// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-matmul-op-pattern"

namespace mlir::syna::torq {

static bool isMapWithResults(AffineMap map, ArrayRef<AffineExpr> expectedResults) {
    if (!map || map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
        map.getNumResults() != expectedResults.size()) {
        return false;
    }

    return llvm::all_of(llvm::enumerate(expectedResults), [&](auto indexedExpr) {
        return map.getResult(indexedExpr.index()) == indexedExpr.value();
    });
}

static bool isIdentityMap(AffineMap map, int64_t rank) {
    if (!map || map.getNumDims() != rank) {
        return false;
    }

    SmallVector<AffineExpr> identityResults;
    identityResults.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
        identityResults.push_back(getAffineDimExpr(i, map.getContext()));
    }
    return isMapWithResults(map, identityResults);
}

static bool isPrefix2DMap(AffineMap map) {
    if (!map) {
        return false;
    }
    auto d0 = getAffineDimExpr(0, map.getContext());
    auto d1 = getAffineDimExpr(1, map.getContext());
    return isMapWithResults(map, {d0, d1});
}

static bool isFinalMatmulRhsBroadcastMap(AffineMap map) {
    if (!map) {
        return false;
    }
    auto d1 = getAffineDimExpr(1, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    return isMapWithResults(map, {d1, d2});
}

static bool isThirdDimZeroMap(AffineMap map) {
    if (!map) {
        return false;
    }
    auto d0 = getAffineDimExpr(0, map.getContext());
    auto d1 = getAffineDimExpr(1, map.getContext());
    auto zero = getAffineConstantExpr(0, map.getContext());
    return isMapWithResults(map, {d0, d1, zero});
}

static bool isFirstThirdDimMap(AffineMap map) {
    if (!map) {
        return false;
    }
    auto d0 = getAffineDimExpr(0, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    return isMapWithResults(map, {d0, d2});
}

static bool isSecondDimZeroMap(AffineMap map) {
    if (!map) {
        return false;
    }
    auto d0 = getAffineDimExpr(0, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    auto zero = getAffineConstantExpr(0, map.getContext());
    return isMapWithResults(map, {d0, zero, d2});
}

static Value getSingleYieldedValue(linalg::GenericOp genericOp) {
    auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return {};
    }
    return yieldOp.getOperand(0);
}

static bool isYieldedBlockArgument(linalg::GenericOp genericOp, unsigned argNumber) {
    Value yieldedValue = getSingleYieldedValue(genericOp);
    auto arg = yieldedValue ? dyn_cast<BlockArgument>(yieldedValue) : BlockArgument();
    return arg && arg.getArgNumber() == argNumber;
}

template <typename BinaryOp> static BinaryOp getSingleYieldedOp(linalg::GenericOp genericOp) {
    Value yieldedValue = getSingleYieldedValue(genericOp);
    if (!yieldedValue) {
        return {};
    }
    return yieldedValue.template getDefiningOp<BinaryOp>();
}

template <typename BinaryOp>
static bool matchYieldedBinaryBlockArgs(
    linalg::GenericOp genericOp, unsigned &lhsArgNumber, unsigned &rhsArgNumber
) {
    auto binaryOp = getSingleYieldedOp<BinaryOp>(genericOp);
    if (!binaryOp) {
        return false;
    }

    auto lhsArg = dyn_cast<BlockArgument>(binaryOp.getLhs());
    auto rhsArg = dyn_cast<BlockArgument>(binaryOp.getRhs());
    if (!lhsArg || !rhsArg) {
        return false;
    }

    lhsArgNumber = lhsArg.getArgNumber();
    rhsArgNumber = rhsArg.getArgNumber();
    return true;
}

static FailureOr<Value> getCollapsedMatmulInput(Value input, RankedTensorType &matmulInputType) {
    matmulInputType = dyn_cast<RankedTensorType>(input.getType());
    if (!matmulInputType || matmulInputType.getRank() != 3 || matmulInputType.getShape()[0] != 1 ||
        ShapedType::isDynamicShape(matmulInputType.getShape())) {
        return failure();
    }

    if (auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>()) {
        auto collapseOp = expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
        if (!collapseOp) {
            return failure();
        }
        return collapseOp.getSrc();
    }

    auto finalGenericOp = input.getDefiningOp<linalg::GenericOp>();
    if (!finalGenericOp || finalGenericOp.getNumDpsInputs() != 1 ||
        finalGenericOp.getNumResults() != 1 || !isYieldedBlockArgument(finalGenericOp, 0)) {
        return failure();
    }

    auto collapsedType = dyn_cast<RankedTensorType>(finalGenericOp.getInputs()[0].getType());
    auto resultType = dyn_cast<RankedTensorType>(finalGenericOp.getResult(0).getType());
    if (!collapsedType || !resultType || collapsedType.getRank() != 2 ||
        resultType != matmulInputType ||
        collapsedType.getShape() != matmulInputType.getShape().drop_front()) {
        return failure();
    }

    auto maps = finalGenericOp.getIndexingMapsArray();
    if (maps.size() != 2 || !isFinalMatmulRhsBroadcastMap(maps[0]) || !isIdentityMap(maps[1], 3)) {
        return failure();
    }

    auto collapseOp = finalGenericOp.getInputs()[0].getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) {
        return failure();
    }
    return collapseOp.getSrc();
}

static Value peelBlockParamBroadcast(Value value, ArrayRef<int64_t> blockedShape) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1 ||
        !isYieldedBlockArgument(genericOp, 0)) {
        return value;
    }

    auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
    if (!inputType || !resultType || resultType.getRank() != 3 ||
        resultType.getShape() != blockedShape || ShapedType::isDynamicShape(inputType.getShape())) {
        return value;
    }

    auto maps = genericOp.getIndexingMapsArray();
    if (maps.size() != 2 || !isIdentityMap(maps[1], 3)) {
        return value;
    }

    if (inputType.getRank() == 2 && inputType.getShape() == blockedShape.drop_back() &&
        isPrefix2DMap(maps[0])) {
        return genericOp.getInputs()[0];
    }

    if (inputType.getRank() == 2 && inputType.getShape()[0] == blockedShape[0] &&
        inputType.getShape()[1] == blockedShape[2] && isFirstThirdDimMap(maps[0])) {
        return genericOp.getInputs()[0];
    }

    if (inputType.getRank() == 3) {
        if (inputType.getShape() == blockedShape && isIdentityMap(maps[0], 3)) {
            return genericOp.getInputs()[0];
        }
        if (inputType.getShape()[0] == blockedShape[0] &&
            inputType.getShape()[1] == blockedShape[1] && inputType.getShape()[2] == 1 &&
            isThirdDimZeroMap(maps[0])) {
            return genericOp.getInputs()[0];
        }
        if (inputType.getShape()[0] == blockedShape[0] && inputType.getShape()[1] == 1 &&
            inputType.getShape()[2] == blockedShape[2] && isSecondDimZeroMap(maps[0])) {
            return genericOp.getInputs()[0];
        }
    }

    return value;
}

enum class BlockQuantizedMatmulInputLayout {
    OutputColumnBlocked,
    ReductionDimBlocked,
};

struct BlockQuantizedMatmulInputMatch {
    Value weight;
    Value scale;
    Value zeroPoint;
    RankedTensorType weightType;
    RankedTensorType scaleType;
    RankedTensorType zeroPointType;
    BlockQuantizedMatmulInputLayout layout;
    bool weightIsUnsigned = false;
    bool zeroPointIsUnsigned = false;
};

static FailureOr<AffineMap> getOutputColumnBlockParamMap(
    RankedTensorType type, int64_t rows, int64_t groups, int64_t blockSize, MLIRContext *ctx
) {
    if (!type || ShapedType::isDynamicShape(type.getShape())) {
        return failure();
    }

    auto d1 = getAffineDimExpr(1, ctx);
    auto d2 = getAffineDimExpr(2, ctx);
    auto blockSizeExpr = getAffineConstantExpr(blockSize, ctx);
    auto groupExpr = d2.floorDiv(blockSizeExpr);

    if (type.getRank() == 1) {
        if (type.getShape()[0] != rows * groups) {
            return failure();
        }
        return AffineMap::get(3, 0, {d1 * getAffineConstantExpr(groups, ctx) + groupExpr}, ctx);
    }

    if (type.getRank() == 2) {
        if (type.getShape()[0] != rows || type.getShape()[1] != groups) {
            return failure();
        }
        return AffineMap::get(3, 0, {d1, groupExpr}, ctx);
    }

    if (type.getRank() == 3) {
        if (type.getShape()[0] != rows || type.getShape()[1] != groups) {
            return failure();
        }
        if (type.getShape()[2] == blockSize) {
            return AffineMap::get(3, 0, {d1, groupExpr, d2 % blockSizeExpr}, ctx);
        }
        if (type.getShape()[2] == 1) {
            return AffineMap::get(3, 0, {d1, groupExpr, getAffineConstantExpr(0, ctx)}, ctx);
        }
    }

    return failure();
}

static FailureOr<AffineMap> getReductionDimBlockParamMap(
    RankedTensorType type, int64_t groups, int64_t blockSize, int64_t cols, MLIRContext *ctx
) {
    if (!type || ShapedType::isDynamicShape(type.getShape())) {
        return failure();
    }

    auto d1 = getAffineDimExpr(1, ctx);
    auto d2 = getAffineDimExpr(2, ctx);
    auto groupExpr = d1.floorDiv(getAffineConstantExpr(blockSize, ctx));
    auto laneExpr = d1 % getAffineConstantExpr(blockSize, ctx);

    if (type.getRank() == 1) {
        if (type.getShape()[0] != groups * cols) {
            return failure();
        }
        return AffineMap::get(3, 0, {groupExpr * getAffineConstantExpr(cols, ctx) + d2}, ctx);
    }

    if (type.getRank() == 2) {
        if (type.getShape()[0] != groups || type.getShape()[1] != cols) {
            return failure();
        }
        return AffineMap::get(3, 0, {groupExpr, d2}, ctx);
    }

    if (type.getRank() == 3) {
        if (type.getShape()[0] != groups || type.getShape()[2] != cols) {
            return failure();
        }
        if (type.getShape()[1] == blockSize) {
            return AffineMap::get(3, 0, {groupExpr, laneExpr, d2}, ctx);
        }
        if (type.getShape()[1] == 1) {
            return AffineMap::get(3, 0, {groupExpr, getAffineConstantExpr(0, ctx), d2}, ctx);
        }
    }

    return failure();
}

static FailureOr<AffineMap> getBlockParamMap(
    RankedTensorType type, ArrayRef<int64_t> blockedShape, BlockQuantizedMatmulInputLayout layout,
    MLIRContext *ctx
) {
    if (layout == BlockQuantizedMatmulInputLayout::OutputColumnBlocked) {
        return getOutputColumnBlockParamMap(
            type, blockedShape[0], blockedShape[1], blockedShape[2], ctx
        );
    }
    return getReductionDimBlockParamMap(
        type, blockedShape[0], blockedShape[1], blockedShape[2], ctx
    );
}

static bool isFloatTensorWithElementType(Value value, Type elementType) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    return type && type.getElementType() == elementType &&
           (elementType.isBF16() || elementType.isF32());
}

static bool isRankIdentityMap(AffineMap map, int64_t rank) {
    if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0 ||
        map.getNumResults() != rank) {
        return false;
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (map.getResult(i) != getAffineDimExpr(i, map.getContext())) {
            return false;
        }
    }
    return true;
}

static LogicalResult matchIntToFpGeneric(
    Value value, RankedTensorType blockedType, Value &weight, RankedTensorType &weightType,
    bool &weightIsUnsigned
) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1) {
        return failure();
    }

    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
    weightType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    if (!resultType || !weightType || resultType != blockedType || weightType.getRank() != 3 ||
        weightType.getShape() != blockedType.getShape() ||
        ShapedType::isDynamicShape(weightType.getShape()) ||
        !weightType.getElementType().isInteger()) {
        return failure();
    }

    auto maps = genericOp.getIndexingMapsArray();
    if (maps.size() != 2 || !isIdentityMap(maps[0], 3) || !isIdentityMap(maps[1], 3)) {
        return failure();
    }

    auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return failure();
    }

    auto yieldedValue = yieldOp.getOperand(0);
    auto sitofpOp = yieldedValue.getDefiningOp<arith::SIToFPOp>();
    auto uitofpOp = yieldedValue.getDefiningOp<arith::UIToFPOp>();
    auto inputArg = sitofpOp
                        ? dyn_cast<BlockArgument>(sitofpOp.getIn())
                        : (uitofpOp ? dyn_cast<BlockArgument>(uitofpOp.getIn()) : BlockArgument());
    if (!inputArg || inputArg.getArgNumber() != 0) {
        return failure();
    }

    weight = genericOp.getInputs()[0];
    weightIsUnsigned = static_cast<bool>(uitofpOp);
    return success();
}

static Value peelIntToFpParam(Value value, Type fpElementType, bool &isUnsigned) {
    if (auto collapseOp = value.getDefiningOp<tensor::CollapseShapeOp>()) {
        return peelIntToFpParam(collapseOp.getSrc(), fpElementType, isUnsigned);
    }

    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1) {
        return value;
    }

    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
    auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    if (!resultType || !inputType || resultType.getShape() != inputType.getShape() ||
        resultType.getElementType() != fpElementType || !inputType.getElementType().isInteger() ||
        ShapedType::isDynamicShape(inputType.getShape())) {
        return value;
    }

    auto maps = genericOp.getIndexingMapsArray();
    if (maps.size() != 2 || !isRankIdentityMap(maps[0], inputType.getRank()) ||
        !isRankIdentityMap(maps[1], resultType.getRank())) {
        return value;
    }

    auto yieldOp = dyn_cast_or_null<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return value;
    }

    auto yieldedValue = yieldOp.getOperand(0);
    auto sitofpOp = yieldedValue.getDefiningOp<arith::SIToFPOp>();
    auto uitofpOp = yieldedValue.getDefiningOp<arith::UIToFPOp>();
    auto inputArg = sitofpOp
                        ? dyn_cast<BlockArgument>(sitofpOp.getIn())
                        : (uitofpOp ? dyn_cast<BlockArgument>(uitofpOp.getIn()) : BlockArgument());
    if (!inputArg || inputArg.getArgNumber() != 0) {
        return value;
    }

    isUnsigned = static_cast<bool>(uitofpOp);
    return genericOp.getInputs()[0];
}

static LogicalResult matchSubChain(
    Value value, RankedTensorType blockedType, Value &weight, Value &zeroPoint,
    RankedTensorType &weightType, RankedTensorType &zeroPointType, bool &weightIsUnsigned,
    bool &zeroPointIsUnsigned, BlockQuantizedMatmulInputLayout layout
) {
    auto subOp = value.getDefiningOp<linalg::GenericOp>();
    if (!subOp || subOp.getNumDpsInputs() != 2 || subOp.getNumResults() != 1) {
        return failure();
    }

    auto resultType = dyn_cast<RankedTensorType>(subOp.getResult(0).getType());
    if (!resultType || resultType != blockedType) {
        return failure();
    }

    auto maps = subOp.getIndexingMapsArray();
    if (maps.size() != 3 || !isIdentityMap(maps[0], 3) || !isIdentityMap(maps[2], 3)) {
        return failure();
    }

    unsigned lhsArg = 0;
    unsigned rhsArg = 0;
    if (!matchYieldedBinaryBlockArgs<arith::SubFOp>(subOp, lhsArg, rhsArg) || lhsArg != 0 ||
        rhsArg != 1) {
        return failure();
    }

    if (failed(matchIntToFpGeneric(
            subOp.getInputs()[0], blockedType, weight, weightType, weightIsUnsigned
        ))) {
        return failure();
    }

    zeroPoint = peelBlockParamBroadcast(subOp.getInputs()[1], blockedType.getShape());
    zeroPoint = peelIntToFpParam(zeroPoint, blockedType.getElementType(), zeroPointIsUnsigned);
    zeroPointType = dyn_cast<RankedTensorType>(zeroPoint.getType());
    if (!zeroPointType ||
        (!zeroPointType.getElementType().isInteger() &&
         !isFloatTensorWithElementType(zeroPoint, blockedType.getElementType()))) {
        return failure();
    }

    MLIRContext *ctx = subOp.getContext();
    if (failed(getBlockParamMap(zeroPointType, blockedType.getShape(), layout, ctx))) {
        return failure();
    }
    return success();
}

static FailureOr<BlockQuantizedMatmulInputLayout>
getBlockQuantizedMatmulInputLayout(ArrayRef<int64_t> blockedShape, ArrayRef<int64_t> inputShape) {
    if (blockedShape[0] == inputShape[1] && blockedShape[1] > 0 && blockedShape[2] > 0 &&
        blockedShape[1] * blockedShape[2] == inputShape[2]) {
        return BlockQuantizedMatmulInputLayout::OutputColumnBlocked;
    }

    if (blockedShape[0] > 0 && blockedShape[1] > 0 && blockedShape[2] == inputShape[2] &&
        blockedShape[0] * blockedShape[1] == inputShape[1]) {
        return BlockQuantizedMatmulInputLayout::ReductionDimBlocked;
    }

    return failure();
}

static LogicalResult matchSeparatedBlockQuantizedMatmulInput(
    Value blockedInput, RankedTensorType inputType, BlockQuantizedMatmulInputMatch &match
) {
    auto blockedType = dyn_cast<RankedTensorType>(blockedInput.getType());
    if (!blockedType || blockedType.getRank() != 3 ||
        blockedType.getElementType() != inputType.getElementType() ||
        ShapedType::isDynamicShape(blockedType.getShape())) {
        return failure();
    }

    auto blockedShape = blockedType.getShape();
    auto inputShape = inputType.getShape();
    FailureOr<BlockQuantizedMatmulInputLayout> layout =
        getBlockQuantizedMatmulInputLayout(blockedShape, inputShape);
    if (failed(layout)) {
        return failure();
    }

    auto mulOp = blockedInput.getDefiningOp<linalg::GenericOp>();
    if (!mulOp || mulOp.getNumDpsInputs() != 2 || mulOp.getNumResults() != 1) {
        return failure();
    }

    auto maps = mulOp.getIndexingMapsArray();
    if (maps.size() != 3 || !isIdentityMap(maps[0], 3) || !isIdentityMap(maps[2], 3)) {
        return failure();
    }

    unsigned lhsArg = 0;
    unsigned rhsArg = 0;
    if (!matchYieldedBinaryBlockArgs<arith::MulFOp>(mulOp, lhsArg, rhsArg)) {
        return failure();
    }

    SmallVector<std::pair<unsigned, unsigned>, 2> candidates;
    if (lhsArg < 2 && rhsArg < 2 && lhsArg != rhsArg) {
        candidates.push_back({lhsArg, rhsArg});
        candidates.push_back({rhsArg, lhsArg});
    }

    for (auto [subInputIdx, scaleInputIdx] : candidates) {
        Value weight;
        Value zeroPoint;
        RankedTensorType weightType;
        RankedTensorType zeroPointType;
        bool weightIsUnsigned = false;
        bool zeroPointIsUnsigned = false;
        if (failed(matchSubChain(
                mulOp.getInputs()[subInputIdx], blockedType, weight, zeroPoint, weightType,
                zeroPointType, weightIsUnsigned, zeroPointIsUnsigned, *layout
            ))) {
            continue;
        }

        Value scale = peelBlockParamBroadcast(mulOp.getInputs()[scaleInputIdx], blockedShape);
        auto scaleType = dyn_cast<RankedTensorType>(scale.getType());
        if (!scaleType || !isFloatTensorWithElementType(scale, blockedType.getElementType())) {
            continue;
        }

        MLIRContext *ctx = mulOp.getContext();
        if (failed(getBlockParamMap(scaleType, blockedShape, *layout, ctx))) {
            continue;
        }

        match = {weight,        scale,   zeroPoint,        weightType,         scaleType,
                 zeroPointType, *layout, weightIsUnsigned, zeroPointIsUnsigned};
        return success();
    }

    return failure();
}

static Value extendToI32(Value value, PatternRewriter &rewriter, Location loc, bool isUnsigned) {
    auto intType = cast<IntegerType>(value.getType());
    if (intType.getWidth() == 32) {
        return value;
    }
    if (isUnsigned) {
        return arith::ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), value);
    }
    return arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), value);
}

/// Rewrite a block-quantized dequant chain into a single linalg.generic with
/// simple indexing maps (no floordiv/mod), producing the dequantized weight.
/// The dequantized weight is then collapsed/expanded and fed into the existing
/// batch_matmul, which is kept as-is.
///
/// For ReductionDimBlocked (K = G * B, weight is [G, B, N]):
///   3 dims: (d0=G, d1=B, d2=N)
///   Maps:
///     weight[G,B,N]:   (d0, d1, d2)   -- identity
///     scale:           simple broadcast map
///     zp:              simple broadcast map
///     output[G,B,N]:   (d0, d1, d2)   -- identity
///   Then: collapse [G,B,N] -> [K,N], expand -> [1,K,N], feed to batch_matmul
///
/// For OutputColumnBlocked (N = G * B, weight is [K, G, B]):
///   3 dims: (d0=K, d1=G, d2=B)
///   Maps:
///     weight[K,G,B]:   (d0, d1, d2)   -- identity
///     scale:           simple broadcast map
///     zp:              simple broadcast map
///     output[K,G,B]:   (d0, d1, d2)   -- identity
///   Then: collapse [K,G,B] -> [K,N], expand -> [1,K,N], feed to batch_matmul
///
/// Returns the dequantized matmul input (tensor<1xKxNxfp>) or failure.
static FailureOr<Value>
rewriteBlockQuantizedMatmulInputSimple(Value input, PatternRewriter &rewriter, Location loc) {
    OpBuilder::InsertionGuard guard(rewriter);

    RankedTensorType inputType;
    auto blockedInput = getCollapsedMatmulInput(input, inputType);
    if (failed(blockedInput)) {
        return failure();
    }

    BlockQuantizedMatmulInputMatch match;
    if (failed(matchSeparatedBlockQuantizedMatmulInput(*blockedInput, inputType, match))) {
        return failure();
    }

    auto blockedShape = match.weightType.getShape();
    MLIRContext *ctx = rewriter.getContext();
    Type fpType = inputType.getElementType();

    // Build simple 3D maps for scale/zp based on the blocked weight shape.
    // The dequant generic operates in the natural blocked shape [G,B,N] or [K,G,B].
    // We need simple broadcast maps for scale/zp (no div/mod).
    auto getSimpleDequantParamMap = [&](RankedTensorType paramType) -> FailureOr<AffineMap> {
        if (!paramType || ShapedType::isDynamicShape(paramType.getShape())) {
            return failure();
        }
        auto shape = paramType.getShape();
        auto zero = getAffineConstantExpr(0, ctx);
        auto d0 = getAffineDimExpr(0, ctx);
        auto d1 = getAffineDimExpr(1, ctx);
        auto d2 = getAffineDimExpr(2, ctx);

        if (match.layout == BlockQuantizedMatmulInputLayout::ReductionDimBlocked) {
            // blockedShape = [G, B, N], dims: (d0=G, d1=B, d2=N)
            int64_t G = blockedShape[0], B = blockedShape[1], N = blockedShape[2];
            if (paramType.getRank() == 1) {
                if (shape[0] == N)
                    return AffineMap::get(3, 0, {d2}, ctx);
                if (shape[0] == G)
                    return AffineMap::get(3, 0, {d0}, ctx);
                return failure();
            }
            if (paramType.getRank() == 2) {
                if (shape[0] == G && shape[1] == N)
                    return AffineMap::get(3, 0, {d0, d2}, ctx);
                if (shape[0] == 1 && shape[1] == N)
                    return AffineMap::get(3, 0, {zero, d2}, ctx);
                return failure();
            }
            if (paramType.getRank() == 3) {
                if (shape[0] == G && shape[2] == N) {
                    if (shape[1] == 1)
                        return AffineMap::get(3, 0, {d0, zero, d2}, ctx);
                    if (shape[1] == B)
                        return AffineMap::get(3, 0, {d0, d1, d2}, ctx);
                }
                return failure();
            }
            return failure();
        }

        // OutputColumnBlocked: blockedShape = [K, G, B], dims: (d0=K, d1=G, d2=B)
        int64_t K = blockedShape[0], G = blockedShape[1], B = blockedShape[2];
        if (paramType.getRank() == 1) {
            if (shape[0] == G)
                return AffineMap::get(3, 0, {d1}, ctx);
            if (shape[0] == K)
                return AffineMap::get(3, 0, {d0}, ctx);
            return failure();
        }
        if (paramType.getRank() == 2) {
            if (shape[0] == K && shape[1] == G)
                return AffineMap::get(3, 0, {d0, d1}, ctx);
            if (shape[0] == 1 && shape[1] == G)
                return AffineMap::get(3, 0, {zero, d1}, ctx);
            return failure();
        }
        if (paramType.getRank() == 3) {
            if (shape[0] == K && shape[1] == G) {
                if (shape[2] == 1)
                    return AffineMap::get(3, 0, {d0, d1, zero}, ctx);
                if (shape[2] == B)
                    return AffineMap::get(3, 0, {d0, d1, d2}, ctx);
            }
            return failure();
        }
        return failure();
    };

    auto scaleMap = getSimpleDequantParamMap(match.scaleType);
    auto zpMap = getSimpleDequantParamMap(match.zeroPointType);
    if (failed(scaleMap) || failed(zpMap)) {
        return failure();
    }

    // The dequant generic output has the same shape as the weight: [G,B,N] or [K,G,B]
    auto dequantOutputType = RankedTensorType::get(blockedShape, fpType);
    AffineMap weightMap = rewriter.getMultiDimIdentityMap(3);
    AffineMap outMap = rewriter.getMultiDimIdentityMap(3);

    SmallVector<AffineMap> indexingMaps = {weightMap, *scaleMap, *zpMap, outMap};
    SmallVector<utils::IteratorType> iteratorTypes(3, utils::IteratorType::parallel);

    Value outputInit = tensor::EmptyOp::create(rewriter, loc, blockedShape, fpType);

    bool zpIsInt = match.zeroPointType.getElementType().isInteger();
    bool wIsUnsigned = match.weightIsUnsigned;
    bool zpIsUnsigned = match.zeroPointIsUnsigned;

    auto newGeneric = linalg::GenericOp::create(
        rewriter, loc, TypeRange{dequantOutputType},
        ValueRange{match.weight, match.scale, match.zeroPoint}, ValueRange{outputInit},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
            // args[0] = weight, args[1] = scale, args[2] = zp, args[3] = out(init)
            Value centeredWeight;
            if (zpIsInt) {
                Value wI32 = extendToI32(args[0], rewriter, bodyLoc, wIsUnsigned);
                Value zpI32 = extendToI32(args[2], rewriter, bodyLoc, zpIsUnsigned);
                Value shifted = arith::SubIOp::create(b, bodyLoc, wI32, zpI32);
                centeredWeight = arith::SIToFPOp::create(b, bodyLoc, fpType, shifted);
            }
            else {
                Value wFp = wIsUnsigned
                                ? arith::UIToFPOp::create(b, bodyLoc, fpType, args[0]).getResult()
                                : arith::SIToFPOp::create(b, bodyLoc, fpType, args[0]).getResult();
                centeredWeight = arith::SubFOp::create(b, bodyLoc, wFp, args[2]);
            }
            Value scaledWeight = arith::MulFOp::create(b, bodyLoc, centeredWeight, args[1]);
            linalg::YieldOp::create(b, bodyLoc, scaledWeight);
        }
    );

    // Collapse the blocked output to 2D: [G,B,N] -> [K,N] or [K,G,B] -> [K,N]
    Value dequantResult = newGeneric.getResult(0);
    SmallVector<ReassociationIndices> collapseReassoc;
    if (match.layout == BlockQuantizedMatmulInputLayout::ReductionDimBlocked) {
        // [G, B, N] -> [G*B, N] = [K, N]
        collapseReassoc = {{0, 1}, {2}};
    }
    else {
        // [K, G, B] -> [K, G*B] = [K, N]
        collapseReassoc = {{0}, {1, 2}};
    }
    auto inputShape = inputType.getShape();
    auto collapsed2DType = RankedTensorType::get({inputShape[1], inputShape[2]}, fpType);
    auto collapsed = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsed2DType, dequantResult, collapseReassoc
    );

    // Expand from [K, N] to [1, K, N] to match the batch_matmul input shape
    SmallVector<ReassociationIndices> expandReassoc = {{0, 1}, {2}};
    auto expandedResult = tensor::ExpandShapeOp::create(
        rewriter, loc, inputType, collapsed.getResult(), expandReassoc
    );

    return expandedResult.getResult();
}

/// Folds input operations by removing unnecessary collapse shape and generic operations.
///
/// This function optimizes input values by:
/// 1. Removing tensor::CollapseShapeOp if present and storing it for later restoration
/// 2. Checking if the input is produced by a linalg::GenericOp with single input and output
/// 3. Verifying that the generic operation preserves the tensor type
/// 4. Replacing the input with the generic operation's input
/// 5. Restoring the collapse shape operation around the folded input if it existed
///
/// @param input [in/out] The input value to fold. Modified in-place to point to the folded value.
/// @param rewriter [in] The pattern rewriter used to modify operations in-place.
///
/// @note This function modifies the input operand and any related collapse shape operations.
///       It performs no rewrites if the input is not produced by the expected operation patterns.
///
static void foldInput(Value &input, PatternRewriter &rewriter) {
    Value currentInput = input;

    auto collapseOp = dyn_cast_or_null<tensor::CollapseShapeOp>(currentInput.getDefiningOp());
    if (collapseOp)
        currentInput = collapseOp.getOperand();

    auto foldOp = dyn_cast_or_null<linalg::GenericOp>(currentInput.getDefiningOp());
    if (!foldOp)
        return;

    if (foldOp.getNumDpsInputs() != 1 || foldOp.getNumResults() != 1)
        return;

    auto resultType = dyn_cast<RankedTensorType>(foldOp.getResultTypes().front());
    auto inputType = dyn_cast<RankedTensorType>(foldOp.getInputs()[0].getType());
    if (resultType != inputType)
        return;

    if (collapseOp) {
        rewriter.modifyOpInPlace(collapseOp, [&]() { collapseOp->setOperand(0, currentInput); });
        return;
    }

    input = foldOp.getInputs()[0];
}

/// Checks if input is produced by a linalg::GenericOp with a constant input and expands it.
///
/// This function optimizes constant inputs by:
/// 1. Verifying the input is produced by a linalg::GenericOp with single input and output
/// 2. Checking if the generic operation's input is a constant value
/// 3. Validating output rank equals input rank + 1 with matching dimensions
/// 4. Creating a tensor::ExpandShapeOp to replace the generic operation
/// 5. Updating the input to point to the expanded result
///
/// @param input [in/out] The input value to optimize. Modified in-place if a constant pattern is
/// found.
/// @param rewriter [in] The pattern rewriter used to create new operations.
///
/// @note Only modifies input if it matches the constant expansion pattern.
///
static void expandConstantInput(Value &input, PatternRewriter &rewriter) {
    auto definingOp = dyn_cast_or_null<linalg::GenericOp>(input.getDefiningOp());
    if (!definingOp) {
        return;
    }
    if (definingOp.getNumDpsInputs() != 1 || definingOp.getNumResults() != 1) {
        return;
    }

    auto resultType = dyn_cast<RankedTensorType>(definingOp.getResultTypes().front());
    auto inputType = dyn_cast<RankedTensorType>(definingOp.getInputs()[0].getType());

    if (resultType == inputType)
        return;

    // to check if output rank = input rank + 1
    // output has more dim than input shape
    auto outputShape = resultType.getShape();
    auto inputShape = inputType.getShape();
    if (outputShape.size() != inputShape.size() + 1) {
        return;
    }
    for (auto it : llvm::zip(inputShape, outputShape.drop_front())) {
        if (std::get<0>(it) != std::get<1>(it)) {
            return;
        }
    }

    // When the broadcast dimension is 1, expand_shape is a zero-copy reshape and always valid
    // regardless if the input is constant or tensor.
    // Thus, bail out only if bcast_dim != 1 and input != arith.constant.
    if (outputShape[0] != 1) {
        TypedAttr constAttr;
        if (!matchPattern(definingOp.getInputs()[0], m_Constant(&constAttr))) {
            return;
        }
    }

    // create a new expandshapeOp to replace the genericOp
    SmallVector<int64_t> newShape(outputShape.begin(), outputShape.end());
    auto elementType = inputType.getElementType();
    auto outType = RankedTensorType::get(newShape, elementType);

    SmallVector<ReassociationIndices> reassoc;
    int rank = outputShape.size();
    if (rank <= 1) {
        for (int i = 0; i < rank; ++i)
            reassoc.push_back({i});
    }
    else {
        reassoc.push_back({0, 1});
        for (int i = 2; i < rank; ++i) {
            reassoc.push_back({i});
        }
    }

    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, definingOp.getLoc(), outType, definingOp.getInputs()[0], reassoc
    );

    input = expandOp.getResult();
}

/// Check if a batch_matmul input comes from a broadcast-batch pattern:
///   collapse_shape(linalg.generic(broadcast original [1,K,N] → [1,B,K,N]))
/// Returns the broadcast input index (0 or 1), or -1 if no match.
/// Also returns the original un-broadcast source via `broadcastSource`.
static int detectBroadcastBatchInput(linalg::BatchMatmulOp srcOp, Value &broadcastSource) {
    for (int i : {0, 1}) {
        Value input = srcOp.getInputs()[i];
        auto collapseOp = input.getDefiningOp<tensor::CollapseShapeOp>();
        if (!collapseOp)
            continue;
        // first reassociation group must combine exactly two dims [[0,1],...]
        auto reassoc = collapseOp.getReassociationIndices();
        if (reassoc.size() < 2 || reassoc[0].size() != 2)
            continue;
        // pre-collapse dim 0 should be 1 (the original batch-1 dim before broadcast)
        auto srcType = collapseOp.getSrcType();
        if (srcType.getDimSize(0) != 1)
            continue;
        // source of collapse_shape should be a broadcast linalg.generic
        auto genericOp = collapseOp.getSrc().getDefiningOp<linalg::GenericOp>();
        if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1)
            continue;
        // identity body: only a yield op
        if (genericOp.getBody()->getOperations().size() != 1)
            continue;
        // verify indexing maps are projected permutations (no transpose or shuffle)
        auto maps = genericOp.getIndexingMapsArray();
        bool validMaps = true;
        for (auto &map : maps) {
            if (!map.isProjectedPermutation()) {
                validMaps = false;
                break;
            }
        }
        if (!validMaps)
            continue;
        // verify it's actually a broadcast (output has more elements than input)
        auto inputType = cast<RankedTensorType>(genericOp.getInputs()[0].getType());
        auto outputType = cast<RankedTensorType>(genericOp.getResultTypes()[0]);
        if (inputType.getNumElements() >= outputType.getNumElements())
            continue;

        broadcastSource = genericOp.getInputs()[0];
        return i;
    }
    return -1;
}

/// Replace a batch_matmul with a broadcast-batch input by a linalg.generic that
/// performs the matmul with broadcast expressed via indexing maps. This avoids
/// materializing the broadcast buffer entirely.
///
/// batch_matmul([B,M,K], collapse([1,B,K,N])) where rhs came from broadcasting [1,K,N]
/// becomes:
///   linalg.generic {
///     indexing_maps = [(b,m,n,k)->(b,m,k), (b,m,n,k)->(0,k,n), (b,m,n,k)->(b,m,n)]
///     iterator_types = ["parallel","parallel","parallel","reduction"]
///   } ins(A:[B,M,K], B_orig:[1,K,N]) outs(C:[B,M,N]) { mulf + addf }
///
static LogicalResult replaceBatchMatmulWithBroadcastGeneric(
    linalg::BatchMatmulOp srcOp, Value broadcastSource, int broadcastIdx, PatternRewriter &rewriter
) {
    Location loc = srcOp.getLoc();

    // Determine which input is broadcast and which is the "real" batched input
    Value batchedInput = srcOp.getInputs()[1 - broadcastIdx];
    Value output = srcOp.getResults()[0];
    auto outputType = cast<RankedTensorType>(output.getType());
    auto batchedType = cast<RankedTensorType>(batchedInput.getType());

    int64_t batchSize = batchedType.getDimSize(0);
    (void)batchSize;

    // Build indexing maps for broadcast batch matmul:
    //   A[b,m,k] * B[0,k,n] -> C[b,m,n]  (if broadcastIdx == 1)
    //   A[0,m,k] * B[b,k,n] -> C[b,m,n]  (if broadcastIdx == 0)
    MLIRContext *ctx = rewriter.getContext();
    // dims: (b=d0, m=d1, n=d2, k=d3)
    AffineMap lhsMap, rhsMap, outMap;
    if (broadcastIdx == 1) {
        // lhs = A[b,m,k], rhs = B[0,k,n]
        lhsMap = AffineMap::get(
            4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)},
            ctx
        );
        rhsMap = AffineMap::get(
            4, 0,
            {getAffineConstantExpr(0, ctx), getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)}, ctx
        );
    }
    else {
        // lhs = A[0,m,k], rhs = B[b,k,n]
        lhsMap = AffineMap::get(
            4, 0,
            {getAffineConstantExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)}, ctx
        );
        rhsMap = AffineMap::get(
            4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)},
            ctx
        );
    }
    outMap = AffineMap::get(
        4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx
    );

    SmallVector<AffineMap> indexingMaps = {lhsMap, rhsMap, outMap};
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, // batch
        utils::IteratorType::parallel, // M
        utils::IteratorType::parallel, // N
        utils::IteratorType::reduction // K
    };

    // Use the existing output (from linalg.fill) as the init
    Value init = srcOp.getDpsInits()[0];

    // Order inputs as [lhs, rhs] matching the maps above
    Value lhs = (broadcastIdx == 1) ? batchedInput : broadcastSource;
    Value rhs = (broadcastIdx == 1) ? broadcastSource : batchedInput;

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, outputType, ValueRange{lhs, rhs}, ValueRange{init}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] = lhs element, args[1] = rhs element, args[2] = accumulator
            Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
            Value add = arith::AddFOp::create(b, loc, args[2], mul);
            linalg::YieldOp::create(b, loc, add);
        }
    );

    rewriter.replaceOp(srcOp, genericOp.getResults());
    return success();
}

/// Optimization pattern for batch matrix multiplication operations.
///
/// This pattern optimizes linalg::BatchMatmulOp by folding unnecessary operations
/// on inputs that may have been added during upstream transformations from torch,
/// onnx, or tosa dialects. The optimization removes redundant reshape, collapse,
/// and generic operations to simplify subsequent lowering and avoid unnecessary
/// broadcast operations.
///
/// The pattern applies the following optimizations in sequence:
/// 1. Folds input operations by removing unnecessary collapse shape wrappers
/// 2. Expands constant inputs that were wrapped in generic operations
/// 3. Detects broadcast-batch inputs and replaces the batch_matmul with a
///    linalg.generic that expresses the broadcast via indexing maps, avoiding
///    materialization of the broadcast buffer
/// 4. Updates the matmul operation's operands to point to the optimized inputs
/// 5. Fuses separated block-quantized input dequant operations into one matmul-shaped generic
/// 5. Marks the operation as optimized to prevent redundant passes
///
class BatchMatmulOpPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp srcOp, PatternRewriter &rewriter) const {

        if (srcOp->getAttrOfType<BoolAttr>("matmulOptimized")) {
            // Avoid applying the same transformation twice.
            return failure();
        }

        Value input0 = srcOp.getInputs()[0];
        Value input1 = srcOp.getInputs()[1];

        // Try the simple dequant pattern first (no div/mod in maps).
        // This produces a separate dequant linalg.generic and keeps the batch_matmul.
        if (auto simpleLhs =
                rewriteBlockQuantizedMatmulInputSimple(input0, rewriter, srcOp.getLoc());
            succeeded(simpleLhs)) {
            input0 = *simpleLhs;
        }
        if (auto simpleRhs =
                rewriteBlockQuantizedMatmulInputSimple(input1, rewriter, srcOp.getLoc());
            succeeded(simpleRhs)) {
            input1 = *simpleRhs;
        }

        /// TODO: Replace the sequential foldInput calls with a while-loop based approach to
        /// iteratively apply folding transformations until a fixed point is reached. This would
        /// handle cases where folding one input enables additional folding opportunities on the
        /// other input or vice versa, ensuring more aggressive optimization and potentially
        /// exposing additional canonicalization opportunities that single-pass folding would miss.
        foldInput(input0, rewriter);
        foldInput(input1, rewriter);

        expandConstantInput(input0, rewriter);
        expandConstantInput(input1, rewriter);

        // Detect broadcast-batch pattern and replace with a linalg.generic that
        // expresses the broadcast via indexing maps. This avoids materializing
        // the broadcast buffer and lets tile-and-fuse handle it naturally.
        Value broadcastSource;
        int broadcastIdx = detectBroadcastBatchInput(srcOp, broadcastSource);
        if (broadcastIdx >= 0) {
            // Only apply when the broadcast source is rank 3 ([1,K,N]).
            // Rank-2 sources ([K,N]) would produce a linalg.generic with a
            // rank-2 operand that the TORQ backend cannot convert back to
            // torq_hl::MatMulOp (findBatchMatmulBcastIdx expects 3-result maps).
            auto broadcastType = dyn_cast<RankedTensorType>(broadcastSource.getType());
            if (broadcastType && broadcastType.getRank() == 3) {
                return replaceBatchMatmulWithBroadcastGeneric(
                    srcOp, broadcastSource, broadcastIdx, rewriter
                );
            }
        }

        rewriter.modifyOpInPlace(srcOp, [&]() {
            srcOp->setOperand(0, input0);
            srcOp->setOperand(1, input1);
        });

        srcOp->setAttr("matmulOptimized", BoolAttr::get(srcOp->getContext(), true));

        return success();
    }
};

void populateOptimizeMatmuOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<BatchMatmulOpPattern>(context);
}

} // namespace mlir::syna::torq
