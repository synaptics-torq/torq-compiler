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
#include "mlir/IR/DialectResourceBlobManager.h"
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

enum class HeadBroadcastMode { Enable, Disable };

static llvm::cl::opt<HeadBroadcastMode> clHeadBroadcastMode(
    "torq-head-broadcast-mode",
    llvm::cl::desc("Choose how to handle a batch matmul operand built by repeating the heads "
                   "of a smaller key/value cache tensor"),
    llvm::cl::values(
        clEnumValN(
            HeadBroadcastMode::Enable, "enable",
            "Query-head groups that share one cache head are rewritten to read the "
            "small cache with query-head groups folded into M, so the repeated-head "
            "tensors (the full-size K^T transpose and the expanded cache) are not "
            "materialized. Covers the grouped form and the single-cache-head form, "
            "both lowered through the existing batch matmul path. "
            "Value-preserving; on by default"
        ),
        clEnumValN(
            HeadBroadcastMode::Disable, "disable",
            "Leave repeated-head cache chains in place; the pre-existing "
            "batch-1 broadcast fold is unaffected"
        )
    ),
    llvm::cl::init(HeadBroadcastMode::Enable)
);

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

/// Matches a batch matmul input [1,K,N] that lifts a per-tensor dequantized
/// constant weight, the form the importer emits for a DequantizeLinear with
/// scalar scale and zero point:
///   %w = linalg.generic ins(%cst : tensor<KxNxi8>) {
///          extui/extsi (or none); [subi %zp]; sitofp; mulf %scale }
///   input = tensor.expand_shape %w [[0, 1], [2]]
///        or linalg.generic (d0,d1,d2)->(d1,d2) broadcast of %w
/// with %zp (i32) and %scale captured constants. Fills `match` with the
/// equivalent single-block form over new constants: the weight as [1,K,N],
/// scale and zero point as [1,1,N] splats. A uint8 weight is re-encoded as
/// int8 (w ^ 0x80, zp - 128), which preserves w - zp, so that consumers that
/// read the weight as signed (the mixed-quant matmul) accept it.
/// Facts matched out of a constant per-tensor dequant generic (see
/// matchConstantPerTensorDequant).
struct PerTensorDequantMatch {
    Attribute weightAttr;
    RankedTensorType weightType;
    FloatAttr scaleAttr;
    int64_t zeroPoint = 0;
    bool isUnsigned = false;
};

/// Matches a linalg.generic computing mulf(sitofp([subi](ext(w), zp)), scale)
/// with identity maps over a constant int8 weight and nothing else -- the form
/// the importer emits for a per-tensor DequantizeLinear. The zero point is
/// adjusted for the uint8 -> int8 re-encode (zp - 128) and checked against the
/// 8-bit range.
static LogicalResult
matchConstantPerTensorDequant(linalg::GenericOp dq, PerTensorDequantMatch &dequant) {
    if (!dq || dq.getNumDpsInputs() != 1 || dq.getNumResults() != 1 ||
        !llvm::all_of(dq.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); }))
        return failure();
    auto weightType = dyn_cast<RankedTensorType>(dq.getInputs()[0].getType());
    Attribute weightAttr;
    if (!weightType || !weightType.getElementType().isInteger(8) ||
        !matchPattern(dq.getInputs()[0], m_Constant(&weightAttr)))
        return failure();

    // Body: mulf(sitofp([subi](ext(w), zp)), scale), nothing else.
    Block *body = dq.getBody();
    auto mulf = getSingleYieldedValue(dq).getDefiningOp<arith::MulFOp>();
    auto sitofp = mulf ? mulf.getLhs().getDefiningOp<arith::SIToFPOp>() : arith::SIToFPOp();
    if (!sitofp)
        return failure();
    auto subi = sitofp.getIn().getDefiningOp<arith::SubIOp>();
    Value integer = subi ? subi.getLhs() : sitofp.getIn();
    auto extui = integer.getDefiningOp<arith::ExtUIOp>();
    auto extsi = integer.getDefiningOp<arith::ExtSIOp>();
    Value weightArg = extui ? extui.getIn() : extsi ? extsi.getIn() : integer;
    if (weightArg != body->getArgument(0) || (subi && !extui && !extsi) ||
        body->getOperations().size() != 3 + bool(subi) + bool(extui || extsi))
        return failure();
    FloatAttr scaleAttr;
    if (!matchPattern(mulf.getRhs(), m_Constant(&scaleAttr)))
        return failure();
    int64_t zeroPoint = 0;
    if (subi) {
        IntegerAttr zpAttr;
        if (!matchPattern(subi.getRhs(), m_Constant(&zpAttr)))
            return failure();
        zeroPoint = zpAttr.getInt();
    }
    bool isUnsigned = static_cast<bool>(extui);
    if (isUnsigned)
        zeroPoint -= 128;
    if (zeroPoint < -128 || zeroPoint > 127)
        return failure();
    dequant.weightAttr = weightAttr;
    dequant.weightType = weightType;
    dequant.scaleAttr = scaleAttr;
    dequant.zeroPoint = zeroPoint;
    dequant.isUnsigned = isUnsigned;
    return success();
}

static LogicalResult matchPerTensorDequantMatmulInput(
    Value input, RankedTensorType &inputType, BlockQuantizedMatmulInputMatch &match,
    PatternRewriter &rewriter, Location loc
) {
    inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 3 || inputType.getDimSize(0) != 1 ||
        !inputType.hasStaticShape())
        return failure();
    Value lifted;
    if (auto expand = input.getDefiningOp<tensor::ExpandShapeOp>()) {
        lifted = expand.getSrc();
    }
    else if (auto lift = input.getDefiningOp<linalg::GenericOp>()) {
        auto maps = lift.getIndexingMapsArray();
        if (lift.getNumDpsInputs() != 1 || !isYieldedBlockArgument(lift, 0) ||
            !isFinalMatmulRhsBroadcastMap(maps[0]) || !isIdentityMap(maps[1], 3))
            return failure();
        lifted = lift.getInputs()[0];
    }
    auto dq = lifted ? lifted.getDefiningOp<linalg::GenericOp>() : linalg::GenericOp();
    if (!dq ||
        dq.getResult(0).getType() !=
            RankedTensorType::get(inputType.getShape().drop_front(), inputType.getElementType()))
        return failure();

    PerTensorDequantMatch dequant;
    if (failed(matchConstantPerTensorDequant(dq, dequant)))
        return failure();
    Attribute weightAttr = dequant.weightAttr;
    RankedTensorType weightType = dequant.weightType;
    FloatAttr scaleAttr = dequant.scaleAttr;
    int64_t zeroPoint = dequant.zeroPoint;
    bool isUnsigned = dequant.isUnsigned;

    // The constants of the single-block form.
    int64_t N = weightType.getDimSize(1);
    auto i8 = rewriter.getI8Type();
    auto blockedType = RankedTensorType::get({1, weightType.getDimSize(0), N}, i8);
    Attribute blockedAttr;
    if (auto dense = dyn_cast<DenseIntElementsAttr>(weightAttr)) {
        if (isUnsigned)
            dense = cast<DenseIntElementsAttr>(dense.mapValues(i8, [](const APInt &v) {
                return v ^ APInt(8, 0x80);
            }));
        blockedAttr = dense.reshape(blockedType);
    }
    else if (auto resource = dyn_cast<DenseResourceElementsAttr>(weightAttr)) {
        AsmResourceBlob *blob = resource.getRawHandle().getBlob();
        if (!blob || (int64_t)blob->getData().size() != blockedType.getNumElements())
            return failure();
        if (isUnsigned) {
            SmallVector<char> bytes(blob->getData());
            for (char &b : bytes)
                b ^= static_cast<char>(0x80);
            blockedAttr = DenseElementsAttr::getFromRawBuffer(blockedType, bytes);
        }
        else {
            blockedAttr = DenseResourceElementsAttr::get(blockedType, resource.getRawHandle());
        }
    }
    else {
        return failure();
    }
    auto paramShape = SmallVector<int64_t>{1, 1, N};
    match.weightType = blockedType;
    match.scaleType = RankedTensorType::get(paramShape, scaleAttr.getType());
    match.zeroPointType = RankedTensorType::get(paramShape, i8);
    match.weight = arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(blockedAttr));
    match.scale = arith::ConstantOp::create(
        rewriter, loc, DenseElementsAttr::get(match.scaleType, ArrayRef<Attribute>{scaleAttr})
    );
    match.zeroPoint = arith::ConstantOp::create(
        rewriter, loc,
        DenseElementsAttr::get(
            match.zeroPointType,
            ArrayRef<Attribute>{IntegerAttr::get(i8, APInt(8, zeroPoint, /*isSigned=*/true))}
        )
    );
    match.layout = BlockQuantizedMatmulInputLayout::ReductionDimBlocked;
    return success();
}

/// Rewrite a block-quantized dequant chain into a single linalg.generic with
/// simple indexing maps (no floordiv/mod), producing the dequantized weight.
/// The dequantized weight is then collapsed/expanded and fed into the existing
/// batch_matmul, which is kept as-is. A per-tensor dequantized constant weight
/// (see matchPerTensorDequantMatmulInput) is rewritten as a single block.
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
    BlockQuantizedMatmulInputMatch match;
    auto blockedInput = getCollapsedMatmulInput(input, inputType);
    if ((failed(blockedInput) ||
         failed(matchSeparatedBlockQuantizedMatmulInput(*blockedInput, inputType, match))) &&
        failed(matchPerTensorDequantMatmulInput(input, inputType, match, rewriter, loc))) {
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

    // Only peel a genuine no-op generic (body just yields its single input, e.g. a
    // layout/copy). A real unary elementwise op (abs, negate, mulf %in,%in, ...) also has
    // one input, one result and matching input/result types, but it performs a computation;
    // folding it would silently discard that computation and rewire the matmul to read the
    // op's input. Require an identity body, matching
    // getCollapsedMatmulInput/peelBlockParamBroadcast.
    if (!isYieldedBlockArgument(foldOp, 0))
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

    // Guard before dereferencing below: a non-RankedTensorType operand/result
    // yields a null type from dyn_cast, which would crash at getShape().
    if (!resultType || !inputType)
        return;

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

    // A tensor.expand_shape can only introduce size-1 dimensions; it preserves
    // element count and cannot express a broadcast. The reassociation built
    // below ([[0,1],[2],...]) folds the new leading dimension into the source's
    // first dimension, so it is only valid when that leading dimension is 1
    // (a zero-copy reshape, e.g. [K,N] -> [1,K,N]).
    //
    // When outputShape[0] != 1 the generic is a genuine batch broadcast
    // ([K,N] -> [B,K,N]). Rewriting it as an expand_shape produces a result
    // whose element count differs from its source; the greedy rewriter then
    // constant-folds it (tensor::ExpandShapeOp::fold -> DenseElementsAttr::reshape)
    // and asserts on the element-count mismatch (see issue #1767). This is true
    // even when the source is a constant, so leave such broadcasts in place and
    // let normal lowering materialize them -- mirroring the (already working)
    // non-constant broadcast-weight path.
    if (outputShape[0] != 1) {
        return;
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

/// The data input of a link that commutes with a head broadcast
/// [1,K,N] -> [1,B,K,N] feeding a batch matmul (see detectBroadcastBatchInput),
/// or null if `op` is not such a link:
///   - a transpose that keeps dims 0 and 1 (unit batch, broadcast heads) in place;
///   - an elementwise generic whose other inputs are constants read at fixed
///     indices (e.g. a scalar scale).
static Value getBroadcastCommutingLinkInput(Operation *op) {
    if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
        auto perm = transpose.getPermutation();
        if (perm.size() != 4 || perm[0] != 0 || perm[1] != 1)
            return {};
        return transpose.getInput();
    }
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic || generic.getNumResults() != 1 || !generic.isAllParallelLoops() ||
        generic.hasIndexSemantics() ||
        generic.payloadUsesValueFromOperand(generic.getDpsInitOperand(0)) ||
        generic.getNumLoops() != 4 ||
        !generic.getMatchingIndexingMap(generic.getDpsInitOperand(0)).isIdentity())
        return {};
    Value data;
    for (OpOperand *input : generic.getDpsInputOperands()) {
        AffineMap map = generic.getMatchingIndexingMap(input);
        if (map.isIdentity() && !data)
            data = input->get();
        else if (!map.isConstant() || !input->get().getDefiningOp<arith::ConstantOp>())
            return {};
    }
    return data;
}

/// Replays the broadcast-commuting `links` (in data-flow order) on the
/// un-broadcast source, so they run once instead of once per head. The links
/// are rank-4 ops ([1, batch, d2, d3]); the source is rank 3 with the
/// leading unit dimension removed.
static Value
replayLinksBeforeBroadcast(Value source, ArrayRef<Operation *> links, PatternRewriter &rewriter) {
    MLIRContext *ctx = rewriter.getContext();
    int64_t srcRank = cast<RankedTensorType>(source.getType()).getRank();
    assert(srcRank == 3 && "source must be rank 3");
    AffineMap id = AffineMap::getMultiDimIdentityMap(srcRank, ctx);
    for (Operation *op : links) {
        Location loc = op->getLoc();
        auto sourceType = cast<RankedTensorType>(source.getType());
        if (auto transpose = dyn_cast<linalg::TransposeOp>(op)) {
            // Output dims 2 and 3 are source dims 1 and 2.
            auto perm = transpose.getPermutation();
            SmallVector<int64_t> smallPerm = {0, perm[2] - 1, perm[3] - 1};
            Value init = tensor::EmptyOp::create(
                rewriter, loc, applyPermutation(sourceType.getShape(), smallPerm),
                sourceType.getElementType()
            );
            source =
                linalg::TransposeOp::create(rewriter, loc, source, init, smallPerm).getResult()[0];
            continue;
        }
        auto generic = cast<linalg::GenericOp>(op);
        Value init = tensor::EmptyOp::create(
            rewriter, loc, sourceType.getShape(), getElementTypeOrSelf(generic.getResult(0))
        );
        SmallVector<Value> inputs;
        SmallVector<AffineMap> maps;
        for (OpOperand *input : generic.getDpsInputOperands()) {
            AffineMap map = generic.getMatchingIndexingMap(input);
            bool isData = map.isIdentity();
            inputs.push_back(isData ? source : input->get());
            maps.push_back(isData ? id : AffineMap::get(srcRank, 0, map.getResults(), ctx));
        }
        maps.push_back(id);
        auto replayed = linalg::GenericOp::create(
            rewriter, loc, TypeRange{init.getType()}, inputs, ValueRange{init}, maps,
            SmallVector<utils::IteratorType>(srcRank, utils::IteratorType::parallel)
        );
        rewriter.cloneRegionBefore(
            generic.getRegion(), replayed.getRegion(), replayed.getRegion().end()
        );
        source = replayed.getResult(0);
    }
    return source;
}

/// Check if a batch_matmul input comes from a broadcast-batch pattern:
///   collapse_shape(links(linalg.generic(broadcast original [1,K,N] → [1,B,K,N])))
/// where `links` is a possibly empty single-use chain of ops that commute with
/// the broadcast (see getBroadcastCommutingLinkInput), e.g. the K^T transpose
/// and scalar scale of a grouped-query attention key.
/// Returns the broadcast input index (0 or 1), or -1 if no match.
/// Also returns the original un-broadcast source via `broadcastSource` and the
/// links in data-flow order via `links`.
static int detectBroadcastBatchInput(
    linalg::BatchMatmulOp srcOp, Value &broadcastSource, SmallVectorImpl<Operation *> &links
) {
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
        // walk up the broadcast-commuting links to the broadcast
        links.clear();
        Value value = collapseOp.getSrc();
        while (Operation *def = value.getDefiningOp()) {
            Value linkInput = getBroadcastCommutingLinkInput(def);
            if (!linkInput || !value.hasOneUse())
                break;
            links.push_back(def);
            value = linkInput;
        }
        std::reverse(links.begin(), links.end());
        // source of the chain should be a broadcast linalg.generic
        auto genericOp = value.getDefiningOp<linalg::GenericOp>();
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
        // the links may only move dims 2 and 3, so the broadcast must be over
        // dim 1 alone: [1,K,N] -> [1,B,K,N], and nothing else may read it
        if (!links.empty()) {
            MLIRContext *ctx = genericOp.getContext();
            if (!value.hasOneUse() || !isYieldedBlockArgument(genericOp, 0) ||
                !maps[1].isIdentity() ||
                maps[0] != AffineMap::get(
                               4, 0,
                               {getAffineDimExpr(0, ctx), getAffineDimExpr(2, ctx),
                                getAffineDimExpr(3, ctx)},
                               ctx
                           ))
                continue;
        }

        broadcastSource = genericOp.getInputs()[0];
        return i;
    }
    return -1;
}

namespace {
/// The repeated-head cache chain above a batch_matmul operand, the source of
/// the un-broadcast cache, and its K-form links (data-flow order).
struct HeadBroadcastChain {
    Value smallSource;
    int64_t group = 0;
    SmallVector<Operation *, 2> links;
};

/// The splat-scale link of the K-form stack: a rank-4 elementwise generic
/// with a single identity data input (block arg 0), the remaining inputs
/// constants read at fixed indices, its init never read, and a body of one
/// `arith.mulf` of the data and the first constant input, then the yield.
static bool isSplatScaleLink(Operation *op) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic || !getBroadcastCommutingLinkInput(op))
        return false;
    auto &ops = generic.getBody()->getOperations();
    if (!llvm::hasNItems(ops, 2))
        return false;
    auto mulf = dyn_cast<arith::MulFOp>(&*ops.begin());
    auto yield = dyn_cast<linalg::YieldOp>(std::prev(ops.end()));
    if (!mulf || !yield || yield.getOperand(0) != mulf)
        return false;
    auto dataArg = dyn_cast<BlockArgument>(mulf.getLhs());
    auto constArg = dyn_cast<BlockArgument>(mulf.getRhs());
    return dataArg && dataArg.getArgNumber() == 0 && constArg && constArg.getArgNumber() == 1;
}

/// Check if a batch_matmul input is the small head-broadcast cache: a rank-4
/// [1,KV,S,D] tensor repeated over a group of G query heads by an identity
/// broadcast generic, then the merge collapse into the head batch, and either
///   K form: the observed link stack (K^T transpose perm [0,1,3,2], then the
///           splat scale) on the expanded heads, then the batch collapse into
///           the matmul;
///   V form: the batch collapse into the matmul directly.
/// KV may be 1. Only RHS cache chains with single-token LHS operands match;
/// grouping query heads into M then preserves a plain batch matmul.
bool detectHeadBroadcastChain(linalg::BatchMatmulOp matmul, HeadBroadcastChain &chain) {
    MLIRContext *ctx = matmul.getContext();
    auto gExpr = getAffineDimExpr(1, ctx);
    auto headMap = AffineMap::get(
        5, 0, {getAffineDimExpr(0, ctx), gExpr, getAffineDimExpr(3, ctx), getAffineDimExpr(4, ctx)},
        ctx
    );
    auto mergeReassoc = SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}, {4}};
    auto vBatchReassoc = SmallVector<ReassociationIndices>{{0, 1, 2}, {3}, {4}};
    auto kBatchReassoc = SmallVector<ReassociationIndices>{{0, 1}, {2}, {3}};

    Value input = matmul.getInputs()[1];
    auto collapseOp = input.getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp || collapseOp.getReassociationIndices().size() != 3 ||
        ShapedType::isDynamicShape(collapseOp.getSrcType().getShape()))
        return false;
    // The rank-5 result of the head broadcast [1,KV,S,D] -> [1,KV,G,S,D]
    // (group dim inserted between the cache head and the last two dims) and,
    // for the K form, the merge collapse between it and the links.
    tensor::CollapseShapeOp mergeOp;
    Value broadcastResult;
    if (collapseOp.getReassociationIndices() == vBatchReassoc &&
        collapseOp.getSrcType().getRank() == 5) {
        // V form: the batch collapse reads the broadcast directly.
        if (collapseOp.getSrcType().getDimSize(0) != 1 || !collapseOp->hasOneUse())
            return false;
        broadcastResult = collapseOp.getSrc();
    }
    else if (collapseOp.getReassociationIndices() == kBatchReassoc &&
             collapseOp.getSrcType().getRank() == 4 && collapseOp.getSrcType().getDimSize(0) == 1) {
        // K form: the observed link stack, verbatim: the K^T transpose
        // (perm [0,1,3,2]) followed by the splat scale, each single-use,
        // above the merge collapse.
        Value chainTop = collapseOp.getSrc();
        Operation *scaleOp = chainTop.getDefiningOp();
        if (!chainTop.hasOneUse() || !scaleOp || !isSplatScaleLink(scaleOp))
            return false;
        Value transposeOut = cast<linalg::GenericOp>(scaleOp).getInputs()[0];
        auto transpose = dyn_cast_or_null<linalg::TransposeOp>(transposeOut.getDefiningOp());
        if (!transposeOut.hasOneUse() || !transpose ||
            !transpose.getPermutation().equals({0, 1, 3, 2}))
            return false;
        Value mergeOut = transpose.getInput();
        if (!mergeOut.hasOneUse())
            return false;
        mergeOp = dyn_cast_or_null<tensor::CollapseShapeOp>(mergeOut.getDefiningOp());
        if (!mergeOp || !mergeOp->hasOneUse() ||
            mergeOp.getReassociationIndices() != mergeReassoc ||
            mergeOp.getSrcType().getRank() != 5 || mergeOp.getSrcType().getDimSize(0) != 1)
            return false;
        chain.links.push_back(transpose);
        chain.links.push_back(scaleOp);
        broadcastResult = mergeOp.getSrc();
    }
    else {
        return false;
    }
    auto broadcast = broadcastResult.getDefiningOp<linalg::GenericOp>();
    if (!broadcast || !broadcastResult.hasOneUse() || broadcast.getNumDpsInputs() != 1 ||
        broadcast.getNumResults() != 1 || broadcast.getNumLoops() != 5 ||
        !broadcast.isAllParallelLoops() || broadcast.getBody()->getOperations().size() != 1 ||
        !isYieldedBlockArgument(broadcast, 0))
        return false;
    auto maps = broadcast.getIndexingMapsArray();
    if (maps.size() != 2 || maps[0] != headMap || !maps[1].isIdentity())
        return false;
    auto srcType = dyn_cast<RankedTensorType>(broadcast.getInputs()[0].getType());
    auto resType = dyn_cast<RankedTensorType>(broadcast.getResult(0).getType());
    if (!srcType || !resType || srcType.getRank() != 4 || resType.getRank() != 5 ||
        !srcType.hasStaticShape() || !resType.hasStaticShape() || srcType.getShape()[0] != 1 ||
        resType.getShape()[0] != 1 || resType.getShape()[1] != srcType.getShape()[1] ||
        resType.getShape()[2] <= 1 ||
        resType.getShape().drop_front(3) != srcType.getShape().drop_front(2))
        return false;
    int64_t kv = srcType.getShape()[1];
    int64_t group = resType.getShape()[2];
    int64_t batch = kv * group;
    // Single-token decode: the other operand is [H,1,.] and the matmul's
    // head batch is exactly the expanded batch.
    auto otherType = dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
    auto outType = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!otherType || !outType || otherType.getRank() != 3 || outType.getRank() != 3 ||
        ShapedType::isDynamicShape(otherType.getShape()) ||
        ShapedType::isDynamicShape(outType.getShape()) || otherType.getShape()[0] != batch ||
        outType.getShape()[0] != batch || outType.getShape()[1] != 1 ||
        srcType.getElementType() != otherType.getElementType() ||
        srcType.getElementType() != outType.getElementType() ||
        (!srcType.getElementType().isF32() && !srcType.getElementType().isBF16()))
        return false;
    chain.smallSource = broadcast.getInputs()[0];
    chain.group = group;
    return true;
}
} // namespace

/// Check if a batch_matmul input comes *directly* (no intervening collapse_shape)
/// from a broadcast linalg.generic that broadcasts a 2-D [K,N] value over the batch
/// dim to [B,K,N] with B >= 2:
///   linalg.generic { maps = [(d0,d1,d2)->(d1,d2), (d0,d1,d2)->(d0,d1,d2)] }
///                  ins([K,N]) outs([B,K,N]) { yield %in }
/// This is the form expandConstantInput leaves behind for a constant weight once the
/// #1767 fix declines to rewrite the broadcast as an (invalid) expand_shape. Returns
/// the broadcast input index (0 or 1), or -1; the un-broadcast 2-D source is returned
/// via `broadcastSource`.
static int detectDirectBatchBroadcastInput(linalg::BatchMatmulOp srcOp, Value &broadcastSource) {
    for (int i : {0, 1}) {
        Value input = srcOp.getInputs()[i];
        auto genericOp = input.getDefiningOp<linalg::GenericOp>();
        if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1)
            continue;
        // identity body: only a yield of the single input block argument
        if (genericOp.getBody()->getOperations().size() != 1 ||
            !isYieldedBlockArgument(genericOp, 0))
            continue;
        auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
        auto outputType = dyn_cast<RankedTensorType>(genericOp.getResultTypes()[0]);
        if (!inputType || !outputType || inputType.getRank() != 2 || outputType.getRank() != 3 ||
            ShapedType::isDynamicShape(outputType.getShape()))
            continue;
        // output must be [B,K,N] with B >= 2 and trailing [K,N] == input
        if (outputType.getShape()[0] < 2 ||
            outputType.getShape().drop_front() != inputType.getShape())
            continue;
        // maps: input drops the batch dim ((d0,d1,d2)->(d1,d2)); output is identity
        auto maps = genericOp.getIndexingMapsArray();
        if (maps.size() != 2 || !isFinalMatmulRhsBroadcastMap(maps[0]) ||
            !isIdentityMap(maps[1], 3))
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
    auto b = getAffineDimExpr(0, ctx);
    auto m = getAffineDimExpr(1, ctx);
    auto n = getAffineDimExpr(2, ctx);
    auto k = getAffineDimExpr(3, ctx);
    auto zero = getAffineConstantExpr(0, ctx);
    if (broadcastIdx == 1) {
        // lhs = A[b,m,k], rhs = the small operand
        lhsMap = AffineMap::get(4, 0, {b, m, k}, ctx);
        rhsMap = AffineMap::get(4, 0, {zero, k, n}, ctx);
    }
    else {
        // lhs = the small operand, rhs = B[b,k,n]
        lhsMap = AffineMap::get(4, 0, {zero, m, k}, ctx);
        rhsMap = AffineMap::get(4, 0, {b, k, n}, ctx);
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
    genericOp->setAttr(TORQ_BROADCAST_MATMUL, rewriter.getUnitAttr());

    rewriter.replaceOp(srcOp, genericOp.getResults());
    return success();
}

/// At the IREE 3.12 pin the unit batch of a [1,M,K] x [K,N] matmul is collapsed
/// before this pass runs. Downstream patterns (the block-quant input rewriting
/// here, MixedQuantMatmulRewrite in LinalgToTorqHL) only recognize
/// linalg.batch_matmul, so restore the canonical rank-3 form and let them fire
/// as they did at 3.10. Two rank-2 forms are recognized:
/// (a) the result's only user is an expand_shape back to [1,M,N] -- replace the
///     expand with the batch result;
/// (b) the weight arrives as the collapse of a rank-3 blocked dequant chain
///     (the mixed-quant signature) and the result feeds other reshapes (e.g.
///     the attention-heads split) -- insert a collapse back to [M,N] after the
///     batch result for the original consumers.
/// Any other rank-2 matmul is a genuine GEMM and is left alone.
class RaiseUnitBatchMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const {
        Value result = srcOp.getResult(0);
        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
        auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
        auto matmulType = dyn_cast<RankedTensorType>(result.getType());
        if (!lhsType || !rhsType || !matmulType || lhsType.getRank() != 2 ||
            rhsType.getRank() != 2 || ShapedType::isDynamicShape(lhsType.getShape()) ||
            ShapedType::isDynamicShape(rhsType.getShape()) ||
            ShapedType::isDynamicShape(matmulType.getShape()))
            return failure();

        tensor::ExpandShapeOp expandOp;
        if (result.hasOneUse())
            expandOp = dyn_cast<tensor::ExpandShapeOp>(*result.user_begin());

        bool matchedUnitBatchExpand = false;
        if (expandOp) {
            auto resultType = dyn_cast<RankedTensorType>(expandOp.getResultType());
            auto reassoc = expandOp.getReassociationIndices();
            matchedUnitBatchExpand =
                resultType && resultType.getRank() == 3 && resultType.getShape()[0] == 1 &&
                !ShapedType::isDynamicShape(resultType.getShape()) &&
                resultType.getShape().drop_front() == matmulType.getShape() &&
                reassoc.size() == 2 && reassoc[0] == ReassociationIndices{0, 1} &&
                reassoc[1] == ReassociationIndices{2};
        }

        bool matchedBlockedWeight = false;
        if (!matchedUnitBatchExpand) {
            if (auto collapse = rhs.getDefiningOp<tensor::CollapseShapeOp>()) {
                auto srcType = dyn_cast<RankedTensorType>(collapse.getSrcType());
                matchedBlockedWeight =
                    srcType && srcType.getRank() == 3 &&
                    collapse.getSrc().getDefiningOp<linalg::GenericOp>() != nullptr;
            }
        }

        // The per-tensor dequant matcher only recognizes the [1,K,N] batch
        // form; raise the rank-2 matmul 3.12 collapsed to so it can fire (a
        // decode matmul then lowers to mixed_quant_matmul).
        bool matchedPerTensorDequantWeight = false;
        if (!matchedUnitBatchExpand && !matchedBlockedWeight) {
            PerTensorDequantMatch dequant;
            matchedPerTensorDequantWeight = succeeded(
                matchConstantPerTensorDequant(rhs.getDefiningOp<linalg::GenericOp>(), dequant)
            );
        }

        if (!matchedUnitBatchExpand && !matchedBlockedWeight && !matchedPerTensorDequantWeight)
            return failure();

        Location loc = srcOp.getLoc();
        int64_t M = matmulType.getShape()[0];
        int64_t N = matmulType.getShape()[1];
        int64_t K = lhsType.getShape()[1];
        SmallVector<ReassociationIndices> batchReassoc = {{0, 1}, {2}};

        auto expandToBatch = [&](Value v, int64_t d1, int64_t d2) {
            auto elemType = cast<RankedTensorType>(v.getType()).getElementType();
            auto expType = RankedTensorType::get({1, d1, d2}, elemType);
            return tensor::ExpandShapeOp::create(rewriter, loc, expType, v, batchReassoc)
                .getResult();
        };

        Value lhs3 = expandToBatch(lhs, M, K);
        Value rhs3 = expandToBatch(rhs, K, N);
        Value init3 = expandToBatch(srcOp.getDpsInits()[0], M, N);
        auto batchType = RankedTensorType::get({1, M, N}, matmulType.getElementType());
        auto batchOp = linalg::BatchMatmulOp::create(
            rewriter, loc, TypeRange{batchType}, ValueRange{lhs3, rhs3}, ValueRange{init3}
        );

        if (matchedUnitBatchExpand) {
            rewriter.replaceOp(expandOp, batchOp.getResult(0));
        }
        else {
            auto collapsed = tensor::CollapseShapeOp::create(
                rewriter, loc, matmulType, batchOp.getResult(0), batchReassoc
            );
            rewriter.replaceAllUsesWith(result, collapsed.getResult());
        }
        rewriter.eraseOp(srcOp);
        return success();
    }
};

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

        // Only the broadcast folds below assume standard batch/m/n/k roles.
        bool hasDefaultMaps = srcOp.getIndexingMapsArray() ==
                              linalg::BatchMatmulOp::getDefaultIndexingMaps(srcOp.getContext());

        Value input0 = srcOp.getInputs()[0];
        Value input1 = srcOp.getInputs()[1];
        Value origInput0 = input0;
        Value origInput1 = input1;

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
        SmallVector<Operation *> links;
        int broadcastIdx =
            hasDefaultMaps ? detectBroadcastBatchInput(srcOp, broadcastSource, links) : -1;
        if (broadcastIdx >= 0) {
            // Only apply when the broadcast source is rank 3 ([1,K,N]).
            // Rank-2 sources ([K,N]) would produce a linalg.generic with a
            // rank-2 operand that the TORQ backend cannot convert back to
            // torq_hl::MatMulOp (findBatchMatmulBcastIdx expects 3-result maps).
            auto broadcastType = dyn_cast<RankedTensorType>(broadcastSource.getType());
            if (broadcastType && broadcastType.getRank() == 3) {
                rewriter.setInsertionPoint(srcOp);
                broadcastSource = replayLinksBeforeBroadcast(broadcastSource, links, rewriter);
                return replaceBatchMatmulWithBroadcastGeneric(
                    srcOp, broadcastSource, broadcastIdx, rewriter
                );
            }
        }

        // Detect the repeated-head cache form: a [1,KV,S,D] cache head tensor
        // broadcast over a group of G query heads (KV > 1: grouped heads; KV == 1:
        // one head shared by all), expanded to [1,KV,G,S,D] and collapsed into
        // the matmul's head batch.
        if (hasDefaultMaps && clHeadBroadcastMode == HeadBroadcastMode::Enable) {
            HeadBroadcastChain chain;
            if (detectHeadBroadcastChain(srcOp, chain)) {
                rewriter.setInsertionPoint(srcOp);
                Location loc = srcOp.getLoc();
                auto cacheType = cast<RankedTensorType>(chain.smallSource.getType());
                int64_t kv = cacheType.getDimSize(1);
                auto elemType = cacheType.getElementType();
                Value cache = tensor::CollapseShapeOp::create(
                    rewriter, loc,
                    RankedTensorType::get(cacheType.getShape().drop_front(), elemType),
                    chain.smallSource, ArrayRef<ReassociationIndices>{{0, 1}, {2}, {3}}
                );
                cache = replayLinksBeforeBroadcast(cache, chain.links, rewriter);
                // [KV*G,1,K] -> [KV,G,K], preserving head order. Reshape the
                // accumulator identically so nonzero init values are preserved.
                auto groupHeads = [&](Value value) -> Value {
                    auto type = cast<RankedTensorType>(value.getType());
                    int64_t width = type.getDimSize(2);
                    Value expanded = tensor::ExpandShapeOp::create(
                        rewriter, loc, RankedTensorType::get({kv, chain.group, 1, width}, elemType),
                        value, ArrayRef<ReassociationIndices>{{0, 1}, {2}, {3}}
                    );
                    return tensor::CollapseShapeOp::create(
                        rewriter, loc, RankedTensorType::get({kv, chain.group, width}, elemType),
                        expanded, ArrayRef<ReassociationIndices>{{0}, {1, 2}, {3}}
                    );
                };
                Value lhs = groupHeads(srcOp.getInputs()[0]);
                Value init = groupHeads(srcOp.getDpsInits()[0]);
                Value result = linalg::BatchMatmulOp::create(
                                   rewriter, loc, ValueRange{lhs, cache}, ValueRange{init}
                )
                                   .getResult(0);
                auto resultType = cast<RankedTensorType>(srcOp.getResult(0).getType());
                Value expanded = tensor::ExpandShapeOp::create(
                    rewriter, loc,
                    RankedTensorType::get({kv, chain.group, 1, resultType.getDimSize(2)}, elemType),
                    result, ArrayRef<ReassociationIndices>{{0}, {1, 2}, {3}}
                );
                rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
                    srcOp, resultType, expanded, ArrayRef<ReassociationIndices>{{0, 1}, {2}, {3}}
                );
                return success();
            }
        }

        // Constant-weight batch broadcast left in place by expandConstantInput (the
        // #1767 fix): the matmul input is directly a [K,N] -> [B,K,N] broadcast generic.
        // Absorb it into the matmul via indexing maps instead of materializing the
        // [B,K,N] buffer, by first expanding the source [K,N] -> [1,K,N] (a zero-copy
        // reshape) and reusing the rank-3 broadcast path -- which emits a 3-result
        // (0,k,n) map the TORQ backend (findBatchMatmulBcastIdx) already lowers.
        Value directSource;
        int directIdx = hasDefaultMaps ? detectDirectBatchBroadcastInput(srcOp, directSource) : -1;
        if (directIdx >= 0) {
            auto srcType = cast<RankedTensorType>(directSource.getType());
            SmallVector<int64_t> expShape = {1, srcType.getShape()[0], srcType.getShape()[1]};
            auto expType = RankedTensorType::get(expShape, srcType.getElementType());
            SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}};
            Value expanded = tensor::ExpandShapeOp::create(
                rewriter, srcOp.getLoc(), expType, directSource, reassoc
            );
            return replaceBatchMatmulWithBroadcastGeneric(srcOp, expanded, directIdx, rewriter);
        }

        // Idempotence: if no input was actually changed do not touch the op.
        if (input0 == origInput0 && input1 == origInput1)
            return failure();

        rewriter.modifyOpInPlace(srcOp, [&]() {
            srcOp->setOperand(0, input0);
            srcOp->setOperand(1, input1);
        });

        srcOp->setAttr("matmulOptimized", BoolAttr::get(srcOp->getContext(), true));

        return success();
    }
};

// torch-mlir (ConvertAtenMatmulOp) accumulates a bf16 matmul in f32 and truncates the result to
// bf16 in a separate generic. The torq matmul can write bf16 directly, so the f32 result only
// adds a full f32 tensor between the two ops and a separate conversion pass. Write bf16 from the
// matmul instead.
class FoldTruncfIntoFloatMatmulPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp truncOp, PatternRewriter &rewriter) const {
        if (truncOp.getNumDpsInputs() != 1 || truncOp.getNumDpsInits() != 1 ||
            !linalg::isElementwise(truncOp) ||
            !llvm::all_of(truncOp.getIndexingMapsArray(), [](AffineMap m) {
                return m.isIdentity();
            })) {
            return failure();
        }
        Block *body = truncOp.getBody();
        auto truncf = body->getTerminator()->getOperand(0).getDefiningOp<arith::TruncFOp>();
        if (!truncf || truncf.getIn() != body->getArgument(0) ||
            !llvm::hasNItems(body->getOperations(), 2)) {
            return failure();
        }

        Value acc = truncOp.getDpsInputOperand(0)->get();
        auto matmul = acc.getDefiningOp<linalg::LinalgOp>();
        if (!matmul || !isa<linalg::MatmulOp, linalg::BatchMatmulOp>(matmul) || !acc.hasOneUse()) {
            return failure();
        }
        auto outType = cast<RankedTensorType>(truncOp.getResult(0).getType());
        if (!getElementTypeOrSelf(acc.getType()).isF32() || !outType.getElementType().isBF16() ||
            !outType.hasStaticShape() || !llvm::all_of(matmul.getDpsInputs(), [](Value v) {
                return getElementTypeOrSelf(v.getType()).isBF16();
            })) {
            return failure();
        }
        auto fill = matmul.getDpsInitOperand(0)->get().getDefiningOp<linalg::FillOp>();
        if (!fill || !fill->hasOneUse() || !matchPattern(fill.value(), m_AnyZeroFloat())) {
            return failure();
        }

        rewriter.setInsertionPoint(matmul);
        Location loc = matmul.getLoc();
        Value zero = arith::ConstantOp::create(
            rewriter, loc, rewriter.getZeroAttr(outType.getElementType())
        );
        Value empty =
            tensor::EmptyOp::create(rewriter, loc, outType.getShape(), outType.getElementType());
        Value init = linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
        // Build a new op rather than cloning: the body of a named op follows its result type.
        SmallVector<NamedAttribute> attrs =
            llvm::to_vector(llvm::make_filter_range(matmul->getAttrs(), [](NamedAttribute attr) {
                return attr.getName() != "operandSegmentSizes";
            }));
        SmallVector<Value> inputs(matmul.getDpsInputs());
        Value bf16Matmul;
        if (isa<linalg::MatmulOp>(matmul)) {
            bf16Matmul =
                linalg::MatmulOp::create(rewriter, loc, TypeRange{outType}, inputs, init, attrs)
                    .getResult(0);
        }
        else {
            bf16Matmul = linalg::BatchMatmulOp::create(
                             rewriter, loc, TypeRange{outType}, inputs, init, attrs
            )
                             .getResult(0);
        }
        rewriter.replaceOp(truncOp, bf16Matmul);
        return success();
    }
};

void populateOptimizeMatmuOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseUnitBatchMatmulPattern>(context);
    patterns.add<BatchMatmulOpPattern>(context);
    patterns.add<FoldTruncfIntoFloatMatmulPattern>(context);
}

} // namespace mlir::syna::torq
