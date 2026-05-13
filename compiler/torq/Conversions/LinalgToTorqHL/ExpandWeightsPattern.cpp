// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include <limits>

namespace mlir::syna::torq {

static bool isIdentity3DMap(AffineMap map) {
    if (!map || map.getNumDims() != 3 || map.getNumSymbols() != 0 || map.getNumResults() != 3) {
        return false;
    }
    auto d0 = getAffineDimExpr(0, map.getContext());
    auto d1 = getAffineDimExpr(1, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    return map.getResult(0) == d0 && map.getResult(1) == d1 && map.getResult(2) == d2;
}

static bool isBlockQuantizedWeightMap(AffineMap map, int64_t blockSize) {
    if (!map || map.getNumDims() != 3 || map.getNumSymbols() != 0 || map.getNumResults() != 3 ||
        blockSize <= 0) {
        return false;
    }
    auto d1 = getAffineDimExpr(1, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    auto blockSizeExpr = getAffineConstantExpr(blockSize, map.getContext());
    return map.getResult(0) == d1 && map.getResult(1) == d2.floorDiv(blockSizeExpr) &&
           map.getResult(2) == d2 % blockSizeExpr;
}

static bool isBlockQuantizedParamMap(
    AffineMap map, RankedTensorType type, int64_t rows, int64_t groups, int64_t blockSize
) {
    if (!map || !type || map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
        map.getNumResults() != 3 || blockSize <= 0 || ShapedType::isDynamicShape(type.getShape()) ||
        type.getRank() != 3 || type.getShape()[0] != rows || type.getShape()[1] != groups ||
        type.getShape()[2] != 1) {
        return false;
    }

    auto d1 = getAffineDimExpr(1, map.getContext());
    auto d2 = getAffineDimExpr(2, map.getContext());
    auto groupExpr = d2.floorDiv(getAffineConstantExpr(blockSize, map.getContext()));
    return map.getResult(0) == d1 && map.getResult(1) == groupExpr &&
           map.getResult(2) == getAffineConstantExpr(0, map.getContext());
}

static arith::ConstantOp getConstantLikeTensorOp(Value value) {
    if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
        return constOp;
    }

    auto sliceOp = value.getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp) {
        return {};
    }
    return sliceOp.getSource().getDefiningOp<arith::ConstantOp>();
}

// Some linalg forms materialize a per-block zero point as a full-block splat.
// If the constant is a SplatElementsAttr, every lane has the same value, so it
// can be represented by the compact [rows, groups, 1] shape expected by
// ExpandWeightsOp.
static Value compactFullBlockSplatParam(
    Location loc, Value value, int64_t blockSize, PatternRewriter &rewriter
) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 3 || ShapedType::isDynamicShape(type.getShape()) ||
        type.getShape()[2] != blockSize) {
        return value;
    }

    auto constOp = getConstantLikeTensorOp(value);
    auto splatAttr =
        constOp ? dyn_cast<SplatElementsAttr>(constOp.getValue()) : SplatElementsAttr();
    if (!splatAttr) {
        return value;
    }

    auto compactType = RankedTensorType::get(
        {type.getShape()[0], type.getShape()[1], 1}, type.getElementType(), type.getEncoding()
    );
    auto compactAttr = SplatElementsAttr::get(compactType, splatAttr.getSplatValue<Attribute>());
    return arith::ConstantOp::create(rewriter, loc, compactType, compactAttr);
}

static FailureOr<Value> createInputZeroPointBiasScale(
    Location loc, Value zeroPoints, RankedTensorType outputType, PatternRewriter &rewriter,
    linalg::GenericOp anchorOp
) {
    auto type = dyn_cast<RankedTensorType>(zeroPoints.getType());
    if (!type || type.getRank() != 3 || ShapedType::isDynamicShape(type.getShape()) ||
        type.getShape()[2] != 1) {
        return failure();
    }

    auto constOp = getConstantLikeTensorOp(zeroPoints);
    if (!constOp) {
        return failure();
    }

    Type outputElementType = outputType.getElementType();
    if (outputElementType.isBF16() || outputElementType.isF32()) {
        SmallVector<APFloat> biasValues;
        biasValues.reserve(type.getShape()[0] * type.getShape()[1]);
        if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
            for (APInt value : denseAttr.getValues<APInt>()) {
                biasValues.emplace_back(static_cast<float>(-value.getSExtValue()));
            }
        }
        else if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
            for (APFloat value : denseAttr.getValues<APFloat>()) {
                bool ignored;
                value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &ignored);
                value.changeSign();
                biasValues.push_back(value);
            }
        }
        else {
            return failure();
        }

        return createFConst(
                   rewriter, anchorOp, biasValues, {type.getShape()[0], type.getShape()[1], 1}
        )
            .getResult();
    }

    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
        SmallVector<int32_t> biasScaleValues;
        biasScaleValues.reserve(type.getShape()[0] * type.getShape()[1] * 2);
        for (APInt value : denseAttr.getValues<APInt>()) {
            biasScaleValues.push_back(static_cast<int32_t>(-value.getSExtValue()));
            biasScaleValues.push_back(1);
        }

        auto biasScaleType = RankedTensorType::get(
            {type.getShape()[0], type.getShape()[1], 2}, rewriter.getI32Type(), type.getEncoding()
        );
        auto biasScaleAttr = DenseIntElementsAttr::get(biasScaleType, biasScaleValues);
        return arith::ConstantOp::create(rewriter, loc, biasScaleType, biasScaleAttr).getResult();
    }

    return failure();
}

static BlockArgument getWeightArgFromSIToFP(arith::SIToFPOp sitofpOp) {
    if (!sitofpOp) {
        return {};
    }
    if (auto weightArg = dyn_cast<BlockArgument>(sitofpOp.getIn())) {
        return weightArg;
    }
    auto weightExt = sitofpOp.getIn().getDefiningOp<arith::ExtSIOp>();
    return weightExt ? dyn_cast<BlockArgument>(weightExt.getIn()) : BlockArgument();
}

static BlockArgument getIntegerBlockArg(Value value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
        return arg;
    }
    if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
        return dyn_cast<BlockArgument>(extOp.getIn());
    }
    if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
        return dyn_cast<BlockArgument>(extOp.getIn());
    }
    return {};
}

static bool matchBlockQuantizedIntegerBody(linalg::GenericOp genericOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return false;
    }

    auto muliOp = yieldOp.getOperand(0).getDefiningOp<arith::MulIOp>();
    if (!muliOp) {
        return false;
    }
    auto scaleArg = getIntegerBlockArg(muliOp.getRhs());
    if (!scaleArg || scaleArg.getArgNumber() != 1) {
        return false;
    }

    auto subOp = muliOp.getLhs().getDefiningOp<arith::SubIOp>();
    if (!subOp) {
        return false;
    }
    auto weightArg = getIntegerBlockArg(subOp.getLhs());
    auto zpArg = getIntegerBlockArg(subOp.getRhs());
    return weightArg && zpArg && weightArg.getArgNumber() == 0 && zpArg.getArgNumber() == 2;
}

static bool matchBlockQuantizedFloatBody(linalg::GenericOp genericOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return false;
    }

    auto mulfOp = yieldOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulfOp) {
        return false;
    }
    auto scaleArg = dyn_cast<BlockArgument>(mulfOp.getRhs());
    if (!scaleArg || scaleArg.getArgNumber() != 1) {
        return false;
    }

    if (auto subfOp = mulfOp.getLhs().getDefiningOp<arith::SubFOp>()) {
        auto weightArg = getWeightArgFromSIToFP(subfOp.getLhs().getDefiningOp<arith::SIToFPOp>());
        auto zpArg = dyn_cast<BlockArgument>(subfOp.getRhs());
        return weightArg && zpArg && weightArg.getArgNumber() == 0 && zpArg.getArgNumber() == 2;
    }

    auto sitofpOp = mulfOp.getLhs().getDefiningOp<arith::SIToFPOp>();
    if (!sitofpOp) {
        return false;
    }
    auto subOp = sitofpOp.getIn().getDefiningOp<arith::SubIOp>();
    if (!subOp) {
        return false;
    }
    auto weightArg = getIntegerBlockArg(subOp.getLhs());
    auto zpArg = getIntegerBlockArg(subOp.getRhs());
    return weightArg && zpArg && weightArg.getArgNumber() == 0 && zpArg.getArgNumber() == 2;
}

static IntegerAttr getBlockSizeAttr(MLIRContext *context, Value packedWeights) {
    auto packedWeightType = dyn_cast<RankedTensorType>(packedWeights.getType());
    if (!packedWeightType || packedWeightType.getRank() != 3 ||
        ShapedType::isDynamicShape(packedWeightType.getShape())) {
        return {};
    }

    int64_t blockSize = packedWeightType.getShape()[2];
    if (blockSize <= 0 || blockSize > std::numeric_limits<int32_t>::max()) {
        return {};
    }
    return IntegerAttr::get(IntegerType::get(context, 32), blockSize);
}

static bool isBlockQuantizedExpandWeightsOp(linalg::GenericOp genericOp) {
    if (!genericOp || genericOp.getNumDpsInputs() != 3 || genericOp.getNumDpsInits() != 1) {
        return false;
    }

    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
    bool isFloatDequant =
        resultType && (resultType.getElementType().isBF16() || resultType.getElementType().isF32());
    bool isIntegerDequant = resultType && resultType.getElementType().isInteger(8);
    if (!resultType || resultType.getRank() != 3 || resultType.getShape()[0] != 1 ||
        (!isFloatDequant && !isIntegerDequant) ||
        ShapedType::isDynamicShape(resultType.getShape())) {
        return false;
    }

    auto weightType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto scaleType = dyn_cast<RankedTensorType>(genericOp.getInputs()[1].getType());
    auto zeroPointType = dyn_cast<RankedTensorType>(genericOp.getInputs()[2].getType());
    if (!weightType || !scaleType || !zeroPointType || weightType.getRank() != 3 ||
        ShapedType::isDynamicShape(weightType.getShape())) {
        return false;
    }

    auto weightElementType = weightType.getElementType();
    if (!weightElementType.isInteger(4)) {
        return false;
    }
    if (isFloatDequant) {
        if (scaleType.getElementType() != resultType.getElementType()) {
            return false;
        }
        if (!zeroPointType.getElementType().isInteger() &&
            zeroPointType.getElementType() != resultType.getElementType()) {
            return false;
        }
    }
    else if (!scaleType.getElementType().isInteger(8) ||
             !zeroPointType.getElementType().isInteger(8)) {
        return false;
    }

    auto resultShape = resultType.getShape();
    auto weightShape = weightType.getShape();
    int64_t rows = resultShape[1];
    int64_t cols = resultShape[2];
    int64_t groups = weightShape[1];
    int64_t blockSize = weightShape[2];
    if (weightShape[0] != rows || groups <= 0 || blockSize <= 0 || groups * blockSize != cols) {
        return false;
    }

    auto indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 4 || !isBlockQuantizedWeightMap(indexingMaps[0], blockSize) ||
        !isBlockQuantizedParamMap(indexingMaps[1], scaleType, rows, groups, blockSize) ||
        !isBlockQuantizedParamMap(indexingMaps[2], zeroPointType, rows, groups, blockSize) ||
        !isIdentity3DMap(indexingMaps[3])) {
        return false;
    }

    return isFloatDequant ? matchBlockQuantizedFloatBody(genericOp)
                          : matchBlockQuantizedIntegerBody(genericOp);
}

static FailureOr<Value> buildBlockQuantizedExpandWeightsOp(
    Location loc, Value packedWeights, Value scales, Value zeroPoints, RankedTensorType outputType,
    PatternRewriter &rewriter, linalg::GenericOp anchorOp
) {
    Type elementType = outputType.getElementType();
    if (!elementType.isInteger(8) && !elementType.isBF16() && !elementType.isF32()) {
        return failure();
    }

    IntegerAttr blockSizeAttr = getBlockSizeAttr(rewriter.getContext(), packedWeights);
    if (!blockSizeAttr) {
        return failure();
    }
    int64_t blockSize = blockSizeAttr.getInt();
    zeroPoints = compactFullBlockSplatParam(loc, zeroPoints, blockSize, rewriter);
    FailureOr<Value> inputZpBiasScale =
        createInputZeroPointBiasScale(loc, zeroPoints, outputType, rewriter, anchorOp);
    if (failed(inputZpBiasScale)) {
        return failure();
    }

    auto [outMin, outMax] = getDTypeRange(outputType.getElementType());
    auto expandWeightsOp = torq_hl::ExpandWeightsOp::create(
        rewriter, loc, outputType, createInitTensor(anchorOp, rewriter, outputType),
        *inputZpBiasScale, outMin, outMax, scales, packedWeights, static_cast<uint32_t>(blockSize)
    );
    return expandWeightsOp.getOutput();
}

struct ExpandWeightsPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    ExpandWeightsPattern(MLIRContext *context) : OpRewritePattern(context, /*benefit=*/2) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        if (!isBlockQuantizedExpandWeightsOp(genericOp)) {
            return failure();
        }

        auto outputType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
        if (!outputType) {
            return failure();
        }

        FailureOr<Value> expandedWeights = buildBlockQuantizedExpandWeightsOp(
            genericOp.getLoc(), genericOp.getInputs()[0], genericOp.getInputs()[1],
            genericOp.getInputs()[2], outputType, rewriter, genericOp
        );
        if (failed(expandedWeights)) {
            return failure();
        }

        rewriter.replaceOp(genericOp, *expandedWeights);
        return success();
    }
};

void populateLinalgToTorqHLExpandWeightsPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.insert<ExpandWeightsPattern>(context);
}

} // namespace mlir::syna::torq
