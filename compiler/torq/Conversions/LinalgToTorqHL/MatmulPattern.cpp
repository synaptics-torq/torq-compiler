// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <optional>

#define DEBUG_TYPE "linalg-torq-matmul-pattern"

namespace mlir::syna::torq {

namespace {

Value stripTensorShapeCasts(Value value) {
    while (true) {
        if (auto expandOp = value.getDefiningOp<tensor::ExpandShapeOp>()) {
            value = expandOp.getSrc();
            continue;
        }
        if (auto collapseOp = value.getDefiningOp<tensor::CollapseShapeOp>()) {
            value = collapseOp.getSrc();
            continue;
        }
        break;
    }
    return value;
}

arith::ConstantOp getConstantLikeTensorOp(Value value) {
    while (true) {
        if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
            return constOp;
        }

        auto sliceOp = value.getDefiningOp<tensor::ExtractSliceOp>();
        if (!sliceOp) {
            return {};
        }
        value = sliceOp.getSource();
    }
}

Value getIntToFPInput(Value value) {
    if (auto sitofpOp = value.getDefiningOp<arith::SIToFPOp>()) {
        return sitofpOp.getIn();
    }
    if (auto uitofpOp = value.getDefiningOp<arith::UIToFPOp>()) {
        return uitofpOp.getIn();
    }
    return {};
}

BlockArgument getWeightArgFromIntToFP(Value value) {
    Value intInput = getIntToFPInput(value);
    if (!intInput) {
        return {};
    }
    if (auto weightArg = dyn_cast<BlockArgument>(intInput)) {
        return weightArg;
    }
    auto weightExt = intInput.getDefiningOp<arith::ExtSIOp>();
    if (!weightExt) {
        return {};
    }
    return dyn_cast<BlockArgument>(weightExt.getIn());
}

BlockArgument getIntegerBlockArg(Value value) {
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

bool matchMatmulRhsBlockQuantizedBody(linalg::GenericOp genericOp) {
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
        auto weightArg = getWeightArgFromIntToFP(subfOp.getLhs());
        auto zpArg = dyn_cast<BlockArgument>(subfOp.getRhs());
        return weightArg && zpArg && weightArg.getArgNumber() == 0 && zpArg.getArgNumber() == 2;
    }

    Value intToFPInput = getIntToFPInput(mulfOp.getLhs());
    if (!intToFPInput) {
        return false;
    }
    auto subOp = intToFPInput.getDefiningOp<arith::SubIOp>();
    if (!subOp) {
        return false;
    }
    auto weightArg = getIntegerBlockArg(subOp.getLhs());
    auto zpArg = getIntegerBlockArg(subOp.getRhs());
    return weightArg && zpArg && weightArg.getArgNumber() == 0 && zpArg.getArgNumber() == 2;
}

bool usesUnsignedIntegerZeroPoint(linalg::GenericOp genericOp) {
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return false;
    }

    auto mulfOp = yieldOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulfOp) {
        return false;
    }

    Value intToFPInput = getIntToFPInput(mulfOp.getLhs());
    auto subOp = intToFPInput ? intToFPInput.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
    if (!subOp) {
        return false;
    }

    if (subOp.getRhs().getDefiningOp<arith::ExtUIOp>()) {
        return true;
    }

    return false;
}

std::optional<APFloat::Semantics> getFloatSemantics(Type type) {
    if (type.isBF16()) {
        return APFloat::S_BFloat;
    }
    if (type.isF32()) {
        return APFloat::S_IEEEsingle;
    }
    return std::nullopt;
}

uint64_t readLittleEndianBits(ArrayRef<char> data, int64_t elementIndex, unsigned byteWidth) {
    uint64_t bits = 0;
    int64_t offset = elementIndex * byteWidth;
    for (unsigned byte = 0; byte < byteWidth; ++byte) {
        bits |= static_cast<uint64_t>(static_cast<uint8_t>(data[offset + byte])) << (8 * byte);
    }
    return bits;
}

FailureOr<std::vector<APFloat>> getFloatConstantValues(arith::ConstantOp constOp) {
    auto type = dyn_cast<RankedTensorType>(constOp.getType());
    if (!type || ShapedType::isDynamicShape(type.getShape())) {
        return failure();
    }

    if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(constOp.getValue())) {
        std::vector<APFloat> values;
        values.reserve(denseAttr.getNumElements());
        llvm::append_range(values, denseAttr.getValues<APFloat>());
        return values;
    }

    auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(constOp.getValue());
    std::optional<APFloat::Semantics> semantics = getFloatSemantics(type.getElementType());
    if (!resourceAttr || !semantics) {
        return failure();
    }

    unsigned bitWidth = type.getElementType().getIntOrFloatBitWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0 || bitWidth > 64) {
        return failure();
    }
    unsigned byteWidth = bitWidth / 8;
    ArrayRef<char> data = resourceAttr.getData();
    if (data.size() != static_cast<size_t>(type.getNumElements() * byteWidth)) {
        return failure();
    }

    std::vector<APFloat> values;
    values.reserve(type.getNumElements());
    for (int64_t idx = 0, count = type.getNumElements(); idx < count; ++idx) {
        uint64_t bits = readLittleEndianBits(data, idx, byteWidth);
        values.emplace_back(APFloat::EnumToSemantics(*semantics), APInt(bitWidth, bits));
    }
    return values;
}

FailureOr<std::vector<APInt>> getIntegerConstantValues(arith::ConstantOp constOp) {
    auto type = dyn_cast<RankedTensorType>(constOp.getType());
    if (!type || ShapedType::isDynamicShape(type.getShape())) {
        return failure();
    }

    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
        std::vector<APInt> values;
        values.reserve(denseAttr.getNumElements());
        llvm::append_range(values, denseAttr.getValues<APInt>());
        return values;
    }

    auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(constOp.getValue());
    auto integerType = dyn_cast<IntegerType>(type.getElementType());
    if (!resourceAttr || !integerType) {
        return failure();
    }

    unsigned bitWidth = integerType.getWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0 || bitWidth > 64) {
        return failure();
    }
    unsigned byteWidth = bitWidth / 8;
    ArrayRef<char> data = resourceAttr.getData();
    if (data.size() != static_cast<size_t>(type.getNumElements() * byteWidth)) {
        return failure();
    }

    std::vector<APInt> values;
    values.reserve(type.getNumElements());
    for (int64_t idx = 0, count = type.getNumElements(); idx < count; ++idx) {
        values.emplace_back(bitWidth, readLittleEndianBits(data, idx, byteWidth));
    }
    return values;
}

Value maybeSliceLike(Location loc, Value value, Value like, PatternRewriter &rewriter) {
    SmallVector<tensor::ExtractSliceOp> sliceOps;
    Value current = like;
    while (auto sliceOp = current.getDefiningOp<tensor::ExtractSliceOp>()) {
        sliceOps.push_back(sliceOp);
        current = sliceOp.getSource();
    }

    for (tensor::ExtractSliceOp sliceOp : llvm::reverse(sliceOps)) {
        auto valueType = dyn_cast<RankedTensorType>(value.getType());
        auto likeSourceType = dyn_cast<RankedTensorType>(sliceOp.getSource().getType());
        if (!valueType || !likeSourceType || valueType.getShape() != likeSourceType.getShape()) {
            return value;
        }

        value = tensor::ExtractSliceOp::create(
            rewriter, loc, value, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
            sliceOp.getMixedStrides()
        );
    }

    return value;
}

FailureOr<Value>
collapseReductionDimBlockParam(Location loc, Value value, PatternRewriter &rewriter) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || ShapedType::isDynamicShape(type.getShape())) {
        return failure();
    }
    if (type.getRank() == 2) {
        return value;
    }
    if (type.getRank() != 3 || type.getShape()[1] != 1) {
        return failure();
    }

    auto collapsedType = RankedTensorType::get(
        {type.getShape()[0], type.getShape()[2]}, type.getElementType(), type.getEncoding()
    );
    SmallVector<ReassociationIndices> reassociation = {{0}, {1, 2}};
    return tensor::CollapseShapeOp::create(rewriter, loc, collapsedType, value, reassociation)
        .getResult();
}

FailureOr<Value> buildReductionDimScaleBias(
    Location loc, Value scales, Value zeroPoints, RankedTensorType outputType,
    PatternRewriter &rewriter, linalg::GenericOp anchorOp
) {
    arith::ConstantOp scaleConstOp = getConstantLikeTensorOp(scales);
    arith::ConstantOp zeroPointConstOp = getConstantLikeTensorOp(zeroPoints);
    if (!scaleConstOp || !zeroPointConstOp) {
        return failure();
    }

    auto scaleConstType = dyn_cast<RankedTensorType>(scaleConstOp.getType());
    auto zeroPointConstType = dyn_cast<RankedTensorType>(zeroPointConstOp.getType());
    if (!scaleConstType || !zeroPointConstType ||
        scaleConstType.getShape() != zeroPointConstType.getShape() ||
        ShapedType::isDynamicShape(scaleConstType.getShape())) {
        return failure();
    }

    FailureOr<std::vector<APFloat>> maybeScaleValues = getFloatConstantValues(scaleConstOp);
    if (failed(maybeScaleValues)) {
        return failure();
    }
    Type outputElementType = outputType.getElementType();
    if (!outputElementType.isBF16() && !outputElementType.isF32()) {
        return failure();
    }
    // Floating ACT/BRAM consumes fp32 bias even when the expanded weights are bf16.
    const llvm::fltSemantics &biasSemantics = APFloat::IEEEsingle();

    std::vector<APFloat> biasValues;
    biasValues.reserve(scaleConstType.getNumElements());
    const bool zeroPointIsUnsigned = usesUnsignedIntegerZeroPoint(anchorOp);
    if (FailureOr<std::vector<APInt>> zeroPointValues = getIntegerConstantValues(zeroPointConstOp);
        succeeded(zeroPointValues)) {
        if (zeroPointValues->size() != maybeScaleValues->size()) {
            return failure();
        }
        for (auto [scaleValue, zeroPointValue] :
             llvm::zip_equal(*maybeScaleValues, *zeroPointValues)) {
            bool ignored;
            scaleValue.convert(biasSemantics, APFloat::rmNearestTiesToEven, &ignored);
            int64_t zeroPoint = zeroPointIsUnsigned
                                    ? static_cast<int64_t>(zeroPointValue.getZExtValue())
                                    : zeroPointValue.getSExtValue();
            APFloat biasValue(static_cast<float>(zeroPoint));
            biasValue.convert(biasSemantics, APFloat::rmNearestTiesToEven, &ignored);
            biasValue.changeSign();
            scaleValue.multiply(biasValue, APFloat::rmNearestTiesToEven);
            biasValues.push_back(scaleValue);
        }
    }
    else if (FailureOr<std::vector<APFloat>> zeroPointValues =
                 getFloatConstantValues(zeroPointConstOp);
             succeeded(zeroPointValues)) {
        if (zeroPointValues->size() != maybeScaleValues->size()) {
            return failure();
        }
        for (auto [scaleValue, zeroPointValue] :
             llvm::zip_equal(*maybeScaleValues, *zeroPointValues)) {
            bool ignored;
            scaleValue.convert(biasSemantics, APFloat::rmNearestTiesToEven, &ignored);
            zeroPointValue.convert(biasSemantics, APFloat::rmNearestTiesToEven, &ignored);
            zeroPointValue.changeSign();
            scaleValue.multiply(zeroPointValue, APFloat::rmNearestTiesToEven);
            biasValues.push_back(scaleValue);
        }
    }
    else {
        anchorOp.emitError("expected integer or floating-point input zero-point for "
                           "reduction-dimension expand_weights bias");
        return failure();
    }

    Value scaleBias = createFConst(rewriter, anchorOp, biasValues, scaleConstType.getShape());
    if (zeroPoints.getDefiningOp<tensor::ExtractSliceOp>()) {
        scaleBias = maybeSliceLike(loc, scaleBias, zeroPoints, rewriter);
    }
    else if (scales.getDefiningOp<tensor::ExtractSliceOp>()) {
        scaleBias = maybeSliceLike(loc, scaleBias, scales, rewriter);
    }
    return collapseReductionDimBlockParam(loc, scaleBias, rewriter);
}

/// Matches the simplified block-quantized dequant linalg.generic produced by
/// rewriteBlockQuantizedMatmulInputSimple in OptimizeMatmulOpPattern.
///
/// The generic has the form:
///   linalg.generic {
///     indexing_maps = [identity, broadcast_map, broadcast_map, identity],
///     iterator_types = ["parallel", "parallel", "parallel"]
///   } ins(%weight : tensor<GxBxNxi8>, %scale : tensor<Gx1xNxbf16>,
///         %zp : tensor<Gx1xNxi8>)
///     outs(%out : tensor<GxBxNxbf16>) {
///       extsi/extui weight, extsi/extui zp, subi, sitofp, mulf scale
///   }
///
/// Where the broadcast maps use constant-0 for the block dimension (d1).
/// Weight shape [G, B, N] with G * B == K (ReductionDimBlocked).
bool matchBlockQuantizedExpandWeights(linalg::GenericOp genericOp) {
    if (!genericOp || genericOp.getNumDpsInputs() != 3 || genericOp.getNumDpsInits() != 1 ||
        !genericOp.isAllParallelLoops()) {
        return false;
    }

    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResultTypes().front());
    if (!resultType || resultType.getRank() != 3 ||
        !(resultType.getElementType().isBF16() || resultType.getElementType().isF32()) ||
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
    if (!(weightType.getElementType().isInteger(4) || weightType.getElementType().isInteger(8))) {
        return false;
    }
    if (scaleType.getElementType() != resultType.getElementType()) {
        return false;
    }
    if (!zeroPointType.getElementType().isInteger() &&
        zeroPointType.getElementType() != resultType.getElementType()) {
        return false;
    }

    // Weight map must be identity, output map must be identity
    auto indexingMaps = genericOp.getIndexingMapsArray();
    if (indexingMaps.size() != 4 || !indexingMaps[0].isIdentity() ||
        !indexingMaps[3].isIdentity()) {
        return false;
    }

    // Scale and zp maps: must be simple (only dim exprs and constants, no div/mod).
    // They may use constant-0 for broadcast dims, e.g. (d0,d1,d2)->(d0,0,d2).
    auto isSimpleBroadcastMap = [](AffineMap map) {
        if (map.getNumSymbols() != 0)
            return false;
        for (auto expr : map.getResults()) {
            if (!isa<AffineDimExpr>(expr) && !isa<AffineConstantExpr>(expr))
                return false;
        }
        return true;
    };
    if (!isSimpleBroadcastMap(indexingMaps[1]) || !isSimpleBroadcastMap(indexingMaps[2])) {
        return false;
    }

    // Verify weight shape makes sense: [G, B, N] where G*B could be K
    auto weightShape = weightType.getShape();
    if (weightShape[0] <= 0 || weightShape[1] <= 0 || weightShape[2] <= 0) {
        return false;
    }

    // Verify scale/zp are constants
    if (!getConstantLikeTensorOp(genericOp.getInputs()[1]) ||
        !getConstantLikeTensorOp(genericOp.getInputs()[2])) {
        return false;
    }

    // Verify the body matches the dequant pattern:
    // (weight - zp) * scale  (either integer sub or float sub variant)
    return matchMatmulRhsBlockQuantizedBody(genericOp);
}

/// Get the output type that the dequant generic + reshape chain produces when
/// used as a matmul input. Traces through collapse_shape/expand_shape to find
/// the final batch_matmul input type.
RankedTensorType getDequantMatmulInputType(linalg::GenericOp genericOp, Value matmulInput) {
    auto inputType = dyn_cast<RankedTensorType>(matmulInput.getType());
    if (!inputType) {
        return {};
    }
    // The matmul input should be rank-3 [1, K, N]
    if (inputType.getRank() != 3 || inputType.getShape()[0] != 1) {
        return {};
    }
    auto weightShape = cast<RankedTensorType>(genericOp.getInputs()[0].getType()).getShape();
    int64_t K = weightShape[0] * weightShape[1];
    int64_t N = weightShape[2];
    if (inputType.getShape()[1] != K || inputType.getShape()[2] != N) {
        return {};
    }
    return inputType;
}

FailureOr<Value> maybeFoldBlockQuantizedMatmulInput(
    Value input, PatternRewriter &rewriter,
    const std::optional<IntegerAttr> &maybeFuseGroupAttr = std::nullopt
) {
    Value baseInput = stripTensorShapeCasts(input);
    auto genericOp = baseInput.getDefiningOp<linalg::GenericOp>();
    if (!matchBlockQuantizedExpandWeights(genericOp)) {
        return failure();
    }

    auto outputType = getDequantMatmulInputType(genericOp, input);
    if (!outputType) {
        return failure();
    }

    // Marking mode: mark the generic op and shape ops with fuse group, return early.
    if (maybeFuseGroupAttr) {
        markOpFuseGroup(genericOp, rewriter, maybeFuseGroupAttr);
        // Also mark the intermediate shape cast ops
        for (Value current = input; current != baseInput;) {
            Operation *defOp = current.getDefiningOp();
            if (!defOp)
                break;
            if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(defOp)) {
                markOpFuseGroup(expandOp, rewriter, maybeFuseGroupAttr);
                current = expandOp.getSrc();
                continue;
            }
            if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(defOp)) {
                markOpFuseGroup(collapseOp, rewriter, maybeFuseGroupAttr);
                current = collapseOp.getSrc();
                continue;
            }
            break;
        }
        return input;
    }

    // Lowering mode: build the ExpandWeightsOp
    auto weightType = cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    ArrayRef<int64_t> weightShape = weightType.getShape();
    int64_t groups = weightShape[0];
    int64_t blockSize = weightShape[1];
    int64_t cols = weightShape[2];

    if (groups <= 0 || blockSize <= 0 || cols <= 0 ||
        blockSize > std::numeric_limits<int32_t>::max()) {
        return failure();
    }

    // The output type for ExpandWeightsOp is [1, K, N] where K = groups * blockSize
    Location loc = genericOp.getLoc();

    // Flatten weight: [G, B, N] -> [K, N]
    auto flattenedWeightType = RankedTensorType::get(
        {groups * blockSize, cols}, weightType.getElementType(), weightType.getEncoding()
    );
    SmallVector<ReassociationIndices> weightReassociation = {{0, 1}, {2}};
    Value flattenedWeights = tensor::CollapseShapeOp::create(
        rewriter, loc, flattenedWeightType, genericOp.getInputs()[0], weightReassociation
    );

    Value scales = genericOp.getInputs()[1];
    Value zeroPoints = genericOp.getInputs()[2];

    FailureOr<Value> collapsedScales = collapseReductionDimBlockParam(loc, scales, rewriter);
    if (failed(collapsedScales)) {
        return failure();
    }

    FailureOr<Value> scaleBias =
        buildReductionDimScaleBias(loc, scales, zeroPoints, outputType, rewriter, genericOp);
    if (failed(scaleBias)) {
        return failure();
    }

    auto [outMin, outMax] = getDTypeRange(outputType.getElementType());
    auto expandWeightsOp = torq_hl::ExpandWeightsOp::create(
        rewriter, loc, outputType, createInitTensor(genericOp, rewriter, outputType), *scaleBias,
        outMin, outMax, *collapsedScales, flattenedWeights, static_cast<uint32_t>(blockSize)
    );
    return expandWeightsOp.getOutput();
}

bool isMatmulFusibleFp8CastOp(linalg::GenericOp genericOp) {
    if (!genericOp) {
        return false;
    }

    if (genericOp.getNumDpsInputs() != 1) {
        return false;
    }

    if (genericOp.getNumDpsInits() != 1) {
        return false;
    }

    if (!genericOp.isAllParallelLoops()) {
        return false;
    }

    if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
            return map.isIdentity();
        })) {
        return false;
    }

    if (genericOp.payloadUsesValueFromOperand(genericOp.getDpsInitOperand(0))) {
        return false;
    }

    auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto resultType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
    if (!inputType || !resultType || inputType.getShape() != resultType.getShape() ||
        !isa<Float8E5M2Type>(inputType.getElementType()) || !resultType.getElementType().isBF16()) {
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return false;
    }

    auto extOp = yieldOp.getOperand(0).getDefiningOp<arith::ExtFOp>();
    if (!extOp || extOp.getIn().getType() != inputType.getElementType() ||
        extOp.getResult().getType() != resultType.getElementType()) {
        return false;
    }

    auto inputArg = dyn_cast<BlockArgument>(extOp.getIn());
    return inputArg && inputArg.getOwner() == genericOp.getBody() && inputArg.getArgNumber() == 0;
}

FailureOr<Value> buildFp8CastExpandWeightsOp(
    Location loc, Value packedWeights, RankedTensorType outputType, PatternRewriter &rewriter,
    linalg::GenericOp anchorOp
) {
    if (!outputType || !outputType.getElementType().isBF16() ||
        ShapedType::isDynamicShape(outputType.getShape())) {
        return failure();
    }

    auto packedWeightType = dyn_cast<RankedTensorType>(packedWeights.getType());
    if (!packedWeightType || !isa<Float8E5M2Type>(packedWeightType.getElementType()) ||
        ShapedType::isDynamicShape(packedWeightType.getShape())) {
        return failure();
    }

    ArrayRef<int64_t> packedShape = packedWeightType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (outputType.getRank() == packedWeightType.getRank() + 1) {
        if (outputShape.empty() || outputShape.front() != 1) {
            return failure();
        }
        outputShape = outputShape.drop_front();
    }
    if (packedShape != outputShape || outputShape.size() < 2) {
        return failure();
    }

    int64_t rows = outputShape[outputShape.size() - 2];
    int64_t cols = outputShape.back();
    if (rows <= 0 || cols <= 0 || cols > std::numeric_limits<int32_t>::max()) {
        return failure();
    }

    const llvm::fltSemantics &bf16 = llvm::APFloat::BFloat();
    std::vector<APFloat> zeroBias(rows, APFloat(bf16, "0.0"));
    std::vector<APFloat> oneScale(rows, APFloat(bf16, "1.0"));
    SmallVector<int64_t, 3> biasShape{rows, 1, 1};
    SmallVector<int64_t, 2> scaleShape{rows, 1};
    Value scaleBias = createFConst(rewriter, anchorOp, zeroBias, biasShape);
    Value scale = createFConst(rewriter, anchorOp, oneScale, scaleShape);

    auto [outMin, outMax] = getDTypeRange(outputType.getElementType());
    auto expandWeightsOp = torq_hl::ExpandWeightsOp::create(
        rewriter, loc, outputType, createInitTensor(anchorOp, rewriter, outputType), scaleBias,
        outMin, outMax, scale, packedWeights, static_cast<uint32_t>(cols)
    );
    return expandWeightsOp.getOutput();
}

FailureOr<Value> maybeFoldFp8CastMatmulInput(
    Value input, PatternRewriter &rewriter,
    const std::optional<IntegerAttr> &maybeFuseGroupAttr = std::nullopt
) {
    Value baseInput = stripTensorShapeCasts(input);
    auto genericOp = baseInput.getDefiningOp<linalg::GenericOp>();
    if (!isMatmulFusibleFp8CastOp(genericOp)) {
        return failure();
    }

    // Marking mode: mark the generic op and return early (no IR mutation).
    if (markOpFuseGroup(genericOp, rewriter, maybeFuseGroupAttr)) {
        return input;
    }

    auto outputType = dyn_cast<RankedTensorType>(input.getType());
    if (!outputType) {
        return failure();
    }

    return buildFp8CastExpandWeightsOp(
        genericOp.getLoc(), genericOp.getInputs()[0], outputType, rewriter, genericOp
    );
}

/// Check if a linalg.generic represents a broadcasted batch matmul:
///   C[b,m,n] += A[b,m,k] * B[0,k,n]  (or A[0,m,k] * B[b,k,n])
/// Returns the broadcast input index (0 or 1), or std::nullopt if no match.
std::optional<int> findBatchMatmulBcastIdx(linalg::GenericOp op) {
    // Must have 2 inputs, 1 output, 4 iteration dims
    if (op.getNumDpsInputs() != 2 || op.getNumResults() != 1)
        return std::nullopt;

    auto iterTypes = op.getIteratorTypesArray();
    if (iterTypes.size() != 4)
        return std::nullopt;

    // Iterator types: [parallel, parallel, parallel, reduction]
    if (iterTypes[0] != utils::IteratorType::parallel ||
        iterTypes[1] != utils::IteratorType::parallel ||
        iterTypes[2] != utils::IteratorType::parallel ||
        iterTypes[3] != utils::IteratorType::reduction)
        return std::nullopt;

    // Check body: mulf + addf (or muli + addi for integer)
    auto &body = op.getBody()->getOperations();
    if (body.size() != 3) // mul, add, yield
        return std::nullopt;

    auto it = body.begin();
    Operation *firstOp = &*it++;
    Operation *secondOp = &*it++;
    if (!isa<arith::MulFOp, arith::MulIOp>(firstOp) || !isa<arith::AddFOp, arith::AddIOp>(secondOp))
        return std::nullopt;

    // Check indexing maps for broadcast pattern:
    // One input has constant 0 in its batch dim expression
    auto maps = op.getIndexingMapsArray();
    // Output map: (d0, d1, d2, d3) -> (d0, d1, d2)
    auto outMap = maps[2];
    if (outMap.getNumResults() != 3 ||
        outMap.getResult(0) != getAffineDimExpr(0, op.getContext()) ||
        outMap.getResult(1) != getAffineDimExpr(1, op.getContext()) ||
        outMap.getResult(2) != getAffineDimExpr(2, op.getContext()))
        return std::nullopt;

    // Check which input has constant-0 batch dim
    for (int i : {0, 1}) {
        auto map = maps[i];
        if (map.getNumResults() != 3)
            return std::nullopt;
        if (mlir::isa<AffineConstantExpr>(map.getResult(0)) &&
            mlir::cast<AffineConstantExpr>(map.getResult(0)).getValue() == 0) {
            return i;
        }
    }
    return std::nullopt;
}

/// Convert a broadcast-batch-matmul linalg.generic to torq_hl::MatMulOp.
/// The generic has indexing maps with constant-0 on one input's batch dim.
struct BroadcastBatchMatmulGenericConversion : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    BroadcastBatchMatmulGenericConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // TODO: Add fuse group marking support for broadcast batch matmul.
        if (_markFuseGroups) {
            return failure();
        }

        std::optional<int> broadcastIdx = findBatchMatmulBcastIdx(srcOp);
        if (!broadcastIdx)
            return failure();

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(srcResultType.getElementType());

        // Input order in the generic: [lhs, rhs]
        Value input1 = srcOp.getInputs()[0]; // lhs
        Value input2 = srcOp.getInputs()[1]; // rhs

        rewriter.replaceOpWithNewOp<torq_hl::MatMulOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            0, outMin, outMax, 0, createI32Const(rewriter, srcOp, interleave(bias, scale)), input1,
            input2
        );

        return success();
    }
};

template <typename OpTy> struct MatmulOpConversion final : public OpRewritePattern<OpTy> {
  private:
    const bool _markFuseGroups;

  public:
    MatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<OpTy>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(OpTy srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        std::optional<IntegerAttr> maybeFuseGroupAttr = std::nullopt;
        if (_markFuseGroups) {
            maybeFuseGroupAttr = srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
        }

        Value lhs = srcOp.getOperand(0);
        Value rhs = srcOp.getOperand(1);

        bool foldedLhsIntoDequantOp = false;
        if (FailureOr<Value> foldedLhs =
                maybeFoldBlockQuantizedMatmulInput(lhs, rewriter, maybeFuseGroupAttr);
            succeeded(foldedLhs)) {
            lhs = *foldedLhs;
            foldedLhsIntoDequantOp = true;
        }
        else if (FailureOr<Value> foldedLhs =
                     maybeFoldFp8CastMatmulInput(lhs, rewriter, maybeFuseGroupAttr);
                 succeeded(foldedLhs)) {
            lhs = *foldedLhs;
            foldedLhsIntoDequantOp = true;
        }

        bool foldedRhsIntoDequantOp = false;
        if (FailureOr<Value> foldedRhs =
                maybeFoldBlockQuantizedMatmulInput(rhs, rewriter, maybeFuseGroupAttr);
            succeeded(foldedRhs)) {
            rhs = *foldedRhs;
            foldedRhsIntoDequantOp = true;
        }
        else if (FailureOr<Value> foldedRhs =
                     maybeFoldFp8CastMatmulInput(rhs, rewriter, maybeFuseGroupAttr);
                 succeeded(foldedRhs)) {
            rhs = *foldedRhs;
            foldedRhsIntoDequantOp = true;
        }

        if (_markFuseGroups) {
            // maybeFold* already marked the upstream fusible ops
            markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr);
            return success();
        }

        if (foldedLhsIntoDequantOp || foldedRhsIntoDequantOp) {
            rewriter.modifyOpInPlace(srcOp, [&]() {
                if (foldedLhsIntoDequantOp) {
                    srcOp->setOperand(0, lhs);
                }
                if (foldedRhsIntoDequantOp) {
                    srcOp->setOperand(1, rhs);
                }
            });
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(srcResultType.getElementType());

        Value biasScale = createI32Const(rewriter, srcOp, interleave(bias, scale));
        rewriter.replaceOpWithNewOp<torq_hl::MatMulOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            0, outMin, outMax, 0, biasScale, lhs, rhs
        );

        return success();
    }
};

} // namespace

void populateLinalgToTorqHLMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // Make sure to add the new class to isTorqMatmulOp function if here is changed.
    patterns.insert<MatmulOpConversion<linalg::BatchMatmulOp>>(context, markFuseGroups);
    patterns.insert<MatmulOpConversion<linalg::DotOp>>(context, markFuseGroups);
    patterns.insert<MatmulOpConversion<linalg::MatvecOp>>(context, markFuseGroups);

    // TODO: Add fuse group marking support for broadcast batch matmul.
    patterns.insert<BroadcastBatchMatmulGenericConversion>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
