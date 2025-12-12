// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-pre-pattern"

namespace mlir::syna::torq {

static DenseElementsAttr expandWeightsForDilation(
    DenseElementsAttr weights, ArrayRef<int64_t> dilations, PatternRewriter &rewriter,
    int64_t maxKernelSize = 7
) {
    auto weightType = mlir::cast<ShapedType>(weights.getType());
    auto shape = weightType.getShape();
    auto elemType = weightType.getElementType();

    int64_t kh = shape[shape.size() - 2];
    int64_t kw = shape[shape.size() - 1];
    int64_t dh = dilations[0], dw = dilations[1];

    int64_t khNew = kh + (kh - 1) * (dh - 1);
    int64_t kwNew = kw + (kw - 1) * (dw - 1);

    if (khNew > maxKernelSize || kwNew > maxKernelSize)
        return nullptr;

    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[newShape.size() - 2] = khNew;
    newShape[newShape.size() - 1] = kwNew;

    Location loc = rewriter.getUnknownLoc();
    auto weightsConst = rewriter.create<arith::ConstantOp>(loc, weights);
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, newShape, elemType);

    TypedAttr zeroAttr;
    if (elemType.isBF16() || elemType.isF16() || elemType.isF32()) {
        zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    }
    else if (elemType.isInteger()) {
        zeroAttr = rewriter.getIntegerAttr(elemType, 0);
    }
    else {
        return nullptr;
    }

    auto zeroConst = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto filledTensor =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zeroConst}, ValueRange{emptyTensor});

    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes;
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(shape.size()));

    SmallVector<AffineExpr> outputExprs;
    for (size_t i = 0; i < shape.size() - 2; ++i) {
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    outputExprs.push_back(rewriter.getAffineDimExpr(shape.size() - 2) * dh);
    outputExprs.push_back(rewriter.getAffineDimExpr(shape.size() - 1) * dw);
    indexingMaps.push_back(AffineMap::get(shape.size(), 0, outputExprs, rewriter.getContext()));

    for (size_t i = 0; i < shape.size(); ++i) {
        iteratorTypes.push_back(utils::IteratorType::parallel);
    }

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, RankedTensorType::get(newShape, elemType), ValueRange{weightsConst},
        ValueRange{filledTensor.getResult(0)}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
        }
    );

    genericOp->setAttr("torq-compile-time-const", rewriter.getBoolAttr(true));
    auto result = computeConstant(genericOp.getResult(0));

    return result;
}

// Helper function to convert tensor.insert_slice with stride > 1 to InterleavedInsertOp
// and calculate padding for transposed convolution
static std::optional<std::pair<Value, PaddingInfo>> convertStridedInsertSliceToInterleaved(
    Value input, Location loc, PatternRewriter &rewriter, Operation *parentOp
) {
    auto insertSliceOp = input.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) {
        return std::nullopt;
    }

    auto staticStrides = insertSliceOp.getStaticStrides();
    auto staticOffsets = insertSliceOp.getStaticOffsets();
    auto staticSizes = insertSliceOp.getStaticSizes();

    // Check for stride > 1 in any dimension (interleaving pattern)
    int interleavedDim = -1;
    int64_t strideValue = 1;
    for (size_t i = 0; i < staticStrides.size(); ++i) {
        if (staticStrides[i] > 1) {
            interleavedDim = i;
            strideValue = staticStrides[i];
            break;
        }
    }

    // Only handle stride-2 interleaving
    if (interleavedDim < 0 || strideValue != 2) {
        return std::nullopt;
    }

    Value source = insertSliceOp.getSource();
    Value dest = insertSliceOp.getDest();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto destType = cast<RankedTensorType>(dest.getType());
    auto destShape = destType.getShape();
    auto elemType = sourceType.getElementType();

    // Calculate padding from insert_slice offsets
    // Format: [left, right, top, bottom] for NCHW
    int64_t topPadding = staticOffsets[interleavedDim];
    int64_t interleavedSize = staticSizes[interleavedDim] * strideValue;
    int64_t bottomPadding = destShape[interleavedDim] - topPadding - interleavedSize;

    // Build interleaved shape (output size after interleaving)
    SmallVector<int64_t> interleavedShape4D;
    for (size_t i = 0; i < staticSizes.size(); ++i) {
        if (i == static_cast<size_t>(interleavedDim)) {
            interleavedShape4D.push_back(staticSizes[i] * strideValue);
        }
        else {
            interleavedShape4D.push_back(staticSizes[i]);
        }
    }

    auto interleavedResultType = RankedTensorType::get(interleavedShape4D, elemType);
    Value interleavedInit = rewriter.create<tensor::EmptyOp>(loc, interleavedShape4D, elemType);

    // Set clipping values based on data type
    int32_t output_min, output_max;
    if (elemType.isInteger(8)) {
        output_min = -128;
        output_max = 127;
    }
    else if (elemType.isInteger(16)) {
        output_min = -32768;
        output_max = 32767;
    }
    else if (elemType.isBF16()) {
        output_min = 0xff800000; // -inf in bf16 (as int32 bits)
        output_max = 0x7f800000; // +inf in bf16 (as int32 bits)
    }
    else {
        return std::nullopt;
    }

    // Create weights tensor [1, 0] or [1.0, 0.0] for stride-2
    Value weights;
    if (elemType.isInteger(8)) {
        std::vector<int8_t> weightsData = {1, 0};
        weights = createI8Const(rewriter, *parentOp, weightsData, llvm::ArrayRef<int64_t>{2});
    }
    else if (elemType.isInteger(16)) {
        std::vector<int16_t> weightsData = {1, 0};
        weights = createI16Const(rewriter, *parentOp, weightsData, llvm::ArrayRef<int64_t>{2});
    }
    else if (elemType.isBF16()) {
        const llvm::fltSemantics &bf16 = llvm::APFloat::BFloat();
        std::vector<llvm::APFloat> weightsData = {
            llvm::APFloat(bf16, "1.0"), llvm::APFloat(bf16, "0.0")
        };
        weights = createFConst(rewriter, *parentOp, weightsData, llvm::ArrayRef<int64_t>{2});
    }
    else {
        return std::nullopt;
    }

    // Create InterleavedInsertOp
    auto interleavedOp = rewriter.create<torq_hl::InterleavedInsertOp>(
        loc, interleavedResultType, interleavedInit, rewriter.getI32IntegerAttr(strideValue),
        rewriter.getI32IntegerAttr(output_min), rewriter.getI32IntegerAttr(output_max), weights,
        source
    );

    Value interleavedOutput = interleavedOp.getOutput();

    // Apply padding to the interleaved output if needed
    if (topPadding > 0 || bottomPadding > 0) {
        // Build padded shape: add top and bottom padding to the interleaved dimension
        SmallVector<int64_t> paddedShape4D = interleavedShape4D;
        paddedShape4D[interleavedDim] = destShape[interleavedDim]; // Use original dest size

        Value paddedInit = rewriter.create<tensor::EmptyOp>(loc, paddedShape4D, elemType);

        // Fill with zeros
        TypedAttr fillValue;
        if (elemType.isBF16()) {
            fillValue = rewriter.getIntegerAttr(rewriter.getIntegerType(16), 0);
        }
        else if (elemType.isInteger(8)) {
            fillValue = rewriter.getIntegerAttr(rewriter.getIntegerType(8), 0);
        }
        else if (elemType.isInteger(16)) {
            fillValue = rewriter.getIntegerAttr(rewriter.getIntegerType(16), 0);
        }
        else {
            fillValue = rewriter.getZeroAttr(elemType);
        }

        Value fillValueAsValue = rewriter.create<arith::ConstantOp>(loc, fillValue);
        auto fillOp = rewriter.create<linalg::FillOp>(
            loc, ValueRange{fillValueAsValue}, ValueRange{paddedInit}
        );

        // Insert the interleaved output at the correct offset
        SmallVector<OpFoldResult> offsets(paddedShape4D.size(), rewriter.getIndexAttr(0));
        offsets[interleavedDim] = rewriter.getIndexAttr(topPadding);

        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : interleavedShape4D) {
            sizes.push_back(rewriter.getIndexAttr(dim));
        }

        SmallVector<OpFoldResult> strides(paddedShape4D.size(), rewriter.getIndexAttr(1));

        interleavedOutput = rewriter.create<tensor::InsertSliceOp>(
            loc, interleavedOutput, fillOp.getResult(0), offsets, sizes, strides
        );

        LLVM_DEBUG({
            llvm::dbgs() << "Applied padding [" << topPadding << ", " << bottomPadding
                         << "] to InterleavedInsertOp output\n";
        });
    }

    // Return zero padding for Conv2D since padding is already applied
    PaddingInfo padInfo;
    padInfo.lrtbPad = {0, 0, 0, 0};
    padInfo.padValue = 0;

    LLVM_DEBUG({
        llvm::dbgs(
        ) << "Converted strided insert_slice to InterleavedInsertOp with padding applied\n";
    });

    return std::make_pair(interleavedOutput, padInfo);
}

template <class LinalgConvOp, class TorqConvOp>
struct Conv2dConvert : public OpRewritePattern<LinalgConvOp> {
  private:
    using MatchFn = bool(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape, ArrayRef<int64_t> padShape
    );

    const int _channelDim;          // Channel dimension index in data shape
    const Permutation _dataPerm;    // Dim permutation for data transpose
    const Permutation _weightsPerm; // Weights permutation for weight transpose
    const int _shift8b;             // Scale shift for 8-bit integer operations
    const int _shift16b;            // Scale shift for 16-bit integer operations
    MatchFn *_matchFn;              // Function to match the convolution operation
    const bool _markFuseGroups;     // When true, mark the TI operations, don't convert.
    const bool
        _2DNchwChw; // set nchw/nhwc info from linalg conv op which give accurate input layout

  public:
    using OpRewritePattern<LinalgConvOp>::OpRewritePattern;
    Conv2dConvert(
        MLIRContext *context, int channelDim, const Permutation &dataPerm,
        const Permutation &weightsPerm, int shift8b, int shift16b, MatchFn *matchFn,
        bool markFuseGroups, bool isNchw = false
    )
        : OpRewritePattern<LinalgConvOp>(context), _channelDim(channelDim), _dataPerm(dataPerm),
          _weightsPerm(weightsPerm), _shift8b(shift8b), _shift16b(shift16b), _matchFn(matchFn),
          _markFuseGroups(markFuseGroups), _2DNchwChw(isNchw) {}

    LogicalResult matchAndRewrite(LinalgConvOp convOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        constexpr int groups = 1; // We don't use it
        const auto loc = convOp.getLoc();

        // Get the input, weights, and output of the original operation
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];

        auto inputType = llvm::cast<RankedTensorType>(input.getType());
        auto shape = inputType.getShape();
        auto weightType = llvm::cast<RankedTensorType>(weights.getType());
        auto weightShape = weightType.getShape();
        // Layout: NHWC → height is dim 1,  NCHW → height is dim 2
        const int heightDim = (_channelDim == 3) ? 1 : 2;
        bool isConv1D = (inputType.getRank() == 4 && shape[heightDim] == 1);
        bool isDepthwise =
            llvm::isa<linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNchwChwOp>(&convOp);
        if (isConv1D && !isDepthwise) {
            return rewriteAsConv1D(convOp, rewriter);
        }

        // First, check if input has strided insert_slice pattern (but don't convert yet)
        bool hasStridedInsertSlice = false;
        if (auto insertSliceOp = input.getDefiningOp<tensor::InsertSliceOp>()) {
            auto staticStrides = insertSliceOp.getStaticStrides();
            for (size_t i = 0; i < staticStrides.size(); ++i) {
                if (staticStrides[i] == 2) {
                    hasStridedInsertSlice = true;
                    break;
                }
            }
        }
        // Get preliminary padding info without modifying IR yet
        // For strided insert slice cases, no preliminary padding needed (handled later)
        // For regular cases, we'll compute padding after validation passes
        PaddingInfo prelimPadInfo{{0, 0, 0, 0}, 0};

        // For regular convs (no strided insert), peek at padding without consuming it
        // We need this for the match function check below
        if (!hasStridedInsertSlice) {
            // Just peek at padding values without modifying IR
            if (auto padOp = input.getDefiningOp<tensor::PadOp>()) {
                auto lp = padOp.getStaticLow();
                auto hp = padOp.getStaticHigh();
                if (lp.size() == 4 && hp.size() == 4) {
                    // Use correct dimension indices based on layout
                    // NCHW: [N, C, H, W] -> H=dim2, W=dim3
                    // NHWC: [N, H, W, C] -> H=dim1, W=dim2
                    int hDim = (_channelDim == 3) ? 1 : 2;
                    int wDim = (_channelDim == 3) ? 2 : 3;
                    prelimPadInfo.lrtbPad = {
                        lp[wDim], lp[hDim], hp[wDim], hp[hDim]
                    }; // [left, top, right, bottom]
                }
            }
        }

        // Todo: Capability check for depthwise conv should be moved to a helper function
        // Check for depthwise conv specific constraints
        if (llvm::isa<linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNchwChwOp>(&convOp
            )) {

            auto strides = convOp.getStrides().template getValues<int64_t>();
            // hk kernel
            if (strides[0] > 2 || strides[0] != strides[1]) {
                return rewriter.notifyMatchFailure(
                    convOp, "asymmetric strides or stride > 2 not supported by DW"
                );
            }
            // EK kernel
            bool isBF16 =
                inputType.getElementType().isBF16() && weightType.getElementType().isBF16();
            // nchw_chw -> in_cxkhxkw, nhwc_hwc -> khxkwxin_c

            if (isBF16 && strides[0] != 1) {
                return rewriter.notifyMatchFailure(convOp, "DW-bf16 only support stride 1");
            }

            // Dilation check removed - now handled by weight expansion below
            // (dilations > 1 will be converted to dilation = 1 via weight expansion)
        }

        // Check if we can support this layer
        if (_matchFn && !_matchFn(shape, weightShape, prelimPadInfo.lrtbPad)) {
            return rewriter.notifyMatchFailure(
                convOp, "Conv does not match expected kernel dimension or padding"
            );
        }

        // Fold any per-channel bias
        const auto outType = cast<RankedTensorType>(output.getType());
        const int outChannelCount = outType.getShape()[_channelDim];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outChannelCount, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold operations that take care of zero-point in weight quantization if present
        int weightZp = foldForwardWeightZp(output);

        // Fold any additional per-channel bias
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold scale and clamp. This is mandatory for integer operations.
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, false);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                convOp, "Expected scale and clamp info for integer operations"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Convert weights to the required format
        DenseIntOrFPElementsAttr weightAttr;
        auto transposedWeights = transposeValue(weights, _weightsPerm, loc, rewriter);
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overlflow
            constexpr int scaleFactor = 2;
            transposedWeights =
                rescaleValue(transposedWeights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        auto dilations = convOp.getDilations().template getValues<int64_t>();
        SmallVector<int64_t> dilationVec(dilations.begin(), dilations.end());
        std::vector<int64_t> finalDilationVec;

        if (llvm::any_of(dilations, [](int64_t d) { return d > 1; })) {
            weightAttr = computeConstant(transposedWeights);
            if (!weightAttr)
                return rewriter.notifyMatchFailure(
                    convOp, "failed to compute weights for dilation expansion"
                );

            auto expanded = expandWeightsForDilation(weightAttr, dilationVec, rewriter);
            if (!expanded)
                return rewriter.notifyMatchFailure(
                    convOp, "expanded kernel size exceeds 7x7 limit"
                );

            weightAttr = mlir::cast<DenseIntOrFPElementsAttr>(expanded);
            finalDilationVec = {1, 1};
        }
        else {
            weightAttr = computeConstant(transposedWeights);
            finalDilationVec = attrValues(convOp.getDilations());
        }

        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                convOp, "Failed to create constant for transposed weights"
            );
        }

        // NOW that all validation passed (including weight creation),
        // convert strided insert_slice to InterleavedInsertOp OR fold backward padding
        PaddingInfo padInfo;
        if (hasStridedInsertSlice) {
            if (auto result =
                    convertStridedInsertSliceToInterleaved(input, loc, rewriter, convOp)) {
                input = result->first;
                padInfo = result->second;
                // Update input type and shape after conversion
                inputType = cast<RankedTensorType>(input.getType());
                shape = inputType.getShape();
            }
            else {
                // Fallback to regular padding if conversion failed
                // Use correct layout: NCHW if channelDim==1, NHWC if channelDim==3
                bool isNchw = (_channelDim == 1);
                padInfo = foldBackwardPadding(input, rewriter, isNchw);
            }
        }
        else {
            // For regular convs, NOW fold the padding into padInfo (this modifies IR)
            // Use correct layout: NCHW if channelDim==1, NHWC if channelDim==3
            bool isNchw = (_channelDim == 1);
            padInfo = foldBackwardPadding(input, rewriter, isNchw);
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with input/output in the expected format
        input = transposeValue(input, _dataPerm, loc, rewriter);
        bool nhwcInput = _channelDim == 3 && _dataPerm.empty();
        auto torqOutType = transposeType(output.getType(), _dataPerm);

        auto torqConvOp = rewriter.create<TorqConvOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), padInfo.padValue, 0,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, padInfo.lrtbPad,
            attrValues(convOp.getStrides()), finalDilationVec, torq_hl::VectorizationModeEnum::None,
            torqWeights, biasScale, input, nhwcInput
        );
        auto torqOut = transposeValue(torqConvOp.getOutput(), _dataPerm.reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    template <typename LinalgConv>
    LogicalResult rewriteAsConv1D(LinalgConv convOp, PatternRewriter &rewriter) const {
        if (!isa<linalg::Conv2DNhwcHwcfOp>(convOp)) {
            return rewriter.notifyMatchFailure(
                convOp, "Only linalg::Conv2DNhwcHwcfOp can be rewritten as Conv1D"
            );
        }
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        auto loc = convOp.getLoc();
        constexpr int weightZp = 0;
        constexpr int groups = 1;
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];
        ::mlir::DenseIntElementsAttr stridesAttr = convOp.getStrides();
        auto strideValue = stridesAttr.getValues<int64_t>()[1];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto outputType = cast<RankedTensorType>(output.getType());
        auto outElemType = outputType.getElementType();
        bool isInt = outElemType.isInteger();
        int outChannels = outputType.getShape()[_channelDim];

        VectorIntOrFloat bias(outChannels, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannels, _shift8b, _shift16b);
        if (!scInfo && isInt)
            return failure();

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(transposeValue(weights, _weightsPerm, loc, rewriter));
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        // TODO: Torq weights should be reorderes in multiple channels cases;
        if (!torqWeights)
            return failure();

        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Note: op is Conv2DNhwcHwcfOp
        int64_t batch = inputType.getShape()[0];
        int64_t channels = inputType.getShape()[3];
        int64_t out_len = outputType.getShape()[2];
        int64_t outputChannels = outputType.getShape()[3];

        auto weightType = cast<RankedTensorType>(weights.getType());
        int64_t filter_len = weightType.getShape()[1];

        int64_t op_rows = filter_len;
        int64_t op_cols = out_len;

        llvm::SmallVector<int64_t> transposedShape = {batch, channels, op_rows, op_cols};
        RankedTensorType transposedType =
            RankedTensorType::get(transposedShape, inputType.getElementType());

        llvm::SmallVector<int64_t> permVals = {1, 0};
        auto permAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), permVals);

        auto torqOutType = transposeType(output.getType(), _dataPerm);

        // Decide whether to use Conv1D with reduction or TransposeReshape + Conv1D
        // The former is completely generic but probably less efficient for single-channel cases
        // The latter is more efficient but only works for single-channel input and outputs.
        bool useConv1dWithReduce = channels > 1 || outputChannels > 1;
        if (useConv1dWithReduce) {
            input = transposeValue(input, _dataPerm, loc, rewriter);
            // Create type for Conv1D output with an extra dimension at the end.
            // This will be reduced later with linalg.reduce.
            llvm::SmallVector<int64_t> torqOutShape(
                torqOutType.getShape().begin(), torqOutType.getShape().end()
            );
            torqOutShape.push_back(filter_len);
            torqOutType = RankedTensorType::get(torqOutShape, torqOutType.getElementType());
        }
        else {
            auto transposeReshape = rewriter.create<torq_hl::TransposeReshapeOp>(
                loc, transposedType, createInitTensor(convOp, rewriter, transposedType),
                attrValues(convOp.getStrides()), weightType.getShape(), permAttr, input
            );
            input = transposeReshape.getOutput();
            // Reset stride to 1 for Conv1DOp as the actual stride is handled in TransposeReshape
            strideValue = 1;
        }

        llvm::SmallVector<int64_t> zeroPad(4, 0);
        llvm::SmallVector<int64_t> stride = {strideValue};

        auto torqConv1Op = rewriter.create<torq_hl::Conv1DOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), 0, weightZp,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, zeroPad, stride,
            attrValues(convOp.getDilations()), torq_hl::VectorizationModeEnum::None, torqWeights,
            biasScale, input
        );
        Value torqOut = torqConv1Op.getOutput();

        if (useConv1dWithReduce) {
            // Add linalg.reduce to remove the extra dimension
            // Create reducedType from torqOutType by removing the last dimension
            auto reducedShape = torqOutType.getShape().drop_back();

            // Create a tensor filled with zeros of type torqOutType.getElementType()
            Value zeroValue = createZeroConstant(rewriter, loc, torqOutType.getElementType());
            auto cEmpty =
                rewriter.create<tensor::EmptyOp>(loc, reducedShape, torqOutType.getElementType());
            Value zeroTensor =
                rewriter.create<linalg::FillOp>(loc, ValueRange{zeroValue}, ValueRange{cEmpty})
                    .result();
            linalg::ReduceOp reduceOp = rewriter.create<linalg::ReduceOp>(
                loc, ValueRange{torqOut}, ValueRange{zeroTensor}, 4,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                    );
                }
            );

            torqOut = reduceOp->getResult(0);
        }
        // // Overwrite torqOut with init tensor for debugging
        // torqOut = createInitTensor(convOp, rewriter, cast<RankedTensorType>(torqOut.getType()));
        // // Fill input with 1s for debugging
        // torqOut = rewriter.create<torq_hl::FillOp>(
        //     loc, cast<RankedTensorType>(torqOut.getType()), torqOut,
        //     rewriter.getI32IntegerAttr(/*0x3f800000*//*0x00003f80*/0)
        // ).getOutput();

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }
};

struct PoolingNhwcMaxOpConversion : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    PoolingNhwcMaxOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNhwcMaxOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }
        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResults()[0];

        auto attrStrides = attrValues(srcOp.getStrides());
        if (attrStrides.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two strides for PoolingNhwcMaxOp"
            );
        }
        if (attrStrides[0] > 2 || attrStrides[1] > 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected strides <= 2 for PoolingNhwcMaxOp");
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        const std::vector<int8_t> weight = {1};

        PaddingInfo padInfo = foldBackwardPadding(input, rewriter);

        auto kernels = mlir::cast<RankedTensorType>(srcOp.getInputs()[1].getType()).getShape();
        if (kernels.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingNhwcMaxOp"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        auto dataPerm =
            srcResultType.getRank() == 4 ? Permutation::nhwc2nchw() : Permutation::none();

        input = transposeValue(input, dataPerm, loc, rewriter);
        srcResultType = transposeType(srcResultType, dataPerm);

        auto maxpoolOp = rewriter.create<torq_hl::MaxPool2dOp>(
            loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType), padInfo.padValue,
            attrStrides, padInfo.lrtbPad, kernels,
            createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );
        auto result = transposeValue(maxpoolOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), result);

        return success();
    }
};

template <class OpTy> struct FCMatmulOpConversion : public OpRewritePattern<OpTy> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern<OpTy>::OpRewritePattern;

    FCMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<OpTy>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(OpTy srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        const auto loc = srcOp.getLoc();
        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 input and 1 output");
        }

        Value inputA = srcOp.getInputs()[0];
        Value inputB = srcOp.getInputs()[1]; // weights
        Value output = srcOp.getResultTensors()[0];

        auto inputAType = llvm::cast<RankedTensorType>(inputA.getType());
        auto inputBType = llvm::cast<RankedTensorType>(inputB.getType());
        auto outputType = llvm::cast<RankedTensorType>(output.getType());
        if (inputAType.getRank() != 2 || inputBType.getRank() != 2 || outputType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects 2D inputs and outputs"
            );
        }
        auto inputAShape = inputAType.getShape();
        if (inputAShape[0] != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects inputA shape[0] == 1"
            );
        }

        auto outputChannelCount = outputType.getShape()[1];
        bool isInt = outputType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp)) {
        }

        // check if output user is expand_shape
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected scale and clamp info for integer operations"
            );
        }

        // check if output is a tensor::CollapseShapeOp
        if (output.hasOneUse() && (isa<tensor::CollapseShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        // Prepare weights
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(inputB.getDefiningOp())) {
            inputB = transposeOp.getInput();
            // NOTE: inputB changed, re-get its type if need to process related
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {inputA, inputB}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(inputB);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to create constant for transposed weights"
            );
        }
        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // get new output type as above various changes for output
        outputType = llvm::cast<RankedTensorType>(output.getType());

        auto fcOp = rewriter.create<torq_hl::FullyConnectedOp>(
            loc, outputType, createInitTensor(srcOp, rewriter, outputType), input_zp,
            0, // weight zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, inputA
        );
        rewriter.replaceOp(output.getDefiningOp(), fcOp.getOutput());

        return success();
    }
};

struct Conv2DMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }

        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        Value output = srcOp.getResultTensors()[0];

        // Ensure inputs and output are 2D
        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
        auto outType = llvm::cast<RankedTensorType>(output.getType());

        if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
            outType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Conv2DMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Check if the Conv2D input (lhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(lhs.getDefiningOp())) {
            lhs = extractSlice.getSource();
        }
        if (!lhs.getDefiningOp<tensor::CollapseShapeOp>() &&
            !isCollapseOrExpandShapeGeneric(lhs.getDefiningOp())) {
            return rewriter.notifyMatchFailure(srcOp, "LHS is not collapsed from 4D");
        }
        Value input = lhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || inputType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Match transpose on weight
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp())) {
            rhs = transposeOp.getInput();
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(rhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(rhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[1];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }
        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, rhs}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = rhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        finalType = convertTypeNHWCtoNCHW(finalType);
        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;
        input = transposeValue(input, Permutation::nhwc2nchw(), loc, rewriter);

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        auto torqOut =
            transposeValue(conv2dOp.getOutput(), Permutation::nhwc2nchw().reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DMatmulOpConversion success\n"; });
        return success();
    }
};

// Conv2DNchwMatmulOpConversion pattern
// Input NCHW, weight OIXY, Ouput NCHW
// This pattern detects and fuses the following operation chain:
//   %A = input, %collapsed = weight
//   %collapsed = tensor.collapse_shape %input [[0, 1], [2, 3]]
//                 : tensor<1xKx1x1xbf16> into tensor<Kx1xbf16>
//   %init = tensor.empty() : tensor<Nx1xf32>
//   %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<Nx1xf32>)
//   %matmul = linalg.matmul ins(%A, %collapsed : tensor<NxKxbf16>, tensor<Kx1xbf16>)
//                          outs(%fill : tensor<Nx1xf32>) -> tensor<Nx1xf32>
//   %expanded = tensor.expand_shape %matmul [[0, 1], [2, 3]]
//                 output_shape [1, N, 1, 1]
//                 : tensor<Nx1xf32> into tensor<1xNx1x1xf32>
//   %add = linalg.generic ins(%expanded, %bias)
//                          outs(%tmp)
//                          {body: (%x, %b) => arith.addf %x, %b}
//   %trunc = linalg.generic ins(%add)
//                            outs(%out)
//                            {body: (%x) => arith.truncf %x}
//   %output = %trunc
struct Conv2DNchwMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DNchwMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    static linalg::GenericOp getNextGenericWithTruncf(Value value) {
        if (!value.hasOneUse())
            return nullptr;

        auto genericOp = dyn_cast<linalg::GenericOp>(*value.getUsers().begin());
        if (!genericOp)
            return nullptr;

        for (Operation &op : genericOp.getBody()->getOperations())
            if (isa<arith::TruncFOp>(op))
                return genericOp;

        return nullptr;
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        Value output = srcOp.getResultTensors()[0];

        // Ensure inputs and output are 2D
        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
        auto outType = llvm::cast<RankedTensorType>(output.getType());

        if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
            outType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Conv2DNchwMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Check if the Conv2D input (rhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(rhs.getDefiningOp())) {
            rhs = extractSlice.getSource();
        }
        if (!rhs.getDefiningOp<tensor::CollapseShapeOp>() &&
            !isCollapseOrExpandShapeGeneric(lhs.getDefiningOp())) {
            return rewriter.notifyMatchFailure(srcOp, "LHS is not collapsed from 4D");
        }
        Value input = rhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || inputType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(lhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(lhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[0];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        // Check if the next op is linalg.generic with add and update bias, input_zp & weightZP
        // values
        if (!foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
            return rewriter.notifyMatchFailure(srcOp, "Next op is not linalg.generic with add");
        }

        // Check if the next op is linalg.generic with Trunc
        auto truncGeneric = getNextGenericWithTruncf(output);
        if (!truncGeneric)
            return rewriter.notifyMatchFailure(srcOp, "Next op is not linalg.generic with truncf");

        // Get the result tensor from the linalg.generic
        output = truncGeneric.getResultTensors()[0];
        finalType = cast<RankedTensorType>(output.getType());

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }

        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, rhs}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = lhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        rewriter.replaceOp(output.getDefiningOp(), conv2dOp.getOutput());

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DNchwMatmulOpConversion success\n"; });
        return success();
    }
};

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgMatmul\n";
        auto loc = convOp.getLoc();

        // Extract tensors and shapes
        Value input = convOp.getInputs()[0];   // Input tensor [N,C,W]
        Value filter = convOp.getInputs()[1];  // Filter tensor [F,C,Kw]
        Value output = convOp.getOutputs()[0]; // Output tensor [N,F,Ow]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Extract dimensions
        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        if (inputShape.size() != 3 || filterShape.size() != 3 || outputShape.size() != 3) {
            return rewriter.notifyMatchFailure(convOp, "Expected 3D tensors for Conv1D");
        }

        // Extract convolution parameters
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );

        int64_t N = inputShape[0];       // Batch size
        int64_t C = inputShape[1];       // Input channels
        int64_t F = filterShape[0];      // Output channels/filters
        int64_t Kw = filterShape[2];     // Kernel width
        int64_t Ow = outputShape[2];     // Output width
        int64_t stride = strides[0];     // Stride value
        int64_t dilation = dilations[0]; // Dilation value

        // Step 1: Unfold the input tensor using im2col approach
        // Each position in the output corresponds to a patch of the input
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        // Create a tensor to hold the unfolded input
        // Shape: [Ow, C*Kw] - each row contains a full patch for one output position
        SmallVector<int64_t> unfoldedShape = {Ow, C * Kw};
        auto unfoldedType = RankedTensorType::get(unfoldedShape, elemType);
        auto unfoldedInit = rewriter.create<tensor::EmptyOp>(loc, unfoldedShape, elemType);

        // Create the im2col transformation using a linalg.generic
        SmallVector<AffineExpr> unfoldIndexExprs;
        auto dim0 = rewriter.getAffineDimExpr(0); // Output position (Ow dimension)
        auto dim1 = rewriter.getAffineDimExpr(1); // Input channel and kernel position

        // dim1 / Kw gives us the channel index
        auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
        // dim1 % Kw gives us the kernel position
        auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
        // Calculate input position: outputPos * stride + kernelIdx * dilation
        auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                            kernelIdx * rewriter.getAffineConstantExpr(dilation);

        unfoldIndexExprs.push_back(rewriter.getAffineConstantExpr(0)); // N dimension (batch)
        unfoldIndexExprs.push_back(channelIdx);                        // C dimension (channels)
        unfoldIndexExprs.push_back(inputPosExpr);                      // W dimension (width)

        auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
        auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

        // Create the generic op for unfolding with explicit iterator types
        SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

        auto im2col = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{unfoldedType}, ValueRange{input}, ValueRange{unfoldedInit},
            ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap}, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            }
        );

        // Set torq.im2col attribute so that we can easily recognize this op during tiling
        im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
        auto unfoldedInput = im2col.getResult(0);

        // Step 2: Reshape the filter tensor from [F, C, Kw] to [F, C*Kw]
        SmallVector<int64_t> reshapedFilterShape = {F, C * Kw};
        auto reshapedFilterType =
            RankedTensorType::get(reshapedFilterShape, filterType.getElementType());
        auto reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );

        // Step 3: Create the matmul operation
        // We'll do: [F, C*Kw] @ [Ow, C*Kw]^T -> [F, Ow]
        // First, we need to transpose the unfolded input
        SmallVector<int64_t> transposedUnfoldedShape = {C * Kw, Ow};
        // auto transposedUnfoldedType = RankedTensorType::get(transposedUnfoldedShape, elemType);
        auto transposedUnfoldedInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedUnfoldedShape, elemType);

        auto transposedUnfolded = rewriter.create<linalg::TransposeOp>(
            loc, unfoldedInput, transposedUnfoldedInit, ArrayRef<int64_t>{1, 0}
        );

        // Create the matmul output tensor [F, Ow]
        SmallVector<int64_t> matmulResultShape = {F, Ow};
        auto matmulResultType = RankedTensorType::get(matmulResultShape, outputElemType);
        auto matmulInit = rewriter.create<tensor::EmptyOp>(loc, matmulResultShape, outputElemType);

        // Perform the actual matmul
        // Perform the actual matmul
        SmallVector<Value> inputs;
        inputs.push_back(reshapedFilter.getResult());
        inputs.push_back(transposedUnfolded.getResults()[0]);

        SmallVector<Value> outputs;
        outputs.push_back(matmulInit.getResult());

        auto matmulOp =
            rewriter.create<linalg::MatmulOp>(loc, TypeRange{matmulResultType}, inputs, outputs);

        // Step 4: Reshape the result back to [N, F, Ow]
        if (N == 1) {
            // Simply reshape to add the batch dimension
            auto finalResult = rewriter.create<tensor::ExpandShapeOp>(
                loc, matmulResultType, matmulOp.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, finalResult);
        }
        else {
            return rewriter.notifyMatchFailure(
                convOp, "Batched Conv1D not supported in this pattern"
            );
        }

        return success();
    }
};

struct Conv1DNcwFcwToLinalgConv2DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgConv2D\n";
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Add height dimension (1) to input: [N,C,W] -> [N,C,1,W]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], 1, inputType.getShape()[2]},
            inputType.getElementType()
        );

        auto expandedInput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedInputType, input, inputReassoc);

        // Transpose to NHWC format: [N,C,1,W] -> [N,1,W,C]
        SmallVector<int64_t> inputPerm = {0, 2, 3, 1};
        Value nhwcInput = transposeValue(expandedInput, inputPerm, loc, rewriter);

        // Add height dimension to filter: [F,C,W] -> [F,C,1,W]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], 1, filterType.getShape()[2]},
            filterType.getElementType()
        );

        auto expandedFilter =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedFilterType, filter, filterReassoc);

        // Transpose to HWCF format: [F,C,1,W] -> [1,W,C,F]
        SmallVector<int64_t> filterPerm = {2, 3, 1, 0};
        Value hwcfFilter = transposeValue(expandedFilter, filterPerm, loc, rewriter);

        // Add height dimension to output: [N,F,W] -> [N,F,1,W]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], 1, outputType.getShape()[2]},
            outputType.getElementType()
        );

        auto expandedOutput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedOutputType, output, outputReassoc);

        // Transpose to NHWC format: [N,F,1,W] -> [N,1,W,F]
        SmallVector<int64_t> outputPerm = {0, 2, 3, 1};
        Value nhwcOutput = transposeValue(expandedOutput, outputPerm, loc, rewriter);

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Convert 1D strides/dilations to 2D (add height dimension)
        SmallVector<int64_t> strides2d = {1};
        strides2d.push_back(stridesAttr.getValues<int64_t>()[0]);
        SmallVector<int64_t> dilations2d = {1};
        dilations2d.push_back(dilationsAttr.getValues<int64_t>()[0]);

        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        // Create Conv2D
        auto conv2d = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc, nhwcOutput.getType(), ValueRange{nhwcInput, hwcfFilter}, ValueRange{nhwcOutput},
            stridesAttr2d, dilationsAttr2d
        );

        // Transpose result back: [N,1,W,F] -> [N,F,1,W]
        Value transposedResult = transposeValue(conv2d.getResult(0), {0, 3, 1, 2}, loc, rewriter);

        // Collapse height dimension: [N,F,1,W] -> [N,F,W]
        auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
            loc, outputType, transposedResult, outputReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());
        return success();
    }
};

// Pattern to handle interleaved tensor.insert_slice operations
// This pattern detects insert_slice with stride > 1, which indicates
// an interleaving/upsampling operation (common in transposed convolutions)
struct InterleavedInsertSlicePattern : public OpRewritePattern<tensor::InsertSliceOp> {
    using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::InsertSliceOp insertSliceOp, PatternRewriter &rewriter) const override {

        // Get the source tensor being inserted
        Value source = insertSliceOp.getSource();

        // Get the destination tensor (should be zero-filled for interleaving)
        Value dest = insertSliceOp.getDest();

        // Get static offsets, sizes, and strides
        auto staticOffsets = insertSliceOp.getStaticOffsets();
        auto staticSizes = insertSliceOp.getStaticSizes();
        auto staticStrides = insertSliceOp.getStaticStrides();

        // Check if this is a dynamic insert_slice (not supported)
        if (llvm::any_of(
                staticOffsets, [](int64_t offset) { return ShapedType::isDynamic(offset); }
            ) ||
            llvm::any_of(staticSizes, [](int64_t size) { return ShapedType::isDynamic(size); }) ||
            llvm::any_of(staticStrides, [](int64_t stride) {
                return ShapedType::isDynamic(stride);
            })) {
            return rewriter.notifyMatchFailure(
                insertSliceOp, "Dynamic offsets, sizes, or strides not supported"
            );
        }

        // Check if any stride is > 1 (indicating interleaving)
        bool hasInterleaving = false;
        int interleavedDim = -1;
        int64_t strideValue = 1;

        for (size_t i = 0; i < staticStrides.size(); ++i) {
            if (staticStrides[i] > 1) {
                if (hasInterleaving) {
                    // Multiple dimensions with stride > 1 not supported
                    return rewriter.notifyMatchFailure(
                        insertSliceOp, "Multiple interleaved dimensions not supported"
                    );
                }
                hasInterleaving = true;
                interleavedDim = i;
                strideValue = staticStrides[i];
            }
        }

        // If no interleaving, not our pattern
        if (!hasInterleaving) {
            return rewriter.notifyMatchFailure(
                insertSliceOp, "No interleaving detected (all strides = 1)"
            );
        }

        // Only support stride-2 upsampling (common in transposed convolution)
        if (strideValue != 2) {
            return rewriter.notifyMatchFailure(
                insertSliceOp, "Only stride-2 interleaving is supported (current stride: " +
                                   std::to_string(strideValue) + ")"
            );
        }

        // Get type information
        auto sourceType = cast<RankedTensorType>(source.getType());
        auto destType = cast<RankedTensorType>(dest.getType());

        LLVM_DEBUG({
            llvm::dbgs() << "Source type: " << sourceType << "\n";
            llvm::dbgs() << "Dest type: " << destType << "\n";
        });

        // Expand source to 4D NCHW format if needed
        Value source4D = source;
        auto sourceShape = sourceType.getShape();

        if (sourceType.getRank() == 2) {
            // Expand 2D [H, W] to 4D [1, 1, H, W] (NCHW format)
            SmallVector<ReassociationIndices> expandReassoc = {{0, 1, 2}, {3}};
            SmallVector<int64_t> expanded4DShape = {1, 1, sourceShape[0], sourceShape[1]};
            auto expanded4DType =
                RankedTensorType::get(expanded4DShape, sourceType.getElementType());
            source4D = rewriter.create<tensor::ExpandShapeOp>(
                insertSliceOp.getLoc(), expanded4DType, source, expandReassoc
            );
            sourceType = expanded4DType;
            sourceShape = expanded4DShape;
        }

        // Build the interleaved shape based on 4D NCHW format
        // interleavedDim refers to the dimension in the original insert_slice (4D coords)
        // After expansion, we work with 4D shapes
        SmallVector<int64_t> interleavedShape4D;
        for (size_t i = 0; i < staticSizes.size(); ++i) {
            if (i == static_cast<size_t>(interleavedDim)) {
                // This dimension is interleaved: multiply by stride
                interleavedShape4D.push_back(staticSizes[i] * strideValue);
            }
            else {
                // Keep the size from insert_slice
                interleavedShape4D.push_back(staticSizes[i]);
            }
        }

        // Create the result type for InterleavedInsert (4D NCHW, without padding)
        auto interleavedResultType =
            RankedTensorType::get(interleavedShape4D, sourceType.getElementType());

        // Create init tensor for the interleaved output (4D NCHW, without padding)
        Value interleavedInit = rewriter.create<tensor::EmptyOp>(
            insertSliceOp.getLoc(), interleavedShape4D, sourceType.getElementType()
        );

        // Get element type for determining data type-specific values
        auto elemType = sourceType.getElementType();

        // Set clipping values based on data type
        int32_t output_min, output_max;
        if (elemType.isInteger(8)) {
            output_min = -128; // int8 min
            output_max = 127;  // int8 max
        }
        else if (elemType.isInteger(16)) {
            output_min = -32768; // int16 min
            output_max = 32767;  // int16 max
        }
        else if (elemType.isBF16()) {
            output_min = 0xff800000; // -inf in bf16 (as int32 bits)
            output_max = 0x7f800000; // +inf in bf16 (as int32 bits)
        }
        else {
            return rewriter.notifyMatchFailure(
                insertSliceOp, "Unsupported element type for clipping values"
            );
        }

        // Create weights tensor with interleaving pattern
        // For stride-2: [1, 0] for int8, [1.0, 0.0] for bf16/int16
        // Weight type should match input data type
        Value weights;
        if (elemType.isInteger(8)) {
            // For int8 input, use int8 weights
            std::vector<int8_t> weightsData = {1, 0};
            weights =
                createI8Const(rewriter, insertSliceOp, weightsData, llvm::ArrayRef<int64_t>{2});
        }
        else if (elemType.isInteger(16)) {
            // For int16 input, use int16 weights
            std::vector<int16_t> weightsData = {1, 0};
            weights =
                createI16Const(rewriter, insertSliceOp, weightsData, llvm::ArrayRef<int64_t>{2});
        }
        else if (elemType.isBF16()) {
            // For bf16 input, use bf16 weights (0x3f80 = 1.0, 0x0000 = 0.0)
            std::vector<int16_t> weightsData = {0x3f80, 0x0000};
            weights =
                createI16Const(rewriter, insertSliceOp, weightsData, llvm::ArrayRef<int64_t>{2});
        }
        else {
            return rewriter.notifyMatchFailure(
                insertSliceOp, "Unsupported element type for interleaved insert"
            );
        }

        auto interleavedOp = rewriter.create<torq_hl::InterleavedInsertOp>(
            insertSliceOp.getLoc(), interleavedResultType, interleavedInit,
            rewriter.getI32IntegerAttr(strideValue), rewriter.getI32IntegerAttr(output_min),
            rewriter.getI32IntegerAttr(output_max), weights, source4D
        );

        // Replace the insert_slice with the InterleavedInsertOp output directly
        // No padding handling needed - output is exactly 32x2 (interleaved size)
        rewriter.replaceOp(insertSliceOp, interleavedOp.getOutput());

        LLVM_DEBUG({
            llvm::dbgs() << "Successfully converted interleaved insert_slice to TorqHL op\n";
        });

        return success();
    }
};

// Checker methods for convolutions with input: NHWC, weights: HWC(F) or NCHW, weights: CHW(F)
template <int channelIndex> struct Check {
    static constexpr int kh = channelIndex == 3 ? 0 : 1;
    static constexpr int ih = kh + 1;
    static constexpr int iw = ih + 1, kw = kh + 1;
    static constexpr int maxKerHW = 9;
    using Shape = ArrayRef<int64_t>;

    // Check that the kernel shape is small enough
    static bool isKerSmall(Shape iShape, Shape wShape, Shape padShape) {
        return iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] <= maxKerHW &&
               wShape[kw] <= maxKerHW;
    }

    // Check that the kernel shape is equal to the input shape (without padding)
    static bool isKerEqInput(Shape iShape, Shape wShape, Shape padShape) {
        bool noPadding = llvm::all_of(padShape, [](auto p) { return p == 0; });
        return noPadding && iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] > 1 &&
               wShape[kw] > 1 && iShape[ih] == wShape[kh] && iShape[iw] == wShape[kw];
    }
};

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv2dConvert<linalg::Conv2DNhwcHwcfOp, syna::torq_hl::Conv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwcf2fchw(), 28, 12,
        Check<3>::isKerSmall, markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwc2chw(), 20, 12, Check<3>::isKerSmall,
        markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::none(), Permutation::none(), 20, 12, Check<3>::isKerEqInput,
        markFuseGroups
    );

    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNchwChwOp, torq_hl::DepthwiseConv2DOp>>(
        context, 1, Permutation::none(), Permutation::none(), 20, 12, Check<1>::isKerSmall,
        markFuseGroups, true
    );
    patterns.insert<Conv2dConvert<linalg::Conv2DNchwFchwOp, syna::torq_hl::Conv2DOp>>(
        context, 1, Permutation::none(), Permutation::none(), 28, 12, Check<1>::isKerSmall,
        markFuseGroups, true
    );

    patterns.insert<PoolingNhwcMaxOpConversion>(context, markFuseGroups);

    patterns.insert<FCMatmulOpConversion<linalg::MatmulOp>>(context, markFuseGroups);
    patterns.insert<FCMatmulOpConversion<linalg::MatmulTransposeBOp>>(context, markFuseGroups);

    patterns.insert<Conv2DMatmulOpConversion>(context, markFuseGroups);
    patterns.insert<Conv2DNchwMatmulOpConversion>(context, markFuseGroups);
    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    else {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }
}

} // namespace mlir::syna::torq
