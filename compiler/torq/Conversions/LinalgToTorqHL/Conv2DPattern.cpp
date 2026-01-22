// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
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

#define DEBUG_TYPE "linalg-torq-conv2d-pattern"

namespace mlir::syna::torq {

static FailureOr<Value> expandWeightsForDilation(
    Value weights, ArrayRef<int64_t> dilations, PatternRewriter &rewriter, int64_t maxKernelSize = 7
) {
    auto weightType = mlir::cast<ShapedType>(weights.getType());
    auto shape = weightType.getShape();
    auto elemType = weightType.getElementType();

    int64_t kh = shape[shape.size() - 2];
    int64_t kw = shape[shape.size() - 1];
    int64_t dh = dilations[0], dw = dilations[1];

    int64_t khNew = kh + (kh - 1) * (dh - 1);
    int64_t kwNew = kw + (kw - 1) * (dw - 1);

    if (khNew > maxKernelSize || kwNew > maxKernelSize) {
        LLVM_DEBUG(llvm::dbgs() << "Expanded kernel size exceeds limit\n");
        return failure();
    }

    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[newShape.size() - 2] = khNew;
    newShape[newShape.size() - 1] = kwNew;

    Location loc = rewriter.getUnknownLoc();
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, newShape, elemType);

    TypedAttr zeroAttr;
    if (elemType.isBF16() || elemType.isF16() || elemType.isF32()) {
        zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    }
    else if (elemType.isInteger()) {
        zeroAttr = rewriter.getIntegerAttr(elemType, 0);
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported element type for dilation expansion\n");
        return failure();
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
        loc, RankedTensorType::get(newShape, elemType), ValueRange{weights},
        ValueRange{filledTensor.getResult(0)}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
        }
    );

    genericOp->setAttr("torq-compile-time-const", rewriter.getBoolAttr(true));

    return genericOp.getResult(0);
}

// Helper function to convert tensor.insert_slice with stride > 1 to InterleavedInsertOp
// and calculate padding for transposed convolution
static mlir::FailureOr<Value> convertToInterleaved(
    Value input, PatternRewriter &rewriter, Operation *parentOp, bool hasStridedInsertSlice
) {
    if (!hasStridedInsertSlice) {
        return failure();
    }
    auto insertSliceOp = input.getDefiningOp<tensor::InsertSliceOp>();
    if (!insertSliceOp) {
        return failure();
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
        return failure();
    }

    Value source = insertSliceOp.getSource();
    Value dest = insertSliceOp.getDest();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto destType = cast<RankedTensorType>(dest.getType());
    auto destShape = destType.getShape();
    auto elemType = sourceType.getElementType();
    auto loc = input.getLoc();

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
        return failure();
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
        return failure();
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

    LLVM_DEBUG({
        llvm::dbgs(
        ) << "Converted strided insert_slice to InterleavedInsertOp with padding applied\n";
    });
    return interleavedOutput;
}

// Static helper functions for Conv2D conversion
template <class LinalgConvOp>
static bool checkDW1DStride1(LinalgConvOp convOp, Value weights, int channelDim) {
    auto weightType = llvm::cast<RankedTensorType>(weights.getType());
    auto weightShape = weightType.getShape();

    int weightHeightDim = (channelDim == 3) ? 2 : 1;
    bool filterHeightIs1 = (weightShape[weightHeightDim] == 1);
    auto strides = convOp.getStrides().template getValues<int64_t>();
    bool strideHeightIs1 = (strides[0] == 1);

    if (filterHeightIs1 && strideHeightIs1) {
        // Check if expanded kernel size exceeds 7x7 limit of regular 2D depthwise conv
        // Only use 1D stride-1 path if expansion would exceed this limit
        auto dilations = convOp.getDilations().template getValues<int64_t>();
        int64_t kh = weightShape[weightShape.size() - 2];
        int64_t kw = weightShape[weightShape.size() - 1];
        int64_t dh = dilations[0], dw = dilations[1];
        int64_t khExpanded = kh + (kh - 1) * (dh - 1);
        int64_t kwExpanded = kw + (kw - 1) * (dw - 1);

        // Use 1D stride-1 path only if expanded kernel exceeds 7x7 (2D limit)
        if (khExpanded > 7 || kwExpanded > 7) {
            return true;
        }
    }
    return false;
}

static bool isStridedInsertSlice(Value input, bool isDW1DStride1) {
    if (isDW1DStride1) {
        return false;
    }
    if (auto insertSliceOp = input.getDefiningOp<tensor::InsertSliceOp>()) {
        auto staticStrides = insertSliceOp.getStaticStrides();
        for (size_t i = 0; i < staticStrides.size(); ++i) {
            if (staticStrides[i] == 2) {
                return true;
            }
        }
    }
    return false;
}

static PaddingInfo populatePadInfo(Value input, bool hasStridedInsertSlice, int channelDim) {
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
                int hDim = (channelDim == 3) ? 1 : 2;
                int wDim = (channelDim == 3) ? 2 : 3;
                prelimPadInfo.lrtbPad = {
                    lp[wDim], lp[hDim], hp[wDim], hp[hDim]
                }; // [left, top, right, bottom]
            }
        }
    }
    return prelimPadInfo;
}

static mlir::FailureOr<Value> getDilatedWts(
    Value weights, std::vector<int64_t> &finalDilationVec, bool isDW1DStride1,
    PatternRewriter &rewriter
) {
    SmallVector<int64_t> dilationVec(finalDilationVec.begin(), finalDilationVec.end());

    auto maxKernelSize = isDW1DStride1 ? 1024 : 7;

    if (llvm::any_of(dilationVec, [](int64_t d) { return d > 1; })) {
        auto expanded = expandWeightsForDilation(weights, dilationVec, rewriter, maxKernelSize);
        if (failed(expanded)) {
            LLVM_DEBUG(llvm::dbgs() << "expanded kernel size exceeds 7x7 limit\n");
            return failure();
        }
        finalDilationVec = {1, 1};
        weights = *expanded;
    }
    return weights;
}

static bool modifyWeightWithZp(
    Value &weights, int32_t weightZp, VectorIntOrFloat &bias, ScaleClampInfo &scInfo,
    PatternRewriter &rewriter
) {
    if (weightZp != 0) {
        // Divide weights and wZp by two so we can safely add them together without overlflow
        constexpr int scaleFactor = 2;
        weights = rescaleValue(weights, scaleFactor, -weightZp / 2, weights.getLoc(), rewriter);
        // Same for the bias
        for (auto &val : bias.ints) {
            val /= scaleFactor;
        }
        // Reduce the scaling shift to compensate
        // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
        scInfo.scaleShift -= 1;
        return true;
    }
    return false;
}

static Value preConversion(
    Value input, Permutation dataPerm, Location loc, PatternRewriter &rewriter, bool isNchw,
    bool isDW1DStride1
) {
    if (!isNchw && isDW1DStride1) {
        return input;
    }
    if (isNchw && !isDW1DStride1) {
        return input;
    }
    if (isNchw && isDW1DStride1 && dataPerm == Permutation::none()) {
        dataPerm = Permutation::nhwc2nchw().reverse();
    }
    LLVM_DEBUG(llvm::dbgs() << "Transposing input before Conv2D\n";
               llvm::dbgs() << "IsNCHW: " << isNchw << ", isDW1DStride1: " << isDW1DStride1 << "\n";
               llvm::dbgs() << "Using data permutation: "; for (auto d
                                                                : dataPerm) {
                   llvm::dbgs() << d << " ";
               } llvm::dbgs() << "\n";);
    auto newInput = transposeValue(input, dataPerm, loc, rewriter);
    return newInput;
}

static Value createOutput(
    Value output, Permutation dataPerm, PatternRewriter &rewriter, bool isNchw, bool isDW1DStride1
) {
    if (!isNchw && isDW1DStride1) {
        dataPerm = Permutation::none();
    }
    if (isNchw && !isDW1DStride1) {
        dataPerm = Permutation::none();
    }
    if (isNchw && isDW1DStride1 && dataPerm == Permutation::none()) {
        dataPerm = Permutation::nhwc2nchw().reverse();
    }
    auto outTy = mlir::cast<RankedTensorType>(output.getType());
    auto torqOutType = transposeType(outTy, dataPerm);
    auto newOutput = createInitTensor(output, rewriter, torqOutType);
    return newOutput;
}

static Value preConversionWeights(
    Value weights, const Permutation &_weightsPerm, int32_t weightZp, VectorIntOrFloat &bias,
    ScaleClampInfo &scInfo, PatternRewriter &rewriter
) {
    auto transposedWeights = transposeValue(weights, _weightsPerm, weights.getLoc(), rewriter);

    modifyWeightWithZp(transposedWeights, weightZp, bias, scInfo, rewriter);
    return transposedWeights;
}

static Value postConversion(
    Value output, Permutation dataPerm, bool isNchw, bool isDW1DStride1, PatternRewriter &rewriter
) {
    auto loc = output.getLoc();
    if (!isNchw && isDW1DStride1) {
        return output;
    }
    if (isNchw && !isDW1DStride1) {
        return output;
    }
    if (isNchw && isDW1DStride1 && dataPerm == Permutation::none()) {
        dataPerm = Permutation::nhwc2nchw().reverse();
    }
    auto newOutput = transposeValue(output, dataPerm.reverse(), loc, rewriter);
    return newOutput;
}

static bool isLinalgDW(Operation *op) {
    return llvm::isa<linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNchwChwOp>(op);
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

        // Guard: Skip conversion if this Conv2D was created from Conv1D and marked
        // to prevent infinite loop. This prevents Conv2dConvert from converting
        // Conv2D (height=1) back to Conv1D, which would cause an infinite loop
        // with Conv1DNcwFcwToLinalgConv2DPattern.
        if (convOp->hasAttr("torq.skip_conv2d_to_conv1d")) {
            LLVM_DEBUG(llvm::dbgs() << "Skipping Conv2D to Conv1D conversion\n");
            return rewriter.notifyMatchFailure(
                convOp, "Skipping Conv2D to Conv1D conversion to prevent infinite loop"
            );
        }

        bool isDepthwise = isLinalgDW(convOp.getOperation());
        // Todo: Capability check for depthwise conv should be moved to a helper function
        // Check for depthwise conv specific constraints
        if (isDepthwise) {

            auto strides = convOp.getStrides().template getValues<int64_t>();
            // hk kernel
            if (strides[1] != 1 && (strides[0] > 2 || strides[0] != strides[1])) {
                return rewriter.notifyMatchFailure(
                    convOp, "asymmetric strides or stride > 2 not supported by DW"
                );
            }
            // Dilation check removed - now handled by weight expansion below
            // (dilations > 1 will be converted to dilation = 1 via weight expansion)
        }

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

        bool isDW1DStride1 = false;
        // Check if this is a depthwise 1D case
        if (isConv1D && isDepthwise) {
            isDW1DStride1 = checkDW1DStride1(convOp, weights, _channelDim);
            LLVM_DEBUG(
                if (isDW1DStride1) llvm::dbgs() << "Detected depthwise 1D stride-1 convolution\n"
            );
        }

        if (isConv1D && !isDepthwise) {
            // FIXME : Handle non-depthwise Conv1D case separately for now
            LLVM_DEBUG(llvm::dbgs() << "Rewriting Conv2D (1D) to Conv1D\n");
            return rewriteAsConv1D(convOp, rewriter);
        }
        int groups = isDW1DStride1 ? inputType.getShape()[_channelDim] : 1;
        const auto loc = convOp.getLoc();

        // First, check if input has strided insert_slice pattern (but don't convert yet)
        bool hasStridedInsertSlice = isStridedInsertSlice(input, isDW1DStride1);
        LLVM_DEBUG({
            if (hasStridedInsertSlice) {
                llvm::dbgs() << "Input has strided insert_slice pattern\n";
            }
        });
        // Get preliminary padding info without modifying IR yet
        // For strided insert slice cases, no preliminary padding needed (handled later)
        // For regular cases, we'll compute padding after validation passes
        PaddingInfo prelimPadInfo = populatePadInfo(input, hasStridedInsertSlice, _channelDim);

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

        LLVM_DEBUG({
            llvm::dbgs() << "Folding per-channel bias\n";
            llvm::dbgs() << "Current output\n";
            output.print(llvm::dbgs());
        });

        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }
        LLVM_DEBUG({
            llvm::dbgs() << "Output after bias folding\n";
            output.print(llvm::dbgs());
        });

        // Fold operations that take care of zero-point in weight quantization if present
        int weightZp = foldForwardWeightZp(output);
        LLVM_DEBUG({
            llvm::dbgs() << "Output after weights folding\n";
            output.print(llvm::dbgs());
            llvm::dbgs() << "Weight Zp folded: " << weightZp << "\n";
        });

        // Fold any additional per-channel bias
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }
        LLVM_DEBUG({
            llvm::dbgs() << "Output after more bias folding\n";
            output.print(llvm::dbgs());
        });

        // Fold scale and clamp. This is mandatory for integer operations.
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, false);
        LLVM_DEBUG({
            llvm::dbgs() << "Output after more scale folding\n";
            output.print(llvm::dbgs());
        });
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
        auto torqWeights =
            preConversionWeights(weights, _weightsPerm, weightZp, bias, scInfo, rewriter);

        auto dilations = convOp.getDilations().template getValues<int64_t>();
        std::vector<int64_t> finalDilationVec(dilations.begin(), dilations.end());
        auto mWts = getDilatedWts(torqWeights, finalDilationVec, isDW1DStride1, rewriter);
        if (!failed(mWts)) {
            torqWeights = *mWts;
        }
        auto weightsAttr = computeConstant(torqWeights);
        if (!weightsAttr) {
            return rewriter.notifyMatchFailure(convOp, "Weights must be constant");
        }

        torqWeights = createConst(weightsAttr, rewriter, weights.getLoc());

        bool isNchw = _2DNchwChw;
        // NOW that all validation passed (including weight creation),
        // convert strided insert_slice to InterleavedInsertOp OR fold backward padding
        PaddingInfo padInfo{{0, 0, 0, 0}, 0};
        if (failed(convertToInterleaved(input, rewriter, convOp, hasStridedInsertSlice))) {
            // Fallback to regular padding if conversion failed
            // Use correct layout: NCHW if channelDim==1, NHWC if channelDim==3
            padInfo = foldBackwardPadding(input, rewriter, isNchw);
        }

        inputType = cast<RankedTensorType>(input.getType());
        shape = inputType.getShape();

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with input/output in the expected format
        input = preConversion(input, _dataPerm, loc, rewriter, isNchw, isDW1DStride1);
        bool nhwcInput = (!isNchw && _dataPerm == Permutation::none()) || isDW1DStride1;
        auto newConvOutput = createOutput(output, _dataPerm, rewriter, isNchw, isDW1DStride1);
        auto torqOutType = newConvOutput.getType();

        Value outV;
        if (isDepthwise) {
            outV = rewriter
                       .create<torq_hl::DepthwiseConv2DOp>(
                           loc, torqOutType, newConvOutput, padInfo.padValue, 0, scInfo.zp,
                           scInfo.min, scInfo.max, scInfo.scaleShift, groups, padInfo.lrtbPad,
                           attrValues(convOp.getStrides()), finalDilationVec,
                           torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, input,
                           isDW1DStride1, false, isDW1DStride1
                       )
                       .getResult(0);
        }
        else {
            outV = rewriter
                       .create<torq_hl::Conv2DOp>(
                           loc, torqOutType, newConvOutput, padInfo.padValue, 0, scInfo.zp,
                           scInfo.min, scInfo.max, scInfo.scaleShift, groups, padInfo.lrtbPad,
                           attrValues(convOp.getStrides()), finalDilationVec,
                           torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, input,
                           nhwcInput
                       )
                       .getResult(0);
        }
        auto torqOut = postConversion(outV, _dataPerm, isNchw, isDW1DStride1, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    Value createConv1DOutput(
        Value output, Value weights, Permutation dataPerm, bool useConv1dWithReduce,
        PatternRewriter &rewriter
    ) const {
        auto outTy = transposeType(output.getType(), dataPerm);
        if (useConv1dWithReduce) {
            auto weightType = cast<RankedTensorType>(weights.getType());
            int64_t filter_len = weightType.getShape()[1];
            outTy = mlir::cast<RankedTensorType>(output.getType());
            llvm::SmallVector<int64_t> outSh(outTy.getShape().begin(), outTy.getShape().end());
            outSh.push_back(filter_len);
            outTy = RankedTensorType::get(outSh, outTy.getElementType());
        }
        return createInitTensor(output, rewriter, outTy);
    }

    bool checkConv1dWithReduce(Value input, Value output) const {
        auto inputType = mlir::cast<RankedTensorType>(input.getType());
        auto outputType = mlir::cast<RankedTensorType>(output.getType());
        auto channels = inputType.getShape()[3];
        auto outputChannels = outputType.getShape()[3];
        // Check if input and output types are compatible with Conv1D + Reduce
        bool compatible = channels > 1 || outputChannels > 1;
        return compatible;
    }

    template <typename LinalgConv>
    Value preConversionConv1D(
        LinalgConv convOp, Value input, Value weights, Value output, bool useConv1dWithReduce,
        Permutation dataPerm, PatternRewriter &rewriter
    ) const {
        if (useConv1dWithReduce) {
            input = transposeValue(input, dataPerm, input.getLoc(), rewriter);
            // Create type for Conv1D output with an extra dimension at the end.
            // This will be reduced later with linalg.reduce.
        }
        else {
            auto outTy = mlir::cast<RankedTensorType>(output.getType());
            int64_t out_len = outTy.getShape()[2];

            auto weightType = cast<RankedTensorType>(weights.getType());
            int64_t filter_len = weightType.getShape()[1];
            int64_t op_rows = filter_len;
            int64_t op_cols = out_len;

            // Note: op is Conv2DNhwcHwcfOp
            auto inputType = mlir::cast<RankedTensorType>(input.getType());
            int64_t batch = inputType.getShape()[0];
            int64_t channels = inputType.getShape()[3];

            llvm::SmallVector<int64_t> transposedShape = {batch, channels, op_rows, op_cols};
            RankedTensorType transposedType =
                RankedTensorType::get(transposedShape, inputType.getElementType());

            llvm::SmallVector<int64_t> permVals = {1, 0};
            auto permAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), permVals);

            auto transposeReshape = rewriter.create<torq_hl::TransposeReshapeOp>(
                input.getLoc(), transposedType, createInitTensor(convOp, rewriter, transposedType),
                attrValues(convOp.getStrides()), weightType.getShape(), permAttr, input
            );
            input = transposeReshape.getOutput();
        }
        return input;
    }

    Value finalizeConv1DConversion(
        Value torqOut, bool useConv1dWithReduce, PatternRewriter &rewriter
    ) const {
        auto loc = torqOut.getLoc();
        auto torqOutType = mlir::cast<RankedTensorType>(torqOut.getType());
        if (useConv1dWithReduce) {
            auto reducedShape = torqOutType.getShape().drop_back();
            auto elemType = torqOutType.getElementType();
            Type reduceElemType = elemType;
            Value reduceInput = torqOut;
            bool needsF32Conversion = elemType.isF32();

            if (needsF32Conversion) {
                reduceElemType = rewriter.getBF16Type();
                auto bf16Type = RankedTensorType::get(torqOutType.getShape(), reduceElemType);
                reduceInput = rewriter.create<arith::TruncFOp>(loc, bf16Type, torqOut);
            }

            Value zeroValue = createZeroConstant(rewriter, loc, reduceElemType);
            auto cEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedShape, reduceElemType);
            Value zeroTensor =
                rewriter.create<linalg::FillOp>(loc, ValueRange{zeroValue}, ValueRange{cEmpty})
                    .result();
            linalg::ReduceOp reduceOp = rewriter.create<linalg::ReduceOp>(
                loc, ValueRange{reduceInput}, ValueRange{zeroTensor}, 4,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                    );
                }
            );

            torqOut = reduceOp->getResult(0);
            if (needsF32Conversion) {
                auto f32Type = RankedTensorType::get(reducedShape, elemType);
                torqOut = rewriter.create<arith::ExtFOp>(loc, f32Type, torqOut);
            }
        }

        // // Overwrite torqOut with init tensor for debugging
        // torqOut = createInitTensor(convOp, rewriter, cast<RankedTensorType>(torqOut.getType()));
        // // Fill input with 1s for debugging
        // torqOut = rewriter.create<torq_hl::FillOp>(
        //     loc, cast<RankedTensorType>(torqOut.getType()), torqOut,
        //     rewriter.getI32IntegerAttr(/*0x3f800000*//*0x00003f80*/0)
        // ).getOutput();

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);
        return torqOut;
    }

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

        // Decide whether to use Conv1D with reduction or TransposeReshape + Conv1D
        // The former is completely generic but probably less efficient for single-channel cases
        // The latter is more efficient but only works for single-channel input and outputs.
        bool useConv1dWithReduce = checkConv1dWithReduce(input, output);

        strideValue = useConv1dWithReduce ? strideValue : 1;

        input = preConversionConv1D(
            convOp, input, weights, output, useConv1dWithReduce, _dataPerm, rewriter
        );
        auto conv1DOut =
            createConv1DOutput(output, weights, _dataPerm, useConv1dWithReduce, rewriter);

        auto torqOutType = conv1DOut.getType();

        llvm::SmallVector<int64_t> zeroPad(4, 0);
        llvm::SmallVector<int64_t> stride = {strideValue};

        auto torqConv1Op = rewriter.create<torq_hl::Conv1DOp>(
            loc, torqOutType, conv1DOut, 0, weightZp, scInfo.zp, scInfo.min, scInfo.max,
            scInfo.scaleShift, groups, zeroPad, stride, attrValues(convOp.getDilations()),
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, input
        );
        Value torqOut = torqConv1Op.getOutput();

        torqOut = finalizeConv1DConversion(torqOut, useConv1dWithReduce, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
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

void populateLinalgToTorqHLConv2DPatterns(
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
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNchwChwOp, torq_hl::DepthwiseConv2DOp>>(
        context, 1, Permutation::none(), Permutation::none(), 20, 12, Check<1>::isKerSmall,
        markFuseGroups, true
    );
    patterns.insert<Conv2dConvert<linalg::Conv2DNchwFchwOp, syna::torq_hl::Conv2DOp>>(
        context, 1, Permutation::none(), Permutation::none(), 28, 12, Check<1>::isKerSmall,
        markFuseGroups, true
    );
}

} // namespace mlir::syna::torq
