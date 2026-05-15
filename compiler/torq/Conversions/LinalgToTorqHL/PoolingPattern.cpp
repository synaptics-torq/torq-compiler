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

#define DEBUG_TYPE "linalg-torq-pooling-pattern"

namespace mlir::syna::torq {

template <bool IsNCHW>
static Value applyInputTranspose(
    Value input, RankedTensorType &resultType, Location loc, PatternRewriter &rewriter
) {
    if constexpr (IsNCHW) {
        // NCHW is already in correct format, no transpose needed
        return input;
    }
    else {
        // NHWC needs transpose to NCHW
        if (resultType.getRank() == 4) {
            auto dataPerm = Permutation::nhwc2nchw();
            resultType = transposeType(resultType, dataPerm);
            return transposeValue(input, dataPerm, loc, rewriter);
        }
        return input;
    }
}

template <bool IsNCHW>
static Value applyOutputTranspose(
    Value output, RankedTensorType resultType, Location loc, PatternRewriter &rewriter
) {
    if constexpr (IsNCHW) {
        return output;
    }
    else {
        if (resultType.getRank() == 4) {
            auto dataPerm = Permutation::nhwc2nchw();
            return transposeValue(output, dataPerm.reverse(), loc, rewriter);
        }
        return output;
    }
}

static bool is1DPooling(llvm::ArrayRef<int64_t> kernels, llvm::ArrayRef<int64_t> strides) {
    return (kernels[0] == 1 && strides[0] == 1) || (kernels[1] == 1 && strides[1] == 1);
}

/// Look for a downstream linalg.generic that performs an elementwise mulf or divf
/// by a constant. If found, return the constant scale value and the generic op.
/// For divf, the scale is the reciprocal of the divisor.
static std::pair<float, linalg::GenericOp> tryExtractFusedScale(Value output) {
    if (!output.hasOneUse())
        return {1.0f, nullptr};

    auto genericOp = dyn_cast<linalg::GenericOp>(*output.getUsers().begin());
    if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
        genericOp.getNumReductionLoops() != 0)
        return {1.0f, nullptr};

    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return {1.0f, nullptr};

    auto mulOp = dyn_cast<arith::MulFOp>(yieldOp.getOperand(0).getDefiningOp());
    auto divOp = dyn_cast<arith::DivFOp>(yieldOp.getOperand(0).getDefiningOp());
    if (!mulOp && !divOp)
        return {1.0f, nullptr};

    Value lhs = mulOp ? mulOp.getLhs() : divOp.getLhs();
    Value rhs = mulOp ? mulOp.getRhs() : divOp.getRhs();

    bool lhsIsBlockArg =
        isa<BlockArgument>(lhs) && cast<BlockArgument>(lhs).getOwner() == genericOp.getBody();
    bool rhsIsBlockArg =
        isa<BlockArgument>(rhs) && cast<BlockArgument>(rhs).getOwner() == genericOp.getBody();

    arith::ConstantOp constOp = nullptr;
    if (lhsIsBlockArg && !rhsIsBlockArg)
        constOp = rhs.getDefiningOp<arith::ConstantOp>();
    else if (rhsIsBlockArg && !lhsIsBlockArg)
        constOp = lhs.getDefiningOp<arith::ConstantOp>();

    if (!constOp)
        return {1.0f, nullptr};

    auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!floatAttr)
        return {1.0f, nullptr};

    float value = floatAttr.getValueAsDouble();
    if (divOp) {
        if (value == 0.0f)
            return {1.0f, nullptr};
        value = 1.0f / value;
    }
    return {value, genericOp};
}

template <typename PoolingOpType, bool IsNCHW>
struct PoolingMaxOpConversionBase : public OpRewritePattern<PoolingOpType> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern<PoolingOpType>::OpRewritePattern;
    PoolingMaxOpConversionBase(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<PoolingOpType>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(PoolingOpType srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }
        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResults()[0];

        auto attrStrides = attrValuesAsVec(srcOp.getStrides());
        if (attrStrides.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two strides for PoolingMaxOp"
            );
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        PaddingInfo padInfo = foldBackwardPadding(input, rewriter, IsNCHW);

        auto kernel = srcOp.getInputs()[1];
        auto kernelsShape = mlir::cast<RankedTensorType>(kernel.getType()).getShape();
        if (kernelsShape.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingMaxOp"
            );
        }

        // Check stride constraints after determining if 1D pooling
        bool is1D = is1DPooling(kernelsShape, attrStrides);
        if (!is1D) {
            // For 2D pooling, enforce stride limits
            constexpr int64_t maxStride = IsNCHW ? 4 : 2;
            if (attrStrides[0] > maxStride || attrStrides[1] > maxStride) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Stride exceeds maximum supported for 2D pooling"
                );
            }
        }
        // For 1D pooling, allow larger strides in the pooling dimension

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, kernel}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        input = applyInputTranspose<IsNCHW>(input, srcResultType, loc, rewriter);

        auto elementType = srcResultType.getElementType();
        int32_t outputMin, outputMax;
        Value weightConst;

        if (elementType.isInteger(8)) {
            outputMin = -128;
            outputMax = 127;
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }
        else if (elementType.isInteger(16)) {
            outputMin = -32768;
            outputMax = 32767;
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }
        else if (elementType.isBF16()) {
            outputMin = 0xff800000; // -inf in bf16 (as int32 bits)
            outputMax = 0x7f800000; // +inf in bf16 (as int32 bits)
            // Create BF16 weight with value 1.0
            auto bf16Type = rewriter.getBF16Type();
            auto weightType = RankedTensorType::get({1, 1, 1, 1}, bf16Type);
            auto bf16One = rewriter.getFloatAttr(bf16Type, 1.0);
            auto weightAttr = DenseElementsAttr::get(weightType, bf16One);
            weightConst = arith::ConstantOp::create(rewriter, loc, weightType, weightAttr);
        }
        else {
            outputMin = std::numeric_limits<int32_t>::min();
            outputMax = std::numeric_limits<int32_t>::max();
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }

        auto maxpoolOp = torq_hl::MaxPool2dOp::create(
            rewriter, loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType),
            padInfo.padValue, outputMin, outputMax, attrStrides, padInfo.lrtbPad, kernelsShape,
            weightConst, createI32Const(rewriter, srcOp, interleave(bias, scale)), input,
            /*segment_output=*/false
        );

        Value result =
            applyOutputTranspose<IsNCHW>(maxpoolOp.getOutput(), srcResultType, loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), result);

        return success();
    }
};

struct PoolingNhwcMaxOpConversion
    : public PoolingMaxOpConversionBase<linalg::PoolingNhwcMaxOp, false> {
    using PoolingMaxOpConversionBase::PoolingMaxOpConversionBase;
};

struct PoolingNchwMaxOpConversion
    : public PoolingMaxOpConversionBase<linalg::PoolingNchwMaxOp, true> {
    using PoolingMaxOpConversionBase::PoolingMaxOpConversionBase;
};

struct PoolingNhwcSumOpConversion : public OpRewritePattern<linalg::PoolingNhwcSumOp> {
  public:
    using LinalgOpT = linalg::PoolingNhwcSumOp;
    using TorqOp = torq_hl::AvgPool2DOp;
    using OpRewritePattern<LinalgOpT>::OpRewritePattern;

    PoolingNhwcSumOpConversion(MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups)
        : OpRewritePattern<LinalgOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgOpT linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        // Get the inputs and output of the original operation
        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();
        if (kernelShape.size() != 2 || kernelShape[0] != in_s[1] || kernelShape[1] != in_s[2]) {
            return rewriter.notifyMatchFailure(linalgOp, "Only kernel == whole frame supported");
        }
        int itemsPerChannel = in_s[1] * in_s[2];

        auto dataPerm = Permutation::none();

        bool isInt = outType.getElementType().isInteger();
        if (!isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Only integer pooling supported");
        }

        const int channelDimension = 3;
        int32_t channelCount = in_s[channelDimension];

        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, channelCount, _shift8b, _shift16b, true);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Cannot fold forward scale/clamp");
        }

        // Don't fold input rescale here. Input zp is applied after pooling in the ForwardScale
        ScaleInfo scaleInput0;

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input0, kernel}, rewriter,
                linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Generate torq_hl op with input in the expected format
        input0 = transposeValue(input0, dataPerm, loc, rewriter);

        // Compute scale and bias vectors
        const std::vector<int32_t> scale(
            channelCount, int32_t(scaleInput0.scale * (1 << scInfo.scaleShift) / itemsPerChannel)
        );
        const std::vector<int32_t> bias(channelCount, -scInfo.bias);

        // Prepare bias (and scale for integer ops)
        auto biasScale = createConst(interleave(bias, scale), rewriter, loc);

        // Prepare weights
        auto weights = createI8Const(
            rewriter, linalgOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        // Generate torq_hl op with output in the expected format
        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = TorqOp::create(
            rewriter, linalgOp.getLoc(), torqOutType,
            createInitTensor(linalgOp, rewriter, torqOutType), scInfo.zp, scInfo.zp, scInfo.min,
            scInfo.max, scInfo.scaleShift, weights, biasScale, input0
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

struct PoolingNchwSumOpConversion : public OpRewritePattern<linalg::PoolingNchwSumOp> {
  public:
    using LinalgOpT = linalg::PoolingNchwSumOp;
    using TorqOp = torq_hl::AvgPool2DOp;
    using OpRewritePattern<LinalgOpT>::OpRewritePattern;

    PoolingNchwSumOpConversion(MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups)
        : OpRewritePattern<LinalgOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgOpT linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();
        if (kernelShape.size() != 2 || kernelShape[0] != in_s[2] || kernelShape[1] != in_s[3]) {
            return rewriter.notifyMatchFailure(linalgOp, "Only kernel == whole frame supported");
        }
        int itemsPerChannel = in_s[2] * in_s[3];

        // AvgPool2DOp expects NHWC input; transpose NCHW -> NHWC
        auto dataPerm = Permutation::nhwc2nchw().reverse();

        bool isInt = outType.getElementType().isInteger();
        bool isBF16 = outType.getElementType().isBF16();
        if (!isInt && !isBF16) {
            return rewriter.notifyMatchFailure(linalgOp, "Only integer or bf16 pooling supported");
        }

        const int channelDimension = 1;
        int32_t channelCount = in_s[channelDimension];

        // Try to fold a downstream elementwise mulf/divf generic op into the avgpool weight.
        auto [scaleValue, fusedGenericOp] = tryExtractFusedScale(output);

        if (_markFuseGroups) {
            // If we found a downstream fused generic, update output so that
            // markFuseGroupBackward walks through it and marks it in the same group.
            if (fusedGenericOp) {
                output = fusedGenericOp.getResult(0);
            }
            markFuseGroupBackward(
                output, {input0, kernel}, rewriter,
                linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Transpose input NCHW -> NHWC
        input0 = transposeValue(input0, dataPerm, loc, rewriter);

        Value weights;
        Value biasScale;
        int32_t inputZp = 0;
        int32_t outputZp = 0;
        int32_t outputMin = 0;
        int32_t outputMax = 0;
        int32_t shiftFactor = 0;

        if (isInt) {
            ScaleClampInfo scInfo =
                foldForwardScaleClamp(output, channelCount, _shift8b, _shift16b, true);
            if (!scInfo) {
                return rewriter.notifyMatchFailure(linalgOp, "Cannot fold forward scale/clamp");
            }
            ScaleInfo scaleInput0;
            const std::vector<int32_t> scale(
                channelCount,
                int32_t(scaleInput0.scale * (1 << scInfo.scaleShift) / itemsPerChannel)
            );
            const std::vector<int32_t> bias(channelCount, -scInfo.bias);
            biasScale = createConst(interleave(bias, scale), rewriter, loc);
            weights = createI8Const(
                rewriter, linalgOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
            );
            inputZp = scInfo.zp;
            outputZp = scInfo.zp;
            outputMin = scInfo.min;
            outputMax = scInfo.max;
            shiftFactor = scInfo.scaleShift;
        }
        else {
            const llvm::fltSemantics &bf16sem = APFloat::BFloat();
            std::vector<llvm::APFloat> weightsData(
                1, llvm::APFloat(bf16sem, std::to_string(scaleValue))
            );
            weights = createConst(weightsData, rewriter, loc);
            std::vector<float> biasScaleData{0.0f};
            biasScale = createConst(biasScaleData, rewriter, loc);
            outputMin = 0xff800000;
            outputMax = 0x7f800000;
        }

        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = TorqOp::create(
            rewriter, linalgOp.getLoc(), torqOutType,
            createInitTensor(linalgOp, rewriter, torqOutType), inputZp, outputZp, outputMin,
            outputMax, shiftFactor, weights, biasScale, input0
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        if (fusedGenericOp) {
            rewriter.replaceOp(fusedGenericOp, torqOut);
            rewriter.eraseOp(linalgOp);
        }
        else {
            rewriter.replaceOp(linalgOp, torqOut);
        }
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

/// Convert linalg.pooling_nchw_sum (local average pooling) to DepthwiseConv2D.
///
/// A depthwise conv2d can implement local average pooling because the two
/// operations are mathematically equivalent when the depthwise weights are
/// initialized correctly:
///   AvgPool(x) = 1/(KH·KW) · Σ(xᵢ)
///   DWConv(x)  = Σ(xᵢ · wᵢ) + bias
/// With wᵢ = 1/(KH·KW) and bias = 0, DWConv(x) = AvgPool(x).
///
/// This pattern:
/// 1. Matches linalg.pooling_nchw_sum.
/// 2. Looks for a downstream mulf/divf generic that scales by 1/(KH·KW).
/// 3. Creates depthwise conv2d weights of shape [C, 1, KH, KW] filled with
///    that scale value.
/// 4. Sets bias to zeros and groups = channelCount (making it depthwise).
/// 5. Replaces both the pooling op and the scale generic with a single
///    DepthwiseConv2DOp.
struct PoolingNchwSumOpToDW2DConversion : public OpRewritePattern<linalg::PoolingNchwSumOp> {
    using OpRewritePattern<linalg::PoolingNchwSumOp>::OpRewritePattern;

    PoolingNchwSumOpToDW2DConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::PoolingNchwSumOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNchwSumOp linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();

        // Only match local pooling (kernel != whole frame)
        if (kernelShape.size() != 2 || (kernelShape[0] == in_s[2] && kernelShape[1] == in_s[3])) {
            return rewriter.notifyMatchFailure(linalgOp, "Only local pooling supported");
        }

        // Only bf16 for now
        if (!outType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(linalgOp, "Only bf16 supported");
        }

        PaddingInfo padInfo = foldBackwardPadding(input0, rewriter, true);

        // Look for downstream elementwise mulf/divf generic that applies a scale.
        auto [scaleValue, fusedGenericOp] = tryExtractFusedScale(output);
        if (!fusedGenericOp) {
            return rewriter.notifyMatchFailure(linalgOp, "No downstream scale generic found");
        }

        // Verify scale roughly matches 1/(KH*KW).  Use a loose tolerance because
        // the constant may be in BF16 (epsilon ~ 7.8e-3).
        float expectedScale = 1.0f / static_cast<float>(kernelShape[0] * kernelShape[1]);
        if (std::abs(scaleValue - expectedScale) > 1e-2f) {
            return rewriter.notifyMatchFailure(linalgOp, "Scale does not match 1/(KH*KW)");
        }

        if (_markFuseGroups) {
            Value fuseOutput = fusedGenericOp ? fusedGenericOp.getResult(0) : output;
            markFuseGroupBackward(
                fuseOutput, {input0, kernel}, rewriter,
                linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        int32_t channelCount = static_cast<int32_t>(in_s[1]);

        // Create depthwise conv2d weights [C, 1, KH, KW] filled with scaleValue.
        auto bf16Type = rewriter.getBF16Type();
        auto weightType =
            RankedTensorType::get({channelCount, 1, kernelShape[0], kernelShape[1]}, bf16Type);
        auto weightAttr =
            DenseElementsAttr::get(weightType, rewriter.getFloatAttr(bf16Type, scaleValue));
        auto weights = arith::ConstantOp::create(rewriter, loc, weightType, weightAttr);

        std::vector<float> biasData(channelCount, 0.0f);
        auto biasScale = createConst(biasData, rewriter, loc);

        auto initTensor = createInitTensor(linalgOp, rewriter, outType);

        auto strides = attrValuesAsVec(linalgOp.getStrides());
        auto dilations = attrValuesAsVec(linalgOp.getDilations());

        auto dwOp = torq_hl::DepthwiseConv2DOp::create(
            rewriter, loc, outType, initTensor,
            /*input_zp=*/0, /*weight_zp=*/0, /*output_zp=*/0,
            /*output_min=*/0xff800000, /*output_max=*/0x7f800000, /*shift_factor=*/0,
            /*groups=*/channelCount, padInfo.lrtbPad, strides, dilations,
            torq_hl::VectorizationModeEnum::None, weights, biasScale, input0,
            /*isDw1dStride1=*/false, /*segment_output=*/false, /*nhwc_input=*/false
        );

        rewriter.replaceOp(fusedGenericOp, dwOp.getOutput());
        rewriter.eraseOp(linalgOp);
        return success();
    }

  private:
    const bool _markFuseGroups;
};

void populateLinalgToTorqHLPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<PoolingNhwcMaxOpConversion>(context, markFuseGroups);
    patterns.insert<PoolingNchwMaxOpConversion>(context, markFuseGroups);
    patterns.insert<PoolingNhwcSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
    patterns.insert<PoolingNchwSumOpToDW2DConversion>(context, markFuseGroups);
    patterns.insert<PoolingNchwSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
}

} // namespace mlir::syna::torq
