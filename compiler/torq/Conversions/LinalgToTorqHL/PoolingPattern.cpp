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

        auto kernels = mlir::cast<RankedTensorType>(srcOp.getInputs()[1].getType()).getShape();
        if (kernels.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingMaxOp"
            );
        }

        // Check stride constraints after determining if 1D pooling
        bool is1D = is1DPooling(kernels, attrStrides);
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
                output, {input}, rewriter,
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
            weightConst = rewriter.create<arith::ConstantOp>(loc, weightType, weightAttr);
        }
        else {
            outputMin = std::numeric_limits<int32_t>::min();
            outputMax = std::numeric_limits<int32_t>::max();
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }

        auto maxpoolOp = rewriter.create<torq_hl::MaxPool2dOp>(
            loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType), padInfo.padValue,
            outputMin, outputMax, attrStrides, padInfo.lrtbPad, kernels, weightConst,
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input,
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

void populateLinalgToTorqHLPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<PoolingNhwcMaxOpConversion>(context, markFuseGroups);
    patterns.insert<PoolingNchwMaxOpConversion>(context, markFuseGroups);
    patterns.insert<PoolingNhwcSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
}

} // namespace mlir::syna::torq
