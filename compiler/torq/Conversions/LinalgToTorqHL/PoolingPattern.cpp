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
        auto torqOp = rewriter.create<TorqOp>(
            linalgOp.getLoc(), torqOutType, createInitTensor(linalgOp, rewriter, torqOutType),
            scInfo.zp, scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, weights, biasScale,
            input0
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
    patterns.insert<PoolingNhwcSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
}

} // namespace mlir::syna::torq
