// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-valid-pad"

namespace mlir::syna::torq {

namespace {

class ConvertConvValidPadToSamePadPattern : public OpRewritePattern<torq_hl::Conv2DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::Conv2DOp conv2dOp, PatternRewriter &rewriter) const override {

        auto inputType = llvm::cast<RankedTensorType>(conv2dOp.getInput().getType());
        if (!inputType)
            return failure();

        auto input_shape = inputType.getShape();
        if (input_shape.size() != 4)
            return failure();

        auto weight_type = llvm::cast<RankedTensorType>(conv2dOp.getWeights().getType());
        if (!weight_type)
            return failure();

        auto weight_shape = weight_type.getShape();
        if (weight_shape.size() != 4)
            return failure();

        const uint32_t ksize_x = weight_shape[3];
        const uint32_t ksize_y = weight_shape[2];

        int32_t stride = conv2dOp.getStride()[0];

        int32_t pad_left = conv2dOp.getPad()[0];
        int32_t pad_right = conv2dOp.getPad()[1];
        int32_t pad_top = conv2dOp.getPad()[2];
        int32_t pad_bottom = conv2dOp.getPad()[3];

        if (!(pad_top == 0 && pad_left == 0 && pad_right == 0 && pad_bottom == 0 && ksize_x > 1 &&
              ksize_y > 1))
            return failure();

        int64_t total_pad_h = 0, total_pad_w = 0;
        if (stride == 1) {
            // For stride = 1, SAME padding requires total padding = kernel_size - 1
            // This ensures output size remains the same as input size
            total_pad_h = ksize_y - 1;
            total_pad_w = ksize_x - 1;
        }
        else if (stride == 2) {
            // Total padding = (output_size - 1) * stride + kernel_size - input_size
            // This is the formula for SAME padding with stride > 1
            int64_t output_h = (input_shape[2] + 1) / 2;
            int64_t output_w = (input_shape[3] + 1) / 2;
            total_pad_h = std::max((output_h - 1) * 2 + ksize_y - input_shape[2], int64_t(0));
            total_pad_w = std::max((output_w - 1) * 2 + ksize_x - input_shape[3], int64_t(0));
        }
        else {
            return failure();
        }

        if (total_pad_h == 0 && total_pad_w == 0)
            return failure();
        // Split the total padding into top/bottom and left/right
        // Usually SAME padding is split symmetrically, but if the total is odd,
        // we assign the extra padding to the bottom and right sides (TensorFlow-style)
        int64_t new_pad_top = total_pad_h / 2;
        int64_t new_pad_bottom = total_pad_h - new_pad_top;
        int64_t new_pad_left = total_pad_w / 2;
        int64_t new_pad_right = total_pad_w - new_pad_left;

        auto output_type = llvm::dyn_cast<RankedTensorType>(conv2dOp.getInit().getType());
        auto output_shape = output_type.getShape();
        SmallVector<int64_t, 4> same_pad_conv_output_shape = {
            output_shape[0], output_shape[1], output_shape[2], output_shape[3]
        };

        same_pad_conv_output_shape[2] += (new_pad_top + new_pad_bottom);
        same_pad_conv_output_shape[3] += (new_pad_left + new_pad_right);

        auto new_output_type =
            RankedTensorType::get(same_pad_conv_output_shape, output_type.getElementType());

        // Create new SAME pad attribute
        SmallVector<int64_t, 4> newPads{new_pad_left, new_pad_right, new_pad_top, new_pad_bottom};
        auto newPadAttr = rewriter.getDenseI64ArrayAttr(newPads);

        rewriter.setInsertionPoint(conv2dOp);
        auto loc = conv2dOp.getLoc();

        auto ConvInit = rewriter.create<tensor::EmptyOp>(
            conv2dOp.getLoc(), new_output_type.getShape(), new_output_type.getElementType()
        );

        auto samepadConv = rewriter.create<torq_hl::Conv2DOp>(
            loc, new_output_type, ConvInit.getResult(), conv2dOp.getInputZp(),
            conv2dOp.getWeightZp(), conv2dOp.getOutputZp(), conv2dOp.getOutputMin(),
            conv2dOp.getOutputMax(), conv2dOp.getShiftFactor(), conv2dOp.getGroups(), newPadAttr,
            conv2dOp.getStride(), conv2dOp.getDilation(), conv2dOp.getVectorizationMode(),
            conv2dOp.getWeights(), conv2dOp.getScaleBias(), conv2dOp.getInput()
        );

        auto offsets = createVector({0, 0, new_pad_top, new_pad_left}, rewriter);
        auto sizes = createVector(
            {output_shape[0], output_shape[1], output_shape[2], output_shape[3]}, rewriter
        );
        auto slice_strides = createVector({1, 1, 1, 1}, rewriter);
        auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            loc, samepadConv.getOutput(), offsets, sizes, slice_strides
        );

        rewriter.replaceOp(conv2dOp, extractSliceOp.getResult());
        return success();
    }
};

class ValidToSamePadPass : public ValidToSamePadPassBase<ValidToSamePadPass> {
  public:
    using ValidToSamePadPassBase<ValidToSamePadPass>::ValidToSamePadPassBase;
    void runOnOperation() override;
};

void ValidToSamePadPass::runOnOperation() {
    MLIRContext *ctx = getOperation().getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertConvValidPadToSamePadPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createValidToSamePadPass() {
    return std::make_unique<ValidToSamePadPass>();
}

} // namespace mlir::syna::torq
