// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <cstring>

#define DEBUG_TYPE "torq-fuse-act-with-conv"

namespace mlir::syna::torq {

namespace {

static int32_t floatBits(const llvm::APFloat &apf) {
    float f = apf.convertToFloat();
    int32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// Fuses a "clamp" torq_hl.act into a preceding conv/depthwise_conv/fc op by
// transferring min_fp/max_fp as output_min/output_max on the producer, then
// removing the act op entirely.
template <typename ProducerOp>
class FuseClampActIntoProducer : public OpRewritePattern<torq_hl::ActOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::ActOp actOp, PatternRewriter &rewriter) const override {
        if (actOp.getName() != "clamp")
            return failure();

        auto producerOp = actOp.getInput().getDefiningOp<ProducerOp>();
        if (!producerOp)
            return failure();

        if (!producerOp.getOutput() || !producerOp.getOutput().hasOneUse())
            return failure();

        int32_t newOutMin = floatBits(actOp.getMinFp());
        int32_t newOutMax = floatBits(actOp.getMaxFp());

        bool alreadyFused =
            (newOutMin == producerOp.getOutputMin() && newOutMax == producerOp.getOutputMax());

        LLVM_DEBUG(
            llvm::dbgs() << "[FuseActWithConv] fusing clamp into "
                         << producerOp->getName().getStringRef() << " ["
                         << producerOp.getOutputMin() << ", " << producerOp.getOutputMax()
                         << "] -> [" << newOutMin << ", " << newOutMax << "]"
                         << (alreadyFused ? " (already fused, removing redundant act)" : "") << "\n"
        );

        if (!alreadyFused) {
            rewriter.modifyOpInPlace(producerOp, [&] {
                producerOp.setOutputMin(newOutMin);
                producerOp.setOutputMax(newOutMax);
            });
        }
        rewriter.replaceOp(actOp, producerOp.getOutput());
        return success();
    }
};

class FuseActWithConvPass : public FuseActWithConvBase<FuseActWithConvPass> {
  public:
    using FuseActWithConvBase<FuseActWithConvPass>::FuseActWithConvBase;

    void runOnOperation() override {
        MLIRContext *ctx = getOperation().getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<FuseClampActIntoProducer<torq_hl::Conv2DOp>>(ctx);
        patterns.add<FuseClampActIntoProducer<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<FuseClampActIntoProducer<torq_hl::FullyConnectedOp>>(ctx);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFuseActWithConvPass() {
    return std::make_unique<FuseActWithConvPass>();
}

} // namespace mlir::syna::torq
