// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-torqhl-optimize-seg"

namespace mlir::syna::torq_hl {

template <typename T> void setSegmentAttr(T op) {
    op->setAttr(op.getSegmentOutputAttrName(), BoolAttr::get(op.getContext(), true));
}

// simple check if conv2d is tiled, we have to skip segmenting it if it need to tile
// as current conv2d xy-tiling segmented to wrong data
bool isConv2DTiled(torq_hl::Conv2DOp op) {
    auto datasize = getShapeTypeDataSize(op.getInit().getType()) +
                    getShapeTypeDataSize(op.getInput().getType()) +
                    getShapeTypeDataSize(op.getWeights().getType()) +
                    getShapeTypeDataSize(op.getScaleBias().getType());

    // TODO: get lram size from somewhere
    if (datasize > 1024 * 400) {
        return true;
    }
    return false;
}

class SegmentOptimizePattern : public OpRewritePattern<torq_hl::SegmentationOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::SegmentationOp segOp, PatternRewriter &rewriter) const override {
        auto op = segOp.getInput().getDefiningOp();

        if (!op)
            return failure();

        bool segment_output =
            TypeSwitch<Operation *, bool>(op)
                .Case<torq_hl::Conv2DOp, torq_hl::DepthwiseConv2DOp, torq_hl::AddOp>([&](auto op) {
                    if (isa<torq_hl::Conv2DOp>(op) && isConv2DTiled(cast<torq_hl::Conv2DOp>(op))) {
                        return false;
                    }

                    setSegmentAttr(op);
                    return true;
                })
                .Default([&](Operation *op) { return false; });

        if (segment_output) {
            segOp.getOutput().replaceAllUsesWith(segOp.getInput());
            rewriter.eraseOp(segOp);
            return success();
        }

        return failure();
    }
};

class TorqHLOptimizeSegmentationPass
    : public TorqHLOptimizeSegmentationBase<TorqHLOptimizeSegmentationPass> {
  public:
    using TorqHLOptimizeSegmentationBase::TorqHLOptimizeSegmentationBase;
    void runOnOperation() override;
};

void TorqHLOptimizeSegmentationPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<SegmentOptimizePattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLOptimizeSegmentationPass() {
    return std::make_unique<TorqHLOptimizeSegmentationPass>();
}

} // namespace mlir::syna::torq_hl
