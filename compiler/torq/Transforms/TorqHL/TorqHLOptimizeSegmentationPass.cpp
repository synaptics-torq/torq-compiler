// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-torqhl-optimize-seg"

namespace mlir::syna::torq_hl {

template <typename T> void setSegmentAttr(T op) {
    op->setAttr(op.getSegmentOutputAttrName(), BoolAttr::get(op.getContext(), true));
}

template <typename T>
static void
modifyForSegmentedOutput(T op, torq_hl::SegmentationOp segOp, PatternRewriter &rewriter) {
    setSegmentAttr(op);
    PatternRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op.getOperation());
    auto outputType = llvm::cast<RankedTensorType>(segOp.getOutput().getType());
    auto newInit = tensor::EmptyOp::create(
        rewriter, op->getLoc(), outputType.getShape(), outputType.getElementType()
    );
    rewriter.modifyOpInPlace(op, [&]() {
        op->setOperand(0, newInit.getResult());
        op->getResult(0).setType(outputType);
    });
}

// Merges duplicate SegmentationOps that share the same input value and segmentation parameters.
// When a value is consumed by two or more SegmentationOps with identical (hSegments, wSegments),
// all but the first are replaced with the first, avoiding redundant segmentation work.
class DeduplicateSegmentationPattern : public OpRewritePattern<torq_hl::SegmentationOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::SegmentationOp segOp, PatternRewriter &rewriter) const override {
        Value input = segOp.getInput();
        int64_t hSeg = segOp.getHSegments();
        int64_t wSeg = segOp.getWSegments();

        // Walk all users of the same input looking for an equivalent SegmentationOp that was
        // already created (i.e. dominates segOp in the IR).
        for (Operation *user : input.getUsers()) {
            if (user == segOp)
                continue;
            auto otherSeg = dyn_cast<torq_hl::SegmentationOp>(user);
            if (!otherSeg)
                continue;
            if (otherSeg.getHSegments() != hSeg || otherSeg.getWSegments() != wSeg)
                continue;
            // Prefer the op that appears first in the IR so the replacement is valid.
            // dominates() requires both ops to be in the same region; fall back to a simple
            // block-offset check when they are in the same block.
            bool otherDominates = false;
            if (otherSeg->getBlock() == segOp->getBlock()) {
                otherDominates = otherSeg->isBeforeInBlock(segOp);
            }
            else {
                // Different blocks — skip; a more elaborate dominance check would be needed.
                continue;
            }
            if (!otherDominates)
                continue;

            // otherSeg dominates segOp and is equivalent — replace segOp with it.
            rewriter.replaceOp(segOp, otherSeg.getOutput());
            return success();
        }
        return failure();
    }
};

class SegmentOptimizePattern : public OpRewritePattern<torq_hl::SegmentationOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::SegmentationOp segOp, PatternRewriter &rewriter) const override {
        if (segOp.getHSegments() != 2 || segOp.getWSegments() != 2) {
            // Only 4-quandrants segmentation can be fused at the moment
            return failure();
        }
        auto op = segOp.getInput().getDefiningOp();
        if (!op) {
            return failure();
        }
        if (!op->getResult(0).hasOneUse()) {
            // No need to segment if the output is consumed by only one op
            return failure();
        }

        // If the defining op has any user that is NOT a SegmentationOp,
        // we cannot set segment_output because that user expects
        // non-segmented output.
        for (auto *user : op->getResult(0).getUsers()) {
            if (!isa<torq_hl::SegmentationOp>(user)) {
                return failure();
            }
        }

        bool segment_output =
            TypeSwitch<Operation *, bool>(op)
                .Case<torq_hl::Conv2DOp, torq_hl::DepthwiseConv2DOp, torq_hl::AddOp>([&](auto op) {
                    modifyForSegmentedOutput(op, segOp, rewriter);
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
    : public impl::TorqHLOptimizeSegmentationBase<TorqHLOptimizeSegmentationPass> {
  public:
    using TorqHLOptimizeSegmentationBase::TorqHLOptimizeSegmentationBase;
    void runOnOperation() override;
};

void TorqHLOptimizeSegmentationPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    // First deduplicate SegmentationOps that share the same input and parameters.
    patterns.add<DeduplicateSegmentationPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }

    // Then try to fuse segmentation with producer operations
    RewritePatternSet fusePatterns(ctx);
    fusePatterns.add<SegmentOptimizePattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(fusePatterns)))) {
        return signalPassFailure();
    }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLOptimizeSegmentationPass() {
    return std::make_unique<TorqHLOptimizeSegmentationPass>();
}

} // namespace mlir::syna::torq_hl
