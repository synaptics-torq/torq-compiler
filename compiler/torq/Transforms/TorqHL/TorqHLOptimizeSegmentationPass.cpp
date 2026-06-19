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
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
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
        // Currently only op with even W dimension can be fused.
        // H can be even or odd since this makes no difference in how segmenation is computed.
        auto inputType = llvm::cast<RankedTensorType>(segOp.getInput().getType());
        if (!inputType || inputType.getRank() != 4) {
            return failure();
        }
        if (inputType.getShape()[3] % 2) {
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

// Returns true if `v` is loop-invariant with respect to `forallOp`:
// i.e., it does not transitively depend on any induction variable or
// shared_outs block argument of the forall body.
// `visited` prevents re-visiting values in cyclic/diamond graphs.
static bool
isLoopInvariant(Value v, scf::ForallOp forallOp, llvm::SmallPtrSetImpl<Value> &visited) {
    if (!visited.insert(v).second)
        return true; // already confirmed invariant

    // Block arguments of the forall body (IVs + shared_outs) are loop-variant.
    if (auto ba = dyn_cast<BlockArgument>(v))
        return ba.getParentBlock() != forallOp.getBody();

    Operation *defOp = v.getDefiningOp();
    // Defined outside the forall — trivially invariant.
    if (!forallOp->isProperAncestor(defOp))
        return true;

    // Recursively check every operand of the defining op.
    return llvm::all_of(defOp->getOperands(), [&](Value operand) {
        return isLoopInvariant(operand, forallOp, visited);
    });
}

// Recursively collect all ops inside `forallOp` that must be moved out to
// make `v` available outside. Ops are inserted into `opsToMove` in
// topological order (deepest dependency first → correct move order).
static void
collectOpsToHoist(Value v, scf::ForallOp forallOp, llvm::SetVector<Operation *> &opsToMove) {
    if (isa<BlockArgument>(v))
        return;
    Operation *defOp = v.getDefiningOp();
    if (!forallOp->isProperAncestor(defOp))
        return; // already outside, nothing to collect
    if (opsToMove.contains(defOp))
        return; // already collected
    // Recurse on operands first (depth-first → topo order).
    for (Value operand : defOp->getOperands())
        collectOpsToHoist(operand, forallOp, opsToMove);
    opsToMove.insert(defOp);
}

// Hoists a SegmentationOp from inside a scf.forall to just before the forall.
//
// The pattern fires when all data operands (weights, scale_bias, input) are
// loop-invariant — they do not transitively depend on any induction variable
// or shared_outs arg of the forall. Any loop-invariant ops inside the forall
// that produce those values are also moved out first (mini-LICM).
// The init tensor is always recreated fresh outside the loop.
//
// Example:
//   BEFORE:
//     scf.forall (%iv = 0 to 8 step 4) {
//         %empty         = tensor.empty()
//         %filled        = torq_hl.fill(%empty)          // loop-invariant
//         %inserted      = tensor.insert_slice ... into %filled  // loop-invariant
//         %seg = torq_hl.segmentation(%inserted : 1x256x14x56)
//         conv2d(..., %seg)
//     }
//
//   AFTER:
//     %empty    = tensor.empty()                         // hoisted
//     %filled   = torq_hl.fill(%empty)                   // hoisted
//     %inserted = tensor.insert_slice ... into %filled   // hoisted
//     %seg      = torq_hl.segmentation(%inserted)        // hoisted
//     scf.forall (%iv = 0 to 8 step 4) {
//         conv2d(..., %seg)
//     }
class HoistSegmentationOutOfForallPattern : public OpRewritePattern<torq_hl::SegmentationOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::SegmentationOp segOp, PatternRewriter &rewriter) const override {
        // Only fire if the segmentation is inside a forall.
        auto forallOp = segOp->getParentOfType<scf::ForallOp>();
        if (!forallOp)
            return failure();

        // Check that weights, scale_bias and input are all loop-invariant.
        // The init operand is excluded — we always recreate it outside.
        auto checkInvariant = [&](Value v) -> bool {
            llvm::SmallPtrSet<Value, 16> visited;
            return isLoopInvariant(v, forallOp, visited);
        };

        if (!checkInvariant(segOp.getWeights()) || !checkInvariant(segOp.getScaleBias()) ||
            !checkInvariant(segOp.getInput()))
            return failure();

        // Collect all ops inside the forall that need to move out,
        // in topological order (dependencies before dependents).
        llvm::SetVector<Operation *> opsToMove;
        collectOpsToHoist(segOp.getWeights(), forallOp, opsToMove);
        collectOpsToHoist(segOp.getScaleBias(), forallOp, opsToMove);
        collectOpsToHoist(segOp.getInput(), forallOp, opsToMove);

        // Move the collected ops out, then the segmentation itself.
        // rewriter.moveOpBefore preserves operand dominance because we move
        // in topological order and insert just before the forall.
        for (Operation *op : opsToMove)
            rewriter.moveOpBefore(op, forallOp);

        // Recreate the init outside and build the hoisted segmentation.
        rewriter.setInsertionPoint(forallOp);
        Location loc = segOp->getLoc();
        auto outputType = llvm::cast<RankedTensorType>(segOp.getOutput().getType());
        Value newInit = tensor::EmptyOp::create(
            rewriter, loc, outputType.getShape(), outputType.getElementType()
        );
        auto hoistedSeg = torq_hl::SegmentationOp::create(
            rewriter, loc, outputType, newInit, segOp.getHSegmentsAttr(), segOp.getWSegmentsAttr(),
            segOp.getWeights(), segOp.getScaleBias(), segOp.getInput()
        );

        rewriter.replaceOp(segOp, hoistedSeg.getOutput());
        return success();
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

    // Finally, hoist any SegmentationOp inside a scf.forall to just before the
    // forall, provided its data operands (input, weights, scale_bias) are all
    // loop-invariant. This ensures the segmentation runs once rather than once
    // per iteration.
    RewritePatternSet hoistPatterns(ctx);
    hoistPatterns.add<HoistSegmentationOutOfForallPattern>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(hoistPatterns)))) {
        return signalPassFailure();
    }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLOptimizeSegmentationPass() {
    return std::make_unique<TorqHLOptimizeSegmentationPass>();
}

} // namespace mlir::syna::torq_hl
