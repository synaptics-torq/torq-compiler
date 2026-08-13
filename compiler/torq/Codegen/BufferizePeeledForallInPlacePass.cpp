// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BufferizePeeledForallInPlacePass
// --------------------------------
// LinalgSlicing peels a slice-dim into a head (an scf.forall over the slices)
// plus a tail. After EliminateEmptyTensors the head looks like this:
//
//   %e   = tensor.empty()
//   %es  = tensor.extract_slice %e[R]
//   %f   = scf.forall shared_outs(%arg = %es) { ... parallel_insert covering R }
//   %ins = tensor.insert_slice %f into %dst[R]        // %dst = output load
//
// EliminateEmptyTensors points the insert at the output binding but leaves the
// forall init on the orphaned tensor.empty %e. The forall then bufferizes into a
// temp, and %ins copies that temp into %dst. That copy is an XRAM->LRAM->XRAM
// round-trip of the head region and doubles the DMA.
//
// This pass points the extract_slice at %dst instead of %e. The forall then
// bufferizes in place and the insert_slice folds to identity.
//
// Preconditions: %e is a tensor.empty, the extract feeds only this forall init,
// the extract and insert cover the same region R, and %dst dominates the extract.
// scf.forall writes disjoint regions per iteration.

#include "PassesDetail.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "torq-bufferize-peeled-forall-in-place"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

struct ForallOutputInPlace : OpRewritePattern<tensor::InsertSliceOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::InsertSliceOp ins, PatternRewriter &rewriter) const override {
        auto res = dyn_cast<OpResult>(ins.getSource());
        if (!res || !res.hasOneUse())
            return failure();
        auto forallOp = res.getDefiningOp<scf::ForallOp>();
        if (!forallOp)
            return failure();

        Value init = forallOp.getOutputs()[res.getResultNumber()];
        auto extractSlice = init.getDefiningOp<tensor::ExtractSliceOp>();
        if (!extractSlice || !extractSlice.getSource().getDefiningOp<tensor::EmptyOp>())
            return failure();

        // The extract must feed only this forall init, otherwise retargeting its
        // source would affect other readers.
        if (!extractSlice->hasOneUse())
            return failure();

        Value dst = ins.getDest();
        if (extractSlice.getSource() == dst)
            return failure(); // already in place

        // The extract and the insert must cover the same region.
        if (extractSlice.getMixedOffsets() != ins.getMixedOffsets() ||
            extractSlice.getMixedSizes() != ins.getMixedSizes() ||
            extractSlice.getMixedStrides() != ins.getMixedStrides())
            return failure();

        // %dst must dominate the extract.
        if (Operation *dstDef = dst.getDefiningOp())
            if (dstDef->getBlock() != extractSlice->getBlock() ||
                !dstDef->isBeforeInBlock(extractSlice))
                return failure();

        rewriter.modifyOpInPlace(extractSlice, [&] {
            extractSlice.getSourceMutable().assign(dst);
        });
        return success();
    }
};

struct BufferizePeeledForallInPlacePass
    : public impl::BufferizePeeledForallInPlaceBase<BufferizePeeledForallInPlacePass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ForallOutputInPlace>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createBufferizePeeledForallInPlacePass() {
    return std::make_unique<BufferizePeeledForallInPlacePass>();
}

} // namespace mlir::syna::torq
