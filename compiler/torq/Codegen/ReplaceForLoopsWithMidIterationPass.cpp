// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#include <cstdint>

#define DEBUG_TYPE "torq-replace-for-loops-with-mid-iteration"

namespace mlir::syna::torq {

namespace {

struct ReduceForOp : public OpRewritePattern<scf::ForOp> {
    using OpRewritePattern<scf::ForOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
        rewriter.setInsertionPoint(forOp);
        std::optional<int64_t> lb = getConstIntValue(forOp.getLowerBound());
        std::optional<int64_t> ub = getConstIntValue(forOp.getUpperBound());
        std::optional<int64_t> step = getConstIntValue(forOp.getStep());

        if (!lb || !ub || !step)
            return failure();

        int64_t tripCount = mlir::constantTripCount(
                                forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), true,
                                scf::computeUbMinusLb
        )
                                .value()
                                .getSExtValue();

        // For loops with 2 or fewer iterations, neither iteration is an interior middle tile;
        // both are border tiles with potentially asymmetric halo padding. Keeping both iterations
        // ensures fit_tile_to_memory evaluates both border halos and allocates a full-tensor
        // destination LRAM buffer, matching check_tiling_succeeded. For tripCount >= 3, reducing
        // to the middle tile (tripCount / 2) safely tests the maximum interior halo.
        if (tripCount <= 2)
            return failure();

        int64_t midPoint = *lb + ((tripCount / 2) * *step);

        Value newLb = mlir::getValueOrCreateConstantIndexOp(
            rewriter, forOp.getLoc(), rewriter.getIndexAttr(midPoint)
        );

        Value newUb = mlir::getValueOrCreateConstantIndexOp(
            rewriter, forOp.getLoc(), rewriter.getIndexAttr(midPoint + *step)
        );

        forOp.setLowerBound(newLb);
        forOp.setUpperBound(newUb);

        return success();
    }
};

class ReplaceForLoopsWithMidIterationPass
    : public impl::ReplaceForLoopsWithMidIterationBase<ReplaceForLoopsWithMidIterationPass> {
    using ReplaceForLoopsWithMidIterationBase<
        ReplaceForLoopsWithMidIterationPass>::ReplaceForLoopsWithMidIterationBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ReduceForOp>(patterns.getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createReplaceForLoopsWithMidIterationPass() {
    return std::make_unique<ReplaceForLoopsWithMidIterationPass>();
}

} // namespace mlir::syna::torq
