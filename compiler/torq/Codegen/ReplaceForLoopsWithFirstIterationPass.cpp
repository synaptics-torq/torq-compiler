// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-replace-for-loops-with-first-iteration"

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

        if (*ub <= *lb + *step)
            return failure();

        Value newUb = mlir::getValueOrCreateConstantIndexOp(
            rewriter, forOp.getLoc(), rewriter.getIndexAttr(*lb + *step)
        );

        forOp.setUpperBound(newUb);

        return success();
    }
};

class ReplaceForLoopsWithFirstIterationPass
    : public impl::ReplaceForLoopsWithFirstIterationBase<ReplaceForLoopsWithFirstIterationPass> {
    using ReplaceForLoopsWithFirstIterationBase<
        ReplaceForLoopsWithFirstIterationPass>::ReplaceForLoopsWithFirstIterationBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<ReduceForOp>(patterns.getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createReplaceForLoopsWithFirstIterationPass() {
    return std::make_unique<ReplaceForLoopsWithFirstIterationPass>();
}

} // namespace mlir::syna::torq
