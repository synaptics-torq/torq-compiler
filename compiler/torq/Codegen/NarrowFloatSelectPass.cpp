// Copyright 2025 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "torq-narrow-float-select"

namespace mlir::syna::torq {

namespace {

// select(c, extf(a), extf(b)) -> extf(select(c, a, b))
//
// extf is exact, so selecting in the narrow type is equivalent. Widened
// selects otherwise survive to LLVM on soft-float CSS targets, where each
// f32 select and the truncf that stores it back pull in soft-float libcalls
// and round-to-nearest-even expansions (e.g. the LinalgExt sort sweep became
// select-based upstream). Mirrors the NarrowExtremum folds MLIR has for
// maximumf/minimumf, which don't cover arith.select.
struct NarrowSelectOfExtF : OpRewritePattern<arith::SelectOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::SelectOp op, PatternRewriter &rewriter) const override {
        auto trueExt = op.getTrueValue().getDefiningOp<arith::ExtFOp>();
        auto falseExt = op.getFalseValue().getDefiningOp<arith::ExtFOp>();
        if (!trueExt || !falseExt || trueExt.getIn().getType() != falseExt.getIn().getType()) {
            return failure();
        }
        Value narrow = arith::SelectOp::create(
            rewriter, op.getLoc(), op.getCondition(), trueExt.getIn(), falseExt.getIn()
        );
        rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, op.getType(), narrow);
        return success();
    }
};

// truncf(extf(x)) -> x when x's type matches the truncf result type. The
// round trip through the wider type is exact, but MLIR's own fold requires
// signaling-NaN round-tripping, which bf16->f32 fails, so the pair would
// otherwise survive to ArithExpandOps and become a round-to-nearest-even
// expansion with a NaN check. Keeping the original bits is also the more
// faithful behavior for data-movement kernels (sort/select permutations).
struct FoldTruncFExtFRoundTrip : OpRewritePattern<arith::TruncFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::TruncFOp op, PatternRewriter &rewriter) const override {
        auto extOp = op.getOperand().getDefiningOp<arith::ExtFOp>();
        if (!extOp || extOp.getIn().getType() != op.getType()) {
            return failure();
        }
        rewriter.replaceOp(op, extOp.getIn());
        return success();
    }
};

class NarrowFloatSelectPass : public impl::NarrowFloatSelectBase<NarrowFloatSelectPass> {
  public:
    using NarrowFloatSelectBase::NarrowFloatSelectBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<NarrowSelectOfExtF, FoldTruncFExtFRoundTrip>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createNarrowFloatSelectPass() {
    return std::make_unique<NarrowFloatSelectPass>();
}

} // namespace mlir::syna::torq
