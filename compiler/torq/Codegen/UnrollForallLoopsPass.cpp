// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-unroll-forall-loops"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// This pattern unrolls a scf.forall operation by replicating the loop body n-1 times.
// The forall loop is used to parallelize the execution of an operation on multiple HW slices
// so its body will be terminated by single Wait and a single Store operation, in order.
// Since by definition all the iterations are independent, it is safe to move all the Wait and Store
// operations to the end of the unrolled loop to parallelize execution without further analysis.
// Code derived from generateUnrolledLoop in mlir/lib/Dialect/SCF/Utils/Utils.cpp
class ForallOpPattern : public OpRewritePattern<scf::ForallOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(scf::ForallOp forallOp, PatternRewriter &rewriter) const override {

        assert(
            forallOp.getStaticLowerBound().size() == 1 && forallOp.getDynamicLowerBound().empty() &&
            "expected single static lower bound in forall op"
        );
        assert(
            forallOp.getStaticUpperBound().size() == 1 && forallOp.getDynamicUpperBound().empty() &&
            "expected single static upper bound in forall op"
        );
        assert(
            forallOp.getStaticStep().size() == 1 && forallOp.getDynamicStep().empty() &&
            "expected single static step size in forall op"
        );

        int64_t lb = forallOp.getStaticLowerBound()[0];
        int64_t step = forallOp.getStaticStep()[0];

        std::optional<APInt> unrollFactor = mlir::constantTripCount(
            forallOp.getLowerBound(rewriter)[0], forallOp.getUpperBound(rewriter)[0],
            forallOp.getStep(rewriter)[0], /*isSigned=*/true, scf::computeUbMinusLb
        );

        assert(unrollFactor && "trip count is not a constant");
        uint64_t unrollCount = unrollFactor->getZExtValue();

        // track the original operations in the body loop
        SmallVector<Operation *> originalOps;
        for (auto &op : forallOp.getBody()->getOperations()) {
            originalOps.push_back(&op);
        }

        // create an operand map for each unrolled iteration
        SmallVector<IRMapping> operandMaps(unrollCount - 1);

        // insert in the operand map the induction variable
        if (!forallOp.getInductionVar(0).use_empty()) {
            for (unsigned i = 1; i < unrollCount; i++) {
                Value ivUnroll = arith::ConstantOp::create(
                    rewriter, forallOp.getLoc(), rewriter.getIndexAttr(lb + i * step)
                );
                operandMaps[i - 1].map(forallOp.getInductionVar(0), ivUnroll);
            }
        }

        SmallVector<Operation *> operationsToClone;

        // Scan the original operations and accumulate them until the block
        // terminator. When the terminator is reached, or for NSS when the
        // WaitProgramOp is reached, clone the collected ops for the remaining
        // unrolled iterations.
        {
            PatternRewriter::InsertionGuard insertionGuard(rewriter);
            for (auto op : originalOps) {
                bool cssWait = false;
                if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
                    auto invocationType =
                        cast<torq_hl::InvocationType>(waitOp.getInvocation().getType());
                    cssWait = invocationType.getExecutor() == torq_hl::Executor::CSS;
                }

                if ((isa<torq_hl::WaitProgramOp>(op) && !cssWait) ||
                    op->getBlock()->getTerminator() == op) {
                    rewriter.setInsertionPoint(op);
                    for (unsigned i = 1; i < unrollCount; i++) {
                        // Clone the original loop body operations
                        for (auto prevOp : operationsToClone) {
                            rewriter.clone(*prevOp, operandMaps[i - 1]);
                        }
                    }
                    operationsToClone.clear();
                }
                operationsToClone.push_back(op);
            }
        } // destroy insertionGuard

        // Loop now has a single iteration so we can remove it
        forallOp.setStaticUpperBound(lb + step);
        if (failed(forallOp.promoteIfSingleIteration(rewriter))) {
            assert(false && "Unrolling of scf.forall failed");
        }

        return success();
    }
};

static LogicalResult unrollForallLoops(Operation *op) {

    RewritePatternSet unrollPatterns(op->getContext());

    unrollPatterns.add<ForallOpPattern>(op->getContext());

    return applyPatternsGreedily(op, std::move(unrollPatterns));
}

class UnrollForallLoopsPass : public impl::UnrollForallLoopsBase<UnrollForallLoopsPass> {
  public:
    UnrollForallLoopsPass() = default;
    UnrollForallLoopsPass(const UnrollForallLoopsPass &pass) {}

    void runOnOperation() override;
};

void UnrollForallLoopsPass::runOnOperation() {
    if (failed(unrollForallLoops(getOperation()))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollForallLoopsPass() {
    return std::make_unique<UnrollForallLoopsPass>();
}

} // namespace mlir::syna::torq
