// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

class SpecializeTransposeOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp generic, PatternRewriter &rewriter) const override {
        // Only single input / single output generics
        if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1)
            return failure();

        Block &block = generic.getRegion().front();
        auto yieldOp = dyn_cast<linalg::YieldOp>(block.getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1)
            return failure();
        // The yielded value must be a block argument (i.e. an incoming element)
        auto bbArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
        if (!bbArg)
            return failure();
        // For a typical linalg.generic that computes a transpose the yielded
        // operand is the first input block argument (index 0).
        if (bbArg.getArgNumber() != 0)
            return failure();

        // FIXME: generalize this for non-identity output maps.
        // Ensure that the output indexing map is identity.
        if (!generic.getIndexingMapMatchingResult(generic->getResult(0)).isIdentity())
            return failure();

        auto inputIndexingMap = generic.getMatchingIndexingMap(&generic.getInputsMutable()[0]);
        if (!inputIndexingMap.isPermutation())
            return failure();

        SmallVector<int64_t> perm;
        for (auto dim : inputIndexingMap.getResults()) {
            if (auto dimExpr = dyn_cast<AffineDimExpr>(dim)) {
                perm.push_back(dimExpr.getPosition());
            }
            else {
                return failure();
            }
        }

        // Create a linalg.transpose replacing the generic.
        Value input = generic.getOperation()->getOperand(0);
        rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
            generic, input, generic.getDpsInits()[0], rewriter.getDenseI64ArrayAttr(perm)
        );
        return success();
    }
};

// Helper to register the pattern into a pattern list.
void populateSpecializeTransposeOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<SpecializeTransposeOpPattern>(ctx);
}

} // namespace mlir::syna::torq
