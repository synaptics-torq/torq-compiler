// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-remove-isnan"

namespace mlir::syna::torq {

namespace {

struct WrappedSelectMatch {
    linalg::GenericOp selectGeneric;
    OpOperand *falseOperand;
};

SmallVector<WrappedSelectMatch> matchWrappedSelectOps(linalg::GenericOp isNanOp) {
    SmallVector<WrappedSelectMatch> matches;
    Value isNanResult = isNanOp.getResult(0);
    for (OpOperand &use : isNanResult.getUses()) {
        auto selectGeneric = dyn_cast<linalg::GenericOp>(use.getOwner());
        if (!selectGeneric)
            continue;
        // Ensure this particular use is a DPS input, not an init.
        if (!selectGeneric.isDpsInput(&use))
            continue;
        // Exactly three inputs (one init) and one result.
        if (selectGeneric.getNumDpsInputs() != 3 || selectGeneric.getNumDpsInits() != 1 ||
            selectGeneric.getNumResults() != 1)
            continue;
        if (selectGeneric.getNumParallelLoops() != selectGeneric.getNumLoops() ||
            !selectGeneric.getRegion().hasOneBlock())
            continue;

        Block *body = selectGeneric.getBody();
        if (body->getOperations().size() != 2)
            continue;

        auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1)
            continue;
        auto selectOp = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
        if (!selectOp)
            continue;

        // The particular use of the IsNaN result must feed the select condition.
        BlockArgument conditionArg = selectGeneric.getMatchingBlockArgument(&use);
        if (selectOp.getCondition() != conditionArg)
            continue;

        // The false branch must directly map to another generic input.
        auto falseArg = dyn_cast<BlockArgument>(selectOp.getFalseValue());
        if (!falseArg || falseArg.getOwner() != body)
            continue;
        OpOperand *falseOperand = selectGeneric.getMatchingOpOperand(falseArg);
        if (!falseOperand || !selectGeneric.isDpsInput(falseOperand))
            continue;

        // Forwarding is valid only when the false input and result use the same
        // indexing and have exactly the same tensor type.
        OpOperand *outputOperand = selectGeneric.getDpsInitOperand(0);
        if (selectGeneric.getMatchingIndexingMap(falseOperand) !=
            selectGeneric.getMatchingIndexingMap(outputOperand))
            continue;
        if (falseOperand->get().getType() != selectGeneric.getResult(0).getType())
            continue;

        matches.push_back(WrappedSelectMatch{selectGeneric, falseOperand});
    }
    return matches;
}

class RemoveIsNanPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp isNanOp, PatternRewriter &rewriter) const override {
        // Exactly one tensor input, init, and result.
        if (isNanOp.getNumDpsInputs() != 1 || isNanOp.getNumDpsInits() != 1 ||
            isNanOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(isNanOp, "expected one input, init, and result");
        }
        // Pointwise elementwise operation.
        if (isNanOp.getNumParallelLoops() != isNanOp.getNumLoops() ||
            !llvm::all_of(isNanOp.getIndexingMapsArray(), [](AffineMap map) {
                return map.isIdentity();
            })) {
            return rewriter.notifyMatchFailure(
                isNanOp, "expected all-parallel iterators and identity indexing maps"
            );
        }
        // Input is a floating tensor; result is the same shape with i1 elements.
        auto inputType = dyn_cast<RankedTensorType>(isNanOp.getInputs()[0].getType());
        auto resultType = dyn_cast<RankedTensorType>(isNanOp.getResult(0).getType());
        if (!inputType || !resultType || !isa<FloatType>(inputType.getElementType()) ||
            !resultType.getElementType().isInteger(1) ||
            inputType.getShape() != resultType.getShape()) {
            return rewriter.notifyMatchFailure(isNanOp, "unexpected input/output types");
        }
        // Must have only one block with exactly cmpf + linalg.yield.
        if (!isNanOp.getRegion().hasOneBlock()) {
            return rewriter.notifyMatchFailure(isNanOp, "expected single-block generic");
        }
        Block *body = isNanOp.getBody();
        if (body->getOperations().size() != 2) {
            return rewriter.notifyMatchFailure(isNanOp, "expected exactly 2 ops in generic");
        }
        auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1) {
            return rewriter.notifyMatchFailure(isNanOp, "expected one directly yielded value");
        }
        auto cmpOp = yieldOp.getOperand(0).getDefiningOp<arith::CmpFOp>();
        if (!cmpOp || cmpOp.getPredicate() != arith::CmpFPredicate::UNE) {
            return rewriter.notifyMatchFailure(isNanOp, "expected directly yielded arith.cmpf une");
        }
        // Ensure NaN check semantics: cmpf should compare its input against itself
        Value inputArg = body->getArgument(0);
        if (cmpOp.getLhs() != inputArg || cmpOp.getRhs() != inputArg) {
            return rewriter.notifyMatchFailure(
                isNanOp, "expected arith.cmpf to compare input against itself"
            );
        }

        for (auto &selectOpMatch : matchWrappedSelectOps(isNanOp)) {
            Value falseInput = selectOpMatch.falseOperand->get();
            rewriter.replaceOp(selectOpMatch.selectGeneric, falseInput);
        }
        // Erase the cmpf or replace with an all false constant in case it has other consumers
        if (isNanOp->use_empty()) {
            rewriter.eraseOp(isNanOp);
            LLVM_DEBUG(llvm::dbgs() << "[torq-remove-isnan] removed IsNaN check\n");
        }
        else {
            auto falseAttr = SplatElementsAttr::get(resultType, rewriter.getBoolAttr(false));
            rewriter.replaceOpWithNewOp<arith::ConstantOp>(isNanOp, falseAttr);
            LLVM_DEBUG(
                llvm::dbgs() << "[torq-remove-isnan] replaced IsNaN check with all false constant\n"
            );
        }
        return success();
    }
};

class RemoveIsNanPass : public impl::RemoveIsNanBase<RemoveIsNanPass> {
  public:
    using RemoveIsNanBase::RemoveIsNanBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<RemoveIsNanPattern>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRemoveIsNanPass() {
    return std::make_unique<RemoveIsNanPass>();
}

} // namespace mlir::syna::torq
