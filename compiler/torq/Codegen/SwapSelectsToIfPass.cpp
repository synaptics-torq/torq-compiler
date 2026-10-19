// Copyright 2025 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "torq-swap-selects-to-if"

namespace mlir::syna::torq {

namespace {

// select(c, a, b) -> store M[I] paired with select(c, b, a) -> store M[J],
// where a was loaded from M[I] and b from M[J] with no writes to M in
// between, is an in-place conditional swap:
//
//     scf.if (!c) { store b -> M[I]; store a -> M[J] }
//
// The branchless form writes both locations on every iteration and forces
// the condition producer (e.g. a soft-float compare libcall) to stay live
// across both stores; on soft-float CSS targets LLVM even re-issues the
// compare. The LinalgExt sort sweep became select-based upstream in 3.12,
// which roughly doubled the executed instructions of bf16 sort kernels.
struct SwapSelectStoresToIf : OpRewritePattern<memref::StoreOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::StoreOp store, PatternRewriter &rewriter) const override {
        if (store.getNontemporal()) {
            return failure();
        }
        auto sel = store.getValue().getDefiningOp<arith::SelectOp>();
        if (!sel || !sel.getCondition().getType().isInteger(1)) {
            return failure();
        }
        Value a = sel.getTrueValue();
        Value b = sel.getFalseValue();

        // a must come straight from the store's own location.
        auto aLoad = a.getDefiningOp<memref::LoadOp>();
        if (!aLoad || aLoad.getMemref() != store.getMemref() ||
            aLoad.getIndices() != store.getIndices()) {
            return failure();
        }

        // Find the sibling store of the swapped select pair.
        for (Operation *user : b.getUsers()) {
            auto sel2 = dyn_cast<arith::SelectOp>(user);
            if (!sel2 || sel2.getCondition() != sel.getCondition() || sel2.getTrueValue() != b ||
                sel2.getFalseValue() != a) {
                continue;
            }
            for (Operation *sel2User : sel2->getUsers()) {
                auto otherStore = dyn_cast<memref::StoreOp>(sel2User);
                if (!otherStore || otherStore == store || otherStore.getNontemporal() ||
                    otherStore->getBlock() != store->getBlock()) {
                    continue;
                }
                // b must come straight from the sibling store's location.
                auto bLoad = b.getDefiningOp<memref::LoadOp>();
                if (!bLoad || bLoad.getMemref() != otherStore.getMemref() ||
                    bLoad.getIndices() != otherStore.getIndices()) {
                    continue;
                }
                // No writes to either memref between the loads and the stores
                // (the two stores of the motif itself excluded).
                if (isClobbered(aLoad, store, store, otherStore) ||
                    isClobbered(bLoad, otherStore, store, otherStore)) {
                    continue;
                }
                return rewritePair(store, otherStore, sel, a, b, rewriter);
            }
        }
        return failure();
    }

  private:
    static bool isClobbered(
        memref::LoadOp load, Operation *stop, memref::StoreOp ignore1, memref::StoreOp ignore2
    ) {
        for (Operation *op = load->getNextNode(); op && op != stop; op = op->getNextNode()) {
            if (op == ignore1 || op == ignore2) {
                continue;
            }
            auto other = dyn_cast<memref::StoreOp>(op);
            if (other && other.getMemref() == load.getMemref()) {
                return true;
            }
        }
        return false;
    }

    LogicalResult rewritePair(
        memref::StoreOp store, memref::StoreOp otherStore, arith::SelectOp sel, Value a, Value b,
        PatternRewriter &rewriter
    ) const {
        Location loc = store.getLoc();
        Operation *insertPt =
            store->isBeforeInBlock(otherStore) ? (Operation *)otherStore : (Operation *)store;
        rewriter.setInsertionPoint(insertPt);
        Value trueVal = arith::ConstantIntOp::create(rewriter, loc, 1, 1);
        Value notC = arith::XOrIOp::create(rewriter, loc, sel.getCondition(), trueVal);
        auto ifOp = scf::IfOp::create(rewriter, loc, notC, /*withElseRegion=*/false);
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(ifOp.thenBlock());
            memref::StoreOp::create(rewriter, loc, b, store.getMemref(), store.getIndices());
            memref::StoreOp::create(
                rewriter, loc, a, otherStore.getMemref(), otherStore.getIndices()
            );
        }
        rewriter.eraseOp(store);
        rewriter.eraseOp(otherStore);
        return success();
    }
};

class SwapSelectsToIfPass : public impl::SwapSelectsToIfBase<SwapSelectsToIfPass> {
  public:
    using SwapSelectsToIfBase::SwapSelectsToIfBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<SwapSelectStoresToIf>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSwapSelectsToIfPass() {
    return std::make_unique<SwapSelectsToIfPass>();
}

} // namespace mlir::syna::torq
