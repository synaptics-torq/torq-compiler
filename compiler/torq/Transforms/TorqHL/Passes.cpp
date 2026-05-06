// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::syna::torq_hl {

#define DEBUG_TYPE "torq-hl-op-transform"
class TorqHlOpTransformPass : public impl::TorqHlOpTransformBase<TorqHlOpTransformPass> {
  public:
    using TorqHlOpTransformBase::TorqHlOpTransformBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        RewritePatternSet patterns(ctx);
        populateTorqHLConv2DBigStridePatterns(ctx, patterns, false);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHlOpTransformPass() {
    return std::make_unique<TorqHlOpTransformPass>();
}

//===---------------------------------------------------------------------===//
// TorqHLFoldTableConstantPass
//===---------------------------------------------------------------------===//

// Folds torq_hl.table ops whose dynamic_table is a compile-time constant
// (torq_hl.const loaded via torq_hl.load) into a static_table attribute.
// TableOps and LoadOps for the same buffer are sorted in dominance order and
// walked together. A const-backed LoadOp opens a group that records subsequent
// TableOps until the next LoadOp. Each TableOp in an active group is folded
// with the current constant. A LoadOp is erased only once all TableOps in its
// group are folded.
class TorqHLFoldTableConstantPass
    : public impl::TorqHLFoldTableConstantBase<TorqHLFoldTableConstantPass> {
  public:
    using TorqHLFoldTableConstantBase::TorqHLFoldTableConstantBase;

    void runOnOperation() override {
        auto funcOp = getOperation();

        // Collect unfolded TableOps keyed by their dynamic_table buffer.
        DenseMap<Value, SmallVector<Operation *>> opsByDynTable;
        funcOp->walk([&](torq_hl::TableOp op) {
            if (op.getStaticTable())
                return;
            if (Value dynTable = op.getDynamicTable())
                opsByDynTable[dynTable].push_back(op.getOperation());
        });

        if (opsByDynTable.empty())
            return;

        // Append LoadOps that write into any tracked buffer.
        funcOp->walk([&](torq_hl::LoadOp op) {
            if (opsByDynTable.count(op.getOutput()))
                opsByDynTable[op.getOutput()].push_back(op.getOperation());
        });

        // Sort each combined list in dominance order.
        DominanceInfo &dom = getAnalysis<DominanceInfo>();
        for (auto &[dynTable, ops] : opsByDynTable)
            llvm::sort(ops, [&](Operation *a, Operation *b) {
                return dom.properlyDominates(a, b);
            });

        for (auto &[dynTable, ops] : opsByDynTable) {
            torq_hl::ConstOp activeConst;
            SmallVector<std::pair<torq_hl::LoadOp, SmallVector<torq_hl::TableOp>>> groups;

            for (Operation *op : ops) {
                if (auto loadOp = dyn_cast<torq_hl::LoadOp>(op)) {
                    activeConst = nullptr;
                    auto constOp = loadOp.getInput().getDefiningOp<torq_hl::ConstOp>();
                    if (constOp && dyn_cast<DenseElementsAttr>(constOp.getValue())) {
                        activeConst = constOp;
                        groups.push_back({loadOp, {}});
                    }
                }
                else if (auto tableOp = dyn_cast<torq_hl::TableOp>(op)) {
                    if (!activeConst)
                        continue;
                    groups.back().second.push_back(tableOp);
                    auto values = llvm::to_vector(
                        cast<DenseElementsAttr>(activeConst.getValue()).getValues<int32_t>()
                    );
                    tableOp.setStaticTableAttr(DenseI32ArrayAttr::get(tableOp.getContext(), values)
                    );
                    tableOp.getDynamicTableMutable().clear();
                }
            }

            // Erase each LoadOp (and its ConstOp source) if all its TableOps were folded.
            // Reverse order avoids use-after-erase when ConstOps are shared.
            for (auto &[loadOp, coveredTables] : llvm::reverse(groups)) {
                if (!llvm::all_of(coveredTables, [](torq_hl::TableOp t) {
                        return bool(t.getStaticTable());
                    }))
                    continue;
                auto constOp = loadOp.getInput().getDefiningOp<torq_hl::ConstOp>();
                loadOp.erase();
                if (constOp && constOp->use_empty())
                    constOp.erase();
            }
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLFoldTableConstantPass() {
    return std::make_unique<TorqHLFoldTableConstantPass>();
}

//===---------------------------------------------------------------------===//
// Register TorqHL Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Transforms/TorqHL/Passes.h.inc"
} // namespace

void registerTorqHLPasses() {
    // Generated.
    registerPasses();
}

} // namespace mlir::syna::torq_hl
