// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-overlap-nss-code-loads"

namespace mlir::syna::torq {

namespace {

class OverlapNSSCodeLoadsPass : public impl::OverlapNSSCodeLoadsBase<OverlapNSSCodeLoadsPass> {
  public:
    using OverlapNSSCodeLoadsBase::OverlapNSSCodeLoadsBase;
    void runOnOperation() override;
};

// Moves the code loads of an NSS program block before the block's last wait, when that wait is
// the block's last start or wait. A code load fills the inactive code area, which nothing in the
// block uses, so it can run while the program the wait is for still runs. When the block ends
// with a program running (its last start or wait is a start), the loads already overlap it.
// Returns the number of loads moved.
static int overlapCodeLoads(Block &block) {
    Operation *lastStartOrWait = nullptr;
    SmallVector<torq_hl::LoadOp> codeLoads;
    for (auto &op : block) {
        if (isa<torq_hl::StartProgramOp, torq_hl::WaitProgramOp>(op)) {
            lastStartOrWait = &op;
        }
        else if (auto loadOp = dyn_cast<torq_hl::LoadOp>(op)) {
            if (loadOp.getInput().getDefiningOp<torq_hl::GetBlockOp>()) {
                codeLoads.push_back(loadOp);
            }
        }
    }

    auto waitOp = dyn_cast_if_present<torq_hl::WaitProgramOp>(lastStartOrWait);
    if (!waitOp) {
        return 0;
    }

    int moved = 0;
    for (auto loadOp : codeLoads) {
        if (!waitOp->isBeforeInBlock(loadOp)) {
            continue;
        }
        auto getBlockOp = loadOp.getInput().getDefiningOp<torq_hl::GetBlockOp>();
        if (getBlockOp->getBlock() == &block && waitOp->isBeforeInBlock(getBlockOp)) {
            getBlockOp->moveBefore(waitOp);
        }
        loadOp->moveBefore(waitOp);
        moved++;
    }
    return moved;
}

void OverlapNSSCodeLoadsPass::runOnOperation() {
    auto funcOp = getOperation();

    int moved = 0;
    for (auto programOp : funcOp.getFunctionBody().getOps<torq_hl::ProgramOp>()) {
        if (programOp.getProgram().getType().getExecutor() != torq_hl::Executor::NSS) {
            continue;
        }
        for (auto &block : programOp.getBody()) {
            moved += overlapCodeLoads(block);
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "torq-overlap-nss-code-loads: moved " << moved << " loads\n");
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOverlapNSSCodeLoadsPass() {
    return std::make_unique<OverlapNSSCodeLoadsPass>();
}

} // namespace mlir::syna::torq
