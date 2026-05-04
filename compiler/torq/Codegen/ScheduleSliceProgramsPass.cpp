// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-schedule-slice-tasks"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// assign an executor_id to each start_program operation
static LogicalResult
scheduleSliceTasks(Region &region, torq_hl::Executor executor, int executorCount) {

    SmallVector<bool> sliceBusy(executorCount, false);

    for (auto &op : region.getOps()) {

        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {

            auto invocationOp =
                startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

            if (!invocationOp) {
                return op.emitError() << "must use an invocation created by a create_invocation op";
            }

            if (!invocationOp.getExecutorId()) {

                // schedule the program on the first available slice

                auto it = llvm::find(sliceBusy, false);

                if (it == sliceBusy.end()) {
                    return op.emitError() << "all slices busy, cannot allocate a executor_id";
                }

                int availableSlice = std::distance(sliceBusy.begin(), it);

                invocationOp.setExecutorId(APInt(64, availableSlice));
                sliceBusy[availableSlice] = true;
            }
            else {

                // mark the executor being used as busy
                auto executorId = invocationOp.getExecutorId()->getZExtValue();

                if (sliceBusy[executorId]) {
                    return op.emitError() << "executor is already busy";
                }

                sliceBusy[executorId] = true;
            }
        }

        else if (auto sliceWaitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {

            auto invocationOp =
                sliceWaitOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

            if (!invocationOp) {
                return op.emitError() << "must use an invocation created by a create_invocation op";
            }

            sliceBusy[invocationOp.getExecutorId()->getZExtValue()] = false;
        }
    }

    return success();
}

class ScheduleSliceProgramsPass
    : public impl::ScheduleSliceProgramsBase<ScheduleSliceProgramsPass> {
  public:
    ScheduleSliceProgramsPass() = default;
    ScheduleSliceProgramsPass(const ScheduleSliceProgramsPass &pass) {}

    void runOnOperation() override;
};

void ScheduleSliceProgramsPass::runOnOperation() {
    if (failed(scheduleSliceTasks(
            getOperation().getFunctionBody(), torq_hl::Executor::Slice, HwInfo::slice_count
        ))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createScheduleSliceProgramsPass() {
    return std::make_unique<ScheduleSliceProgramsPass>();
}

} // namespace mlir::syna::torq
