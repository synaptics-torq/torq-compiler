// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-assign-objects-identifiers"

namespace mlir::syna::torq {

namespace {

class AssignObjectsIdentifiersPass
    : public AssignObjectsIdentifiersBase<AssignObjectsIdentifiersPass> {
  public:
    AssignObjectsIdentifiersPass() = default;
    AssignObjectsIdentifiersPass(const AssignObjectsIdentifiersPass &pass) {}

    void runOnOperation() override;
};

void AssignObjectsIdentifiersPass::runOnOperation() {
    auto funcOp = getOperation();

    OpBuilder builder(funcOp.getContext());

    // Assign job id to each NSS invocation that will be used for torq-rt interoperability
    auto jobId = 0;
    for (auto op : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {
        if (op.getInvocation().getType().getExecutor() != torq_hl::Executor::NSS) {
            continue;
        }

        op->setAttr("torq-job-id", builder.getI32IntegerAttr(jobId));

        jobId++;
    }

    // Assign allocation id to each new memref
    // this will be used by the runtime to track allocations
    auto allocationId = 0;
    for (auto &op : funcOp.getFunctionBody().getOps()) {

        SmallVector<int64_t> allocationIds;

        for (auto result : op.getResults()) {
            if (auto memrefType = dyn_cast<MemRefType>(result.getType())) {
                allocationIds.push_back(allocationId);
                allocationId++;
            }
        }

        if (!allocationIds.empty()) {
            op.setAttr("torq-buffer-ids", builder.getDenseI64ArrayAttr(allocationIds));
        }
    }

    // Assign action numbers to all the runtime actions
    auto actionId = 0;
    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (isDerivedMemRefOperation(&op)) {
            continue;
        }
        if (isa<torq_hl::ProgramOp, torq_hl::CreateInvocationOp, torq_hl::DescriptorOp,
                torq_hl::ConstOp, torq_hl::MapBindingOp, func::ReturnOp, torq_hl::ImportProgramOp,
                torq_hl::GetBlockOp, torq_hw::DispatchProfilingOp, arith::ConstantOp,
                bufferization::ToMemrefOp>(op)) {
            continue; // skip these ops
        }

        op.setAttr("torq-action-id", builder.getI32IntegerAttr(actionId));

        actionId++;
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignObjectsIdentifiersPass() {
    return std::make_unique<AssignObjectsIdentifiersPass>();
}

} // namespace mlir::syna::torq
