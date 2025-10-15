// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-add-deallocation"

namespace mlir::syna::torq {

namespace {

class AddDeallocationPass : public AddDeallocationBase<AddDeallocationPass> {
  public:
    using AddDeallocationBase<AddDeallocationPass>::AddDeallocationBase;

    void runOnOperation() override;
};

void AddDeallocationPass::runOnOperation() {
    auto funcOp = getOperation();

    // FIXME: this could be implemented using BufferViewFlowAnalysis

    // last user of each allocation
    DenseMap<Value, Operation *> lastAllocationUser;

    // values allocated by alloc op in allocation order
    SetVector<Value> allocations;

    // the allocated memref corresponding to any derived memref
    // (the allocated memref is an alias of itself)
    DenseMap<Value, Value> aliases;

    // list start operation for each invocation
    DenseMap<Value, torq_hl::StartProgramOp> invocationStart;

    // scope in which an allocation is created
    DenseMap<Value, Operation *> scope;

    // walk the IR in order of execution and mark the last user of each allocation
    auto result = getOperation().walk([&](Operation *op) {
        LLVM_DEBUG({
            llvm::dbgs() << "Processing op:\n";
            op->dump();
        });

        // skip all operations within programs (notably slice programs)
        if (op->getParentOfType<torq_hl::ProgramOp>()) {
            return WalkResult::skip();
        }

        auto operands = op->getOperands();

        // check if the operations is creating an alias and track it accordingly
        if (isDerivedMemRefOperation(op)) {
            auto &baseMemRef = getDerivedMemRefBase(op);

            // the base memref may be an alias of an allocation
            auto baseMemRefAllocation = aliases.at(baseMemRef.get());

            // mark the output of the operation to be an alias of the original allocation
            aliases[op->getResult(0)] = baseMemRefAllocation;
        }
        else {
            // track all the memref results as aliases of itself
            for (auto result : op->getResults()) {
                if (isa<MemRefType>(result.getType())) {
                    aliases[result] = result;
                }
            }
        }

        // when processing a start op record the mapping startop/invocation
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
            invocationStart[startOp.getInvocation()] = startOp;
        }
        // when processing a wait op use the operands of the corresponding start
        else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
            operands = invocationStart.at(waitOp.getInvocation())->getOperands();
        }
        // if it is an allocation operation, track it as an alias of itself
        else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
            allocations.insert(allocOp.getResult());
            scope.try_emplace(allocOp.getResult(), op->getParentOp());
        }

        // mark the scope of all the allocations that the operation uses
        for (auto operand : operands) {

            LLVM_DEBUG({
                llvm::dbgs() << "considering operand:\n";
                operand.dump();
            });

            if (!isa<MemRefType>(operand.getType())) {
                continue;
            }

            auto baseMemref = aliases.at(operand);

            LLVM_DEBUG({
                llvm::dbgs() << "base memref:\n";
                baseMemref.dump();
            });

            // we can't handle values that escape the scope for the moment
            if (op->getBlock()->getTerminator() == op) {
                op->emitError("terminator op cannot be the last user of an allocation");
                return WalkResult::interrupt();
            }

            if (allocations.contains(baseMemref)) {

                auto scopeOp = scope.at(baseMemref);

                // since we want to always deallocate in the same scope
                // as the allocation we look for the parent of this operation
                // as the last user of the allocation
                Operation *lastUser = op;
                while (lastUser->getParentOp() != scopeOp) {
                    lastUser = lastUser->getParentOp();
                }

                lastAllocationUser[baseMemref] = lastUser;
            }
        }

        return WalkResult::advance();
    });

    // there was a problem during the analysis
    if (result.wasInterrupted()) {
        return signalPassFailure();
    }

    OpBuilder builder(funcOp);

    // go over all the allocations we may want to deallocate and insert
    // a deallocation after the last use
    for (auto alloc : allocations) {
        auto user = lastAllocationUser.at(alloc);
        builder.setInsertionPointAfter(user);
        builder.create<memref::DeallocOp>(user->getLoc(), alloc);
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAddDeallocationPass() {
    return std::make_unique<AddDeallocationPass>();
}

} // namespace mlir::syna::torq
