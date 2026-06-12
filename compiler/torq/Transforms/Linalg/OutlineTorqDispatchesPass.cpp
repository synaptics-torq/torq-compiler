// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "outline-torq-dispatches"

namespace mlir::syna::torq {

using namespace mlir::iree_compiler;

namespace {

static DenseSet<Operation *> getTorqOperations(FunctionOpInterface funcOp) {
    // Find all operations that have not been transferred to local (excluding transfer ops
    // themselves)

    assert(
        funcOp.getCallableRegion()->hasOneBlock() &&
        "Expected function body to have at most one block"
    );

    DenseSet<Operation *> torqOps;

    for (Operation &op : funcOp.getCallableRegion()->front()) {

        // never outline these operations
        if (isa<IREE::Flow::TensorTransferOp, func::ReturnOp, IREE::Util::ReturnOp,
                IREE::HAL::TensorImportOp, arith::ConstantOp>(op)) {
            continue;
        }

        // never outline operations that do not produce tensors, a flow region cannot return them
        if (llvm::any_of(op.getResults(), [](Value v) { return !isa<TensorType>(v.getType()); })) {
            continue;
        }

        // scan the operands, if any of them have been transferred to local, then this is not a
        // torq operation, otherwise it is a torq operation
        bool isTorqOp = true;

        for (auto operand : op.getOperands()) {
            if (auto transferOp = operand.getDefiningOp<IREE::Flow::TensorTransferOp>()) {
                if (auto target = dyn_cast<IREE::HAL::DevicePromiseAttr>(transferOp.getTarget())) {
                    if (target.getDevice() == "local") {
                        isTorqOp = false;
                        break;
                    }
                }
            }
        }

        if (isTorqOp) {
            torqOps.insert(&op);
        }
    }

    return torqOps;
}

static SmallVector<SmallVector<Operation *>> groupTorqOperations(FunctionOpInterface funcOp) {

    auto torqOps = getTorqOperations(funcOp);

    SmallVector<SmallVector<Operation *>> dispatchGroups;
    DenseSet<Operation *> currentGroupUsers;

    for (auto &op : funcOp.getCallableRegion()->front()) {
        if (torqOps.contains(&op)) {

            // create a new dispatch if necessary
            if (dispatchGroups.empty()) {
                dispatchGroups.emplace_back();
                currentGroupUsers.clear();
            }

            // put the operation in the current dispatch group
            dispatchGroups.back().push_back(&op);

            // add all the users of this operation to the users
            // of the current dispatch group, so that we can
            // decide when to start a new dispatch group
            for (auto result : op.getResults()) {
                for (auto user : result.getUsers()) {

                    // add to the list of users
                    currentGroupUsers.insert(user);

                    // add all the parents of the user (in case a value is used by an operation
                    // in a region of another sibiling operation)
                    while (true) {

                        if (user == funcOp.getOperation()) {
                            break;
                        }

                        currentGroupUsers.insert(user);
                        user = user->getParentOp();
                    }
                }
            }
        }
        else {
            // if this operation is not a torq operation, check if it is a user of the current
            // dispatch group
            if (currentGroupUsers.contains(&op)) {
                // if it is, then we need to start a new dispatch group
                dispatchGroups.emplace_back();
                currentGroupUsers.clear();
            }
        }
    }

    // remove the last group if it is empty
    if (!dispatchGroups.empty() && dispatchGroups.back().empty()) {
        dispatchGroups.pop_back();
    }

    return dispatchGroups;
}

static DenseSet<Operation *> findAllGroupOps(SmallVector<Operation *> dispatchGroup) {
    // find all operations in the dispatch group, including nested operations

    DenseSet<Operation *> groupOps;

    for (auto op : dispatchGroup) {
        op->walk([&](Operation *nestedOp) { groupOps.insert(nestedOp); });
    }

    return groupOps;
}

static DenseSet<Operation *>
findOpsUsedOutsideGroup(SmallVector<Operation *> dispatchGroup, DenseSet<Operation *> groupOps) {
    // find all the operations in the dispatch group that are used outside the dispatch group

    DenseSet<Operation *> opsUsedOutsideGroup;

    for (auto op : dispatchGroup) {
        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (!groupOps.contains(user)) {
                    opsUsedOutsideGroup.insert(op);
                    break;
                }
            }
        }
    }

    return opsUsedOutsideGroup;
}

static DenseSet<Operation *> findOpsToCopy(
    SmallVector<Operation *> dispatchGroup, DenseSet<Operation *> groupOps,
    DenseSet<Operation *> opsUsedOutsideGroup
) {
    // find all the operations in the dispatch group that we need to copy in the dispatch group
    // instead of moving them

    DenseSet<Operation *> opsToCopy;

    for (auto op : dispatchGroup) {

        // operations that are not used outside the group do not need to be copied
        if (!opsUsedOutsideGroup.contains(op)) {
            continue;
        }

        // dispatch groups can return only tensors, so we copy operations that produce anything else
        bool producesNonTensor =
            llvm::any_of(op->getResults(), [](Value v) { return !isa<TensorType>(v.getType()); });

        if (producesNonTensor) {
            opsToCopy.insert(op);

            SmallVector<Operation *> worklist{op};

            // for the moment we also copy all the operations used by this operation,
            // we could avoid this by splitting the dispatch region appropriately
            while (!worklist.empty()) {
                auto currentOp = worklist.pop_back_val();

                for (auto operand : currentOp->getOperands()) {
                    if (auto definingOp = operand.getDefiningOp()) {

                        // if the defining operation is already marked to be copied we can stop
                        // walking if the defining operation is not in the group we can also stop
                        // walking
                        if (opsToCopy.contains(definingOp) || !groupOps.contains(definingOp)) {
                            continue;
                        }

                        opsToCopy.insert(definingOp);
                        worklist.push_back(definingOp);
                    }
                }
            }

            continue;
        }
    }

    return opsToCopy;
}

static SmallVector<Value> findGroupResults(
    SmallVector<Operation *> dispatchGroup, DenseSet<Operation *> groupOps,
    DenseSet<Operation *> opsToCopy, DenseSet<Operation *> usedOutsideGroup
) {
    // find all values that are produced by operations used outside the group but not produced by
    // operations that we will copy

    SmallVector<Value> groupValues;

    for (auto op : usedOutsideGroup) {

        // we don't need to return values that are produce by operations we will copy
        // the operations outside the group that need them will just use the copied
        // version of the value
        if (opsToCopy.contains(op)) {
            continue;
        }

        for (auto result : op->getResults()) {
            for (auto user : result.getUsers()) {
                if (!groupOps.contains(user)) {
                    // if the user is not in the group then we need to return this value from the
                    // group
                    groupValues.push_back(result);
                    break;
                }
            }
        }
    }

    return groupValues;
}

static void outlineDispatchGroups(
    FunctionOpInterface funcOp, SmallVector<SmallVector<Operation *>> dispatchGroups
) {

    // for each dispatch group, outline it into a new function and replace the original operations
    // with a call to the new function
    IRRewriter rewriter(funcOp.getContext());

    for (auto &dispatchGroup : dispatchGroups) {

        auto groupOps = findAllGroupOps(dispatchGroup);
        auto usedOutsideGroup = findOpsUsedOutsideGroup(dispatchGroup, groupOps);
        auto opsToCopy = findOpsToCopy(dispatchGroup, groupOps, usedOutsideGroup);
        auto groupResults = findGroupResults(dispatchGroup, groupOps, opsToCopy, usedOutsideGroup);

        auto groupResultTypes =
            llvm::to_vector<4>(llvm::map_range(groupResults, [](Value v) { return v.getType(); }));

        rewriter.setInsertionPoint(dispatchGroup.back());

        auto dispatchRegionOp = IREE::Flow::DispatchRegionOp::create(
            rewriter, funcOp.getLoc(), groupResultTypes, /*result_dims=*/ValueRange{},
            /*workload=*/ValueRange{}
        );

        dispatchRegionOp.getBody().emplaceBlock();

        rewriter.setInsertionPointToStart(&dispatchRegionOp.getBody().front());

        IRMapping mapping;
        for (auto op : dispatchGroup) {
            // clone the operation into the dispatch region
            rewriter.clone(*op, mapping);
        }

        // add a terminator to the group
        auto results = llvm::to_vector<4>(llvm::map_range(groupResults, [&](Value v) {
            return mapping.lookupOrDefault(v);
        }));
        IREE::Flow::ReturnOp::create(rewriter, funcOp.getLoc(), results);

        // replace the original operations with the results of the dispatch region
        for (auto [oldValue, newValue] : llvm::zip(groupResults, dispatchRegionOp.getResults())) {
            rewriter.replaceAllUsesWith(oldValue, newValue);
        }

        // delete the original operations (in reverse order so we remove
        // dependencies correctly), except the ones we want to copy
        for (auto op : llvm::reverse(dispatchGroup)) {
            if (!opsToCopy.contains(op)) {
                rewriter.eraseOp(op);
            }
        }
    }
}

class OutlineTorqDispatchesPass
    : public impl::OutlineTorqDispatchesBase<OutlineTorqDispatchesPass> {
  public:
    void runOnOperation() override {

        for (auto funcOp : getOperation().getOps<FunctionOpInterface>()) {
            auto dispatchGroups = groupTorqOperations(funcOp);
            outlineDispatchGroups(funcOp, dispatchGroups);
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOutlineTorqDispatchesPass() {
    return std::make_unique<OutlineTorqDispatchesPass>();
}

} // namespace mlir::syna::torq