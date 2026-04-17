// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-annotate-tied-operands"

namespace mlir::syna::torq {
namespace {

/// Walk the SSA def-use chain **backwards** from `root` and collect every
/// `BlockArgument` of `entryBlock` that is transitively reachable.
static void
traceToFuncArgs(Value root, Block *entryBlock, SmallPtrSetImpl<BlockArgument> &reachedArgs) {
    SmallVector<Value, 32> worklist;
    llvm::DenseSet<Value> visited;
    worklist.push_back(root);

    while (!worklist.empty()) {
        Value val = worklist.pop_back_val();
        if (!visited.insert(val).second)
            continue;

        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
            if (blockArg.getOwner() == entryBlock)
                reachedArgs.insert(blockArg);
            // Don't traverse into other blocks' arguments.
            continue;
        }

        Operation *defOp = val.getDefiningOp();
        if (!defOp)
            continue;

        // Walk all operands of the defining operation.
        for (Value operand : defOp->getOperands())
            worklist.push_back(operand);

        // Also walk into nested regions (e.g. linalg.generic bodies,
        // scf.for, etc.) — the region's yield operands feed back into
        // the parent op's results, but operands of ops inside the region
        // may also reference outer values via block arguments.
        for (Region &region : defOp->getRegions()) {
            for (Block &block : region) {
                if (auto *terminator = block.getTerminator()) {
                    for (Value operand : terminator->getOperands())
                        worklist.push_back(operand);
                }
            }
        }
    }
}

struct AnnotateTiedOperandsPass : public impl::AnnotateTiedOperandsBase<AnnotateTiedOperandsPass> {

    void runOnOperation() override {
        ModuleOp module = getOperation();
        OpBuilder builder(module.getContext());

        module.walk([&](func::FuncOp funcOp) {
            if (funcOp.isExternal())
                return;

            auto funcType = funcOp.getFunctionType();
            unsigned numResults = funcType.getNumResults();
            if (numResults == 0)
                return;

            // Only handle single-block functions for now.
            if (!funcOp.getBody().hasOneBlock())
                return;

            Block &entryBlock = funcOp.getBody().front();

            // Find the return op; bail if there are multiple.
            func::ReturnOp returnOp = nullptr;
            bool multipleReturns = false;
            funcOp.walk([&](func::ReturnOp op) {
                if (returnOp)
                    multipleReturns = true;
                returnOp = op;
            });
            if (!returnOp || multipleReturns)
                return;

            // Track which arguments have already been claimed by a tied
            // result so each argument is tied at most once.
            // Pre-populate from any existing annotations to avoid conflicts.
            llvm::DenseSet<unsigned> claimedArgs;
            for (unsigned i = 0; i < numResults; ++i) {
                if (auto attr = funcOp.getResultAttr(i, "iree.abi.tied")) {
                    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
                        claimedArgs.insert(intAttr.getInt());
                }
            }

            for (unsigned i = 0; i < numResults; ++i) {
                // Skip results that are already annotated.
                if (funcOp.getResultAttr(i, "iree.abi.tied"))
                    continue;

                Value returnVal = returnOp.getOperand(i);
                Type resultType = returnVal.getType();

                // Only handle ranked tensor types.
                auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
                if (!resultTensorType)
                    continue;

                // Trace backward to function arguments.
                SmallPtrSet<BlockArgument, 4> reachedArgs;
                traceToFuncArgs(returnVal, &entryBlock, reachedArgs);

                // Filter to arguments with the exact same ranked tensor type
                // that have not already been claimed.
                SmallVector<BlockArgument, 2> candidates;
                for (BlockArgument arg : reachedArgs) {
                    if (arg.getType() == resultType && !claimedArgs.contains(arg.getArgNumber())) {
                        candidates.push_back(arg);
                    }
                }

                if (candidates.size() != 1)
                    continue; // Ambiguous or no match — skip.

                unsigned argIndex = candidates[0].getArgNumber();
                claimedArgs.insert(argIndex);

                funcOp.setResultAttr(i, "iree.abi.tied", builder.getI64IntegerAttr(argIndex));

                LLVM_DEBUG(
                    llvm::dbgs() << "Tied result #" << i << " -> arg #" << argIndex << " in @"
                                 << funcOp.getName() << "\n"
                );
            }
        });
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateTiedOperandsPass() {
    return std::make_unique<AnnotateTiedOperandsPass>();
}

} // namespace mlir::syna::torq
