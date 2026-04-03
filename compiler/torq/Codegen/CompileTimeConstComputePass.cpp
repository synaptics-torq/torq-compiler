// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-compute-compile-time-const"

namespace mlir::syna::torq {

// The pass looks for operations marked as compile-time-const and
// replaces them with constant operations computed at compile time.
class CompileTimeConstComputePass
    : public impl::CompileTimeConstComputeBase<CompileTimeConstComputePass> {
  public:
    using CompileTimeConstComputeBase::CompileTimeConstComputeBase;

    void runOnOperation() override {
        SmallVector<Value> valuesToProcess;

        auto funcOp = getOperation();
        funcOp->walk([&](Operation *op) {
            if (op->getNumResults() != 1) {
                LLVM_DEBUG({
                    llvm::dbgs(
                    ) << "Skipping compile-time const op with unsupported result count: ";
                    op->dump();
                });
                return WalkResult::advance();
            }

            if (!isCompileTimeConst(op)) {
                return WalkResult::advance();
            }

            valuesToProcess.push_back(op->getResult(0));
            return WalkResult::advance();
        });

        if (valuesToProcess.empty()) {
            return;
        }

        auto maybeConstValues = computeAllArithConst(valuesToProcess, true, {});
        if (failed(maybeConstValues)) {
            LLVM_DEBUG({ llvm::dbgs() << "Failed to compute compile-time constants in batch\n"; });
            return;
        }

        for (auto [idx, v] : llvm::enumerate(valuesToProcess)) {
            v.replaceAllUsesWith((*maybeConstValues)[idx]);
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstComputePass() {
    return std::make_unique<CompileTimeConstComputePass>();
}
} // namespace mlir::syna::torq
