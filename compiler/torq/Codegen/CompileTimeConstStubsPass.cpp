// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-compile-time-const-stubs"

namespace mlir::syna::torq {

// The pass looks for operations marked as compile-time-const and
// replaces them with stubs.
class CompileTimeConstStubsPass
    : public impl::CompileTimeConstStubsBase<CompileTimeConstStubsPass> {
  public:
    using CompileTimeConstStubsBase::CompileTimeConstStubsBase;

    void runOnOperation() override {
        SmallVector<Operation *> opsToProcess;
        auto funcOp = getOperation();
        funcOp->walk([&](Operation *op) {
            if (!isCompileTimeConst(op)) {
                return WalkResult::advance();
            }
            opsToProcess.push_back(op);
            return WalkResult::advance();
        });

        OpBuilder builder(&getContext());

        for (auto op : opsToProcess) {
            builder.setInsertionPoint(op);

            OpResult val = op->getResult(0);
            ShapedType shapeType = cast<mlir::ShapedType>(val.getType());

            auto emptyOp = tensor::EmptyOp::create(builder, op->getLoc(), shapeType, {});
            val.replaceAllUsesWith(emptyOp.getResult());
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstStubsPass() {
    return std::make_unique<CompileTimeConstStubsPass>();
}
} // namespace mlir::syna::torq
