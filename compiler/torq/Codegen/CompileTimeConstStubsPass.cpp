// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
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
        auto funcOp = getOperation();
        auto valuesToProcess = collectAllCompileTimeConstOps(funcOp);
        funcOp->walk([&](Operation *op) {
            if (!isCompileTimeConst(op)) {
                return WalkResult::advance();
            }
            IRRewriter rewriter(op->getContext());
            removeCompileTimeConst(op, rewriter);
            return WalkResult::skip();
        });
        OpBuilder builder(&getContext());

        for (auto val : valuesToProcess) {
            builder.setInsertionPoint(val.getDefiningOp());

            ShapedType shapeType = cast<mlir::ShapedType>(val.getType());

            auto emptyOp = tensor::EmptyOp::create(builder, val.getLoc(), shapeType, {});
            val.replaceAllUsesWith(emptyOp.getResult());
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstStubsPass() {
    return std::make_unique<CompileTimeConstStubsPass>();
}
} // namespace mlir::syna::torq
