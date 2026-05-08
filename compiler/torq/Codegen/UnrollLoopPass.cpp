// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

namespace {

struct UnrollLoopPass : public impl::UnrollLoopBase<UnrollLoopPass> {
    void runOnOperation() override {
        auto funcOp = getOperation();
        funcOp->walk([](scf::ForOp forOp) { (void)loopUnrollFull(forOp); });
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollLoopPass() {
    return std::make_unique<UnrollLoopPass>();
}

} // namespace mlir::syna::torq
