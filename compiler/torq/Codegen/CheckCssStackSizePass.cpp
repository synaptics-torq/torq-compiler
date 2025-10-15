// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#define DEBUG_TYPE "torq-check-css-stack-size"

// using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class CheckCssStackSizePass : public CheckCssStackSizeBase<CheckCssStackSizePass> {
  public:
    using CheckCssStackSizeBase::CheckCssStackSizeBase;

    void runOnOperation() override;
};

void CheckCssStackSizePass::runOnOperation() {

    int totalAllocs = 0;

    getOperation().walk([&](memref::AllocaOp allocaOp) {
        totalAllocs += getEncodedTotalSizeBytes(allocaOp.getResult().getType());
    });

    if (totalAllocs > HwInfo::css_stack_size) {
        getOperation().emitError()
            << "CSS program allocations of " << totalAllocs
            << " bytes exceed maximum CSS stack size of " << HwInfo::css_stack_size;
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCheckCssStackSizePass() {
    return std::make_unique<CheckCssStackSizePass>();
}

} // namespace mlir::syna::torq
