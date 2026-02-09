// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::syna::torq {

//===---------------------------------------------------------------------===//
// Register Linalg Optimization For Torq Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Transforms/Linalg/Passes.h.inc"
} // namespace

void registerOptimizeLinalgForTorqPasses() {
    // Generated.
    registerPasses();
}

void registerTorqTypeConversionPasses() {
    // Generated.
    registerPasses();
}

} // namespace mlir::syna::torq
