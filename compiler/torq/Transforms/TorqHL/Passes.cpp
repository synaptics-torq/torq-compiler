// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::syna::torq_hl {

//===---------------------------------------------------------------------===//
// Register TorqHL Passes
//===---------------------------------------------------------------------===//

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Transforms/TorqHL/Passes.h.inc"
} // namespace

void registerTorqHLPasses() {
    // Generated.
    registerPasses();
}

} // namespace mlir::syna::torq_hl
