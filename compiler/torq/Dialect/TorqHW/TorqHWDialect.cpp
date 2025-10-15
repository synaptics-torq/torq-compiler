// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Dialect/TorqHW/TorqHWAttrs.h"

#include "torq/Dialect/TorqHW/TorqHWDialect.cpp.inc"
#include "torq/Dialect/TorqHW/TorqHWOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir::syna::torq_hw {

void TorqHWDialect::initialize() {
    initializeTorqHWAttrs();

    addOperations<
#define GET_OP_LIST
#include "torq/Dialect/TorqHW/TorqHWOps.cpp.inc"
        >();
}

} // namespace mlir::syna::torq_hw
