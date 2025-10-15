// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq_hl {

#define GEN_PASS_CLASSES
#define GEN_PASS_DECL
// to get Pass Base class implement constructor with options and others automatically
#define GEN_PASS_DEF_TORQHLOPTIMIZESEGMENTATION
#define GEN_PASS_DEF_FORMDISPATCHREGIONS
#include "torq/Transforms/TorqHL/Passes.h.inc"

} // namespace mlir::syna::torq_hl
