// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Passes.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

#define GEN_PASS_DECL
#define GEN_PASS_CLASSES
#include "torq/Conversions/TosaToTorqHL/Passes.h.inc"

} // namespace mlir::syna::torq
