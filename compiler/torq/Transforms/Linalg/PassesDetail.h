// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

#define GEN_PASS_DECL

#define GEN_PASS_DEF_FOLDCONSTANTS
#define GEN_PASS_DEF_GENERALIZELINALGNAMEDOPS
#define GEN_PASS_DEF_OPTIMIZELINALGFORTORQ
#define GEN_PASS_DEF_OPTIMIZETRANSPOSELAYOUT
#define GEN_PASS_DEF_TORQCONVERTF16TOBF16
#define GEN_PASS_DEF_TORQDEMOTEF32TOBF16
#define GEN_PASS_DEF_TORQDEMOTEI64TOI32

#include "torq/Transforms/Linalg/Passes.h.inc"

} // namespace mlir::syna::torq
