// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

namespace mlir::syna::torq {

FailureOr<DenseIntOrFPElementsAttr>
computeValue(Value value, bool recursive, const std::vector<Value> &assumeZero);

// Return a dense elements vector with the recursively computed content of the constant
// (nullptr if not const)
DenseIntOrFPElementsAttr
computeConstant(Value value, bool recursive = true, const std::vector<Value> &assumeZero = {});

// Return a dense elements vector with the computed content of the constant (nullptr if not const)
// assumeZero value, if specified, is considered to be a zero-filled tensor
DenseIntOrFPElementsAttr computeConstant(
    linalg::LinalgOp linalgOp, bool recursive = true, const std::vector<Value> &assumeZero = {}
);

} // namespace mlir::syna::torq
