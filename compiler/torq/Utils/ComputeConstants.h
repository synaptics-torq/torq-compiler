// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

namespace mlir::syna::torq {

// Return a dense elements vector with the recursively computed content of the constant
// (nullptr if not const)
FailureOr<Attribute>
computeConstAttr(Value value, bool recursive = true, const std::vector<Value> &assumeZero = {});

// Return a dense elements vector with the recursively computed content of the constant
// (nullptr if not const)
FailureOr<Value>
computeArithConst(Value value, bool recursive = true, const std::vector<Value> &assumeZero = {});

// Return a dense elements vector with the computed content of the constant (nullptr if not const)
// assumeZero value, if specified, is considered to be a zero-filled tensor
FailureOr<Value> computeArithConst(
    linalg::LinalgOp linalgOp, bool recursive = true, const std::vector<Value> &assumeZero = {}
);

} // namespace mlir::syna::torq
