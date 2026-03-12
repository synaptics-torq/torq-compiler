// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::syna::torq {

// Return a dense elements vector with the recursively computed content of the constant
// (nullptr if not const)
FailureOr<Attribute>
computeConstAttr(Value value, bool recursive = true, llvm::ArrayRef<Value> assumeZero = {});

// Return dense element attributes for all values using one batched compute path.
FailureOr<SmallVector<Attribute>> computeAllConstAttr(
    llvm::SmallVectorImpl<Value> &values, bool recursive = true,
    llvm::ArrayRef<Value> assumeZero = {}
);

// Return a dense elements vector with the recursively computed content of the constant
// (nullptr if not const)
FailureOr<Value>
computeArithConst(Value value, bool recursive = true, llvm::ArrayRef<Value> assumeZero = {});

// Return arith.constant values for all values using one batched compute path.
FailureOr<SmallVector<Value>> computeAllArithConst(
    llvm::SmallVectorImpl<Value> &values, bool recursive = true,
    llvm::ArrayRef<Value> assumeZero = {}
);

// Return a dense elements vector with the computed content of the constant (nullptr if not const)
// assumeZero value, if specified, is considered to be a zero-filled tensor
FailureOr<Value> computeArithConst(
    linalg::LinalgOp linalgOp, bool recursive = true, llvm::ArrayRef<Value> assumeZero = {}
);

// Return arith.constant values for all operators using one batched compute path.
FailureOr<SmallVector<Value>> computeAllArithConst(
    llvm::SmallVectorImpl<mlir::Operation *> &ops, bool recursive = true,
    llvm::ArrayRef<Value> assumeZero = {}
);

} // namespace mlir::syna::torq
