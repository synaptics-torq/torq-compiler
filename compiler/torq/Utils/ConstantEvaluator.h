// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

#include <memory>

namespace mlir::syna::torq {

// Computes compile-time constants from the data of the constants they depend on, without the
// JIT. It supports the ops of the usual weight-preparation cones: tensor.empty, linalg.fill,
// tensor slices and reshapes, torq_hl and linalg transposes, generics that copy or compute with
// integer arith ops (including integer reductions), scf.for and scf.forall with constant
// bounds, and scalar ops that fold.
class ConstantEvaluator {
  public:
    // With `recursive` unset, only the op that defines the value is computed: its operands must
    // be constants. Values in `assumeZero` are never computed here.
    ConstantEvaluator(bool recursive, llvm::ArrayRef<Value> assumeZero);
    ~ConstantEvaluator();

    // Returns the value of `value`, or null if an op isn't supported or the result contains
    // elements no op has written. Results are reused between calls.
    DenseElementsAttr evaluate(Value value);

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace mlir::syna::torq
