// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_TRANSFORMS_TOSA_PASSES_TD_
#define TORQ_TRANSFORMS_TOSA_PASSES_TD_

include "mlir/Pass/PassBase.td"

def OptimizeTosaForTorq :
    InterfacePass<"torq-optimize-tosa-for-torq", "mlir::FunctionOpInterface"> {

  let summary = "Rewrites TOSA operations in a form more suitable for execution on Torq";
  let description = [{
    TOSA to TOSA rewrites that put TOSA operations in a form the Torq kernels support. They run
    before the upstream TOSA decompositions and the lowering to linalg.
  }];

  let constructor = "mlir::syna::torq::createOptimizeTosaForTorqPass()";

  let dependentDialects = [
    "tensor::TensorDialect",
    "tosa::TosaDialect"
  ];
}

#endif // TORQ_TRANSFORMS_TOSA_PASSES_TD_
