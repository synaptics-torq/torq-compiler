// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TORQHL_GENERIC_TRANSFORMS_PASSES_TD_
#define IREE_TORQHL_GENERIC_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SimplifyGeneric :
    InterfacePass<"torq-simplify-generic",
         "mlir::FunctionOpInterface"> {

  let summary = "Re-arrange and fuse torq_hl.generic operations to reduce the number of operations";

  let constructor =
      "mlir::syna::torq_hl::createSimplifyGenericPass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
        "syna::torq_hl::TorqHLDialect"
    ];
}

def MergeBiasScaleTensors :
    InterfacePass<"torq-merge-weight-scale-tensors",
         "mlir::FunctionOpInterface"> {

  let summary = "Rewrite all torq_hl.generic operations to use bias and scale coming from the same tensor";

  let constructor =
      "mlir::syna::torq_hl::createMergeBiasScaleTensorsPass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
        "syna::torq_hl::TorqHLDialect"
    ];
}

#endif // IREE_TORQHL_GENERIC_TRANSFORMS_PASSES_TD_
