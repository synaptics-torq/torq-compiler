// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LINALG_TO_TORQ_GENERIC_OP
#define IREE_LINALG_TO_TORQ_GENERIC_OP

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LinalgToTorqHLGeneric :
    InterfacePass<"torq-specialize-to-torqhl",
         "mlir::FunctionOpInterface"> {

    let summary = "Transform linalg.generic operations to torq.generic operations";

    let constructor =
        "mlir::syna::torq::createLinalgToTorqHLGenericPass()";

    let dependentDialects = [
        "linalg::LinalgDialect",
        "syna::torq_hl::TorqHLDialect"
    ];

    let options = [
      Option<"specializeConstantComputations", "specialize-constant-computations", "bool",
              /*default=*/"false",
             "Do not specialize linalg operations that have constant results">
    ];
}

def SpecializeLinalgGenericOp :
    InterfacePass<"torq-specialize-linalg-generic-op",
                  "mlir::FunctionOpInterface"> {

    let summary = "Specialize linalg.generic operations";

    let description = [{
        Specialize linalg.generic operations
    }];

    let constructor =
        "mlir::syna::torq::createSpecializeLinalgGenericOpPass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect",
    ];
}
#endif // IREE_LINALG_TO_TORQ_GENERIC_OP
