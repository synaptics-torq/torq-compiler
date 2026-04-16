// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_TORQHLTOLINALG_PASSES
#define IREE_CODEGEN_TORQHLTOLINALG_PASSES

include "mlir/Pass/PassBase.td"

def TorqHLToLinalgConversion :
    InterfacePass<"torq-torqhl-to-linalg-conversion",
                  "mlir::FunctionOpInterface"> {

    let summary = "Lower TorqHL operators to Linalg dialect";

    let description = [{
        Converts TorqHL operations to the Linalg dialect so they can be
        processed by standard downstream passes (e.g. JIT constant evaluation).
    }];

    let constructor =
        "mlir::syna::torq::createTorqHLToLinalgConversionPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::linalg::LinalgDialect"
    ];
}

#endif // IREE_CODEGEN_TORQHLTOLINALG_PASSES
