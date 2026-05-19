// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_TOSA_TO_LINALG_PASSES
#define IREE_CODEGEN_TOSA_TO_LINALG_PASSES

include "mlir/Pass/PassBase.td"

def TosaToLinalgConversion :
    InterfacePass<"torq-tosa-to-linalg-conversion",
                  "mlir::FunctionOpInterface"> {

    let summary = "Lower TOSA operators to Linalg dialect";

    let description = [{
        Converts from various input dialects to the Linalg dialect
    }];

    let constructor =
        "mlir::syna::torq::createTosaToLinalgConversionPass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect"
    ];
}

#endif // IREE_CODEGEN_TOSA_TO_LINALG_PASSES
