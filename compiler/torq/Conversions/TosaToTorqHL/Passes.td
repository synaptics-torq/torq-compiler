// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SYNPHL_TO_HW_PASSES
#define IREE_CODEGEN_SYNPHL_TO_HW_PASSES

include "mlir/Pass/PassBase.td"

def TosaToTorqHLConversion :
    InterfacePass<"torq-tosa-to-torqhl-conversion",
                  "mlir::FunctionOpInterface"> {

    let summary = "Lower TOSA operators to TorqHL and linalg dialect";

    let description = [{
        Converts from various input dialects to the TorqHL dialect
    }];

    let constructor =
        "mlir::syna::torq::createTosaToTorqHLConversionPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::linalg::LinalgDialect"
    ];
}

#endif // IREE_CODEGEN_SYNPHL_TO_HW_PASSES
