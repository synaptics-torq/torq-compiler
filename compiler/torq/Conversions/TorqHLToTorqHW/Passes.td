// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SYNPHL_TO_HW_PASSES
#define IREE_CODEGEN_SYNPHL_TO_HW_PASSES

include "mlir/Pass/PassBase.td"

def ConvertSliceProgramToTorqHw :
    Pass<"torq-convert-slice-program-to-torqhw", "torq_hl::ProgramOp"> {
        let summary = "Convert operations in a slice program to torq_hw";
        let constructor = "mlir::syna::torq::createConvertSliceProgramToTorqHwPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "syna::torq_hw::TorqHWDialect"
        ];
}

def ConvertNssProgramToTorqHw :
    Pass<"torq-convert-nss-program-to-torqhw", "torq_hl::ProgramOp"> {
        let summary = "Convert operations in a nss program to torq_hw";
        let constructor = "mlir::syna::torq::createConvertNssProgramToTorqHwPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "syna::torq_hw::TorqHWDialect"
        ];
}

#endif // IREE_CODEGEN_SYNPHL_TO_HW_PASSES
