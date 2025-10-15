// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SYNPHL_TO_HW_PASSES
#define IREE_CODEGEN_SYNPHL_TO_HW_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def FoldConstants :
    InterfacePass<"torq-fold-constants",
                  "mlir::FunctionOpInterface"> {

    let summary = "Computes all operations that are constant";

    let constructor = "mlir::syna::torq::createFoldConstantsPass()";

    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

def GeneralizeLinalgNamedOps :
    InterfacePass<"torq-generalize-linalg-named-ops",
                  "mlir::FunctionOpInterface"> {

    let summary = "Converts linalg named ops to linalg generic ops";

    let constructor =
        "mlir::syna::torq::createGeneralizeLinalgNamedOpsPass()";

    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

def OptimizeLinalgForTorq :
    InterfacePass<"torq-optimize-linalg-for-torq",
                  "mlir::FunctionOpInterface"> {

    let summary = "Rewrites linalg operations in a form more suitable for execution on Torq";

    let constructor =
        "mlir::syna::torq::createOptimizeLinalgForTorqPass()";

    let options = RewritePassUtils.options;

    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}


#endif // IREE_CODEGEN_SYNPHL_TO_HW_PASSES
