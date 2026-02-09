// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SYNPHL_TO_HW_PASSES
#define IREE_CODEGEN_SYNPHL_TO_HW_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

//===---------------------------------------------------------------------===//
// Torq type-conversion passes.
//===---------------------------------------------------------------------===//

def TorqDemoteF32ToBF16 : Pass<"torq-demote-f32-to-bf16", "mlir::ModuleOp"> {
  let summary = "Demotes float32 tensors to bfloat16 tensors";
  let description = [{
    Converts eligible float32 payloads to bfloat16. This pass is intended to
    run early in the Torq pipeline and may introduce precision loss.
  }];
  let constructor = "mlir::syna::torq::createTorqDemoteF32ToBF16Pass()";
  let dependentDialects = ["arith::ArithDialect"];
}

def TorqConvertF16ToBF16 : Pass<"torq-convert-f16-to-bf16", "mlir::ModuleOp"> {
  let summary = "Converts float16 tensors to bfloat16 tensors";
  let description = [{
    Converts float16 payloads to bfloat16. This pass is intended to run early in
    the Torq pipeline and may introduce precision loss.
  }];
  let constructor = "mlir::syna::torq::createTorqConvertF16ToBF16Pass()";
  let dependentDialects = ["arith::ArithDialect"];
}

def TorqDemoteI64ToI32 : Pass<"torq-demote-i64-to-i32", "mlir::ModuleOp"> {
  let summary = "Demotes int64 tensors to int32 tensors";
  let description = [{
    Converts eligible int64 payloads to int32. This pass is intended to run
    early in the Torq pipeline and may introduce truncation.
  }];
  let constructor = "mlir::syna::torq::createTorqDemoteI64ToI32Pass()";
  let dependentDialects = ["arith::ArithDialect"];
}

//===---------------------------------------------------------------------===//

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
