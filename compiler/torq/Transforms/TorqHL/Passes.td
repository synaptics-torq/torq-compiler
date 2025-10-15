// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_
#define IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// TORQ passes (keep alphabetical)
//===---------------------------------------------------------------------===//

def TorqHLOptimizeSegmentation :
    InterfacePass<"torq-torqhl-optimize-segmentation",
         "mlir::FunctionOpInterface"> {
  let summary = "Optimize TorqHL segmentation Op";
  let description = [{
    Optimize
  }];
  let constructor =
      "mlir::syna::torq_hl::createTorqHLOptimizeSegmentationPass()";
    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
    ];
}

def FormDispatchRegions :
    InterfacePass<"torq-form-dispatch-regions",
         "mlir::FunctionOpInterface"> {

  let summary = "Group TorqHL operations into dispatch regions";

  let constructor =
      "mlir::syna::torq_hl::createFormDispatchRegionsPass()";

  let dependentDialects = [
        "bufferization::BufferizationDialect",
        "iree_compiler::IREE::HAL::HALDialect",
        "iree_compiler::IREE::Flow::FlowDialect",
        "syna::torq_hl::TorqHLDialect"
    ];

  let options = [
    Option<"disableDispatchFusion", "disable-dispatch-fusion", "bool",
            /*default=*/"false",
           "Do not fuse all operations into a single dispatch">
  ];
}

#endif // IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_
