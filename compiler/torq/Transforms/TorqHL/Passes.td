// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_
#define IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

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

def TorqHLFoldTableConstant :
    InterfacePass<"torq-hl-fold-table-constant",
         "mlir::FunctionOpInterface"> {
  let summary = "Fold constant dynamic_table into static_table on TorqHL TableOp";
  let description = [{
    Walks every torq_hl.table op.  When the dynamic_table operand is backed by
    a compile-time constant (torq_hl.const loaded via torq_hl.load), the
    DenseI32ArrayAttr is materialised as static_table so that downstream
    patterns can lower without runtime memory traffic.
  }];
  let constructor =
      "mlir::syna::torq_hl::createTorqHLFoldTableConstantPass()";
  let dependentDialects = [
      "syna::torq_hl::TorqHLDialect",
  ];
}

def ConvertUtilGlobalsToTorchLiterals :
    Pass<"torq-convert-util-globals-to-torch-literals", "mlir::ModuleOp"> {

  let summary = "Convert util.global constants to torch.vtensor.literal";

  let description = [{
    Folds util.global constants consumed by torch_c.from_builtin_tensor into
    torch.vtensor.literal ops. This allows iree-turbine's parameter
    externalization output to be processed by the standard torch backend
    pipeline, which expects torch tensor literals rather than util.global
    loads.
  }];

  let constructor =
      "mlir::syna::torq_hl::createConvertUtilGlobalsToTorchLiteralsPass()";

  let dependentDialects = [
        "::mlir::BuiltinDialect",
    ];

}

def InlineDenseResources :
    Pass<"torq-inline-dense-resources", "mlir::ModuleOp"> {

  let summary = "Inline DenseResourceElementsAttr to DenseElementsAttr";

  let description = [{
    Converts all DenseResourceElementsAttr attributes to DenseElementsAttr.
    This works around a torch-mlir/IREE bug where bf16 dense_resource constants
    get their result type incorrectly promoted to f32 during lowering.
  }];

  let constructor =
      "mlir::syna::torq_hl::createInlineDenseResourcesPass()";

  let dependentDialects = [
        "::mlir::BuiltinDialect"
    ];

}

#endif // IREE_COMPILER_DIALECT_TORQHL_TRANSFORMS_PASSES_TD_
