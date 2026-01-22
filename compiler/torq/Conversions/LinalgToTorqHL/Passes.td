// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_SYNPHL_TO_HW_PASSES
#define IREE_CODEGEN_SYNPHL_TO_HW_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def LinalgToTorqHLConversion :
    InterfacePass<"torq-linalg-to-torqhl-conversion",
                  "mlir::FunctionOpInterface"> {

    let summary = "Lower linalg operators to TorqHL";

    let description = [{
        Converts linalg to the TorqHL dialect
    }];

    let constructor =
        "mlir::syna::torq::createLinalgToTorqHLConversionPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::linalg::LinalgDialect"
    ];
}

def LinalgToTorqHLPreConversion :
    InterfacePass<"torq-linalg-to-torqhl-pre-conversion",
      "mlir::FunctionOpInterface"> {

    let summary = "Lower linalg operators to TorqHL before tiling";
    
    let description = [{
        Converts linalg to the TorqHL dialect before tiling
    }];
    
    let constructor =
        "mlir::syna::torq::createLinalgToTorqHLPreConversionPass()";
    
    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::linalg::LinalgDialect"
    ];

    let options = RewritePassUtils.options;
}

def TensorToLinalg :
    InterfacePass<"tensor-to-linalg",
      "mlir::FunctionOpInterface"> {

    let summary = "Convert tensor collapse_shape and expand_shape to linalg.generic";

    let description = [{
        Convert tensor collapse_shape and expand_shape to linalg.generic so they don't break tile-fusing.
    }];

    let constructor =
        "mlir::syna::torq::createTensorToLinalgPass()";

    let dependentDialects = [
        "::mlir::tensor::TensorDialect",
        "::mlir::linalg::LinalgDialect"
    ];

    let options = RewritePassUtils.options;
}

def MarkPatternsForTileAndFuse :
    InterfacePass<"torq-mark-patterns-for-tile-and-fuse",
      "mlir::FunctionOpInterface"> {

    let summary = "Identify patterns that tile-and-fuse should not break";

    let description = [{
        This pass runs rewrite patterns from other passes, but without doing the rewrites. Instead, it marks the operations involved in the match of each pattern. This marking allows tile-and-fuse to make sure operations that belong to a single pattern will be fused in the same tile.
    }];

    let constructor =
        "mlir::syna::torq::createMarkPatternsForTileAndFusePass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect"
    ];

    let options = RewritePassUtils.options;
}

def ArithToTorqHLConversion :
    InterfacePass<"torq-arith-to-torqhl-conversion",
      "mlir::FunctionOpInterface"> {

    let summary = "Lower arith operators to TorqHL";

    let description = [{
        Converts arith to the TorqHL dialect
    }];

    let constructor =
        "mlir::syna::torq::createArithToTorqHLConversionPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "arith::ArithDialect"
    ];
}


#endif // IREE_CODEGEN_SYNPHL_TO_HW_PASSES
