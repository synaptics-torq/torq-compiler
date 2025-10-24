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

def MarkPatternsForSuperTiling :
    InterfacePass<"torq-mark-patterns-for-super-tiling",
      "mlir::FunctionOpInterface"> {

    let summary = "Identify patterns that super tiling should not break";

    let description = [{
        This pass runs rewrite patterns from other passes, but without doing the rewrites. Instead, it marks the operations involved in the match of each pattern. This marking allows super tiling to make sure operations that belong to a single pattern will be fused in the same tile.
    }];

    let constructor =
        "mlir::syna::torq::createMarkPatternsForSuperTilingPass()";

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

def LramTile:
    InterfacePass<"torq-lram-tile",
                  "mlir::FunctionOpInterface"> {

    let summary = "Tile operations to fit in LRAM";

    let description = [{
        Tile operations to fit LRAM constraints
    }];

    let constructor =
        "mlir::syna::torq::createLramTilePass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect"
    ];
}

def DtcmTile:
    InterfacePass<"torq-dtcm-tile",
                  "mlir::FunctionOpInterface"> {

    let summary = "Tile operations to fit in DTCM";

    let description = [{
        Tile operations to fit DTCM constraints
    }];

    let constructor =
        "mlir::syna::torq::createDtcmTilePass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect"
    ];
}


def UnrollLoop:
    InterfacePass<"torq-unroll-loop",
                  "mlir::FunctionOpInterface"> {

    let summary = "Unroll loops";

    let description = [{
       Unroll loop
    }];

    let constructor =
        "mlir::syna::torq::createUnrollLoopPass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect",
        // "::mlir::scf::SCFDialect"
    ];
}

#endif // IREE_CODEGEN_SYNPHL_TO_HW_PASSES
