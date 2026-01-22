// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_TORQ_PASSES
#define IREE_CODEGEN_TORQ_PASSES

include "mlir/Pass/PassBase.td"

//===---------------------------------------------------------------------===//
// TORQ passes (keep alphabetical)
//===---------------------------------------------------------------------===//

def ValidToSamePadPass
    : InterfacePass<"torq-convert-conv-valid-to-same", "mlir::FunctionOpInterface"> {

  let summary = "Converts ops with VALID padding into SAME padding by inserting padding and slicing.";
  
  let constructor = "mlir::syna::torq::createValidToSamePadPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def CheckCssStackSize
    : InterfacePass<"torq-check-css-stack-size", "mlir::FunctionOpInterface"> {

  let summary = "Checks that a CSS program doesn't exceed the stack size.";

  let constructor = "mlir::syna::torq::createCheckCssStackSizePass()";

  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def TORQLowerExecutableTarget
    :  Pass<"iree-torq-lower-executable-target-pass", "ModuleOp"> {

    let summary = "Lower the executable target to TORQ using one of the "
                "IREE::HAL::DispatchLoweringPassPipeline";
    let constructor = "mlir::syna::torq::createTORQLowerExecutableTargetPass()";

    // TODO: this list should be generated from the passes we call
    let dependentDialects = [
        "linalg::LinalgDialect",
        "memref::MemRefDialect",
        "bufferization::BufferizationDialect",
        "scf::SCFDialect",
        "syna::torq_hl::TorqHLDialect",
        "syna::torq_hw::TorqHWDialect"
    ];

}

def MapBindings
    : InterfacePass<"torq-map-bindings", "mlir::FunctionOpInterface"> {

    let summary = "Lower all the hal.interface.binding.subspan ops to torq_hl.map_binding ops";
    let constructor = "mlir::syna::torq::createMapBindingsPass()";

    let dependentDialects = [
        "iree_compiler::IREE::HAL::HALDialect",
        "syna::torq_hl::TorqHLDialect"
    ];
}

def LowerArithConstants
    : InterfacePass<"torq-lower-arith-constants", "mlir::FunctionOpInterface"> {

    let summary = "Lower all the arith.constant ops to torq_hl.const ops";
    let constructor = "mlir::syna::torq::createLowerArithConstantsPass()";

    let dependentDialects = [
        "bufferization::BufferizationDialect",
        "syna::torq_hl::TorqHLDialect"
    ];
}

def OutlineSliceTasks :
    InterfacePass<"torq-outline-slice-tasks",
         "mlir::FunctionOpInterface"> {
  let summary = "Wrap all operators that run on slices into a TaskOp";
  let constructor =
      "mlir::syna::torq::createOutlineSliceProgramsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];

}

def AssignOperationsToCpuPrograms : Pass<"torq-assign-operations-to-cpu-programs", "ModuleOp"> {

  let summary = "Create programs for the CPUs (CSS/host)";
  let description = "Moves suitable operations into CPU programs that will be later outlined";

  let constructor = "mlir::syna::torq::createAssignOperationsToCpuProgramsPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect"
  ];

  let options = [
    Option<"disableCSS", "disable-css", "bool",
              /*default=*/"false",
             "Do not execute operations on CSS">,
    Option<"disableHost", "disable-host", "bool",
            /*default=*/"false",
            "Do not execute operations on the host">
  ];

}


def OutlineCpuPrograms :
    Pass<"torq-outline-cpu-tasks", "ModuleOp"> {
  let summary = "Wrap code that needs to run on CSS or host into func.func operators";
  let constructor = "mlir::syna::torq::createOutlineCpuProgramsPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect"
  ];

}

def CompileCpuPrograms :
    Pass<"torq-compile-css-tasks", "ModuleOp"> {
  let summary = "Compile all the CSS tasks source code to a binary";
  let constructor = "mlir::syna::torq::createCompileCpuProgramsPass()";

  let dependentDialects = [
    "iree_compiler::IREE::HAL::HALDialect"
  ];
}

def LowerCallProgramToStartWait :
    InterfacePass<"torq-lower-css-calls-to-start-wait",
         "mlir::FunctionOpInterface"> {

  let summary = "Converts all torq_hl.call instructions to torq_hl.css_start and torq_hl.css_wait";
  let constructor = "mlir::syna::torq::createLowerCallProgramToStartWaitPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect"
  ];
}

def OutlineNSSPrograms :
    InterfacePass<"torq-outline-nss-programs",
         "mlir::FunctionOpInterface"> {
  let summary = "Outlines NSS programs into torq_hl.program ops";
  let constructor =
      "mlir::syna::torq::createOutlineNSSProgramsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];

}

def ResolveInvocationArguments :
    InterfacePass<"torq-resolve-invocation-arguments",
         "mlir::FunctionOpInterface"> {
  let summary = "Copy in the invocation operation the arguments and executor id found in the corresponding torq_hl.start_program op";
  let constructor =
      "mlir::syna::torq::createResolveInvocationArgumentsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];

}

def AssignAddresses :
    InterfacePass<"torq-assign-addresses",
         "mlir::FunctionOpInterface"> {
  let summary = "Assign an addresses to all ops that declare regions in memory (torq_hl::AllocOp, arith::ConstantOp, etc.)";
  let constructor =
      "mlir::syna::torq::createAssignAddressesPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "bufferization::BufferizationDialect"
    ];

}

def TileAndFuse :
    InterfacePass<"torq-tile-and-fuse",
         "mlir::FunctionOpInterface"> {
  let summary = "Tile operations and fuse producers into the tiling loop";
  let constructor =
      "mlir::syna::torq::createTileAndFusePass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
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

def TorqHlTile :
    InterfacePass<"torq-torqhl-tile",
         "mlir::FunctionOpInterface"> {
  let summary = "Tile all torq_hl operators so that the operands fit in LRAM";
  let constructor =
      "mlir::syna::torq::createTorqHlTilePass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "tensor::TensorDialect"
    ];

}

def Slicing :
           InterfacePass<"torq-slicing",
                         "mlir::FunctionOpInterface"> {
    let summary = "Slice all torq_hl operators so that the they can be parallelized";
    let constructor =
        "mlir::syna::torq::createSlicingPass()";
    
    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "bufferization::BufferizationDialect",
        "tensor::TensorDialect"
    ];
    
}

def AddDeallocation :
    InterfacePass<"torq-add-deallocation",
         "mlir::FunctionOpInterface"> {
  let summary = "Add LRAM buffer deallocation ops when done using a buffer";
  let constructor =
      "mlir::syna::torq::createAddDeallocationPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];

}

def Profiling:
    InterfacePass<"torq-profiling", "mlir::FunctionOpInterface"> {
        let summary = "Torq profiling pass";
        let constructor = "mlir::syna::torq::createProfilingPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect", "syna::torq_hw::TorqHWDialect"
        ];
}

def KernelSelection :
    InterfacePass<"torq-kernel-selection", "mlir::FunctionOpInterface"> {
        let summary = "Kernel selection";
        let constructor = "mlir::syna::torq::createKernelSelectionPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "bufferization::BufferizationDialect"
        ];
}

def FoldConvert :
    InterfacePass<"torq-fold-convert", "mlir::FunctionOpInterface"> {
        let summary = "Fold tensor conversion operations";
        let constructor = "mlir::syna::torq::createFoldConvertPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
        ];
}

#ifdef ENABLE_TORQ_GENERIC
def FoldPValueInits :
    InterfacePass<"torq-fold-pvalue-inits", "mlir::FunctionOpInterface"> {
        let summary = "Folds filling of P value initialization to arith::const ops";
        let constructor = "mlir::syna::torq::createFoldPValueInitsPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
        ];
}
#endif // ENABLE_TORQ_GENERIC

def ResolveAddresses :
    InterfacePass<"torq-resolve-addresses", "mlir::FunctionOpInterface"> {
        let summary = "Computes all the addresses used by torq_hw operators and set them as attributes";
        let constructor = "mlir::syna::torq::createResolveAddressesPass()";
        let dependentDialects = [
            "syna::torq_hw::TorqHWDialect"
        ];
}

def AssignObjectsIdentifiers :
    InterfacePass<"torq-assign-object-indentifiers", "mlir::FunctionOpInterface"> {
        let summary = "Annotates runtime objects with their unique identifiers ";
        let constructor = "mlir::syna::torq::createAssignObjectsIdentifiersPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "syna::torq_hw::TorqHWDialect"
        ];
}

def EncodeTensors :
    InterfacePass<"torq-encode-tensors", "mlir::FunctionOpInterface"> {
        let summary = "Encodes tensor data for torq_hl operations that require it";
        let constructor = "mlir::syna::torq::createEncodeTensorsPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
        ];
}

def ScalarsToTensors : 
    InterfacePass<"torq-scalars-to-tensors", "mlir::FunctionOpInterface"> {
        let summary = "Convert scalar values to tensors";
        let constructor = "mlir::syna::torq::createScalarsToTensorsPass()";
        let dependentDialects = [
            "tensor::TensorDialect"
        ];
}

def CompileTimeConstCompute : 
    InterfacePass<"torq-compile-time-const-compute", "mlir::FunctionOpInterface"> {
        let summary = "Compute values at compile time for operations marked as compile-time-const";
        let constructor = "mlir::syna::torq::createCompileTimeConstComputePass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "tensor::TensorDialect"
        ];
}

def MarkHostExecutor :
    InterfacePass<"torq-mark-host-executor",
                  "mlir::FunctionOpInterface"> {

    let summary = "Mark host executor";

    let constructor =
        "mlir::syna::torq::createMarkHostExecutorPass()";

    let dependentDialects = [
        "linalg::LinalgDialect"
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

#endif // IREE_CODEGEN_TORQ_PASSES
