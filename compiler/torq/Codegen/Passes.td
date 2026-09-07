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

def AnalyzeTensorSizes
    : Pass<"torq-analyze-tensor-sizes", "ModuleOp"> {

  let summary = "Analyze tensor sizes in the module";
  
  let constructor = "mlir::syna::torq::createAnalyzeTensorSizesPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect"
  ];
}

def ConvertTensorPadToLinalg
    : Pass<"torq-convert-tensor-pad-to-linalg", "func::FuncOp"> {

  let summary = "Convert tensor.pad ops to linalg.generic + tensor.insert_slice";

  let description = [{
    Convert tensor.pad ops to linalg.generic + tensor.insert_slice so that
    IREE's LLVMCPU backend can compile them in standalone CPU programs.
  }];

  let constructor = "mlir::syna::torq::createConvertTensorPadToLinalgPass()";

  let dependentDialects = [
    "linalg::LinalgDialect",
    "tensor::TensorDialect"
  ];
}

def DecomposeTensorConcat
    : Pass<"torq-decompose-tensor-concat", "func::FuncOp"> {

  let summary = "Decompose tensor.concat ops to insert_slice form";

  let description = [{
    Decompose tensor.concat ops to a chain of tensor.insert_slice ops so that
    downstream destination-passing-style conversion can tie the tensor chain to
    the dispatch output buffer.
  }];

  let constructor = "mlir::syna::torq::createDecomposeTensorConcatPass()";

  let dependentDialects = [
    "tensor::TensorDialect"
  ];
}

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
        "syna::torq_hw::TorqHWDialect",
        "mlir::tosa::TosaDialect"
    ];

}

def MapBindings
    : InterfacePass<"torq-map-bindings", "mlir::FunctionOpInterface"> {

    let summary = "Lower all the hal.interface.binding.subspan ops to torq_hl.map_binding ops";
    let constructor = "mlir::syna::torq::createMapBindingsPass()";

    let dependentDialects = ["iree_compiler::IREE::HAL::HALDialect",
                             "syna::torq_hl::TorqHLDialect",
                             "mlir::memref::MemRefDialect"];
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

def OutlineSlicePrograms :
    InterfacePass<"torq-outline-slice-programs",
         "mlir::FunctionOpInterface"> {
  let summary = "Wrap all operators that run on slices into slice programs";
  let constructor =
      "mlir::syna::torq::createOutlineSliceProgramsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];
}

def UnrollForallLoops :
    InterfacePass<"torq-unroll-forall-loops",
         "mlir::FunctionOpInterface"> {
  let summary = "Unroll scf.forall loops used for slice parallelism";
  let constructor =
      "mlir::syna::torq::createUnrollForallLoopsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];
}

def ScheduleSlicePrograms :
    InterfacePass<"torq-schedule-slice-programs",
         "mlir::FunctionOpInterface"> {
  let summary = "Assign executor IDs to slice start_program operations";
  let constructor =
      "mlir::syna::torq::createScheduleSliceProgramsPass()";

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

def AttachX86CpuObjects :
    Pass<"torq-attach-x86-cpu-objects",
         "::mlir::iree_compiler::IREE::HAL::ExecutableOp"> {
  let summary = "Attach a hal.executable.object to every x86_64 host-CPU variant";
  let description = [{
    For every `hal.executable.variant` whose target triple is `x86_64-*`, attach
    an `#hal.executable.object<{path = "libclang_rt.builtins-x86_64.a", data = ...}>`
    entry to its `objects` attribute. IREE's LLVMCPU plugin links that archive
    at the link stage, supplying soft-float helpers like `__truncsfbf2` that
    LLVM emits when lowering `int -> bf16` casts on CPUs without native bf16.

    The archive bytes are embedded inside the compiler binary at build time via
    `.incbin` (see `host_x86_objects.S`), so the pass needs neither an external
    archive name option nor a `--iree-hal-executable-object-search-path` flag;
    the archive travels with the binary.

    Intended to be invoked via IREE's
    `--iree-hal-preprocess-executables-with=builtin.module(torq-attach-x86-cpu-objects)`
    flag. The `builtin.module(...)` wrapper is a trigger marker; the inner
    pipeline is anchored at `hal.executable`.
  }];
  let constructor = "mlir::syna::torq::createAttachX86CpuObjectsPass()";

  let dependentDialects = [
    "iree_compiler::IREE::HAL::HALDialect"
  ];
}

def CompileSliceInvocations :
    InterfacePass<"torq-compile-slice-invocations",
         "mlir::FunctionOpInterface"> {
  let summary = "Compile slice invocation operations";
  let constructor = "mlir::syna::torq::createCompileSliceInvocationsPass()";
  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect"
  ];
}

def CompileNSSInvocations :
    InterfacePass<"torq-compile-nss-invocations", "mlir::FunctionOpInterface"> {
  let summary = "Compile all the NSS invocations";
  let constructor = "mlir::syna::torq::createCompileNSSInvocationsPass()";

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect",
    "syna::torq_hw::TorqHWDialect"
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

def LowerHostCopiesToNpu :
    InterfacePass<"torq-lower-host-copies-to-npu",
         "mlir::FunctionOpInterface"> {

  let summary = "Lower host-side memory copies to NPU operations";
  let constructor = "mlir::syna::torq::createLowerHostCopiesToNpuPass()";
  let options=[
    Option<"lramSize", "lram-size", "int64_t",
           /*default=*/"0",
           "Size of LRAM in bytes. Used to determine whether a copy can be done by the NPU or not">
  ];

  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect",
    "mlir::memref::MemRefDialect"
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

def SegmentNSSPrograms :
    InterfacePass<"torq-segment-nss-programs", "mlir::FunctionOpInterface"> {
  let summary = "Segment NSS programs into smaller blocks that fit the NSS block size";
  let constructor =
      "mlir::syna::torq::createSegmentNSSProgramsPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];

}

def AssignNSSProgramsAddresses :
    InterfacePass<"torq-assign-nss-programs-addresses",
         "mlir::FunctionOpInterface"> {
  let summary = "Measure NSS program sizes and assign their XRAM addresses";
  let description = [{
    Compiles each NSS program once to measure its real block sizes, shrinks the
    code sections, packs the programs in XRAM, and updates the code block loads with the assigned addresses.

    The measuring compile serializes the lowered torq_hw tasks, so the pass must
    run after ConvertNssProgramToTorqHw and ResolveAddresses. It runs immediately
    before CompileNSSInvocations.
  }];
  let constructor =
      "mlir::syna::torq::createAssignNSSProgramsAddressesPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "syna::torq_hw::TorqHWDialect"
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

def RecoverLramResidency :
    InterfacePass<"torq-recover-lram-residency",
         "mlir::FunctionOpInterface"> {
  let summary = "Reduce peak LRAM residency in functions whose address allocation would fail";
  let description = [{
    Runs immediately before `torq-assign-lram-addresses`, and only in the real
    pipeline: the tile-fit probe pipelines that TileAndFuse builds must not include
    it, because the probe only reports "fits" or "does not fit" and would therefore
    pick a tile that only fits thanks to the extra DMA these transforms cost.
  }];
  let constructor =
      "mlir::syna::torq::createRecoverLramResidencyPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "bufferization::BufferizationDialect",
        "mlir::memref::MemRefDialect"
    ];
}

def AssignLramAddresses :
    InterfacePass<"torq-assign-lram-addresses",
         "mlir::FunctionOpInterface"> {
  let summary = "Assign addresses to all ops that declare regions in LRAM";
  let constructor =
      "mlir::syna::torq::createAssignLramAddressesPass()";

  let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "bufferization::BufferizationDialect"
    ];

}

def AssignDtcmItcmXramAddresses :
    InterfacePass<"torq-assign-dtcm-itcm-xram-addresses",
         "mlir::FunctionOpInterface"> {
  let summary = "Assign addresses to all ops that declare regions in DTCM, ITCM, and XRAM";
  let constructor =
      "mlir::syna::torq::createAssignDtcmItcmXramAddressesPass()";

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

  let options = [
      Option<"sliceCount", "slice-count", "int64_t",
             /*default=*/"0",
             "Number of hardware slices available for slicing">
  ];

  let dependentDialects = [
        "linalg::LinalgDialect",
    ];

}

def PeelTileLoops :
    InterfacePass<"torq-peel-tile-loops",
         "mlir::FunctionOpInterface"> {
  let summary = "Peel iterations of tile loops, to make all shapes inside the loops static";
  let constructor =
      "mlir::syna::torq::createPeelTileLoopsPass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
    ];

}

def UnrollDynamicShapeLoop :
    InterfacePass<"torq-unroll-dynamic-shape-loop",
         "mlir::FunctionOpInterface"> {
  let summary = "Unroll for loops that have dynamic shapes in their body";
  let constructor =
      "mlir::syna::torq::createUnrollDynamicShapeLoopPass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
    ];

}

def ReplaceForLoopsWithMidIteration :
    InterfacePass<"torq-replace-for-loops-with-mid-iteration",
         "mlir::FunctionOpInterface"> {
  let summary = "Intended to run from inside the Tile and Fuse pass. Replace scf.for loops with their middle iteration.";
  let constructor =
      "mlir::syna::torq::createReplaceForLoopsWithMidIterationPass()";

  let dependentDialects = [
        "linalg::LinalgDialect",
    ];

}

def TileReductionForLram:
    InterfacePass<"torq-tile-reduction-for-lram",
         "mlir::FunctionOpInterface"> {

    let summary = "Tile reduction dimensions of linalg ops to fit in LRAM";

    let description = [{
        Tiles reduction dimensions of linalg.generic ops whose input tensors exceed LRAM capacity.
        This ensures that downstream TorqHL conversion only sees LRAM-sized inputs.
    }];

    let constructor =
        "mlir::syna::torq::createTileReductionForLramPass()";

    let dependentDialects = [
        "::mlir::linalg::LinalgDialect",
        "::mlir::scf::SCFDialect"
    ];
}

def TorqHLTileLargeInputs:
    InterfacePass<"torq-hl-tile-large-inputs",
                  "mlir::FunctionOpInterface"> {

    let summary = "Tile TorqHL ops whose input tensors exceed a size threshold";

    let description = [{
        After TorqHL conversion, tiles ops (conv, matmul, pool) whose any single
        input tensor element count exceeds --torq-hl-tile-input-size-threshold,
        using a fixed tile size per parallel domain.
    }];

    let constructor =
        "mlir::syna::torq::createTorqHLTileLargeInputsPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::scf::SCFDialect"
    ];
}

def SwapConvertExtractSlice:
    InterfacePass<"torq-swap-convert-extract-slice",
                  "mlir::FunctionOpInterface"> {

    let summary = "Swap extract_slice(convert) to convert(extract_slice)";

    let description = [{
        After tiling, a tile's extract_slice often reads from a torq_hl.convert
        that converted its whole source tensor up front. Slices the conversion's
        pre-conversion input instead, so only the region a tile actually reads is
        ever converted.
    }];

    let constructor =
        "mlir::syna::torq::createSwapConvertExtractSlicePass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "::mlir::tensor::TensorDialect"
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

def StripTorqExecutorAttr :
    InterfacePass<"torq-strip-executor-attr",
                  "mlir::FunctionOpInterface"> {
    let summary = "Strip torq-executor attributes from torq_hl operations";
    let constructor =
        "mlir::syna::torq::createStripTorqExecutorAttrPass()";

    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect"
    ];
}

def CoalesceElementwiseChains :
        InterfacePass<"torq-coalesce-elementwise-chains",
                      "mlir::FunctionOpInterface"> {
    let summary = "Coalesce pure-elementwise generic chains into one fuse group before slicing";

    let constructor =
        "mlir::syna::torq::createCoalesceElementwiseChainsPass()";

    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

def FuseTransposeIntoGroup :
        InterfacePass<"torq-fuse-transpose-into-group",
                      "mlir::FunctionOpInterface"> {
    let summary = "Fuse a producer layout transpose into the group it feeds, before slicing";

    let constructor =
        "mlir::syna::torq::createFuseTransposeIntoGroupPass()";

    let dependentDialects = [
        "linalg::LinalgDialect"
    ];
}

def LinalgSlicing :
           InterfacePass<"torq-linalg-slicing",
                         "mlir::FunctionOpInterface"> {
    let summary = "Slice pattern-fuse-groups so that they can be parallelized.";

    let constructor =
        "mlir::syna::torq::createLinalgSlicingPass()";

    let options = [
        Option<"sliceCount", "slice-count", "int64_t",
               /*default=*/"0",
               "Number of hardware slices available for slicing">
    ];

    let dependentDialects = [
        "linalg::LinalgDialect",
        "tensor::TensorDialect",
        "scf::SCFDialect"
    ];
}

def BufferizePeeledForallInPlace :
           InterfacePass<"torq-bufferize-peeled-forall-in-place",
                         "mlir::FunctionOpInterface"> {
    let summary = "Point a peeled slice-head forall's tensor.empty init at its output "
                  "destination so it bufferizes in place (drops a redundant copy).";

    let constructor =
        "mlir::syna::torq::createBufferizePeeledForallInPlacePass()";

    let dependentDialects = [
        "tensor::TensorDialect",
        "scf::SCFDialect"
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

def AnnotateDmaAndSliceCycles :
        InterfacePass<"torq-annotate-dma-and-slice-cycles", "mlir::FunctionOpInterface"> {
    let summary = "Annotate DMA/CDMA transfers and slice tasks with estimated cycles/bytes";
    let constructor = "mlir::syna::torq::createAnnotateDmaAndSliceCyclesPass()";
    let dependentDialects = [
        "syna::torq_hl::TorqHLDialect",
        "syna::torq_hw::TorqHWDialect"
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
        let options=[
          Option<"lramSize", "lram-size", "int64_t",
                 /*default=*/"0",
                 "Size of LRAM in bytes. Used to determine whether a copy can be done by the NPU or not">
        ];

        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
        ];
}

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

def EliminateRedundantLramLoads :
    InterfacePass<"torq-eliminate-redundant-lram-loads", "mlir::FunctionOpInterface"> {
        let summary = "Reuse LRAM buffers for repeated read-only loads within a region";
        let constructor = "mlir::syna::torq::createEliminateRedundantLramLoadsPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
        ];
}

def MergeEmptyTensorPairs :
    InterfacePass<"torq-merge-empty-tensor-pairs", "mlir::FunctionOpInterface"> {
        let summary = "Merge sibling tensor.empty pairs left behind by EliminateEmptyTensorsPass";
        let constructor = "mlir::syna::torq::createMergeEmptyTensorPairsPass()";
        let dependentDialects = [
            "tensor::TensorDialect"
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

def SplitConstants :
    InterfacePass<"torq-split-constants", "mlir::FunctionOpInterface"> {
  let summary = "Split large tensor constants at their extract_slice users";
  let constructor = "mlir::syna::torq::createSplitConstantsPass()";
  let dependentDialects = [
    "syna::torq_hl::TorqHLDialect",
    "mlir::memref::MemRefDialect"
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

def CompileTimeConstOutline : 
    InterfacePass<"torq-compile-time-const-outline", "mlir::FunctionOpInterface"> {
        let summary = "Outline operations marked as compile-time-const into separate functions";
        let constructor = "mlir::syna::torq::createCompileTimeConstOutlinePass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "tensor::TensorDialect"
        ];
}

def CompileTimeConstStubs :
    InterfacePass<"torq-compile-time-const-stubs", "mlir::FunctionOpInterface"> {
        let summary = "Replace operations marked as compile-time-const with fake values";
        let constructor = "mlir::syna::torq::createCompileTimeConstStubsPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect",
            "tensor::TensorDialect"
        ];
}

def CreateGlobals :
    Pass<"torq-create-globals", "ModuleOp"> {
        let summary = "Replace all memref.alloc used in a dispatch with memref.global objects";
        let constructor = "mlir::syna::torq::createCreateGlobalsPass()";
        let dependentDialects = [
            "syna::torq_hl::TorqHLDialect"
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

def ExecutorAssignment :
    Pass<"torq-executor-assignment", "func::FuncOp"> {

    let summary = "Assign executors to operations based on capability profile";

    let description = [{
        Reads an executor assignment map (generated by Python capability profiler)
        and marks each operation with the appropriate executor (NSS/CSS/Host).
    }];

    let constructor = "mlir::syna::torq::createExecutorAssignmentPass()";

    let dependentDialects = [
        "linalg::LinalgDialect",
        "syna::torq_hl::TorqHLDialect"
    ];

    let options = [
        Option<"executorMapFile", "executor-map", "std::string",
               /*default=*/"\"\"",
               "Path to JSON file with executor assignments from runtime profiling">
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
