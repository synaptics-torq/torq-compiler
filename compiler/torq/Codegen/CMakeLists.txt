# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(sources
    AddDeallocationPass.cpp
    AssignAddressesPass.cpp
    AssignObjectsIdentifiersPass.cpp
    AssignOperationsToCpuProgramsPass.cpp
    BufferizationUtils.cpp
    CheckCssStackSizePass.cpp
    CompileCpuProgramsPass.cpp
    CompileTimeConstComputePass.cpp
    EncodeTensorsPass.cpp
    FoldConvertPass.cpp
    KernelSelectionPass.cpp
    LinalgTilePass.cpp
    LowerArithConstantsPass.cpp
    LowerCallProgramToStartWaitPass.cpp
    MapBindingsPass.cpp
    MarkHostExecutorPass.cpp
    OutlineCpuProgramsPass.cpp
    OutlineNSSProgramsPass.cpp
    OutlineSliceProgramsPass.cpp
    OutlineUtils.cpp
    Passes.cpp
    Pool.cpp
    ProfilingPass.cpp
    ResolveAddressesPass.cpp
    ResolveInvocationArgumentsPass.cpp
    ScalarsToTensorsPass.cpp
    SlicingPass.cpp
    TORQLowerExecutableTargetPass.cpp
    TorqComprehensiveBufferizePass.cpp
    TorqHlTilePass.cpp
    TileAndFusePass.cpp
    UnrollLoopPass.cpp
    ValidPadToSamePadPass.cpp
    VirtualMemory.cpp
)

if(TORQ_ENABLE_TORQ_GENERIC)
  list(APPEND sources FoldPValueInitsPass.cpp)
endif()

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)

iree_cc_library(
  NAME
    Codegen
  HDRS
    PassesDetail.h
    Passes.h
    Passes.h.inc
  SRCS
    ${sources}
  DEPS
    ::PassesIncGen
    MLIRSupport
    iree::compiler::Codegen::Common::TransformDialectInterpreterPass
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::IR::HALDialect
    iree::compiler::Dialect::LinalgExt::IR
    iree::compiler::Dialect::LinalgExt::Transforms
    iree::compiler::Utils
    IREELinalgTransformDialectPasses
    iree::compiler::plugins::target::LLVMCPU::LLVMIRPasses
    MLIRFuncDialect
    MLIRFunctionInterfaces
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPass
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSCFUtils
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorToLLVM
    MLIRVectorToSCF
    MLIRVectorTransforms
    iree::target::torq::Utils
    iree::target::torq::Conversions::TosaToTorqHL
    iree::target::torq::Conversions::LinalgToTorqHL
    iree::target::torq::Conversions::TorqHLToTorqHW
    iree::target::torq::Transforms::Linalg::Transforms    
    iree::target::torq::Codegen::css_bootstrap::torq_kernel_riscv
)

add_subdirectory("css_bootstrap")
