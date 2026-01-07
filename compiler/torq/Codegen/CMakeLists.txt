# Copyright 2024 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(sources
    AddDeallocationPass.cpp
    AssignAddressesPass.cpp
    KernelSelectionPass.cpp
    LowerArithConstantsPass.cpp
    MapBindingsPass.cpp
    LowerCallProgramToStartWaitPass.cpp
    OutlineSliceProgramsPass.cpp
    AssignOperationsToCpuProgramsPass.cpp
    OutlineCpuProgramsPass.cpp
    OutlineUtils.cpp
    CompileCpuProgramsPass.cpp
    Passes.cpp
    ProfilingPass.cpp
    OutlineNSSProgramsPass.cpp
    SlicingPass.cpp
    TorqComprehensiveBufferizePass.cpp
    TORQLowerExecutableTargetPass.cpp
    TorqHlTilePass.cpp
    ValidPadToSamePadPass.cpp
    ResolveAddressesPass.cpp
    TileAndFusePass.cpp
    ResolveInvocationArgumentsPass.cpp
    AssignObjectsIdentifiersPass.cpp
    BufferizationUtils.cpp
    Pool.cpp
    VirtualMemory.cpp
    CheckCssStackSizePass.cpp
    EncodeTensorsPass.cpp
    ScalarsToTensorsPass.cpp
    FoldConvertPass.cpp
    CompileTimeConstComputePass.cpp
    MarkHostExecutorPass.cpp
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
