// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

//===---------------------------------------------------------------------===//
// TORQ passes
//===---------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>> createValidToSamePadPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCheckCssStackSizePass();

std::unique_ptr<OperationPass<ModuleOp>> createTORQLowerExecutableTargetPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMapBindingsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerArithConstantsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignAddressesPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineSliceProgramsPass();

std::unique_ptr<OperationPass<ModuleOp>>
createAssignOperationsToCpuProgramsPass(bool disableCss = false, bool disableHost = false);

std::unique_ptr<OperationPass<ModuleOp>> createOutlineCpuProgramsPass();

std::unique_ptr<OperationPass<ModuleOp>> createCompileCpuProgramsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerCallProgramToStartWaitPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineNSSProgramsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHlTilePass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSlicingPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAddDeallocationPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createProfilingPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createKernelSelectionPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldPValueInitsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createResolveAddressesPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileAndFusePass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createResolveInvocationArgumentsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignObjectsIdentifiersPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConvertPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createEncodeTensorsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createScalarsToTensorsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstComputePass();

void addTorqComprehensiveBufferizePasses(
    OpPassManager &funcPassManager,
    std::optional<bufferization::BufferizationOptions::AllocationFn> allocationFn,
    std::optional<bufferization::BufferizationOptions::MemCpyFn> memCpyFn,
    std::optional<bufferization::BufferizationOptions::DefaultMemorySpaceFn> memSpaceFn
);

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkHostExecutorPass();

//----------------------------------------------------------------------------//
// Registration
//----------------------------------------------------------------------------//

void registerCodegenTORQPasses();

} // namespace mlir::syna::torq
