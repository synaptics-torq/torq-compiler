// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

//===---------------------------------------------------------------------===//
// Torq type-conversion passes.
//===---------------------------------------------------------------------===//
std::unique_ptr<OperationPass<ModuleOp>> createTorqConvertAllDTypesPass();
std::unique_ptr<OperationPass<ModuleOp>> createTorqNarrowWeightConstPass();
//===---------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvertNhwcOpToNchwPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRaiseResizeNearestNeighborPass();

// `restoreRank` also folds collapse -> generic -> expand round trips, rebuilding the generic at
// the higher rank; only valid before dispatch formation.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFoldUnitExtentWrapperPass(bool restoreRank = false);

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCollapseUnrolledConcatPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeTransposeLayoutPass();

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateLinalgAffinitiesPass();

std::unique_ptr<OperationPass<ModuleOp>> createOutlineTorqDispatchesPass();

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateTiedOperandsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollStaticScfForLoopsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSinglePixelConvToMatmulPass();

std::unique_ptr<OperationPass<ModuleOp>> createTorqAnnotateTorqResourcesPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDecomposeConvToSpaceToDepthPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRemoveIsNanPass();

std::unique_ptr<Pass> createDecomposeConvsForHardwarePass();

//----------------------------------------------------------------------------//
// Registration
//----------------------------------------------------------------------------//

void registerLinalgForTorqPasses();
void registerConvertNhwcOpToNchwPasses();
void buildTorqTypeConversionPipeline(OpPassManager &passManager);

} // namespace mlir::syna::torq
