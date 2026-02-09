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
std::unique_ptr<OperationPass<ModuleOp>> createTorqDemoteF32ToBF16Pass();
std::unique_ptr<OperationPass<ModuleOp>> createTorqConvertF16ToBF16Pass();
std::unique_ptr<OperationPass<ModuleOp>> createTorqDemoteI64ToI32Pass();
//===---------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConstantsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createGeneralizeLinalgNamedOpsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass();

//----------------------------------------------------------------------------//
// Registration
//----------------------------------------------------------------------------//

void registerOptimizeLinalgForTorqPasses();
void registerTorqTypeConversionPasses();
void buildTorqTypeConversionPipeline(OpPassManager &passManager);

} // namespace mlir::syna::torq
