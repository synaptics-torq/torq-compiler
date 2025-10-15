// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <memory>

namespace mlir::syna::torq_hl {

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLOptimizeSegmentationPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSimplifyGenericPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMergeBiasScaleTensorsPass();

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFormDispatchRegionsPass(bool disableDispatchFusion = false);

//----------------------------------------------------------------------------//
// Registration
//----------------------------------------------------------------------------//

void registerTorqHLPasses();

} // namespace mlir::syna::torq_hl
