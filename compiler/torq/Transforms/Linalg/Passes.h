// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConstantsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createGeneralizeLinalgNamedOpsPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDecomposeLinalgOpsPass();

} // namespace mlir::syna::torq
