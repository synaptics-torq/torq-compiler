// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::syna::torq {

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLPreConversionPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTensorToLinalgPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkPatternsForTileAndFusePass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLConversionPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTensorToTorqHLConversionPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>> createArithToTorqHLConversionPass();

#ifdef ENABLE_TORQ_GENERIC
std::unique_ptr<InterfacePass<FunctionOpInterface>> createSpecializeLinalgGenericOpPass();
#endif // ENABLE_TORQ_GENERIC

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgToTorqHLGenericPass(bool specializeConstantComputations = false);

void registerLinalgToTorqHLPasses();

} // namespace mlir::syna::torq
