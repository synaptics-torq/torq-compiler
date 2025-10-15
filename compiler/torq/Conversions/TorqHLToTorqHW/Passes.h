// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Pass/Pass.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

std::unique_ptr<OperationPass<torq_hl::ProgramOp>> createConvertSliceProgramToTorqHwPass();
std::unique_ptr<OperationPass<torq_hl::ProgramOp>> createConvertNssProgramToTorqHwPass();

void registerTorqHLToTorqHWPasses();

} // namespace mlir::syna::torq
