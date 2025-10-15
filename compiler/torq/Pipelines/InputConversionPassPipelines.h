// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::syna::torq {

//===---------------------------------------------------------------------===//
// TORQ pass pipelines
//===---------------------------------------------------------------------===//

void registerTosaTransformPassPipeline();

void registerTorchTransformPassPipeline();

/// Creates the pipeline used to proces the input from tosa to torq_hl operators
void buildTosaToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager);

void buildTorchToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager);

void buildLinalgToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager);

} // namespace mlir::syna::torq
