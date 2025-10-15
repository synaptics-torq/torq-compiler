// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir::syna::torq {

/// Populates passes needed to lower linalg/arith/math ops to TORQ ops via
/// the structured ops path.
void buildTORQCodegenPassPipeline(OpPassManager &variantPassManager);

} // namespace mlir::syna::torq
