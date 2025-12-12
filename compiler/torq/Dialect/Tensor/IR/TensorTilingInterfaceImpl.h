// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/DialectRegistry.h"

namespace mlir::syna::torq::tensor {

/// Registers additional external models for Tiling interface for tensor ops.
/// Currently, it registers:
///
/// * TilingInterface for `tensor.collapse_shape`, and `tensor.expand_shape`.
///
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace mlir::syna::torq::tensor
