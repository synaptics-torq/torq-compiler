// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::syna::torq {

void populateOptimizeElementwiseBinaryOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
