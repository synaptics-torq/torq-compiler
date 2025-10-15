// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

void populateTOSAToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
