// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

// Lowers a transpose conv with a stride on one spatial axis to one stride-1 phase conv.
void populateTransposeConvPhaseConvPatterns(MLIRContext *context, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
