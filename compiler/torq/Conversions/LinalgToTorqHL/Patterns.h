// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::syna::torq {

void populateLinalgToActPatterns(MLIRContext *context, RewritePatternSet &patterns);
void populateLinalgToAluPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPrePatternsLowPrio(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateArithToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns);

using ControlFoldingFn = std::function<bool(Operation *operation)>;

void populateTorqConstantFoldLinalgOperations(
    RewritePatternSet &patterns, const ControlFoldingFn &controlFn
);

void populateTensorToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
