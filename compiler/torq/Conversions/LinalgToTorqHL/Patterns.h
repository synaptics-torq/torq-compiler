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
void populateLinalgToTorqHLConv1DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgConv2DToTorqHLConv1DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLConv2DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLConv2DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLFCPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLClampPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPrePatternsLowPrio(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLEWBinaryPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLReduceMeanPatternsBeforeMarking(
    MLIRContext *context, RewritePatternSet &patterns
);
void populateLinalgToTorqHLReduceMeanPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateArithToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateTensorToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateSoftmaxPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateSigmoidPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateExpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateLinalgToTorqHLMulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateLinalgToTorqHLExtractPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

} // namespace mlir::syna::torq
