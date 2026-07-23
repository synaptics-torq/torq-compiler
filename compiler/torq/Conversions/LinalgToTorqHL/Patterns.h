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
void populateLinalgToTorqHLQConv2DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLQMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLQEWBinaryPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLQSigmoidPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLConv2DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLConv1DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLFCPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLMatmulPatterns(
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
void populateLinalgToTorqHLQMulDivPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLQPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateLinalgToTorqHLQReducePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateLinalgToTorqHLEWBinaryPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLReduceMeanPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLQuantizePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);
void populateLinalgToTorqHLPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateArithToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateTensorToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateTrigPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateSoftmaxPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateSigmoidPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateExpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateGeluPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateLinalgToTorqHLMulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateLinalgToTorqHLExtractPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
);

void populateLinalgToTorqHLExpandWeightsPatterns(MLIRContext *context, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
