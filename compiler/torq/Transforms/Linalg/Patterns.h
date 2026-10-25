// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::syna::torq {

void populateOptimizeConv1DPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateOptimizeElementwiseBinaryOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateOptimizeSelectPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateDecomposeLinalgOpsPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateSpecializeTransposeOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

// Value-preserving transpose normalization
void populateEliminateTransposePatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateOptimizeMatmuOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

// Fold k=1/n=1 matmuls (a single-term contraction) into the elementwise
// product of their operands.
void populateFoldScalarMatmulPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateRaiseSoftmaxOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateRaiseDynamicQuantizeOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateRaiseIntMeanToPoolingPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateRaiseMatMulIntegerOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateOptimizeArithElementwiseBinaryOpPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns
);

void populateOptimizePowPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateCastI32MulPatterns(FunctionOpInterface funcOp, IRRewriter &rewriter);

// Split max poolings with kernels larger than the HW limits in chains of smaller max poolings
void populateDecomposeLargeMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Compute max poolings from the result of another max pooling of the same input when their windows
// are nested, e.g. the 9x9 and 13x13 pools of a YOLO SPP block from its 5x5 pool
void shareMaxPoolStages(FunctionOpInterface funcOp, IRRewriter &rewriter);

void populateSwishActivationPatterns(MLIRContext *context, RewritePatternSet &patterns);
void populatePreCalcI8TablePatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateFuseReluClampWithTruncfPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateAbsorbDecomposedWzpCorrectionPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

// Converts to standardized representation of the many ops that may not be handled by
// canonicalization patterns, to make them easier to match for torq-specific patterns
void populateCommonStandardizationPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void convertNhwcOpToNchwOp(FunctionOpInterface funcOp);

} // namespace mlir::syna::torq
