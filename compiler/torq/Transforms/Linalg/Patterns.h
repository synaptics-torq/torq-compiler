// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::linalg {
class Conv1DNcwFcwOp;
}

namespace mlir::syna::torq {

void populateOptimizeConv1DPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

// Shared claim predicate for the conv1d GEMM form (dilation 1, all-zero init,
// no bias input; a single output point whose window is the whole input, or a
// few output points of a stride > 2 conv with batch 1):
// DecomposeConvsForHardwarePass lowers it to linalg.matmul
// (Conv1DtoMatmulPattern).  The conv1d->conv2d promotion must bail exactly
// where this pattern claims, no wider and no narrower: a conv it does not
// claim (say, a bias folded into the init) must still be promoted, or nothing
// lowers it.
bool isConv1dMatmulCandidate(linalg::Conv1DNcwFcwOp op);

// Registers the conv1d->conv2d promotion patterns for strides in
// [minStride, maxStride]; see the pattern for what the range means.
void populateConv1dPromotionPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, int64_t minStride, int64_t maxStride
);

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

// Rewrite a tensor.pad that appends a scalar constant along one axis into a
// tensor.concat of the source and a constant-filled tensor.
void populatePadToConcatPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateRaiseSoftmaxOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

void populateRaiseDynamicQuantizeOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateRaiseIntMeanToPoolingPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateRaiseMatMulIntegerOpPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateOptimizeArithElementwiseBinaryOpPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns
);

void populateOptimizePowPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Replace a tanh GELU decomposed into one elementwise generic per step with one generic that
// GeluOpPattern lowers
void populateFuseDecomposedGeluPatterns(MLIRContext *context, RewritePatternSet &patterns);

void populateCastI32MulPatterns(FunctionOpInterface funcOp, IRRewriter &rewriter);

// Replace quantize(maxpool(pad(dequantize(x)))) with an int8 maxpool on x when the quantize undoes
// the dequantize
void populateFoldMaxPoolQuantizationPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Split max poolings with kernels larger than the HW limits in chains of smaller max poolings
void populateDecomposeLargeMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Split stride-4 max poolings in a stride-2 max pooling and a 1x1 stride-2 max pooling
void populateSplitStride4MaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Split max poolings with other strides (e.g. 3, or different along H and W) in a stride-1 max
// pooling and exact 1D max poolings
void populateSplitStridedMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns);

// Lower horizontal 1D max poolings with the vertical kernel between two transposes swapping H and W
void populateSwapHorizontalMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns);

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
