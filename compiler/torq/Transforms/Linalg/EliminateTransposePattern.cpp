// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

enum class EliminateTransposeMode { Enable, Disable };

static llvm::cl::opt<EliminateTransposeMode> clEliminateTransposeMode(
    "torq-eliminate-transpose-mode",
    llvm::cl::desc("Choose how to handle data-preserving Transpose ops"),
    llvm::cl::values(
        clEnumValN(
            EliminateTransposeMode::Enable, "enable",
            "Replace linalg.transpose ops whose permutation preserves the flat "
            "(row-major) element order with the equivalent free reshape "
            "(tensor.collapse_shape/expand_shape). Value-preserving "
            "normalization; on by default"
        ),
        clEnumValN(
            EliminateTransposeMode::Disable, "disable",
            "Leave all linalg.transpose ops as is (data-preserving "
            "transposes keep their transpose kernel)"
        )
    ),
    llvm::cl::init(EliminateTransposeMode::Enable)
);

namespace {

/// True when `Transpose(x, perm)` produces the same flat (row-major) element
/// order as `x`, i.e. it is equivalent to `Reshape(x, permuted_shape)`.
///
/// Exact test: unit dims carry no data, so the transpose preserves the flat order iff it keeps the
/// non-unit input dims in their original relative order. O(rank), so no numel cap is needed.
bool isDataPreservingTranspose(ArrayRef<int64_t> inShape, ArrayRef<int64_t> perm) {
    int64_t prev = -1;
    for (int64_t p : perm) {
        if (inShape[p] == 1)
            continue;
        if (p < prev)
            return false;
        prev = p;
    }
    return true;
}

class EliminateTransposePattern : public OpRewritePattern<linalg::TransposeOp> {
  public:
    explicit EliminateTransposePattern(MLIRContext *ctx)
        : OpRewritePattern<linalg::TransposeOp>(ctx) {
        setDebugName("EliminateTransposePattern");
    }

    LogicalResult
    matchAndRewrite(linalg::TransposeOp transpose, PatternRewriter &rewriter) const override {
        auto inputTy = dyn_cast<RankedTensorType>(transpose.getInput().getType());
        auto resultTy = dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
        if (!inputTy || !resultTy)
            return rewriter.notifyMatchFailure(transpose, "expected ranked tensor operands");
        // Static shapes are required to prove data-preservation
        if (!inputTy.hasStaticShape() || !resultTy.hasStaticShape())
            return rewriter.notifyMatchFailure(transpose, "expected static shapes");
        if (inputTy.getRank() == 0)
            return rewriter.notifyMatchFailure(transpose, "expected non-scalar tensor");
        // Conservative: skip zero-sized tensors.
        if (inputTy.getNumElements() == 0)
            return rewriter.notifyMatchFailure(transpose, "expected non-empty tensor");
        if (!isDataPreservingTranspose(inputTy.getShape(), transpose.getPermutation()))
            return rewriter.notifyMatchFailure(
                transpose, "real layout transpose (OptimizeTransposeLayoutPass territory)"
            );

        rewriter.replaceOp(
            transpose, emitValuePreservingReshape(
                           rewriter, transpose.getLoc(), transpose.getInput(), inputTy, resultTy
                       )
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[torq-eliminate-transpose] folded data-preserving "
                            "transpose "
                         << inputTy << " -> " << resultTy << "\n"
        );
        return success();
    }
};

} // namespace

void populateEliminateTransposePatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    if (clEliminateTransposeMode != EliminateTransposeMode::Enable)
        return;
    patterns.add<EliminateTransposePattern>(ctx);
}

} // namespace mlir::syna::torq
