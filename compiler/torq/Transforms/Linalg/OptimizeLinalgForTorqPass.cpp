// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

class OptimizeLinalgForTorqPass : public OptimizeLinalgForTorqBase<OptimizeLinalgForTorqPass> {
  public:
    void runOnOperation() override {

        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();
        RewritePatternSet patterns(ctx);

        populateOptimizeElementwiseBinaryOpPatterns(ctx, patterns);
        populateDecomposeLinalgOpsPatterns(ctx, patterns);
        populateSpecializeTransposeOpPatterns(ctx, patterns);
        linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
        // Configure disabled/enabled patterns based on pass options.
        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass() {
    return std::make_unique<OptimizeLinalgForTorqPass>();
}

} // namespace mlir::syna::torq
