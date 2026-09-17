// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

namespace {

class OptimizeLinalgForTorqPass
    : public impl::OptimizeLinalgForTorqBase<OptimizeLinalgForTorqPass> {
  public:
    void runOnOperation() override {

        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        auto buildPatterns = [&]() {
            RewritePatternSet patterns(ctx);

            // Convert linalg ops to more specific linalg ops that are easier to match for
            // torq patterns.
            populateCommonStandardizationPatterns(ctx, patterns);

            populateOptimizeConv1DPatterns(ctx, patterns);

            populateOptimizeMatmuOpPatterns(ctx, patterns);

            populateOptimizeElementwiseBinaryOpPatterns(ctx, patterns);

            populateOptimizePowPatterns(ctx, patterns);

            populateRaiseSoftmaxOpPatterns(ctx, patterns);

            populateRaiseDynamicQuantizeOpPatterns(ctx, patterns);

            populateOptimizeSelectPatterns(ctx, patterns);

            populateDecomposeLinalgOpsPatterns(ctx, patterns);

            populateSpecializeTransposeOpPatterns(ctx, patterns);

            linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);

            populateOptimizeArithElementwiseBinaryOpPatterns(ctx, patterns);
            // The fused fp32-clamp + truncf->bf16 generic is lowered to a single
            // torq_hl.act (clamp in f32, emit bf16) by ClampOpConversion::matchFusedClampTruncf.
            populateFuseReluClampWithTruncfPatterns(ctx, patterns);
            populateAbsorbDecomposedWzpCorrectionPatterns(ctx, patterns);

            // Configure disabled/enabled patterns based on pass options.
            return FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);
        };

        // RaiseMatMulInteger consumes the zero point the DynamicQuantizeLinear raise
        // emits, and both are linalg.generic patterns the greedy driver orders by
        // worklist position rather than by dependency: in one set the MatMulInteger
        // raise can be tried before the DQL raise fires and never retried, so it runs
        // in its own round, after this set is at fixpoint.
        if (failed(applyPatternsGreedily(getOperation(), buildPatterns()))) {
            return signalPassFailure();
        }

        // The raise cannot match without an i32 matmul, so models without one skip the
        // round entirely. When it runs, it contains only the raise: the other groups
        // are already at fixpoint, and re-running them could rewrite the freshly
        // raised correction chain, whose exact shape the LinalgToTorqHL mark and
        // materialize sides expect verbatim.
        bool hasI32Matmul = false;
        funcOp.walk([&](Operation *op) {
            if (hasI32Matmul)
                return WalkResult::skip();
            ShapedType resultTy;
            if (auto matmul = dyn_cast<linalg::MatmulOp>(op))
                resultTy = dyn_cast<ShapedType>(matmul.getResult(0).getType());
            else if (auto batchMatmul = dyn_cast<linalg::BatchMatmulOp>(op))
                resultTy = dyn_cast<ShapedType>(batchMatmul.getResult(0).getType());
            if (resultTy && resultTy.getElementType().isInteger(32))
                hasI32Matmul = true;
            return WalkResult::advance();
        });
        if (hasI32Matmul) {
            RewritePatternSet raisePatterns(ctx);
            populateRaiseMatMulIntegerOpPatterns(ctx, raisePatterns);
            auto frozenRaisePatterns = FrozenRewritePatternSet(
                std::move(raisePatterns), disabledPatterns, enabledPatterns
            );
            if (failed(applyPatternsGreedily(getOperation(), frozenRaisePatterns))) {
                return signalPassFailure();
            }
        }

        IRRewriter rewriter(ctx);
        populateCastI32MulPatterns(funcOp, rewriter);

        // NOTE: Swish pattern optimization should happen in a separate pass
        // after the above castI32MulPatterns have been applied.
        RewritePatternSet SwishPatterns(ctx);
        populatePreCalcI8TablePatterns(ctx, SwishPatterns);
        populateSwishActivationPatterns(ctx, SwishPatterns);
        auto frozenSwishPatterns =
            FrozenRewritePatternSet(std::move(SwishPatterns), disabledPatterns, enabledPatterns);
        if (failed(applyPatternsGreedily(getOperation(), frozenSwishPatterns))) {
            return signalPassFailure();
        }

        // NOTE: Mul fusion should happen in a separate pass after the swish
        // pattern optimization above.
        // Run mul fusion in a separate round so that the swish pattern
        // (which needs the mul op to be a separate generic) fires first.
        RewritePatternSet fusionPatterns(ctx);
        auto frozenFusionPatterns =
            FrozenRewritePatternSet(std::move(fusionPatterns), disabledPatterns, enabledPatterns);
        if (failed(applyPatternsGreedily(getOperation(), frozenFusionPatterns))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass() {
    return std::make_unique<OptimizeLinalgForTorqPass>();
}

} // namespace mlir::syna::torq
