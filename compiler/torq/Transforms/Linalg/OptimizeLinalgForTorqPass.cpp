// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

namespace {

// Rewrite a keepdims reduction, whose output maps index unit axes with constant
// zero (e.g. (d0,d1,d2,d3) -> (d0,d1,d2,0)), into one with projected-permutation
// output maps plus a tensor.expand_shape back to the keepdims shape. Only those
// output axes are dropped: the loops, the inputs and the other unit dims (such as
// a batch of 1) are left as they are.
void dropKeepdimsOutputAxes(IRRewriter &rewriter, linalg::GenericOp op) {
    int64_t numInputs = op.getNumDpsInputs();
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    SmallVector<std::optional<SmallVector<ReassociationIndices>>> reassocs;
    SmallVector<Type> resultTypes;
    for (auto [i, init] : llvm::enumerate(op.getOutputs())) {
        AffineMap &map = maps[numInputs + i];
        auto outType = cast<RankedTensorType>(init.getType());
        if (map.isProjectedPermutation()) {
            reassocs.push_back(std::nullopt);
            resultTypes.push_back(outType);
            continue;
        }
        SmallVector<int64_t> keptPos;
        SmallVector<AffineExpr> keptExprs;
        if (failed(splitKeepdimsOutputMap(map, outType, keptPos, keptExprs)))
            return;
        SmallVector<int64_t> reducedShape;
        for (int64_t p : keptPos)
            reducedShape.push_back(outType.getDimSize(p));
        map = AffineMap::get(map.getNumDims(), 0, keptExprs, op.getContext());
        reassocs.push_back(keepdimsReassociation(keptPos, outType.getRank()));
        resultTypes.push_back(RankedTensorType::get(reducedShape, outType.getElementType()));
    }

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    SmallVector<Value> inits;
    for (auto [i, init] : llvm::enumerate(op.getOutputs())) {
        Value newInit = init;
        if (reassocs[i])
            newInit = tensor::CollapseShapeOp::create(
                rewriter, loc, cast<RankedTensorType>(resultTypes[i]), init, *reassocs[i]
            );
        inits.push_back(newInit);
    }
    auto newOp = linalg::GenericOp::create(
        rewriter, loc, resultTypes, op.getInputs(), inits, maps, op.getIteratorTypesArray()
    );
    newOp->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

    SmallVector<Value> results;
    for (auto [i, result] : llvm::enumerate(newOp.getResults())) {
        Value newResult = result;
        if (reassocs[i])
            newResult = tensor::ExpandShapeOp::create(
                rewriter, loc, op.getResult(i).getType(), result, *reassocs[i]
            );
        results.push_back(newResult);
    }
    rewriter.replaceOp(op, results);
}

class OptimizeLinalgForTorqPass
    : public impl::OptimizeLinalgForTorqBase<OptimizeLinalgForTorqPass> {
  public:
    void runOnOperation() override {

        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        IRRewriter rewriter(ctx);

        auto buildPatterns = [&]() {
            RewritePatternSet patterns(ctx);

            // Convert linalg ops to more specific linalg ops that are easier to match for
            // torq patterns.
            populateCommonStandardizationPatterns(ctx, patterns);

            populateOptimizeConv1DPatterns(ctx, patterns);

            populateOptimizeMatmuOpPatterns(ctx, patterns);

            // Fold k=1/n=1 matmuls (a single-term contraction) into the elementwise
            // product of their operands, so a row-scale multiplication is not run as
            // a matmul kernel.
            populateFoldScalarMatmulPatterns(ctx, patterns);

            populateOptimizeElementwiseBinaryOpPatterns(ctx, patterns);

            populateOptimizePowPatterns(ctx, patterns);

            populateFuseDecomposedGeluPatterns(ctx, patterns);

            populateRaiseSoftmaxOpPatterns(ctx, patterns);

            populateRaiseDynamicQuantizeOpPatterns(ctx, patterns);

            populateRaiseIntMeanToPoolingPatterns(ctx, patterns);

            populateOptimizeSelectPatterns(ctx, patterns);

            populateDecomposeLinalgOpsPatterns(ctx, patterns);

            populateSpecializeTransposeOpPatterns(ctx, patterns);

            // Fold data-preserving transposes into free reshapes.
            populateEliminateTransposePatterns(ctx, patterns);

            linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);

            // A pad that appends a scalar constant along one axis is a concat
            // of its source and a constant-filled tensor.
            populatePadToConcatPatterns(ctx, patterns);

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

        // Raise compound operations such as softmax before reshaping their reductions.
        // Keep-dims reductions index unit output axes with constant zero.
        // Normalize those axes before tiling and reduction matching, both of
        // which require projected-permutation output maps.
        funcOp.walk([&](linalg::GenericOp op) {
            if (!op.hasPureTensorSemantics() || !op.getNumReductionLoops())
                return;
            auto maps = op.getIndexingMapsArray();
            auto outputMaps = llvm::drop_begin(maps, op.getNumDpsInputs());
            if (llvm::all_of(outputMaps, [](AffineMap map) {
                    return map.isProjectedPermutation();
                }))
                return;
            dropKeepdimsOutputAxes(rewriter, op);
        });

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

            // After the raise, which takes the constant-weight corrections: the ones left
            // correct a matmul of two activations, and run on the slices once split. The
            // elementwise patterns give the split's adds the broadcast inputs their kernel needs.
            RewritePatternSet zeroPointPatterns(ctx);
            populateSplitMatmulZeroPointCorrectionPatterns(ctx, zeroPointPatterns);
            populateOptimizeElementwiseBinaryOpPatterns(ctx, zeroPointPatterns);
            if (failed(applyPatternsGreedily(getOperation(), std::move(zeroPointPatterns)))) {
                return signalPassFailure();
            }
        }

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

        // Quantized maxpools must be in int8 before sharing their stages: a maxpool computed from
        // another one no longer reads the dequantize
        RewritePatternSet maxPoolPatterns(ctx);
        populateFoldMaxPoolQuantizationPatterns(ctx, maxPoolPatterns);
        auto frozenMaxPoolPatterns =
            FrozenRewritePatternSet(std::move(maxPoolPatterns), disabledPatterns, enabledPatterns);
        if (failed(applyPatternsGreedily(getOperation(), frozenMaxPoolPatterns))) {
            return signalPassFailure();
        }

        shareMaxPoolStages(funcOp, rewriter);

        // After the quantization folding (a split quantized maxpool would no longer match it) and
        // the stage sharing (a stride-4 maxpool can read a stride-1 one)
        RewritePatternSet splitMaxPoolPatterns(ctx);
        populateSplitStride4MaxPoolPatterns(ctx, splitMaxPoolPatterns);
        // Other strides, e.g. 3 or different along H and W
        populateSplitStridedMaxPoolPatterns(ctx, splitMaxPoolPatterns);
        // Horizontal 1D maxpools, including the 1D stages of the strided split, before the NHWC to
        // NCHW conversion so that the transposes compose with the layout ones
        populateSwapHorizontalMaxPoolPatterns(ctx, splitMaxPoolPatterns);
        // Then kernels larger than the HW limits (including the stride-2 stage of a stride-4
        // maxpool), after the stage sharing which may already reduce them
        populateDecomposeLargeMaxPoolPatterns(ctx, splitMaxPoolPatterns);
        auto frozenSplitMaxPoolPatterns = FrozenRewritePatternSet(
            std::move(splitMaxPoolPatterns), disabledPatterns, enabledPatterns
        );
        if (failed(applyPatternsGreedily(getOperation(), frozenSplitMaxPoolPatterns))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeLinalgForTorqPass() {
    return std::make_unique<OptimizeLinalgForTorqPass>();
}

} // namespace mlir::syna::torq
