// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

namespace {

static bool hasPermutation(linalg::TransposeOp transposeOp, ArrayRef<int64_t> expected) {
    return transposeOp.getPermutation() == expected;
}

static bool hasStaticShape(RankedTensorType type, ArrayRef<int64_t> expected) {
    if (!type || !type.hasStaticShape() || type.getRank() != static_cast<int64_t>(expected.size()))
        return false;

    return type.getShape() == expected;
}

static bool hasConv1DExpandReassociation(tensor::ExpandShapeOp expandOp) {
    auto reassociation = expandOp.getReassociationIndices();
    return reassociation.size() == 2 && reassociation[0].size() == 2 && reassociation[0][0] == 0 &&
           reassociation[0][1] == 1 && reassociation[1].size() == 1 && reassociation[1][0] == 2;
}

// Lower a `linalg.matmul` produced by Conv1D-as-matmul to torq_hl.fully_connected,
// folding the trailing per-channel bias and any truncf/clamp or
// requantize/clamp chain into the fully_connected op. This eliminates dangling
// elementwise linalg.generic ops that have no NSS/slice lowering, and routes
// pointwise Conv1D layers to NPU instead of leaving them on Host.
//
// Pattern is element-type agnostic: it handles both the bf16 path and the
// integer (i8/i32) requantize path produced by quantized ONNX models. The
// post-matmul layout chain (transpose -> optional expand_shape) is shared
// between the two; only the bias/scale folding differs.
//
// Layout assumed (created by Conv1DNcwFcwToLinalgMatmulPattern):
//   %im2col      : [Ow, K]       (K = C * Kw)
//   %filter_T    : [K, F]        (linalg.transpose of [F, K]; canonicalization
//                                 may have folded this into a constant, in
//                                 which case the matmul RHS already lives in
//                                 [K, F] and we re-introduce the transpose)
//   %matmul      : [Ow, F]       (element type matches the conv1d output)
//   %trans       = linalg.transpose %matmul -> [F, Ow]
//   %expand      = tensor.expand_shape %trans -> [1, F, Ow]   (optional)
//
// followed by either:
//   bf16 path:
//     %add   = linalg.generic addf(%expand, %bias_f32) -> [1, F, Ow] f32
//     %trunc = linalg.generic truncf(%add)             -> [1, F, Ow] bf16
//   integer path:
//     %add   = linalg.generic addi(%expand, %bias_i32) -> [1, F, Ow] i32
//     %req   = linalg.generic apply_scale + clamp + trunci -> [1, F, Ow] i8
//
// We emit:
//   %fc          = torq_hl.fully_connected(%im2col, %weights[F,K], %bias)
//                                                                 -> [Ow, F]
//   %trans2      = linalg.transpose %fc                            -> [F, Ow]
//   %expand2     = tensor.expand_shape %trans2                     -> [1, F, Ow]
//
// where %bias is `[F]` for the bf16 path and an interleaved `[2*F]`
// bias/scale buffer for the integer path. The bottom-most folded op (truncf
// or requantize) is replaced with %expand2; the matmul, post-matmul
// transpose and expand_shape become dead and are DCE'd.
struct Conv1DMatmulToTorqHlFCPattern : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const int _shift8b;
    const int _shift16b;
    const bool _markFuseGroups;

  public:
    Conv1DMatmulToTorqHlFCPattern(
        MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups
    )
        // Higher benefit than Conv2DMatmulOpConversion's default (1) so we win
        // when the matmul is conv1d-derived.
        : OpRewritePattern(context, /*benefit=*/2), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(matmulOp)) {
            return rewriter.notifyMatchFailure(matmulOp, "Already marked");
        }

        auto loc = matmulOp.getLoc();

        Value im2col = matmulOp.getInputs()[0];
        Value transposedFilter = matmulOp.getInputs()[1];
        Value matmulResult = matmulOp.getResult(0);

        auto matmulType = dyn_cast<RankedTensorType>(matmulResult.getType());
        if (!matmulType || !matmulType.hasStaticShape() || matmulType.getRank() != 2) {
            return rewriter.notifyMatchFailure(matmulOp, "Expected static 2D matmul output");
        }
        int64_t Ow = matmulType.getShape()[0];
        int64_t F = matmulType.getShape()[1];

        auto im2colType = dyn_cast<RankedTensorType>(im2col.getType());
        auto transposedFilterType = dyn_cast<RankedTensorType>(transposedFilter.getType());
        if (!im2colType || !transposedFilterType || !im2colType.hasStaticShape() ||
            !transposedFilterType.hasStaticShape() || im2colType.getRank() != 2 ||
            transposedFilterType.getRank() != 2) {
            return rewriter.notifyMatchFailure(matmulOp, "Expected static 2D matmul inputs");
        }
        int64_t K = im2colType.getShape()[1];

        // Recover [F, K] weights. Canonicalization may fold the introduced
        // transpose into a constant, leaving the matmul RHS already in [K, F].
        auto filterTransposeOp = transposedFilter.getDefiningOp<linalg::TransposeOp>();
        Value weights = transposedFilter;
        bool transposeWeightsForFC = true;
        if (filterTransposeOp) {
            if (!hasPermutation(filterTransposeOp, {1, 0})) {
                return rewriter.notifyMatchFailure(
                    matmulOp, "Expected 2D linalg.transpose feeding matmul rhs"
                );
            }
            weights = filterTransposeOp.getInput();
            transposeWeightsForFC = false;
        }

        auto weightsType = dyn_cast<RankedTensorType>(weights.getType());
        if (transposeWeightsForFC) {
            if (!hasStaticShape(weightsType, {K, F})) {
                return rewriter.notifyMatchFailure(
                    matmulOp, "Unexpected transposed weights layout"
                );
            }
        }
        else if (!hasStaticShape(weightsType, {F, K})) {
            return rewriter.notifyMatchFailure(matmulOp, "Unexpected weights layout");
        }

        // Walk forward through the post-matmul layout chain:
        //   matmul[Ow,F] -> linalg.transpose[F,Ow] -> tensor.expand_shape[N,F,Ow]
        Value forwardWalk = matmulResult;
        auto matmulTransposeOp = getSingleUser<linalg::TransposeOp>(forwardWalk);
        if (!matmulTransposeOp || !hasPermutation(matmulTransposeOp, {1, 0})) {
            return rewriter.notifyMatchFailure(
                matmulOp, "Expected post-matmul 2D linalg.transpose"
            );
        }
        forwardWalk = matmulTransposeOp.getResults()[0];

        // An expand_shape may be present between the transpose and elementwise
        // chain. After it, F lives at dim 1 of [N, F, Ow]; without it, F lives
        // at dim 0 of [F, Ow].
        bool hasExpandShape = false;
        if (auto expandOp = getSingleUser<tensor::ExpandShapeOp>(forwardWalk)) {
            if (!hasConv1DExpandReassociation(expandOp)) {
                return rewriter.notifyMatchFailure(
                    matmulOp, "Unexpected post-transpose expand_shape reassociation"
                );
            }
            forwardWalk = expandOp.getResult();
            hasExpandShape = true;
        }

        auto preFoldType = dyn_cast<RankedTensorType>(forwardWalk.getType());
        if (!preFoldType || !preFoldType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(
                matmulOp, "non-static ranked tensor after layout walk"
            );
        }
        if (hasExpandShape) {
            auto preFoldShape = preFoldType.getShape();
            if (preFoldType.getRank() != 3 || preFoldShape[1] != F || preFoldShape[2] != Ow) {
                return rewriter.notifyMatchFailure(matmulOp, "Unexpected expand_shape layout");
            }
        }

        const int channelDim = hasExpandShape ? 1 : 0;

        // Capture per-channel bias along F (biasVec starts at all-zeros).
        bool isInt = preFoldType.getElementType().isInteger();
        VectorIntOrFloat biasVec(F, isInt);
        Value foldStart = forwardWalk;
        while (foldForwardPerChannelAdd(forwardWalk, channelDim, biasVec)) {
            // Keep folding chained bias adds into biasVec.
        }

        // Absorb a trailing truncf and any clamp. For float anchors, scale data
        // may remain empty even when `forwardWalk` advances past truncf/clamp.
        ScaleClampInfo scInfo = foldForwardScaleClamp(forwardWalk, F, _shift8b, _shift16b);

        // If neither bias nor truncf/clamp was absorbed, there is nothing to
        // fuse, so fall through to the default Conv2DMatmulOpConversion.
        if (forwardWalk == foldStart) {
            return rewriter.notifyMatchFailure(matmulOp, "No bias or truncf to fuse");
        }
        if (isInt && !scInfo) {
            return rewriter.notifyMatchFailure(
                matmulOp, "Expected integer scale/clamp after Conv1D matmul"
            );
        }

        Type fcElemType = cast<RankedTensorType>(forwardWalk.getType()).getElementType();

        if (_markFuseGroups) {
            markFuseGroupBackward(
                forwardWalk, {im2col, weights}, rewriter,
                matmulOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Mark filter constants as compile-time const so they can be packed
        // into the static section.
        if (transposeWeightsForFC) {
            weights = transposeValue(weights, SmallVector<int64_t>{1, 0}, loc, rewriter);
        }
        if (auto wDef = weights.getDefiningOp()) {
            setCompileTimeConstAttr(wDef);
        }

        // Build the torq_hl.fully_connected:
        //   input   : [Ow, K]
        //   weights : [F,  K]
        //   bias    : [F] for float, interleaved bias/scale [2*F] for integer
        //   output  : [Ow, F] in fcElemType
        auto fcOutType = RankedTensorType::get({Ow, F}, fcElemType);
        Value fcInit = createInitTensor(matmulOp, rewriter, fcOutType);

        Value biasValue =
            isInt ? createConst(interleave(biasVec.ints, scInfo.scaleNpu), rewriter, loc)
                  : createConst(biasVec.floats, rewriter, loc);

        // scInfo carries the default output attributes plus any folded
        // truncf/clamp data, so the FC op can consume it unconditionally.
        auto fcOp = torq_hl::FullyConnectedOp::create(
            rewriter, loc, fcOutType, fcInit, /*inputZp=*/0, /*weightZp=*/0, scInfo.zp, scInfo.min,
            scInfo.max, scInfo.scaleShift, torq_hl::VectorizationModeEnum::None, weights, biasValue,
            im2col
        );
        Value finalResult = fcOp.getResult(0); // [Ow, F]

        // Rebuild the original output layout.
        // First transpose [Ow, F] -> [F, Ow].
        finalResult = transposeValue(finalResult, SmallVector<int64_t>{1, 0}, loc, rewriter);

        // Optionally re-expand to [N, F, Ow] when the upstream chain had one.
        if (hasExpandShape) {
            int64_t N = preFoldType.getShape()[0];
            auto expandedTy = RankedTensorType::get({N, F, Ow}, fcElemType);
            finalResult = tensor::ExpandShapeOp::create(
                              rewriter, loc, expandedTy, finalResult,
                              ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            ).getResult();
        }

        // Replace the bottom-most fused op (truncf or bias add) with the new
        // FC + transpose + expand chain. The matmul, the post-matmul transpose
        // and the expand_shape become dead and are DCE'd by the canonicalizer
        // that runs at the end of the pre-conversion pass.
        rewriter.replaceOp(forwardWalk.getDefiningOp(), finalResult);

        return success();
    }
};

} // namespace

void populateLinalgToTorqHLConv1DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv1DMatmulToTorqHlFCPattern>(context, 28, 12, markFuseGroups);
}

} // namespace mlir::syna::torq
