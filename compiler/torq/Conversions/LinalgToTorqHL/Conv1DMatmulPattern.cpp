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

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-conv1d-matmul-pattern"
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

    Value replaceWithTorqMatmul(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const {
        // Fallback rewrite when no fusible conv/fc chain is present:
        // emit plain torq_hl.matmul with neutral bias/scale parameters.
        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Falling back to torq_hl.matmul for: " << srcOp << "\n"
        );
        auto outTy = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(outTy.getElementType());

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        Value biasScale = createI32Const(rewriter, srcOp, interleave(bias, scale));

        auto matMulOp = torq_hl::MatMulOp::create(
            rewriter, srcOp->getLoc(), outTy, createInitTensor(srcOp, rewriter, outTy), 0, outMin,
            outMax, 0, biasScale, srcOp.getOperand(0), srcOp.getOperand(1)
        );

        // linalg::MatmulOp is accumulating, but torq_hl::MatmulOp is not, so we do the addition
        // explicitly.
        FailureOr<Value> resultVal =
            addInitToResult(srcOp.getOutputs().front(), matMulOp.getResult(0), rewriter);
        assert(succeeded(resultVal) && "failed to add init value");

        rewriter.replaceOp(srcOp, *resultVal);

        return *resultVal;
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Attempting match on: " << matmulOp << "\n");

        if (_markFuseGroups && isMarkedFuseGroup(matmulOp)) {
            LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Skipping already-marked fuse group\n");
            return rewriter.notifyMatchFailure(matmulOp, "Already marked");
        }

        // QDQ quantized chains are owned by QMatmulToFCConvert. This pattern has
        // higher benefit and is tried first on every linalg.matmul, so without
        // this check it can spuriously match a quantized FC matmul and
        // permanently steal it from QMatmulToFCConvert.
        if (isQuantizedMatmulChain(matmulOp)) {
            return rewriter.notifyMatchFailure(
                matmulOp, "quantized matmul handled by QMatmulToFCConvert"
            );
        }

        auto loc = matmulOp.getLoc();

        Value im2col = matmulOp.getInputs()[0];
        Value weights = matmulOp.getInputs()[1];
        Value matmulResult = matmulOp.getResult(0);

        auto matmulType = dyn_cast<RankedTensorType>(matmulResult.getType());
        if (!matmulType || !matmulType.hasStaticShape() || matmulType.getRank() != 2) {
            return rewriter.notifyMatchFailure(matmulOp, "Expected static 2D matmul output");
        }
        int64_t Ow = matmulType.getShape()[0];
        int64_t F = matmulType.getShape()[1];
        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Matmul output shape: [" << Ow << ", " << F << "]\n"
        );

        auto im2colType = dyn_cast<RankedTensorType>(im2col.getType());
        auto weightsType = dyn_cast<RankedTensorType>(weights.getType());
        if (!im2colType || !weightsType || !im2colType.hasStaticShape() ||
            !weightsType.hasStaticShape() || im2colType.getRank() != 2 ||
            weightsType.getRank() != 2) {
            return rewriter.notifyMatchFailure(matmulOp, "Expected static 2D matmul inputs");
        }
        int64_t K = im2colType.getShape()[1];

        // Recover [K, F] weights. Canonicalization folds the filter transpose into the
        // constant, so the matmul RHS lands directly as [K, F].

        if (!hasStaticShape(weightsType, {K, F})) {
            return rewriter.notifyMatchFailure(matmulOp, "Unexpected weights layout");
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Weights shape: [" << K << ", " << F
                         << "], im2col shape: [" << Ow << ", " << K << "]\n"
        );

        // Build fusion plan and compute bias/scale using PatternUtils helpers
        auto output = matmulOp.getResult(0);
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Building fusion plan\n");
        FailureOr<FusionPlan> fusionPlanOr = buildFusionPlanAndRebindOutput(output);
        if (failed(fusionPlanOr) || !fusionPlanOr->isFusable()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[" DEBUG_TYPE "] Fusion plan not fusable"
                             << (failed(fusionPlanOr) ? " (failed to build)" : "") << "\n"
            );
            if (_markFuseGroups) {
                // Discovery-only mode: mark the chain and defer material rewrite.
                LLVM_DEBUG(
                    llvm::dbgs() << "[" DEBUG_TYPE "] Marking fuse group (non-fusable path)\n"
                );
                markFuseGroupBackward(
                    output, {im2col, weights}, rewriter,
                    matmulOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                );
                return success();
            }
            replaceWithTorqMatmul(matmulOp, rewriter);
            return success();
        }
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Fusion plan built successfully\n");

        RankedTensorType finalType = cast<RankedTensorType>(output.getType());
        // Compute per-channel bias value and scale/clamp info from fused chain
        std::optional<Value> optionalWeightZpV;
        ScaleClampInfo scInfo = getDefaultScaleClampInfo(finalType, output.getDefiningOp());

        FailureOr<Value> biasV = computeBiasAndRescaleInfo(
            *fusionPlanOr, fusionPlanOr->channelDim.value(), optionalWeightZpV, scInfo
        );
        if (failed(biasV)) {
            LLVM_DEBUG({ llvm::dbgs() << "computeBias: no bias found, setting zero bias\n"; });
            biasV = getDefaultBiasScale(matmulOp, finalType, rewriter);
            fusionPlanOr->channelDim = 0; // default to batch dim for zero bias
        }

        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Scale/clamp info: zp=" << scInfo.zp
                         << " min=" << scInfo.min << " max=" << scInfo.max
                         << " shift=" << scInfo.scaleShift << "\n"
        );

        Type fcElemType = finalType.getElementType();

        if (_markFuseGroups) {
            LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Marking fuse group (fusable path)\n");
            markFuseGroupBackward(
                output, {im2col, weights}, rewriter,
                matmulOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        for (auto &op : llvm::reverse(fusionPlanOr->opsToFuse)) {
            if (op->use_empty()) {
                // Remove now-dead ops from fused tail after bias/scale extraction.
                rewriter.eraseOp(op);
            }
        }
        // Mark filter constants as compile-time const so they can be packed
        // into the static section.
        if (auto wDef = weights.getDefiningOp()) {
            weights = createCompileTimeConstOp(wDef, rewriter).value_or(weights);
        }

        {
            rewriter.setInsertionPoint(output.getDefiningOp());
            // Build the torq_hl.fully_connected:
            //   $input   (input_slot) : im2col [Ow, K]   → N=Ow, IC=K
            //   $weights (weight_slot): filterT [K, F]   → IC=K, OC=F
            //   $bias                 : [F] for float, interleaved bias/scale [2*F] for integer
            //   output                : [Ow, F] in fcElemType
            // is_batch_scaled=false so bias applies per-OC=F (per-output-channel).
            LLVM_DEBUG(
                llvm::dbgs() << "[" DEBUG_TYPE "] Emitting torq_hl.fully_connected: [" << Ow << ", "
                             << F << "] elem=" << fcElemType << " expand_shape="
                             << (fusionPlanOr->includedExpandShape ? "yes" : "no") << "\n"
            );
            auto fcOutType = RankedTensorType::get({Ow, F}, fcElemType);
            Value fcInit = createInitTensor(matmulOp, rewriter, fcOutType);

            bool isBatchBias = fusionPlanOr->channelDim.value() == 0;
            if (fusionPlanOr->includedExpandShape) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[" DEBUG_TYPE
                                    "] Checking expand_shape reassociation for channel dim\n"
                );
                // If there is an expand_shape, the channel dim is not necessarily the first dim of
                // the fully_connected output. Check the expand_shape reassociation to see if the
                // channel dim is the first or second dim of the expanded output.
                auto expandOp = cast<tensor::ExpandShapeOp>(*fusionPlanOr->includedExpandShape);
                auto reassoc = expandOp.getReassociationIndices();
                if (reassoc[0].size() >= 2 && reassoc[0][0] == 0 && reassoc[0][1] == 1) {
                    isBatchBias = fusionPlanOr->channelDim.value() == 0 ||
                                  fusionPlanOr->channelDim.value() == 1;
                }
            }
            if (fusionPlanOr->includedTranspose) {
                isBatchBias = !isBatchBias;
            }

            // scInfo carries the default output attributes plus any folded
            // truncf/clamp data, so the FC op can consume it unconditionally.
            auto fcOp = torq_hl::FullyConnectedOp::create(
                rewriter, loc, fcOutType, fcInit, /*inputZp=*/0, /*weightZp=*/0, scInfo.zp,
                scInfo.min, scInfo.max, scInfo.scaleShift, torq_hl::VectorizationModeEnum::None,
                weights, *biasV, im2col, isBatchBias
            );
            Value finalResult = fcOp.getResult(0); // [Ow, F]

            // Rebuild the original output layout.
            if (fusionPlanOr->includedTranspose) {
                auto transposeOp = cast<linalg::TransposeOp>(*fusionPlanOr->includedTranspose);
                auto perm = Permutation(transposeOp.getPermutation());
                finalResult = transposeValue(finalResult, perm, loc, rewriter);
            }

            if (fusionPlanOr->includedExpandShape) {
                auto expandOp = cast<tensor::ExpandShapeOp>(*fusionPlanOr->includedExpandShape);
                auto reassoc = expandOp.getReassociationIndices();
                auto expandType = cast<RankedTensorType>(expandOp.getType());

                finalResult =
                    tensor::ExpandShapeOp::create(
                        rewriter, loc, RankedTensorType::get(expandType.getShape(), fcElemType),
                        finalResult, reassoc
                    )
                        .getResult();
            }

            // Replace the bottom-most fused op (truncf or bias add) with the new
            // FC + transpose + expand chain. The matmul, the post-matmul transpose
            // and the expand_shape become dead and are DCE'd by the canonicalizer
            // that runs at the end of the pre-conversion pass.
            LLVM_DEBUG(
                llvm::dbgs() << "[" DEBUG_TYPE "] Successfully replaced Conv1D matmul [" << F
                             << ", " << Ow << "] (K=" << K << ") with torq_hl.fully_connected\n"
            );
            rewriter.replaceOp(output.getDefiningOp(), finalResult);

            return success();
        }
    }
};

} // namespace

void populateLinalgToTorqHLConv1DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv1DMatmulToTorqHlFCPattern>(context, 28, 12, markFuseGroups);
}

} // namespace mlir::syna::torq
