// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-conv2d-matmul-pattern"

namespace mlir::syna::torq {

// NCHW pattern
// Input NCHW, weight OIXY, Ouput NCHW
// %cst_1 is weights
// %collapse is input
// %9 = linalg.matmul ins(%cst_1, %collapsed : tensor<8192x64xbf16>, tensor<64x1xbf16>) outs(%8 :
// tensor<8192x1xf32>) -> tensor<8192x1xf32>

// This pattern detects and fuses the following operation chain:
// NHWC pattern
// %collapsed is input
// %transposed is weights transposed from OxI to IxO
// %7 = linalg.matmul ins(%collapsed, %transposed : tensor<3136x144xi8>, tensor<144x24xi8>) outs(%6
// : tensor<3136x24xi32>) -> tensor<3136x24xi32>
struct Conv2DMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const int _channelDim;
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DMatmulOpConversion(MLIRContext *context, int channelDim, bool markFuseGroups)
        : OpRewritePattern(context), _channelDim(channelDim), _markFuseGroups(markFuseGroups) {}

    bool isNCHWMatmul(linalg::MatmulOp matmulOp) const {
        // Heuristic: NHWC path typically has an explicit transpose on RHS weights.
        // If RHS is already direct (no transpose), treat as NCHW-style layout.
        Value rhs = matmulOp.getInputs()[1];

        if (auto transposeOp =
                llvm::dyn_cast_if_present<linalg::TransposeOp>(rhs.getDefiningOp())) {
            return false;
        }
        return true;
    }

    Value preConversion(Value input, PatternRewriter &rewriter, bool isNCHW, bool isFC) const {
        // FIXME: Handling 1x1x1x1 case. Need to figure out if this is applicable for more cases
        auto inputTy = cast<RankedTensorType>(input.getType());
        if (inputTy.getRank() < 4 && !isFC) {
            // Conv lowering expects 4D activation. If input was flattened for matmul,
            // reconstruct a minimal 4D shape before layout normalization.
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointAfterValue(input);
            auto shape = inputTy.getShape();

            SmallVector<int64_t, 4> newShape;
            if (isNCHW) {
                newShape = {shape[0], shape[1], 1, 1};
            }
            else {
                newShape = {1, 1, shape[0], shape[1]};
            }

            auto newType = RankedTensorType::get(newShape, inputTy.getElementType());
            auto maybeReassoc = getReassociationIndicesForCollapse(newShape, shape);
            if (!maybeReassoc) {
                llvm::report_fatal_error("Failed to get reassociation indices for collapse", true);
            }
            auto reassoc = *maybeReassoc;
            input = rewriter.create<tensor::ExpandShapeOp>(input.getLoc(), newType, input, reassoc);
        }
        Permutation dataPerm = Permutation::nhwc2nchw();
        if (isFC || isNCHW) {
            // FC and native NCHW paths do not need NHWC->NCHW permutation.
            dataPerm = Permutation::none();
        }
        auto transposed = transposeValue(input, dataPerm, input.getLoc(), rewriter);

        return transposed;
    }

    Value preConversionWeights(
        Value weights, std::optional<Value> weightZpV, ScaleClampInfo &scInfo,
        PatternRewriter &rewriter, bool isNCHW, bool isFC
    ) const {
        if (!isNCHW || isFC) {
            // Canonicalize weights to [O, I] orientation expected by downstream lowering.
            weights =
                transposeValue(weights, SmallVector<int64_t, 4>{1, 0}, weights.getLoc(), rewriter);
        }
        auto weightTy = cast<RankedTensorType>(weights.getType());
        assert(weightTy.getRank() == 2 && "Expected weights to be 2D after collapsing from 4D");

        if (!isFC) {
            // Conv path expects 4D filter tensor, expand 2D matmul weights to [O, I, 1, 1].
            auto weightShape = weightTy.getShape();
            SmallVector<int64_t, 4> newShape = {weightShape[0], weightShape[1], 1, 1};
            SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2, 3}};
            auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
                weights.getLoc(), RankedTensorType::get(newShape, weightTy.getElementType()),
                weights, reassoc
            );
            weights = expandShapeOp.getResult();
        }
        if (weightZpV) {
            // Fold weight zero-point directly into constant weights when available.
            auto modifiedWeights = buildWeightWithZp(weights, *weightZpV, rewriter);
            if (succeeded(modifiedWeights)) {
                LLVM_DEBUG(llvm::dbgs() << "Folded weight zero-point into weights\n");
                weights = *modifiedWeights;
            }
        }
        // Weights may become static payload.
        setCompileTimeConstAttr(weights.getDefiningOp());
        return weights;
    }

    Value postConversion(Value output, PatternRewriter &rewriter, bool isNCHW, bool isFC) const {
        // Undo preConversion layout normalization to match original graph convention.
        Permutation dataPerm = Permutation::nhwc2nchw().reverse();
        if (isFC || isNCHW) {
            dataPerm = Permutation::none();
        }
        auto transposed = transposeValue(output, dataPerm, output.getLoc(), rewriter);
        return transposed;
    }

    Value createOutput(Value output, PatternRewriter &rewriter, bool isNCHW, bool isFC) const {
        // Create destination init tensor in the layout expected by the Torq op.
        Permutation dataPerm = Permutation::nhwc2nchw();
        if (isFC || isNCHW) {
            dataPerm = Permutation::none();
        }
        auto finalType = cast<RankedTensorType>(output.getType());
        finalType = transposeType(finalType, dataPerm);
        Value initTensor = createInitTensor(*output.getDefiningOp(), rewriter, finalType);
        return initTensor;
    }

    Value replaceWithTorqMatmul(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const {
        // Fallback rewrite when no fusible conv/fc chain is present:
        // emit plain torq_hl.matmul with neutral bias/scale parameters.
        auto outTy = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(outTy.getElementType());

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        Value biasScale = createI32Const(rewriter, srcOp, interleave(bias, scale));

        auto matMulOp = rewriter.replaceOpWithNewOp<torq_hl::MatMulOp>(
            srcOp, outTy, createInitTensor(srcOp, rewriter, outTy), 0, outMin, outMax, 0, biasScale,
            srcOp.getOperand(0), srcOp.getOperand(1)
        );
        return matMulOp.getResult(0);
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        // High-level flow:
        // 1) recognize matmul-as-conv/fc and collect fusible trailing ops,
        // 2) derive per-channel bias + rescale metadata from that fusion chain,
        // 3) rewrite into torq_hl Conv2D/FullyConnected with normalized layout/weights.
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }

        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        Value output = srcOp.getResultTensors()[0];

        // Ensure inputs and output are 2D
        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
        auto outType = llvm::cast<RankedTensorType>(output.getType());

        if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
            outType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Conv2DMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Build fusion plan and compute bias/scale using PatternUtils helpers
        FailureOr<FusionPlan> fusionPlanOr = buildFusionPlanAndRebindOutput(output);
        if (failed(fusionPlanOr) || !fusionPlanOr->isFusable()) {
            if (_markFuseGroups) {
                // Discovery-only mode: mark the chain and defer material rewrite.
                markFuseGroupBackward(
                    output, {lhs, rhs}, rewriter,
                    srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                );
                return success();
            }
            replaceWithTorqMatmul(srcOp, rewriter);
            return success();
        }

        // If there is an expand_shape user, use it to determine 4D output shape
        tensor::ExpandShapeOp outputExpandOp = nullptr;
        if (output.hasOneUse() && isa<tensor::ExpandShapeOp>(*output.getUsers().begin())) {
            outputExpandOp = cast<tensor::ExpandShapeOp>(*output.getUsers().begin());
            output = outputExpandOp.getResult();
        }
        RankedTensorType finalType = cast<RankedTensorType>(output.getType());

        bool isNCHW = isNCHWMatmul(srcOp);
        // A matmul always has rank-2 output, so start by assuming FC.
        // Override to conv when an ExpandShapeOp reveals real spatial dims.
        // For NCHW matmuls the operands are in weights×input order; the conv
        // path handles the required swap while the FC path does not, so always
        // keep NCHW 4D outputs on the conv path.
        bool isFC = true;
        if (finalType.getRank() == 4) {
            if (isNCHW) {
                isFC = false;
            }
            else {
                auto shape = finalType.getShape();
                bool hasSpatial = (shape[1] > 1 || shape[2] > 1);
                if (hasSpatial)
                    isFC = false;
            }
        }

        Value input = lhs;
        Value weights = rhs;
        int channelDim = _channelDim;
        if (isNCHW && !isFC) {
            // NCHW matmul form stores operands in opposite positions vs NHWC path.
            // Swap roles so downstream conversion logic sees canonical (input, weights).
            input = rhs;
            weights = lhs;
            channelDim = 1;
        }
        // Check if the Conv2D input (lhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        // For FC lowering, keep tiled extract_slice on the activation input.
        // Peeling it here would expand back to the full source tensor and can
        // break tile-local memory planning downstream.
        if (!isFC) {
            while (auto extractSlice =
                       llvm::dyn_cast_if_present<tensor::ExtractSliceOp>(input.getDefiningOp())) {
                input = extractSlice.getSource();
            }
        }
        if (!isFC &&
            input.getDefiningOp<tensor::CollapseShapeOp>(
            )) { // FIXME In some cases the input is collapsed to 2D from a 3D. Will be removed once
                 // we a better way to detect the channel dimension in such cases
            input = input.getDefiningOp()->getOperand(0);
        }

        // Check weights are supported
        auto weightElemType = cast<RankedTensorType>(weights.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        if (_markFuseGroups) {
            // Discovery-only mode: mark the chain and defer material rewrite.
            // check if the weight input is transposed, then move the weight mark to the transpose
            if (auto transposeOp =
                    llvm::dyn_cast_if_present<linalg::TransposeOp>(weights.getDefiningOp())) {
                weights = transposeOp.getInput();
            }

            markFuseGroupBackward(
                output, {input, weights}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Compute per-channel bias value and scale/clamp info from fused chain
        std::optional<Value> optionalWeightZpV;
        FailureOr<Value> biasV =
            computeBiasForMatmul(*fusionPlanOr, channelDim, optionalWeightZpV, isFC);
        if (failed(biasV)) {
            replaceWithTorqMatmul(srcOp, rewriter);
            return success();
        }

        ScaleClampInfo scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        biasV = computeRescaleInfo(*fusionPlanOr, *biasV, scInfo);
        if (failed(biasV)) {
            replaceWithTorqMatmul(srcOp, rewriter);
            return success();
        }

        // Erase in reverse order to avoid invalidating users while pruning folded tail ops.
        for (auto &op : llvm::reverse(fusionPlanOr->opsToFuse)) {
            if (op->use_empty()) {
                LLVM_DEBUG({
                    llvm::dbgs() << "Erasing op in fusion plan: ";
                    op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
                    llvm::dbgs() << "\n";
                });
                rewriter.eraseOp(op);
            }
        }
        LLVM_DEBUG({
            assert(
                succeeded(verify(srcOp->getParentOp())) && "Parent function verification failed"
            );
        });

        {
            rewriter.setInsertionPoint(output.getDefiningOp());

            auto vectorizationMode = torq_hl::VectorizationModeEnum::None;

            input = preConversion(input, rewriter, /*isNchw=*/isNCHW, isFC);

            auto torqWeights =
                preConversionWeights(weights, optionalWeightZpV, scInfo, rewriter, isNCHW, isFC);

            auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
            auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
            auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

            // Input zero-point is not folded here; treat as 0
            int32_t input_zp = -128;

            Value initTensor = createOutput(output, rewriter, isNCHW, isFC);
            finalType = cast<RankedTensorType>(initTensor.getType());
            Value torqOut;
            if (!isFC) {
                // Non-2D outputs are treated as conv lowering.
                auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
                    loc, finalType, initTensor,
                    input_zp, // input_zp
                    0,        // weight_zp
                    scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
                    1,        // groups
                    pad,      // pad
                    stride,   // stride
                    dilation, // dilation
                    vectorizationMode, torqWeights, *biasV, input
                );

                torqOut = postConversion(conv2dOp.getOutput(), rewriter, isNCHW, isFC);
            }
            else {
                // FullyConnected always operates on rank-2 tensors.
                // If the output was expanded beyond rank 2 (e.g. by an
                // absorbed ExpandShapeOp), collapse to rank 2 for the FC op
                // and expand back afterwards.
                RankedTensorType fcType = finalType;
                if (finalType.getRank() > 2) {
                    auto shape = finalType.getShape();
                    int64_t batchDims = 1;
                    for (int i = 0; i < finalType.getRank() - 1; ++i)
                        batchDims *= shape[i];
                    fcType = RankedTensorType::get(
                        {batchDims, shape[finalType.getRank() - 1]}, finalType.getElementType()
                    );
                    initTensor = createInitTensor(*output.getDefiningOp(), rewriter, fcType);
                }
                auto fcOp = rewriter.create<torq_hl::FullyConnectedOp>(
                    loc, fcType, initTensor, input_zp,
                    0, // weight zp
                    scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
                    torq_hl::VectorizationModeEnum::None, torqWeights, *biasV, input
                );
                torqOut = fcOp.getResult(0);
                if (fcType != finalType) {
                    auto reassoc =
                        getReassociationIndicesForCollapse(finalType.getShape(), fcType.getShape());
                    assert(reassoc && "Failed to get reassociation for FC expand");
                    torqOut =
                        rewriter.create<tensor::ExpandShapeOp>(loc, finalType, torqOut, *reassoc);
                }
            }
            rewriter.replaceOp(output.getDefiningOp(), torqOut);
        }

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DMatmulOpConversion success\n"; });
        return success();
    }
};

void populateLinalgToTorqHLConv2DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv2DMatmulOpConversion>(context, 3, markFuseGroups);
}

} // namespace mlir::syna::torq
