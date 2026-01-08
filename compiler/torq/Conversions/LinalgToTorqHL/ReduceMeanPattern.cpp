// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-reducemean-pattern"

namespace mlir::syna::torq {

struct ReduceMeanPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp.getNumDpsInputs() != 1 || srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "ReducemeanPattern Expected 1 input/init");
        }

        auto inType = dyn_cast_or_null<RankedTensorType>(srcOp.getInputs()[0].getType());
        auto output = srcOp.getResult(0);
        auto outType = dyn_cast_or_null<RankedTensorType>(output.getType());

        if (!inType || !outType || !inType.getElementType().isBF16() ||
            !outType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(
                srcOp, "ReducemeanPattern only bf16 supported for now"
            );
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected yield");
        }

        auto divFOp = dyn_cast_or_null<arith::DivFOp>(yieldOp.getValues()[0].getDefiningOp());
        if (!divFOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected div op");
        }

        if (!isa<BlockArgument>(divFOp.getLhs())) {
            return rewriter.notifyMatchFailure(srcOp, "Div lhs must be block arg");
        }

        auto divRhs = divFOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!divRhs) {
            return rewriter.notifyMatchFailure(srcOp, "Div rhs must be constant");
        }

        auto divConstAttr = dyn_cast_or_null<FloatAttr>(divRhs.getValue());
        if (!divConstAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Div constant must be float");
        }

        double divConst = divConstAttr.getValueAsDouble();
        if (divConst <= 0.0) {
            return rewriter.notifyMatchFailure(srcOp, "Div constant must be positive");
        }

        // if output is used by CollapseShape, fold collapseShape op
        if (output.hasOneUse() && (isa<tensor::CollapseShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        auto reducesumOp = srcOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!reducesumOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected sum reduction");
        }

        auto reducesumYield = dyn_cast<linalg::YieldOp>(reducesumOp.getBody()->getTerminator());
        if (!reducesumYield || !isa<arith::AddFOp>(reducesumYield.getValues()[0].getDefiningOp())) {
            return rewriter.notifyMatchFailure(
                reducesumOp, "ReducesumPattern Expected AddFOp reduction"
            );
        }

        SmallVector<unsigned> reductionDims;
        reducesumOp.getReductionDims(reductionDims);
        if (reductionDims.size() < 1) {
            return rewriter.notifyMatchFailure(
                reducesumOp, "ReducesumPattern expected reduction loop > 0"
            );
        }

        SmallVector<unsigned> parallelDims;
        reducesumOp.getParallelDims(parallelDims);

        Value input = reducesumOp.getInputs()[0];

        // reduceMean has batch and its iteratetype is parallel
        SmallVector<uint64_t, 4> permVec;
        permVec.push_back(0);
        permVec.append(reductionDims.begin(), reductionDims.end());
        for (int i = 1; i < parallelDims.size(); i++) {
            permVec.push_back(parallelDims[i]);
        }

        // avgpool kernel request nhwc
        auto loc = srcOp.getLoc();
        Permutation dataPerm(permVec.begin(), permVec.end());
        input = transposeValue(input, dataPerm, loc, rewriter);

        // Scale = 1 / meanConst for mean calculation
        float meanValue = 1.0f / divConst;
        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weightsData(1, llvm::APFloat(bf16, std::to_string(meanValue)));
        auto weights = createConst(weightsData, rewriter, srcOp.getLoc());

        std::vector<float> biasScaleData{0.0};
        auto biasScale = createConst(biasScaleData, rewriter, srcOp.getLoc());

        auto avgpoolOutType = dyn_cast_or_null<RankedTensorType>(output.getType());

        rewriter.replaceOpWithNewOp<torq_hl::AvgPool2DOp>(
            srcOp, avgpoolOutType, createInitTensor(reducesumOp, rewriter, avgpoolOutType), 0, 0,
            0xff800000, 0x7f800000, 0, weights, biasScale, input
        );

        return success();
    }
};

// ReduceMeanConvert: detects a pattern of sum-reduction along the last axis followed by
// division by the reduced axis size, and lowers it to torq_hl::ReduceMeanOp.
// Example matched IR (bf16 shown but pattern supports bf16/f32):
//   %sum = linalg.generic {iterator_types=["parallel","reduction"]}
//            ins(%x: tensor<NxMxt>) outs(%init: tensor<Nxt>) { %r = arith.addf %in, %out; yield %r
//            }
//   %mean = linalg.generic {iterator_types=["parallel"]}
//            ins(%sum: tensor<Nxt>) outs(%out: tensor<Nxt>) { %q = arith.divf %in, %cst_M; yield %q
//            }
struct ReduceMeanConvert : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp meanOp, PatternRewriter &rewriter) const override {
        if (meanOp.getNumDpsInputs() != 1 || meanOp.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(meanOp, "Expected 1 input/init");

        auto iters = meanOp.getIteratorTypesArray();
        if (iters.empty() ||
            llvm::any_of(iters, [](auto t) { return t != mlir::utils::IteratorType::parallel; }))
            return rewriter.notifyMatchFailure(meanOp, "Expected parallel iterators");

        auto yieldOp = dyn_cast<linalg::YieldOp>(meanOp.getBody()->getTerminator());
        if (!yieldOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected yield");

        Operation *divOp = yieldOp.getValues()[0].getDefiningOp();
        if (!divOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected div op");

        // Only bf16 (DivFOp) is supported
        auto divFOp = dyn_cast<arith::DivFOp>(divOp);
        if (!divFOp)
            return rewriter.notifyMatchFailure(meanOp, "Only bf16 (DivFOp) supported");

        Value divLhs = divFOp.getLhs();
        if (!isa<BlockArgument>(divLhs))
            return rewriter.notifyMatchFailure(meanOp, "Div lhs must be block arg");

        auto divRhs = divFOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!divRhs)
            return rewriter.notifyMatchFailure(meanOp, "Div rhs must be constant");

        auto divConstAttr = dyn_cast<FloatAttr>(divRhs.getValue());
        if (!divConstAttr)
            return rewriter.notifyMatchFailure(meanOp, "Div constant must be float");

        double divConst = divConstAttr.getValueAsDouble();
        if (divConst <= 0.0)
            return rewriter.notifyMatchFailure(meanOp, "Div constant must be positive");

        auto sumOp = meanOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!sumOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected sum reduction");

        auto sumIters = sumOp.getIteratorTypesArray();
        if (llvm::count_if(sumIters, [](auto t) {
                return t == mlir::utils::IteratorType::reduction;
            }) != 1)
            return rewriter.notifyMatchFailure(sumOp, "Expected one reduction");

        // Find the index of the reduction loop in the loop nest
        int reductionLoopIdx = -1;
        for (size_t i = 0; i < sumIters.size(); ++i)
            if (sumIters[i] == mlir::utils::IteratorType::reduction)
                reductionLoopIdx = i;

        if (reductionLoopIdx == -1)
            return rewriter.notifyMatchFailure(sumOp, "No reduction loop found");

        // Map the reduction loop index to the actual input dimension index
        auto inMap = sumOp.getIndexingMapsArray()[0];
        int reductionAxis = -1;
        for (unsigned i = 0; i < inMap.getNumResults(); ++i) {
            if (auto dimExpr = dyn_cast<AffineDimExpr>(inMap.getResult(i))) {
                if (dimExpr.getPosition() == static_cast<unsigned>(reductionLoopIdx)) {
                    reductionAxis = i;
                    break;
                }
            }
        }

        if (reductionAxis == -1)
            return rewriter.notifyMatchFailure(
                sumOp, "Could not map reduction loop to input dimension"
            );

        auto sumYield = dyn_cast<linalg::YieldOp>(sumOp.getBody()->getTerminator());
        if (!sumYield || !isa<arith::AddFOp>(sumYield.getValues()[0].getDefiningOp()))
            return rewriter.notifyMatchFailure(sumOp, "Expected AddFOp reduction");

        auto loc = meanOp.getLoc();
        Value input = sumOp.getInputs()[0];
        auto inputType = cast<RankedTensorType>(input.getType());
        auto resultType = cast<RankedTensorType>(meanOp.getResult(0).getType());
        int64_t reducedDimSize = static_cast<int64_t>(divConst);

        // Reshape to NCHW rank-4
        Value inputNCHW = input;
        int rank = inputType.getRank();
        if (rank == 2) {
            auto s = inputType.getShape();
            // [B, F] -> [B, 1, 1, F] where B is batch (dim 0) and F is features (dim 1)
            auto nchwType = RankedTensorType::get({s[0], 1, 1, s[1]}, inputType.getElementType());
            inputNCHW = rewriter.create<tensor::ExpandShapeOp>(
                loc, nchwType, input, SmallVector<ReassociationIndices>{{0}, {1, 2, 3}}
            );
        }
        else if (rank == 3) {
            auto s = inputType.getShape();
            // [B, H, W] -> [B, 1, H, W] where B is batch (dim 0)
            auto nchwType =
                RankedTensorType::get({s[0], 1, s[1], s[2]}, inputType.getElementType());
            inputNCHW = rewriter.create<tensor::ExpandShapeOp>(
                loc, nchwType, input, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}
            );
        }
        else if (rank != 4)
            return rewriter.notifyMatchFailure(meanOp, "Only rank 2-4 supported");

        // Map reduction axis to NCHW and apply transpose to bring reduction axis to H position
        // ReduceMeanOp always reduces along dimension 2 (H), so we need to permute the input
        // to bring the target axis to position 2, then permute back after the reduction
        int nchwAxis = reductionAxis;
        if (rank == 2) {
            // [B, F] -> [B, 1, 1, F].
            // reductionAxis 0 (batch) -> nchwAxis 0 (N)
            // reductionAxis 1 (features) -> nchwAxis 3 (W)
            if (reductionAxis == 0) {
                nchwAxis = 0; // Reducing batch dimension -> N dimension
            }
            else if (reductionAxis == 1) {
                nchwAxis = 3; // Reducing features dimension -> W dimension
            }
        }
        else if (rank == 3) {
            // [B, H, W] -> [B, 1, H, W]. 0->0, 1->2, 2->3
            if (reductionAxis > 0)
                nchwAxis = reductionAxis + 1;
        }

        // Determine the transpose permutation to bring nchwAxis to position 2
        // - nchwAxis == 0 (N): [0,1,2,3] -> [2,1,0,3] (swap N and H)
        // - nchwAxis == 1 (C): [0,1,2,3] -> [0,2,1,3] (swap C and H)
        // - nchwAxis == 2 (H): identity (no transpose needed)
        // - nchwAxis == 3 (W): [0,1,2,3] -> [0,1,3,2] (swap H and W)
        Value reduceMeanInput = inputNCHW;
        Permutation preTransposePerm;
        bool needsPreTranspose = (nchwAxis != 2);
        if (nchwAxis == 0) {
            preTransposePerm = Permutation({2, 1, 0, 3});
            reduceMeanInput = transposeValue(inputNCHW, preTransposePerm, loc, rewriter);
        }
        else if (nchwAxis == 1) {
            preTransposePerm = Permutation({0, 2, 1, 3});
            reduceMeanInput = transposeValue(inputNCHW, preTransposePerm, loc, rewriter);
        }
        else if (nchwAxis == 3) {
            preTransposePerm = Permutation({0, 1, 3, 2});
            reduceMeanInput = transposeValue(inputNCHW, preTransposePerm, loc, rewriter);
        }

        // Scale = 1/reducedDimSize for mean calculation (e.g., 1/288 = 0.003472)
        float scaleValue = 1.0f / static_cast<float>(reducedDimSize);
        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weightsData(1, llvm::APFloat(bf16, std::to_string(scaleValue)));
        auto weights = createConst(weightsData, rewriter, loc);

        // Create f32 bias tensor
        std::vector<float> biasScaleData{0.0};
        auto biasScale = createConst(biasScaleData, rewriter, loc);

        auto inputShape = cast<RankedTensorType>(reduceMeanInput.getType()).getShape();
        SmallVector<int64_t> reduceMeanOutShape = {inputShape[0], inputShape[1], 1, inputShape[3]};
        auto reduceMeanOutType =
            RankedTensorType::get(reduceMeanOutShape, resultType.getElementType());

        // For bf16: use full f32 range (no clipping)
        float min_f = std::numeric_limits<float>::lowest();
        int32_t output_min = *reinterpret_cast<int32_t *>(&min_f);
        float max_f = std::numeric_limits<float>::max();
        int32_t output_max = *reinterpret_cast<int32_t *>(&max_f);

        auto reduceMeanOp = rewriter.create<torq_hl::ReduceMeanOp>(
            loc, reduceMeanOutType, createInitTensor(meanOp, rewriter, reduceMeanOutType), 0, 0,
            output_min, output_max,                // input_zp=0, output_zp=0, min, max
            0, weights, biasScale, reduceMeanInput // shift_factor=0 for bf16
        );

        // Undo transpose if it was applied (restore original axis positions)
        // The swap permutations are self-inverse, so we use the same permutation
        Value out = reduceMeanOp.getOutput();
        if (needsPreTranspose)
            out = transposeValue(out, preTransposePerm, loc, rewriter);

        // Collapse back to the result rank
        // After transpose-back, out has shape [N, C, H, 1] or [N, C, 1, W] depending on reduction
        // axis
        int outRank = resultType.getRank();
        SmallVector<ReassociationIndices> collapseIndices;

        if (rank == 2) {
            // Input was rank 2, output is rank 1 or 2
            if (outRank == 1)
                collapseIndices = {{0, 1, 2, 3}};
            else // outRank == 2
                collapseIndices = {{0, 1, 2}, {3}};
        }
        else if (rank == 3) {
            // Input was rank 3, output is rank 2 or 3
            // Input [N, H, W] -> Expanded [N, 1, H, W]
            if (outRank == 2) {      // keepdims=false
                if (nchwAxis == 0) { // Reduced N. [1, 1, H, W] -> [H, W]
                    collapseIndices = {{0, 1, 2}, {3}};
                }
                else { // Reduced H or W. [N, 1, 1, W] or [N, 1, H, 1] -> [N, W] or [N, H]
                    collapseIndices = {{0, 1}, {2, 3}};
                }
            }
            else { // outRank == 3, keepdims=true
                // [N, 1, 1, W] -> [N, 1, W]
                // [N, 1, H, 1] -> [N, H, 1]
                // [1, 1, H, W] -> [1, H, W]
                if (nchwAxis == 0) { // Reduced N. [1, 1, H, W] -> [1, H, W]
                    collapseIndices = {{0, 1}, {2}, {3}};
                }
                else {
                    collapseIndices = {{0}, {1, 2}, {3}};
                }
            }
        }
        else if (rank == 4) {
            // Input [N, C, H, W] -> [N, C, H, W] (Identity or Transpose)
            if (outRank == 3) {      // keepdims=false
                if (nchwAxis <= 1) { // Reduced N or C
                    collapseIndices = {{0, 1}, {2}, {3}};
                }
                else { // Reduced H or W
                    collapseIndices = {{0}, {1}, {2, 3}};
                }
            }
            else { // outRank == 4, keepdims=true
                collapseIndices = {{0}, {1}, {2}, {3}};
            }
        }

        out = rewriter.create<tensor::CollapseShapeOp>(loc, resultType, out, collapseIndices);

        rewriter.replaceOp(meanOp, out);
        return success();
    }
};

void populateLinalgToTorqHLReduceMeanPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<ReduceMeanConvert>(context);

    // TODO: refactor with ReduceMeanConvert later soon
    patterns.insert<ReduceMeanPattern>(context);
}

} // namespace mlir::syna::torq