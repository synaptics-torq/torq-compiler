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
#include "torq/Utils/TorqUtils.h"

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
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-conv2d-matmul-pattern"

namespace mlir::syna::torq {

struct Conv2DMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
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

        // Check if the Conv2D input (lhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(lhs.getDefiningOp())) {
            lhs = extractSlice.getSource();
        }
        if (!lhs.getDefiningOp<tensor::CollapseShapeOp>()) {
            // FIXME Maybe there are cases where there is no CollapseShapeOp and the rank is already
            // 2D !
            return rewriter.notifyMatchFailure(srcOp, "LHS is not collapsed from 4D");
        }
        Value input = lhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());

        if (!inputType || inputType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Match transpose on weight
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp())) {
            rhs = transposeOp.getInput();
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(rhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(rhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[1];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && isa<tensor::ExpandShapeOp>(*output.getUsers().begin())) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }
        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, rhs}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = rhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        finalType = convertTypeNHWCtoNCHW(finalType);
        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;
        input = transposeValue(input, Permutation::nhwc2nchw(), loc, rewriter);

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        auto torqOut =
            transposeValue(conv2dOp.getOutput(), Permutation::nhwc2nchw().reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DMatmulOpConversion success\n"; });
        return success();
    }
};

// Conv2DNchwMatmulOpConversion pattern
// Input NCHW, weight OIXY, Ouput NCHW
// This pattern detects and fuses the following operation chain:
//   %A = input, %collapsed = weight
//   %collapsed = tensor.collapse_shape %input [[0, 1], [2, 3]]
//                 : tensor<1xKx1x1xbf16> into tensor<Kx1xbf16>
//   %init = tensor.empty() : tensor<Nx1xf32>
//   %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<Nx1xf32>)
//   %matmul = linalg.matmul ins(%A, %collapsed : tensor<NxKxbf16>, tensor<Kx1xbf16>)
//                          outs(%fill : tensor<Nx1xf32>) -> tensor<Nx1xf32>
//   %expanded = tensor.expand_shape %matmul [[0, 1], [2, 3]]
//                 output_shape [1, N, 1, 1]
//                 : tensor<Nx1xf32> into tensor<1xNx1x1xf32>
//   %add = linalg.generic ins(%expanded, %bias)
//                          outs(%tmp)
//                          {body: (%x, %b) => arith.addf %x, %b}
//   %trunc = linalg.generic ins(%add)
//                            outs(%out)
//                            {body: (%x) => arith.truncf %x}
//   %output = %trunc
struct Conv2DNchwMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DNchwMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    static linalg::GenericOp getNextGenericWithTruncf(Value value) {
        if (!value.hasOneUse())
            return nullptr;

        auto genericOp = dyn_cast<linalg::GenericOp>(*value.getUsers().begin());
        if (!genericOp)
            return nullptr;

        for (Operation &op : genericOp.getBody()->getOperations())
            if (isa<arith::TruncFOp>(op))
                return genericOp;

        return nullptr;
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
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
                srcOp, "Conv2DNchwMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Check if the Conv2D input (rhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(rhs.getDefiningOp())) {
            rhs = extractSlice.getSource();
        }
        if (!rhs.getDefiningOp<tensor::CollapseShapeOp>() &&
            !isCollapseOrExpandShapeGeneric(rhs.getDefiningOp())) {
            return rewriter.notifyMatchFailure(srcOp, "RHS is not collapsed from 4D");
        }
        Value input = rhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(lhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(lhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[0];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        // Check if the next op is linalg.generic with add and update bias, input_zp & weightZP
        // values
        if (!foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
            return rewriter.notifyMatchFailure(srcOp, "Next op is not linalg.generic with add");
        }

        // Check if the next op is linalg.generic with Trunc
        auto truncGeneric = getNextGenericWithTruncf(output);
        if (!truncGeneric)
            return rewriter.notifyMatchFailure(srcOp, "Next op is not linalg.generic with truncf");

        // Get the result tensor from the linalg.generic
        output = truncGeneric.getResultTensors()[0];
        finalType = cast<RankedTensorType>(output.getType());

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            // For float ops, only clamp (min/max) is set; scales are not required
            else if (!scInfo.hasClamp())
                scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }

        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        // padding input rank to be output rank
        if (inputType.getRank() < finalType.getRank()) {
            SmallVector<ReassociationIndices> reassoc;
            int rank = finalType.getRank();
            if (rank <= 1) {
                for (int i = 0; i < rank; ++i)
                    reassoc.push_back({i});
            }
            else {
                for (int i = 0; i < rank - 2; ++i) {
                    reassoc.push_back({i});
                }
                reassoc.push_back({rank - 2, rank - 1});
            }
            auto inputShape = inputType.getShape();
            SmallVector<int64_t, 8> newInShape(inputShape.begin(), inputShape.end());
            newInShape.resize(finalType.getRank(), 1);
            auto newTy = RankedTensorType::get(newInShape, inputType.getElementType());
            input = rewriter.create<tensor::ExpandShapeOp>(loc, newTy, input, reassoc).getResult();
        }
        else if (inputType.getRank() > finalType.getRank()) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input rank <= output rank");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {lhs, input}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = lhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        rewriter.replaceOp(output.getDefiningOp(), conv2dOp.getOutput());

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DNchwMatmulOpConversion success\n"; });
        return success();
    }
};

void populateLinalgToTorqHLConv2DMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv2DMatmulOpConversion>(context, markFuseGroups);
    patterns.insert<Conv2DNchwMatmulOpConversion>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
