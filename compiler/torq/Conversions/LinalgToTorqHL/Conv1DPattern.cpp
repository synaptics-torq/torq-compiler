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

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Analysis/DataFlowFramework.h"
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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-conv1d-pattern"

namespace mlir::syna::torq {

/// Lowers linalg::GenericOp (from Conv1D with preserved kernel dim) to torq_hl::Conv1DOp.
///
/// This pattern matches a 5D linalg.generic op produced by Conv1DNcwFcwToGenericConv1DPattern
/// and lowers it directly to torq_hl.conv1d. The 5D structure [N, F, 1, Ow, Kw] is preserved
/// in the output, allowing downstream patterns to handle the kernel reduction.
///
/// Input/Output Shapes (NCHW layout):
///   - Input: [N, C, 1, W] (4D)
///   - Filter: [F, C, 1, Kw] (4D)
///   - Output: [N, F, 1, Ow, Kw] (5D), element type may be bf16 if truncf is fused
///
/// Features:
///   - Extracts stride/dilation from the generic op's affine indexing maps
///   - Folds per-channel bias into the operation
///   - Fuses scale/clamp operations (foldForwardScaleClamp)
///   - Fuses truncf (f32 -> bf16) either from:
///     a) Inside the generic body (legacy pattern), or
///     b) As a separate op following the generic (current pattern with reduce sum)
///   - Supports fuse group marking for backend optimization
///   - Marks filter as compile-time constant
struct LinalgGenericConv1DToTorqHLConv1DPattern : public OpRewritePattern<linalg::GenericOp> {
  private:
    const int _shift8b;  // Scale shift for 8-bit integer operations
    const int _shift16b; // Scale shift for 16-bit integer operations
    const bool _markFuseGroups;

  public:
    LinalgGenericConv1DToTorqHLConv1DPattern(
        MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups
    )
        : OpRewritePattern<linalg::GenericOp>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        auto loc = genericOp.getLoc();

        // Check if already marked as fuse group
        if (_markFuseGroups && isMarkedFuseGroup(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "Already marked");
        }

        // Check if this is a 5D generic op (output should be 5D)
        if (genericOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected single result");
        }

        auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
        if (resultType.getRank() != 5) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 5D output tensor");
        }

        // Check that we have exactly 2 inputs (input tensor and filter)
        if (genericOp.getInputs().size() != 2) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 2 input operands");
        }

        Value input = genericOp.getInputs()[0];
        Value filter = genericOp.getInputs()[1];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());

        // Expected shapes (NCHW layout):
        // input: [N, C, 1, W] - 4D expanded from [N, C, W]
        // filter: [F, C, 1, Kw] - 4D expanded from [F, C, Kw]
        // output: [N, F, 1, Ow, Kw] - NCHW layout with preserved kernel dim

        if (inputType.getRank() != 4 || filterType.getRank() != 4) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 4D input and filter tensors");
        }

        // Validate input shape: [N, C, 1, W]
        if (inputType.getShape()[2] != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected input shape [N, C, 1, W]");
        }

        // Validate filter shape: [F, C, 1, Kw]
        if (filterType.getShape()[2] != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected filter shape [F, C, 1, Kw]");
        }

        auto outShape = resultType.getShape();
        if (outShape.size() != 5 || outShape[2] != 1) {
            return rewriter.notifyMatchFailure(
                genericOp, "Expected output shape [N, F, 1, Ow, Kw]"
            );
        }

        int64_t N = outShape[0];
        int64_t F = outShape[1];
        int64_t Ow = outShape[3];
        int64_t Kw = outShape[4];

        auto elemType = resultType.getElementType();

        // We expect exactly 3 affine maps: 2 inputs + 1 output
        auto maps = genericOp.getIndexingMapsArray();
        if (maps.size() != 3) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 3 indexing maps");
        }

        // Extract stride and dilation from input affine map
        // Input map: (n, f, kh, ow, kw) -> (n, 0, kh, ow * stride + kw * dilation)
        int64_t strideValue = 1;
        int64_t dilationValue = 1;

        auto inputMap = maps[0];
        // The 4th result expr should be: ow * stride + kw * dilation
        if (inputMap.getNumResults() == 4) {
            auto widthExpr = inputMap.getResult(3);
            // Try to extract stride and dilation from the affine expression
            // Expression should be: dim(3) * stride + dim(4) * dilation
            if (auto binOp = llvm::dyn_cast<AffineBinaryOpExpr>(widthExpr)) {
                if (binOp.getKind() == mlir::AffineExprKind::Add) {
                    auto lhs = binOp.getLHS();
                    auto rhs = binOp.getRHS();
                    // Check lhs: ow * stride (ow is dim 3)
                    if (auto mulOp = llvm::dyn_cast<AffineBinaryOpExpr>(lhs)) {
                        if (mulOp.getKind() == mlir::AffineExprKind::Mul) {
                            if (auto constExpr =
                                    llvm::dyn_cast<AffineConstantExpr>(mulOp.getRHS())) {
                                strideValue = constExpr.getValue();
                            }
                        }
                    }
                    // Check rhs: kw * dilation (kw is dim 4)
                    if (auto mulOp = llvm::dyn_cast<AffineBinaryOpExpr>(rhs)) {
                        if (mulOp.getKind() == mlir::AffineExprKind::Mul) {
                            if (auto constExpr =
                                    llvm::dyn_cast<AffineConstantExpr>(mulOp.getRHS())) {
                                dilationValue = constExpr.getValue();
                            }
                        }
                    }
                }
            }
        }

        // Should be all parallel: (parallel, parallel, parallel, parallel, parallel)
        auto iterators = genericOp.getIteratorTypesArray();
        if (iterators.size() != 5) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 5 iterator types");
        }

        // The generic op should perform a multiply operation
        if (genericOp.getBody()->getOperations().size() < 2) {
            return rewriter.notifyMatchFailure(genericOp, "Generic body too small");
        }

        // Find and validate the actual computation operation
        arith::MulFOp mulOp = nullptr;
        for (auto &op : genericOp.getBody()->getOperations()) {
            if (auto mul = llvm::dyn_cast<arith::MulFOp>(&op)) {
                mulOp = mul;
                break;
            }
        }

        if (!mulOp) {
            return rewriter.notifyMatchFailure(genericOp, "Expected arith.mulf operation in body");
        }

        // Get the generic op result
        auto genericResult = genericOp.getResult(0);

        // Determine the output type for torq_hl.conv1d and check for truncf to fuse.
        // Two cases:
        // 1. truncf is inside the generic body (legacy fused pattern) - use bf16 output
        // 2. truncf is a separate linalg.generic following the generic (current pattern) - fuse it
        // here
        Type torqConv1dOutputType = elemType;
        linalg::GenericOp followingTruncfGeneric = nullptr;

        // Case 1: Look for arith.truncf in the generic body
        for (auto &op : genericOp.getBody()->getOperations()) {
            if (auto truncfOp = llvm::dyn_cast<arith::TruncFOp>(&op)) {
                // Found truncf fused in body, use bf16 output type
                torqConv1dOutputType = truncfOp.getResult().getType();
                break;
            }
        }

        // Case 2: Check if there's a truncf generic following the generic (f32 -> bf16)
        // This happens when OptimizeConv1DPattern inserts truncf before reduce sum.
        // The truncf is wrapped in a linalg.generic, not a bare arith.truncf.
        // We can fuse it into torq_hl.conv1d for efficiency.
        if (genericResult.hasOneUse()) {
            auto userOp = genericResult.use_begin()->getOwner();
            if (auto genericUser = llvm::dyn_cast<linalg::GenericOp>(userOp)) {
                // Check if it's a truncf generic: f32 input -> bf16 output
                auto srcType = dyn_cast<RankedTensorType>(genericUser.getInputs()[0].getType());
                auto dstType = dyn_cast<RankedTensorType>(genericUser.getResult(0).getType());
                if (srcType && dstType && srcType.getElementType().isF32() &&
                    dstType.getElementType().isBF16()) {
                    // Verify it contains a truncf operation
                    bool hasTruncf = false;
                    for (auto &op : genericUser.getBody()->getOperations()) {
                        if (llvm::isa<arith::TruncFOp>(&op)) {
                            hasTruncf = true;
                            break;
                        }
                    }
                    if (hasTruncf) {
                        // Found truncf generic following the conv generic, fuse it into conv1d
                        torqConv1dOutputType = dstType.getElementType();
                        followingTruncfGeneric = genericUser;
                    }
                }
            }
        }

        // Create 5D output for torq_hl.conv1d: [N, F, 1, Ow, Kw]
        // This preserves the height dimension structure (NCHW layout)
        SmallVector<int64_t> conv1dOutShape = {N, F, 1, Ow, Kw};
        auto conv1dOutType = RankedTensorType::get(conv1dOutShape, torqConv1dOutputType);
        auto conv1dOutInit =
            tensor::EmptyOp::create(rewriter, loc, conv1dOutShape, torqConv1dOutputType);

        // Fold forward bias from the output users
        bool isInt = torqConv1dOutputType.isInteger();
        VectorIntOrFloat bias(F, isInt);
        Value outputValue = genericResult;
        while (foldForwardPerChannelAdd(outputValue, 1, bias)) {
        }

        // Fold forward scale and clamp info
        ScaleClampInfo scInfo = foldForwardScaleClamp(outputValue, F, _shift8b, _shift16b);

        // Get weight zero point
        Value filterValue = filter;
        int weightZp = foldForwardWeightZp(filterValue);

        int64_t inputZp = 0;
        int32_t groups = 1;

        SmallVector<int64_t> pad(4, 0);
        SmallVector<int64_t> stride = {strideValue};
        SmallVector<int64_t> dilation = {dilationValue};

        // Create bias tensor
        Value biasValue = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                                : createConst(bias.floats, rewriter, loc);

        // Mark fuse group if requested
        if (_markFuseGroups) {
            markFuseGroupBackward(
                outputValue, {input, filter}, rewriter,
                genericOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Mark filter as compile-time constant if applicable
        if (filter.getDefiningOp()) {
            setCompileTimeConstAttr(filter.getDefiningOp());
        }

        // Create torq_hl.conv1d operation (outputs 5D [N, F, 1, Ow, Kw])
        auto torqConv1dOp = torq_hl::Conv1DOp::create(
            rewriter, loc, conv1dOutType, conv1dOutInit, inputZp, weightZp, scInfo.zp, scInfo.min,
            scInfo.max, scInfo.scaleShift, groups, pad, stride, dilation,
            torq_hl::VectorizationModeEnum::None, filter, biasValue, input
        );

        // Replace the generic op with torq_hl.conv1d
        rewriter.replaceOp(genericOp, torqConv1dOp.getResults());

        // If we fused a following truncf generic, erase it as well since conv1d now outputs bf16
        if (followingTruncfGeneric) {
            // The truncf generic's users should now use the conv1d result directly
            // Since we already replaced genericOp with conv1d, and truncf was using genericOp,
            // we need to make sure truncf's users are redirected to conv1d's bf16 output
            rewriter.replaceOp(followingTruncfGeneric, torqConv1dOp.getResults());
        }

        return success();
    }
};

void populateLinalgConv2DToTorqHLConv1DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // Generic 5D conv1d pattern (from Conv1DNcwFcwToGenericConv1DPattern)
    patterns.insert<LinalgGenericConv1DToTorqHLConv1DPattern>(context, 28, 12, markFuseGroups);
}

} // namespace mlir::syna::torq
