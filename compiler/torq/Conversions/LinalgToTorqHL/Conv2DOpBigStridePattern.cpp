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

#define DEBUG_TYPE "torq-conv2d-big-stride-pattern"

namespace mlir::syna::torq {

Value getSpaceToDepth(Value input, int sh, int sw, PatternRewriter &rewriter) {
    auto loc = input.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto shape = inputType.getShape();
    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];

    // Expand [n, c, h, w] -> [n, c, h/sh, sh, w/sw, sw]
    // Map: [0]->[0], [1]->[1], [2]->[2,3], [3]->[4,5]
    SmallVector<int64_t, 6> expandedShape = {n, c, h / sh, sh, w / sw, sw};
    auto expandedType = RankedTensorType::get(expandedShape, elementType);
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, input, ArrayRef<ReassociationIndices>{{0}, {1}, {2, 3}, {4, 5}}
    );

    // Transpose to [n, c, sh, sw, h/sh, w/sw]
    const SmallVector<int64_t, 6> perm = {0, 1, 3, 5, 2, 4};
    Value transposed = transposeValue(expanded, perm, input.getLoc(), rewriter);

    // Collapse [n, c, sh, sw, h/sh, w/sw] to [n, c*sh*sw, h/sh, w/sw]
    auto outType = RankedTensorType::get({n, c * sh * sw, h / sh, w / sw}, elementType);
    Value out = rewriter.create<tensor::CollapseShapeOp>(
        loc, outType, transposed, ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}, {4}, {5}}
    );

    return out;
}

struct Conv2DOpBigStride : public OpRewritePattern<syna::torq_hl::Conv2DOp> {
  private:
    typedef syna::torq_hl::Conv2DOp ConvOp;

    const bool _markFuseGroups;

  public:
    using OpRewritePattern<ConvOp>::OpRewritePattern;
    Conv2DOpBigStride(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<ConvOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(op)) {
            return rewriter.notifyMatchFailure(op, "Already marked");
        }

        const auto loc = op.getLoc();
        auto strides = op.getStride();
        Value weights = op.getWeights();
        arith::ConstantOp constOp = weights.getDefiningOp<arith::ConstantOp>();
        auto convWeights = cast<DenseElementsAttr>(constOp.getValue());
        auto weightType = llvm::cast<RankedTensorType>(convWeights.getType());
        auto weightShape = weightType.getShape().vec();
        bool supportedStrides = strides.size() == 2 && ((strides[0] == 1 && strides[1] == 1) ||
                                                        (strides[0] == 2 && strides[1] == 2));
        if (supportedStrides) {
            // We can handle these strides directly in our HW kernel, nothing to change
            return failure();
        }

        bool kernelEqStride = strides.size() == 2 && weightShape.size() == 4 &&
                              weightShape[2] == strides[0] && weightShape[3] == strides[1];
        if (!kernelEqStride) {
            // The shape of the conv kernel is different from the strides, we can't do anything
            return rewriter.notifyMatchFailure(op, "Kernel shape different from strides");
        }

        // This is a 2D conv with strides [h,w] with kern size == strides
        // Rewrite it as (space2depth + conv2d(stride1) with weights [O,h*w*I,H/sh,W/sw])
        auto weightElementType = weightType.getElementType();
        auto sh = strides[0];
        auto sw = strides[1];

        if (_markFuseGroups) {
            markOpFuseGroup(
                op, rewriter, op->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Convert input from [N, C, H, W] to [N, C*h*w, H/sh, W/sw]
        Value inToDepth = getSpaceToDepth(op.getInput(), sh, sw, rewriter);

        // Reshape weights from [OIHW] to [O,I*h*w,H/sh,W/sw]
        auto newWeightType = RankedTensorType::get(
            {weightShape[0], weightShape[1] * sh * sw, weightShape[2] / sh, weightShape[2] / sw},
            weightElementType
        );
        auto newWeightAttr = DenseElementsAttr::get(newWeightType, convWeights.getRawData());
        auto torqWeights = rewriter.create<arith::ConstantOp>(loc, newWeightType, newWeightAttr);

        // Update conv2d
        rewriter.modifyOpInPlace(op, [&]() {
            op.setOperand(op.getInputMutable().getOperandNumber(), inToDepth);
            op.setOperand(op.getWeightsMutable().getOperandNumber(), torqWeights);
            op.setStride({1, 1});
        });

        return success();
    }
};

void populateTorqHLConv2DBigStridePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<Conv2DOpBigStride>(context, markFuseGroups);
}

} // namespace mlir::syna::torq