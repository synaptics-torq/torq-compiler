// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-matmul-op-pattern"

namespace mlir::syna::torq {

/// Folds input operations by removing unnecessary collapse shape and generic operations.
///
/// This function optimizes input values by:
/// 1. Removing tensor::CollapseShapeOp if present and storing it for later restoration
/// 2. Checking if the input is produced by a linalg::GenericOp with single input and output
/// 3. Verifying that the generic operation preserves the tensor type
/// 4. Replacing the input with the generic operation's input
/// 5. Restoring the collapse shape operation around the folded input if it existed
///
/// @param input [in/out] The input value to fold. Modified in-place to point to the folded value.
/// @param rewriter [in] The pattern rewriter used to modify operations in-place.
///
/// @note This function modifies the input operand and any related collapse shape operations.
///       It performs no rewrites if the input is not produced by the expected operation patterns.
///
static void foldInput(Value &input, PatternRewriter &rewriter) {
    auto collapseOp = dyn_cast_or_null<tensor::CollapseShapeOp>(input.getDefiningOp());
    if (collapseOp) {
        input = input.getDefiningOp()->getOperand(0);
    }

    auto foldOp = dyn_cast_or_null<linalg::GenericOp>(input.getDefiningOp());
    if (!foldOp) {
        return;
    }
    if (foldOp.getNumDpsInputs() != 1 || foldOp.getNumResults() != 1) {
        return;
    }

    auto resultType = dyn_cast<RankedTensorType>(foldOp.getResultTypes().front());
    auto inputType = dyn_cast<RankedTensorType>(foldOp.getInputs()[0].getType());
    if (resultType != inputType) {
        return;
    }
    input = foldOp.getInputs()[0];

    if (collapseOp) {
        rewriter.modifyOpInPlace(collapseOp, [&]() { collapseOp->setOperand(0, input); });
        input = collapseOp.getResult();
    }
}

/// Checks if input is produced by a linalg::GenericOp with a constant input and expands it.
///
/// This function optimizes constant inputs by:
/// 1. Verifying the input is produced by a linalg::GenericOp with single input and output
/// 2. Checking if the generic operation's input is a constant value
/// 3. Validating output rank equals input rank + 1 with matching dimensions
/// 4. Creating a tensor::ExpandShapeOp to replace the generic operation
/// 5. Updating the input to point to the expanded result
///
/// @param input [in/out] The input value to optimize. Modified in-place if a constant pattern is
/// found.
/// @param rewriter [in] The pattern rewriter used to create new operations.
///
/// @note Only modifies input if it matches the constant expansion pattern.
///
static void expandConstantInput(Value &input, PatternRewriter &rewriter) {
    auto definingOp = dyn_cast_or_null<linalg::GenericOp>(input.getDefiningOp());
    if (!definingOp) {
        return;
    }
    if (definingOp.getNumDpsInputs() != 1 || definingOp.getNumResults() != 1) {
        return;
    }
    TypedAttr constAttr;
    if (!matchPattern(definingOp.getInputs()[0], m_Constant(&constAttr))) {
        return;
    }

    auto resultType = dyn_cast<RankedTensorType>(definingOp.getResultTypes().front());
    auto inputType = dyn_cast<RankedTensorType>(definingOp.getInputs()[0].getType());

    if (resultType == inputType)
        return;

    // to check if output rank = input rank + 1
    // output has more dim than input shape
    auto outputShape = resultType.getShape();
    auto inputShape = inputType.getShape();
    if (outputShape.size() != inputShape.size() + 1) {
        return;
    }
    for (auto it : llvm::zip(inputShape, outputShape.drop_front())) {
        if (std::get<0>(it) != std::get<1>(it)) {
            return;
        }
    }

    // create a new expandshapeOp to replace the genericOp
    SmallVector<int64_t> newShape(outputShape.begin(), outputShape.end());
    auto elementType = inputType.getElementType();
    auto outType = RankedTensorType::get(newShape, elementType);

    SmallVector<ReassociationIndices> reassoc;
    int rank = outputShape.size();
    if (rank <= 1) {
        for (int i = 0; i < rank; ++i)
            reassoc.push_back({i});
    }
    else {
        reassoc.push_back({0, 1});
        for (int i = 2; i < rank; ++i) {
            reassoc.push_back({i});
        }
    }

    auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
        definingOp.getLoc(), outType, definingOp.getInputs()[0], reassoc
    );

    input = expandOp.getResult();
}

/// Optimization pattern for batch matrix multiplication operations.
///
/// This pattern optimizes linalg::BatchMatmulOp by folding unnecessary operations
/// on inputs that may have been added during upstream transformations from torch,
/// onnx, or tosa dialects. The optimization removes redundant reshape, collapse,
/// and generic operations to simplify subsequent lowering and avoid unnecessary
/// broadcast operations.
///
/// The pattern applies the following optimizations in sequence:
/// 1. Folds input operations by removing unnecessary collapse shape wrappers
/// 2. Expands constant inputs that were wrapped in generic operations
/// 3. Updates the matmul operation's operands to point to the optimized inputs
/// 4. Marks the operation as optimized to prevent redundant passes
///
class BatchMatmulOpPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp srcOp, PatternRewriter &rewriter) const {

        if (srcOp->getAttrOfType<BoolAttr>("matmulOptimized")) {
            // Avoid applying the same transformation twice.
            return failure();
        }

        Value input0 = srcOp.getInputs()[0];
        Value input1 = srcOp.getInputs()[1];

        /// TODO: Replace the sequential foldInput calls with a while-loop based approach to
        /// iteratively apply folding transformations until a fixed point is reached. This would
        /// handle cases where folding one input enables additional folding opportunities on the
        /// other input or vice versa, ensuring more aggressive optimization and potentially
        /// exposing additional canonicalization opportunities that single-pass folding would miss.
        foldInput(input0, rewriter);
        foldInput(input1, rewriter);

        expandConstantInput(input0, rewriter);
        expandConstantInput(input1, rewriter);

        // modify in-place the two inputs of the matmulOp
        rewriter.modifyOpInPlace(srcOp, [&]() {
            srcOp->setOperand(0, input0);
            srcOp->setOperand(1, input1);
        });

        srcOp->setAttr("matmulOptimized", BoolAttr::get(srcOp->getContext(), true));

        return success();
    }
};

void populateOptimizeMatmuOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<BatchMatmulOpPattern>(context);
}

} // namespace mlir::syna::torq