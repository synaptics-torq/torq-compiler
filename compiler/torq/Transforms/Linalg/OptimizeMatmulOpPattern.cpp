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

#include "mlir/Dialect/Arith/IR/Arith.h"
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
    Value currentInput = input;

    auto collapseOp = dyn_cast_or_null<tensor::CollapseShapeOp>(currentInput.getDefiningOp());
    if (collapseOp)
        currentInput = collapseOp.getOperand();

    auto foldOp = dyn_cast_or_null<linalg::GenericOp>(currentInput.getDefiningOp());
    if (!foldOp)
        return;

    if (foldOp.getNumDpsInputs() != 1 || foldOp.getNumResults() != 1)
        return;

    auto resultType = dyn_cast<RankedTensorType>(foldOp.getResultTypes().front());
    auto inputType = dyn_cast<RankedTensorType>(foldOp.getInputs()[0].getType());
    if (resultType != inputType)
        return;

    if (collapseOp) {
        rewriter.modifyOpInPlace(collapseOp, [&]() { collapseOp->setOperand(0, currentInput); });
        return;
    }

    input = foldOp.getInputs()[0];
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

    // When the broadcast dimension is 1, expand_shape is a zero-copy reshape and always valid
    // regardless if the input is constant or tensor.
    // Thus, bail out only if bcast_dim != 1 and input != arith.constant.
    if (outputShape[0] != 1) {
        TypedAttr constAttr;
        if (!matchPattern(definingOp.getInputs()[0], m_Constant(&constAttr))) {
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

    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, definingOp.getLoc(), outType, definingOp.getInputs()[0], reassoc
    );

    input = expandOp.getResult();
}

/// Check if a batch_matmul input comes from a broadcast-batch pattern:
///   collapse_shape(linalg.generic(broadcast original [1,K,N] → [1,B,K,N]))
/// Returns the broadcast input index (0 or 1), or -1 if no match.
/// Also returns the original un-broadcast source via `broadcastSource`.
static int detectBroadcastBatchInput(linalg::BatchMatmulOp srcOp, Value &broadcastSource) {
    for (int i : {0, 1}) {
        Value input = srcOp.getInputs()[i];
        auto collapseOp = input.getDefiningOp<tensor::CollapseShapeOp>();
        if (!collapseOp)
            continue;
        // first reassociation group must combine exactly two dims [[0,1],...]
        auto reassoc = collapseOp.getReassociationIndices();
        if (reassoc.size() < 2 || reassoc[0].size() != 2)
            continue;
        // pre-collapse dim 0 should be 1 (the original batch-1 dim before broadcast)
        auto srcType = collapseOp.getSrcType();
        if (srcType.getDimSize(0) != 1)
            continue;
        // source of collapse_shape should be a broadcast linalg.generic
        auto genericOp = collapseOp.getSrc().getDefiningOp<linalg::GenericOp>();
        if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumResults() != 1)
            continue;
        // identity body: only a yield op
        if (genericOp.getBody()->getOperations().size() != 1)
            continue;
        // verify indexing maps are projected permutations (no transpose or shuffle)
        auto maps = genericOp.getIndexingMapsArray();
        bool validMaps = true;
        for (auto &map : maps) {
            if (!map.isProjectedPermutation()) {
                validMaps = false;
                break;
            }
        }
        if (!validMaps)
            continue;
        // verify it's actually a broadcast (output has more elements than input)
        auto inputType = cast<RankedTensorType>(genericOp.getInputs()[0].getType());
        auto outputType = cast<RankedTensorType>(genericOp.getResultTypes()[0]);
        if (inputType.getNumElements() >= outputType.getNumElements())
            continue;

        broadcastSource = genericOp.getInputs()[0];
        return i;
    }
    return -1;
}

/// Replace a batch_matmul with a broadcast-batch input by a linalg.generic that
/// performs the matmul with broadcast expressed via indexing maps. This avoids
/// materializing the broadcast buffer entirely.
///
/// batch_matmul([B,M,K], collapse([1,B,K,N])) where rhs came from broadcasting [1,K,N]
/// becomes:
///   linalg.generic {
///     indexing_maps = [(b,m,n,k)->(b,m,k), (b,m,n,k)->(0,k,n), (b,m,n,k)->(b,m,n)]
///     iterator_types = ["parallel","parallel","parallel","reduction"]
///   } ins(A:[B,M,K], B_orig:[1,K,N]) outs(C:[B,M,N]) { mulf + addf }
///
static LogicalResult replaceBatchMatmulWithBroadcastGeneric(
    linalg::BatchMatmulOp srcOp, Value broadcastSource, int broadcastIdx, PatternRewriter &rewriter
) {
    Location loc = srcOp.getLoc();

    // Determine which input is broadcast and which is the "real" batched input
    Value batchedInput = srcOp.getInputs()[1 - broadcastIdx];
    Value output = srcOp.getResults()[0];
    auto outputType = cast<RankedTensorType>(output.getType());
    auto batchedType = cast<RankedTensorType>(batchedInput.getType());

    int64_t batchSize = batchedType.getDimSize(0);
    (void)batchSize;

    // Build indexing maps for broadcast batch matmul:
    //   A[b,m,k] * B[0,k,n] -> C[b,m,n]  (if broadcastIdx == 1)
    //   A[0,m,k] * B[b,k,n] -> C[b,m,n]  (if broadcastIdx == 0)
    MLIRContext *ctx = rewriter.getContext();
    // dims: (b=d0, m=d1, n=d2, k=d3)
    AffineMap lhsMap, rhsMap, outMap;
    if (broadcastIdx == 1) {
        // lhs = A[b,m,k], rhs = B[0,k,n]
        lhsMap = AffineMap::get(
            4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)},
            ctx
        );
        rhsMap = AffineMap::get(
            4, 0,
            {getAffineConstantExpr(0, ctx), getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)}, ctx
        );
    }
    else {
        // lhs = A[0,m,k], rhs = B[b,k,n]
        lhsMap = AffineMap::get(
            4, 0,
            {getAffineConstantExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(3, ctx)}, ctx
        );
        rhsMap = AffineMap::get(
            4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(3, ctx), getAffineDimExpr(2, ctx)},
            ctx
        );
    }
    outMap = AffineMap::get(
        4, 0, {getAffineDimExpr(0, ctx), getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx
    );

    SmallVector<AffineMap> indexingMaps = {lhsMap, rhsMap, outMap};
    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel, // batch
        utils::IteratorType::parallel, // M
        utils::IteratorType::parallel, // N
        utils::IteratorType::reduction // K
    };

    // Use the existing output (from linalg.fill) as the init
    Value init = srcOp.getDpsInits()[0];

    // Order inputs as [lhs, rhs] matching the maps above
    Value lhs = (broadcastIdx == 1) ? batchedInput : broadcastSource;
    Value rhs = (broadcastIdx == 1) ? broadcastSource : batchedInput;

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, outputType, ValueRange{lhs, rhs}, ValueRange{init}, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] = lhs element, args[1] = rhs element, args[2] = accumulator
            Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
            Value add = arith::AddFOp::create(b, loc, args[2], mul);
            linalg::YieldOp::create(b, loc, add);
        }
    );

    rewriter.replaceOp(srcOp, genericOp.getResults());
    return success();
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
/// 3. Detects broadcast-batch inputs and replaces the batch_matmul with a
///    linalg.generic that expresses the broadcast via indexing maps, avoiding
///    materialization of the broadcast buffer
/// 4. Updates the matmul operation's operands to point to the optimized inputs
/// 5. Marks the operation as optimized to prevent redundant passes
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

        // Detect broadcast-batch pattern and replace with a linalg.generic that
        // expresses the broadcast via indexing maps. This avoids materializing
        // the broadcast buffer and lets tile-and-fuse handle it naturally.
        Value broadcastSource;
        int broadcastIdx = detectBroadcastBatchInput(srcOp, broadcastSource);
        if (broadcastIdx >= 0) {
            return replaceBatchMatmulWithBroadcastGeneric(
                srcOp, broadcastSource, broadcastIdx, rewriter
            );
        }

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
