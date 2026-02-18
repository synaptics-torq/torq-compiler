// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-raise-sofmax"

namespace mlir::syna::torq {

bool checkIdentityLikeIndexingMaps(linalg::GenericOp op) {
    for (auto [operandIdx, indexingMap] : llvm::enumerate(op.getIndexingMapsArray())) {
        if (indexingMap.isIdentity()) {
            continue;
        }

        for (auto [idx, result] : llvm::enumerate(indexingMap.getResults())) {
            if (auto dim = dyn_cast<AffineDimExpr>(result)) {
                if (dim.getPosition() != idx) {
                    return false;
                }
            }
            else if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
                if (constExpr.getValue() != 0) {
                    return false;
                }
                auto operandType = op.getOperand(operandIdx).getType();
                if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
                    if (shapedType.getDimSize(idx) != 1) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <class T> SmallVector<Value> findElementwiseOpInputs(Value value) {
    SmallVector<Value> inputs;
    auto op = value.getDefiningOp<linalg::GenericOp>();
    if (!op) {
        return inputs;
    }
    // Make sure there is only 1 output
    if (op.getNumDpsInits() != 1) {
        return inputs;
    }
    // Make sure there is no permutations or other indexing maps than identity
    if (!checkIdentityLikeIndexingMaps(op)) {
        return inputs;
    }
    // Make sure the terminator of the generic op returns the result of a truncf op
    auto innerOp = op.getBody()->getTerminator()->getOperand(0).getDefiningOp<T>();
    if (!innerOp) {
        return inputs;
    }
    if (innerOp->getNumOperands() != op.getNumDpsInputs()) {
        return inputs;
    }
    for (unsigned i = 0; i < op.getNumDpsInputs(); ++i) {
        auto innerInput = dyn_cast<BlockArgument>(innerOp->getOperands()[i]);
        // Make sure the inner op input(s) are the input(s) of the generic op
        if (!innerInput || innerInput.getArgNumber() != i ||
            innerInput.getOwner() != op.getBody()) {
            return inputs;
        }
        inputs.push_back(op.getInputs()[i]);
    }
    return inputs;
}

template <class T>
SmallVector<Value> findReductionOpInputs(Value value, unsigned int reductionDim) {
    SmallVector<Value> inputs;
    auto op = value.getDefiningOp<linalg::GenericOp>();
    if (!op) {
        return inputs;
    }
    // Make sure the reduction dim is valid
    if (reductionDim >= op.getNumLoops()) {
        return inputs;
    }
    // Make sure the reduction dim is valid
    if (op.getIteratorTypesArray()[reductionDim] != utils::IteratorType::reduction) {
        return inputs;
    }
    // Make sure there is only 1 output
    if (op.getNumDpsInits() != 1) {
        return inputs;
    }
    // Make sure there is no permutations or other indexing maps than identity
    if (!checkIdentityLikeIndexingMaps(op)) {
        return inputs;
    }
    // Make sure the terminator of the generic op returns the result of a truncf op
    auto innerOp = op.getBody()->getTerminator()->getOperand(0).getDefiningOp<T>();
    if (!innerOp) {
        return inputs;
    }
    if (innerOp.getNumOperands() != op.getNumOperands()) {
        return inputs;
    }
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto innerOperand = dyn_cast<BlockArgument>(innerOp->getOperands()[i]);
        // Make sure the inner op operands are the operands of the generic op
        if (!innerOperand || innerOperand.getArgNumber() != i ||
            innerOperand.getOwner() != op.getBody()) {
            return SmallVector<Value>();
        }
        if (op.isDpsInput(&op->getOpOperand(i))) {
            inputs.push_back(op->getOperand(i));
        }
    }
    return inputs;
}

/// Matches a linalg.generic operation that broadcasts along the last dimension.
/// Pattern:
///   %broadcast = linalg.generic
///     {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
///                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
///      iterator_types = ["parallel", "parallel", "parallel"]}
///     ins(%input : tensor<1x1916xf32>) outs(%out : tensor<1x1916x2xf32>) {
///       ^bb0(%in: f32, %out: f32):
///         linalg.yield %in : f32
///     } -> tensor<1x1916x2xf32>
///
/// @param value The value to check if it's a broadcast operation
/// @return The input to the broadcast operation if matched, nullptr otherwise
Value matchBroadcastOp(Value value) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp) {
        return nullptr;
    }

    // Should have exactly one input
    if (genericOp.getNumDpsInputs() != 1) {
        return nullptr;
    }

    // Check that body just yields the input
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return nullptr;
    }

    auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != genericOp.getBody()) {
        return nullptr;
    }

    // Verify it's a broadcast along the last dimension
    auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());

    if (!inputType || !outputType) {
        return nullptr;
    }

    // Output should have one more dimension than input
    if (outputType.getRank() != inputType.getRank() + 1) {
        return nullptr;
    }

    return genericOp.getInputs()[0];
}

/// Matches a tensor.expand_shape operation that adds a trailing dimension of size 1.
/// Supports two patterns:
/// 1. Splitting last dimension [1] into [1, 1]
/// 2. Splitting last dimension [N] into [N, 1]
///
/// Pattern 1:
///   %expanded = tensor.expand_shape %input [[0], [1], ..., [n-1], [n, n+1]]
///   : tensor<d0xd1x...xd_{n-1}x1xf32> into tensor<d0xd1x...xd_{n-1}x1x1xf32>
///
/// Pattern 2:
///   %expanded = tensor.expand_shape %input [[0], [1, 2]]
///   : tensor<1x1916xf32> into tensor<1x1916x1xf32>
///
/// @param value The value to check if it's an expand_shape operation
/// @return The input to the expand_shape operation if matched, nullptr otherwise
Value matchExpandShapeOp(Value value) {
    auto expandOp = value.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
        return nullptr;
    }

    auto inputType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
    if (!inputType) {
        return nullptr;
    }

    auto outputType = dyn_cast<RankedTensorType>(expandOp.getResult().getType());
    if (!outputType) {
        return nullptr;
    }

    // Verify output rank is exactly one more than input rank
    if (outputType.getRank() != inputType.getRank() + 1) {
        return nullptr;
    }

    auto reassociations = expandOp.getReassociationIndices();
    int inputRank = inputType.getRank();

    // Verify the reassociation groups: [[0], [1], ..., [n-1], [n, n+1]]
    // Should have (inputRank) groups total
    if (reassociations.size() != inputRank) {
        return nullptr;
    }

    // Check all groups except the last one: each should contain exactly one dimension
    for (int i = 0; i < inputRank - 1; ++i) {
        if (reassociations[i].size() != 1 || reassociations[i][0] != i) {
            return nullptr;
        }
    }

    // Check the last group: should contain exactly two consecutive dimensions
    int lastGroupIdx = inputRank - 1;
    if (reassociations[lastGroupIdx].size() != 2 ||
        reassociations[lastGroupIdx][0] != lastGroupIdx ||
        reassociations[lastGroupIdx][1] != lastGroupIdx + 1) {
        return nullptr;
    }

    // Verify the last dimension of output is 1 (the newly added trailing dimension)
    if (outputType.getDimSize(inputRank) != 1) {
        return nullptr;
    }

    // Verify the second-to-last output dimension matches the last input dimension
    if (outputType.getDimSize(inputRank - 1) != inputType.getDimSize(inputRank - 1)) {
        return nullptr;
    }

    return expandOp.getSrc();
}

/// Matches either a tensor.expand_shape or linalg.generic broadcast operation
/// @param value The value to check
/// @return The input to the broadcast/expand operation if matched, nullptr otherwise
Value matchBroadcastOrExpandShape(Value value) {
    Value result = matchExpandShapeOp(value);
    if (result) {
        return result;
    }
    return matchBroadcastOp(value);
}

/// Matches a linalg.generic that computes reciprocal: 1.0 / input
/// The constant 1.0 can be either:
/// 1. A captured value from outside the block (e.g., %cst_3)
/// 2. A constant defined inside the block
///
/// Pattern:
///   %recip = linalg.generic ins(%sum) outs(%out) {
///     ^bb0(%in: f32, %out: f32):
///       %result = arith.divf %cst_one, %in : f32
///       linalg.yield %result : f32
///   }
///
/// @param value The value to check if it's a reciprocal operation
/// @return The input to the reciprocal (the denominator) if matched, nullptr otherwise
Value matchReciprocalOp(Value value) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp) {
        return nullptr;
    }

    // Should have exactly one input
    if (genericOp.getNumDpsInputs() != 1) {
        return nullptr;
    }

    // Should have exactly one output
    if (genericOp.getNumDpsInits() != 1) {
        return nullptr;
    }

    // Check the body contains divf yielded directly
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return nullptr;
    }

    auto divOp = yieldOp.getOperand(0).getDefiningOp<arith::DivFOp>();
    if (!divOp) {
        return nullptr;
    }

    // Check that the divisor (rhs) is the block argument corresponding to the input
    auto rhsArg = dyn_cast<BlockArgument>(divOp.getRhs());
    if (!rhsArg || rhsArg.getArgNumber() != 0 || rhsArg.getOwner() != genericOp.getBody()) {
        return nullptr;
    }

    // Check that the numerator (lhs) is a constant 1.0
    // It could be a captured value from outside or a constant inside the block
    Value lhs = divOp.getLhs();

    // Try to get the constant value - could be from inside or outside the block
    FloatAttr constValue;
    if (auto constOp = lhs.getDefiningOp<arith::ConstantOp>()) {
        constValue = dyn_cast<FloatAttr>(constOp.getValue());
    }

    if (!constValue || !constValue.getValue().isExactlyValue(1.0)) {
        return nullptr;
    }

    return genericOp.getInputs()[0];
}

class RaiseSoftmaxOnnx : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {

        auto truncInput = findElementwiseOpInputs<arith::TruncFOp>(op.getResult(0));
        if (truncInput.size() != 1) {
            return failure();
        }
        auto divInputs = findElementwiseOpInputs<arith::DivFOp>(truncInput[0]);
        if (divInputs.size() != 2) {
            return failure();
        }
        auto expInputs = findElementwiseOpInputs<math::ExpOp>(divInputs[0]);
        if (expInputs.size() != 1) {
            return failure();
        }
        auto addInputs = findReductionOpInputs<arith::AddFOp>(divInputs[1], 3);
        if (addInputs.size() != 1) {
            return failure();
        }
        if (addInputs[0] != divInputs[0]) {
            return failure();
        }
        auto subInputs = findElementwiseOpInputs<arith::SubFOp>(expInputs[0]);
        if (subInputs.size() != 2) {
            return failure();
        }
        auto expandInput = matchExpandShapeOp(subInputs[1]);
        if (!expandInput) {
            return failure();
        }
        auto maxInputs = findReductionOpInputs<arith::MaximumFOp>(expandInput, 3);
        if (maxInputs.size() != 1) {
            return failure();
        }
        if (maxInputs[0] != subInputs[0]) {
            return failure();
        }
        auto extInputs = findElementwiseOpInputs<arith::ExtFOp>(subInputs[0]);
        if (extInputs.size() != 1) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
            op, op.getResultTypes(), extInputs[0], op.getOutputs()[0], 3
        );

        return success();
    }
};

/// Matches the decomposed softmax pattern and rewrites it to linalg.softmax
///
/// Softmax formula: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// Pattern matched (working backwards from result):
///   %result = mul(%exp, %recip)
///   %recip = 1.0 / %sum_expanded
///   %sum_expanded = expand_shape(%sum)
///   %sum = reduce_add(%exp)
///   %exp = exp(%sub)
///   %sub = %input - %max_expanded
///   %max_expanded = expand_shape(%max)
///   %max = reduce_max(%input)
///
/// Constraints verified:
///   - sum and max reduce the same dimension
///   - max and sub operate on the same input
///   - sum reduces the exp output
///
/// If the original tosa.mul has shift != 0, it lowers to arith.mulf + shift operations, which
/// won't match the simple arith::MulFOp pattern here.
class RaiseDecomposedSoftmax : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        // Match the final multiplication (exp * reciprocal)
        auto mulInputs = findElementwiseOpInputs<arith::MulFOp>(op.getResult(0));
        if (mulInputs.size() != 2) {
            return failure();
        }

        // Multiplication is commutative, so try both orderings
        // First try: mulInputs[0] = exp, mulInputs[1] = recip
        // Second try: mulInputs[0] = recip, mulInputs[1] = exp
        for (int ordering = 0; ordering < 2; ++ordering) {
            Value expValue = mulInputs[ordering];
            Value recipBroadcastValue = mulInputs[1 - ordering];

            // Match reciprocal broadcast (could be a linalg.generic that broadcasts a collapsed
            // reciprocal)
            Value recipValue = matchBroadcastOrExpandShape(recipBroadcastValue);
            if (!recipValue) {
                recipValue = recipBroadcastValue;
            }
            else {
                auto collapseOp = recipValue.getDefiningOp<tensor::CollapseShapeOp>();
                if (collapseOp) {
                    recipValue = collapseOp.getSrc();
                }
            }

            // Match reciprocal computation (1.0 / sum)
            Value sumExpandedValue = matchReciprocalOp(recipValue);
            if (!sumExpandedValue) {
                continue; // Try other ordering
            }

            // Match sum expansion (tensor.expand_shape)
            Value sumReducedValue = matchExpandShapeOp(sumExpandedValue);
            if (!sumReducedValue) {
                continue;
            }

            // Match sum reduction (linalg.reduce with addf)
            auto sumReduceOp = sumReducedValue.getDefiningOp<linalg::ReduceOp>();
            if (!sumReduceOp) {
                continue;
            }

            auto sumYield = dyn_cast<linalg::YieldOp>(sumReduceOp.getBody()->getTerminator());
            if (!sumYield || sumYield.getNumOperands() != 1) {
                continue;
            }
            auto sumAddOp = sumYield.getOperand(0).getDefiningOp<arith::AddFOp>();
            if (!sumAddOp) {
                continue;
            }

            auto sumDimensions = sumReduceOp.getDimensions();
            if (sumDimensions.size() != 1) {
                continue;
            }
            int64_t reductionDim = sumDimensions[0];

            // Verify input to sum is exp
            if (sumReduceOp.getInputs()[0] != expValue) {
                continue;
            }

            // Match exponential (math.exp)
            auto expInputs = findElementwiseOpInputs<math::ExpOp>(expValue);
            if (expInputs.size() != 1) {
                continue;
            }
            Value subValue = expInputs[0];

            // Match subtraction (input - max)
            auto subInputs = findElementwiseOpInputs<arith::SubFOp>(subValue);
            if (subInputs.size() != 2) {
                continue;
            }
            Value inputValue = subInputs[0];
            Value maxBroadcastValue = subInputs[1];

            // Match max broadcast (tensor.expand_shape or linalg.generic broadcast)
            Value maxReducedValue = matchBroadcastOrExpandShape(maxBroadcastValue);
            if (!maxReducedValue) {
                continue;
            }

            // Match max reduction (linalg.reduce with maximumf)
            auto maxReduceOp = maxReducedValue.getDefiningOp<linalg::ReduceOp>();
            if (!maxReduceOp) {
                continue;
            }

            auto maxYield = dyn_cast<linalg::YieldOp>(maxReduceOp.getBody()->getTerminator());
            if (!maxYield || maxYield.getNumOperands() != 1) {
                continue;
            }
            auto maxOp = maxYield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
            if (!maxOp) {
                continue;
            }

            auto maxDimensions = maxReduceOp.getDimensions();
            if (maxDimensions.size() != 1 || maxDimensions[0] != reductionDim) {
                continue;
            }

            // Verify input to max is the same as input to subtraction
            if (maxReduceOp.getInputs()[0] != inputValue) {
                continue;
            }

            // All constraints matched - rewrite to linalg.softmax
            auto softmaxOp = rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
                op, op.getResultTypes(), inputValue, op.getOutputs()[0], reductionDim
            );
            setTargetExecutorIfForced(softmaxOp, rewriter, "softmax");
            return success();
        }

        return failure();
    }
};

void populateRaiseSoftmaxOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseSoftmaxOnnx>(context);
    patterns.add<RaiseDecomposedSoftmax>(context);
}

} // namespace mlir::syna::torq
