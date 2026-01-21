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

/// Matches a tensor.expand_shape operation that splits the last dimension into two unit dimensions.
/// Works for tensors of arbitrary rank by splitting the last dimension [1] into [1, 1].
///
/// Pattern:
///   %expanded = tensor.expand_shape %input [[0], [1], ..., [n-1], [n, n+1]]
///   output_shape [d0, d1, ..., d_{n-1}, 1, 1]
///   : tensor<d0xd1x...xd_{n-1}x1xf32> into tensor<d0xd1x...xd_{n-1}x1x1xf32>
///
/// Example (rank 3 input):
///   %expanded = tensor.expand_shape %7 [[0], [1], [2, 3]] output_shape [1, 8, 1, 1]
///   : tensor<1x8x1xf32> into tensor<1x8x1x1xf32>
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

    // Verify the last dimension of input is 1 (to be split into [1, 1])
    if (inputType.getDimSize(inputRank - 1) != 1) {
        return nullptr;
    }

    // Verify the last two dimensions of output are both 1
    if (outputType.getDimSize(inputRank - 1) != 1 || outputType.getDimSize(inputRank) != 1) {
        return nullptr;
    }

    return expandOp.getSrc();
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

void populateRaiseSoftmaxOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseSoftmaxOnnx>(context);
}

} // namespace mlir::syna::torq
