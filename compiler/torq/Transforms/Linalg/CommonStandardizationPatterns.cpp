
// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-elementwise-binary-op-pattern"

namespace mlir::syna::torq {

namespace {

// True for a rank-1 single-element input with source map `(d...) -> (0)`,
// as emitted by upstream's `tosa-to-linalg-named` lowering.
bool isUnitConstSrcInput(linalg::GenericOp op, OpOperand &input) {
    assert(input.getOwner() == op.getOperation() && "input must be an operand of op");
    auto rtt = dyn_cast<RankedTensorType>(input.get().getType());
    if (!rtt || rtt.getRank() != 1 || !rtt.hasStaticShape() || rtt.getDimSize(0) != 1)
        return false;
    AffineMap srcMap = op.getMatchingIndexingMap(&input);
    if (srcMap.getNumResults() != 1)
        return false;
    auto cst = dyn_cast<AffineConstantExpr>(srcMap.getResult(0));
    return cst && cst.getValue() == 0;
}

bool isElementwiseNIn1Out(linalg::GenericOp srcOp) {
    if (srcOp.getNumResults() != 1) {
        return false;
    }
    if (srcOp.getNumDpsInits() != 1) {
        return false;
    }
    if (srcOp.getNumReductionLoops() != 0) {
        return false;
    }
    if (srcOp.getNumParallelLoops() != srcOp.getNumLoops()) {
        return false;
    }

    AffineMap outMap = srcOp.getMatchingIndexingMap(srcOp.getDpsInitOperand(0));

    for (OpOperand *inputOperand : srcOp.getDpsInputOperands()) {
        Value v = inputOperand->get();
        auto rtt = dyn_cast<RankedTensorType>(v.getType());
        if (!rtt) {
            return false;
        }
        if (rtt.getRank() == 0) {
            continue;
        }
        if (isUnitConstSrcInput(srcOp, *inputOperand)) {
            continue;
        }
        if (srcOp.getMatchingIndexingMap(inputOperand) != outMap) {
            return false;
        }
    }
    return true;
}
} // namespace

/// The reshape that consumes `value` as its source as its only use, or null.
static Operation *nextReshapeInChain(Value value) {
    if (!value.hasOneUse() || value.use_begin()->getOperandNumber() != 0)
        return nullptr;
    Operation *next = value.use_begin()->getOwner();
    return isa<tensor::ReshapeOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp>(next) ? next
                                                                                        : nullptr;
}

/// Number of non-unit dims (at least 1).
static int64_t reducedRank(RankedTensorType type) {
    return std::max<int64_t>(1, llvm::count_if(type.getShape(), [](int64_t d) { return d != 1; }));
}

/// Standardizes reshapes into tensor.collapse_shape / tensor.expand_shape: a
/// chain of consecutive reshape ops (tensor.reshape, collapse_shape,
/// expand_shape, each the single-use source of the next) starting at the
/// anchor is replaced by its net reshape, emitted as a single collapse/expand
/// or, for a mid-shape net, as a collapse+expand pair through the reduced
/// shape. A lone tensor.reshape is a one-op chain.
template <typename ReshapeOpTy>
class ReshapeToCollapseExpand : public OpRewritePattern<ReshapeOpTy> {
  public:
    ReshapeToCollapseExpand(MLIRContext *ctx) : OpRewritePattern<ReshapeOpTy>(ctx) {
        this->setDebugName("ReshapeToCollapseExpand");
    }

    LogicalResult matchAndRewrite(ReshapeOpTy op, PatternRewriter &rewriter) const override {
        SmallVector<Operation *> chain = {op};
        while (Operation *next = nextReshapeInChain(chain.back()->getResult(0)))
            chain.push_back(next);
        for (Operation *link : chain) {
            auto srcTy = dyn_cast<RankedTensorType>(link->getOperand(0).getType());
            auto dstTy = dyn_cast<RankedTensorType>(link->getResult(0).getType());
            if (!srcTy || !dstTy || !srcTy.hasStaticShape() || !dstTy.hasStaticShape())
                return rewriter.notifyMatchFailure(op, "expected static shapes");
        }

        auto srcTy = cast<RankedTensorType>(chain.front()->getOperand(0).getType());
        auto dstTy = cast<RankedTensorType>(chain.back()->getResult(0).getType());
        bool hasReshape = llvm::any_of(chain, llvm::IsaPred<tensor::ReshapeOp>);
        // A lone tensor.reshape that changes the number of non-unit dims is left
        // to its consumers; a lone collapse/expand is already standard.
        if (chain.size() == 1 && (!hasReshape || reducedRank(srcTy) != reducedRank(dstTy)))
            return rewriter.notifyMatchFailure(op, "single reshape left as is");
        // A collapse+expand pair whose net is mid-shape is already at its 2-op
        // floor; refolding it would ping-pong in the greedy driver.
        if (chain.size() == 2 && !hasReshape && srcTy != dstTy &&
            !getReassociationIndicesForReshape(srcTy, dstTy))
            return rewriter.notifyMatchFailure(op, "2-op chain at mid-shape floor");

        Value replacement =
            emitValuePreservingReshape(rewriter, op.getLoc(), op->getOperand(0), srcTy, dstTy);
        rewriter.replaceAllUsesWith(chain.back()->getResult(0), replacement);
        // Erase tail-first: every intermediate result is used only by the next op.
        for (Operation *link : llvm::reverse(chain))
            rewriter.eraseOp(link);
        return success();
    }
};

void populateCommonStandardizationPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<
        ReshapeToCollapseExpand<tensor::ReshapeOp>,
        ReshapeToCollapseExpand<tensor::CollapseShapeOp>,
        ReshapeToCollapseExpand<tensor::ExpandShapeOp>>(context);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    patterns.add<ComposeExpandOfCollapseOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp>>(context
    );
}
} // namespace mlir::syna::torq