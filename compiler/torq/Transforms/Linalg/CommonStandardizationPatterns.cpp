
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
/// @brief Rewrite scalar-like broadcast inputs of an elementwise generic into
/// a uniform rank-1 form so broadcast pattern matchers pick them up.
///
/// A scalar-like input is a single value broadcast across the iteration space.
/// Two equivalent forms occur:
///   1. Rank-0 input (`tensor<T>`).
///   2. Rank-1 single-element input (`tensor<1xT>`) with source map
///      `(d...) -> (0)`.
///
/// Both are rewritten to a `tensor<1xT>` input with source map
/// `(d...) -> (d_k)` where `d_k` is a unit output dim.
class PromoteScalarsTo1D : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const {

        // Avoid broadcasted inputs
        if (srcOp->getAttrOfType<BoolAttr>("broadcasted")) {
            return failure();
        }

        if (!isElementwiseNIn1Out(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "not elementwise N-in 1-out linalg.generic");
        }

        auto resultType = dyn_cast<RankedTensorType>(srcOp.getResultTypes().front());
        if (!resultType) {
            return rewriter.notifyMatchFailure(srcOp, "expected ranked tensor result type\n");
        }

        if (resultType.getRank() == 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "result is scalar, no need to promote scalars\n"
            );
        }

        if (srcOp.getNumReductionLoops() != 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "only handle elementwise generics without reductions\n"
            );
        }

        SmallVector<AffineMap> indexingMaps = srcOp.getIndexingMapsArray();
        SmallVector<Value> newOperands(srcOp->getOperands().begin(), srcOp->getOperands().end());
        // Collect inputs to rewrite: rank-0 (needs rank promotion + map rewrite),
        // or rank-1 unit-extent with const-0 source map (needs only map rewrite).
        SmallVector<unsigned> inputIdxs;
        SmallVector<bool> needsRankPromotion;
        for (auto it : llvm::enumerate(srcOp.getDpsInputs())) {
            auto rtt = dyn_cast<RankedTensorType>(it.value().getType());
            if (!rtt)
                return rewriter.notifyMatchFailure(srcOp, "non-ranked input\n");
            if (rtt.getRank() == 0) {
                inputIdxs.push_back(it.index());
                needsRankPromotion.push_back(true);
                continue;
            }
            if (isUnitConstSrcInput(srcOp, *srcOp.getDpsInputOperand(it.index()))) {
                inputIdxs.push_back(it.index());
                needsRankPromotion.push_back(false);
            }
        }
        if (inputIdxs.empty()) {
            return rewriter.notifyMatchFailure(srcOp, "no input to rewrite\n");
        }

        // Pick a unit output dim to pair with the size-1 source dim. Without one,
        // the rank-1 source can't be expressed as a `linalg.broadcast` source.
        int64_t unitDim = -1;
        for (int64_t d = 0, e = resultType.getRank(); d < e; ++d) {
            if (!resultType.isDynamicDim(d) && resultType.getDimSize(d) == 1) {
                unitDim = d;
                break;
            }
        }
        if (unitDim < 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "no unit output dim to pair with the rank-1 source\n"
            );
        }

        const int64_t numLoops = srcOp.getNumLoops();
        auto unitDimMap = AffineMap::get(
            /*dimCount=*/numLoops,
            /*symbolCount=*/0,
            /*results=*/ArrayRef<AffineExpr>{rewriter.getAffineDimExpr(unitDim)},
            rewriter.getContext()
        );

        for (auto [dpsIdx, promote] : llvm::zip(inputIdxs, needsRankPromotion)) {
            unsigned operandNumber = srcOp.getDpsInputOperand(dpsIdx)->getOperandNumber();
            indexingMaps[operandNumber] = unitDimMap;
            if (!promote)
                continue;
            Value input = srcOp.getDpsInputs()[dpsIdx];
            if (failed(promoteScalar(srcOp, input, rewriter))) {
                return rewriter.notifyMatchFailure(
                    srcOp, "failed to promote input " + Twine(dpsIdx) + "\n"
                );
            }
            newOperands[operandNumber] = input;
        }

        rewriter.modifyOpInPlace(srcOp, [&]() {
            for (auto [dpsIdx, promote] : llvm::zip(inputIdxs, needsRankPromotion)) {
                if (!promote)
                    continue;
                unsigned operandNumber = srcOp.getDpsInputOperand(dpsIdx)->getOperandNumber();
                srcOp->setOperand(operandNumber, newOperands[operandNumber]);
            }
            srcOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(indexingMaps));
        });

        return success();
    }
};

class ReshapeToCollapseExpand : public OpRewritePattern<tensor::ReshapeOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult matchAndRewrite(tensor::ReshapeOp op, PatternRewriter &rewriter) const {
        //////////Support Functions///////////
        auto calcRank = [&](ArrayRef<int64_t> srcShape) {
            int rank = 0;
            for (auto dim : srcShape) {
                if (dim != 1)
                    rank++;
            }
            if (rank == 0)
                rank = 1;
            return rank;
        };

        auto findCollapseShape = [](ArrayRef<int64_t> shape) {
            SmallVector<int64_t> collapseShape;
            for (auto d : shape) {
                if (d == 1) {
                    continue;
                }
                collapseShape.push_back(d);
            }
            if (collapseShape.empty()) {
                collapseShape.push_back(1);
            }
            return collapseShape;
        };
        //////////////End of Support Functions//////////////

        auto srcType = mlir::dyn_cast<RankedTensorType>(op.getSource().getType());
        auto dstType = mlir::dyn_cast<RankedTensorType>(op.getResult().getType());
        if (!srcType || !dstType || !srcType.hasStaticShape() || !dstType.hasStaticShape())
            return failure();

        ArrayRef<int64_t> srcShape = srcType.getShape();
        ArrayRef<int64_t> dstShape = dstType.getShape();
        if (calcRank(srcShape) != calcRank(dstShape)) {
            return failure();
        }

        const int srcRank = srcType.getRank();
        const int dstRank = dstType.getRank();
        if (auto direct = getReassociationIndicesForReshape(srcType, dstType)) {
            Value out;
            if (dstRank < srcRank) {
                out = tensor::CollapseShapeOp::create(
                    rewriter, op.getLoc(), dstType, op.getSource(), *direct
                );
            }
            else if (dstRank > srcRank) {
                out = tensor::ExpandShapeOp::create(
                    rewriter, op.getLoc(), dstType, op.getSource(), *direct
                );
            }
            /// same rank: fall through to mid-shape decomposition

            if (out) {
                rewriter.replaceOp(op, out);
                return success();
            }
        }

        SmallVector<int64_t> collapseShape = findCollapseShape(srcShape);
        SmallVector<int64_t> dstCollapseShape = findCollapseShape(dstShape);
        if (collapseShape != dstCollapseShape) {
            op.emitError("ReshapeToCollapseExpand: cannot optimize reshape op due to different "
                         "collapse shape");
            return failure();
        }

        auto collapseSrc = getReassociationIndicesForCollapse(srcShape, collapseShape);
        auto collapseDst = getReassociationIndicesForCollapse(dstShape, dstCollapseShape);
        if (!collapseSrc || !collapseDst) {
            return failure();
        }

        auto midType = RankedTensorType::get(collapseShape, srcType.getElementType());
        Value collapsed = tensor::CollapseShapeOp::create(
            rewriter, op.getLoc(), midType, op.getSource(), *collapseSrc
        );
        Value expanded =
            tensor::ExpandShapeOp::create(rewriter, op.getLoc(), dstType, collapsed, *collapseDst);

        rewriter.replaceOp(op, expanded);
        return success();
    }
};

void populateCommonStandardizationPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<PromoteScalarsTo1D>(context);
    patterns.add<ReshapeToCollapseExpand>(context);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    patterns.add<ComposeExpandOfCollapseOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp>>(context
    );
}
} // namespace mlir::syna::torq