// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ShapeUtils.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <vector>

#define DEBUG_TYPE "torq-shape-utils"

namespace mlir::syna::torq {

static void broadcastInput(
    linalg::LinalgOp srcOp, Value &input, SmallVector<int64_t> &dims,
    ArrayRef<int64_t> inputNewShape, ArrayRef<int64_t> outputShape,
    llvm::ArrayRef<AffineMap> newIndexingMaps, PatternRewriter &rewriter
) {
    /////////Support Functions///////////

    auto addReshapeOp = [](Value &input, ArrayRef<int64_t> newShape, PatternRewriter &rewriter) {
        auto type = dyn_cast<RankedTensorType>(input.getType());
        auto elementType = type.getElementType();

        auto outType = RankedTensorType::get(newShape, elementType);
        std::vector<int64_t> shVec(newShape.begin(), newShape.end());
        Value shValue = createConst(shVec, rewriter, input.getLoc());
        auto reshapeOp =
            rewriter.create<tensor::ReshapeOp>(input.getLoc(), outType, input, shValue);

        return reshapeOp.getResult();
    };

    auto addBcastOp = [](linalg::LinalgOp srcOp, Value &input, llvm::ArrayRef<int64_t> bcastShape,
                         SmallVector<int64_t> &dims, PatternRewriter &rewriter) {
        auto type = dyn_cast<RankedTensorType>(input.getType());
        auto elementType = type.getElementType();

        auto outType = RankedTensorType::get(bcastShape, elementType);
        auto bcastOp = rewriter.create<linalg::BroadcastOp>(
            srcOp.getLoc(), input, createInitTensor(srcOp, rewriter, outType), dims
        );
        auto gOp = linalg::generalizeNamedOp(rewriter, bcastOp);
        if (failed(gOp)) {
            return bcastOp.getResults()[0];
        }

        return gOp->getResults()[0];
    };

    //////////////End of Support Functions//////////////

    ScaleInfo scaleInfo;
    Value srcInput = input;
    while (foldBackwardRescale(input, scaleInfo)) {
    }

    // Collect the rescale GenericOps between srcOp and the current input
    SmallVector<linalg::GenericOp> rescaleOps;
    while (srcInput != input) {
        auto rescaleOp = srcInput.getDefiningOp<linalg::GenericOp>();
        rescaleOps.push_back(rescaleOp);
        srcInput = rescaleOp.getInputs()[0];
    }

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputShape = inputType.getShape();
    bool needReshape = !inputNewShape.empty() && !llvm::equal(inputNewShape, inputShape);
    if (needReshape) {
        input = addReshapeOp(input, inputNewShape, rewriter);
    }

    // Add a broadcast iff there are explicit bcast dims
    if (!dims.empty()) {
        input = addBcastOp(srcOp, input, outputShape, dims, rewriter);
    }

    // Propagate the reshape+broadcast through the rescale GenericOps
    while (!rescaleOps.empty()) {
        linalg::GenericOp rescaleOp = rescaleOps.back();
        rescaleOps.pop_back();

        auto output = rescaleOp.getOutputs()[0];
        auto outputType = dyn_cast<RankedTensorType>(output.getType());
        auto elementType = outputType.getElementType();

        auto emptyOp =
            rewriter.create<tensor::EmptyOp>(rescaleOp.getLoc(), outputShape, elementType);

        linalg::GenericOp newOp = rewriter.create<linalg::GenericOp>(
            rescaleOp.getLoc(), TypeRange{emptyOp.getResult().getType()}, ValueRange{input},
            ValueRange{emptyOp.getResult()}, newIndexingMaps,
            SmallVector<utils::IteratorType>{outputShape.size(), utils::IteratorType::parallel}
        );

        // Copy the region
        IRMapping mapping;
        rescaleOp->getRegion(0).cloneInto(&newOp.getRegion(), newOp.getRegion().begin(), mapping);

        input = newOp->getResult(0);
    }
}

LogicalResult collapseShapeWithDim(Value &input, int dim, PatternRewriter &rewriter) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();

    llvm::SmallVector<llvm::SmallVector<int64_t, 2>> newShape;

    bool collapse = false;
    for (int i = 0; i < shape.size(); i++) {
        // push everything if already collapsed
        if (collapse) {
            newShape.push_back({i});
            continue;
        }

        if (i != dim) {
            if (i + 1 != dim && i - 1 != dim) {
                newShape.push_back({i});
            }

            continue;
        }

        if (i - 1 >= 0) {
            newShape.push_back({i - 1, i});
        }
        else if (i + 1 < shape.size()) {
            newShape.push_back({i, i + 1});
        }
        else {
            newShape.push_back({i});
        }
        collapse = true;
    }

    auto squeezeDim = [](ArrayRef<int64_t> shape, int dim) {
        SmallVector<int64_t> newShape(shape.begin(), shape.end());

        int64_t rank = shape.size();
        if (rank == 0) {
            return newShape;
        }

        if (dim < 0 || dim >= rank) {
            return newShape;
        }

        if (shape[dim] != 1) {
            return newShape;
        }

        newShape.clear();
        for (int i = 0; i < rank; i++) {
            if (i != dim) {
                newShape.push_back(shape[i]);
            }
        }
        return newShape;
    };

    SmallVector<int64_t> squeezedShape;
    squeezedShape = squeezeDim(shape, dim);

    if (squeezedShape.size() == shape.size()) {
        // no change
        return failure();
    }

    auto outType = RankedTensorType::get(squeezedShape, elementType);
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        input.getLoc(), outType, input, ArrayRef<ReassociationIndices>{newShape}
    );

    input = collapseOp.getResult();

    return success();
}

// this helper func used to change input value, if return failure means input no change
// this func collapse tensor which input/output shape size is the same but input need broadcast
// 1x21x1 -> 1x21x2100, linalg.broadcast dosen't support input/output rank the same but broadcast
// on one dim, we collapse 1x21x1 to 1x21 to make sure linalg.broadcast input rank+broadcast rank ==
// output rank
LogicalResult collapseValue(
    Value &input, SmallVector<int64_t> &dims, int outputShapeSize, PatternRewriter &rewriter
) {
    auto type = dyn_cast<RankedTensorType>(input.getType());
    auto shape = type.getShape();

    if (dims.size() > 0 && shape.size() == outputShapeSize) {

        auto d = dims[0];

        // for now we don't support broadcast dims > 1
        // 1x21x1x1 -> 1x21x68x3 broadcast on dims[2, 3]
        if (dims.size() > 1) {
            return failure();
        }

        // 1x21x2 -> 1x21x68 is not broadcast case on the dim 2
        if (shape[d] != 1) {
            return failure();
        }

        if (failed(collapseShapeWithDim(input, d, rewriter))) {
            return failure();
        }
    }

    return success();
}

LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, SmallVectorImpl<Value> &inputs, PatternRewriter &rewriter) {
    /////////Support Functions///////////

    // This inspects affine indexing maps to find which output loop dims
    // are missing in each input (→ broadcast candidates).
    auto getBroadcastDimsFromMap = [](AffineMap inputMap,
                                      AffineMap outputMap) -> SmallVector<int64_t> {
        llvm::SmallDenseSet<unsigned> usedDims;
        for (auto expr : inputMap.getResults()) {
            if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr)) {
                usedDims.insert(dimExpr.getPosition());
            }
        }

        SmallVector<int64_t> broadcastDims;
        for (unsigned i = 0; i < outputMap.getNumDims(); i++) {
            if (!usedDims.contains(i)) {
                broadcastDims.push_back(i);
            }
        }
        return broadcastDims;
    };

    auto calcBcastShapeAndDims = [](ArrayRef<int64_t> outputShape, SmallVector<int64_t> &shape,
                                    SmallVector<int64_t> &dims) {
        // Reshape the input to match with linalg.broadcast requirements
        // 128x1 to be broadcasted to 1x128x32 need to be reshaped to 1x128
        SmallVector<int64_t> newShape;
        SmallVector<int64_t> newDims;
        for (int i = 0; i < outputShape.size(); ++i) {
            if (outputShape[i] == 1) {
                newShape.push_back(outputShape[i]);
                continue;
            }
            if (!llvm::is_contained(dims, i)) {
                newShape.push_back(outputShape[i]);
            }
            else {
                newDims.push_back(i);
            }
        }
        dims = newDims;
        shape = newShape;
    };

    //////////////End of Support Functions//////////////

    auto outputType = dyn_cast<RankedTensorType>(srcOp->getResult(0).getType());
    if (!outputType) {
        return failure();
    }
    auto outputShape = outputType.getShape();

    // Fetch maps for inputs and the output
    SmallVector<AffineMap> indexingMaps = srcOp.getIndexingMapsArray();
    AffineMap outputMap = indexingMaps.back();

    SmallVector<SmallVector<int64_t>> reshapeShapes(inputs.size());
    SmallVector<SmallVector<int64_t>> broadcastDims(inputs.size());
    bool needsTransform = false;
    SmallVector<bool> needsTransformPerInput(inputs.size(), false);

    for (auto it : llvm::enumerate(inputs)) {
        auto type = dyn_cast<RankedTensorType>(it.value().getType());
        if (!type) {
            return failure();
        }
        auto shape = type.getShape();
        if (shape.size() > outputShape.size()) {
            assert(false && "Input shape size is larger than output shape size");
        }

        broadcastDims[it.index()] = getBroadcastDimsFromMap(indexingMaps[it.index()], outputMap);
        if (broadcastDims[it.index()].empty()) {
            continue;
        }

        calcBcastShapeAndDims(outputShape, reshapeShapes[it.index()], broadcastDims[it.index()]);

        LLVM_DEBUG({
            llvm::dbgs() << "broadcastInputs: input " << it.index() << " reshapeShapes=[";
            for (size_t i = 0; i < reshapeShapes[it.index()].size(); ++i) {
                if (i) {
                    llvm::dbgs() << ",";
                }
                llvm::dbgs() << reshapeShapes[it.index()][i];
            }
            llvm::dbgs() << "] broadcastDims=[";
            for (size_t i = 0; i < broadcastDims[it.index()].size(); ++i) {
                if (i) {
                    llvm::dbgs() << ",";
                }
                llvm::dbgs() << broadcastDims[it.index()][i];
            }
            llvm::dbgs() << "]\n";
        });

        size_t expectedRank = outputShape.size() - broadcastDims[it.index()].size();
        if (reshapeShapes[it.index()].size() != expectedRank) {
            llvm::errs() << "broadcastInputs: input " << it.index() << " reshape rank "
                         << reshapeShapes[it.index()].size() << " doesn't match expected "
                         << expectedRank
                         << " for broadcast dims = " << broadcastDims[it.index()].size() << "\n";
            return failure();
        }
        bool needsReshape =
            !reshapeShapes[it.index()].empty() && !llvm::equal(reshapeShapes[it.index()], shape);
        if (!broadcastDims[it.index()].empty() || needsReshape) {
            needsTransform = true;
            needsTransformPerInput[it.index()] = true;
        }
    }

    if (!needsTransform) {
        return success();
    }

    SmallVector<AffineMap> newIndexingMaps = {outputMap, outputMap};
    for (auto it : llvm::enumerate(inputs)) {
        if (!needsTransformPerInput[it.index()]) {
            continue;
        }
        broadcastInput(
            srcOp, inputs[it.index()], broadcastDims[it.index()], reshapeShapes[it.index()],
            outputShape, newIndexingMaps, rewriter
        );
    }

    return success();
}

LogicalResult promoteScalar(linalg::LinalgOp srcOp, Value &input, PatternRewriter &rewriter) {

    auto addReshape1D = [&](Value &input) {
        auto type = dyn_cast<RankedTensorType>(input.getType());
        auto elementType = type.getElementType();
        SmallVector<int64_t> newShape = {1};
        auto outType = RankedTensorType::get(newShape, elementType);
        std::vector<int64_t> shVec(newShape.begin(), newShape.end());
        Value shValue = createConst(shVec, rewriter, input.getLoc());
        auto reshapeOp =
            rewriter.create<tensor::ReshapeOp>(input.getLoc(), outType, input, shValue);
        return reshapeOp.getResult();
    };

    auto rank = dyn_cast<RankedTensorType>(input.getType()).getRank();
    assert(rank >= 0 && "Tensor rank must be >= 0");
    if (rank > 0) {
        return failure();
    }
    input = addReshape1D(input);
    return success();
}

static bool isElementwiseNIn1Out(linalg::GenericOp srcOp) {
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
        if (srcOp.getMatchingIndexingMap(inputOperand) != outMap) {
            return false;
        }
    }
    return true;
}

LogicalResult
PromoteScalarsTo1D::matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const {

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
        return rewriter.notifyMatchFailure(srcOp, "result is scalar, no need to promote scalars\n");
    }

    if (srcOp.getNumReductionLoops() != 0) {
        return rewriter.notifyMatchFailure(
            srcOp, "only handle elementwise generics without reductions\n"
        );
    }

    SmallVector<AffineMap> indexingMaps = srcOp.getIndexingMapsArray();
    SmallVector<Value> newOperands(srcOp->getOperands().begin(), srcOp->getOperands().end());
    SmallVector<unsigned> scalarInputIdxs;
    for (auto it : llvm::enumerate(srcOp.getDpsInputs())) {
        auto rtt = dyn_cast<RankedTensorType>(it.value().getType());
        if (!rtt)
            return rewriter.notifyMatchFailure(srcOp, "non-ranked input\n");
        if (rtt.getRank() == 0)
            scalarInputIdxs.push_back(it.index());
    }
    if (scalarInputIdxs.empty()) {
        return rewriter.notifyMatchFailure(srcOp, "no rank-0 input to promote\n");
    }

    for (unsigned dpsIdx : scalarInputIdxs) {
        Value input = srcOp.getDpsInputs()[dpsIdx];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType) {
            return rewriter.notifyMatchFailure(
                srcOp, "expected ranked tensor type for input " + Twine(dpsIdx) + "\n"
            );
        }
        if (failed(promoteScalar(srcOp, input, rewriter))) {
            return rewriter.notifyMatchFailure(
                srcOp, "failed to promote input " + Twine(dpsIdx) + "\n"
            );
        }
        const int64_t numLoops = srcOp.getNumLoops();
        auto c0 = rewriter.getAffineConstantExpr(0);
        unsigned operandNumber = srcOp.getDpsInputOperand(dpsIdx)->getOperandNumber();
        newOperands[operandNumber] = input;
        indexingMaps[operandNumber] = AffineMap::get(
            /*dimCount=*/numLoops,
            /*symbolCount=*/0,
            /*results=*/ArrayRef<AffineExpr>{c0}, rewriter.getContext()
        );
    }

    rewriter.modifyOpInPlace(srcOp, [&]() {
        for (unsigned dpsIdx : scalarInputIdxs) {
            unsigned operandNumber = srcOp.getDpsInputOperand(dpsIdx)->getOperandNumber();
            srcOp->setOperand(operandNumber, newOperands[operandNumber]);
        }
        srcOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(indexingMaps));
    });

    return success();
}

LogicalResult
ReshapeToCollapseExpand::matchAndRewrite(tensor::ReshapeOp op, PatternRewriter &rewriter) const {
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
            out = rewriter.create<tensor::CollapseShapeOp>(
                op.getLoc(), dstType, op.getSource(), *direct
            );
        }
        else if (dstRank > srcRank) {
            out = rewriter.create<tensor::ExpandShapeOp>(
                op.getLoc(), dstType, op.getSource(), *direct
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
    Value collapsed = rewriter.create<tensor::CollapseShapeOp>(
        op.getLoc(), midType, op.getSource(), *collapseSrc
    );
    Value expanded =
        rewriter.create<tensor::ExpandShapeOp>(op.getLoc(), dstType, collapsed, *collapseDst);

    rewriter.replaceOp(op, expanded);
    return success();
}

} // namespace mlir::syna::torq
