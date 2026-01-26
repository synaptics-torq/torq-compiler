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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-elementwise-binary-op-pattern"

namespace mlir::syna::torq {

static LogicalResult collapseShapeWithDim(Value &input, int dim, PatternRewriter &rewriter) {
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
static LogicalResult collapseValue(
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

    // Add a reshape and broadcast.
    // TODO: can dims be empty?
    if (inputNewShape.size() != outputShape.size() && !dims.empty()) {
        input = addReshapeOp(input, inputNewShape, rewriter);
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

static LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, Value &input1, Value &input2, PatternRewriter &rewriter) {
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

    auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
    auto input1Shape = input1Type.getShape();
    auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
    auto input2Shape = input2Type.getShape();

    auto outputType = dyn_cast<RankedTensorType>(srcOp->getResult(0).getType());
    auto outputShape = outputType.getShape();

    // this case should never happen
    if (input1Shape.size() > outputShape.size() || input2Shape.size() > outputShape.size()) {
        llvm::errs() << "input1 shape size: " << input1Shape.size()
                     << ", input2 shape size: " << input2Shape.size()
                     << ", output shape size: " << outputShape.size() << "\n";
        assert(false && "Input shape size is larger than output shape size");
    }

    if (input1Shape.size() == 0 && input2Shape.size() == 0) {
        assert(false && "Input shape size is 0");
    }

    // Fetch maps for both inputs and the output
    SmallVector<AffineMap> indexingMaps = srcOp.getIndexingMapsArray();
    AffineMap input1Map = indexingMaps[0];
    AffineMap input2Map = indexingMaps[1];
    AffineMap outputMap = indexingMaps.back();

    SmallVector<AffineMap> newIndexingMaps = {outputMap, outputMap};

    // Compute initial broadcast dims
    SmallVector<int64_t> dims1 = getBroadcastDimsFromMap(input1Map, outputMap);
    SmallVector<int64_t> dims2 = getBroadcastDimsFromMap(input2Map, outputMap);
    if (dims1.empty() && dims2.empty()) {
        return success();
    }

    SmallVector<int64_t> input1NewShape;
    calcBcastShapeAndDims(outputShape, input1NewShape, dims1);
    SmallVector<int64_t> input2NewShape;
    calcBcastShapeAndDims(outputShape, input2NewShape, dims2);

    if (dims1.size() > 1 && input1NewShape.size() != 1) {
        return failure();
    }
    if (dims2.size() > 1 && input2NewShape.size() != 1) {
        return failure();
    }

    broadcastInput(srcOp, input1, dims1, input1NewShape, outputShape, newIndexingMaps, rewriter);
    broadcastInput(srcOp, input2, dims2, input2NewShape, outputShape, newIndexingMaps, rewriter);

    return success();
}

static LogicalResult
promoteScalar(linalg::LinalgOp srcOp, Value &input, PatternRewriter &rewriter) {

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

/// @brief Promote scalar operands to 1D tensors via a symbolic reshape
/// @note Only matches elementwise binary ops for now
///
/// The purpose of this promotion is to ensure that ops with scalar operands
/// are picked up by `BroadcastElementwiseBinaryOpPattern`
///
/// Simlpe example: `tensor<i16>` -> `tensor<1xi16>`
class PromoteScalarsTo1D : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        // Avoid broadcasted inputs
        if (srcOp->getAttrOfType<BoolAttr>("broadcasted")) {
            return failure();
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

        if (srcOp.getNumDpsInputs() != 2 || srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "expected binary linalg.generic with a single output\n"
            );
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
};

// elementwise binary ops with 2 inputs
class BroadcastElementwiseBinaryOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    bool isScalarFromRecursiveRescale(const Value &v) const {
        Value input = v;
        ScaleInfo scaleInfo;

        // if input is from tensor.expandshape, assign input to expandop's input
        if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(input.getDefiningOp())) {
            input = expandOp.getSrc();
        }

        while (foldBackwardRescale(input, scaleInfo)) {
        }

        linalg::GenericOp rescaleOp = input.getDefiningOp<linalg::GenericOp>();

        if (!rescaleOp) {
            LLVM_DEBUG({ llvm::errs() << "Value input definingOp is not linalg.generic op\n"; });
            return false;
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(rescaleOp.getBody()->getTerminator());
        if (!yieldOp) {
            LLVM_DEBUG({ llvm::errs() << "There is no yield in linalg.generic body\n"; });
            return false;
        }

        auto yieldValues = yieldOp.getValues();
        if (yieldValues.size() != 1) {
            LLVM_DEBUG({ llvm::errs() << "Linalg.yield operand is not 1 \n"; });
            return false;
        }

        tosa::ApplyScaleOp applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();
        if (!applyScaleOp) {
            LLVM_DEBUG({ llvm::errs() << "apply scale op does not exist\n"; });
            return false;
        }

        Value value = applyScaleOp.getValue();
        if (!value) {
            LLVM_DEBUG({ llvm::errs() << "applyScaleOp cannot get value\n"; });
            return false;
        }

        auto constOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp());
        if (!constOp) {
            LLVM_DEBUG({ llvm::errs() << "defining op is not constant op\n"; });
            return false;
        }

        int32_t data = 0;
        if (!getIntegerConstantValue(constOp, &data)) {
            LLVM_DEBUG({ llvm::errs() << "cannot get integer constant value\n"; });
            return false;
        }

        return true;
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp->getAttrOfType<BoolAttr>("broadcasted")) {
            return failure();
        }

        if (srcOp.getInputs().empty() || srcOp.getInputs().size() > 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects generic op with 2 (or 1) inputs\n"
            );
        }

        if (srcOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects generic op with 1 output\n"
            );
        }

        auto resultType = dyn_cast<RankedTensorType>(srcOp.getResultTypes().front());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor result type\n"
            );
        }
        auto resultElemType = resultType.getElementType();
        if (resultElemType.isF64()) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern does not support f64\n"
            );
        }

        auto srcOpSize = srcOp.getNumDpsInputs();
        if (srcOpSize == 1) {
            return failure();
        }

        Value input1 = srcOp.getInputs()[0];
        Value input2 = srcOp.getInputs()[1];

        auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
        if (!input1Type) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor input1 type\n"
            );
        }
        auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
        if (!input2Type) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern expects ranked tensor input2 type\n"
            );
        }

        if (input1Type == resultType && input2Type == resultType) {
            return failure();
        }

        Operation *eleOp = getElementwiseBinaryOp(srcOp, true);

        arith::ShRSIOp shrsiOp1;
        bool isRRSOp = isRoundingRightShiftOp(srcOp, shrsiOp1);

        if (!eleOp && !isRRSOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "generic op is not expected elementwise binary op\n"
            );
        }

        // TODO: add more elementwise binary ops
        if (eleOp && !isa<arith::AddIOp>(eleOp) && !isa<arith::AddFOp>(eleOp) &&
            !isa<arith::SubIOp>(eleOp) && !isa<arith::SubFOp>(eleOp) &&
            !isa<arith::MulIOp>(eleOp) && !isa<arith::MulFOp>(eleOp) &&
            !isa<arith::DivFOp>(eleOp) && !isa<arith::CmpIOp>(eleOp)) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern only supports add/sub/mul/cmp ..\n"
            );
        }

        auto isConstantValue = [](Value val) {
            if (auto definingOp = val.getDefiningOp()) {
                return isa<arith::ConstantOp>(definingOp);
            }
            return false;
        };

        auto rank1 = input1Type.getRank();
        auto rank2 = input2Type.getRank();

        if (rank1 == 0 || rank2 == 0) {
            return rewriter.notifyMatchFailure(
                srcOp, "one of input or both input rank is 0, no need broadcast\n"
            );
        }

        // if one input is rank=1 constant, no need to broadcast
        if ((isConstantValue(input1) && rank1 < 2) || (isConstantValue(input2) && rank2 < 2)) {
            return rewriter.notifyMatchFailure(
                srcOp, "one of input or both input is rank=1 constant, no need broadcast\n"
            );
        }

        // TODO: add more recursive scalar input processing for elementwise binary ops
        // right now we only handle add/sub with recurive scalar input processing
        if (isa<arith::AddIOp>(eleOp) || isa<arith::SubIOp>(eleOp)) {
            if (isScalarFromRecursiveRescale(input1) || isScalarFromRecursiveRescale(input2)) {
                return rewriter.notifyMatchFailure(
                    srcOp, "one of input or both input is recurive scalar, no need broadcast\n"
                );
            }
        }

        if (failed(broadcastInputs(srcOp, input1, input2, rewriter))) {
            return rewriter.notifyMatchFailure(
                srcOp, "failed to broadcast inputs for elementwise binary op\n"
            );
        }

        AffineMap m = srcOp.getMatchingIndexingMap(srcOp.getDpsInitOperand(0));
        SmallVector<AffineMap> newIndexingMaps = {m, m, m};

        // update the inputs and related attr of srcOp
        rewriter.modifyOpInPlace(srcOp, [&]() {
            srcOp->setOperand(0, input1);
            srcOp->setOperand(1, input2);
            srcOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newIndexingMaps));
        });

        srcOp->setAttr("broadcasted", BoolAttr::get(srcOp->getContext(), true));

        return success();
    }
};

struct ReshapeToCollapseExpand : public OpRewritePattern<tensor::ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ReshapeOp op, PatternRewriter &rewriter) const override {
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
};

void populateOptimizeElementwiseBinaryOpPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.add<PromoteScalarsTo1D>(context);
    patterns.add<BroadcastElementwiseBinaryOpPattern>(context);
    patterns.add<ReshapeToCollapseExpand>(context);
    tensor::populateReassociativeReshapeFoldingPatterns(patterns);
    patterns.add<ComposeExpandOfCollapseOp<tensor::ExpandShapeOp, tensor::CollapseShapeOp>>(context
    );
}

} // namespace mlir::syna::torq
