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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
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

static LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, Value &input1, Value &input2, PatternRewriter &rewriter) {
    auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
    auto input1Shape = input1Type.getShape();
    auto input1ElementType = input1Type.getElementType();
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

    // if input rank is 1, expand input shape to output rank
    // find the dim postion in output shape equal to input shape
    // e.g.
    auto expandTensor = [&](Value input, ArrayRef<int64_t> outputShape) -> Value {
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        auto inputElementType = inputType.getElementType();

        if (inputShape.size() == 1 && outputShape.size() > 1) {
            int dim = -1;
            for (int i = 0; i < outputShape.size(); i++) {
                if (outputShape[i] == inputShape[0]) {
                    dim = i;
                    break;
                }
            }

            if (dim == -1) {
                return input;
            }

            SmallVector<int64_t> newShape(outputShape.size(), 1);
            newShape[dim] = inputShape[0];

            SmallVector<int64_t, 2> reassociationIndices;
            for (int i = 0; i < newShape.size(); i++) {
                reassociationIndices.push_back(i);
            }

            auto newType = RankedTensorType::get(newShape, inputElementType);
            auto expandOp = rewriter.create<tensor::ExpandShapeOp>(
                input.getLoc(), newType, input, ArrayRef<ReassociationIndices>{reassociationIndices}
            );

            return expandOp.getResult();
        }

        return input;
    };

    auto input1Rank = dyn_cast<RankedTensorType>(input1.getType()).getRank();
    auto input2Rank = dyn_cast<RankedTensorType>(input2.getType()).getRank();
    if (input1Rank == 1 && outputShape.size() > 1) {
        input1 = expandTensor(input1, outputShape);
    }
    if (input2Rank == 1 && outputShape.size() > 1) {
        input2 = expandTensor(input2, outputShape);
    }

    // If a tensor is full of size-1 dims, collapse it to [1].
    auto collapseTensor = [&](Value input, PatternRewriter &rewriter) -> Value {
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        auto inputElementType = inputType.getElementType();

        SmallVector<int64_t, 2> collapseShape;
        for (int i = 0; i < inputShape.size(); i++) {
            collapseShape.push_back(i);
        }

        bool allOne = true;
        for (int i = 0; i < inputShape.size(); i++) {
            if (inputShape[i] != 1) {
                allOne = false;
                break;
            }
        }
        if (allOne) {
            auto outType = RankedTensorType::get({1}, inputElementType);
            auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
                input.getLoc(), outType, input, ArrayRef<ReassociationIndices>{collapseShape}
            );

            return collapseOp.getResult();
        }

        return input;
    };

    input1 = collapseTensor(input1, rewriter);
    input2 = collapseTensor(input2, rewriter);

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

    // Fetch maps for both inputs and the output
    SmallVector<AffineMap> indexingMaps = srcOp.getIndexingMapsArray();
    AffineMap input1Map = indexingMaps[0];
    AffineMap input2Map = indexingMaps[1];
    AffineMap outputMap = indexingMaps.back();

    // Recompute shapes (after potential collapse)
    input1Shape = dyn_cast<RankedTensorType>(input1.getType()).getShape();
    input2Shape = dyn_cast<RankedTensorType>(input2.getType()).getShape();

    // Compute initial broadcast dims
    SmallVector<int64_t> dims1 = getBroadcastDimsFromMap(input1Map, outputMap);
    SmallVector<int64_t> dims2 = getBroadcastDimsFromMap(input2Map, outputMap);

    // When ranks match, drop dims where input and output already match (no broadcast).
    auto filterBySize = [](SmallVector<int64_t> &dims, ArrayRef<int64_t> inputShape,
                           ArrayRef<int64_t> outputShape) {
        SmallVector<int64_t> filtered;
        for (auto d : dims) {
            if (d < inputShape.size() && inputShape[d] == outputShape[d]) {
                continue;
            }
            filtered.push_back(d);
        }
        return filtered;
    };

    if (input1Shape.size() == outputShape.size()) {
        dims1 = filterBySize(dims1, input1Shape, outputShape);
    }
    if (input2Shape.size() == outputShape.size()) {
        dims2 = filterBySize(dims2, input2Shape, outputShape);
    }

    if (dims1.empty() && dims2.empty()) {
        return success();
    }

    auto recalculateBroadcastDims = [](Value input, ArrayRef<int64_t> outputShape) {
        auto inputShape = dyn_cast<RankedTensorType>(input.getType()).getShape();
        SmallVector<int64_t> broadcastDims;

        for (unsigned i = inputShape.size(); i < outputShape.size(); i++) {
            broadcastDims.push_back(i);
        }
        return broadcastDims;
    };

    // Detect pre-broadcasted maps that contain constants (like affine_map<(d0,d1,d2)->(0,d1,0)>).
    auto hasConstants = [](AffineMap map) {
        for (auto expr : map.getResults()) {
            if (llvm::isa<AffineConstantExpr>(expr)) {
                return true;
            }
        }
        return false;
    };

    // Remember ranks before collapse
    unsigned originalInput1Rank = input1Shape.size();
    unsigned originalInput2Rank = input2Shape.size();

    (void)collapseValue(input1, dims1, outputShape.size(), rewriter);
    (void)collapseValue(input2, dims2, outputShape.size(), rewriter);

    // Refresh shapes and detect whether we collapsed in this pass
    input1Shape = dyn_cast<RankedTensorType>(input1.getType()).getShape();
    input2Shape = dyn_cast<RankedTensorType>(input2.getType()).getShape();

    bool input1WasCollapsed = (input1Shape.size() != originalInput1Rank);
    bool input2WasCollapsed = (input2Shape.size() != originalInput2Rank);

    if (!dims1.empty() && input1Shape.size() < outputShape.size() && hasConstants(input1Map) &&
        !input1WasCollapsed) {
        dims1 = recalculateBroadcastDims(input1, outputShape);
        SmallVector<int64_t> filtered1;
        for (auto d : dims1) {
            if (d >= input1Shape.size()) {
                filtered1.push_back(d);
            }
        }
        dims1 = filtered1;
    }
    if (!dims2.empty() && input2Shape.size() < outputShape.size() && hasConstants(input2Map) &&
        !input2WasCollapsed) {
        dims2 = recalculateBroadcastDims(input2, outputShape);
        SmallVector<int64_t> filtered2;
        for (auto d : dims2) {
            if (d >= input2Shape.size()) {
                filtered2.push_back(d);
            }
        }
        dims2 = filtered2;
    }

    // Don't create linalg.broadcast for same-rank inputs.
    if (input1Shape.size() == outputShape.size()) {
        if (!dims1.empty() && !hasConstants(input1Map)) {
            // Same-rank + dims present but no constants in map is inconsistent.
            // Either an upstream issue or we skipped a needed collapse step.
            LLVM_DEBUG(
                llvm::dbgs()
                << "[broadcastInputs] Warning: same-rank dims for input1 without constants in map\n"
            );
            // Optionally: return failure() to surface the inconsistency early.
        }
        dims1.clear();
    }
    if (input2Shape.size() == outputShape.size()) {
        if (!dims2.empty() && !hasConstants(input2Map)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[broadcastInputs] Warning: same-rank dims for input2 without constants in map\n"
            );
        }
        dims2.clear();
    }

    if (!dims1.empty()) {

        auto broadcastOutputType = RankedTensorType::get(outputShape, input1ElementType);

        input1 = rewriter
                     .create<linalg::BroadcastOp>(
                         srcOp.getLoc(), input1,
                         createInitTensor(srcOp, rewriter, broadcastOutputType), dims1
                     )
                     .getResults()[0];
    }

    if (!dims2.empty()) {

        auto broadcastOutputType = RankedTensorType::get(outputShape, input1ElementType);
        input2 = rewriter
                     .create<linalg::BroadcastOp>(
                         srcOp.getLoc(), input2,
                         createInitTensor(srcOp, rewriter, broadcastOutputType), dims2
                     )
                     .getResults()[0];
    }

    return success();
}

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
            !isa<arith::MulIOp>(eleOp) && !isa<arith::MulFOp>(eleOp)) {
            return rewriter.notifyMatchFailure(
                srcOp, "elementwise binary op pattern only supports add/sub/mul ..\n"
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

void populateOptimizeElementwiseBinaryOpPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.add<BroadcastElementwiseBinaryOpPattern>(context);
}

} // namespace mlir::syna::torq
