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

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

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
    // e.g. 21 -> 1x1x21
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

    // if input shape is all 1, we need to collapse it to be 1
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

    // align input shape to output shape for broadcasting
    auto dimvec = [](ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape) {
        SmallVector<int64_t> dims;

        if (inputShape.size() <= outputShape.size()) {
            for (int i = 0; i < outputShape.size(); i++) {
                if (i < inputShape.size()) {
                    if (inputShape[i] != outputShape[i]) {
                        dims.push_back(i);
                    }
                }
                else {
                    dims.push_back(i);
                }
            }
        }
        return dims;
    };

    input1Shape = dyn_cast<RankedTensorType>(input1.getType()).getShape();
    input2Shape = dyn_cast<RankedTensorType>(input2.getType()).getShape();
    SmallVector<int64_t> dims1, dims2;
    dims1 = dimvec(input1Shape, outputShape);
    dims2 = dimvec(input2Shape, outputShape);

    (void)collapseValue(input1, dims1, outputShape.size(), rewriter);
    (void)collapseValue(input2, dims2, outputShape.size(), rewriter);

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

        // if op is tagged with torq.fuse.group, unnecessary to do broadcast
        // as it will be fused later probably
        if (auto groupAttr = srcOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
            bool needBroadcast = false;

            // FIXME: for now we only check addOp and will remove all these group id check soon
            if (isa<arith::AddIOp>(eleOp)) {

                Value operand1 = srcOp.getInputs()[0];
                Value operand2 = srcOp.getInputs()[1];

                ScaleInfo scaleInput1, scaleInput2;
                if (foldBackwardRescale(operand1, scaleInput1) ||
                    foldBackwardRescale(operand2, scaleInput2)) {
                    needBroadcast = true;
                }
            }

            if (!needBroadcast) {
                return failure();
            }
        }

        auto rank1 = input1Type.getRank();
        auto rank2 = input2Type.getRank();

        if (rank1 == 0 || rank2 == 0) {
            return failure();
        }

        if (failed(broadcastInputs(srcOp, input1, input2, rewriter))) {
            return rewriter.notifyMatchFailure(
                srcOp, "failed to broadcast inputs for elementwise binary op\n"
            );
        }

        // make sure all inputs/outputs have the same affine maps as elementwise binary op requires
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
