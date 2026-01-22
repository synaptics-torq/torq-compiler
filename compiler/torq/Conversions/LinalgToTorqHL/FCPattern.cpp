// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-fc-pattern"

namespace mlir::syna::torq {

template <class OpTy> struct FCMatmulOpConversion : public OpRewritePattern<OpTy> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern<OpTy>::OpRewritePattern;

    FCMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<OpTy>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(OpTy srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        const auto loc = srcOp.getLoc();
        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 input and 1 output");
        }

        Value inputA = srcOp.getInputs()[0];
        Value inputB = srcOp.getInputs()[1]; // weights
        Value output = srcOp.getResultTensors()[0];

        auto inputAType = llvm::cast<RankedTensorType>(inputA.getType());
        auto inputBType = llvm::cast<RankedTensorType>(inputB.getType());
        auto outputType = llvm::cast<RankedTensorType>(output.getType());
        if (inputAType.getRank() != 2 || inputBType.getRank() != 2 || outputType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects 2D inputs and outputs"
            );
        }
        auto inputAShape = inputAType.getShape();
        if (inputAShape[0] != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects inputA shape[0] == 1"
            );
        }

        auto outputChannelCount = outputType.getShape()[1];
        bool isInt = outputType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp)) {
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected scale and clamp info for integer operations"
            );
        }

        // check if output is a tensor::CollapseShapeOp
        if (output.hasOneUse() && isa<tensor::CollapseShapeOp>(*output.getUsers().begin())) {
            output = output.getUsers().begin()->getResult(0);
        }

        // Prepare weights
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(inputB.getDefiningOp())) {
            inputB = transposeOp.getInput();
            // NOTE: inputB changed, re-get its type if need to process related
        }

        inputBType = llvm::cast<RankedTensorType>(inputB.getType());
        auto inputBShape = inputBType.getShape();
        if (inputBShape[0] != outputChannelCount) {
            // above logic already checked inputB(weight) rank=2
            inputB = transposeValue(inputB, {1, 0}, loc, rewriter);
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {inputA, inputB}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(inputB);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to create constant for transposed weights"
            );
        }
        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // get new output type as above various changes for output
        outputType = llvm::cast<RankedTensorType>(output.getType());
        inputAType = llvm::cast<RankedTensorType>(inputA.getType());

        assert(outputType.getRank() >= 2 && "TORQ FC op expects output tensor rank >= 2");
        if (outputType.getRank() > 2) {
            auto outputShape = outputType.getShape();
            for (int i = 0; i < outputShape.size() - 2; ++i) {
                assert(outputShape[i] == 1 && "TORQ FC op expects extra dimensions to be 1");
            }
        }
        assert(inputAType.getRank() == 2 && "TORQ FC op expects input tensor rank == 2");
        auto weightType = llvm::cast<RankedTensorType>(torqWeights.getType());
        assert(weightType.getRank() == 2 && "TORQ FC op expects weight rank == 2");

        auto fcOp = rewriter.create<torq_hl::FullyConnectedOp>(
            loc, outputType, createInitTensor(srcOp, rewriter, outputType), input_zp,
            0, // weight zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, inputA
        );
        rewriter.replaceOp(output.getDefiningOp(), fcOp.getOutput());

        return success();
    }
};

void populateLinalgToTorqHLFCPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<FCMatmulOpConversion<linalg::MatmulOp>>(context, markFuseGroups);
    patterns.insert<FCMatmulOpConversion<linalg::MatmulTransposeBOp>>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
