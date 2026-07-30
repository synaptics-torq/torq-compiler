// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/QuantPatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "linalg-torq-q-reduce-pattern"

namespace mlir::syna::torq {

// Quantized reduction op patterns.
//
// QReduceMeanConvert matches a DQ -> sum-reduction -> div-by-count -> Q chain
// that represents ONNX ReduceMean (global average over spatial dims) after
// torch-mlir lowering, and fuses it into a single torq_hl.avg_pool2d op.
//
// Example matched IR:
//   %dq = linalg.generic {parallel x 4}    i8 -> f32  (dequant)
//   %sum = linalg.generic {parallel, parallel, reduction, reduction}
//            f32 -> f32  (arith.addf reduction)
//   %mean = linalg.generic {parallel x 2}  f32 -> f32  (arith.divf %in, %count)
//   %q = linalg.generic {parallel x 2}     f32 -> i8   (quant)
//   %out = tensor.expand_shape %q ...      [1,4] -> [1,4,1,1]

namespace {

struct QReduceMeanConvert : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

    // Match the mean (div-by-constant) generic.
    // Returns the divisor value (the number of reduced elements) on success.
    LogicalResult matchMeanGeneric(
        linalg::GenericOp meanOp, linalg::GenericOp &sumOp, double &reducedElements,
        PatternRewriter &rewriter
    ) const {
        if (meanOp.getNumDpsInputs() != 1 || meanOp.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(meanOp, "mean op must have 1 input/init");

        if (meanOp.getNumReductionLoops() != 0 || meanOp.getNumParallelLoops() == 0)
            return rewriter.notifyMatchFailure(meanOp, "mean op must be elementwise parallel");

        auto yieldOp = dyn_cast<linalg::YieldOp>(meanOp.getBody()->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1)
            return rewriter.notifyMatchFailure(meanOp, "mean op must have single yield");

        Operation *yieldDefOp = yieldOp.getOperand(0).getDefiningOp();
        auto divOp = dyn_cast_or_null<arith::DivFOp>(yieldDefOp);
        if (!divOp)
            return rewriter.notifyMatchFailure(meanOp, "mean op body must be divf");

        if (!isa<BlockArgument>(divOp.getLhs()) ||
            cast<BlockArgument>(divOp.getLhs()).getOwner() != meanOp.getBody())
            return rewriter.notifyMatchFailure(meanOp, "div lhs must be block argument");

        auto constOp = divOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!constOp)
            return rewriter.notifyMatchFailure(meanOp, "div rhs must be constant");

        auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
        if (!floatAttr)
            return rewriter.notifyMatchFailure(meanOp, "div constant must be float");

        double count = floatAttr.getValueAsDouble();
        if (count <= 0.0)
            return rewriter.notifyMatchFailure(meanOp, "divisor must be positive");
        reducedElements = count;

        // The div input must come from a sum reduction generic.
        sumOp = meanOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!sumOp)
            return rewriter.notifyMatchFailure(meanOp, "mean input must come from generic");

        if (sumOp.getNumReductionLoops() == 0)
            return rewriter.notifyMatchFailure(sumOp, "sum op must have reduction loops");

        auto sumYield = dyn_cast<linalg::YieldOp>(sumOp.getBody()->getTerminator());
        if (!sumYield || sumYield.getNumOperands() != 1)
            return rewriter.notifyMatchFailure(sumOp, "sum op must have single yield");

        if (!dyn_cast_or_null<arith::AddFOp>(sumYield.getOperand(0).getDefiningOp()))
            return rewriter.notifyMatchFailure(sumOp, "sum op must reduce via addf");

        return success();
    }

    // Match the full DQ -> sum -> div -> Q chain rooted at the mean generic.
    // The caller handles the sum/mean generics in the middle; the shared helpers
    // match the DQ and Q.
    LogicalResult matchQReduceMeanChain(
        linalg::GenericOp meanOp, PatternRewriter &rewriter, linalg::GenericOp &sumOp,
        QuantizedOpChain &chain, Value &dequantInput, int64_t &numChannels, double &reducedElements
    ) const {
        if (failed(matchMeanGeneric(meanOp, sumOp, reducedElements, rewriter)))
            return failure();

        // Match dequant feeding the sum reduction.
        Value dequantValue = sumOp.getInputs()[0];
        if (failed(matchInputQuantization(dequantValue, rewriter, chain)))
            return rewriter.notifyMatchFailure(meanOp, "failed to match Q dequant");

        auto dequantResultType = dyn_cast<ShapedType>(chain.inputDequantOp.getResult(0).getType());
        if (!dequantResultType || !dequantResultType.getElementType().isF32())
            return rewriter.notifyMatchFailure(meanOp, "dequant output must be f32");

        dequantInput = chain.inputDequantOp.getInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(dequantInput.getType());
        if (!inputType || !inputType.getElementType().isInteger(8) || inputType.getRank() != 4)
            return rewriter.notifyMatchFailure(meanOp, "dequant input must be 4-D int8");

        numChannels = inputType.getShape()[1];
        if (numChannels <= 0)
            return rewriter.notifyMatchFailure(meanOp, "non-positive channel count");

        int64_t expectedReducedElements = inputType.getShape()[2] * inputType.getShape()[3];
        if (expectedReducedElements <= 0 ||
            std::llround(reducedElements) != expectedReducedElements)
            return rewriter.notifyMatchFailure(meanOp, "divisor does not match H*W");

        // Match quant consuming the mean output.
        if (failed(matchOutputQuantization(meanOp.getResult(0), rewriter, chain)))
            return rewriter.notifyMatchFailure(meanOp, "failed to match Q quant");

        return success();
    }

    // Build the per-channel scale_bias tensor: [bias_0, mult, bias_1, mult, ...].
    Value buildScaleBias(
        linalg::GenericOp meanOp, int64_t numChannels, int32_t multiplier, int32_t bias32,
        PatternRewriter &rewriter
    ) const {
        std::vector<int32_t> scalePerChannel(numChannels, multiplier);
        std::vector<int32_t> biasPerChannel(numChannels, bias32);
        return createI32Const(rewriter, meanOp, interleave(biasPerChannel, scalePerChannel));
    }

  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;
    QReduceMeanConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/3),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp meanOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(meanOp))
            return rewriter.notifyMatchFailure(meanOp, "already marked");

        linalg::GenericOp sumOp;
        QuantizedOpChain chain;
        Value dequantInput;
        int64_t numChannels;
        double reducedElements;
        if (failed(matchQReduceMeanChain(
                meanOp, rewriter, sumOp, chain, dequantInput, numChannels, reducedElements
            )))
            return failure();

        if (_markFuseGroups) {
            auto fuseGroupAttr = meanOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(meanOp, "missing fuse group id");
            markFuseGroupBackward(
                chain.quantOp.getResult(0), {dequantInput}, rewriter, fuseGroupAttr
            );
            return success();
        }

        if (chain.outputInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(meanOp, "quant scale is zero");

        // Combined rescale: input_scale / (reduced_elements * output_scale).
        double avgScale = 1.0 / reducedElements;
        double rescaleRatio = chain.inputInfo.scale * avgScale / chain.outputInfo.scale;

        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(rescaleRatio, multiplier, shift))
            return rewriter.notifyMatchFailure(meanOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QReduceMeanConvert] multiplier=" << multiplier
                                << " shift=" << shift << " reducedElements=" << reducedElements
                                << " avgScale=" << avgScale << "\n";);

        int32_t bias32 =
            -static_cast<int32_t>(reducedElements * static_cast<int64_t>(chain.inputInfo.zp));
        Value scaleBias = buildScaleBias(meanOp, numChannels, multiplier, bias32, rewriter);

        Location loc = meanOp.getLoc();
        auto nchwToNhwc = Permutation::nchw2nhwc();
        Value nhwcInput = transposeValue(dequantInput, nchwToNhwc, loc, rewriter);

        // ReduceMean over HxW produces [N, C, 1, 1] in NCHW, i.e. [N, 1, 1, C] in NHWC.
        auto dequantInputType = cast<RankedTensorType>(dequantInput.getType());
        auto inputShape = dequantInputType.getShape();
        auto quantResultType = cast<RankedTensorType>(chain.quantOp.getResult(0).getType());
        auto nhwcOutputType = RankedTensorType::get(
            {inputShape[0], 1, 1, inputShape[1]}, quantResultType.getElementType()
        );

        Value weights = createI8Const(
            rewriter, meanOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        int32_t outputZp = static_cast<int32_t>(std::llround(chain.outputInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(chain.outputInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(chain.outputInfo.max));

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(chain.quantOp);

        auto avgPoolOp = torq_hl::AvgPool2DOp::create(
            rewriter, loc, nhwcOutputType, createInitTensor(meanOp, rewriter, nhwcOutputType),
            /*input_zp=*/chain.inputInfo.zp, outputZp, outputMin, outputMax, shift, weights,
            scaleBias, nhwcInput
        );

        Value torqOut = transposeValue(avgPoolOp.getOutput(), nchwToNhwc.reverse(), loc, rewriter);

        // If the original quant output was collapsed (e.g. [N, C]), collapse back.
        if (quantResultType.getRank() == 2) {
            torqOut = tensor::CollapseShapeOp::create(
                rewriter, loc, quantResultType, torqOut,
                SmallVector<ReassociationIndices>{{0}, {1, 2, 3}}
            );
        }

        rewriter.replaceOp(chain.quantOp, torqOut);
        rewriter.eraseOp(meanOp);
        rewriter.eraseOp(sumOp);
        rewriter.eraseOp(chain.inputDequantOp);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQReducePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QReduceMeanConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
