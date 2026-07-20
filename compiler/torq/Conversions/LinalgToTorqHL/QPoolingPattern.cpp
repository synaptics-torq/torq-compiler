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

#define DEBUG_TYPE "linalg-torq-q-pooling-pattern"

namespace mlir::syna::torq {

// Quantized pooling lowering patterns.

namespace {

// Try to match a downstream elementwise divf/mulf generic that scales the
// pooling result by a constant.  Returns the scale value; scaleOp is set to
// the matched generic (or null if no scale is present).
double tryExtractAvgScale(Value output, linalg::GenericOp &scaleOp) {
    scaleOp = getSingleUser<linalg::GenericOp>(output);
    if (!scaleOp || scaleOp.getNumDpsInputs() != 1 || scaleOp.getNumDpsInits() != 1 ||
        scaleOp.getNumReductionLoops() != 0)
        return 1.0;

    auto yieldOp = dyn_cast<linalg::YieldOp>(scaleOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return 1.0;

    auto mulOp = dyn_cast<arith::MulFOp>(yieldOp.getOperand(0).getDefiningOp());
    auto divOp = dyn_cast<arith::DivFOp>(yieldOp.getOperand(0).getDefiningOp());
    if (!mulOp && !divOp)
        return 1.0;

    Value lhs = mulOp ? mulOp.getLhs() : divOp.getLhs();
    Value rhs = mulOp ? mulOp.getRhs() : divOp.getRhs();

    bool lhsIsBlockArg =
        isa<BlockArgument>(lhs) && cast<BlockArgument>(lhs).getOwner() == scaleOp.getBody();
    bool rhsIsBlockArg =
        isa<BlockArgument>(rhs) && cast<BlockArgument>(rhs).getOwner() == scaleOp.getBody();

    arith::ConstantOp constOp = nullptr;
    if (lhsIsBlockArg && !rhsIsBlockArg)
        constOp = rhs.getDefiningOp<arith::ConstantOp>();
    else if (rhsIsBlockArg && !lhsIsBlockArg)
        constOp = lhs.getDefiningOp<arith::ConstantOp>();

    if (!constOp)
        return 1.0;

    auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!floatAttr)
        return 1.0;

    double value = floatAttr.getValueAsDouble();
    if (divOp) {
        if (value == 0.0)
            return 1.0;
        value = 1.0 / value;
    }
    return value;
}

struct QGlobalAveragePoolConvert : public OpRewritePattern<linalg::PoolingNchwSumOp> {
  private:
    const bool _markFuseGroups;

    // Match the dequant → pool → [avg scale] → quant chain.
    LogicalResult matchQPoolingChain(
        linalg::PoolingNchwSumOp poolOp, Value output, PatternRewriter &rewriter,
        linalg::GenericOp &dequantOp, DequantInfo &dInfo, double &avgScale,
        linalg::GenericOp &avgScaleOp, linalg::GenericOp &quantOp, QuantInfo &qInfo,
        int64_t &numChannels, int64_t &reducedElements
    ) const {
        Value input0 = poolOp.getInputs()[0];
        Value kernel = poolOp.getInputs()[1];
        auto inputType = dyn_cast<RankedTensorType>(input0.getType());
        auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
        if (!inputType || !kernelType || kernelType.getRank() != 2)
            return rewriter.notifyMatchFailure(poolOp, "invalid types");

        auto inputShape = inputType.getShape();
        auto kernelShape = kernelType.getShape();
        if (inputShape.size() != 4 || kernelShape[0] != inputShape[2] ||
            kernelShape[1] != inputShape[3])
            return rewriter.notifyMatchFailure(poolOp, "not global pooling");

        numChannels = inputShape[1];
        reducedElements = inputShape[2] * inputShape[3];
        if (reducedElements <= 0)
            return rewriter.notifyMatchFailure(poolOp, "non-positive frame size");

        // Match dequant.
        dequantOp = dyn_cast<linalg::GenericOp>(input0.getDefiningOp());
        if (!dequantOp || !matchDequantGeneric(dequantOp, dInfo.scale, dInfo.zp) ||
            !dequantOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q dequant");

        auto dequantResultType = dyn_cast<ShapedType>(dequantOp.getResult(0).getType());
        if (!dequantResultType || !dequantResultType.getElementType().isF32())
            return rewriter.notifyMatchFailure(poolOp, "dequant output must be f32");

        // Extract optional avg scale generic.
        avgScale = tryExtractAvgScale(output, avgScaleOp);
        Value quantInput = avgScaleOp ? avgScaleOp.getResult(0) : output;

        if (avgScaleOp && !avgScaleOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "avg scale op has multiple uses");

        // Match quant.
        auto quantUsers = quantInput.getUsers();
        if (quantUsers.empty())
            return rewriter.notifyMatchFailure(poolOp, "quant input has no users");
        quantOp = dyn_cast<linalg::GenericOp>(*quantUsers.begin());
        if (!quantOp || !matchQuantGeneric(quantOp, qInfo.scale, qInfo.zp, qInfo.min, qInfo.max))
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q quant");

        return success();
    }

    // Build the per-channel scale_bias tensor for the pool.
    Value buildQPoolingScaleBias(
        linalg::PoolingNchwSumOp poolOp, int64_t numChannels, int32_t multiplier, int32_t bias32,
        PatternRewriter &rewriter
    ) const {
        std::vector<int32_t> scalePerChannel(numChannels, multiplier);
        std::vector<int32_t> biasPerChannel(numChannels, bias32);
        return createI32Const(rewriter, poolOp, interleave(biasPerChannel, scalePerChannel));
    }

    // Create quantized AvgPool2DOp and replace the original chain.
    LogicalResult rewriteQPoolingChain(
        linalg::PoolingNchwSumOp poolOp, linalg::GenericOp dequantOp, linalg::GenericOp avgScaleOp,
        linalg::GenericOp quantOp, const DequantInfo &dInfo, const QuantInfo &qInfo,
        Value scaleBias, int32_t shift, PatternRewriter &rewriter
    ) const {
        Location loc = poolOp.getLoc();

        int32_t outputZp = static_cast<int32_t>(std::llround(qInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(qInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(qInfo.max));

        auto nchwToNhwc = Permutation::nchw2nhwc();
        Value nhwcInput = transposeValue(dequantOp.getInputs()[0], nchwToNhwc, loc, rewriter);
        auto nhwcOutputType = transposeType(quantOp.getResult(0).getType(), nchwToNhwc);

        Value weights = createI8Const(
            rewriter, poolOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(quantOp);

        auto avgPoolOp = torq_hl::AvgPool2DOp::create(
            rewriter, loc, nhwcOutputType, createInitTensor(poolOp, rewriter, nhwcOutputType),
            /*input_zp=*/dInfo.zp, outputZp, outputMin, outputMax, shift, weights, scaleBias,
            nhwcInput
        );

        Value torqOut = transposeValue(avgPoolOp.getOutput(), nchwToNhwc.reverse(), loc, rewriter);

        rewriter.replaceOp(quantOp, torqOut);
        if (avgScaleOp)
            rewriter.eraseOp(avgScaleOp);
        rewriter.eraseOp(poolOp);
        rewriter.eraseOp(dequantOp);
        return success();
    }

  public:
    using OpRewritePattern<linalg::PoolingNchwSumOp>::OpRewritePattern;
    QGlobalAveragePoolConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::PoolingNchwSumOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNchwSumOp poolOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(poolOp))
            return rewriter.notifyMatchFailure(poolOp, "already marked");

        linalg::GenericOp dequantOp;
        DequantInfo dInfo;
        double avgScale;
        linalg::GenericOp avgScaleOp;
        linalg::GenericOp quantOp;
        QuantInfo qInfo;
        int64_t numChannels, reducedElements;
        if (failed(matchQPoolingChain(
                poolOp, poolOp.getResultTensors()[0], rewriter, dequantOp, dInfo, avgScale,
                avgScaleOp, quantOp, qInfo, numChannels, reducedElements
            )))
            return failure();

        if (_markFuseGroups) {
            auto fuseGroupAttr = poolOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(poolOp, "missing fuse group id");
            markFuseGroupBackward(
                quantOp.getResult(0), {dequantOp.getInputs()[0]}, rewriter, fuseGroupAttr
            );
            return success();
        }

        if (qInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(poolOp, "quant scale is zero");

        double rescaleRatio = dInfo.scale * avgScale / qInfo.scale;

        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(rescaleRatio, multiplier, shift))
            return rewriter.notifyMatchFailure(poolOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QGlobalAveragePoolConvert] multiplier=" << multiplier
                                << " shift=" << shift << " reducedElements=" << reducedElements
                                << " avgScale=" << avgScale << "\n";);

        int32_t bias32 = -static_cast<int32_t>(reducedElements * static_cast<int64_t>(dInfo.zp));

        Value scaleBias = buildQPoolingScaleBias(poolOp, numChannels, multiplier, bias32, rewriter);

        return rewriteQPoolingChain(
            poolOp, dequantOp, avgScaleOp, quantOp, dInfo, qInfo, scaleBias, shift, rewriter
        );
    }
};

template <typename PoolingOpType, bool IsNCHW>
struct QMaxPoolConvert : public OpRewritePattern<PoolingOpType> {
  private:
    const bool _markFuseGroups;

    // Match the DQ -> (optional pad) -> MaxPool -> Q chain.
    LogicalResult matchQMaxPoolChain(
        PoolingOpType poolOp, PatternRewriter &rewriter, linalg::GenericOp &dequantOp,
        DequantInfo &dInfo, Value &dequantInput, RankedTensorType &dequantInputType,
        linalg::GenericOp &quantOp, QuantInfo &qInfo, PaddingInfo &padInfo, tensor::PadOp &padOp,
        linalg::FillOp &fillOp
    ) const {
        if (poolOp.getInputs().size() != 2 || poolOp.getResults().size() != 1)
            return rewriter.notifyMatchFailure(poolOp, "unexpected operand/result count");

        Value originalPoolInput = poolOp.getInputs()[0];
        Value poolOutput = poolOp.getResultTensors()[0];

        auto poolInputType = dyn_cast<RankedTensorType>(originalPoolInput.getType());
        if (!poolInputType || !poolInputType.getElementType().isF32())
            return rewriter.notifyMatchFailure(poolOp, "pool input is not f32");

        padOp = originalPoolInput.getDefiningOp<tensor::PadOp>();
        fillOp = poolOp.getOutputs()[0].template getDefiningOp<linalg::FillOp>();

        // Fold any tensor.pad that precedes the pool.
        Value poolInput = originalPoolInput;
        padInfo = foldBackwardPadding(poolInput, rewriter, IsNCHW, poolOutput);
        if (padInfo.lrtbPad.empty())
            return rewriter.notifyMatchFailure(poolOp, "failed to fold padding");

        // The pad source must be a dequant generic.
        dequantOp = dyn_cast<linalg::GenericOp>(poolInput.getDefiningOp());
        if (!dequantOp || !matchDequantGeneric(dequantOp, dInfo.scale, dInfo.zp) ||
            !dequantOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q dequant");

        dequantInput = dequantOp.getInputs()[0];
        dequantInputType = dyn_cast<RankedTensorType>(dequantInput.getType());
        if (!dequantInputType || !dequantInputType.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(poolOp, "dequant input must be int8");

        // Pool output must feed a single quant generic.
        if (!poolOutput.hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "pool output has multiple uses");

        quantOp = dyn_cast<linalg::GenericOp>(*poolOutput.getUsers().begin());
        if (!quantOp || !matchQuantGeneric(quantOp, qInfo.scale, qInfo.zp, qInfo.min, qInfo.max))
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q quant");

        // Direct int8 MaxPool is valid when input/output quant params match.
        if (dInfo.scale != qInfo.scale)
            return rewriter.notifyMatchFailure(poolOp, "mismatched quant scales");
        if (dInfo.zp != static_cast<int32_t>(std::llround(qInfo.zp)))
            return rewriter.notifyMatchFailure(poolOp, "mismatched quant zero points");

        // The pad fill value must be the minimum representable int8 value,
        // which is also the typical zero-point (-128).
        if (dInfo.zp != static_cast<int32_t>(std::llround(qInfo.min)))
            return rewriter.notifyMatchFailure(
                poolOp, "zero-point must equal output min for maxpool padding"
            );

        return success();
    }

    // Emit torq_hl.maxpool2d and replace the matched chain.
    LogicalResult rewriteQMaxPoolChain(
        PoolingOpType poolOp, linalg::GenericOp dequantOp, const DequantInfo &dInfo,
        Value dequantInput, RankedTensorType dequantInputType, linalg::GenericOp quantOp,
        const QuantInfo &qInfo, const PaddingInfo &padInfo, tensor::PadOp padOp,
        linalg::FillOp fillOp, PatternRewriter &rewriter
    ) const {
        Location loc = poolOp.getLoc();

        auto strides = attrValuesAsVec(poolOp.getStrides());
        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getInputs()[1].getType());
        auto kernelShape = kernelType.getShape();

        // Transpose input to NCHW if the pool is NHWC.
        Value transposedInput = dequantInput;
        if constexpr (!IsNCHW) {
            transposedInput = transposeValue(dequantInput, Permutation::nhwc2nchw(), loc, rewriter);
        }

        auto quantOutputType = dyn_cast<RankedTensorType>(quantOp.getResult(0).getType());
        RankedTensorType hwOutputType = quantOutputType;
        if constexpr (!IsNCHW) {
            hwOutputType = transposeType(quantOutputType, Permutation::nhwc2nchw());
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        Value weightConst = createI8Const(
            rewriter, poolOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );
        Value biasScaleConst = createI32Const(rewriter, poolOp, interleave(bias, scale));

        int32_t outputMin = static_cast<int32_t>(std::llround(qInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(qInfo.max));

        rewriter.setInsertionPoint(quantOp);

        auto maxpoolOp = torq_hl::MaxPool2dOp::create(
            rewriter, loc, hwOutputType, createInitTensor(poolOp, rewriter, hwOutputType),
            /*input_zp=*/dInfo.zp, outputMin, outputMax, strides, padInfo.lrtbPad, kernelShape,
            weightConst, biasScaleConst, transposedInput, /*segment_output=*/false
        );

        Value result = maxpoolOp.getOutput();
        if constexpr (!IsNCHW) {
            result = transposeValue(result, Permutation::nchw2nhwc(), loc, rewriter);
        }

        rewriter.replaceOp(quantOp, result);
        rewriter.eraseOp(poolOp);
        rewriter.eraseOp(dequantOp);
        if (padOp)
            rewriter.eraseOp(padOp);
        if (fillOp)
            rewriter.eraseOp(fillOp);
        return success();
    }

  public:
    using OpRewritePattern<PoolingOpType>::OpRewritePattern;
    QMaxPoolConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<PoolingOpType>(context, /*benefit=*/2), _markFuseGroups(markFuseGroups) {
    }

    LogicalResult matchAndRewrite(PoolingOpType poolOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(poolOp))
            return rewriter.notifyMatchFailure(poolOp, "already marked");

        linalg::GenericOp dequantOp, quantOp;
        DequantInfo dInfo;
        QuantInfo qInfo;
        Value dequantInput;
        RankedTensorType dequantInputType;
        PaddingInfo padInfo;
        tensor::PadOp padOp = nullptr;
        linalg::FillOp fillOp = nullptr;
        if (failed(matchQMaxPoolChain(
                poolOp, rewriter, dequantOp, dInfo, dequantInput, dequantInputType, quantOp, qInfo,
                padInfo, padOp, fillOp
            )))
            return failure();

        if (_markFuseGroups) {
            auto fuseGroupAttr = poolOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(poolOp, "missing fuse group id");
            markFuseGroupBackward(quantOp.getResult(0), {dequantInput}, rewriter, fuseGroupAttr);
            return success();
        }

        return rewriteQMaxPoolChain(
            poolOp, dequantOp, dInfo, dequantInput, dequantInputType, quantOp, qInfo, padInfo,
            padOp, fillOp, rewriter
        );
    }
};

} // namespace

void populateLinalgToTorqHLQPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QGlobalAveragePoolConvert>(context, markFuseGroups);
    patterns.insert<QMaxPoolConvert<linalg::PoolingNchwMaxOp, true>>(context, markFuseGroups);
    patterns.insert<QMaxPoolConvert<linalg::PoolingNhwcMaxOp, false>>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
