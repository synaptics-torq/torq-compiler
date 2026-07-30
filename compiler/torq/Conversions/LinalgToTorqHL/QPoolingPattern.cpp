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
#include "torq/Utils/ExecutorAssignment.h"

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

// Fuses DQ -> pooling_nchw_sum -> [div-by-count] -> Q into `torq_hl.AvgPool2DOp`.
// Only matches global average pooling, i.e. the kernel covers the whole HxW
// frame and the spatial output is 1x1.
struct QGlobalAveragePoolConvert : public OpRewritePattern<linalg::PoolingNchwSumOp> {
  private:
    const bool _markFuseGroups;

    // Validate that this is global average pooling: kernel covers the whole HxW
    // frame and produces a 1x1 spatial output.
    LogicalResult checkGlobalPoolShape(
        linalg::PoolingNchwSumOp poolOp, int64_t &numChannels, int64_t &reducedElements,
        PatternRewriter &rewriter
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
        linalg::PoolingNchwSumOp poolOp, linalg::GenericOp avgScaleOp, QuantizedOpChain &chain,
        Value scaleBias, int32_t shift, PatternRewriter &rewriter
    ) const {
        Location loc = poolOp.getLoc();

        int32_t outputZp = static_cast<int32_t>(std::llround(chain.outputInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(chain.outputInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(chain.outputInfo.max));

        auto nchwToNhwc = Permutation::nchw2nhwc();
        Value nhwcInput =
            transposeValue(chain.inputDequantOp.getInputs()[0], nchwToNhwc, loc, rewriter);
        auto nhwcOutputType = transposeType(chain.quantOp.getResult(0).getType(), nchwToNhwc);

        Value weights = createI8Const(
            rewriter, poolOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(chain.quantOp);

        auto avgPoolOp = torq_hl::AvgPool2DOp::create(
            rewriter, loc, nhwcOutputType, createInitTensor(poolOp, rewriter, nhwcOutputType),
            /*input_zp=*/chain.inputInfo.zp, outputZp, outputMin, outputMax, shift, weights,
            scaleBias, nhwcInput
        );

        Value torqOut = transposeValue(avgPoolOp.getOutput(), nchwToNhwc.reverse(), loc, rewriter);

        rewriter.replaceOp(chain.quantOp, torqOut);
        if (avgScaleOp)
            rewriter.eraseOp(avgScaleOp);
        rewriter.eraseOp(poolOp);
        rewriter.eraseOp(chain.inputDequantOp);
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

        int64_t numChannels, reducedElements;
        if (failed(checkGlobalPoolShape(poolOp, numChannels, reducedElements, rewriter)))
            return failure();

        Value poolOutput = poolOp.getResultTensors()[0];
        linalg::GenericOp avgScaleOp;
        double avgScale = tryExtractAvgScale(poolOutput, avgScaleOp);
        Value quantInput = avgScaleOp ? avgScaleOp.getResult(0) : poolOutput;
        if (avgScaleOp && !avgScaleOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "avg scale op has multiple uses");

        QuantizedOpChain chain;
        if (failed(matchInputQuantization(poolOp.getInputs()[0], rewriter, chain)))
            return failure();
        if (failed(matchOutputQuantization(quantInput, rewriter, chain)))
            return failure();

        if (_markFuseGroups) {
            auto fuseGroupAttr = poolOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(poolOp, "missing fuse group id");
            markFuseGroupBackward(
                chain.quantOp.getResult(0), {chain.inputDequantOp.getInputs()[0]}, rewriter,
                fuseGroupAttr
            );
            return success();
        }

        if (chain.outputInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(poolOp, "quant scale is zero");

        double rescaleRatio = chain.inputInfo.scale * avgScale / chain.outputInfo.scale;

        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(rescaleRatio, multiplier, shift))
            return rewriter.notifyMatchFailure(poolOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QGlobalAveragePoolConvert] multiplier=" << multiplier
                                << " shift=" << shift << " reducedElements=" << reducedElements
                                << " avgScale=" << avgScale << "\n";);

        int32_t bias32 =
            -static_cast<int32_t>(reducedElements * static_cast<int64_t>(chain.inputInfo.zp));

        Value scaleBias = buildQPoolingScaleBias(poolOp, numChannels, multiplier, bias32, rewriter);

        return rewriteQPoolingChain(poolOp, avgScaleOp, chain, scaleBias, shift, rewriter);
    }
};

// Local vs. global quantized AveragePool lowering.
//
// Linalg represents AveragePool as `pooling_nchw_sum` followed by a div-by-count.
// * Global average pool: the kernel covers the entire HxW frame and produces a
//   1x1 spatial output.  TORQ has a dedicated `AvgPool2DOp` for this case, so
//   `QGlobalAveragePoolConvert` fuses DQ -> pool -> [div] -> Q into that op.
// * Local average pool: the kernel is smaller than the frame and slides across
//   the spatial dimensions.  TORQ has no dedicated local-average-pool op, so we
//   map the same DQ -> pad -> pool_sum -> div -> Q chain to a
//   `torq_hl.depthwise_conv2d` with unit weights.
//
// A depthwise conv2d can implement quantized local AveragePool because the two
// are mathematically equivalent when the int8 sum is corrected for the input
// zero-point and the fixed-point rescale absorbs the divide-by-count:
//   AvgPool(x) = (1/N) * sum_k (x_i(k) - input_zp) * input_scale
//   DWConv(x)  = sum_k x_i(k) + bias
// With bias = -N * input_zp and rescale = input_scale / (N * output_scale),
// quantizing AvgPool(x) gives the same result as the DW backend's
// `((DWConv(x) * mult) >> shift) + output_zp`.
//
// The DW backend's implicit zero padding uses input_zp, which dequantizes to
// 0.0; that matches the f32 pad constant required by AveragePool when
// count_include_pad == 1.
//
// Convert a quantized local AveragePool (DQ -> pad -> pooling_nchw_sum -> div -> Q)
// to a torq_hl.depthwise_conv2d.
struct QLocalAveragePoolConvert : public OpRewritePattern<linalg::PoolingNchwSumOp> {
  private:
    const bool _markFuseGroups;

    // Match the local AveragePool quantization chain:
    //   DQ -> [pad] -> pool_sum -> div-by-count -> Q
    // The caller handles the optional tensor.pad on the input side and the
    // optional div-by-count generic on the output side; the shared helpers
    // match the DQ and Q.
    LogicalResult matchQLocalAvgPoolChain(
        linalg::PoolingNchwSumOp poolOp, Value output, PatternRewriter &rewriter,
        QuantizedOpChain &chain, double &avgScale, linalg::GenericOp &avgScaleOp,
        tensor::PadOp &padOp, linalg::FillOp &fillOp, int64_t &numChannels, int64_t &kernelElements
    ) const {
        Value input0 = poolOp.getInputs()[0];
        Value kernel = poolOp.getInputs()[1];
        auto inputType = dyn_cast<RankedTensorType>(input0.getType());
        auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
        if (!inputType || !kernelType || kernelType.getRank() != 2)
            return rewriter.notifyMatchFailure(poolOp, "invalid pool types");

        auto inputShape = inputType.getShape();
        auto kernelShape = kernelType.getShape();
        if (inputShape.size() != 4)
            return rewriter.notifyMatchFailure(poolOp, "input is not 4-D");
        // Only local pooling; global average pool is handled by QGlobalAveragePoolConvert.
        if (kernelShape[0] == inputShape[2] && kernelShape[1] == inputShape[3])
            return rewriter.notifyMatchFailure(poolOp, "global average pool, not local");

        numChannels = inputShape[1];
        kernelElements = kernelShape[0] * kernelShape[1];
        if (kernelElements <= 0)
            return rewriter.notifyMatchFailure(poolOp, "non-positive kernel size");

        padOp = input0.getDefiningOp<tensor::PadOp>();
        fillOp = dyn_cast_or_null<linalg::FillOp>(poolOp.getOutputs()[0].getDefiningOp());

        // The pad and fill must be owned exclusively by this pool so we can
        // safely erase them.  A shared pad/fill would be matched by multiple
        // patterns and could be erased twice, corrupting the conversion state.
        if (padOp && !padOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "shared pad");
        if (fillOp && !fillOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "shared fill");

        // Match the input DQ (after walking through the optional pad).
        Value dequantValue = padOp ? padOp.getSource() : input0;
        if (failed(matchInputQuantization(dequantValue, rewriter, chain)))
            return rewriter.notifyMatchFailure(poolOp, "failed to match input DQ");

        auto dequantInput = chain.inputDequantOp.getInputs()[0];
        auto dequantInputType = dyn_cast<RankedTensorType>(dequantInput.getType());
        if (!dequantInputType || !dequantInputType.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(poolOp, "DQ input must be int8");

        // Extract the downstream div-by-count generic.
        avgScale = tryExtractAvgScale(output, avgScaleOp);
        if (avgScale == 1.0)
            return rewriter.notifyMatchFailure(poolOp, "missing div-by-count scale");
        if (avgScaleOp && !avgScaleOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "avg scale op has multiple uses");

        // Match the output Q (consuming the scaled pool output).
        Value quantInput = avgScaleOp ? avgScaleOp.getResult(0) : output;
        if (failed(matchOutputQuantization(quantInput, rewriter, chain)))
            return rewriter.notifyMatchFailure(poolOp, "failed to match output Q");

        return success();
    }

    // Build the per-channel scale_bias tensor: [bias_0, mult, bias_1, mult, ...].
    // Use the deferred constant-resolution path so the DW backend can resolve it
    // at compile time, just like QConv2DPattern does for weighted ops.
    Value buildScaleBias(
        linalg::PoolingNchwSumOp poolOp, int64_t numChannels, int32_t multiplier, int32_t bias32,
        PatternRewriter &rewriter
    ) const {
        Location loc = poolOp.getLoc();
        auto biasTy = RankedTensorType::get({numChannels}, rewriter.getI32Type());
        std::vector<int32_t> biasValues(numChannels, bias32);
        Value bias =
            arith::ConstantOp::create(rewriter, loc, biasTy, rewriter.getI32TensorAttr(biasValues))
                .getResult();
        Value scaleBias = buildDynamicInterleavedBiasScale(bias, multiplier, loc, rewriter);
        if (!scaleBias)
            return nullptr;
        auto folded = createCompileTimeConstOp(scaleBias.getDefiningOp(), rewriter);
        return succeeded(folded) ? *folded : scaleBias;
    }

  public:
    using OpRewritePattern<linalg::PoolingNchwSumOp>::OpRewritePattern;
    QLocalAveragePoolConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::PoolingNchwSumOp>(context, /*benefit=*/3),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNchwSumOp poolOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(poolOp))
            return rewriter.notifyMatchFailure(poolOp, "already marked");

        QuantizedOpChain chain;
        double avgScale;
        linalg::GenericOp avgScaleOp;
        tensor::PadOp padOp = nullptr;
        linalg::FillOp fillOp = nullptr;
        int64_t numChannels, kernelElements;
        if (failed(matchQLocalAvgPoolChain(
                poolOp, poolOp.getResultTensors()[0], rewriter, chain, avgScale, avgScaleOp, padOp,
                fillOp, numChannels, kernelElements
            )))
            return failure();

        // Validate the chain without mutating IR first, then fold the padding so
        // the DW conv can absorb it.  This avoids partial IR mutation when the
        // chain does not actually convert.
        Value poolInput = poolOp.getInputs()[0];
        PaddingInfo padInfo =
            foldBackwardPadding(poolInput, rewriter, /*nchw=*/true, poolOp.getResultTensors()[0]);
        if (padInfo.lrtbPad.empty())
            return rewriter.notifyMatchFailure(poolOp, "failed to fold padding");

        if (_markFuseGroups) {
            auto fuseGroupAttr = poolOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(poolOp, "missing fuse group id");
            markFuseGroupBackward(
                chain.quantOp.getResult(0), {chain.inputDequantOp.getInputs()[0]}, rewriter,
                fuseGroupAttr
            );
            return success();
        }

        if (chain.outputInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(poolOp, "quant scale is zero");

        double rescaleRatio = chain.inputInfo.scale * avgScale / chain.outputInfo.scale;

        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(rescaleRatio, multiplier, shift))
            return rewriter.notifyMatchFailure(poolOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QLocalAveragePoolConvert] multiplier=" << multiplier
                                << " shift=" << shift << " kernelElements=" << kernelElements
                                << " avgScale=" << avgScale << "\n";);

        // The depthwise conv backend does not subtract the input zero-point from
        // each input element, so we fold the dequantization zero-point into the
        // per-channel bias as -input_zp * kernelElements.
        int32_t bias32 =
            -static_cast<int32_t>(kernelElements * static_cast<int64_t>(chain.inputInfo.zp));
        Value scaleBias = buildScaleBias(poolOp, numChannels, multiplier, bias32, rewriter);
        if (!scaleBias)
            return rewriter.notifyMatchFailure(poolOp, "failed to build scale_bias");

        Location loc = poolOp.getLoc();
        Value input = chain.inputDequantOp.getInputs()[0];
        auto quantOutputType = dyn_cast<RankedTensorType>(chain.quantOp.getResult(0).getType());

        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getInputs()[1].getType());
        auto kernelShape = kernelType.getShape();
        int64_t C = numChannels;
        int64_t KH = kernelShape[0];
        int64_t KW = kernelShape[1];
        std::vector<int8_t> weightsData(C * KH * KW, 1);
        Value weights =
            createI8Const(rewriter, poolOp, weightsData, llvm::ArrayRef<int64_t>{C, 1, KH, KW});
        auto foldedWeights = createCompileTimeConstOp(weights.getDefiningOp(), rewriter);
        if (succeeded(foldedWeights))
            weights = *foldedWeights;

        auto strides = attrValuesAsVec(poolOp.getStrides());
        auto dilations = attrValuesAsVec(poolOp.getDilations());

        int32_t outputZp = static_cast<int32_t>(std::llround(chain.outputInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(chain.outputInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(chain.outputInfo.max));

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(chain.quantOp);

        auto dwOp = torq_hl::DepthwiseConv2DOp::create(
            rewriter, loc, quantOutputType, createInitTensor(poolOp, rewriter, quantOutputType),
            /*input_zp=*/chain.inputInfo.zp, /*weight_zp=*/0, outputZp, outputMin, outputMax, shift,
            /*groups=*/static_cast<int32_t>(C), padInfo.lrtbPad, strides, dilations,
            torq_hl::VectorizationModeEnum::None, weights, scaleBias, input,
            /*isDw1dStride1=*/false, /*segment_output=*/false, /*nhwc_input=*/false
        );

        rewriter.replaceOp(chain.quantOp, dwOp.getOutput());
        if (avgScaleOp)
            rewriter.eraseOp(avgScaleOp);
        rewriter.eraseOp(poolOp);
        rewriter.eraseOp(chain.inputDequantOp);
        if (padOp)
            rewriter.eraseOp(padOp);
        if (fillOp)
            rewriter.eraseOp(fillOp);
        return success();
    }
};

template <typename PoolingOpType, bool IsNCHW>
struct QMaxPoolConvert : public OpRewritePattern<PoolingOpType> {
  private:
    const bool _markFuseGroups;

    // Match the DQ -> (optional pad) -> MaxPool -> Q chain.
    // The caller handles the optional tensor.pad; the shared helpers match the
    // DQ and Q.  MaxPool-specific quant-parameter validation stays here.
    LogicalResult matchQMaxPoolChain(
        PoolingOpType poolOp, PatternRewriter &rewriter, QuantizedOpChain &chain,
        Value &dequantInput, RankedTensorType &dequantInputType, PaddingInfo &padInfo,
        tensor::PadOp &padOp, linalg::FillOp &fillOp
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

        // The pad and fill must be owned exclusively by this pool so we can
        // safely erase them.  A shared pad/fill would be matched by multiple
        // patterns and could be erased twice, corrupting the conversion state.
        if (padOp && !padOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "shared pad");
        if (fillOp && !fillOp->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(poolOp, "shared fill");

        // Fold any tensor.pad that precedes the pool.
        Value poolInput = originalPoolInput;
        padInfo = foldBackwardPadding(poolInput, rewriter, IsNCHW, poolOutput);
        if (padInfo.lrtbPad.empty())
            return rewriter.notifyMatchFailure(poolOp, "failed to fold padding");

        // Match the input DQ (after walking through the optional pad).
        if (failed(matchInputQuantization(poolInput, rewriter, chain)))
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q dequant");

        dequantInput = chain.inputDequantOp.getInputs()[0];
        dequantInputType = dyn_cast<RankedTensorType>(dequantInput.getType());
        if (!dequantInputType || !dequantInputType.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(poolOp, "dequant input must be int8");

        // Match the output Q consuming the pool output.
        if (failed(matchOutputQuantization(poolOutput, rewriter, chain)))
            return rewriter.notifyMatchFailure(poolOp, "failed to match Q quant");

        // Direct int8 MaxPool is valid when input/output quant params match.
        if (chain.inputInfo.scale != chain.outputInfo.scale)
            return rewriter.notifyMatchFailure(poolOp, "mismatched quant scales");
        if (chain.inputInfo.zp != static_cast<int32_t>(std::llround(chain.outputInfo.zp)))
            return rewriter.notifyMatchFailure(poolOp, "mismatched quant zero points");

        // The pad fill value must be the minimum representable int8 value,
        // which is also the typical zero-point (-128).
        if (chain.inputInfo.zp != static_cast<int32_t>(std::llround(chain.outputInfo.min)))
            return rewriter.notifyMatchFailure(
                poolOp, "zero-point must equal output min for maxpool padding"
            );

        return success();
    }

    // Emit torq_hl.maxpool2d and replace the matched chain.
    LogicalResult rewriteQMaxPoolChain(
        PoolingOpType poolOp, QuantizedOpChain &chain, Value dequantInput,
        RankedTensorType dequantInputType, const PaddingInfo &padInfo, tensor::PadOp padOp,
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

        auto quantOutputType = dyn_cast<RankedTensorType>(chain.quantOp.getResult(0).getType());
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

        int32_t outputMin = static_cast<int32_t>(std::llround(chain.outputInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(chain.outputInfo.max));

        rewriter.setInsertionPoint(chain.quantOp);

        auto maxpoolOp = torq_hl::MaxPool2dOp::create(
            rewriter, loc, hwOutputType, createInitTensor(poolOp, rewriter, hwOutputType),
            /*input_zp=*/chain.inputInfo.zp, outputMin, outputMax, strides, padInfo.lrtbPad,
            kernelShape, weightConst, biasScaleConst, transposedInput, /*segment_output=*/false
        );

        Value result = maxpoolOp.getOutput();
        if constexpr (!IsNCHW) {
            result = transposeValue(result, Permutation::nchw2nhwc(), loc, rewriter);
        }

        rewriter.replaceOp(chain.quantOp, result);
        rewriter.eraseOp(poolOp);
        rewriter.eraseOp(chain.inputDequantOp);
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

        QuantizedOpChain chain;
        Value dequantInput;
        RankedTensorType dequantInputType;
        PaddingInfo padInfo;
        tensor::PadOp padOp = nullptr;
        linalg::FillOp fillOp = nullptr;
        if (failed(matchQMaxPoolChain(
                poolOp, rewriter, chain, dequantInput, dequantInputType, padInfo, padOp, fillOp
            )))
            return failure();

        if (_markFuseGroups) {
            auto fuseGroupAttr = poolOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(poolOp, "missing fuse group id");
            markFuseGroupBackward(
                chain.quantOp.getResult(0), {dequantInput}, rewriter, fuseGroupAttr
            );
            return success();
        }

        return rewriteQMaxPoolChain(
            poolOp, chain, dequantInput, dequantInputType, padInfo, padOp, fillOp, rewriter
        );
    }
};

} // namespace

void populateLinalgToTorqHLQPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QGlobalAveragePoolConvert>(context, markFuseGroups);
    patterns.insert<QLocalAveragePoolConvert>(context, markFuseGroups);
    patterns.insert<QMaxPoolConvert<linalg::PoolingNchwMaxOp, true>>(context, markFuseGroups);
    patterns.insert<QMaxPoolConvert<linalg::PoolingNhwcMaxOp, false>>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
