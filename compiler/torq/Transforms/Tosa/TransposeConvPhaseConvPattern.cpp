// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/HwInfo.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>

namespace mlir::syna::torq {

namespace {

// The phase conv of a transpose conv that has a stride on one spatial axis. Output
// stride * p + r of the transpose conv is channel r * OC + o of output p of the phase conv. Tap t
// of the phase conv reads in[p + t - halfWidth]. The phase conv has 2 * halfWidth + 1 taps and a
// pad of halfWidth on each side. This is the SAME pad that the HW makes for an odd kernel.
struct OneAxisPhaseConv {
    int64_t halfWidth = 0;
    // kernelTap[r][t] is the transpose conv tap of phase r and phase conv tap t, or -1 for a
    // zero tap.
    SmallVector<SmallVector<int64_t>> kernelTap;
};

// Compute the OneAxisPhaseConv of an axis with `kernelSize` taps, stride `stride` and an output
// pad of `outPadLow` before the first output.
static OneAxisPhaseConv
computeOneAxisPhaseConv(int64_t kernelSize, int64_t stride, int64_t outPadLow) {
    // TOSA: out[stride * i + outPadLow + k] += in[i] * w[k]. So output stride * p + r reads
    // in[p + j] * w[k], with j = (r - outPadLow - k) / stride. Collect (r, j, k) for each tap.
    SmallVector<std::array<int64_t, 3>> taps;
    OneAxisPhaseConv phaseConv;
    for (int64_t r = 0; r < stride; ++r) {
        for (int64_t k = 0; k < kernelSize; ++k) {
            if ((r - outPadLow - k) % stride != 0)
                continue;
            int64_t j = (r - outPadLow - k) / stride;
            taps.push_back({r, j, k});
            phaseConv.halfWidth = std::max(phaseConv.halfWidth, std::abs(j));
        }
    }
    phaseConv.kernelTap.assign(stride, SmallVector<int64_t>(2 * phaseConv.halfWidth + 1, -1));
    for (auto [r, j, k] : taps)
        phaseConv.kernelTap[r][j + phaseConv.halfWidth] = k;
    return phaseConv;
}

// Return the raw data of `values`, with a splat expanded to all elements.
static SmallVector<char> getExpandedRawData(DenseElementsAttr values) {
    ArrayRef<char> raw = values.getRawData();
    if (!values.isSplat())
        return SmallVector<char>(raw);
    SmallVector<char> data;
    data.reserve(raw.size() * values.getNumElements());
    for (int64_t i = 0; i < values.getNumElements(); ++i)
        data.append(raw.begin(), raw.end());
    return data;
}

// Return the OHWI weights of `phaseConv` on spatial axis `axis` (1 is H, 2 is W). `weights` are
// the OHWI weights of the transpose conv, with a kernel size of 1 on the other spatial axis.
static DenseElementsAttr
getOneAxisPhaseWeights(DenseElementsAttr weights, const OneAxisPhaseConv &phaseConv, int64_t axis) {
    auto type = cast<RankedTensorType>(weights.getType());
    ArrayRef<int64_t> shape = type.getShape();
    int64_t outChannels = shape[0];
    int64_t kernelSize = shape[axis];
    int64_t stride = phaseConv.kernelTap.size();
    int64_t taps = 2 * phaseConv.halfWidth + 1;
    SmallVector<int64_t> phaseShape = {stride * outChannels, 1, 1, shape[3]};
    phaseShape[axis] = taps;
    auto phaseType = RankedTensorType::get(phaseShape, type.getElementType());

    // Each (output channel, tap) row holds the input channels. A zero tap stays all zero bytes.
    int64_t elementBytes = type.getElementTypeBitWidth() / 8;
    int64_t rowBytes = shape[3] * elementBytes;
    SmallVector<char> src = getExpandedRawData(weights);
    SmallVector<char> dst(phaseType.getNumElements() * elementBytes, 0);
    for (int64_t r = 0; r < stride; ++r) {
        for (int64_t o = 0; o < outChannels; ++o) {
            for (int64_t t = 0; t < taps; ++t) {
                int64_t k = phaseConv.kernelTap[r][t];
                if (k < 0)
                    continue;
                int64_t srcOffset = (o * kernelSize + k) * rowBytes;
                int64_t dstOffset = ((r * outChannels + o) * taps + t) * rowBytes;
                std::copy_n(src.begin() + srcOffset, rowBytes, dst.begin() + dstOffset);
            }
        }
    }
    return DenseElementsAttr::getFromRawBuffer(phaseType, dst);
}

// Return a constant with the per-channel `values` repeated `count` times. A constant with one
// element applies to all channels, so return it unchanged.
static Value repeatChannels(Value values, int64_t count, Location loc, PatternRewriter &rewriter) {
    DenseElementsAttr attr;
    if (!matchPattern(values, m_Constant(&attr)) || attr.getNumElements() == 1)
        return values;
    SmallVector<Attribute> elements;
    for (int64_t i = 0; i < count; ++i)
        llvm::append_range(elements, attr.getValues<Attribute>());
    auto type =
        RankedTensorType::get({static_cast<int64_t>(elements.size())}, attr.getElementType());
    auto repeated = DenseElementsAttr::get(type, elements);
    return tosa::ConstOp::create(rewriter, loc, type, repeated);
}

// Return true if `value` is a constant zero, as an integer or a float.
static bool isZeroConstant(Value value) {
    return matchPattern(value, m_Zero()) || matchPattern(value, m_AnyZeroFloat());
}

// Return true if `rescaleOp` has constant per-channel parameters, which repeatChannels can copy.
static bool hasConstantRescaleParameters(tosa::RescaleOp rescaleOp) {
    return matchPattern(rescaleOp.getMultiplier(), m_Constant()) &&
           matchPattern(rescaleOp.getShift(), m_Constant());
}

// Return the strided spatial axis (1 is H, 2 is W) of `srcOp` if OneAxisPhaseConv can lower it.
// The other spatial axis must have a kernel of 1, a stride of 1 and no output pad. The output
// must be exactly `stride` times the input on the strided axis. A strided H axis also needs
// W == 1, so that a reshape interleaves the phases.
static FailureOr<int64_t> getOneStridedAxis(tosa::TransposeConv2DOp srcOp) {
    auto inputType = dyn_cast<RankedTensorType>(srcOp.getInput().getType());
    auto weightType = dyn_cast<RankedTensorType>(srcOp.getWeight().getType());
    auto resultType = dyn_cast<RankedTensorType>(srcOp.getType());
    if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
        !weightType.hasStaticShape() || !resultType.hasStaticShape())
        return failure();
    ArrayRef<int64_t> stride = srcOp.getStride();
    ArrayRef<int64_t> outPad = srcOp.getOutPad();
    int64_t axis = stride[1] > 1 ? 2 : 1;
    int64_t other = 3 - axis;
    if (stride[axis - 1] < 2 || stride[other - 1] != 1 || weightType.getDimSize(other) != 1 ||
        outPad[2 * (other - 1)] != 0 || outPad[2 * (other - 1) + 1] != 0)
        return failure();
    if (resultType.getDimSize(axis) != inputType.getDimSize(axis) * stride[axis - 1] ||
        resultType.getDimSize(other) != inputType.getDimSize(other))
        return failure();
    if (axis == 1 && inputType.getDimSize(2) != 1)
        return failure();
    return axis;
}

// The largest ratio of the weight bytes that the zero taps add to the input bytes, for a stride on
// W. Above it, the larger weights cost more than the XRAM round trip of the explicit pad.
// Measured only on the stride (1, 2) audio decoder layers, with 1x3 and 1x5 kernels: a ratio of 6
// or less was faster on SL2610 and SR200. A ratio of 8 or more was slower on at least one chip.
// Measure again before relying on it for other strides or kernels.
constexpr int64_t kMaxAddedWeightToInputRatio = 6;

// The same limit for a stride on H (with W == 1), where the phase conv is a Kx1 conv on a single
// column. Measured on the stride (2, 1) Keras cases: on one slice (SR200, or SL2610 without
// slicing), a ratio of 3 or less was faster and a ratio of 6 slower. Split across two slices
// (SL2610), its extra taps cost more slice time: a 32x1x48 int8 input with 48 output channels
// (ratio 3) was 20 % slower, so the limit there is lower.
constexpr int64_t kMaxAddedWeightToInputRatioH = 3;
constexpr int64_t kMaxAddedWeightToInputRatioHSliced = 2;

// Return true if the zero taps of `phaseConv` add too many weight bytes for the input of
// `srcOp`. The upstream decomposition uses ceil(kernelSize / stride) taps for each phase.
static bool
addsTooManyWeights(tosa::TransposeConv2DOp srcOp, const OneAxisPhaseConv &phaseConv, int64_t axis) {
    auto weightType = cast<RankedTensorType>(srcOp.getWeight().getType());
    auto inputType = cast<RankedTensorType>(srcOp.getInput().getType());
    int64_t stride = phaseConv.kernelTap.size();
    int64_t upstreamTaps = llvm::divideCeil(weightType.getDimSize(axis), stride);
    int64_t addedTaps = 2 * phaseConv.halfWidth + 1 - upstreamTaps;
    if (addedTaps <= 0)
        return false;
    int64_t addedWeightBytes = stride * weightType.getDimSize(0) * weightType.getDimSize(3) *
                               addedTaps * weightType.getElementTypeBitWidth() / 8;
    int64_t inputBytes = inputType.getNumElements() * inputType.getElementTypeBitWidth() / 8;
    int64_t maxRatio = kMaxAddedWeightToInputRatio;
    if (axis == 1)
        maxRatio = TorqHw::get().getSliceCount() > 1 ? kMaxAddedWeightToInputRatioHSliced
                                                     : kMaxAddedWeightToInputRatioH;
    return addedWeightBytes > maxRatio * inputBytes;
}

// Change a transpose conv with a stride s on one spatial axis into one stride-1 conv with s
// times the output channels, and a reshape that interleaves the phases. The upstream
// decomposition makes a conv with an even kernel and a pad on both sides. The HW cannot make
// that pad, so it stays a fill and an insert_slice in XRAM. The conv of this pattern has an odd
// kernel with zero taps, and its pad is the HW SAME pad. When a rescale is the only user of the
// transpose conv, the conv gets a copy of the rescale, so that the rescale stays fused.
struct TransposeConv2DOneAxisPhaseConv : public OpRewritePattern<tosa::TransposeConv2DOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tosa::TransposeConv2DOp srcOp, PatternRewriter &rewriter) const override {
        FailureOr<int64_t> axis = getOneStridedAxis(srcOp);
        if (failed(axis))
            return rewriter.notifyMatchFailure(srcOp, "expected a stride on one axis only");
        DenseElementsAttr weights;
        if (!matchPattern(srcOp.getWeight(), m_Constant(&weights)) ||
            !isZeroConstant(srcOp.getWeightZp()) || !weights.getElementType().isIntOrFloat() ||
            weights.getElementType().getIntOrFloatBitWidth() % 8 != 0)
            return rewriter.notifyMatchFailure(srcOp, "expected whole-byte constant weights");
        if (!matchPattern(srcOp.getBias(), m_Constant()))
            return rewriter.notifyMatchFailure(srcOp, "expected a constant bias");

        int64_t stride = srcOp.getStride()[*axis - 1];
        OneAxisPhaseConv phaseConv = computeOneAxisPhaseConv(
            weights.getType().getDimSize(*axis), stride, srcOp.getOutPad()[2 * (*axis - 1)]
        );
        int64_t maxKernelSize =
            *axis == 2 ? HwInfo::nss_max_kernel_width : HwInfo::nss_max_kernel_height;
        if (2 * phaseConv.halfWidth + 1 > maxKernelSize)
            return rewriter.notifyMatchFailure(srcOp, "phase conv kernel is too large");
        if (addsTooManyWeights(srcOp, phaseConv, *axis))
            return rewriter.notifyMatchFailure(srcOp, "zero taps add too many weight bytes");

        Value conv = buildOneAxisPhaseConv(srcOp, weights, phaseConv, *axis, rewriter);
        tosa::RescaleOp rescaleOp = getOnlyRescaleUser(srcOp);
        if (rescaleOp && hasConstantRescaleParameters(rescaleOp)) {
            // The parameters of the rescale can be defined after the transpose conv.
            rewriter.setInsertionPoint(rescaleOp);
            Value rescale = copyRescale(rescaleOp, conv, stride, rewriter);
            rewriter.replaceOp(rescaleOp, reshapeLike(rescale, rescaleOp.getType(), rewriter));
            rewriter.eraseOp(srcOp);
            return success();
        }
        rewriter.replaceOp(srcOp, reshapeLike(conv, srcOp.getType(), rewriter));
        return success();
    }

  private:
    // Build the phase conv of `srcOp` on its NHWC input.
    static Value buildOneAxisPhaseConv(
        tosa::TransposeConv2DOp srcOp, DenseElementsAttr weights, const OneAxisPhaseConv &phaseConv,
        int64_t axis, PatternRewriter &rewriter
    ) {
        Location loc = srcOp.getLoc();
        int64_t stride = phaseConv.kernelTap.size();
        DenseElementsAttr phaseWeights = getOneAxisPhaseWeights(weights, phaseConv, axis);
        Value weightsValue =
            tosa::ConstOp::create(rewriter, loc, phaseWeights.getType(), phaseWeights);
        Value bias = repeatChannels(srcOp.getBias(), stride, loc, rewriter);

        auto inputType = cast<RankedTensorType>(srcOp.getInput().getType());
        auto resultType = cast<RankedTensorType>(srcOp.getType());
        SmallVector<int64_t> convShape(inputType.getShape());
        convShape[3] = stride * resultType.getDimSize(3);
        SmallVector<int64_t> pad = {0, 0, 0, 0};
        pad[2 * (axis - 1)] = phaseConv.halfWidth;
        pad[2 * (axis - 1) + 1] = phaseConv.halfWidth;
        return tosa::Conv2DOp::create(
            rewriter, loc, RankedTensorType::get(convShape, resultType.getElementType()),
            srcOp.getInput(), weightsValue, bias, srcOp.getInputZp(), srcOp.getWeightZp(),
            rewriter.getDenseI64ArrayAttr(pad), rewriter.getDenseI64ArrayAttr({1, 1}),
            rewriter.getDenseI64ArrayAttr({1, 1}), srcOp.getAccTypeAttr(), srcOp.getLocalBoundAttr()
        );
    }

    // Return the rescale that is the only user of `srcOp`, or a null op.
    static tosa::RescaleOp getOnlyRescaleUser(tosa::TransposeConv2DOp srcOp) {
        if (!srcOp->hasOneUse())
            return nullptr;
        return dyn_cast<tosa::RescaleOp>(*srcOp->getUsers().begin());
    }

    // Apply a copy of `rescaleOp` to `conv`, with the per-channel parameters repeated for each
    // of the `stride` phases.
    static Value
    copyRescale(tosa::RescaleOp rescaleOp, Value conv, int64_t stride, PatternRewriter &rewriter) {
        Location loc = rescaleOp.getLoc();
        Value multiplier = repeatChannels(rescaleOp.getMultiplier(), stride, loc, rewriter);
        Value shift = repeatChannels(rescaleOp.getShift(), stride, loc, rewriter);
        auto convType = cast<RankedTensorType>(conv.getType());
        Type elementType = cast<RankedTensorType>(rescaleOp.getType()).getElementType();
        return tosa::RescaleOp::create(
            rewriter, loc, convType.clone(elementType), conv, multiplier, shift,
            rescaleOp.getInputZp(), rescaleOp.getOutputZp(), rescaleOp.getScale32Attr(),
            rescaleOp.getRoundingModeAttr(), rescaleOp.getPerChannelAttr(),
            rescaleOp.getInputUnsignedAttr(), rescaleOp.getOutputUnsignedAttr()
        );
    }

    // Reshape `value` to the shape of `type`. This interleaves the phases on the strided axis.
    static Value reshapeLike(Value value, Type type, PatternRewriter &rewriter) {
        Location loc = value.getLoc();
        auto shapedType = cast<RankedTensorType>(type);
        Value shape = tosa::getTosaConstShape(rewriter, loc, shapedType.getShape());
        return tosa::ReshapeOp::create(rewriter, loc, shapedType, value, shape);
    }
};

} // namespace

void populateTransposeConvPhaseConvPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<TransposeConv2DOneAxisPhaseConv>(context);
}

} // namespace mlir::syna::torq
