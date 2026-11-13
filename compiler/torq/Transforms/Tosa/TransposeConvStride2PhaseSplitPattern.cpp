// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#include <algorithm>
#include <cstdint>

namespace mlir::syna::torq {

namespace {

// The part of a stride-2 transpose conv along one spatial axis that makes the outputs of one
// parity. Output o = 2 * m + parity uses the kernel taps firstTap + 2 * t, for t < taps. As a
// stride-1 conv with the taps in reverse order, it reads the input with padLow and padHigh zero
// rows added. A negative pad becomes a crop of the input (cropLow and cropHigh).
struct PhaseAxis {
    int64_t firstTap;
    int64_t taps;
    int64_t outSize;
    int64_t padLow;
    int64_t padHigh;
    int64_t cropLow;
    int64_t cropHigh;
};

// Compute the PhaseAxis of `parity` for an axis with `inSize` inputs, `kernelSize` taps,
// `outSize` outputs and an output pad of `outPadLow` before the first output.
static PhaseAxis computePhaseAxis(
    int64_t inSize, int64_t kernelSize, int64_t outSize, int64_t outPadLow, int64_t parity
) {
    PhaseAxis axis;
    // TOSA: out[2 * i + outPadLow + k] += in[i] * w[k]. The tap parity must match the output.
    axis.firstTap = ((parity - outPadLow) % 2 + 2) % 2;
    axis.taps = (kernelSize - axis.firstTap + 1) / 2;
    axis.outSize = (outSize - parity + 1) / 2;
    // Output m reads in[m + base - t] for tap t. The stride-1 window starts at base - (taps - 1).
    int64_t base = (parity - outPadLow - axis.firstTap) / 2;
    int64_t padLow = axis.taps - 1 - base;
    int64_t padHigh = axis.outSize + axis.taps - 1 - inSize - padLow;
    axis.padLow = std::max<int64_t>(padLow, 0);
    axis.padHigh = std::max<int64_t>(padHigh, 0);
    axis.cropLow = std::max<int64_t>(-padLow, 0);
    axis.cropHigh = std::max<int64_t>(-padHigh, 0);
    return axis;
}

// Compute the PhaseAxis of parity 0 and 1 of spatial dim `dim` (1 is H, 2 is W). Return an empty
// vector if a phase has no output, no tap, or a window with no input row.
static SmallVector<PhaseAxis, 2> computePhaseAxes(
    RankedTensorType inputType, RankedTensorType resultType, int64_t kernelSize, int64_t outPadLow,
    int64_t dim
) {
    int64_t inSize = inputType.getDimSize(dim);
    SmallVector<PhaseAxis, 2> axes;
    for (int64_t parity = 0; parity < 2; ++parity) {
        PhaseAxis axis =
            computePhaseAxis(inSize, kernelSize, resultType.getDimSize(dim), outPadLow, parity);
        if (axis.outSize < 1 || axis.taps < 1 || axis.cropLow + axis.cropHigh >= inSize)
            return {};
        axes.push_back(axis);
    }
    return axes;
}

// Return the taps firstTap + 2 * t of `weights` on dim `dim`, for t < taps, in reverse order.
// The taps are tosa.slice ops and a tosa.concat, so the weights can be any value.
static Value selectReversedTaps(
    Value weights, int64_t dim, int64_t firstTap, int64_t taps, Location loc,
    PatternRewriter &rewriter
) {
    auto type = cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> sizes(type.getShape());
    sizes[dim] = 1;
    Value sizeShape = tosa::getTosaConstShape(rewriter, loc, sizes);
    SmallVector<Value> slices;
    for (int64_t t = taps - 1; t >= 0; --t) {
        SmallVector<int64_t> start(type.getRank(), 0);
        start[dim] = firstTap + 2 * t;
        slices.push_back(tosa::SliceOp::create(
            rewriter, loc, type.clone(sizes), weights,
            tosa::getTosaConstShape(rewriter, loc, start), sizeShape
        ));
    }
    if (slices.size() == 1)
        return slices.front();
    sizes[dim] = taps;
    return tosa::ConcatOp::create(rewriter, loc, type.clone(sizes), slices, dim);
}

// The smallest number of output channels for which the split is faster than the upstream
// decomposition. Measured on a stride-2 sweep (8x8 to 32x32 inputs, 16 to 128 channels, kernels
// 3x3 and 4x4, int8 and int16). With 16 channels, the four small convs were 4 % to 48 % slower.
// With 64 or more channels, they were 7 % to 85 % faster on one slice (SR200).
constexpr int64_t kMinOutputChannels = 64;

// Return true if the split is faster than the upstream decomposition for these weights. On two
// slices (SL2610), the 4x wide upstream conv divides well over the slices. There, the split was
// faster only for a kernel that is odd on an axis. The upstream decomposition pads such a kernel
// to an even size, so its conv has more taps than the four phase convs together.
static bool isSplitFaster(RankedTensorType weightType) {
    if (weightType.getDimSize(0) < kMinOutputChannels)
        return false;
    bool oddKernel = weightType.getDimSize(1) % 2 == 1 || weightType.getDimSize(2) % 2 == 1;
    return TorqHw::get().getSliceCount() == 1 || oddKernel;
}

// Return the rescale that is the only user of `srcOp`, or a null op.
static tosa::RescaleOp getOnlyRescaleUser(tosa::TransposeConv2DOp srcOp) {
    if (!srcOp->hasOneUse())
        return nullptr;
    return dyn_cast<tosa::RescaleOp>(*srcOp->getUsers().begin());
}

// Apply a copy of `rescaleOp` to `conv`.
static Value copyRescale(tosa::RescaleOp rescaleOp, Value conv, PatternRewriter &rewriter) {
    auto convType = cast<RankedTensorType>(conv.getType());
    Type elementType = cast<RankedTensorType>(rescaleOp.getType()).getElementType();
    return tosa::RescaleOp::create(
        rewriter, rescaleOp.getLoc(), convType.clone(elementType), conv, rescaleOp.getMultiplier(),
        rescaleOp.getShift(), rescaleOp.getInputZp(), rescaleOp.getOutputZp(),
        rescaleOp.getScale32Attr(), rescaleOp.getRoundingModeAttr(), rescaleOp.getPerChannelAttr(),
        rescaleOp.getInputUnsignedAttr(), rescaleOp.getOutputUnsignedAttr()
    );
}

// Return the NHWC input of `srcOp` without the crop rows and columns of the phase.
static Value cropInput(
    tosa::TransposeConv2DOp srcOp, const PhaseAxis &axisH, const PhaseAxis &axisW,
    PatternRewriter &rewriter
) {
    Value input = srcOp.getInput();
    if (!axisH.cropLow && !axisH.cropHigh && !axisW.cropLow && !axisW.cropHigh)
        return input;
    MLIRContext *ctx = rewriter.getContext();
    ArrayRef<int64_t> inShape = cast<RankedTensorType>(input.getType()).getShape();
    SmallVector<int64_t> sizes = {
        inShape[0], inShape[1] - axisH.cropLow - axisH.cropHigh,
        inShape[2] - axisW.cropLow - axisW.cropHigh, inShape[3]
    };
    return tensor::ExtractSliceOp::create(
        rewriter, srcOp.getLoc(), input,
        getAsIndexOpFoldResult(ctx, {0, axisH.cropLow, axisW.cropLow, 0}),
        getAsIndexOpFoldResult(ctx, sizes), getAsIndexOpFoldResult(ctx, {1, 1, 1, 1})
    );
}

// Build the stride-1 conv of one phase of `srcOp`, with the weight taps of the phase. Then apply
// a copy of `rescaleOp` if it is set.
static Value buildPhaseConv(
    tosa::TransposeConv2DOp srcOp, tosa::RescaleOp rescaleOp, const PhaseAxis &axisH,
    const PhaseAxis &axisW, PatternRewriter &rewriter
) {
    Location loc = srcOp.getLoc();
    Value weights =
        selectReversedTaps(srcOp.getWeight(), 1, axisH.firstTap, axisH.taps, loc, rewriter);
    weights = selectReversedTaps(weights, 2, axisW.firstTap, axisW.taps, loc, rewriter);

    auto srcType = cast<RankedTensorType>(srcOp.getType());
    SmallVector<int64_t> phaseShape = {
        srcType.getDimSize(0), axisH.outSize, axisW.outSize, srcType.getDimSize(3)
    };
    Value conv = tosa::Conv2DOp::create(
        rewriter, loc, srcType.clone(phaseShape), cropInput(srcOp, axisH, axisW, rewriter), weights,
        srcOp.getBias(), srcOp.getInputZp(), srcOp.getWeightZp(),
        rewriter.getDenseI64ArrayAttr({axisH.padLow, axisH.padHigh, axisW.padLow, axisW.padHigh}),
        rewriter.getDenseI64ArrayAttr({1, 1}), rewriter.getDenseI64ArrayAttr({1, 1}),
        srcOp.getAccTypeAttr(), srcOp.getLocalBoundAttr()
    );
    return rescaleOp ? copyRescale(rescaleOp, conv, rewriter) : conv;
}

// Split a stride-2 transpose conv into four stride-1 convs, one for each output parity (phase).
// Each phase result goes into the NHWC output with a stride-2 insert_slice. The upstream
// decomposition makes one conv with 4x the output channels, then a depth-to-space and a crop.
// These only move data through XRAM, and they set the rows of a positive out_pad to 0, not to the
// bias. So the pattern applies to every positive out_pad, and otherwise only if isSplitFaster.
// When a rescale is the only user of the transpose conv, each phase gets a copy of it, so that the
// rescale stays fused.
struct TransposeConv2DStride2PhaseSplit : public OpRewritePattern<tosa::TransposeConv2DOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tosa::TransposeConv2DOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp.getStride() != ArrayRef<int64_t>{2, 2})
            return rewriter.notifyMatchFailure(srcOp, "expected stride 2x2");
        auto inputType = dyn_cast<RankedTensorType>(srcOp.getInput().getType());
        auto weightType = dyn_cast<RankedTensorType>(srcOp.getWeight().getType());
        auto resultType = dyn_cast<RankedTensorType>(srcOp.getType());
        if (!inputType || !weightType || !resultType || !inputType.hasStaticShape() ||
            !weightType.hasStaticShape() || !resultType.hasStaticShape())
            return rewriter.notifyMatchFailure(srcOp, "expected static shapes");
        ArrayRef<int64_t> outPad = srcOp.getOutPad();
        // A positive out_pad adds output rows or columns. The upstream decomposition sets them to
        // 0, not to the bias.
        bool positiveOutPad = llvm::any_of(outPad, [](int64_t pad) { return pad > 0; });
        if (!positiveOutPad && !isSplitFaster(weightType))
            return rewriter.notifyMatchFailure(srcOp, "the upstream decomposition is faster");
        SmallVector<PhaseAxis, 2> axesH =
            computePhaseAxes(inputType, resultType, weightType.getDimSize(1), outPad[0], 1);
        SmallVector<PhaseAxis, 2> axesW =
            computePhaseAxes(inputType, resultType, weightType.getDimSize(2), outPad[2], 2);
        if (axesH.empty() || axesW.empty())
            return rewriter.notifyMatchFailure(srcOp, "a phase has no output, tap or input");

        // The operands of the rescale can be defined after the transpose conv.
        tosa::RescaleOp rescaleOp = getOnlyRescaleUser(srcOp);
        if (rescaleOp)
            rewriter.setInsertionPoint(rescaleOp);
        auto finalType = rescaleOp ? cast<RankedTensorType>(rescaleOp.getType()) : resultType;
        Location loc = srcOp.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        Value result = tensor::EmptyOp::create(
            rewriter, loc, finalType.getShape(), finalType.getElementType()
        );
        for (int64_t parityH = 0; parityH < 2; ++parityH) {
            for (int64_t parityW = 0; parityW < 2; ++parityW) {
                const PhaseAxis &axisH = axesH[parityH];
                const PhaseAxis &axisW = axesW[parityW];
                Value phase = buildPhaseConv(srcOp, rescaleOp, axisH, axisW, rewriter);
                SmallVector<int64_t> sizes = {
                    finalType.getDimSize(0), axisH.outSize, axisW.outSize, finalType.getDimSize(3)
                };
                result = tensor::InsertSliceOp::create(
                    rewriter, loc, phase, result,
                    getAsIndexOpFoldResult(ctx, {0, parityH, parityW, 0}),
                    getAsIndexOpFoldResult(ctx, sizes), getAsIndexOpFoldResult(ctx, {1, 2, 2, 1})
                );
            }
        }

        if (rescaleOp) {
            rewriter.replaceOp(rescaleOp, result);
            rewriter.eraseOp(srcOp);
        }
        else {
            rewriter.replaceOp(srcOp, result);
        }
        return success();
    }
};

} // namespace

void populateTransposeConvStride2PhaseSplitPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.insert<TransposeConv2DStride2PhaseSplit>(context);
}

} // namespace mlir::syna::torq
