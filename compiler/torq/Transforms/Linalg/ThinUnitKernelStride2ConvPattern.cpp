// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Thins the kernel-1 axes of the stride (2, 2) convs that the stride-2 NSS kernels do not run
// correctly (synaptics-torq/torq-compiler-dev#2890).
//
// Output o of an axis with a kernel of 1 and a stride of 2 reads only element 2 * o of the image.
// Thus the conv gives the same result when it reads every second element of this axis with a
// stride of 1:
//
//   conv(image, kernel Kx1, stride (2, 2)) == conv(image[:, :, :, ::2], kernel Kx1, stride (2, 1))
//
// A thinned conv2d has a stride of (2, 1) or (1, 2). The S2D pattern, which runs in the same pass,
// folds it as it does for the other convs with these strides. A thinned depthwise conv has a
// stride of (2, 1) or (1, 1), which its lowering takes.

#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#include <array>
#include <type_traits>

namespace mlir::syna::torq {

namespace {

// Index of the H axis in an NCHW shape. The W axis follows it.
constexpr int64_t kFirstSpatialDim = 2;

// For the H and the W axis: true when the pattern thins the axis.
using ThinnedAxes = std::array<bool, 2>;

// Returns the axes to thin. The stride-2 kernels give wrong results or do not finish when one
// kernel axis is 1. The depthwise kernel also gives wrong results for a 1x1 kernel. Two cases
// have a correct lowering and stay as they are: a 1x1 conv2d and a depthwise conv with a 1xK
// kernel. A thinned conv2d is correct only after the S2D fold, which takes only convs without
// dilation. Thus a conv2d with a dilation on the axis that is not thinned stays as it is.
ThinnedAxes selectAxesToThin(
    ArrayRef<int64_t> kernel, ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
    bool isDepthwise
) {
    const bool unitH = kernel[0] == 1;
    const bool unitW = kernel[1] == 1;
    if (strides[0] != 2 || strides[1] != 2 || (isDepthwise ? !unitW : unitH == unitW)) {
        return {false, false};
    }
    if (!isDepthwise && dilations[unitH ? 1 : 0] != 1) {
        return {false, false};
    }
    return {unitH, unitW};
}

// Returns the tensor.pad that gives `image` its border, when the same pad on the thinned source
// gives the thinned image: the source has a static shape, the pad value is a constant, and the
// thinned axes have no border. Returns null in the other cases.
tensor::PadOp getPadToKeep(Value image, const ThinnedAxes &thinned) {
    auto padOp = image.getDefiningOp<tensor::PadOp>();
    if (!padOp || !padOp.getSourceType().hasStaticShape() || !padOp.getConstantPaddingValue()) {
        return nullptr;
    }
    for (int64_t axis = 0; axis < 2; ++axis) {
        const int64_t dim = kFirstSpatialDim + axis;
        if (thinned[axis] && (padOp.getStaticLow()[dim] != 0 || padOp.getStaticHigh()[dim] != 0)) {
            return nullptr;
        }
    }
    return padOp;
}

// Returns the shape of `type` with the size that each thinned axis has in `outShape`, the shape
// of the conv result: a thinned axis has one element for each output.
SmallVector<int64_t>
getThinnedShape(RankedTensorType type, const ThinnedAxes &thinned, ArrayRef<int64_t> outShape) {
    SmallVector<int64_t> shape(type.getShape());
    for (int64_t axis = 0; axis < 2; ++axis) {
        if (thinned[axis]) {
            shape[kFirstSpatialDim + axis] = outShape[kFirstSpatialDim + axis];
        }
    }
    return shape;
}

// Creates the slice of `source` that has every second element of the thinned axes.
Value createThinnedSlice(
    PatternRewriter &rewriter, Location loc, Value source, const ThinnedAxes &thinned,
    ArrayRef<int64_t> outShape
) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    SmallVector<int64_t> steps(sourceType.getRank(), 1);
    for (int64_t axis = 0; axis < 2; ++axis) {
        if (thinned[axis]) {
            steps[kFirstSpatialDim + axis] = 2;
        }
    }
    MLIRContext *ctx = rewriter.getContext();
    return tensor::ExtractSliceOp::create(
        rewriter, loc, source, SmallVector<OpFoldResult>(steps.size(), rewriter.getIndexAttr(0)),
        getAsIndexOpFoldResult(ctx, getThinnedShape(sourceType, thinned, outShape)),
        getAsIndexOpFoldResult(ctx, steps)
    );
}

// Creates the thinned image of the conv. A border from a tensor.pad stays a tensor.pad of the
// thinned source, so that the conv lowering can fold it into the kernel as before.
Value createThinnedImage(
    PatternRewriter &rewriter, Location loc, Value image, const ThinnedAxes &thinned,
    ArrayRef<int64_t> outShape
) {
    tensor::PadOp padOp = getPadToKeep(image, thinned);
    if (!padOp) {
        return createThinnedSlice(rewriter, loc, image, thinned, outShape);
    }
    Value slice = createThinnedSlice(rewriter, loc, padOp.getSource(), thinned, outShape);
    // A pad without a border adds nothing. The S2D cost gate would see it as a border in memory.
    if (padOp.hasZeroLowPad() && padOp.hasZeroHighPad() && !padOp.getNofold()) {
        return slice;
    }
    auto imageType = cast<RankedTensorType>(image.getType());
    return tensor::PadOp::create(
        rewriter, padOp.getLoc(), imageType.clone(getThinnedShape(imageType, thinned, outShape)),
        slice, padOp.getMixedLowPad(), padOp.getMixedHighPad(), padOp.getConstantPaddingValue(),
        padOp.getNofold()
    );
}

// The S2D pattern of the same pass also matches a stride (2, 2) conv2d with a kernel above the HW
// limit, for example 9x1. If it runs first, it also folds the kernel-1 axis with a kernel that has
// a zero pad to 2, which doubles the channels of the conv. Thus this pattern has a higher benefit:
// the thinned conv has stride (2, 1), and the S2D pattern folds only the H axis.
constexpr unsigned kThinPatternBenefit = 2;

template <typename ConvOpT>
class ThinUnitKernelStride2ConvPattern : public OpRewritePattern<ConvOpT> {
  public:
    explicit ThinUnitKernelStride2ConvPattern(MLIRContext *ctx)
        : OpRewritePattern<ConvOpT>(ctx, kThinPatternBenefit) {}

    LogicalResult matchAndRewrite(ConvOpT op, PatternRewriter &rewriter) const override {
        Value image = op.getInputs()[0];
        Value filter = op.getInputs()[1];
        Value init = op.getOutputs()[0];
        auto imageType = cast<RankedTensorType>(image.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto initType = cast<RankedTensorType>(init.getType());
        if (!imageType.hasStaticShape() || !filterType.hasStaticShape() ||
            !initType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(op, "shapes are not static");
        }

        // The kernel is the last two dims of the conv2d [F, C, KH, KW] and of the depthwise
        // [C, KH, KW] filters.
        constexpr bool isDepthwise = std::is_same_v<ConvOpT, linalg::DepthwiseConv2DNchwChwOp>;
        SmallVector<int64_t> strides(op.getStrides().template getValues<int64_t>());
        SmallVector<int64_t> dilations(op.getDilations().template getValues<int64_t>());
        const ThinnedAxes thinned =
            selectAxesToThin(filterType.getShape().take_back(2), strides, dilations, isDepthwise);
        if (!thinned[0] && !thinned[1]) {
            return rewriter.notifyMatchFailure(op, "no kernel-1 axis to thin");
        }

        // A kernel of 1 has no dilation.
        for (int64_t axis = 0; axis < 2; ++axis) {
            if (thinned[axis]) {
                strides[axis] = 1;
                dilations[axis] = 1;
            }
        }

        Value thinnedImage =
            createThinnedImage(rewriter, op.getLoc(), image, thinned, initType.getShape());
        rewriter.replaceOpWithNewOp<ConvOpT>(
            op, init.getType(), ValueRange{thinnedImage, filter}, ValueRange{init},
            rewriter.getI64TensorAttr(strides), rewriter.getI64TensorAttr(dilations)
        );
        return success();
    }
};

} // namespace

void populateThinUnitKernelStride2ConvPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<ThinUnitKernelStride2ConvPattern<linalg::Conv2DNchwFchwOp>>(ctx);
    patterns.add<ThinUnitKernelStride2ConvPattern<linalg::DepthwiseConv2DNchwChwOp>>(ctx);
}

} // namespace mlir::syna::torq
