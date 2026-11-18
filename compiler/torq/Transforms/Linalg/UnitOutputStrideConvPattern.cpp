// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Gives a stride of 1 to the axes of a depthwise conv that have one output
// (synaptics-torq/torq-compiler-dev#1789).
//
// Output 0 of an axis reads the image elements 0, d, ..., (K - 1) * d, for all strides. Thus the
// stride of an axis with one output has no effect. The NSS depthwise lowering does not take all
// strides, for example (1, 2). A 1D depthwise conv on a W of 1 with a "same" pad and a stride of
// 2 has this stride, and it runs on the NSS as a stride (1, 1) conv.
//
// The pattern also changes the convs that the lowering takes with their strides, for example a
// 3x3 kernel with a stride of (2, 2) on a 3x3 image. They give the same result with a stride of
// (1, 1).
//
// A stride-1 axis has one output only when the image ends at the last element that output 0
// reads. The image can have more elements, for example a W of 4 for a 1x3 kernel with a stride
// of (1, 2). The pattern removes them from the image.

#include "ConvStridePatternUtils.h"
#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <array>

namespace mlir::syna::torq {

namespace {

// For the H and the W axis: the number of elements at the end of the image that the conv does
// not read.
using UnreadElements = std::array<int64_t, 2>;

// Returns the unread elements of the image when each axis with a stride above 1 has one output.
// Then the conv with a stride of (1, 1) on the image without these elements has the same result
// and the same output shape. Returns std::nullopt in the other cases, which include a stride of
// (1, 1). The pattern makes only a stride (1, 1) conv, because the depthwise lowering does not
// take all the strides between (1, 1) and the original one.
std::optional<UnreadElements> getUnreadElementsOfUnitOutputAxes(const StaticConvInfo &conv) {
    UnreadElements unread{0, 0};
    bool hasStride = false;
    for (int64_t axis = 0; axis < 2; ++axis) {
        if (conv.strides[axis] == 1) {
            continue;
        }
        const int64_t dim = kFirstSpatialDim + axis;
        if (conv.initType.getDimSize(dim) != 1) {
            return std::nullopt;
        }
        const int64_t window = (conv.kernel[axis] - 1) * conv.dilations[axis] + 1;
        unread[axis] = conv.imageType.getDimSize(dim) - window;
        hasStride = true;
    }
    if (!hasStride) {
        return std::nullopt;
    }
    return unread;
}

// Returns the shape of `type` without the unread elements.
SmallVector<int64_t> getShapeWithoutUnread(RankedTensorType type, const UnreadElements &unread) {
    SmallVector<int64_t> shape(type.getShape());
    for (int64_t axis = 0; axis < 2; ++axis) {
        shape[kFirstSpatialDim + axis] -= unread[axis];
    }
    return shape;
}

// Creates the slice of `source` without its unread elements.
Value createSliceWithoutUnread(
    PatternRewriter &rewriter, Location loc, Value source, const UnreadElements &unread
) {
    auto sourceType = cast<RankedTensorType>(source.getType());
    return createLeadingSlice(
        rewriter, loc, source, getShapeWithoutUnread(sourceType, unread),
        SmallVector<int64_t>(sourceType.getRank(), 1)
    );
}

// Returns the tensor.pad that gives `image` its border, when a smaller pad of the same source
// can give the image without its unread elements: the source has a static shape, the pad value
// is a constant, and the high pad is static. Returns null in the other cases.
tensor::PadOp getPadToShrink(Value image) {
    auto padOp = image.getDefiningOp<tensor::PadOp>();
    if (!padOp || !padOp.getSourceType().hasStaticShape() || !padOp.getConstantPaddingValue() ||
        !ShapedType::isStaticShape(padOp.getStaticHigh())) {
        return nullptr;
    }
    return padOp;
}

// Creates the result of `padOp` without its unread elements. The high border loses the unread
// elements first, and the source loses the remaining ones. The border stays a tensor.pad, so
// that the conv lowering can fold it into the kernel as before.
Value createPadWithoutUnread(
    PatternRewriter &rewriter, tensor::PadOp padOp, const UnreadElements &unread
) {
    SmallVector<int64_t> high(padOp.getStaticHigh());
    UnreadElements unreadOfSource;
    for (int64_t axis = 0; axis < 2; ++axis) {
        const int64_t dim = kFirstSpatialDim + axis;
        const int64_t unreadOfBorder = std::min(high[dim], unread[axis]);
        high[dim] -= unreadOfBorder;
        unreadOfSource[axis] = unread[axis] - unreadOfBorder;
    }

    Value source = padOp.getSource();
    if (unreadOfSource != UnreadElements{0, 0}) {
        source = createSliceWithoutUnread(rewriter, padOp.getLoc(), source, unreadOfSource);
    }
    // A pad without a border adds nothing.
    if (padOp.hasZeroLowPad() && llvm::all_of(high, [](int64_t size) { return size == 0; }) &&
        !padOp.getNofold()) {
        return source;
    }
    RankedTensorType imageType = padOp.getResultType();
    return tensor::PadOp::create(
        rewriter, padOp.getLoc(), imageType.clone(getShapeWithoutUnread(imageType, unread)), source,
        padOp.getMixedLowPad(), getAsIndexOpFoldResult(rewriter.getContext(), high),
        padOp.getConstantPaddingValue(), padOp.getNofold()
    );
}

// True when output 0 of a strided axis reads only the low border of the tensor.pad that gives
// `image` its border, for example with a low border of 4 for a 1x3 kernel. The image without
// its unread elements would then be a slice of the padded image on this axis, which the conv
// lowering does not fold.
bool readsOnlyLowBorder(Value image, const StaticConvInfo &conv) {
    auto padOp = image.getDefiningOp<tensor::PadOp>();
    if (!padOp) {
        return false;
    }
    ArrayRef<int64_t> low = padOp.getStaticLow();
    for (int64_t axis = 0; axis < 2; ++axis) {
        const int64_t dim = kFirstSpatialDim + axis;
        const int64_t window = (conv.kernel[axis] - 1) * conv.dilations[axis] + 1;
        if (conv.strides[axis] != 1 && !ShapedType::isDynamic(low[dim]) && low[dim] >= window) {
            return true;
        }
    }
    return false;
}

// Creates the image of the conv without its unread elements.
Value createImageWithoutUnread(
    PatternRewriter &rewriter, Location loc, Value image, const UnreadElements &unread
) {
    if (unread == UnreadElements{0, 0}) {
        return image;
    }
    if (tensor::PadOp padOp = getPadToShrink(image)) {
        return createPadWithoutUnread(rewriter, padOp, unread);
    }
    return createSliceWithoutUnread(rewriter, loc, image, unread);
}

// The thinning patterns of the same pass also match some of these convs. This pattern has a
// higher benefit, because a stride (1, 1) conv does not need the thinned slice of the image.
constexpr unsigned kUnitOutputStrideBenefit = 3;

class UnitOutputStrideDepthwisePattern : public OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp> {
  public:
    explicit UnitOutputStrideDepthwisePattern(MLIRContext *ctx)
        : OpRewritePattern(ctx, kUnitOutputStrideBenefit) {}

    LogicalResult
    matchAndRewrite(linalg::DepthwiseConv2DNchwChwOp op, PatternRewriter &rewriter) const override {
        std::optional<StaticConvInfo> conv = getStaticConvInfo(op);
        if (!conv) {
            return rewriter.notifyMatchFailure(op, "shapes are not static");
        }
        std::optional<UnreadElements> unread = getUnreadElementsOfUnitOutputAxes(*conv);
        if (!unread) {
            return rewriter.notifyMatchFailure(op, "the strides are (1, 1) or have an effect");
        }
        if (readsOnlyLowBorder(op.getInputs()[0], *conv)) {
            return rewriter.notifyMatchFailure(op, "output 0 reads only the low border");
        }

        // A new op, not a change of the strides in place: a linalg named op keeps its indexing
        // maps in an attribute after the first read, and these maps have the original strides.
        Value image = createImageWithoutUnread(rewriter, op.getLoc(), op.getInputs()[0], *unread);
        rewriter.replaceOpWithNewOp<linalg::DepthwiseConv2DNchwChwOp>(
            op, conv->initType, ValueRange{image, op.getInputs()[1]}, op.getOutputs(),
            rewriter.getI64TensorAttr({1, 1}), op.getDilations()
        );
        return success();
    }
};

} // namespace

void populateUnitOutputStrideConvPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<UnitOutputStrideDepthwisePattern>(ctx);
}

} // namespace mlir::syna::torq
