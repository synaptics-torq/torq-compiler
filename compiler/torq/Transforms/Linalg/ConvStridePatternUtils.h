// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Helpers of the patterns that change the strides of the NCHW convs in
// DecomposeConvToSpaceToDepthPass: ThinUnitKernelStride2ConvPattern.cpp and
// UnitOutputStrideConvPattern.cpp.

#pragma once

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

namespace mlir::syna::torq {

// Index of the H axis in an NCHW shape. The W axis follows it.
constexpr int64_t kFirstSpatialDim = 2;

// The types and the attributes of a linalg conv2d or depthwise conv in NCHW that has static
// shapes. The kernel, the strides and the dilations have one value for the H axis and one for
// the W axis.
struct StaticConvInfo {
    RankedTensorType imageType;
    RankedTensorType initType;
    // The last two dims of the conv2d [F, C, KH, KW] and of the depthwise [C, KH, KW] filters.
    ArrayRef<int64_t> kernel;
    SmallVector<int64_t> strides;
    SmallVector<int64_t> dilations;
};

// Returns the info of `op`, or std::nullopt when the image, the filter or the init does not have
// a static shape.
template <typename ConvOpT> std::optional<StaticConvInfo> getStaticConvInfo(ConvOpT op) {
    auto imageType = cast<RankedTensorType>(op.getInputs()[0].getType());
    auto filterType = cast<RankedTensorType>(op.getInputs()[1].getType());
    auto initType = cast<RankedTensorType>(op.getOutputs()[0].getType());
    if (!imageType.hasStaticShape() || !filterType.hasStaticShape() || !initType.hasStaticShape()) {
        return std::nullopt;
    }
    return StaticConvInfo{
        imageType, initType, filterType.getShape().take_back(2),
        SmallVector<int64_t>(op.getStrides().template getValues<int64_t>()),
        SmallVector<int64_t>(op.getDilations().template getValues<int64_t>())
    };
}

// Creates the slice of `source` that starts at element 0 of each axis, and has `sizes` elements
// with `steps` between them.
inline Value createLeadingSlice(
    PatternRewriter &rewriter, Location loc, Value source, ArrayRef<int64_t> sizes,
    ArrayRef<int64_t> steps
) {
    MLIRContext *ctx = rewriter.getContext();
    return tensor::ExtractSliceOp::create(
        rewriter, loc, source, SmallVector<OpFoldResult>(sizes.size(), rewriter.getIndexAttr(0)),
        getAsIndexOpFoldResult(ctx, sizes), getAsIndexOpFoldResult(ctx, steps)
    );
}

} // namespace mlir::syna::torq
