// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Lower torq_hl.im2col to a single slice task that reads sliding input windows via an affine
// DEDR NDL and writes the unfolded [Ow, C*Kw] layout via a DEQW NDL. No gather index table is
// materialized: the read addresses are computed by the NDL strides directly, i.e.
//   out[ow, c*Kw + kw] = input[c, ow*stride + kw*dilation]
template <>
LogicalResult Im2ColPattern::transform(torq_hl::Im2ColOp op, PatternRewriter &rewriter) const {
    auto inputType = llvm::cast<MemRefType>(op.getInput().getType());
    auto outputType = llvm::cast<MemRefType>(op.getInit().getType());
    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    if (inputShape.size() != 2 || outputShape.size() != 2) {
        return op.emitError("im2col expects 2D input [C, W] and 2D init [Ow, C*Kw]");
    }

    const int64_t stride = op.getStride();
    const int64_t dilation = op.getDilation();
    const int64_t kernelWidth = op.getKernelWidth();

    const int64_t channels = inputShape[0];
    const int64_t outputWidth = outputShape[0];

    if (channels * kernelWidth != outputShape[1]) {
        return op.emitError("im2col init second dim must equal C*Kw");
    }

    auto inputStrides = getEncodedStridesElements(inputType);
    auto outputStrides = getEncodedStridesElements(outputType);
    const int64_t inChannelStride = inputStrides[0]; // elements between channels (= W when dense)
    const int64_t inWidthStride = inputStrides[1];   // elements between adjacent samples (= 1)
    const int64_t outRowStride = outputStrides[0];   // elements between output rows (= C*Kw)
    const int64_t outColStride = outputStrides[1];   // elements between columns (= 1)

    DType elementType = getDType(inputType.getElementType());
    // The copy is a pure bypass, so view bf16 as int16 (matches IdentityPattern).
    if (elementType == DType::bf16) {
        elementType = DType::int16;
    }

    Slice slice("im2col");

    // Windowed view of the input: dims [Ow, C, Kw] with affine (possibly overlapping) strides.
    //   element[ow][c][kw] = base + ow*stride + c*W + kw*dilation
    LData input(
        {{outputWidth, stride * inWidthStride},
         {channels, inChannelStride},
         {kernelWidth, dilation * inWidthStride}},
        elementType
    );

    // Dense unfolded output: dims [Ow, C, Kw] -> flattened [Ow, C*Kw].
    LData output(
        {{outputWidth, outRowStride},
         {channels, kernelWidth * outColStride},
         {kernelWidth, outColStride}},
        elementType
    );

    For(auto ow = slice.iterate(outputWidth)) {
        For(auto c = slice.iterate(channels)) {
            // Load the full kw window for this (ow, c) and copy it through the ALU/ACT bypass.
            IData idata = slice.iram.load(input[ow][c]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.load(pdata);
            slice.store(output[ow][c], res);
        }
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op, slice.name(), ValueRange{op.getInput()}, // D: input activations (windowed read)
        ValueRange{},                                // W
        ValueRange{},                                // B
        ValueRange{op.getInit()},                    // Q: unfolded output
        ValueRange{},                                // Symbols
        slice.getCfgAttr(rewriter.getContext()),     // Slice configuration
        slice.getNdls()                              // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
