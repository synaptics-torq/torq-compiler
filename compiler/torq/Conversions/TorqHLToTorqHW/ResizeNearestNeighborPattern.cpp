
// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl-resizenn"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

static void resizeNearestNeighborNHWC(LData &input, LData &output, Slice &slice) {
    struct In : NHWC, Vectorized {};
    int vectorSize = slice.act.width(input.elementType());
    input.vectorize(vectorSize);
    auto scaleUp = 2;
    output.reshapeDim(In::H, {-1, scaleUp});
    output.reshapeDim(In::W, {-1, scaleUp});
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto h = slice.iterate(input.dim(In::H))) {
            For(auto hScale = slice.iterate(scaleUp)) {
                For(auto w = slice.iterate(input.dim(In::W))) {
                    For(auto wScale = slice.iterate(scaleUp)) {
                        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                            IData idata = slice.iram.load(input[batch][h][w][iv]);
                            PData pdata = slice.alu.load(idata);
                            QData res = slice.act.load(pdata);
                            slice.append(output[batch][h][hScale][w][wScale], res);
                        }
                    }
                }
            }
        }
    }
}

static void resizeNearestNeighborNCHW(LData &input, LData &output, Slice &slice) {
    struct In : NCHW, Vectorized {};
    auto scaleUp = 2;
    int vectorSize = slice.act.width(input.elementType()) / scaleUp;
    input.vectorize(vectorSize);
    output.reshapeDim(NCHW::H, {-1, scaleUp});
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto c = slice.iterate(input.dim(In::C))) {
            For(auto h = slice.iterate(input.dim(In::H))) {
                For(auto hScale = slice.iterate(scaleUp)) {
                    For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                        IData idata = slice.iram.load(input[batch][c][h][iv]).repeat(scaleUp);
                        PData pdata = slice.alu.load(idata);
                        QData res = slice.act.load(pdata);
                        slice.append(output[batch][c][h][hScale], res);
                    }
                }
            }
        }
    }
}

template <>
LogicalResult ResizeNearestNeighborPattern::transform(
    torq_hl::ResizeNearestNeighborOp op, PatternRewriter &rewriter
) const {

    LData input(op.getInput());
    LData output(op.getInit());
    assert(input.shape().size() == 4 && output.shape().size() == 4);

    // check the last dim are same
    bool isNHWC = (input.shape().back().count == output.shape().back().count);
    Slice slice("ResizeNearestNeighbor");
    // All cases will fall to NCHW format
    if (isNHWC) {
        resizeNearestNeighborNHWC(input, output, slice);
    }
    else {
        resizeNearestNeighborNCHW(input, output, slice);
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,
        slice.name(),                            // Operation to replace
        ValueRange{op.getInput()},               // Input tensor
        ValueRange{},                            // Weights
        ValueRange{},                            // BiasScale tensor,
        ValueRange{op.getInit()},                // Output tensor initializer
        ValueRange{},                            // Symbols
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
