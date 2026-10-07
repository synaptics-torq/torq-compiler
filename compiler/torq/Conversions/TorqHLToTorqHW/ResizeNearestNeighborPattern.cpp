
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

static void
resizeNearestNeighborNHWC(LData &input, LData &output, Slice &slice, int scaleH, int scaleW) {
    struct In : NHWC, Vectorized {};
    int vectorSize = slice.act.width(input.elementType());
    input.vectorize(vectorSize);
    output.reshapeDim(In::H, {-1, scaleH});
    output.reshapeDim(In::W, {-1, scaleW});
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto h = slice.iterate(input.dim(In::H))) {
            For(auto hScale = slice.iterate(scaleH)) {
                For(auto w = slice.iterate(input.dim(In::W))) {
                    For(auto wScale = slice.iterate(scaleW)) {
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

static void
resizeNearestNeighborNCHW(LData &input, LData &output, Slice &slice, int scaleH, int scaleW) {
    struct In : NCHW, Vectorized {};
    bool canVectorRepeat =
        ((scaleW == 1 || scaleW == 2 || scaleW == 4) && input.elementType() == DType::int8);
    int vectorSize = canVectorRepeat ? slice.act.width(input.elementType()) / scaleW : 1;
    input.vectorize(vectorSize);
    if (!canVectorRepeat) {
        output.reshapeDim(NCHW::W, {-1, scaleW});
    }
    output.reshapeDim(NCHW::H, {-1, scaleH});
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto c = slice.iterate(input.dim(In::C))) {
            For(auto h = slice.iterate(input.dim(In::H))) {
                For(auto hScale = slice.iterate(scaleH)) {
                    For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                        if (canVectorRepeat) {
                            IData idata = slice.iram.load(input[batch][c][h][iv]).repeat(scaleW);
                            PData pdata = slice.alu.load(idata);
                            QData res = slice.act.load(pdata);
                            slice.append(output[batch][c][h][hScale], res);
                        }
                        else {
                            For(auto wScale = slice.iterate(scaleW)) {
                                IData idata = slice.iram.load(input[batch][c][h][iv]);
                                PData pdata = slice.alu.load(idata);
                                QData res = slice.act.load(pdata);
                                slice.append(output[batch][c][h][hScale][iv][wScale], res);
                            }
                        }
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
    int scaleH = op.getScaleH();
    int scaleW = op.getScaleW();
    assert(scaleH > 0 && scaleW > 0 && "Expected positive resize scale factors");

    // check the last dim are same
    bool isNHWC = (input.shape().back().count == output.shape().back().count);
    LLVM_DEBUG({
        llvm::dbgs() << "ResizeNearestNeighbor: scaleH=" << scaleH << ", scaleW=" << scaleW
                     << ", layout=" << (isNHWC ? "NHWC" : "NCHW") << "\n";
        llvm::dbgs() << "  input=" << input << "\n";
        llvm::dbgs() << "  output=" << output << "\n";
    });

    Slice slice("ResizeNearestNeighbor");
    // All cases will fall to NCHW format
    if (isNHWC) {
        resizeNearestNeighborNHWC(input, output, slice, scaleH, scaleW);
    }
    else {
        resizeNearestNeighborNCHW(input, output, slice, scaleH, scaleW);
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
