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

// Segment H and W dimensions to 2 segments each (4 quadrants) EE, EO, OE, OO
static torq_hw::SliceTaskOp
segmentToQuadrants(torq_hl::SegmentationOp op, PatternRewriter &rewriter) {
    Slice slice("segmentation-hw");
    LData input(op.getInput());
    LData output(op.getInit());
    assert(input.shape().size() == 4 && output.shape().size() == 4);
    struct In : NCHW, Vectorized {};

    // Vectorize the input data vectors based on the activation width
    int vectorSize = slice.act.width(input.elementType());
    input.fuse({In::H, In::W}).vectorize(vectorSize);

    // Reorganize the output to be partitioned in 4 quadrants by index parity in height and width
    output.partitionByIndexParity2D();

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch = slice.iterate(input.dim(In::C))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                IData idata = slice.iram.load(input[batch][ch][iv]);
                PData pdata = slice.alu.load(idata);
                QData res = slice.act.load(pdata);
                slice.append(output[batch][ch], res);
            }
        }
    }

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

// Segment H dimension to the specified number of segments
// E.g.: ensor with lines [a,b,c,d,e,f] and hSegments == 2 is converted to [a,c,e,b,d,f]
static torq_hw::SliceTaskOp
segmentHDim(torq_hl::SegmentationOp op, PatternRewriter &rewriter, int hSegments) {
    Slice slice("segmentation-h");
    LData input(op.getInput());
    LData output(op.getInit());
    assert(input.shape().size() == 4 && output.shape().size() == 4);
    assert(input.dims() == output.dims());
    struct In : NCHW, Vectorized {
        enum { RowsInSegment = H, Segments };
    };

    // Vectorize the input data vectors based on the activation width
    int vectorSize = slice.act.width(input.elementType());
    input.vectorize(vectorSize);

    // Partition the input and output lines according to the desired number of segments
    assert(input.dim(In::H) % hSegments == 0 && "Row count not multiple of hSegments");
    input.reshapeDim(In::H, {-1, hSegments}, false);
    output.reshapeDim(In::H, {hSegments, -1}, false);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch = slice.iterate(input.dim(In::C))) {
            For(auto row = slice.iterate(input.dim(In::RowsInSegment))) {
                For(auto seg = slice.iterate(input.dim(In::Segments))) {
                    For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                        IData idata = slice.iram.load(input[batch][ch][row][seg][iv]);
                        PData pdata = slice.alu.load(idata);
                        QData res = slice.act.load(pdata);
                        slice.append(output[batch][ch][seg][row], res);
                    }
                }
            }
        }
    }

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

template <>
LogicalResult
SegmentationPattern::transform(torq_hl::SegmentationOp op, PatternRewriter &rewriter) const {
    const int hSegments = op.getHSegments();
    const int wSegments = op.getWSegments();

    torq_hw::SliceTaskOp hwOp;
    if (hSegments == 2 && wSegments == 2) {
        hwOp = segmentToQuadrants(op, rewriter);
    }
    else if (wSegments <= 1) {
        hwOp = segmentHDim(op, rewriter, hSegments);
    }
    else {
        return rewriter.notifyMatchFailure(op, "Unsupported segmentation configuration");
    }

    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
