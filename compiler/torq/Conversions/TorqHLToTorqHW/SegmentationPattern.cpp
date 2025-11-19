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

template <>
LogicalResult
SegmentationPattern::transform(torq_hl::SegmentationOp op, PatternRewriter &rewriter) const {
    Slice slice;
    LData input(op.getInput());
    LData output(op.getInit());
    assert(input.shape().size() == 4 && output.shape().size() == 4);
    struct In : Vectorized {
        enum { N, C, H, W };
    };

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

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, "segmentation", op.getInput(), op.getWeights(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
