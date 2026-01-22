// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-mul"

namespace mlir::syna::torq {

template <>
LogicalResult MulPattern::transform(torq_hl::MulOp op, PatternRewriter &rewriter) const {
    Slice slice("mul");

    using In = Vectorized;
    LData input1(op.getInput1());
    LData input2(op.getInput2());
    LData biasScale(op.getScaleBias());
    LData output(op.getInit());

    const int shift = op.getShift();
    const int outMin = op.getOutputMin();
    const int outMax = op.getOutputMax();

    // Apply implicit broadcasting
    input1.broadcastAs(output);
    input2.broadcastAs(output);

    // Count how many dense inner dimensions we have in input and output
    int inDenseDims = std::min(input1.denseDims(), input2.denseDims());
    int outDenseDims = output.denseDims();
    int denseDims = std::min(inDenseDims, outDenseDims);

    // Fuse and vectorize inputs
    int vectorSize = slice.act.width(input1.elementType(), input2.elementType());
    input1.fuse(denseDims).vectorize(vectorSize);
    input2.fuse(denseDims).vectorize(vectorSize);
    output.fuse(denseDims);

    BData bdata = slice.bram.load(biasScale);
    For(auto i = slice.iterate(input1.dims(0, In::Elements))) {
        IData data1 = slice.iram.load(input1[i]);
        WData data2 = slice.wram.load(input2[i]);
        PData pdata = slice.alu.elementwiseProductAccumulate(data1, data2);
        QData res = slice.act.rescaleClamp(pdata, bdata, shift, 0, outMin, outMax);
        slice.append(output, res);
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, slice.name(), op.getInput1(), op.getInput2(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
