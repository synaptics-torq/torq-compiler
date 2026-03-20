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

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult FMAPattern::transform(torq_hl::FMAOp op, PatternRewriter &rewriter) const {

    auto shift = op.getShiftFactor();
    auto min = op.getOutputMin();
    auto max = op.getOutputMax();
    auto zp = op.getOutputZp();

    LData input(op.getInput());
    LData weights(op.getWeights());
    LData biasScale(op.getScaleBias());
    LData output(op.getInit());

    struct In : Vectorized {
        enum { NonDenseDims };
    };

    Slice slice;
    int vectorSize = slice.act.width(input.elementType(), weights.elementType());
    input.fuse(std::min(input.denseDims(), output.denseDims())).vectorize(vectorSize);

    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    For(auto b = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[b][iv]);
            PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);
            QData res = slice.act.rescaleClamp(pdata, bdata, shift, zp, min, max);
            slice.append(output[b], res);
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, // Operation to replace
        "fma",
        op.getInput(),                           // Input tensor
        op.getWeights(),                         // Weights
        op.getScaleBias(),                       // BiasScale tensor
        op.getInit(),                            // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
