// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

// For each index of the other dimensions, the index along axis of the largest input value (the
// first one if several are equal):
//   output[outer..., inner...] = argmax over r of input[outer..., r, inner...]
template <>
LogicalResult ArgMaxPattern::transform(torq_hl::ArgMaxOp op, PatternRewriter &rewriter) const {
    LData input(op.getInput());
    LData output(op.getInit());
    const int axis = op.getAxis();
    const int rank = input.shape().size();
    if (axis < 0 || axis >= rank) {
        return op.emitError("ArgMax axis out of range");
    }
    // Fuse the inner dimensions that are dense in the input and in the output (with none, as for an
    // argmax on the last axis, fuse() adds a unit dimension)
    const int innerDims = input.shape().size() - axis - 1;
    const int denseDims = std::min({input.denseDims(), output.denseDims(), innerDims});

    Slice slice("arg_max");
    input.fuse(denseDims).vectorize(slice.alu.iWidth(input.elementType()));
    output.fuse(denseDims);
    // The index is in the upper bits of the partials: don't round it up
    slice.act.setRoundingMode(Act::Rounding::off);

    For(auto t = slice.iterate(input.dims(0, axis))) {
        For(auto i = slice.iterate(input.dims(axis + 1, Vectorized::Vectors))) {
            For(auto v = slice.iterate(input.dim(Vectorized::Vectors))) {
                PData pdata;
                For(auto r = slice.iterate(input.dim(axis))) {
                    IData idata = slice.iram.load(input[t][r][i][v]);
                    pdata = slice.alu.accumulate(idata, Alu::Mode::argMax);
                }
                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.load(pdata[av]);
                    slice.append(output[t][i], res);
                }
            }
        }
    }

    rewriter.replaceOp(op, slice.createOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
