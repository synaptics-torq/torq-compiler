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

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

// Each index selects the row of a table written with the input, and the same indices are used for
// all the tables (the output is updated in place). The tables are the last dimension of the output,
// as rows of one element:
//   output[..., indices[n]] = input[..., n]
// input [tables..., W] and output [tables..., K] with the same tables, and W indices: for example
// input [1, 4, 8], indices [8], output [1, 4, 256] (a TOSA scatter in NCH layout). The ACT applies
// the identity bias/scale of the op, which the BRAM data engine reads in turn with the indices.
template <>
LogicalResult ScatterPattern::transform(torq_hl::ScatterOp op, PatternRewriter &rewriter) const {
    LData input(op.getInput());
    LData output(op.getInit());
    LData scaleBias(op.getScaleBias());
    if (input.shape().size() < 2) {
        return op.emitError("Scatter input must have rank >= 2");
    }

    LData indices(op.getIndices());
    if (indices.denseDims() < indices.shape().size()) {
        return op.emitError("Scatter indices must be dense");
    }
    indices.fuse(indices.shape().size()).reshapeDim(-1, {-1, 1});
    if (indices.dim(0) != input.dim(-1)) {
        return op.emitError("Scatter indices must be the same for all the tables");
    }

    // Floats are copied as integers of the same size
    if (isFloat(input.elementType())) {
        const DType intType = sizeofType(input.elementType()) == 2 ? DType::int16 : DType::int32;
        input.bitCast(intType);
        output.bitCast(intType);
    }
    // Tables of rows of one element
    input.reshapeDim(-1, {-1, 1});
    output.reshapeDim(-1, {-1, 1});

    Slice slice("scatter");
    For(auto t = slice.iterate(input.dims(0, -2))) {
        For(auto n = slice.iterate(indices.dim(0))) {
            IData idata = slice.iram.load(input[t][n]);
            BData bdata = slice.bram.load(scaleBias);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.rescaleClamp(
                pdata, bdata, 0, 0, minVal(DType::int32), maxVal(DType::int32)
            );
            slice.storeIndexed(output[t], indices[n], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
