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

#define DEBUG_TYPE "torq-lower-torqhl-gather"

namespace mlir::syna::torq {

// Copy to output[0..count) the rows of a table selected by count indices (1, 2 or 4)
static void copyIndexedRows(
    Slice &slice, const LData &table, const LData &indices, const LData &output, int count
) {
    IData idata = slice.iram.loadIndexed(table, indices);
    // Same as alu.load(), which is limited to the ACT width
    PData pdata = slice.alu.accumulate(idata, Alu::Mode::bitOr);
    // The rows come one after the other in the partials, each one padded as in the IData
    pdata.setShape({{count, idata.dim(-1)}, {1, 1}});
    For(auto r = slice.iterate(count)) {
        QData res = slice.act.clamp(pdata[r], minVal(DType::int32), maxVal(DType::int32));
        slice.store(output[r], res);
    }
}

// Each index selects one row of a table, and the same indices are used for all the tables. The op
// comes in two forms (see ExtractPattern.cpp):
// - Lookups: the tables are the last dimension of the values, each index selects one element of
//   each table:
//     output[..., n] = values[..., indices[n]]
//   values [tables..., K] and output [tables..., W], with W indices: for example values
//   [1, 32, 256], indices [1, 16], output [1, 32, 16]. Rank-2 values [C, K] may come with an
//   output [1, C', W] (C' <= C), from a unit-batch gather collapsed to rank 2.
// - Slice gathers: rank-2 values [R, S] (other than the case above), each index selects a slice of
//   S elements:
//     output[n, ...] = values[i_n, ...]
//   output [index dims..., slice dims...], where the slice dims have S elements: for example values
//   [16, 64], indices [16], output [16, 1, 64]. ExtractPattern gives the element offset i_n * S of
//   each slice instead of i_n. So the kernel sees S tables, one per column s, starting at element s
//   of the values, in which offset i_n * S selects values[i_n, s].
template <>
LogicalResult GatherPattern::transform(torq_hl::GatherOp op, PatternRewriter &rewriter) const {
    LData values(op.getValues());
    LData indices(op.getIndices());
    LData output(op.getInit());
    const int valuesRank = values.shape().size();
    const int indicesRank = indices.shape().size();
    if (valuesRank == 0) {
        return rewriter.notifyMatchFailure(op, "Expected GatherOp values rank >= 1");
    }
    if (indices.denseDims() < indicesRank) {
        return op.emitError("Gather indices must be dense");
    }
    indices.fuse(indicesRank);
    const int indicesCount = indices.dim(0);

    // Lookup with rank-2 values [C, K] and an output [1, C', W]
    const bool unitBatchLookup = valuesRank == 2 && output.shape().size() == 3 &&
                                 indicesRank == 1 && output.dim(-1) == indicesCount &&
                                 output.dim(1) <= values.dim(0);
    const bool sliceGather = valuesRank == 2 && !unitBatchLookup;

    // Floats are copied as integers of the same size
    if (isFloat(values.elementType())) {
        const DType intType = sizeofType(values.elementType()) == 2 ? DType::int16 : DType::int32;
        values.bitCast(intType);
        output.bitCast(intType);
    }

    Slice slice("gather");
    if (sliceGather) {
        const int sliceSize = values.dim(-1);
        // The S tables of rows of one element, table s starting at element s
        values.fuse(2).insertDim(0, {sliceSize, 1}).reshapeDim(-1, {-1, 1});
        // Fuse the slice dims of the output and make them the tables: [S, index dims...]
        int sliceDims = 0;
        for (int64_t size = 1; size < sliceSize;) {
            size *= output.dim(-1 - sliceDims++);
        }
        if (elementCount(output.shape()) != int64_t(sliceSize) * indicesCount ||
            output.denseDims() < sliceDims) {
            return op.emitError("Gather output slices must be dense");
        }
        output.fuse(sliceDims).moveDim(-1, 0).reshapeDim(-1, {-1, 1, 1});
        // Each index as a list of one, with the shape of the index dims of the output
        indices.reshapeDim(0, output.dims(1, -1));

        For(auto t = slice.iterate(values.dim(0))) {
            For(auto n = slice.iterate(output.dims(1, -2))) {
                copyIndexedRows(slice, values[t], indices[n], output[t][n], 1);
            }
        }
    }
    else {
        // A rank-1 lookup is a single table, and a rank-2 one has the unit batch of its output
        if (valuesRank == 1 || unitBatchLookup) {
            values.insertDim(0, {1});
        }
        if (valuesRank == 1) {
            output.insertDim(0, {1});
        }
        // Look up as many indices at a time as fit in 4 bytes of rows (4 int8, 2 int16, 1 int32),
        // the limit of IRam::loadIndexed()
        const int elementSize = sizeofType(values.elementType());
        int groupSize = 4;
        while (groupSize > 1 && (groupSize * elementSize > 4 || indicesCount % groupSize != 0)) {
            groupSize /= 2;
        }
        indices.reshapeDim(0, {-1, groupSize});
        // Tables of rows of one element. The rows dimension of the values is only used to compute
        // the address of the indexed rows.
        values.reshapeDim(-1, {-1, 1});
        output.reshapeDim(-1, {-1, groupSize, 1});

        For(auto t = slice.iterate(output.dims(0, -3))) {
            For(auto g = slice.iterate(indices.dim(0))) {
                copyIndexedRows(slice, values[t], indices[g], output[t][g], groupSize);
            }
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
