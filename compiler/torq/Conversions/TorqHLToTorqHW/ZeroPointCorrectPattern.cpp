// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Both kernels walk a dimension whose neighbours are live -- the next row of the result, the
// next entry of the bias record -- so a vector must not run past the end of one. Keep the width
// power-of-two as well: ACT integer bias records support only 1, 2, or 4 pairs, and the bias-table
// kernel's alternating {bias, scale} records must restart on an even lane boundary.
static int fittingVectorSize(int limit, int64_t count) {
    int size = std::min<int64_t>(limit, count);
    while (size > 1 && ((size & (size - 1)) != 0 || count % size != 0))
        --size;
    return size;
}

// One MAC of the interleaved constant against the factor, with a two-entry ACT bias supplying
// the scale slots. The record is written densely because a strided write of the bias slots
// alone leaves the DEQW without a dense inner dimension.
template <>
LogicalResult ZeroPointBiasTablePattern::transform(
    torq_hl::ZeroPointBiasTableOp op, PatternRewriter &rewriter
) const {
    LData input(op.getInput());
    LData aZp(op.getWeights());
    LData biasScale(op.getScaleBias());
    LData table(op.getInit());

    if (input.dims().size() != 1 || table.dims().size() != 1 || table.dim(0) != input.dim(0)) {
        return op.emitError() << "zero_point_bias_table expects [2*N] input and init";
    }
    // i8 is signless, so an unsigned factor has to say so or the MAC signs it.
    if (op.getWeightsUnsigned())
        aZp.setElementType(toUnsigned(aZp.elementType()));

    Slice slice("zp-bias-table");
    // Per-item is what makes the {bias, scale} pattern repeat every two results, and for an
    // i32 stream against a one-byte weight it costs no ACT width. The record can be narrower
    // than one vector: the row correction's is two entries at M == 1.
    const int vectorSize = fittingVectorSize(
        std::min(
            slice.act.width(input.elementType(), aZp.elementType(), /*biasScalePerItem=*/true),
            slice.alu.iWidth(input.elementType(), aZp.elementType())
        ),
        input.dim(0)
    );
    input.vectorize(vectorSize);
    table.vectorize(vectorSize);
    biasScale.reshapeDim(0, {-1, biasScaleWidth(input.elementType())});

    WData wdata = slice.wram.load(aZp);
    BData bdata = slice.bram.load(biasScale);
    For(auto iv = slice.iterate(input.dim(Vectorized::Vectors))) {
        IData idata = slice.iram.load(input[iv]);
        PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);
        For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
            QData res = slice.act.rescaleClamp(
                pdata[av], bdata, /*shift=*/0, /*zeroPoint=*/0, std::numeric_limits<int32_t>::min(),
                std::numeric_limits<int32_t>::max()
            );
            slice.append(table, res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

// The bias comes from BRAM, a separate NDL stream: a slice task has one data stream and
// `input` needs it, so that is the only way to bring an [N] or [M] operand alongside an
// [M, N] one without materialising it.
//
// The int-to-float cast cannot be folded in. On the board a BRAM bias record combined with
// `ACTMode::I2F` hangs the NPU, reloaded per block or loaded once.
template <>
LogicalResult ZeroPointCorrectPattern::transform(
    torq_hl::ZeroPointCorrectOp op, PatternRewriter &rewriter
) const {
    LData input(op.getInput());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());

    const DType inType = input.elementType();
    if (!isInt(inType) || inType != output.elementType()) {
        return op.emitError() << "zero_point_correct expects matching integer input and result";
    }
    const int64_t n = input.dim(-1);
    if (biasScale.dims().size() != 1) {
        return op.emitError() << "zero_point_correct expects a flat {bias, scale} record";
    }

    // Drop a unit batch so rows and columns are one loop each whatever the source rank was.
    for (LData *data : {&input, &output})
        while (data->dims().size() > 2 && data->dim(0) == 1)
            data->eraseDim(0);
    if (input.dims().size() != 2 || output.dims().size() != 2) {
        return op.emitError() << "zero_point_correct expects a [M, N] result";
    }
    const bool perRow = op.getPerRow();
    if (biasScale.dim(0) != (perRow ? input.dim(0) : n) * biasScaleWidth(inType)) {
        return op.emitError() << "zero_point_correct scale_bias width does not match the axis";
    }

    struct In : Vectorized {
        enum { NonDenseDims };
    };

    Slice slice("zp-correct");
    // A per-row bias is one record per ACT pass and keeps the full width; a per-column bias is
    // one per result, which narrows the ACT and costs a reload per column block.
    const int vectorSize = fittingVectorSize(
        perRow ? slice.act.width(inType)
               : slice.act.width(inType, DType::none, /*biasScalePerItem=*/true),
        n
    );
    // Not fused with the rows: the record repeats every N, or changes per row.
    input.vectorize(vectorSize);
    biasScale.reshapeDim(0, {-1, perRow ? 1 : vectorSize, biasScaleWidth(inType)});

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[ndd][iv]);
            PData pdata = slice.alu.load(idata);
            BData bdata = slice.bram.load(perRow ? biasScale[ndd][0] : biasScale[iv]);
            QData res = slice.act.rescaleClamp(
                pdata, bdata, /*shift=*/0, /*zeroPoint=*/0, op.getOutputMin(), op.getOutputMax()
            );
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
