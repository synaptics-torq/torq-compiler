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

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// This is an example of an independent table lookup operation.
// - The table contains 512 values for lookup.
// - Input is an int16 9.7 fixed-point value, and the output is an interpolated int32 result.
// - The upper signed 9 bits of the input are used to look up the index in the range [0, 511], while
//   the lower 7 bits represent fractional values and are used to calculate the slope during lookup.
// - Since this is an independent operation, the ALU will operate in bypass mode, and the ACT will
//   function in its default mode.
// - ACT can output only 2 lookup values at a time.
template <>
LogicalResult TablePattern::transform(torq_hl::TableOp op, PatternRewriter &rewriter) const {
    Slice slice("table");

    struct In : Vectorized {
        enum { NonDenseDims };
    };

    auto ctx = op.getContext();
    LData input(op.getInput());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());

    // HW ACT LUT bytes are fixed in the slice descriptor; there is no runtime table buffer.
    // ExtractPattern must have set static_table when creating torq_hl.table; runtime LUTs
    // must be torq_hl.gather instead.
    DenseI32ArrayAttr staticTable = op.getStaticTableAttr();
    if (!staticTable) {
        return rewriter.notifyMatchFailure(
            op, "torq_hl.table missing static_table (expected set at Linalg lowering; "
                "non-const LUTs must be torq_hl.gather)"
        );
    }
    ArrayRef<int32_t> lut = staticTable.asArrayRef();
    slice.act.setLUT(lut);

    input.fuse(std::min(input.denseDims(), output.denseDims()))
        .vectorize(HwInfo::table_lookup_count);

    BData bdata = slice.bram.load(biasScale);
    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto i = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[ndd][i]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.rescaleClamp(
                pdata, bdata, 0, 0, minVal(DType::int32), maxVal(DType::int32)
            );
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, slice.name(), op.getInput(), ValueRange{}, op.getScaleBias(), op.getInit(),
        ValueRange{}, slice.getCfgAttr(ctx), slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
