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

    // input
    auto input_type = llvm::dyn_cast<MemRefType>(op.getInput().getType());
    auto input_element_type = input_type.getElementType();
    auto input_element_size = input_element_type.getIntOrFloatBitWidth() / 8;

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_element_type = output_type.getElementType();

    uint32_t weightBytes = 2; // 16b weights
    if (input_element_type.isInteger(32)) {
        weightBytes = 1; // 8b weights
    }

    // alu and act process dbus/wbus two 8/16-bit elements size at a time
    uint32_t data_width = input_element_size * weightBytes;

    auto shift = op.getShiftFactor();
    auto min = op.getOutputMin();
    auto max = op.getOutputMax();
    auto zp = op.getOutputZp();

    const uint32_t frame_size = getEncodedTotalSizeBytes(input_type) / input_element_size;
    uint32_t blockSize = alu_group_width / data_width;
    uint32_t blockCount = div_ceil(frame_size, blockSize);
    uint32_t actWidth = HwInfo::act_width / data_width;
    uint32_t actBlockCount = div_ceil(blockSize, actWidth);

    LData input({blockCount, blockSize}, getDType(input_element_type));
    LData weights({1}, input_element_type.isInteger(32) ? DType::int8 : DType::int16);
    LData biasScale({2}, DType::int32);
    LData output({blockCount, actBlockCount, actWidth}, getDType(output_element_type));

    Slice slice;
    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    For(auto b = slice.iterate(blockCount)) {
        IData idata = slice.iram.load(input[b]);
        PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);

        For(auto a = slice.iterate(actBlockCount)) {
            QData res = slice.act.rescaleClamp(pdata[a], bdata, shift, zp, min, max);
            slice.store(output[b][a], res);
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
