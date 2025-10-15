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

namespace mlir::syna::torq {

template <>
LogicalResult
Avgpool2DPattern::transform(torq_hl::AvgPool2DOp op, PatternRewriter &rewriter) const {
    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    // avgpool input is NHWC
    const uint32_t h = input_shape[1];
    const uint32_t w = input_shape[2];
    const uint32_t input_channel = input_shape[3];

    const uint32_t frame_size = h * w;
    assert(frame_size > 0 && "frame size must be greater than 0");

    uint32_t blockSize = 64;
    uint32_t blockCount = input_channel / blockSize;
    uint32_t frameSizeStride = input_shape[3];

    // input memory read address when we do loop
    LData input({{frame_size, frameSizeStride}, blockCount, blockSize}, DType::int8);

    // Same weight used for the entire computation
    LData weights({1}, DType::int8);

    // all blocks use the same bias and scale
    LData biasScale({2}, DType::int32);

    LData output({blockCount, blockSize / HwInfo::act_width, HwInfo::act_width}, DType::int8);

    Slice slice;
    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    For(auto b = slice.iterate(blockCount)) {
        PData pdata;
        // iterate on the frame of all channel block
        For(auto u = slice.iterate(frame_size)) {
            IData idata = slice.iram.load(input[u][b]); // idata = 64 channel frame_size data
            pdata = slice.alu.scalarProductAccumulate(idata, wdata);
        }

        For(auto a = slice.iterate(blockSize / HwInfo::act_width)) {
            QData res = slice.act.rescaleClamp(
                pdata[a], bdata, op.getShiftFactor(), op.getOutputZp(), op.getOutputMin(),
                op.getOutputMax()
            );
            slice.store(output[b][a], res);
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                                      // Operation to replace
        "avgpool2d",                             // Task name
        op.getInput(),                           // Input tensor
        op.getWeights(),                         // Weights
        op.getScaleBias(),                       // BiasScale tensor,
        op.getInit(),                            // Output tensor initia
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
