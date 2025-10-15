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
ReduceMeanPattern::transform(torq_hl::ReduceMeanOp op, PatternRewriter &rewriter) const {
    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    uint32_t input_channel, height, width;
    input_channel = input_shape[1];
    height = input_shape[2];
    width = input_shape[3];

    const uint32_t frame_size = height * width;
    assert(frame_size > 0 && "frame size must be greater than 0");

    Slice slice;

    const auto dataType = DType::bf16;
    const auto weightType = DType::bf16;
    const uint32_t blockSize = slice.alu.iWidth(dataType, weightType);
    const uint32_t blockCount = div_ceil(width, blockSize);

    auto input_strides = getEncodedStridesElements(input_type);
    uint32_t row_stride = input_strides[2] * sizeofType(DType::bf16);

    LData input({input_channel, {height, row_stride}, blockCount, blockSize}, DType::bf16);
    LData weight({1}, weightType);
    LData biasScale({1}, DType::fp32);

    // For bf16, activation unit combines 2 partials into 1 result, so effective output width is
    // act_width/2
    auto input_element_type = input_type.getElementType();
    const DType inputType = getDType(input_element_type);
    const uint32_t actOutputWidth = slice.act.width(inputType, weightType); // 16 / 2 = 8 for bf16

    LData output(
        {input_channel, blockCount, blockSize / actOutputWidth, actOutputWidth}, DType::bf16
    );

    BData bdata = slice.bram.load(biasScale);
    WData wdata = slice.wram.load(weight);
    For(auto c = slice.iterate(input_channel)) {
        For(auto b = slice.iterate(blockCount)) {
            PData pdata;
            // iterate on the width of all channel block
            For(auto u = slice.iterate(height)) {
                IData idata = slice.iram.load(input[c][u][b]);
                pdata = slice.alu.scalarProductAccumulate(idata, wdata);
            }

            For(auto a = slice.iterate(blockSize / actOutputWidth)) {
                QData res = slice.act.rescaleClamp(
                    pdata[a], bdata, op.getShiftFactor(), op.getOutputZp(), op.getOutputMin(),
                    op.getOutputMax()
                );
                slice.store(output[c][b][a], res);
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                                      // Operation to replace
        "reduce_mean",                           // Task name
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
