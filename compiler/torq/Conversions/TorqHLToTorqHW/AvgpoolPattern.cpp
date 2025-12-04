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
    auto input_element_type = input_type.getElementType();

    const DType inputDType = getDType(input_element_type);
    const DType outputDType =
        getDType(llvm::cast<MemRefType>(op.getInit().getType()).getElementType());
    assert(inputDType == outputDType && "input and output dataType must be the same");

    const DType weightDType =
        getDType(llvm::cast<MemRefType>(op.getWeights().getType()).getElementType());
    auto biasElementType = llvm::cast<MemRefType>(op.getScaleBias().getType()).getElementType();
    const DType biasDType = getDType(biasElementType);

    // avgpool input is NHWC
    const uint32_t input_channel = input_shape[input_shape.size() - 1];

    uint32_t frame_size = 1;
    for (int i = 1; i < input_shape.size() - 1; i++) {
        frame_size *= input_shape[i];
    }
    assert(frame_size > 0 && "frame size must be greater than 0");

    Slice slice;

    const uint32_t blockSize = slice.alu.iWidth(inputDType, weightDType);

    uint32_t blockCount = div_ceil(input_channel, blockSize);
    uint32_t frameSizeStride = input_channel;

    // input memory read address when we do loop
    LData input({{frame_size, frameSizeStride}, blockCount, blockSize}, inputDType);

    // Same weight used for the entire computation
    LData weights({1}, weightDType);

    // all blocks use the same bias and scale
    LData biasScale({biasElementType.isInteger() ? 2 : 1}, biasDType);

    const uint32_t actOutputWidth = slice.act.width(inputDType, weightDType);
    LData output({blockCount, blockSize / actOutputWidth, actOutputWidth}, inputDType);

    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    For(auto b = slice.iterate(blockCount)) {
        PData pdata;
        // iterate on the frame of all channel block
        For(auto u = slice.iterate(frame_size)) {
            IData idata = slice.iram.load(input[u][b]); // idata = 64 channel frame_size data
            pdata = slice.alu.scalarProductAccumulate(idata, wdata);
        }

        For(auto a = slice.iterate(blockSize / actOutputWidth)) {
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
