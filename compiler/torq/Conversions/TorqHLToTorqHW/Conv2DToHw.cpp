// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

LogicalResult convertToHw(torq_hl::Conv2DOp op, PatternRewriter &rewriter) {
    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    // weight
    auto weight_type = llvm::cast<MemRefType>(op.getWeights().getType());
    auto weight_shape = weight_type.getShape();

    if (input_type.getElementType().isInteger(16) || input_type.getElementType().isBF16() ||
        weight_type.getElementType().isInteger(16) || weight_type.getElementType().isBF16() ||
        weight_shape[K_SIZE_X] != 3 || weight_shape[K_SIZE_Y] != 3 ||
        weight_shape[K_SIZE_X] != weight_shape[K_SIZE_Y]) {
        return failure();
    }

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();
    if (output_shape[1] == 1) {
        return failure();
    }

    auto input_strides = getEncodedStridesElements(input_type);
    auto output_strides = getEncodedStridesElements(output_type);

    // input/output data layout is NCHW
    const uint32_t out_h = output_shape[2];
    const uint32_t out_w = output_shape[3];

    const uint32_t ksize_x = weight_shape[K_SIZE_X];
    const uint32_t ksize_y = weight_shape[K_SIZE_Y];

    uint32_t max_input;
    uint32_t max_channels;
    switch (op.getVectorizationMode()) {
    case torq_hl::VectorizationModeEnum::_32x8:
        max_input = 32;
        max_channels = 8;
        break;
    case torq_hl::VectorizationModeEnum::_16x16:
        max_input = 16;
        max_channels = 16;
        break;
    default:
        max_input = 64;
        max_channels = 4;
        break;
    }

    // We want to maximize the ALU usage and process "max_channels" output channels at a time.
    const uint32_t max_out_channels = output_shape[1] >= 4 ? max_channels : 1;
    const uint32_t out_ch_split = div_ceil(output_shape[1], max_out_channels);
    const uint32_t max_in_channels = std::min<uint32_t>(input_shape[1], HwInfo::iram_depth);
    assert(max_in_channels > 0);

    const uint32_t out_frame_size = align_ceil(out_h * out_w, max_input);
    assert(out_frame_size > 0 && "frame size must be greater than 0");

    const uint32_t total_px_block = div_ceil(out_frame_size, max_input);

    int32_t stride = op.getStride()[0];
    assert(stride <= 2 && "stride must be less than or equal to 2");

    int32_t pad_left = op.getPad()[0];
    int32_t pad_right = op.getPad()[1];
    int32_t pad_top = op.getPad()[2];
    int32_t pad_bottom = op.getPad()[3];

    if (stride == 2 && (ksize_x == 1 && ksize_y == 1)) {
        stride = 1;
    }
    if (stride == 2) {
        pad_left = 1;
        pad_right = 1;
        pad_top = 1;
        pad_bottom = 1;
    }

    if (stride != 1) {
        return failure();
    }
    if (pad_left != 1 || pad_right != 1 || pad_top != 1 || pad_bottom != 1) {
        return failure();
    }

    const int32_t kernel_left = weight_shape[3] / 2;
    const int32_t kernel_right = weight_shape[3] - kernel_left - 1;
    const int32_t kernel_top = weight_shape[2] / 2;
    const int32_t kernel_bottom = weight_shape[2] - kernel_top - 1;

    auto ctx = rewriter.getContext();
    auto shift = op.getShiftFactor();
    auto min = op.getOutputMin();
    auto max = op.getOutputMax();
    auto zp = op.getOutputZp();
    bool segmentOutput = op.getSegmentOutput();

    int blockSize = max_input;
    int inputBlockSize = blockSize + pad_left + pad_right;
    int inputChannels = input_shape[1];
    int inputChStride = input_strides[1];
    int outputChGroups = out_ch_split;
    int outputChInGroup = max_out_channels;
    int outputChStride = output_strides[1];
    int rowSize = output_shape[3];
    int blocksInChannel = total_px_block;
    int actBlockSize = HwInfo::act_width;

    LData input(
        {
            {inputChannels, inputChStride},
            {blocksInChannel, blockSize},
            {ksize_y, rowSize},
            inputBlockSize,
        },
        DType::int8
    );
    LData weight({outputChGroups, inputChannels, ksize_y, ksize_x * outputChInGroup}, DType::int8);
    LData biasScale({outputChGroups, outputChInGroup, 2}, DType::int32);
    LData output(
        {
            {outputChGroups, (outputChStride * outputChInGroup)},
            {blocksInChannel, blockSize},
            {outputChInGroup, outputChStride},
            blockSize / actBlockSize,
            actBlockSize,
        },
        DType::int8
    );

    if (segmentOutput) {
        output = LData(
            {{outputChGroups, (outputChStride * outputChInGroup)},
             {outputChInGroup, outputChStride},
             output_shape[2],
             output_shape[3]},
            DType::int8
        );
        output.partitionByIndexParity2D();
    }

    Slice slice;
    slice.setInputChannelShape(input_shape[2], input_shape[3]);
    slice.setKernel({kernel_left, kernel_right, kernel_top, kernel_bottom});
    slice.setPadding({pad_left, pad_right, pad_top, pad_bottom}, op.getInputZp());
    For(auto og = slice.iterate(outputChGroups)) {
        For(auto b = slice.iterate(blocksInChannel, MemDimTag::A)) {
            PData pdata;
            BData bdata = slice.bram.load(biasScale[og]);
            For(auto u = slice.iterate(inputChannels)) {
                For(auto j = slice.iterate(ksize_y, MemDimTag::J)) {
                    IData idata = slice.iram.load(input[u][b][j]);
                    WData wdata = slice.wram.load(weight[og][u][j]);
                    idata.setShape({{ksize_x, 1}, blockSize});
                    wdata.setShape({ksize_x, outputChInGroup});
                    For(auto i = slice.iterate(ksize_x)) {
                        pdata = slice.alu.outerProductAccumulate(idata[i], wdata[i]);
                    }
                }
            }
            For(auto o = slice.iterate(outputChInGroup)) {
                For(auto a = slice.iterate(blockSize / actBlockSize, MemDimTag::A)) {
                    QData res = slice.act.rescaleClamp(pdata[o][a], bdata[o], shift, zp, min, max);
                    if (segmentOutput) {
                        slice.append(output[og][o], res);
                    }
                    else {
                        slice.store(output[og][b][o][a], res);
                    }
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                    // Operation to replace
        "conv2d-ek",           // Task name
        op.getInput(),         // Input tensor
        op.getWeights(),       // Weights
        op.getScaleBias(),     // BiasScale tensor,
        op.getInit(),          // Output tensor initializer
        slice.getCfgAttr(ctx), // Slice configuration
        slice.getNdls()        // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
