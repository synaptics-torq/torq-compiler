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

namespace mlir::syna::torq {

namespace {

struct PoolOutputRegion {
    int64_t y, x, height, width;
};

// Address views keep every region on the original layer's buffers, without slices.
static LData slidingPoolInputView(
    torq_hl::AvgPool2DOp op, PoolOutputRegion region, uint32_t blockSize, uint32_t channelBlocks
) {
    auto [y, x, height, width] = region;
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto inputStrides = getEncodedStridesElements(inputType);
    auto kernel = op.getKernelAttr().asArrayRef();
    auto stride = op.getStrideAttr().asArrayRef();
    auto dilation = op.getDilationAttr().asArrayRef();
    // Output (h,w) starts at (h*strideH,w*strideW); kernel offsets use dilation.
    LData input(op.getInput());
    input.subviewDim(
        1, y * stride[0], (height - 1) * stride[0] + (kernel[0] - 1) * dilation[0] + 1
    );
    input.subviewDim(2, x * stride[1], (width - 1) * stride[1] + (kernel[1] - 1) * dilation[1] + 1);
    input.setShape(
        {{inputType.getShape()[0], inputStrides[0]},
         {height, stride[0] * inputStrides[1]},
         {width, stride[1] * inputStrides[2]},
         {channelBlocks, blockSize},
         {kernel[0], dilation[0] * inputStrides[1]},
         {kernel[1], dilation[1] * inputStrides[2]},
         blockSize}
    );
    return input;
}

static LData slidingPoolOutputView(torq_hl::AvgPool2DOp op, PoolOutputRegion region) {
    auto [y, x, height, width] = region;
    LData output(op.getInit());
    output.subviewDim(1, y, height).subviewDim(2, x, width);
    return output;
}

static void emitSlidingAvgPoolRegion(
    torq_hl::AvgPool2DOp op, PatternRewriter &rewriter, PoolOutputRegion region, int64_t shift,
    int64_t scaleIndex
) {
    auto outputShape = cast<MemRefType>(op.getInit().getType()).getShape();
    auto kernel = op.getKernelAttr().asArrayRef();
    Slice slice("sliding_avgpool2d");
    const uint32_t blockSize = slice.alu.iWidth(DType::int8, DType::int8);
    const uint32_t channelBlocks = div_ceil(outputShape[3], blockSize);
    const uint32_t actWidth = slice.act.width(DType::int8, DType::int8);
    const uint32_t actBlocks = blockSize / actWidth;
    LData input = slidingPoolInputView(op, region, blockSize, channelBlocks);
    LData output = slidingPoolOutputView(op, region);
    LData weights(op.getWeights());
    weights.fuse(weights.shape().size());
    LData biasScale(op.getScaleBias());
    biasScale.fuse(biasScale.shape().size()).subviewDim(0, scaleIndex * 2, 2);
    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);
    For(auto n = slice.iterate(outputShape[0])) {
        For(auto h = slice.iterate(region.height)) {
            For(auto w = slice.iterate(region.width)) {
                For(auto b = slice.iterate(channelBlocks)) {
                    // Start a fresh accumulator for each output pixel/block.
                    PData pdata;
                    For(auto kh = slice.iterate(kernel[0])) {
                        For(auto kw = slice.iterate(kernel[1])) {
                            IData idata = slice.iram.load(input[n][h][w][b][kh][kw]);
                            pdata = slice.alu.scalarProductAccumulate(idata, wdata);
                        }
                    }
                    For(auto a = slice.iterate(actBlocks)) {
                        QData result = slice.act.rescaleClamp(
                            pdata[a], bdata, shift, op.getOutputZp(), op.getOutputMin(),
                            op.getOutputMax()
                        );
                        // Appending to the pixel drops the lanes past its last channel.
                        slice.append(output[n][h][w], result);
                    }
                }
            }
        }
    }
    slice.createSliceTaskOp(rewriter, op.getLoc());
}

static LogicalResult lowerSlidingAvgPool(torq_hl::AvgPool2DOp op, PatternRewriter &rewriter) {
    if (auto regionsAttr = op.getScaleRegionsAttr()) {
        // ACT has one right shift per configuration, so emit one task per region.
        auto regions = regionsAttr.asArrayRef();
        for (size_t i = 0; i < regions.size(); i += 5)
            emitSlidingAvgPoolRegion(
                op, rewriter, {regions[i], regions[i + 1], regions[i + 2], regions[i + 3]},
                regions[i + 4], i / 5
            );
    }
    else {
        auto shape = cast<MemRefType>(op.getInit().getType()).getShape();
        emitSlidingAvgPoolRegion(op, rewriter, {0, 0, shape[1], shape[2]}, op.getShiftFactor(), 0);
    }
    rewriter.eraseOp(op);
    return success();
}

} // namespace

template <>
LogicalResult
Avgpool2DPattern::transform(torq_hl::AvgPool2DOp op, PatternRewriter &rewriter) const {
    if (op.getSlidingPool())
        return lowerSlidingAvgPool(op, rewriter);
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

    // avgpool input is NHWC
    const uint32_t input_channel = input_shape[input_shape.size() - 1];

    uint32_t frame_size = 1;
    for (int i = 1; i < input_shape.size() - 1; i++) {
        frame_size *= input_shape[i];
    }
    assert(frame_size > 0 && "frame size must be greater than 0");

    Slice slice("avgpool2d");

    const uint32_t blockSize = slice.alu.iWidth(inputDType, weightDType);

    uint32_t blockCount = div_ceil(input_channel, blockSize);

    // The input can be strided (e.g. a block transpose pads its output rows): read each frame
    // position at the stride of the last frame dim. The channels and the frame dims must be
    // contiguous so that the frame is a single dimension.
    SmallVector<int64_t> strides = getEncodedStridesElements(input_type);
    int rank = input_shape.size();
    bool contiguous = strides[rank - 1] == 1;
    for (int i = 1; i < rank - 2 && contiguous; i++) {
        contiguous = strides[i] == strides[i + 1] * input_shape[i + 1];
    }
    if (!contiguous) {
        return op.emitError("avgpool input channels and frame dims must be contiguous");
    }
    uint32_t frameSizeStride = rank > 2 ? strides[rank - 2] : input_channel;

    // input memory read address when we do loop
    // (the last block may read past the end of the channels)
    LData input(op.getInput());
    input.setShape({{frame_size, frameSizeStride}, blockCount, blockSize});

    // Same weight used for the entire computation
    LData weights(op.getWeights());
    weights.setShape({1});

    // all blocks use the same bias and scale
    LData biasScale = LData(op.getScaleBias()).subviewDim(0, 0, biasScaleWidth(inputDType));

    // The output is written in whole blocks too
    const uint32_t actOutputWidth = slice.act.width(inputDType, weightDType);
    LData output(op.getInit());
    output.setShape({blockCount, blockSize / actOutputWidth, actOutputWidth});

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

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));

    return success();
}

} // namespace mlir::syna::torq
