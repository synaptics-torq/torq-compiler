// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h" // For hasEkLowering
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Layout of the input/output memref data before vectorization
using Dim = NCHW;

struct Out {
    enum { N, CVectors, CItems, H, W };
};

struct Weight {
    // IC dimension is always 1 for normal depthwise convolutions, but here we use it to represent
    // the input channel groups used to accumulate multiple input channels together when depthwise
    // has been converted from another one with stride [SH, 1] with SH > 1.
    // In this case the DW will have the original output shape [N, C, H, W], but input shape
    // [N, C * SH, H / SH, W] and weight shape [O, SH, KH / SH, KW, o].
    // In both the input and the kernel, the rows have been reorganized in even/odd.
    // More info in: https://github.com/synaptics-torq/torq-compiler-dev/issues/506
    enum { OCVectors, IC, H, W, OCItems };
};

// TODO: find a .h where to put this utility function
void convAdjustPadding(LData &input, const LRTBDim &pad, const LRTBDim &kernelBorder);

// Get subview of input, output, weight and biasScale tensors for the given channel offset and count
static void
getSubview(LData &input, LData &output, LData &weight, LData &biasScale, int chOffs, int chCount) {
    int sbWidth = biasScaleWidth(output.elementType());
    auto wDims = weight.dims();
    int inChGroupSize = weight.dim(Weight::IC);
    int outChVectSize = wDims.size() > Weight::OCItems ? wDims[Weight::OCItems] : 1;
    assert(chOffs % outChVectSize == 0);
    input.subviewDim(Dim::C, chOffs * inChGroupSize, chCount * inChGroupSize);
    output.subviewDim(Dim::C, chOffs, chCount);
    weight.subviewDim(Weight::OCVectors, chOffs / outChVectSize, div_ceil(chCount, outChVectSize));
    biasScale.subviewDim(0, chOffs * sbWidth, chCount * sbWidth);
}

// Lower depthwise 1D stride=1 special case to SliceTaskOp
// Input: NHWC format [N, 1, Width, Channels]
// Weight: Packed format [Ch_outer, Kh, Kw, 32] where Ch_outer = Channels/32
// Lower depthwise 1D stride=1 convolution to hardware
// Processing strategy: For each channel block (16 for BF16, 32 for INT8),
// accumulate across kernel width using element-wise multiply-accumulate
static torq_hw::SliceTaskOp lowerDw1dStride1ToHw(
    torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    if (!hasEkLoweringConv(op)) {
        return {};
    }

    // Define operands in LRAM
    LData input(op.getInput());         // NHWC format: [N, H, W, C]
    LData output(op.getInit());         // NHWC format: [N, H, W, C]
    LData biasScale(op.getScaleBias()); // Bias and scale data
    LData weight(op.getWeights());      // Packed format: [Ch_outer, Kh, Kw, inner]

    // Extract parameters
    int sbWidth = biasScaleWidth(output.elementType());
    // Get the inner tile size from packed weights: 16 for BF16, 32 for INT8
    int outChVectSize = weight.dim(weight.shape().size() - 1);
    const int channelDim = 3; // NHWC: channels at last dimension

    assert(chOffset % outChVectSize == 0);

    // Apply channel subviews
    input.subviewDim(channelDim, chOffset, chCount);
    output.subviewDim(channelDim, chOffset, chCount);
    weight.subviewDim(0, chOffset / outChVectSize, div_ceil(chCount, outChVectSize));
    biasScale.subviewDim(0, chOffset * sbWidth, chCount * sbWidth);

    // Reshape weights from [Ch_outer, Kh, Kw, inner] to [Ch_outer, Kw, inner]
    // Since Kh=1 for depthwise 1D, fuse last 3 dims then reshape
    int kw = weight.dim(3); // Kernel width
    weight.fuse(4);
    weight.reshapeDim(1, {kw, outChVectSize}, false);

    Slice slice("DepthwiseConv1dStride1");

    // Calculate activation block parameters
    const int32_t actBlockSize = slice.act.width(input.elementType(), weight.elementType(), true);
    const int32_t actBlockCount = div_ceil(outChVectSize, actBlockSize);

    // Reshape tensors to split channels into [Ch_outer, outChVectSize]
    input.reshapeDim(channelDim, {-1, outChVectSize}, true);
    output.reshapeDim(channelDim, {-1, outChVectSize}, true);
    biasScale.reshapeDim(0, {-1, actBlockCount, actBlockSize, sbWidth}, true);

    int chOuterCount = weight.dim(0);

    // Main processing loop
    For(auto batch = slice.iterate(input.dim(0))) {
        For(auto ch_block = slice.iterate(chOuterCount)) {
            PData pdata;

            // Accumulate across kernel width
            For(auto kw_pos = slice.iterate(kw)) {
                WData wdata = slice.wram.load(weight[ch_block][kw_pos]);
                IData idata = slice.iram.load(input[batch][0][kw_pos][ch_block]);
                pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);
            }

            // Process through activation in blocks
            For(auto a = slice.iterate(actBlockCount)) {
                BData bdata = slice.bram.load(biasScale[ch_block][a]);
                QData res = slice.act.rescaleClamp(
                    pdata[a], bdata, op.getShiftFactor(), op.getOutputZp(), op.getOutputMin(),
                    op.getOutputMax()
                );
                slice.append(output[batch][0][0][ch_block], res);
            }
        }
    }

    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

static torq_hw::SliceTaskOp lowerDwStride2ToHw(
    torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    struct In : Vectorized {
        enum { N, CVectors, CItems, CInGroup, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps };
    };

    if (!hasEkLoweringConv(op)) {
        return {};
    }

    int stride = op.getStride()[0];

    // Define operands in LRAM
    LData input(op.getInput());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());
    LData weight(op.getWeights());

    if (weight.dims().size() <= Weight::OCItems) {
        weight.insertDim(Weight::OCItems, {1});
    }
    getSubview(input, output, weight, biasScale, chOffset, chCount);

    // FIXME Adjust stride & padding for stride 2 case (also consider all stride values)
    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    LRTBDim pad(op.getPad());

    // Configure convolution parameters
    Slice slice("DepthwiseConv2dStride2");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    slice.setKernel(kernelBorder);

    // For 2x2 (and generally when pad == (kw-1)/2), stride_offset should be 0
    // because the kernel tiles evenly without needing an offset. For other kernels
    // (e.g. 3x3 with same padding), stride_offset=1 is required (same as maxpool stride2).
    int strideOffset = (pad.left == pad.top && pad.left == (kernelDim.w - 1) / stride) ? 0 : 1;
    // FIXME: not sure if needed (from dw sride2 NDL kernel)
    // if (stride == 2 && (ksize_x == 1 && ksize_y == 1)) {
    //     stride = 1;
    //     stride_offset = 0;
    // }

    slice.setStrideOffset(strideOffset);
    slice.setStride(stride);

    slice.setOutputChannelShape(input.dim(Dim::H) / stride, input.dim(Dim::W) / stride);

    if (pad.top) {
        pad.top = kernelBorder.top;
    }
    if (pad.bottom) {
        pad.bottom = kernelBorder.bottom;
    }
    slice.setPadding(pad, op.getInputZp());

    // Get out ch vector size from weight tensor (or less to handle peeled channels without padding)
    int outChVectSize = std::min(weight.dim(Weight::OCItems), chCount);

    // Vectorize input
    // Note that we don't have to limit the vector size to the segment size (H*W/4)
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChVectSize);

    int vectSize = vectStride + std::min(kernelBorder.left /* + kernelBorder.right */, alukw - 1);
    input.fuse({Dim::H, Dim::W}).reshapeDim(Dim::H, {2, 2, -1}).vectorize(vectSize, vectStride);
    // Shape: [N, C, RowQuadrant, ColQuadrant, Vectors, vectSize]

    // Split the input channels in groups to accumulate over the input channels in the same group
    // After the split input.dim(Dim::C) will be equal to output.dim(Dim::C) as for normal dw convs.
    // For standard depthwise convs each group contains exactly 1 input channel.
    int inChGroupSize = weight.dim(Weight::IC);
    assert(input.dim(Dim::C) % inChGroupSize == 0 && "Input channels not multiple of group");
    input.reshapeDim(Dim::C, {-1, inChGroupSize}, false);
    // Shape: [N, C, CInGroup, RowQuadrant, ColQuadrant, Vectors, vectSize]

    // Split the C dimension into CVectors and CItems
    input.reshapeDim(In::CVectors, {-1, outChVectSize}, true);
    // Shape: [N, CVectors, CItems, CInGroup, RowQuadrant, ColQuadrant, Vectors, vectSize]

    // Add additional dimension to scan over the kernelDim.h input rows
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(div_ceil(kernelDim.h, 2), Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);
    // [N, CVectors, CItems, CInGroup, RowQuadrant, ColQuadrant, KernelRows, Vectors, vectSize]

    // Add additional dimension to scan over the kernelDim.w/ColGroupSize input column groups
    ShapeItem colGroupsDim(
        div_ceil(div_ceil(kernelDim.w, 2), alukw), Stride(alukw), ShapeItem::Tag::KernelCols
    );
    input.insertDim(In::KernelColGrps, colGroupsDim);
    // Shape: [N, CVectors, CItems, CInGroup, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps,
    // Vectors, vectSize]

    // Tag quandrants since the HW needs to be aware
    input.getShape()[In::RowQuadrant].tag = ShapeItem::Tag::KernelRows;
    input.getShape()[In::ColQuadrant].tag = ShapeItem::Tag::KernelCols;

    // Reshape output to match the processing layout
    output.reshapeDim(Dim::C, {-1, outChVectSize}, true);
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChVectSize, biasScaleWidth(input.elementType())}, true);

    // If start_pos we have to iterate the corresponding quadrant in reverse order,
    // but the starting offset and stride must be tweaked so we can't use the standard reverse()
    // method. Instead we manually adjust the starting offset and stride of the quadrant dimensions.
    const int32_t start_pos_x = (-kernelBorder.left + strideOffset) & 1;
    const int32_t start_pos_y = (-kernelBorder.top + strideOffset) & 1;
    const int32_t kernel_left_even = (kernelBorder.left - strideOffset + 1) >> 1;
    const int32_t kernel_left_odd = kernelBorder.left - strideOffset - kernel_left_even;
    const int32_t kernel_top_even = (kernelBorder.top - strideOffset + 1) >> 1;
    const int32_t kernel_top_odd = kernelBorder.top - strideOffset - kernel_top_even;
    const int32_t colQStride = input.shape()[In::ColQuadrant].stride.intVal.value();
    const int32_t rowQStride = input.shape()[In::RowQuadrant].stride.intVal.value();
    input.setOffset(input.offset() + (2 * start_pos_y + start_pos_x) * colQStride);
    input.getShape()[In::ColQuadrant].stride =
        colQStride * (start_pos_x == 0 ? 1 : -1) + kernel_left_even - kernel_left_odd;
    input.getShape()[In::RowQuadrant].stride = rowQStride * (start_pos_y == 0 ? 1 : -1) +
                                               output.dim(-1) * (kernel_top_even - kernel_top_odd);

    // Main processing loops. Instead of processing one input vector at a time, we load multiple
    // vectors in iram from neighboring channels. The number of vectors loaded is equal to the
    // weight vectorization (dimension 4 of the weight vector if present).
    // Loading multiple vectors allows to parallelize the iram load (4 cycles) with the
    // processing of the idata by the alu (kernelDim.w cycles).
    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto ic = slice.iterate(weight.dim(Weight::IC))) {
                    For(auto kh = slice.iterate(kernelDim.h)) {
                        For(auto qr = slice.iterate(input.dim(In::RowQuadrant))) {
                            For(auto qc = slice.iterate(input.dim(In::ColQuadrant))) {
                                For(auto kw = slice.iterate(kernelDim.w)) {
                                    WData wdata = slice.wram.load(weight[ocv][ic][kh][kw]);
                                    // Load vectors from neighboring channels
                                    IData idata = slice.iram.load(
                                        input[n][ocv][":"][ic][qr][qc][kh / 2][kw / (alukw * 2)][iv]
                                    );
                                    idata.setShape({{alukw, Stride(1)}, idata.dim(0), vectStride});
                                    pdata = slice.alu.multiScalarProductAccumulate(
                                        idata[kw % 2], wdata
                                    );
                                }
                            }
                        }
                    }
                }
                For(auto o = slice.iterate(outChVectSize)) { // Not necessarily all the pdata
                    BData bdata = slice.bram.load(biasScale[ocv][o]);
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][av], bdata, op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.append(output[n][ocv][o], res);
                    }
                }
            }
        }
    }

    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

// Lower torq_hl op to SliceTaskOp
static torq_hw::SliceTaskOp lowerToHw(
    torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    // Layout of the in/out/weight tensors for processing
    struct In : Vectorized {
        enum { N, CVectors, CItems, CInGroup, KernelRows, KernelColGroups };
    };

    if (!hasEkLoweringConv(op)) {
        return {};
    }

    int stride = op.getStride()[0];
    if (stride == 2) {
        return lowerDwStride2ToHw(op, rewriter, taskInitTensor, chOffset, chCount);
    }

    // Define operands in LRAM
    LData input(op.getInput());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());
    LData weight(op.getWeights());

    if (weight.dims().size() <= Weight::OCItems) {
        weight.insertDim(Weight::OCItems, {1});
    }
    getSubview(input, output, weight, biasScale, chOffset, chCount);

    // FIXME Adjust stride & padding for stride 2 case (also consider all stride values)
    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    LRTBDim pad(op.getPad());

    // Configure convolution parameters
    Slice slice("DepthwiseConv2d");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    convAdjustPadding(input, pad, kernelBorder);
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setOutputChannelShape(input.dim(Dim::H), input.dim(Dim::W));

    // Get out ch vector size from weight tensor (or less to handle peeled channels without padding)
    int outChVectSize = std::min(weight.dim(Weight::OCItems), chCount);

    // Vectorize input
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChVectSize);
    int vectSize = vectStride + std::min(kernelBorder.left + kernelBorder.right, alukw - 1);
    input.fuse({Dim::H, Dim::W}).vectorize(vectSize, vectStride);

    // Split the input channels in groups to accumulate over the input channels in the same group
    // After the split input.dim(Dim::C) will be equal to output.dim(Dim::C) as for normal dw convs.
    // For standard depthwise convs each group contains exactly 1 input channel.
    int inChGroupSize = weight.dim(Weight::IC);
    assert(input.dim(Dim::C) % inChGroupSize == 0 && "Input channels not multiple of group");
    input.reshapeDim(Dim::C, {-1, inChGroupSize}, false);

    // Split the C dimension into CVectors and CItems
    input.reshapeDim(In::CVectors, {-1, outChVectSize}, true);

    // Add additional dimension to scan over the kernelDim.h input rows
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(kernelDim.h, Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);

    // Add additional dimension to scan over the kernelDim.w/ColGroupSize input column groups
    ShapeItem colGroupsDim(div_ceil(kernelDim.w, alukw), Stride(alukw), ShapeItem::Tag::KernelCols);
    input.insertDim(In::KernelColGroups, colGroupsDim);

    // Reshape output to match the processing layout
    output.reshapeDim(Dim::C, {-1, outChVectSize}, true);
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChVectSize, biasScaleWidth(input.elementType())}, true);

    // Main processing loops. Instead of processing one input vector at a time, we load multiple
    // vectors in iram from neighboring channels. The number of vectors loaded is equal to the
    // weight vectorization (dimension 4 of the weight vector if present).
    // Loading multiple vectors allows to paralleliza the iram load (4 cycles) with the
    // processing of the idata by the alu (kernelDim.w cycles).
    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto ic = slice.iterate(weight.dim(Weight::IC))) {
                    For(auto kh = slice.iterate(kernelDim.h)) {
                        For(auto kw = slice.iterate(kernelDim.w)) {
                            WData wdata = slice.wram.load(weight[ocv][ic][kh][kw]);
                            // Load vectors from neighboring channels
                            IData idata =
                                slice.iram.load(input[n][ocv][":"][ic][kh][kw / alukw][iv]);
                            idata.setShape({{alukw, Stride(1)}, idata.dim(0), vectStride});
                            pdata =
                                slice.alu.multiScalarProductAccumulate(idata[kw % alukw], wdata);
                        }
                    }
                }
                For(auto o = slice.iterate(outChVectSize)) { // Not necessarily all the pdata
                    BData bdata = slice.bram.load(biasScale[ocv][o]);
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][av], bdata, op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.append(output[n][ocv][o], res);
                    }
                }
            }
        }
    }

    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

LogicalResult convertToHw(torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter) {
    // Check for depthwise 1D stride=1 special case
    bool isDw1dStride1 = op.getIsDw1dStride1();

    Value initValue = op.getInit();

    auto wDims = LData(op.getWeights()).dims();
    // Get the vector size from the last dimension of the weight tensor
    // For depthwise 1D stride=1: weight is [Ch_outer, Kh, Kw, inner] so inner is at index 3
    // For standard depthwise: weight is [Ch_outer, IC, Kh, Kw, inner] so inner is at index 4
    int outChVectSize = wDims.size() > 0 ? wDims[wDims.size() - 1] : 1;

    // For depthwise 1D stride=1, the layout is NHWC, so channel is at dimension 3
    // For regular depthwise 2D, the layout is NCHW, so channel is at dimension 1 (Dim::C)
    LData outputData(op.getInit());
    int outChCount = isDw1dStride1 ? outputData.dim(outputData.shape().size() - 1) // NHWC: last dim
                                   : outputData.dim(Dim::C);                       // NCHW: dim 1
    torq_hw::SliceTaskOp hwOp;

    if (isDw1dStride1) {
        // Use specialized lowering for depthwise 1D stride=1
        LLVM_DEBUG(llvm::dbgs() << "Lowering depthwise 1D stride=1 case\n");

        // Peel-off channels not multiple of output channel grouping
        if (int peeledOutCh = outChCount % outChVectSize) {
            if (!(hwOp = lowerDw1dStride1ToHw(
                      op, rewriter, initValue, outChCount - peeledOutCh, peeledOutCh
                  ))) {
                return failure();
            }
            initValue = hwOp.getQ()[0];
            outChCount -= peeledOutCh;
        }
        if (outChCount > 0) {
            if (!(hwOp = lowerDw1dStride1ToHw(op, rewriter, initValue, 0, outChCount))) {
                return failure();
            }
        }
        rewriter.replaceOp(op, hwOp.getOperation()->getResults());
        return success();
    }

    // Standard depthwise conv2d lowering

    // Peel-off channels not multiple of output channel grouping. This removes any
    // requirement for output channels padding.
    // Here we could check output stride to see if we have padding, in which case peeling not needed
    if (int peeledOutCh = outChCount % outChVectSize) {
        if (!(hwOp = lowerToHw(op, rewriter, initValue, outChCount - peeledOutCh, peeledOutCh))) {
            return failure();
        }
        initValue = hwOp.getQ()[0];
        outChCount -= peeledOutCh;
    }
    if (outChCount > 0) {
        if (!(hwOp = lowerToHw(op, rewriter, initValue, 0, outChCount))) {
            return failure();
        }
    }
    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
