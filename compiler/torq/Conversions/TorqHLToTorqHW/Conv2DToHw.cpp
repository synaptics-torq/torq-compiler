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

// TODO: check if this option is worth it in terms of performance gain
// #define OPTIMIZE_FOR_SMALL_KERNEL

// Layout of the input/output memref data before vectorization
using Dim = NCHW;

// Layout of the in/out/weight tensors for processing
struct In : Vectorized {
    enum { N, C, KernelRows, KernelColGroups };
};

struct Out {
    enum { N, CVectors, CVectorItems, H, W };
};

struct Weight {
    enum { OCVectors, IC, H, W, OCVectorItems };
};

// Adjust input shape according to the padding to be applied
void convAdjustPadding(LData &input, const LRTBDim &pad, const LRTBDim &kernelBorder) {
    assert(pad.left == kernelBorder.left && pad.right == kernelBorder.right && "No HW support");
    assert(pad.top <= kernelBorder.top && pad.bottom <= kernelBorder.bottom);
    if (int validPadLines = kernelBorder.top - pad.top) {
        // NPU always starts fetching data kernel.top rows before the beginning of the data.
        // If (some) valid padding, add an offset to start fetching from the beginning of the frame
        // and reduce the height accordingly
        input.setOffset(
            input.offset() + validPadLines * input.shape()[Dim::H].stride.intVal.value()
        );
        input.getShape()[Dim::H].count -= validPadLines;
    }
    if (int validPadLines = kernelBorder.bottom - pad.bottom) {
        // If valid padding reduce the height accordingly
        input.getShape()[Dim::H].count -= validPadLines;
    }
}

// Get subview of output, weight and biasScale tensors for the given output channel offset and count
static void getSubview(LData &output, LData &weight, LData &biasScale, int chOffs, int chCount) {
    int sbWidth = biasScaleWidth(output.elementType());
    auto wDims = weight.dims();
    int outChVectSize = wDims.size() > Weight::OCVectorItems ? wDims[Weight::OCVectorItems] : 1;
    assert(chOffs % outChVectSize == 0);
    output.subviewDim(Dim::C, chOffs, chCount);
    weight.subviewDim(Weight::OCVectors, chOffs / outChVectSize, div_ceil(chCount, outChVectSize));
    biasScale.subviewDim(0, chOffs * sbWidth, chCount * sbWidth);
}

/*
The classical algorithm to compute a 2D convolution on a 4D NCHW tensor consists of 7 nested loops.
The 4 outermost loops are used to iterate over each position in the NCHW output tensor.
The 3 innermost loops are needed to multiply and accumulate the input region associated with each
output pixel by the corresponding weights (each input region has shape [InputChannels, KH, KW]).
The structure of the kernel below closely resembles this algorithm, with a few changes to take
advantage of the hardware capabilities:
1) Instead of computing one output value at a time, we compute P output values on Q channels in
   parallel. P and Q depend on the ALU mode and data-type, typical is 64x4.
2) We have just 6 loops, not 7. Instead of explicitly iterating over the H,W dimensions we flatten
   both to a single dimension of data vectors. The NPU takes care of inserting padding values
   where needed.
3) To handle input vectors and channel vectors we add the required dimensions to the tensors
4) Moving horizontally on a line to fetch the input vectors (inner loop on KW) would generate a lot
   of LRam traffic to get data that is overlapping except for one pixel. To avoid this we load to
   IRam two additional pixels each time, so that for 3 iterations the ALU input data can be fetched
   directly from IRam. For kernels KW > 3 this is parallelized with the loading of the next vector.
   The data loaded to IRam is resized from a vector of size [P+2] to an array of size
   [{3:stride(1)}, P] so that at each iteration shifting by one pixel the data loaded to the ALU
   can be done simply by indexing the first dimension.
*/

// Lower stride-2 Conv2DOp to SliceTaskOp.
// The spatial domain is decomposed into 4 quadrants (EE/EO/OE/OO) by reshaping the fused H*W
// dimension as {RowQuadrant=2, ColQuadrant=2, Vectors}. The kernel then iterates over both
// quadrants and kernel positions, halving the effective kernel step count along each axis.
static torq_hw::SliceTaskOp lowerConvStride2ToHw(
    torq_hl::Conv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset, int chCount
) {
    struct In : Vectorized {
        enum { N, C, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps };
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

    if (weight.dims().size() <= Weight::OCVectorItems) {
        weight.insertDim(Weight::OCVectorItems, {1});
    }
    // Only output channels, weights and bias are sliced per tile; all input channels are always
    // processed (unlike depthwise where input channels == output channels).
    getSubview(output, weight, biasScale, chOffset, chCount);

    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    LRTBDim pad(op.getPad());

    // Configure convolution parameters
    Slice slice("conv2dStride2");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    slice.setKernel(kernelBorder);

    // For kernels where pad == (kw-1)/2 the kernel tiles evenly (e.g. 2x2), so stride_offset=0.
    // For other kernels (e.g. 3x3 with same padding) stride_offset=1 is required to align the
    // kernel centre to the correct quadrant (same logic as maxpool stride-2).
    int strideOffset = (pad.left == pad.top && pad.left == (kernelDim.w - 1) / stride) ? 0 : 1;

    slice.setStrideOffset(strideOffset);
    slice.setStride(stride);

    // Output spatial shape is halved by stride; must be set before fusing/reshaping input dims.
    slice.setOutputChannelShape(input.dim(Dim::H) / stride, input.dim(Dim::W) / stride);

    if (pad.top) {
        pad.top = kernelBorder.top;
    }
    if (pad.bottom) {
        pad.bottom = kernelBorder.bottom;
    }
    slice.setPadding(pad, op.getInputZp());

    // Get out ch vector size from weight tensor (or less to handle peeled channels without padding)
    int outChVectSize = std::min(weight.dim(Weight::OCVectorItems), chCount);

    // Vectorize input: fuse H*W then split into a 2x2 quadrant grid (the stride-2 decomposition),
    // followed by the spatial vector dimension.
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChVectSize);
    int vectSize = vectStride + std::min(kernelBorder.left /* + kernelBorder.right */, alukw - 1);
    input.fuse({Dim::H, Dim::W}).reshapeDim(Dim::H, {2, 2, -1}).vectorize(vectSize, vectStride);
    // Shape: [N, C, RowQuadrant=2, ColQuadrant=2, Vectors, vectSize]

    // Insert a dimension to step over kernel rows; stride-2 halves the number of row steps
    // because each quadrant step advances by 2 input rows.
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(div_ceil(kernelDim.h, 2), Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);
    // Shape: [N, C, RowQuadrant, ColQuadrant, KernelRows, Vectors, vectSize]

    // Insert a dimension to step over kernel column groups; halved by stride then grouped by alukw.
    ShapeItem colGroupsDim(
        div_ceil(div_ceil(kernelDim.w, 2), alukw), Stride(alukw), ShapeItem::Tag::KernelCols
    );
    input.insertDim(In::KernelColGrps, colGroupsDim);
    // Shape: [N, C, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps, Vectors, vectSize]

    // Tag the quadrant dimensions so the HW descriptor generator treats them as kernel scan dims.
    input.getShape()[In::RowQuadrant].tag = ShapeItem::Tag::KernelRows;
    input.getShape()[In::ColQuadrant].tag = ShapeItem::Tag::KernelCols;

    // Reshape output to match the processing layout
    output.reshapeDim(Dim::C, {-1, outChVectSize});
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChVectSize, biasScaleWidth(input.elementType())});

    // Determine which quadrant (EE/EO/OE/OO) the kernel scan starts from and adjust the quadrant
    // dimension strides so the hardware iterates in the correct phase order.
    // start_pos_{x,y}==0 means start from the even quadrant; ==1 means start from the odd quadrant.
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

    // Main processing loops.
    // Outer loops: batch, output-channel-vector, spatial-pixel-vector.
    // Inner loops: input-channel, kernel-height, row-quadrant, col-quadrant, kernel-width.
    // For each (kh, qr, kw, qc) position:
    //   - kh/2 and kw/(alukw*2) index into the halved KernelRows/KernelColGrps IRAM dims.
    //   - kw%2 selects the even or odd column quadrant tap within the loaded IRAM window.
    //   - outerProductAccumulate: pixel vector [vectStride] x weight row [outChVectSize]
    //     -> accumulates into pdata [outChVectSize, vectStride].
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto ic = slice.iterate(input.dim(In::C))) {
                    For(auto kh = slice.iterate(kernelDim.h)) {
                        For(auto qr = slice.iterate(input.dim(In::RowQuadrant))) {
                            For(auto qc = slice.iterate(input.dim(In::ColQuadrant))) {
                                For(auto kw = slice.iterate(kernelDim.w)) {
                                    WData wdata = slice.wram.load(weight[ocv][ic][kh][kw]);
                                    IData idata = slice.iram.load(
                                        input[batch][ic][qr][qc][kh / 2][kw / (alukw * 2)][iv]
                                    );
                                    idata.setShape({{alukw, Stride(1)}, vectStride});
                                    pdata = slice.alu.outerProductAccumulate(idata[kw % 2], wdata);
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
                        slice.append(output[batch][ocv][o], res);
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
    torq_hl::Conv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset, int chCount
) {
    if (!hasEkLoweringConv(op)) {
        return {};
    }

    int stride = op.getStride()[0];
    if (stride == 2) {
        // For a 1x1 kernel with stride 2, only the EE (top-left) quadrant is used,
        // so the operation is equivalent to a stride-1 conv on the pre-segmented input.
        auto weightShape = cast<ShapedType>(op.getWeights().getType()).getShape();
        int kh = weightShape[Weight::H];
        int kw = weightShape[Weight::W];
        if (kh == 1 && kw == 1) {
            // Fall through to the stride-1 path below (stride variable not used further here).
        }
        else {
            return lowerConvStride2ToHw(op, rewriter, taskInitTensor, chOffset, chCount);
        }
    }

    // Define operands in LRAM
    LData input(op.getInput());
    LData output(op.getInit());
    LData biasScale(op.getScaleBias());
    LData weight(op.getWeights());
    if (weight.dims().size() <= Weight::OCVectorItems) {
        weight.insertDim(Weight::OCVectorItems, {1});
    }
    getSubview(output, weight, biasScale, chOffset, chCount);

    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    LRTBDim pad(op.getPad());

    // Configure convolution parameters
    Slice slice("conv2d");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    convAdjustPadding(input, pad, kernelBorder);
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setOutputChannelShape(input.dim(Dim::H), input.dim(Dim::W));

    // Get out ch vector size from weight tensor (or less to handle peeled channels without padding)
    int outChVectSize = std::min(weight.dim(Weight::OCVectorItems), chCount);

    // Vectorize input
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChVectSize);
    int vectSize = vectStride + std::min(kernelBorder.left + kernelBorder.right, alukw - 1);
    input.fuse({Dim::H, Dim::W}).vectorize(vectSize, vectStride);

    // Add additional dimension to scan over the kernelDim.h input rows
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(kernelDim.h, Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);

    // Add additional dimension to scan over the kernelDim.w/ColGroupSize input column groups
    ShapeItem colGroupsDim(div_ceil(kernelDim.w, alukw), Stride(alukw), ShapeItem::Tag::KernelCols);
    input.insertDim(In::KernelColGroups, colGroupsDim);

    // Reshape output to match the processing layout
    output.reshapeDim(Dim::C, {-1, outChVectSize});
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChVectSize, biasScaleWidth(input.elementType())});

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto ic = slice.iterate(input.dim(In::C))) {
                    For(auto kh = slice.iterate(kernelDim.h)) {
#ifdef OPTIMIZE_FOR_SMALL_KERNEL
                        // For small kernels we can load all weights at once to reduce LRam traffic
                        WData wdata = slice.wram.load(weight[ocv][ic][kh]);
                        For(auto kw = slice.iterate(kernelDim.w)) {
                            IData idata = slice.iram.load(input[batch][ic][kh][kw / alukw][iv]);
                            idata.setShape({{alukw, Stride(1)}, vectStride});
                            pdata = slice.alu.outerProductAccumulate(idata[kw % alukw], wdata[kw]);
                        }
#else
                        For(auto kw = slice.iterate(kernelDim.w)) {
                            WData wdata = slice.wram.load(weight[ocv][ic][kh][kw]);
                            IData idata = slice.iram.load(input[batch][ic][kh][kw / alukw][iv]);
                            idata.setShape({{alukw, Stride(1)}, vectStride});
                            pdata = slice.alu.outerProductAccumulate(idata[kw % alukw], wdata);
                        }
#endif
                    }
                }
                For(auto o = slice.iterate(outChVectSize)) { // Not necessarily all the pdata
                    BData bdata = slice.bram.load(biasScale[ocv][o]);
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][av], bdata, op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.append(output[batch][ocv][o], res);
                    }
                }
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

LogicalResult convertToHw(torq_hl::Conv2DOp op, PatternRewriter &rewriter) {
    Value initValue = op.getInit();
    auto wDims = LData(op.getWeights()).dims();
    int outChVectSize = wDims.size() > Weight::OCVectorItems ? wDims[Weight::OCVectorItems] : 1;
    int outChCount = LData(op.getInit()).dim(Dim::C);
    torq_hw::SliceTaskOp hwOp;

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
