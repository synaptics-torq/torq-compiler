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
        input.setOffset(validPadLines * input.shape()[Dim::H].stride.intVal.value());
        input.getShape()[Dim::H].count -= validPadLines;
    }
    if (int validPadLines = kernelBorder.bottom - pad.bottom) {
        // If valid padding reduce the height accordingly
        input.getShape()[Dim::H].count -= validPadLines;
    }
}

// Get subview of output, weight and biasScale tensors for the given output channel offset and count
static void getSubview(LData &output, LData &weight, LData &biasScale, int chOffs, int chCount) {
    int sbWidth = scaleBiasWidth(output.elementType());
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

// Lower torq_hl op to SliceTaskOp
static torq_hw::SliceTaskOp lowerToHw(
    torq_hl::Conv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset, int chCount
) {
    if (!hasEkLoweringConv(op)) {
        return {};
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

    // FIXME Adjust stride & padding for stride 2 case (also consider all stride values)
    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    LRTBDim pad(op.getPad());

    // Configure convolution parameters
    Slice slice("conv2d");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    convAdjustPadding(input, pad, kernelBorder);
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setInputChannelShape(input.dim(Dim::H), input.dim(Dim::W));

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
    output.reshapeDim(Dim::C, {-1, outChVectSize}, true);
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChVectSize, scaleBiasWidth(input.elementType())}, true);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto ic = slice.iterate(input.dim(In::C))) {
                    For(auto kh = slice.iterate(kernelDim.h)) {
                        WData wdata = slice.wram.load(weight[ocv][ic][kh]);
                        For(auto kw = slice.iterate(kernelDim.w)) {
                            IData idata = slice.iram.load(input[batch][ic][kh][kw / alukw][iv]);
                            idata.setShape({{alukw, Stride(1)}, vectStride});
                            pdata = slice.alu.outerProductAccumulate(idata[kw % alukw], wdata[kw]);
                        }
                    }
                }
                BData bdata = slice.bram.load(biasScale[ocv]);
                For(auto o = slice.iterate(outChVectSize)) { // Not necessarily all the pdata
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][av], bdata[o], op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.append(output[batch][ocv][o], res);
                    }
                }
            }
        }
    }

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
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
