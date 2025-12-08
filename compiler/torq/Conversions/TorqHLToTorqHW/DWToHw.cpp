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
struct In {
    enum { N, CVectors, IVectors, KernelRows, KernelColGroups, CVectorItems };
};

struct Out {
    enum { N, CVectors, CVectorItems, H, W };
};

struct Weight {
    // Here IC dimension is always 1 for depthwise convolution
    enum { OCVectors, IC, H, W, OCVectorItems };
};

// TODO: find a .h where to put this utility function
void convAdjustPadding(LData &input, const LRTBDim &pad, const LRTBDim &kernelBorder);

// Get subview of input, output, weight and biasScale tensors for the given channel offset and count
static void
getSubview(LData &input, LData &output, LData &weight, LData &biasScale, int chOffs, int chCount) {
    int sbWidth = scaleBiasWidth(output.elementType());
    auto wDims = weight.dims();
    int outChVectSize = wDims.size() > Weight::OCVectorItems ? wDims[Weight::OCVectorItems] : 1;
    assert(chOffs % outChVectSize == 0);
    input.subviewDim(Dim::C, chOffs, chCount);
    output.subviewDim(Dim::C, chOffs, chCount);
    weight.subviewDim(Weight::OCVectors, chOffs / outChVectSize, div_ceil(chCount, outChVectSize));
    biasScale.subviewDim(0, chOffs * sbWidth, chCount * sbWidth);
}

// Lower torq_hl op to SliceTaskOp
static torq_hw::SliceTaskOp lowerToHw(
    torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
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
    slice.setInputChannelShape(input.dim(Dim::H), input.dim(Dim::W));

    // Get out ch vector size from weight tensor (or less to handle peeled channels without padding)
    int outChVectSize = std::min(weight.dim(Weight::OCVectorItems), chCount);

    // Vectorize input
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChVectSize);
    int vectSize = vectStride + std::min(kernelBorder.left + kernelBorder.right, alukw - 1);
    input.fuse({Dim::H, Dim::W}).vectorize(vectSize, vectStride);

    // Split the C dimension into CVectors and CVectorItems
    input.reshapeDim(Dim::C, {-1, outChVectSize}, true);

    // Move CVectorItems dimension after IVectors so that we can load 4 IVectors in parallel
    // from neighboring channels (use CVectorItems -2 because Rows & Cols dim not yet inserted)
    input.moveDim(In::CVectors + 1, In::CVectorItems - 2);

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

    // Main processing loops. Instead of processing one input vector at a time, we load multiple
    // vectors in iram from neighboring channels. The number of vectors loaded is equal to the
    // weight vectorization (dimension 4 of the weight vector if present).
    // Loading multiple vectors allows to paralleliza the iram load (4 cycles) with the
    // processing of the idata by the alu (kernelDim.w cycles).
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ocv = slice.iterate(output.dim(Out::CVectors))) {
            For(auto iv = slice.iterate(input.dim(In::IVectors))) {
                PData pdata;
                For(auto kh = slice.iterate(kernelDim.h)) {
                    For(auto kw = slice.iterate(kernelDim.w)) {
                        WData wdata = slice.wram.load(weight[ocv][0][kh][kw]);
                        // Load vectors from neighboring channels
                        IData idata = slice.iram.load(input[batch][ocv][iv][kh][kw / alukw]);
                        idata.setShape({{alukw, Stride(1)}, idata.dim(0), vectStride});
                        pdata = slice.alu.multiScalarProductAccumulate(idata[kw % alukw], wdata);
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

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

LogicalResult convertToHw(torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter) {
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
