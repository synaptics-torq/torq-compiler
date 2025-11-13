// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Layout of the input/output memref data before vectorization
struct Dim {
    enum { N, C, H, W };
};

// Adjust input shape according to the padding to be applied
void adjustPadding(LData &input, const LRTBDim &pad, const LRTBDim &kernelBorder) {
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

LogicalResult convertToHw(torq_hl::Conv2DOp op, PatternRewriter &rewriter) {
    // Layout of the in/out/weight tensors for processing
    struct In : Vectorized {
        enum { N, C, KernelRows };
    };
    struct Out {
        enum { N, ChGroups, ChInGroup, H, W };
    };
    struct Weight {
        enum { OCGroups, IC, H, W, OCInGroup };
    };

    // Define operands in LRAM
    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());
    LData biasScale(op.getScaleBias());

    int stride = op.getStride()[0]; // FIXME: consider all stride values
    LRTBDim pad(op.getPad());
    HWDim kernelDim(weight.dim(Weight::H), weight.dim(Weight::W));
    if (pad.left != 1 || pad.right != 1) {
        // Not supported by HW
        return failure();
    }
    if (stride != 1 || kernelDim.h != 3 || kernelDim.w != 3) {
        // Not supported by this EK kernel
        return failure();
    }

    // FIXME Adjust stride padding for stride 2 case
    if (stride == 2 && (kernelDim.h == 1 && kernelDim.w == 1)) {
        stride = 1;
    }
    if (stride == 2) {
        pad.left = pad.right = pad.top = pad.bottom = 1;
    }

    // Configure convolution parameters
    Slice slice("conv2d");
    LRTBDim kernelBorder;
    kernelBorder.left = kernelDim.w / 2;
    kernelBorder.right = kernelDim.w - kernelBorder.left - 1;
    kernelBorder.top = kernelDim.h / 2;
    kernelBorder.bottom = kernelDim.h - kernelBorder.top - 1;
    adjustPadding(input, pad, kernelBorder);
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setInputChannelShape(input.dim(Dim::H), input.dim(Dim::W));

    // Get channel grouping from weight tensor, otherwise process one channel at a time
    int outChInGroup = weight.dims().size() > Weight::OCInGroup ? weight.dim(Weight::OCInGroup) : 1;

    // Vectorize input and add additional dimension to scan over the kernelDim.h input rows
    int vectStride = slice.alu.iWidth(input.elementType(), weight.elementType(), outChInGroup);
    int vectSize = vectStride + kernelBorder.left + kernelBorder.right;
    int rowSize = output.dim(Out::H);
    ShapeItem rowsDim(kernelDim.h, rowSize, ShapeItem::Tag::KernelRows);
    input.vectorize({Dim::H, Dim::W}, vectSize, vectStride).insertDim(In::KernelRows, rowsDim);

    // Reshape biasScale to match the processing layout
    biasScale.reshapeDim(0, {-1, outChInGroup, scaleBiasEntries(input.elementType())}, true);

    // Reshape output to match the processing layout
    output.reshapeDim(Dim::C, {-1, outChInGroup}, true);
    if (op.getSegmentOutput()) {
        output.partitionByIndexParity2D();
    }

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto og = slice.iterate(output.dim(Out::ChGroups))) {
            For(auto b = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                BData bdata = slice.bram.load(biasScale[og]);
                For(auto u = slice.iterate(input.dim(In::C))) {
                    For(auto j = slice.iterate(kernelDim.h)) {
                        IData idata = slice.iram.load(input[batch][u][j][b]);
                        WData wdata = slice.wram.load(weight[og][u][j]);
                        idata.setShape({{kernelDim.w, 1}, vectStride});
                        wdata.setShape({kernelDim.w, outChInGroup});
                        For(auto i = slice.iterate(kernelDim.w)) {
                            pdata = slice.alu.outerProductAccumulate(idata[i], wdata[i]);
                        }
                    }
                }
                For(auto o = slice.iterate(outChInGroup)) {
                    For(auto a = slice.iterate(pdata.dim(1))) {
                        QData res = slice.act.rescaleClamp(
                            pdata[o][a], bdata[o], op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.append(output[batch][og][o], res);
                    }
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
