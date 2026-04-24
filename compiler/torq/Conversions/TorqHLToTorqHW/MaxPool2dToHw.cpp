// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
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

// Lower MaxPool 1D to hardware using Slice API
static torq_hw::SliceTaskOp lowerMaxPool1DToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    // Layout of the in/out tensors for processing
    struct In : Vectorized {
        enum { N, C, H, KernelH };
    };
    LData input(op.getInput());
    LData output(op.getInit());

    auto kernel = op.getKernel();
    int kh = kernel[0];
    int kw = kernel[1];

    assert(kw == 1 && kh != 1 && "Expected kw=1 and kh!=1 for 1D maxpool");

    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    Slice slice("MaxPool1D");

    int vectSize = slice.alu.iWidth(input.elementType());
    input.vectorize(vectSize);

    auto stride = op.getStride();
    assert(kh == stride[0] && "Kernel height (kh) must be equal to stride[0] for MaxPool1D");

    input.reshapeDim(Dim::H, {-1, kh});

    int in_ch = input.dim(In::C);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch_block = slice.iterate(in_ch)) {
            For(auto op_row = slice.iterate(input.dim(In::H))) {
                For(auto ip_vec = slice.iterate(input.dim(In::Vectors))) {
                    PData pdata;
                    For(auto kh_idx = slice.iterate(kh)) {
                        IData idata =
                            slice.iram.load(input[batch][ch_block][op_row][kh_idx][ip_vec]);
                        pdata = slice.alu.accumulate(idata, torq_hw::ALUOp1Mode::MAX);
                    }
                    For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                        QData res = slice.act.load(pdata[av]);
                        slice.append(output[batch][ch_block][op_row], res);
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

// Pool along W only (kh==1, kw>1) by splitting the dense W dimension into [outputWidth, kw].
static torq_hw::SliceTaskOp lowerMaxPool1DHorizontalToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    using In = NCHW;
    LData input(op.getInput());
    LData output(op.getInit());

    auto kernel = op.getKernel();
    int kh = kernel[0];
    int kw = kernel[1];

    assert(kh == 1 && kw != 1 && "Expected kh=1 and kw!=1 for horizontal 1D maxpool");

    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    Slice slice("MaxPool1DHorizontal");

    auto stride = op.getStride();
    assert(
        kw == stride[1] && "Kernel width (kw) must be equal to stride[1] for horizontal MaxPool1D"
    );

    // kh==1 and strideH==1, so each output row maps directly to the corresponding input row.
    // Keep the native N,C,H,W layout and only split W into [outputWidth, kw] tiles.
    // TODO: This path reduces along the innermost W dimension, so it cannot leverage vectorized
    // reduction like the vertical 1D case. For larger tensors, consider swapping H/W first and
    // reusing the faster H-reduction path.
    input.reshapeDim(Dim::W, {-1, kw});

    int in_ch = input.dim(In::C);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch_block = slice.iterate(in_ch)) {
            For(auto op_row = slice.iterate(input.dim(In::H))) {
                For(auto op_col = slice.iterate(input.dim(In::W))) {
                    PData pdata;
                    IData idata = slice.iram.load(input[batch][ch_block][op_row][op_col]);
                    For(auto kw_idx = slice.iterate(kw)) {
                        pdata = slice.alu.accumulate(idata[kw_idx], torq_hw::ALUOp1Mode::MAX);
                    }
                    QData res = slice.act.load(pdata);
                    slice.store(output[batch][ch_block][op_row][op_col], res);
                }
            }
        }
    }

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

static torq_hw::SliceTaskOp lowerMaxPool2DToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    // Layout of the in/out tensors for processing
    struct In {
        enum { N, C, IVectors, KernelRows };
    };

    if (!hasEkLoweringMaxPool(op)) {
        return {};
    }

    LData input(op.getInput());
    LData output(op.getInit());

    auto kernel = op.getKernel();
    int kh = kernel[0];
    int kw = kernel[1];

    int inputWidth = input.dim(Dim::W);
    int inputHeight = input.dim(Dim::H);

    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    Slice slice("MaxPool2D");

    HWDim kernelDim(kh, kw);
    LRTBDim pad(op.getPad());
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setInputChannelShape(inputHeight, inputWidth);

    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType());
    int vectSize = vectStride + std::min(kernelBorder.left + kernelBorder.right, alukw - 1);
    input.fuse({Dim::H, Dim::W}).vectorize(vectSize, vectStride);

    // Add additional dimension to scan over the kernelDim.h input rows
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(kernelDim.h, Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);

    int inCh = input.dim(In::C);

    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto cv = slice.iterate(inCh)) {
            For(auto iv = slice.iterate(input.dim(In::IVectors))) {
                PData pdata;
                For(auto kh_idx = slice.iterate(kh)) {
                    IData idata = slice.iram.load(input[n][cv][iv][kh_idx]);
                    For(auto kw_idx = slice.iterate(kw)) {
                        idata.setShape({{kw, Stride(1)}, vectStride});
                        pdata = slice.alu.accumulate(idata[kw_idx], torq_hw::ALUOp1Mode::MAX);
                    }
                }

                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.load(pdata[av]);
                    slice.append(output[n][cv], res);
                }
            }
        }
    }

    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

static torq_hw::SliceTaskOp lowerMaxPool2DStride2ToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {

    struct In {
        enum { N, C, RowQuadrant, ColQuadrant, Kh, IVectors };
    };

    if (!hasEkLoweringMaxPool(op)) {
        return {};
    }

    LData input(op.getInput());
    LData output(op.getInit());

    auto kernel = op.getKernel();
    int kh = kernel[0];
    int kw = kernel[1];

    int inputWidth = input.dim(Dim::W);
    int inputHeight = input.dim(Dim::H);

    // auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    // auto input_strides = getEncodedStridesElements(input_type);

    // const int data_part_size = static_cast<int>(input_strides[1]) / 4;
    const int data_part_size = static_cast<int>(inputWidth * inputHeight) / 4;

    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    Slice slice("MaxPool2D");

    HWDim kernelDim(kh, kw);
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    slice.setKernel(kernelBorder);

    LRTBDim pad(op.getPad());
    // For 2x2 (and generally when pad == (kw-1)/2), stride_offset should be 0
    // because the kernel tiles evenly without needing an offset. For other kernels
    // (e.g. 3x3 with same padding), stride_offset=1 is required.
    int strideOffset = (pad.left == pad.top && pad.left == (kw - 1) / 2) ? 0 : 1;
    slice.setStrideOffset(strideOffset);
    slice.setPadding(pad, op.getInputZp());

    int outputWidth = output.dim(Dim::W);
    int outputHeight = output.dim(Dim::H);
    slice.setInputChannelShape(outputHeight, outputWidth);
    slice.setStride(2);

    const int alukw = slice.alu.kerWidth();

    int vectStride = std::min(slice.alu.iWidth(input.elementType()), data_part_size);

    int vectSize = vectStride + std::min(kernelBorder.left, alukw - 1);

    input.fuse({Dim::H, Dim::W}).reshapeDim(Dim::H, {2, 2, -1}).vectorize(vectSize, vectStride);

    input.getShape()[In::RowQuadrant].tag = ShapeItem::Tag::KernelRows;

    int inCh = input.dim(In::C);

    ShapeItem KhstrideDim(div_ceil(kh, 2), output.dim(Dim::W), ShapeItem::Tag::KernelRows);
    input.insertDim(In::Kh, KhstrideDim);

    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto cv = slice.iterate(inCh)) {
            For(auto iv = slice.iterate(input.dim(In::IVectors))) {
                PData pdata;
                For(auto kh_idx = slice.iterate(div_ceil(kh, 2))) {
                    For(auto qr = slice.iterate(input.dim(In::RowQuadrant))) {
                        For(auto qc = slice.iterate(input.dim(In::ColQuadrant))) {
                            IData idata = slice.iram.load(input[n][cv][qr][qc][kh_idx][iv]);
                            For(auto kw_idx = slice.iterate(div_ceil(kw, 2))) {
                                idata.setShape({{kw, Stride(1)}, vectStride});
                                pdata =
                                    slice.alu.accumulate(idata[kw_idx], torq_hw::ALUOp1Mode::MAX);
                            }
                        }
                    }
                }
                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.load(pdata[av]);
                    slice.append(output[n][cv], res);
                }
            }
        }
    }

    torq_hw::Ndls ndls = slice.getNdls();

    if (auto *dedr = ndls.getMemNdl(torq_hw::NdlType::DEDR)) {
        dedr->offset = 0;
    }

    return torq_hw::SliceTaskOp::create(
        rewriter, op.getLoc(), slice.name(), op.getInput(), op.getWeights(), op.getScaleBias(),
        taskInitTensor, slice.getCfgAttr(rewriter.getContext()), ndls
    );
}

LogicalResult convertToHw(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) {
    LData inputData(op.getInput());
    auto kernel = op.getKernel();
    auto stride = op.getStride();
    auto pad = op.getPad();

    // The reshape-based 1D lowerings only handle exact non-overlapping, unpadded windows.
    // Leave padded or tail-dropping cases to the generic fallback path.
    bool hasNoPadding = pad[0] == 0 && pad[1] == 0 && pad[2] == 0 && pad[3] == 0;
    bool isVertical1D = kernel[1] == 1 && stride[1] == 1 && kernel[0] != 1 &&
                        stride[0] == kernel[0] && hasNoPadding &&
                        inputData.dim(Dim::H) % kernel[0] == 0;
    bool isHorizontal1D = kernel[0] == 1 && stride[0] == 1 && kernel[1] != 1 &&
                          stride[1] == kernel[1] && hasNoPadding &&
                          inputData.dim(Dim::W) % kernel[1] == 0;

    Value initValue = op.getInit();
    LData outputData(op.getInit());

    int outChCount = outputData.dim(Dim::C);

    torq_hw::SliceTaskOp hwOp;

    if (isVertical1D) {
        LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 1D (vertical) case\n");
        if (!(hwOp = lowerMaxPool1DToHw(op, rewriter, initValue, 0, outChCount))) {
            return failure();
        }
    }
    else if (isHorizontal1D) {
        LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 1D (horizontal) case\n");
        if (!(hwOp = lowerMaxPool1DHorizontalToHw(op, rewriter, initValue, 0, outChCount))) {
            return failure();
        }
    }
    else {
        int strideH = stride[0];
        int strideW = stride[1];
        if (strideH == 2 || strideW == 2) {
            LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 2D stride-2 case\n");
            if (!(hwOp = lowerMaxPool2DStride2ToHw(op, rewriter, initValue, 0, outChCount))) {
                return failure();
            }
        }
        else {
            LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 2D case\n");
            if (!(hwOp = lowerMaxPool2DToHw(op, rewriter, initValue, 0, outChCount))) {
                return failure();
            }
        }
    }

    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
