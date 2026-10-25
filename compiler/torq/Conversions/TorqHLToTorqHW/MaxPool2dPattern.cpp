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

// Load the partials clamped to [output_min, output_max] of the maxpool, within the range of the
// partials type: a quantized maxpool clamps to the range of its quantize (e.g. [-128, 100]), which
// can be narrower than the data type
static QData loadClamped(Slice &slice, const PData &pdata, torq_hl::MaxPool2dOp op) {
    DType type = pdata.elementType();
    if (!isInt(type)) {
        return slice.act.load(pdata);
    }
    int32_t clipMin = std::max<int32_t>(op.getOutputMin(), minVal(type));
    int32_t clipMax = std::min<int32_t>(op.getOutputMax(), maxVal(type));
    return slice.act.clamp(pdata, clipMin, clipMax);
}

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

    // Each output row is appended separately, and the ACT part of the last (partial) vector of a
    // row beyond the first ACT width gives wrong results (e.g. bf16, whose ALU vectors of 32 are
    // read in 2 ACT vectors of 16, with rows of 18 or 24 elements): use vectors of the ACT width
    // in that case
    int vectSize = slice.alu.iWidth(input.elementType());
    int actWidth = slice.act.width(input.elementType());
    if (input.dim(Dim::W) % vectSize > actWidth) {
        vectSize = actWidth;
    }
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
                        QData res = loadClamped(slice, pdata[av], op);
                        slice.append(output[batch][ch_block][op_row], res);
                    }
                }
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
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
    // Keep the native N,C,H,W layout and split W into [outputWidth, nOuterGroups, tile] tiles.
    //
    // The innermost `tile` is read as the strided 'S' dimension of a single IRAM load, which the
    // hardware descriptor limits to alu.kerWidth(). A wide window (e.g. kw=10) would otherwise emit
    // an S dimension of kw and trip the `dims.sn<=4` assertion in the HW descriptor generator.
    // Picking `tile` as the largest divisor of kw not exceeding that limit keeps each load legal
    // (and avoids over-reading past the window), while the remaining `nOuterGroups` factor is
    // reduced with an outer loop of separate loads -- max is associative, so the per-group partial
    // maxima combine to the same result.
    // TODO: This path reduces along the innermost W dimension, so it cannot leverage vectorized
    // reduction like the vertical 1D case. For larger tensors, consider swapping H/W first and
    // reusing the faster H-reduction path.
    const int maxSDim = slice.alu.kerWidth();
    int tile = 1;
    for (int t = std::min(kw, maxSDim); t >= 1; --t) {
        if (kw % t == 0) {
            tile = t;
            break;
        }
    }
    int nOuterGroups = kw / tile;

    input.reshapeDim(Dim::W, {-1, nOuterGroups, tile});

    int in_ch = input.dim(In::C);

    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch_block = slice.iterate(in_ch)) {
            For(auto op_row = slice.iterate(input.dim(In::H))) {
                For(auto op_col = slice.iterate(input.dim(In::W))) {
                    PData pdata;
                    For(auto kw_group = slice.iterate(nOuterGroups)) {
                        IData idata =
                            slice.iram.load(input[batch][ch_block][op_row][op_col][kw_group]);
                        For(auto kw_idx = slice.iterate(tile)) {
                            pdata = slice.alu.accumulate(idata[kw_idx], torq_hw::ALUOp1Mode::MAX);
                        }
                    }
                    QData res = loadClamped(slice, pdata, op);
                    slice.store(output[batch][ch_block][op_row][op_col], res);
                }
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

static torq_hw::SliceTaskOp lowerMaxPool2DToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    // Layout of the in/out tensors for processing
    struct In {
        enum { N, C, IVectors, KernelRows, KernelColGroups };
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
    // The window of output i covers the input [i - pad.top, i - pad.top + kh - 1], and the HW
    // requires the leading padding to be 0 or the leading kernel border. The symmetric border of an
    // even kernel has the larger half after the center ((k - 1) / 2 before, k / 2 after): swap the
    // halves when the leading padding is k / 2 (e.g. 2x2 with padding 1 then 0, a SAME_LOWER
    // padding)
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    if (pad.top > kernelBorder.top) {
        std::swap(kernelBorder.top, kernelBorder.bottom);
    }
    if (pad.left > kernelBorder.left) {
        std::swap(kernelBorder.left, kernelBorder.right);
    }
    slice.setKernel(kernelBorder);
    slice.setPadding(pad, op.getInputZp());
    slice.setOutputChannelShape(inputHeight, inputWidth);

    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType());
    int vectSize = vectStride + std::min(kernelBorder.left + kernelBorder.right, alukw - 1);
    input.fuse({Dim::H, Dim::W}).vectorize(vectSize, vectStride);

    // Add additional dimension to scan over the kernelDim.h input rows
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(kernelDim.h, Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);

    // Add additional dimension to scan over the kernelDim.w/alukw input column groups:
    // one iram load covers at most alukw kernel columns
    ShapeItem colGroupsDim(div_ceil(kernelDim.w, alukw), Stride(alukw), ShapeItem::Tag::KernelCols);
    input.insertDim(In::KernelColGroups, colGroupsDim);

    int inCh = input.dim(In::C);

    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto cv = slice.iterate(inCh)) {
            For(auto iv = slice.iterate(input.dim(In::IVectors))) {
                PData pdata;
                For(auto kh_idx = slice.iterate(kh)) {
                    For(auto kw_idx = slice.iterate(kw)) {
                        IData idata = slice.iram.load(input[n][cv][iv][kh_idx][kw_idx / alukw]);
                        idata.setShape({{alukw, Stride(1)}, vectStride});
                        pdata =
                            slice.alu.accumulate(idata[kw_idx % alukw], torq_hw::ALUOp1Mode::MAX);
                    }
                }

                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = loadClamped(slice, pdata[av], op);
                    slice.append(output[n][cv], res);
                }
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

// Same structure as the depthwise stride-2 kernel (lowerDwStride2ToHw in DWToHw.cpp) with a MAX
// accumulation instead of the products: the spatial domain is decomposed into 4 quadrants
// (EE/EO/OE/OO) and the kernel iterates over both quadrants and kernel positions.
static torq_hw::SliceTaskOp lowerMaxPool2DStride2ToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    struct In : Vectorized {
        enum { N, C, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps };
    };

    if (!hasEkLoweringMaxPool(op)) {
        return {};
    }

    LData input(op.getInput());
    LData output(op.getInit());

    auto kernel = op.getKernel();
    HWDim kernelDim(kernel[0], kernel[1]);
    int stride = op.getStride()[0];

    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    Slice slice("MaxPool2D");
    LRTBDim kernelBorder = LRTBDim::symmetric(kernelDim);
    slice.setKernel(kernelBorder);

    // For 2x2 (and generally when pad == (kw-1)/2), stride_offset should be 0
    // because the kernel tiles evenly without needing an offset. For other kernels
    // (e.g. 3x3 with same padding), stride_offset=1 is required.
    LRTBDim pad(op.getPad());
    int strideOffset =
        (pad.left == (kernelDim.w - 1) / stride && pad.top == (kernelDim.h - 1) / stride) ? 0 : 1;
    slice.setStrideOffset(strideOffset);
    slice.setStride(stride);

    slice.setOutputChannelShape(
        std::max<int>(1, output.dim(Dim::H)), std::max<int>(1, output.dim(Dim::W))
    );

    // Stride-2 KW/KH>1 integer kernels need both L/R pad enables for the last odd
    // column, even when the op's left pad is 0 (see lowerDwStride2ToHw).
    if ((kernelDim.w > 1 || kernelDim.h > 1) && !isFloat(input.elementType())) {
        pad.left = 1;
        pad.right = 1;
    }
    if (pad.top) {
        pad.top = kernelBorder.top;
    }
    if (pad.bottom) {
        pad.bottom = kernelBorder.bottom;
    }
    slice.setPadding(pad, op.getInputZp());

    // Vectorize input
    const int alukw = slice.alu.kerWidth();
    int vectStride = slice.alu.iWidth(input.elementType());
    int vectSize = vectStride + std::min(kernelBorder.left, alukw - 1);
    input.fuse({Dim::H, Dim::W}).reshapeDim(Dim::H, {2, 2, -1}).vectorize(vectSize, vectStride);
    // Shape: [N, C, RowQuadrant, ColQuadrant, Vectors, vectSize]

    // Add additional dimension to scan over the kernelDim.h input rows; stride-2 halves the
    // number of row steps because each quadrant step advances by 2 input rows.
    int rowSize = output.dim(Dim::W);
    ShapeItem rowsDim(div_ceil(kernelDim.h, 2), Stride(rowSize), ShapeItem::Tag::KernelRows);
    input.insertDim(In::KernelRows, rowsDim);

    // Add additional dimension to scan over the kernel column groups; halved by stride then
    // grouped by alukw
    ShapeItem colGroupsDim(
        div_ceil(div_ceil(kernelDim.w, 2), alukw), Stride(alukw), ShapeItem::Tag::KernelCols
    );
    input.insertDim(In::KernelColGrps, colGroupsDim);
    // Shape: [N, C, RowQuadrant, ColQuadrant, KernelRows, KernelColGrps, Vectors, vectSize]

    // Tag quadrants since the HW needs to be aware
    input.getShape()[In::RowQuadrant].tag = ShapeItem::Tag::KernelRows;
    input.getShape()[In::ColQuadrant].tag = ShapeItem::Tag::KernelCols;
    if (kernelDim.h == 1) {
        input.getShape()[In::RowQuadrant].count = 1;
    }

    // If start_pos we have to iterate the corresponding quadrant in reverse order,
    // but the starting offset and stride must be tweaked so we can't use the standard reverse()
    // method. Instead we manually adjust the starting offset and stride of the quadrant dimensions.
    const int32_t start_pos_x = (-kernelBorder.left + strideOffset) & 1;
    const int32_t start_pos_y = (kernelDim.h == 1) ? 0 : ((-kernelBorder.top + strideOffset) & 1);
    const int32_t kernel_left_even = (kernelBorder.left - strideOffset + 1) >> 1;
    const int32_t kernel_left_odd = kernelBorder.left - strideOffset - kernel_left_even;
    const int32_t kernel_top_even = (kernelBorder.top - strideOffset + 1) >> 1;
    const int32_t kernel_top_odd = kernelBorder.top - strideOffset - kernel_top_even;
    const int32_t colQStride = input.shape()[In::ColQuadrant].stride.intVal.value();
    const int32_t rowQStride = input.shape()[In::RowQuadrant].stride.intVal.value();
    input.setOffset(input.offset() + (2 * start_pos_y + start_pos_x) * colQStride);
    input.getShape()[In::ColQuadrant].stride =
        colQStride * (start_pos_x == 0 ? 1 : -1) + kernel_left_even - kernel_left_odd;
    if (kernelDim.h == 1) {
        input.getShape()[In::RowQuadrant].stride = rowQStride;
    }
    else {
        input.getShape()[In::RowQuadrant].stride =
            rowQStride * (start_pos_y == 0 ? 1 : -1) + rowSize * (kernel_top_even - kernel_top_odd);
    }

    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto c = slice.iterate(input.dim(In::C))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto kh = slice.iterate(kernelDim.h)) {
                    For(auto qr = slice.iterate(input.dim(In::RowQuadrant))) {
                        For(auto qc = slice.iterate(input.dim(In::ColQuadrant))) {
                            For(auto kw = slice.iterate(kernelDim.w)) {
                                IData idata = slice.iram.load(
                                    input[n][c][qr][qc][kh / 2][kw / (alukw * 2)][iv]
                                );
                                idata.setShape({{alukw, Stride(1)}, vectStride});
                                pdata =
                                    slice.alu.accumulate(idata[kw % 2], torq_hw::ALUOp1Mode::MAX);
                            }
                        }
                    }
                }
                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = loadClamped(slice, pdata[av], op);
                    slice.append(output[n][c], res);
                }
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

// 1x1 stride-2 maxpool: a subsampling of the input. The input of a stride-2 maxpool is segmented in
// 4 quadrants (even rows and columns first, see KernelSelectionPass and lowerMaxPool2DStride2ToHw),
// so the result is the first quadrant of each channel: copy it. The stride-2 kernel doesn't support
// 1x1 kernels (the NPU hangs).
static torq_hw::SliceTaskOp lowerMaxPool1x1Stride2ToHw(
    torq_hl::MaxPool2dOp op, PatternRewriter &rewriter, Value taskInitTensor, int chOffset,
    int chCount
) {
    struct In : Vectorized {
        enum { N, C };
    };
    LData input(op.getInput());
    LData output(op.getInit());
    input.subviewDim(Dim::C, chOffset, chCount);
    output.subviewDim(Dim::C, chOffset, chCount);

    int quadrantSize = input.dim(Dim::H) * input.dim(Dim::W) / 4;
    if (input.dim(Dim::H) % 2 || input.dim(Dim::W) % 2 ||
        quadrantSize != output.dim(Dim::H) * output.dim(Dim::W)) {
        return {};
    }

    Slice slice("MaxPool2D");
    // Not accumulating: vectors of the ACT width
    int vectSize = slice.act.width(input.elementType());
    input.fuse({Dim::H, Dim::W}).subviewDim(Dim::H, 0, quadrantSize).vectorize(vectSize);

    For(auto n = slice.iterate(input.dim(In::N))) {
        For(auto c = slice.iterate(input.dim(In::C))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                IData idata = slice.iram.load(input[n][c][iv]);
                PData pdata = slice.alu.load(idata);
                QData res = loadClamped(slice, pdata, op);
                slice.append(output[n][c], res);
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

template <>
LogicalResult
MaxPool2dPattern::transform(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) const {
    LData inputData(op.getInput());
    auto kernel = op.getKernel();
    auto stride = op.getStride();
    auto pad = op.getPad();

    MaxPoolLowering lowering =
        getMaxPoolLowering(kernel, stride, pad, inputData.dim(Dim::H), inputData.dim(Dim::W));

    Value initValue = op.getInit();
    LData outputData(op.getInit());

    int outChCount = outputData.dim(Dim::C);

    torq_hw::SliceTaskOp hwOp;

    if (lowering == MaxPoolLowering::Vertical1D) {
        LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 1D (vertical) case\n");
        if (!(hwOp = lowerMaxPool1DToHw(op, rewriter, initValue, 0, outChCount))) {
            return failure();
        }
    }
    else if (lowering == MaxPoolLowering::Horizontal1D) {
        LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 1D (horizontal) case\n");
        if (!(hwOp = lowerMaxPool1DHorizontalToHw(op, rewriter, initValue, 0, outChCount))) {
            return failure();
        }
    }
    else {
        if (lowering == MaxPoolLowering::None || !hasEkLoweringMaxPool(op)) {
            return op.emitError("maxpool with stride [")
                   << stride[0] << "," << stride[1] << "] and kernel [" << kernel[0] << ","
                   << kernel[1] << "] is not supported on the NSS (strides must be equal and "
                   << "at most 2, kernels at most " << int(HwInfo::nss_max_kernel_height) << "x"
                   << int(HwInfo::nss_max_kernel_width) << ")";
        }
        if (stride[0] == 2 && kernel[0] == 1 && kernel[1] == 1) {
            LLVM_DEBUG(llvm::dbgs() << "Lowering MaxPool 1x1 stride-2 case\n");
            if (!(hwOp = lowerMaxPool1x1Stride2ToHw(op, rewriter, initValue, 0, outChCount))) {
                return op.emitError("1x1 stride-2 maxpool needs an even input height and width");
            }
        }
        else if (stride[0] == 2) {
            // A single stride_offset cannot encode different H and W window phases (e.g.
            // pad=[1,0,0,1] from H-tiling an odd-height pool) and would corrupt silently;
            // ValidToSamePadPass normalizes these, so fail loudly if one slips through.
            if ((pad[0] != (kernel[1] - 1) / 2) != (pad[2] != (kernel[0] - 1) / 2)) {
                return op.emitError("stride-2 maxpool with mixed H/W pad phase (pad=[")
                       << pad[0] << "," << pad[1] << "," << pad[2] << "," << pad[3]
                       << "]) is not lowerable; ValidToSamePadPass should have normalized it";
            }
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
