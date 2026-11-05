// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-gather"

namespace mlir::syna::torq {

// Dimensions of the input and output tensors, before the kernel reshapes them to:
// - input: [N, C', i, j, H, blocks, blockItems] (CRD) or [N, i, j, C', H, blocks, blockItems] (DCR)
// - output: [N, C', H, i, 2 * W]
// with C' = C / 4, i and j the row and column in each 2x2 output block, and each input row of W
// items split into blocks of blockItems items.
using In = NCHW;
using Out = NCHW;

// Replace the last dim of data by blocks of blockSize items. The last block can go past the end.
static void splitLastDim(LData &data, int blocks, int blockSize) {
    Shape shape = data.shape();
    int stride = shape.back().stride.intVal.value();
    shape.back() = ShapeItem(blocks, Stride(blockSize * stride));
    shape.push_back(ShapeItem(blockSize, Stride(stride)));
    data.setShape(shape);
}

template <>
LogicalResult
DepthToSpacePattern::transform(torq_hl::DepthToSpaceOp op, PatternRewriter &rewriter) const {
    // Input [N, C, H, W], output [N, C / 4, H * 2, W * 2] with blocks of 2x2:
    // output[n][c][h * 2 + i][w * 2 + j] = input[n][channel(c, i, j)][h][w]
    LData input(op.getInput());
    LData output(op.getInit());
    LData pattern(op.getWeights());
    const int blockSize = op.getBlockSize();
    if (blockSize != 2 || input.shape().size() != 4) {
        return op.emitError("depth to space only supports 4D tensors and blocks of 2");
    }
    if (sizeofType(input.elementType()) > 2) {
        return op.emitError("depth to space only supports 8 and 16-bit elements");
    }

    // Each output row interleaves two input rows (j = 0, 1): each step multiplies a block of
    // blockItems items of row j, each repeated twice, with a pattern of 0s and 1s that keeps the
    // items at the odd (j = 1) or even (j = 0) positions, and accumulates the two rows.
    // A row of W items is covered by blocks of blockItems items. append() writes the results of a
    // block in whole ACT passes, so each block but the last one must fill whole passes. The last
    // block can go past the end of the row: append() drops the results that don't fit in the row.
    Slice slice("depthtospace");
    const int width = input.dim(In::W);
    const int maxBlockItems = slice.alu.wWidth(input.elementType()) / blockSize;
    const int passSize = slice.act.width(pattern.elementType(), input.elementType());
    assert(maxBlockItems * blockSize % passSize == 0 && "Blocks must fill whole ACT passes");
    const int blocks = div_ceil(width, maxBlockItems);
    const int blockItems = blocks == 1 ? width : maxBlockItems;

    // Input: [N, C', i, j, H, blocks, blockItems] (CRD) or [N, i, j, C', H, blocks, blockItems]
    const int batch = input.dim(In::N);
    const int height = input.dim(In::H);
    const int outChannels = output.dim(Out::C);
    const bool crd = op.getModeType() == torq_hl::DepthToSpaceModeEnum::CRD;
    if (crd) {
        input.reshapeDim(In::C, {outChannels, blockSize, blockSize});
    }
    else {
        input.reshapeDim(In::C, {blockSize, blockSize, outChannels});
    }
    splitLastDim(input, blocks, blockItems);
    auto inputBlock = [&](const Indexes &n, const Indexes &c, const Indexes &i, const Indexes &j,
                          const Indexes &h, const Indexes &b) {
        return crd ? input[n][c][i][j][h][b] : input[n][i][j][c][h][b];
    };

    // Output: [N, C', H, i, 2 * W]
    output.reshapeDim(Out::H, {height, blockSize});

    // Pattern: two rows of 0s and 1s, for j = 0 and j = 1
    pattern.reshapeDim(0, {blockSize, -1}).subviewDim(1, 0, blockItems * blockSize);

    For(auto n = slice.iterate(batch)) {
        For(auto c = slice.iterate(outChannels)) {
            For(auto h = slice.iterate(height)) {
                For(auto i = slice.iterate(blockSize)) {
                    For(auto b = slice.iterate(blocks)) {
                        PData pdata;
                        For(auto j = slice.iterate(blockSize)) {
                            WData wdata =
                                slice.wram.load(inputBlock(n, c, i, j, h, b)).repeat(blockSize);
                            IData idata = slice.iram.load(pattern[j]);
                            pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);
                        }
                        For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                            QData res = slice.act.load(pdata[av]);
                            slice.append(output[n][c][h][i], res);
                        }
                    }
                }
            }
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
