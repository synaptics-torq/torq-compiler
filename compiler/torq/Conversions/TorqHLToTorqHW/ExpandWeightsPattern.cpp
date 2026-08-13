// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "torq-lower-torqhl-expand-weights"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Expand kernel for block-wise expand_weights without bias.
//
// This Kernel compute the following elementwise formula on the input
// packed tensor, per output channel:
//
//   output = (decompressed_tensor(on-fly) * scale)
//
static FailureOr<SliceTaskOp> createBlockExpandWithoutBias(
    torq_hl::ExpandWeightsOp op, PatternRewriter &rewriter, int blockSize
) {

    Slice slice("ExpandWeights-Block-Expand-no-bias");

    struct Weight : Vectorized {
        enum { Blocks, BlockLines };
    };

    LData scale(op.getScale());   // (H/32xW)
    LData weights(op.getInput()); // HxW
    LData output(op.getInit());   // HxW

    if (weights.shape().size() == 2 && output.shape().size() == 3 && output.dim(0) == 1) {
        output.eraseDim(0);
    }

    int vectorSize = slice.act.width(scale.elementType(), weights.elementType());
    weights.vectorize(vectorSize);
    scale.vectorize(vectorSize); // ((H/32xW) /8) x 8

    weights.reshapeDim(0, {-1, blockSize});
    output.reshapeDim(0, {-1, blockSize});

    For(auto block = slice.iterate(weights.dim(Weight::Blocks))) { // iterate over H/32 blocks
        For(auto wv =
                slice.iterate(weights.dim(Weight::Vectors))) { // iterate over vectors in block
            IData sdata = slice.iram.load(scale[block][wv]);
            For(auto blockRow = slice.iterate(weights.dim(Weight::BlockLines))) {
                WData wdata = slice.wram.load(weights[block][blockRow][wv], DType::bf16);
                PData pdata = slice.alu.elementwiseProductAccumulate(sdata, wdata);
                QData res = slice.act.clamp(pdata, op.getOutputMin(), op.getOutputMax());
                slice.append(output[block][blockRow], res);
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

// Expand kernel for block-wise expand_weights with bias.
//
// This Kernel compute the following elementwise formula on the input
// packed weights, per output channel:
//
//   output = (decompressed_weights(on-fly) * scale) - (inputZeroPoint * scale)
//
// Since inputZeroPoint*scale both are constant and has already been converted to bias, this can be
// simplified to:
//
//   output = (decompressed_weights(on-fly) * scale) + (-1 * bias)
static FailureOr<SliceTaskOp>
createBlockExpandWithBias(torq_hl::ExpandWeightsOp op, PatternRewriter &rewriter, int blockSize) {

    Slice slice("ExpandWeights-Block-Expand-with-bias");

    struct Weight : Vectorized {
        enum { Blocks, BlockLines };
    };

    // input is scale
    LData scale(op.getScale());   // (H/32xW)
    LData weights(op.getInput()); // HxW
    LData output(op.getInit());   // HxW
    LData bias(op.getBias());     // HxW

    if (weights.shape().size() == 2 && output.shape().size() == 3 && output.dim(0) == 1) {
        output.eraseDim(0);
    }

    int vectorSize = slice.act.width(scale.elementType(), weights.elementType());
    weights.vectorize(vectorSize);
    scale.vectorize(vectorSize); // ((H/32xW) /8) x 8
    bias.vectorize(vectorSize);  // ((H/32xW) /8) x 8

    weights.reshapeDim(0, {-1, blockSize});
    output.reshapeDim(0, {-1, blockSize});
    bias.reshapeDim(-1, {-1, 1});

    For(auto block = slice.iterate(weights.dim(Weight::Blocks))) { // iterate over H/32 blocks
        For(auto wv =
                slice.iterate(weights.dim(Weight::Vectors))) { // iterate over vectors in block
            IData sdata = slice.iram.load(scale[block][wv]);
            BData bdata = slice.bram.load(bias[block][wv]);
            For(auto blockRow = slice.iterate(weights.dim(Weight::BlockLines))
            ) { // iterate over 32 lines per block
                WData wdata = slice.wram.load(weights[block][blockRow][wv], DType::bf16);
                PData pdata = slice.alu.elementwiseProductAccumulate(sdata, wdata);
                QData res = slice.act.rescaleClamp(
                    pdata, bdata, 0, 0, op.getOutputMin(), op.getOutputMax()
                );
                slice.append(output[block][blockRow], res);
            }
        }
    }

    return slice.createSliceTaskOp(rewriter, op.getLoc());
}

template <>
LogicalResult
ExpandWeightsPattern::transform(torq_hl::ExpandWeightsOp op, PatternRewriter &rewriter) const {

    int blockSize = op.getBlockSize();
    if (blockSize <= 0) {
        return op.emitOpError("expected positive block_size for expand_weights");
    }

    FailureOr<SliceTaskOp> expandWeightsTask =
        op.getBias() ? createBlockExpandWithBias(op, rewriter, blockSize)
                     : createBlockExpandWithoutBias(op, rewriter, blockSize);
    if (failed(expandWeightsTask)) {
        return failure();
    }

    rewriter.replaceOp(op, expandWeightsTask->getOperation());

    return success();
}

} // namespace mlir::syna::torq
