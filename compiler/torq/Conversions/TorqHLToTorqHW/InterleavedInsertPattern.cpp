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

#define DEBUG_TYPE "torq-lower-torqhl-interleaved-insert"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Layout of the input tensor
struct In : NCHW, Vectorized {};

// Layout of the output tensor with explicit stride dimension
struct Out {
    enum { N, C, H, Stride, W };
};

// Weight tensor layout: Simple 1D array [stride_value]
// For stride-2: [1, 0] means write data at pos 0, write zero at pos 1
// No struct needed - weight is indexed directly as weight[stride_pos]

// Lower torq_hl::InterleavedInsertOp to SliceTaskOp
//
// InterleavedInsert is used for transposed convolution upsampling where input rows
// are inserted into output with stride > 1, creating gaps (usually filled with zeros).
//
// For stride-2 example, we interleave entire lines (rows), not individual values:
//   Input channel:      Output channel:
//   1 2 3 4             1 2 3 4
//   5 6 7 8       =>    0 0 0 0
//                       5 6 7 8
//                       0 0 0 0
//
// The weights tensor contains the interleaving pattern (e.g., [1, 0] for stride-2)
// which determines which positions get data vs zeros.
static torq_hw::SliceTaskOp
lowerToHw(torq_hl::InterleavedInsertOp op, PatternRewriter &rewriter, Value taskInitTensor) {
    // Define operands in LRAM
    LData input(op.getInput());
    LData output(op.getInit());
    LData weight(op.getWeights());

    // Validate that input and output tensors are 4D (NCHW format)
    assert(input.dims().size() == 4 && "Expected 4D input tensor");
    assert(output.dims().size() == 4 && "Expected 4D output tensor");

    // Weight is a simple 1D pattern array [0, 1] or [1, 0] for stride-2
    // No need for dimension manipulation like convolution weights

    // Get stride value for interleaving (typically 2 for stride-2 upsampling)
    int strideValue = op.getStrideValue();

    // Configure slice parameters
    Slice slice("InterleavedInsert");

    // Vectorize input for processing
    // Deduce vector size from ALU width to automatically adapt to data type
    // Since we want to handle each line independently, vectorize only the last dimension (W)
    int vectSize = slice.alu.iWidth(input.elementType(), weight.elementType());
    input.vectorize(vectSize);

    // Add an explicit stride dimension to the output after H dimension
    // This allows us to explicitly index into stride positions (e.g., [0, 1] for stride-2)
    // output.insertDim(Out::Stride, {strideValue});
    output.reshapeDim(Out::H, {output.dim(Out::H) / strideValue, strideValue}, true);

    // Main processing loops
    // InterleavedInsert performs data reorganization with stride-based interleaving
    // Weights contain interleaving pattern (e.g., [1,0] or [0,1] for stride-2)
    For(auto batch = slice.iterate(input.dim(In::N))) {
        For(auto ch = slice.iterate(input.dim(In::C))) {
            For(auto h = slice.iterate(input.dim(In::H))) {
                For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                    // Load input data to IRAM
                    IData idata = slice.iram.load(input[batch][ch][h][iv]);

                    // For each stride position (e.g., 0 and 1 for stride-2)
                    For(auto stride_pos = slice.iterate(strideValue)) {
                        // Load interleaving pattern weight for this stride position
                        // Weight is typically [1,0] or [0,1] to select even/odd positions
                        WData wdata = slice.wram.load(weight[stride_pos]);

                        // Multiply input with pattern weight (element-wise)
                        // This selects which data to write (1) or skip (0)
                        PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);

                        For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                            // Clamp output values to specified range
                            QData res =
                                slice.act.clamp(pdata[av], op.getOutputMin(), op.getOutputMax());
                            // Store to output with explicit stride position indexing
                            slice.append(output[batch][ch][h][stride_pos], res);
                        }
                    }
                }
            }
        }
    }

    return rewriter.create<torq_hw::SliceTaskOp>(
        op.getLoc(), slice.name(), op.getInput(), op.getWeights(), Value(), taskInitTensor,
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
}

template <>
LogicalResult InterleavedInsertPattern::transform(
    torq_hl::InterleavedInsertOp op, PatternRewriter &rewriter
) const {
    // InterleavedInsert processes all channels uniformly - no channel peeling needed
    // The operation applies the same interleaving pattern across all channels
    torq_hw::SliceTaskOp hwOp = lowerToHw(op, rewriter, op.getInit());

    if (!hwOp) {
        return failure();
    }

    rewriter.replaceOp(op, hwOp.getOperation()->getResults());
    return success();
}

} // namespace mlir::syna::torq
