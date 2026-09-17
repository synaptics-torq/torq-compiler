// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-mul"

namespace mlir::syna::torq {

// input * scalar: the scalar weight is a single element that the ALU applies to a whole input
// vector natively, so it is loaded once and multiplied with scalarProductAccumulate.
static LogicalResult lowerScalarMul(
    torq_hl::MulOp op, PatternRewriter &rewriter, LData input, LData scalarWeight, LData biasScale,
    LData output, int shift, int outZp, int outMin, int outMax
) {
    Slice slice("mul");

    struct In : Vectorized {
        enum { NonDenseDims };
    };

    input.broadcastAs(output);

    // Multiple bias/scale records form a per-last-dimension table. For a rank-2 multiply, each
    // record belongs to one output column and the sequence is reused for every row. The ACT
    // applies one record per item, so this dimension cannot be fused and BRAM reloads per vector.
    const int recordWidth = biasScaleWidth(biasScale.elementType());
    const bool perLastDimBias = elementCount(biasScale.shape()) > recordWidth;

    int denseDims = std::min(input.denseDims(), output.denseDims());
    if (perLastDimBias)
        denseDims = std::min(denseDims, 1);

    int vectorSize =
        slice.act.width(input.elementType(), scalarWeight.elementType(), perLastDimBias);
    input.fuse(denseDims).vectorize(vectorSize);
    output.fuse(denseDims);

    std::optional<BData> bdata;
    if (perLastDimBias) {
        const int recordCount = elementCount(biasScale.shape()) / recordWidth;
        assert(recordCount == output.dim(-1) && "bias table must match the output's last dim");
        biasScale.forceReshapeDim(
            -1, {(int)div_ceil(recordCount, vectorSize), vectorSize, recordWidth}
        );
    }
    else {
        bdata = slice.bram.load(biasScale);
    }

    WData scalar = slice.wram.load(scalarWeight[Indexes(scalarWeight.shape().size(), 0)]);

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto i = slice.iterate(input.dim(In::Vectors))) {
            IData data1 = slice.iram.load(input[ndd][i]);
            PData pdata = slice.alu.scalarProductAccumulate(data1, scalar);
            // BRAM holds at most the current vector's records, so a per-last-dimension table is
            // reloaded every iteration; a single {bias,scale} pair was loaded once outside.
            BData vectorBias = perLastDimBias ? slice.bram.load(biasScale[i]) : *bdata;
            QData res = slice.act.rescaleClamp(pdata, vectorBias, shift, outZp, outMin, outMax);
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

// input1 * input2: both operands are vectorized and the second is loaded per vector into WRAM,
// so each pair is multiplied with elementwiseProductAccumulate.
static LogicalResult lowerElementwiseMul(
    torq_hl::MulOp op, PatternRewriter &rewriter, LData input1, LData input2, LData biasScale,
    LData output, int shift, int outZp, int outMin, int outMax
) {
    Slice slice("mul");

    struct In : Vectorized {
        enum { NonDenseDims };
    };

    input1.broadcastAs(output);
    input2.broadcastAs(output);

    // Multiple bias/scale records form a per-last-dimension table. For a rank-2 multiply, each
    // record belongs to one output column and the sequence is reused for every row. The ACT
    // applies one record per item, so this dimension cannot be fused and BRAM reloads per vector.
    const int recordWidth = biasScaleWidth(biasScale.elementType());
    const bool perLastDimBias = elementCount(biasScale.shape()) > recordWidth;

    int inDenseDims = std::min(input1.denseDims(), input2.denseDims());
    int denseDims = std::min(inDenseDims, output.denseDims());
    if (perLastDimBias)
        denseDims = std::min(denseDims, 1);

    int vectorSize = slice.act.width(input1.elementType(), input2.elementType(), perLastDimBias);
    input1.fuse(denseDims).vectorize(vectorSize);
    input2.fuse(denseDims).vectorize(vectorSize);
    output.fuse(denseDims);

    std::optional<BData> bdata;
    if (perLastDimBias) {
        const int recordCount = elementCount(biasScale.shape()) / recordWidth;
        assert(recordCount == output.dim(-1) && "bias table must match the output's last dim");
        biasScale.forceReshapeDim(
            -1, {(int)div_ceil(recordCount, vectorSize), vectorSize, recordWidth}
        );
    }
    else {
        bdata = slice.bram.load(biasScale);
    }

    For(auto ndd = slice.iterate(input1.dims(In::NonDenseDims, In::Vectors))) {
        For(auto i = slice.iterate(input1.dim(In::Vectors))) {
            IData data1 = slice.iram.load(input1[ndd][i]);
            PData pdata =
                slice.alu.elementwiseProductAccumulate(data1, slice.wram.load(input2[ndd][i]));
            // BRAM holds at most the current vector's records, so a per-last-dimension table is
            // reloaded every iteration; a single {bias,scale} pair was loaded once outside.
            BData vectorBias = perLastDimBias ? slice.bram.load(biasScale[i]) : *bdata;
            QData res = slice.act.rescaleClamp(pdata, vectorBias, shift, outZp, outMin, outMax);
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

template <>
LogicalResult MulPattern::transform(torq_hl::MulOp op, PatternRewriter &rewriter) const {
    LData input1(op.getInput1());
    LData input2(op.getInput2());
    LData biasScale(op.getScaleBias());
    LData output(op.getInit());

    const int shift = op.getShift();
    const int outMin = op.getOutputMin();
    const int outMax = op.getOutputMax();
    const int outZp = op.getOutputZp();

    // A single-element operand is a scalar multiplier, which the ALU applies to a whole
    // input vector natively.
    // The scalar has to be the weight operand, so swap when it arrives as input1;
    // the product commutes.
    bool scalarWeight = elementCount(output.shape()) > 1 && elementCount(input2.shape()) == 1;
    if (!scalarWeight && elementCount(input1.shape()) == 1 &&
        input1.elementType() == input2.elementType()) {
        std::swap(input1, input2);
        scalarWeight = elementCount(output.shape()) > 1;
    }

    if (scalarWeight)
        return lowerScalarMul(
            op, rewriter, input1, input2, biasScale, output, shift, outZp, outMin, outMax
        );
    return lowerElementwiseMul(
        op, rewriter, input1, input2, biasScale, output, shift, outZp, outMin, outMax
    );
}

} // namespace mlir::syna::torq
