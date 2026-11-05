// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

template <>
LogicalResult SelectPattern::transform(torq_hl::SelectOp op, PatternRewriter &rewriter) const {

    LData weights(op.getInput1());
    LData input1(op.getInput2());
    LData input2(op.getInput3());
    LData output(op.getInit());

    // Handle implicit broadcasting (weights and inputs must have the same rank as output)
    weights.broadcastAs(output);
    input1.broadcastAs(output);
    input2.broadcastAs(output);

    if (isFloat(input1.elementType())) {
        // ALU doen't support select on floating point, cast to int of the same size
        DType intType = getIntDType(sizeofTypeBits(input1.elementType()));
        input1.bitCast(intType);
        input2.bitCast(intType);
        output.bitCast(intType);
    }

    // Dimensions of the input data for processing
    struct In : Vectorized {
        enum {
            InputTensors, // Selects between the two input tensors
            NonDenseDims, // First (non-dense) data dimension if any
        };
    };

    int denseInDims = std::min(input1.denseDims(), input2.denseDims());
    int denseOutDims = output.denseDims();
    int denseDims = std::min(denseInDims, denseOutDims);

    // Vectorize the input tensor
    Slice slice("Select");
    int vectorSizeI = slice.alu.iWidth(input1.elementType(), weights.elementType());
    int vectorSizeW = slice.alu.wWidth(weights.elementType());
    int vectorSize = std::min(vectorSizeI, vectorSizeW);
    input1.fuse(denseDims).vectorize(vectorSize);
    input2.fuse(denseDims).vectorize(vectorSize);
    auto input = stack(rewriter, input1, input2);
    weights.fuse(denseDims).vectorize(vectorSize);

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            PData pdata;
            WData wdata = slice.wram.load(weights[ndd][iv]);
            For(auto it = slice.iterate(input.dim(In::InputTensors))) {
                IData idata = slice.iram.load(input[it][ndd][iv]);
                pdata = slice.alu.elementwiseProductAccumulate(idata, wdata, Alu::Mode::select);
            }
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                QData res = slice.act.load(pdata[av]);
                slice.append(output[ndd], res);
            }
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
