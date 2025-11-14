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

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult SelectPattern::transform(torq_hl::SelectOp op, PatternRewriter &rewriter) const {

    auto ctx = op.getContext();
    LData weights(op.getInput1());
    LData input(op.getInput2());
    LData input2(op.getInput3());
    LData output(op.getInit());

    assert(weights.dims() == input.dims() && input.dims() == input2.dims());

    // SelectOp doesn’t involve computation, so there is no such concepts as floating point or
    // integer. we keep the element type as integer type for simplicity.
    if (input.elementType() == DType::bf16) {
        input.setElementType(DType::int16);
        output.setElementType(DType::int16);
    }
    else if (input.elementType() == DType::fp32) {
        input.setElementType(DType::int32);
        output.setElementType(DType::int32);
    }

    // Dimensions of the input data for processing
    struct In : Vectorized {
        enum {
            InputTensors, // Selects between the two input tensors
            DataDim,      // First (non-dense) data dimension if any
        };
    };

    // Add an additional dimension of size 2 at the beginning to represent the 2 input tensors
    // The difference between the two input addresses is used as offset to fetch the data
    auto inputDiff = getAffineDimExpr(1, ctx) - getAffineDimExpr(0, ctx);
    input.insertDim(In::InputTensors, {2, inputDiff});

    int denseDims = std::min(input.denseDims(), output.denseDims());
    Slice slice;

    // Vectorize the input tensor
    int vectorSizeI = slice.alu.iWidth(input.elementType(), weights.elementType());
    int vectorSizeW = slice.alu.wWidth(weights.elementType());
    int vectorSize = std::min(vectorSizeI, vectorSizeW);

    input.fuse(denseDims).vectorize(vectorSize);
    weights.fuse(denseDims).vectorize(vectorSize);

    For(auto ii = slice.iterate(input.dims(In::DataDim, In::Vectors))) {
        For(auto dv = slice.iterate(input.dim(In::Vectors))) {
            PData pdata;
            WData wdata = slice.wram.load(weights[ii][dv]);
            For(auto i = slice.iterate(input.dim(In::InputTensors))) {
                IData idata = slice.iram.load(input[i][ii][dv]);
                pdata =
                    slice.alu.elementwiseProductAccumulate(idata, wdata, torq_hw::ALUOp1Mode::SEL);
            }
            For(auto a = slice.iterate(pdata.dim(0))) {
                QData res = slice.act.load(pdata[a]);
                slice.append(output[ii], res);
            }
        }
    }

    auto trueInputAddr = rewriter.create<GetAddressOp>(op.getLoc(), op.getInput2()).getAddress();
    auto falseInputAddr = rewriter.create<GetAddressOp>(op.getLoc(), op.getInput3()).getAddress();
    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                                         // Operation to replace
        "select",                                   // Task name
        ValueRange{op.getInput2(), op.getInput3()}, // Input tensor
        ValueRange{op.getInput1()},                 // Weights
        ValueRange{},                               // BiasScale tensor
        ValueRange{op.getInit()},                   // Output tensor initializer
        ValueRange{trueInputAddr, falseInputAddr},  // Symbols used to compute the NDLs
        slice.getCfgAttr(ctx),                      // Slice configuration
        slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
