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

static Alu::Mode opTypeToAluOpMode(torq_hl::ElementwiseOpEnum opType) {
    switch (opType) {
    case torq_hl::ElementwiseOpEnum::BITWISE_AND:
        return Alu::Mode::bitAnd;
    case torq_hl::ElementwiseOpEnum::BITWISE_OR:
        return Alu::Mode::bitOr;
    case torq_hl::ElementwiseOpEnum::BITWISE_XOR:
        return Alu::Mode::bitXor;
    case torq_hl::ElementwiseOpEnum::MINIMUM:
        return Alu::Mode::min;
    case torq_hl::ElementwiseOpEnum::MAXIMUM:
        return Alu::Mode::max;
    case torq_hl::ElementwiseOpEnum::GREATER:
        return Alu::Mode::gt;
    case torq_hl::ElementwiseOpEnum::GREATER_EQUAL:
        return Alu::Mode::ge;
    case torq_hl::ElementwiseOpEnum::EQUAL:
        return Alu::Mode::eq;
    case torq_hl::ElementwiseOpEnum::LOGICAL_AND:
        return Alu::Mode::logicalAnd;
    case torq_hl::ElementwiseOpEnum::LOGICAL_OR:
        return Alu::Mode::logicalOr;
    case torq_hl::ElementwiseOpEnum::LOGICAL_XOR:
        return Alu::Mode::logicalXor;
    case torq_hl::ElementwiseOpEnum::BITWISE_NOT:
        break;
    case torq_hl::ElementwiseOpEnum::LOGICAL_NOT:
        break;
    }
    llvm::errs() << "Unsupported elementwise binary op: " << opType << "\n";
    assert(false && "Unsupported elementwise binary op");
}

template <>
LogicalResult ElementWiseBinaryPattern::transform(
    torq_hl::ElementWiseBinaryOp op, PatternRewriter &rewriter
) const {
    // Define data tensors
    LData input1(op.getInput1());
    LData input2(op.getInput2());
    LData output(op.getInit());

    assert(input1.elementType() == input2.elementType() && "Input element types must match");
    if (op.getIsUnsigned()) {
        input1.bitCast(toUnsigned(input1.elementType()));
        input2.bitCast(toUnsigned(input2.elementType()));
    }
    // TODO: handle broadcasting of input tensors

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

    // Vectorize the input tensors and stack them
    Slice slice(stringifyElementwiseOpEnum(op.getOpType()).str());
    int vectorSize = slice.alu.iWidth(input1.elementType());
    input1.fuse(denseDims).vectorize(vectorSize);
    input2.fuse(denseDims).vectorize(vectorSize);
    auto input = stack(rewriter, input1, input2);
    auto aluAccMode = opTypeToAluOpMode(op.getOpType());

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            PData pdata;
            For(auto it = slice.iterate(input.dim(In::InputTensors))) {
                IData idata = slice.iram.load(input[it][ndd][iv]);
                pdata = slice.alu.accumulate(idata, aluAccMode);
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
