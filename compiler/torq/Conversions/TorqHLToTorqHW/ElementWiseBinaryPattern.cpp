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

static torq_hw::ALUOp1Mode opTypeToAluOpMode(torq_hl::ElementwiseOpEnum opType) {
    switch (opType) {
    case torq_hl::ElementwiseOpEnum::BITWISE_AND:
        return torq_hw::ALUOp1Mode::BAND;
    case torq_hl::ElementwiseOpEnum::BITWISE_OR:
        return torq_hw::ALUOp1Mode::BOR;
    case torq_hl::ElementwiseOpEnum::BITWISE_XOR:
        return torq_hw::ALUOp1Mode::BXOR;
    case torq_hl::ElementwiseOpEnum::MINIMUM:
        return torq_hw::ALUOp1Mode::MIN;
    case torq_hl::ElementwiseOpEnum::MAXIMUM:
        return torq_hw::ALUOp1Mode::MAX;
    case torq_hl::ElementwiseOpEnum::GREATER:
        return torq_hw::ALUOp1Mode::GT;
    case torq_hl::ElementwiseOpEnum::GREATER_EQUAL:
        return torq_hw::ALUOp1Mode::GE;
    case torq_hl::ElementwiseOpEnum::EQUAL:
        return torq_hw::ALUOp1Mode::EQ;
    case torq_hl::ElementwiseOpEnum::LOGICAL_AND:
        return torq_hw::ALUOp1Mode::AND;
    case torq_hl::ElementwiseOpEnum::LOGICAL_OR:
        return torq_hw::ALUOp1Mode::OR;
    case torq_hl::ElementwiseOpEnum::LOGICAL_XOR:
        return torq_hw::ALUOp1Mode::XOR;
    case torq_hl::ElementwiseOpEnum::SUB:
        return torq_hw::ALUOp1Mode::SACC;
    case torq_hl::ElementwiseOpEnum::ADD:
        return torq_hw::ALUOp1Mode::ACC;
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
