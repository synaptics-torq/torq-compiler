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
LogicalResult ElementWiseUnaryPattern::transform(
    torq_hl::ElementWiseUnaryOp op, PatternRewriter &rewriter
) const {

    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    assert(input_shape.size() > 0);

    torq_hw::ALUOp1Mode hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
    torq_hl::ElementwiseOpEnum opType = op.getOpType();

    if (opType == torq_hl::ElementwiseOpEnum::BITWISE_NOT) {
        hwOp1Mode = torq_hw::ALUOp1Mode::BNOT;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::LOGICAL_NOT) {
        hwOp1Mode = torq_hw::ALUOp1Mode::NOT;
    }
    else {
        llvm::errs() << "Unsupported elementwise unary  op: " << opType << "\n";
        return failure();
    }

    Slice slice;
    DType elementType = getDType(input_type.getElementType());
    int blockSize = slice.alu.iWidth(elementType);
    const int blockCount = div_ceil(input_type.getNumElements(), blockSize);
    int actBlockSize = slice.act.width(elementType);

    LData input({blockCount, blockSize}, elementType);
    LData output({blockCount, blockSize / actBlockSize, actBlockSize}, elementType);

    For(auto b = slice.iterate(blockCount)) {
        IData idata = slice.iram.load(input[b]);
        PData pdata = slice.alu.accumulate(idata, hwOp1Mode);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
            slice.store(output[b][a], res);
        }
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        stringifyElementwiseOpEnum(opType),      // Operation to replace
        ValueRange{op.getInput()},               // Input tensor
        ValueRange{},                            // Weights
        ValueRange{},                            // BiasScale tensor,
        ValueRange{op.getInit()},                // Output tensor initializer
        ValueRange{},                            // Symbols
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
