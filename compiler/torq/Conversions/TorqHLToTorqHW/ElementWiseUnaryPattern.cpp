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
LogicalResult ElementWiseUnaryPattern::transform(
    torq_hl::ElementWiseUnaryOp op, PatternRewriter &rewriter
) const {

    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    assert(input_shape.size() > 0);

    Alu::Mode aluMode = Alu::Mode::acc;
    torq_hl::ElementwiseOpEnum opType = op.getOpType();

    if (opType == torq_hl::ElementwiseOpEnum::BITWISE_NOT) {
        aluMode = Alu::Mode::bitNot;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::LOGICAL_NOT) {
        aluMode = Alu::Mode::logicalNot;
    }
    else {
        llvm::errs() << "Unsupported elementwise unary  op: " << opType << "\n";
        return failure();
    }

    Slice slice(std::string("Eltwise-") + std::string(stringifyElementwiseOpEnum(opType)));
    LData input(op.getInput());
    LData output(op.getInit());
    if (input.denseDims() != (int)input.dims().size() ||
        output.denseDims() != (int)output.dims().size()) {
        return op.emitError("elementwise unary input and output must be dense");
    }

    // Process the data as a flat array of blocks; the last block is read and written whole
    int blockSize = slice.act.width(input.elementType());
    input.fuse(input.dims().size()).forceReshapeDim(-1, {-1, blockSize});
    output.fuse(output.dims().size()).forceReshapeDim(-1, {-1, blockSize});

    For(auto b = slice.iterate(input.dim(0))) {
        IData idata = slice.iram.load(input[b]);
        PData pdata = slice.alu.accumulate(idata, aluMode);
        QData res = slice.act.load(pdata);
        slice.store(output[b], res);
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));

    return success();
}

} // namespace mlir::syna::torq
