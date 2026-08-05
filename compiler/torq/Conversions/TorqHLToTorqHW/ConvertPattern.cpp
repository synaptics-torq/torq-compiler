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

// Convert tensor encoding.
template <>
LogicalResult ConvertPattern::transform(torq_hl::ConvertOp op, PatternRewriter &rewriter) const {
    // The input and output can have any number of dimensions with any stride
    struct In : Vectorized {
        enum { NonDenseDims };
    };

    Slice slice("convert");
    LData input(op.getInput());
    LData output(op.getInit());
    assert(input.dims() == output.dims() && "Input and output tensors must have the same shape");

    if (isCompressed(input.elementType())) {
        return op.emitError() << "unsupported sub-byte element type "
                              << cast<MemRefType>(op.getInput().getType()).getElementType();
    }

    // Vectorize input
    int vectorSize = slice.act.width(input.elementType());
    input.fuse(std::min(input.denseDims(), output.denseDims())).vectorize(vectorSize);

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[ndd][iv]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.load(pdata);
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
