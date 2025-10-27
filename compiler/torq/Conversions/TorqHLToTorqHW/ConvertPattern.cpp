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
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());

    // Check that the input/output tensors are compatible
    auto output_type = llvm::cast<MemRefType>(op.getInit().getType());
    if (input_type.getShape() != output_type.getShape()) {
        return op.emitError("Input and output tensors must have the same shape");
    }

    // The input and output can have any number of dimensions with any stride
    Slice slice;
    LData input(input_type);
    LData output(output_type);

    For(auto ii = slice.iterate(input.dims())) {
        IData idata = slice.iram.load(input[ii]);
        PData pdata = slice.alu.load(idata);
        QData res = slice.act.load(pdata);
        slice.store(output[ii], res);
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        "convert",                               // Operation to replace
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
