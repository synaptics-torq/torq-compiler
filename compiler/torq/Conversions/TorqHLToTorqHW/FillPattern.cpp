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
LogicalResult FillPattern::transform(torq_hl::FillOp op, PatternRewriter &rewriter) const {

    // The tensor to be filled can have any number of dimensions with any stride
    // Note: padding areas are not filled
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    Slice slice;
    LData output(output_type);
    For(auto ii = slice.iterate(output.dims())) { slice.store(output[ii], op.getValue()); }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                                      // Operation to replace
        "fill",                                  // Task name
        ValueRange{},                            // Input tensor
        ValueRange{},                            // Weights
        ValueRange{},                            // BiasScale tensor,
        ValueRange{op.getInit()},                // Output tensor initializer
        ValueRange{},                            // Symbols
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
