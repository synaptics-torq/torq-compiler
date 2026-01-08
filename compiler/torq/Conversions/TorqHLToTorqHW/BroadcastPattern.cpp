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

// All the output dims not in dimensions are added to the input as extra dims of size 1
// The indexes in dimensions must be sorted in ascending order
static LData expandDims(const LData &input, const LData &output, ArrayRef<int64_t> dimensions) {
    if (input.shape().size() == output.shape().size()) {
        if (!dimensions.empty()) {
            // This should never happen, makes no sense to specify dimensions if ranks are the same
            llvm::errs() << "Warning: broadcast in/out have same rank but dimensions not empty\n";
        }
        return input;
    }
    assert(input.shape().size() + dimensions.size() == output.shape().size());
    LData expanded({}, input.elementType());
    int inputIx = 0;
    int dimensionsIx = 0;
    for (size_t outDim = 0; outDim < output.shape().size(); outDim++) {
        if (dimensionsIx < dimensions.size() && outDim == dimensions[dimensionsIx]) {
            assert(dimensionsIx == 0 || dimensions[dimensionsIx] > dimensions[dimensionsIx - 1]);
            expanded.getShape().push_back(1);
            dimensionsIx++;
        }
        else {
            assert(input.shape()[inputIx].count == output.shape()[outDim].count);
            expanded.getShape().push_back(input.shape()[inputIx]);
            inputIx++;
        }
    }
    return expanded;
}

template <>
LogicalResult
BroadcastPattern::transform(torq_hl::BroadcastOp op, PatternRewriter &rewriter) const {
    struct In : Vectorized {
        enum { NonDenseDims };
    };

    Slice slice("broadcast");
    LData input(op.getInput());
    LData output(op.getInit());

    // Apply explicit broadcasting (not needed if higher level can ensure input rank == output rank)
    input = expandDims(input, output, op.getDimensions());

    // Apply implicit broadcasting
    input.broadcastAs(output);

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

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                        // Operation to replace
        slice.name(),              // Operation name
        ValueRange{op.getInput()}, // Input tensor
        ValueRange{},              // Weights
        ValueRange{},              // BiasScale tensor,
        ValueRange{op.getInit()},  // Output tensor initializer
        ValueRange{},              // Symbols
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );
    return success();
}

} // namespace mlir::syna::torq
