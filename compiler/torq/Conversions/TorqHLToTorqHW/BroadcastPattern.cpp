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
LogicalResult
BroadcastPattern::transform(torq_hl::BroadcastOp op, PatternRewriter &rewriter) const {

    auto inputType = llvm::cast<MemRefType>(op.getInput().getType());
    auto inputShape = inputType.getShape();

    auto outputType = llvm::cast<MemRefType>(op.getInit().getType());
    auto outputShape = outputType.getShape();

    ArrayRef<int64_t> broadcastDims = op.getDimensions();
    if (broadcastDims.empty()) {
        return failure();
    }
    // if (broadcastDims.size() != 1) {
    //     return rewriter.notifyMatchFailure(
    //         op, "Broadcast dimensions must be a single value for now"
    //     );
    // }

    auto dim = broadcastDims[0];
    if (dim < 0 || dim >= outputShape.size()) {
        return rewriter.notifyMatchFailure(op, "Invalid broadcast dimension");
    }

    uint32_t initSize = 1;
    for (int i = 0; i < inputShape.size(); ++i) {
        initSize *= inputShape[i];
    }

    uint32_t dimSize = outputShape[dim];

    // 1x1x1 -> 1x21x1024
    // if broadcast from 1x1x1, we don't care if dims.size()==1
    if (initSize == 1) {
        dimSize = 1;
        for (int i = 0; i < outputShape.size(); ++i) {
            dimSize *= outputShape[i];
        }
    }

    Shape oShape;
    bool outputIdxReverse = false;
    // 6 -> 6x7 or 6x7 -> 6x7x3 or 6x7 -> 6x7x5x3
    if (dim == inputShape.size() ||
        // 1x21x1 -> 1x21x1024
        (inputShape.size() == outputShape.size() && dim == inputShape.size() - 1) ||
        // 1x1x1 -> 1x21x1
        (initSize == 1)) {

        outputIdxReverse = false;
        oShape = {initSize, dimSize};
    }
    else if (dim == 0) {
        // 6 -> 7x6 or 6x5 -> 7x6x5
        outputIdxReverse = true;
        oShape = {dimSize, initSize};
    }
    else if (dim > 0) {
        // 1x6 -> 1x7x6, 1x1x1x6 -> 1x1x1x7x6
        bool allOne = true;
        if (dim > 0) {
            for (int i = 0; i < dim; i++) {
                if (outputShape[i] != 1) {
                    allOne = false;
                    break;
                }
            }
        }
        if (allOne) {
            outputIdxReverse = true;
            oShape = {dimSize, initSize};
        }
        else {
            return rewriter.notifyMatchFailure(
                op, "Broadcast doesn't support non-zero dim before broadcast dimension"
            );
        }
    }
    else {
        return rewriter.notifyMatchFailure(op, "Broadcast not supported for this case");
    }

    const DType elementType = getDType(inputType.getElementType());

    LData input({initSize}, elementType);
    LData output{oShape, elementType};

    Slice slice;
    For(auto b = slice.iterate(initSize)) {
        IData idata = slice.iram.load(input[b]);
        For(auto a = slice.iterate(dimSize)) {
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.load(pdata);

            if (outputIdxReverse) {
                slice.store(output[a][b], res);
            }
            else {
                slice.store(output[b][a], res);
            }
        }
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        "broadcast",               // Operation to replace
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