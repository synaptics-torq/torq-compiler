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

    SmallVector<int64_t> dims(broadcastDims.begin(), broadcastDims.end());
    llvm::sort(dims.begin(), dims.end());
    const auto outRank = static_cast<int>(outputShape.size());
    const auto inRank = static_cast<int>(inputShape.size());

    for (int64_t d : dims) {
        if (d < 0 || d >= outRank) {
            return rewriter.notifyMatchFailure(op, "Invalid broadcast dimension");
        }
    }

    auto product64 = [](ArrayRef<int64_t> shape) -> uint64_t {
        uint64_t p = 1;
        for (int64_t v : shape) {
            p *= static_cast<uint64_t>(v);
        }
        return p;
    };

    // Special-case: insert a single broadcast dim right before the last input dim.
    // Example: input [1,207,32] -> output [1,207,8,32] with dims=[2].
    if (dims.size() == 1 && outRank == inRank + 1 && dims.front() == inRank - 1) {
        const int64_t insertDim = dims.front();

        // Validate that inputShape matches outputShape with the inserted dim removed.
        for (int i = 0; i < inRank; ++i) {
            const int outIdx = (i < insertDim) ? i : (i + 1);
            if (inputShape[i] != outputShape[outIdx]) {
                return rewriter.notifyMatchFailure(op, "Broadcast insert-dim shape mismatch");
            }
        }

        if (outputShape[insertDim] <= 0) {
            return rewriter.notifyMatchFailure(op, "Broadcast insert-dim shape mismatch");
        }

        const uint64_t prefix = product64(outputShape.take_front(insertDim));
        const uint64_t dimSize = static_cast<uint64_t>(outputShape[insertDim]);
        const uint64_t trailing = product64(outputShape.drop_front(insertDim + 1));

        const DType elementType = getDType(inputType.getElementType());

        // Represent input as [prefix, trailing] and output as [prefix, dimSize, trailing].
        LData input({static_cast<int64_t>(prefix), static_cast<int64_t>(trailing)}, elementType);
        LData output(
            {static_cast<int64_t>(prefix), static_cast<int64_t>(dimSize),
             static_cast<int64_t>(trailing)},
            elementType
        );

        Slice slice;
        For(auto p = slice.iterate(static_cast<int>(prefix))) {
            For(auto t = slice.iterate(static_cast<int>(trailing))) {
                IData idata = slice.iram.load(input[p][t]);
                For(auto r = slice.iterate(static_cast<int>(dimSize))) {
                    PData pdata = slice.alu.load(idata);
                    QData res = slice.act.load(pdata);
                    slice.store(output[p][r][t], res);
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

    auto allPrevOutOnes = [&](int dimIdx) -> bool {
        for (int i = 0; i < dimIdx; ++i) {
            if (outputShape[i] != 1)
                return false;
        }
        return true;
    };

    int primaryDim = -1;

    // Primary-dim selection prefers places that match our simple 2D kernel shape.
    // Prefer append: dim == inRank.
    for (int64_t d : dims) {
        if (d == inRank) {
            primaryDim = static_cast<int>(d);
            break;
        }
    }

    // Prefer same-rank last: dim == inRank - 1.
    // Passes when expanding the last existing dim of a same-rank tensor,
    // e.g. [1x21x1] -> [1x21x1024] at dim=2
    if (primaryDim < 0 && inRank > 0) {
        for (int64_t d : dims) {
            if (d == inRank - 1) {
                primaryDim = static_cast<int>(d);
                break;
            }
        }
    }

    // Try first legal middle insert: 0 < dim < outRank && all previous outputs are 1.
    // Passes when inserting in the middle but the earlier output dims are all 1,
    // which avoids ambiguous layout. This matches our “middle insert” constraint
    if (primaryDim < 0) {
        for (int64_t d : dims) {
            if (d > 0 && d < outRank && allPrevOutOnes(static_cast<int>(d))) {
                primaryDim = static_cast<int>(d);
                break;
            }
        }
    }

    uint64_t initSize = product64(inputShape);

    uint64_t dimSize = static_cast<uint64_t>(
        (initSize == 1) ? product64(outputShape) : // scalar fast-path: fill all
            std::max<int64_t>(outputShape[primaryDim], 1)
    );

    Shape oShape;
    bool outputIdxReverse = false;

    // Append-style (dim==inRank), same-rank last (dim==inRank-1), OR scalar fast-path.
    if (primaryDim == inRank || (inRank == outRank && primaryDim == inRank - 1) ||
        (initSize == 1)) {
        outputIdxReverse = false; // expand across columns
        oShape = {static_cast<int64_t>(initSize), static_cast<int64_t>(dimSize)};
    }
    // Prepend-style (dim==0).
    else if (primaryDim == 0) {
        outputIdxReverse = true; // expand across rows
        oShape = {static_cast<int64_t>(dimSize), static_cast<int64_t>(initSize)};
    }
    // Middle insert (primaryDim>0) but only when all earlier outputs are 1.
    else if (primaryDim > 0) {
        if (!allPrevOutOnes(primaryDim)) {
            return rewriter.notifyMatchFailure(
                op, "Broadcast middle-insert requires all earlier output dims to be 1"
            );
        }
        outputIdxReverse = true;
        oShape = {static_cast<int64_t>(dimSize), static_cast<int64_t>(initSize)};
    }
    else {
        return rewriter.notifyMatchFailure(op, "Unsupported broadcast case");
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