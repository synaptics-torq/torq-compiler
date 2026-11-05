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

static FailureOr<Act::ShiftMode> toShiftMode(torq_hl::ShiftModeEnum opType) {
    switch (opType) {
    case torq_hl::ShiftModeEnum::LSR:
        return Act::ShiftMode::lsr;
    case torq_hl::ShiftModeEnum::LSL:
        return Act::ShiftMode::lsl;
    case torq_hl::ShiftModeEnum::ASR:
        return Act::ShiftMode::asr;
    case torq_hl::ShiftModeEnum::ASL:
        break;
    }
    return failure();
}

// The ACT shifts the values in the first half of its input partials by the amounts in the second
// half, both packed at the element size (see Act::shift). The ALU loads the two inputs side by
// side as int32 words: each step loads width(int32) / 2 words of each input and produces one
// vector of results.
template <>
LogicalResult ElementWiseShiftPattern::transform(
    torq_hl::ElementWiseShiftOp op, PatternRewriter &rewriter
) const {
    FailureOr<Act::ShiftMode> actMode = toShiftMode(op.getOpType());
    if (failed(actMode))
        return op.emitError() << "Unsupported elementwise shift op: " << op.getOpType();

    // Define data tensors
    LData input1(op.getInput1());
    LData input2(op.getInput2());
    LData output(op.getInit());
    const DType elementType = input1.elementType();
    assert(input2.elementType() == elementType && "Input types must match");
    if (input1.offset() != input2.offset())
        return op.emitError() << "Inputs must have the same offset";

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

    // Results per step, and the int32 words of each input they use. Each step loads loadWords
    // words of each input, so for int8 the vectors overlap.
    Slice slice(stringifyShiftModeEnum(op.getOpType()).str());
    const int partials = slice.act.width(DType::int32);
    const int loadWords = partials / 2;
    const int elementSize = sizeofType(elementType);
    const int packFactor = 4 / elementSize;
    const int resultCount = elementSize == 4 ? partials / 2 : partials;
    const int stepWords = resultCount / packFactor;

    // View the dense part of each input as int32 words, rounded up to whole steps: every load is a
    // full vector, and the results past the end of the output are discarded by append(). If even
    // the innermost dimension is not dense the inputs are processed element by element.
    for (LData *input : {&input1, &input2}) {
        input->fuse(denseDims);
        if (denseDims == 0)
            input->insertDim(input->shape().size(), ShapeItem{1});
        input->forceReshapeDim(-1, {(int)align_ceil(input->dim(-1), resultCount)});
        if (packFactor > 1) {
            // The rows must start on word boundaries to be packed
            bool aligned = input->offset() % packFactor == 0;
            for (int i = 0; i < (int)input->shape().size() - 1; ++i) {
                const Stride &stride = input->shape()[i].stride;
                aligned &= !stride.intVal.has_value() || stride.intVal.value() % packFactor == 0;
            }
            if (!aligned)
                return op.emitError() << "Input rows must be aligned to 4 bytes";
            input->bitCast(DType::int32);
        }
        input->vectorize(loadWords, stepWords);
    }
    LData input = LData::stack(rewriter, input1, input2, In::InputTensors);
    output.fuse(denseDims);

    slice.act.setRoundingMode(op.getRound() ? Act::Rounding::nearestUp : Act::Rounding::off);
    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[":"][ndd][iv]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.shift(pdata, elementType, *actMode);
            slice.append(output[ndd], res);
        }
    }

    rewriter.replaceOp(op, slice.createSliceTaskOp(rewriter, op.getLoc()));
    return success();
}

} // namespace mlir::syna::torq
