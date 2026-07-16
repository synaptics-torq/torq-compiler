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
    int32_t val = op.getValue();
    assert(val >= -32768 && val <= 65535 && "Fill value out of range");
    Slice slice;
    LData output(output_type);
    output.fuse(output.denseDims());
    // For sub-word types (i8, i16, bf16), widen to int32 when the fused dense dim is
    // exactly divisible by the factor (4 for i8, 2 for i16/bf16). Divide the last dim
    // count and all outer strides by the factor to keep the shape consistent in int32 units.
    // Outer strides are LRAM-aligned so the division is always exact.
    // Also replicate the fill value into all byte/halfword positions of the int32 so that
    // append(int) receives an already-extended value and passes it directly as pad_value.
    if (sizeofType(output.elementType()) < 4) {
        int factor = 4 / sizeofType(output.elementType());
        if (output.dim(-1) % factor == 0) {
            // Replicate val to fill a 32-bit word (mirrors extendPadValue in Kernel.cpp)
            if (factor == 4) {
                val &= 0xFF;
                val = val | (val << 8) | (val << 16) | (val << 24);
            }
            else {
                val &= 0xFFFF;
                val = val | (val << 16);
            }
            Shape &shape = output.getShape();
            shape.back().count /= factor;
            for (int i = 0; i < (int)shape.size() - 1; ++i) {
                if (shape[i].stride.intVal.has_value())
                    shape[i].stride.intVal = shape[i].stride.intVal.value() / factor;
            }
            output.setElementType(DType::int32);
        }
    }
    int vectorSize = slice.act.width(output.elementType());
    For(auto ndd = slice.iterate(output.dims(0, -1))) {
        For(auto ov = slice.iterate(div_ceil(output.dim(-1), vectorSize))) {
            slice.append(output[ndd], val);
        }
    }

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
