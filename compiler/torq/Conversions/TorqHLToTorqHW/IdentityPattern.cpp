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
LogicalResult IdentityPattern::transform(torq_hl::IdentityOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    assert(input_shape.size() > 0);
    uint32_t total_px = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_px *= input_shape[i];
    }

    bool is_dense = isDenseInMemory(input_type);
    DType elementType = getDType(input_type.getElementType());

    // FIXME: The reinterpret cast from bf16 to i16 was failing.
    // For now, we can treat the input as i16 since it's just a bypass.
    if (elementType == DType::bf16) {
        elementType = DType::int16;
    }
    const int elementSize = sizeofType(elementType);
    auto input_strides = getEncodedStridesElements(input_type);
    Slice slice;
    bool is_size_aligned = (total_px % slice.alu.iWidth(elementType)) == 0;
    if (total_px == 64 && elementType == DType::int8) {
        // This implementation copies a 64-bytes dense input tensor
        // The input tensor in partitioned in g sections reading from last one to exercise
        // the gather feature of DEDR
        // Can be tested with identity-64-i8.mlir
        int blockSize = 64;
        int actBlockSize = 16;
        int g = 4; // This example only works with g == 4

        LData input({{g, -(blockSize / g)}, blockSize / g}, elementType);
        LData output({blockSize / actBlockSize, actBlockSize}, elementType);
        input.setOffset(blockSize - blockSize / g);

        IData idata = slice.iram.load(input);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
            slice.store(output[a.reverse()], res);
        }
    }
    else if (total_px == 64 && elementType == DType::int8 && false) {
        // This implementation copies a 64-bytes dense input tensor by writing each 8-bytes
        // chunk from the end (the result is not a correct memcopy).
        // The act output is partitioned in 2 non-contiguous sections to exercise
        // the scatter feature of DEQW.
        // Can be tested with identity-64-i8.mlir
        int blockSize = 64;
        int actBlockSize = 16;
        int g = 2; // This example only works with g == 2

        LData input({blockSize}, elementType);
        LData output(
            {blockSize / actBlockSize, {g, -(actBlockSize / g)}, actBlockSize / g}, elementType
        );
        output.setOffset(blockSize - actBlockSize / g);

        IData idata = slice.iram.load(input);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
            slice.store(output[a], res);
        }
    }
    else if (false) {
        // This example works only with identity-64-i8.mlir and g == 1, 2 or 4
        // It copies input to output by skipping an element every blockSize/g elements
        int blockSize = 64;
        int actBlockSize = 16;
        int g = 4;
        LData input({{g, (blockSize / g) + 1}, blockSize / g}, elementType);
        LData output({blockSize / actBlockSize, actBlockSize}, elementType);

        IData idata = slice.iram.load(input);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
            slice.store(output[a], res);
        }
    }
    else if (is_dense && is_size_aligned) {
        // Most efficient implementation, data is transferred in blocks
        int blockSize = slice.alu.iWidth(elementType);
        const int blockCount = div_ceil(total_px, blockSize);
        int actBlockSize = slice.act.width(elementType);

        LData input({blockCount, blockSize}, elementType);
        LData output({blockCount, blockSize / actBlockSize, actBlockSize}, elementType);

        // Use reverse iterator. We don't really need it here, it's just to test its usage
        For(auto b = slice.iterate(blockCount).reverse()) {
            IData idata = slice.iram.load(input[b]);
            PData pdata = slice.alu.load(idata);
            For(auto a = slice.iterate(blockSize / actBlockSize)) {
                QData res = slice.act.load(pdata[a]);
                slice.store(output[b][a], res);
            }
        }
    }
#if NOT_YET_IMPLEMENTED
    else if (is_dense) {
        // TODO: use peeling
    }
#endif
    else if (input_shape.back() <= slice.act.width(elementType) && input_strides.back() == 1) {
        // Copy innermost dimension at a time
        // TODO: this can be extended to handle more than one innermost dimension even > act.width
        // as long as data is contiguous
        // The input can have any number of dimensions with any stride (innermost must be dense)
        Shape inputShape;
        for (int i = 0; i < input_shape.size(); ++i) {
            inputShape.push_back({input_shape[i], input_strides[i] * elementSize});
        }
        // The output is a dense version of the input
        Shape outputShape;
        for (int i = 0; i < input_shape.size(); ++i) {
            outputShape.push_back(input_shape[i]);
        }

        LData input(inputShape, elementType);
        LData output(outputShape, elementType);

        For(auto ii = slice.iterate(input.dims(0, -1))) {
            IData idata = slice.iram.load(input[ii]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.load(pdata);
            slice.store(output[ii], res);
        }
    }
    else {
        // Flexible copy: least efficient but most flexible implementation
        // copy one item at a time
        // The input can have any number of dimensions with any stride

        // The output is a dense version of the input
        Shape outputShape;
        for (auto s : input_shape) {
            outputShape.push_back(s);
        }

        LData input(input_type);
        LData output(outputShape, elementType);

        For(auto ii = slice.iterate(input.dims())) {
            IData idata = slice.iram.load(input[ii]);
            PData pdata = slice.alu.load(idata);
            QData res = slice.act.load(pdata);
            slice.store(output[ii], res);
        }
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        "identity",                              // Operation to replace
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
