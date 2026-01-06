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

// Allow to enable the desired test code
#define IDENTITY_TEST_MODE 0

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult IdentityPattern::transform(torq_hl::IdentityOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    uint32_t total_px = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_px *= input_shape[i];
    }

    DType elementType = getDType(input_type.getElementType());

    Slice slice("identity");
    if (IDENTITY_TEST_MODE == 1 && total_px == 64 && elementType == DType::int8) {
        // This implementation copies a 64-bytes dense input tensor
        // The input tensor in partitioned in g sections reading from last one to exercise
        // the gather feature of DEDR
        // Can be tested with identity-64-i8.mlir
        llvm::errs() << "*** Identity using test mode " << IDENTITY_TEST_MODE << "\n";
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
    else if (IDENTITY_TEST_MODE == 2) {
        // This implementation copies a dense input tensor except the last 3 bytes of each block
        // Can be tested with identity-64-i8.mlir
        llvm::errs() << "*** Identity using test mode " << IDENTITY_TEST_MODE << "\n";
        int blockSize = slice.alu.iWidth(elementType);
        int actBlockSize = slice.act.width(elementType);
        const int blockCount = div_ceil(total_px, blockSize);
        int maxOutSize = blockSize - 3;

        LData input({blockCount, blockSize}, elementType);
        LData output({blockCount, maxOutSize}, elementType);

        // slice.setOutputSizeLimit(maxOutSize);
        For(auto b = slice.iterate(blockCount)) {
            IData idata = slice.iram.load(input[b]);
            PData pdata = slice.alu.load(idata);
            For(auto a = slice.iterate(blockSize / actBlockSize)) {
                QData res = slice.act.load(pdata[a]);
                slice.append(output[b], res);
            }
        }
    }
    else if (IDENTITY_TEST_MODE == 3 && total_px == 64 && elementType == DType::int8) {
        // This implementation copies a 64-bytes dense input tensor by writing each 8-bytes
        // chunk from the end (the result is not a correct memcopy).
        // The act output is partitioned in 2 non-contiguous sections to exercise
        // the scatter feature of DEQW.
        // Can be tested with identity-64-i8.mlir
        llvm::errs() << "*** Identity using test mode " << IDENTITY_TEST_MODE << "\n";
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
    else if (IDENTITY_TEST_MODE == 4 && total_px == 64) {
        // This implementation copies a 64-bytes dense input tensor by putting pixels at
        // even and odd positions in the first, second half of the output respectively
        // (the result is not a correct memcopy).
        // This is done using the even-odd feature of DEQW.
        // Can be tested with identity-64-xxx.mlir (xxx is any supported 8- or 16-bits type)
        // /!\ this is processing only one input data block
        llvm::errs() << "*** Identity using test mode " << IDENTITY_TEST_MODE << "\n";
        int blockSize = 64 / sizeofType(elementType);
        int actBlockSize = 16;

#define IDENTITY_EVEN_ODD_WITH_APPEND 1
        LData input({blockSize}, elementType);
        LData output(
#if IDENTITY_EVEN_ODD_WITH_APPEND
            {blockSize},
#else
            {{blockSize / actBlockSize, actBlockSize / 2}, {actBlockSize / 2, 1}, {2, blockSize / 2}
            },
#endif
            elementType
        );
#if IDENTITY_EVEN_ODD_WITH_APPEND
        output.partitionByIndexParity1D();
#endif
        llvm::errs() << "\nOutput after partitioning by index parity output: " << output << "\n";

        IData idata = slice.iram.load(input);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
#if IDENTIY_EVEN_ODD_WITH_APPEND
            slice.append(output, res);
#else
            slice.store(output[a], res);
#endif
        }
    }
    else if (IDENTITY_TEST_MODE == 5) {
        // This example works only with identity-64-i8.mlir and g == 1, 2 or 4
        // It copies input to output by skipping an element every blockSize/g elements
        llvm::errs() << "*** Identity using test mode " << IDENTITY_TEST_MODE << "\n";
        int blockSize = 64;
        int actBlockSize = 16;
        int g = 4;
        LData input({{g, ((blockSize / g) + 1)}, blockSize / g}, elementType);
        LData output({blockSize / actBlockSize, actBlockSize}, elementType);

        IData idata = slice.iram.load(input);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.load(pdata[a]);
            slice.store(output[a], res);
        }
    }
    else {
        struct In : Vectorized {
            enum { NonDenseDims };
        };
        LData input(op.getInput());
        LData output(op.getInit());
        // FIXME: In some cases this op is also used as a bit-cast bf16 -> i16 or i16 -> bf16.
        // For now, we can handle this case as i16 since it's just a bypass.
        if (input.elementType() == DType::bf16 || output.elementType() == DType::bf16) {
            input.setElementType(DType::int16);
            output.setElementType(DType::int16);
        }
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
    }

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        slice.name(),                            // Operation to replace
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
