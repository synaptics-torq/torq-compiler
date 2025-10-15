// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

template <>
LogicalResult MatMulPattern::transform(torq_hl::MatMulOp op, PatternRewriter &rewriter) const {

    // input
    auto type1 = llvm::dyn_cast<MemRefType>(op.getInput1().getType());
    auto input1_shape = type1.getShape();
    auto type2 = llvm::dyn_cast<MemRefType>(op.getInput2().getType());
    auto input2_shape = type2.getShape();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();

    // MATMUL Kernel support matmul, dot product and matrix-vector multiplication
    // loose check rank
    auto rank1 = type1.getRank();
    auto rank2 = type2.getRank();
    if (rank1 < 1 || rank2 < 1) {
        op.emitError() << "Input must be valid matrix or vector, got " << rank1 << " and " << rank2
                       << "\n";
        return failure();
    }

    if (rank1 > 3) {
        op.emitError() << "Input rank must be 1, 2 or 3, got " << rank1 << "\n";
        return failure();
    }

    if (rank1 > 1 && rank2 > 1 && (input1_shape[rank1 - 1] != input2_shape[rank2 - 2])) {
        op.emitError() << "MatMulOp input shapes must match, got " << input1_shape[rank1 - 1]
                       << " and " << input2_shape[rank2 - 2] << "\n";
        return failure();
    }

    auto output_element_type = output_type.getElementType();
    auto output_element_size = output_element_type.getIntOrFloatBitWidth() / 8;
    if (output_element_size != 1 && output_element_size != 2 && output_element_size != 4) {
        op.emitError() << "Output element size must be 1, 2 or 4 bytes, got " << output_element_size
                       << "\n";
        return failure();
    }

    auto input_element_type = type1.getElementType();
    auto in_element_size = input_element_type.getIntOrFloatBitWidth() / 8;
    if (in_element_size != 2 && in_element_size != 1) {
        op.emitError() << "Input element size must be 1 or 2 bytes, got " << in_element_size
                       << "\n";
        return failure();
    }

    if (input_element_type.isBF16() && (!output_element_type.isBF16()) &&
        !output_element_type.isF32()) {
        op.emitError() << "Output element type must be bf16 input, bf16/fp32 output\n";
        return failure();
    }

    SmallVector<int64_t> input1Strides = getEncodedStridesElements(type1);
    SmallVector<int64_t> input2Strides = getEncodedStridesElements(type2);
    int m, k, n;
    int a_row_stride, b_row_stride;

    if (rank1 == 3 && rank2 == 3) { // batch matmul
        // [batch, m, k] x [batch, k, n] = [batch, m, n]
        if (input1_shape[0] > 1) {
            // We can support batch matmul quite easily in theory
            op.emitError() << "Batch > 1 not supported\n";
            return failure();
        }
        m = input1_shape[1];
        a_row_stride = input1Strides[1];
        k = input1_shape[2];
        b_row_stride = input2Strides[1];
        n = output_shape[2];
    }
    else if (rank1 == 2 && rank2 == 2) { // matmul
        // [m, k] x [k, n] = [m, n]
        m = input1_shape[0];
        a_row_stride = input1Strides[0];
        k = input1_shape[1];
        b_row_stride = input2Strides[0];
        n = output_shape[1];
    }
    else if (rank1 == 2 && rank2 == 1) { // matrix-vector multiplication
        // [m, k] x [k] = [m]
        m = input1_shape[0];
        a_row_stride = input1Strides[0];
        k = input1_shape[1];
        b_row_stride = 1;
        n = 1;
    }
    else if (rank1 == 1 && rank2 == 1) { // dot product
        // [k] x [k] = scalar
        m = n = 1;
        a_row_stride = 0;
        k = input1_shape[0];
        b_row_stride = 1;
    }
    else {
        llvm::errs() << "Invalid input shapes\n";
        return failure();
    }

    int32_t act_clip_min = std::numeric_limits<int32_t>::min();
    int32_t act_clip_max = std::numeric_limits<int32_t>::max();

    if (input_element_type.isBF16()) {
        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }

    Slice slice;
    const DType inputType = getDType(input_element_type);
    const DType outputType = getDType(output_element_type);
    int alu_group_width = slice.alu.iWidth(inputType, inputType);
    // Note: we could as well always set groupSize to 1 and simplify the code with minimal or no
    // perf impact. We keep it for now to make it easier to experiment with different versions.
    int groupSize = std::gcd((int)k, slice.alu.wWidth(inputType));
    int actBlockSize = slice.act.width(inputType, inputType);

    if (rank1 == 2 && rank2 == 1) {
        // matrix-vector multiplication
        alu_group_width = 1;
        actBlockSize = 1;
    }
    else if (rank1 == 1 && rank2 == 1) {
        // dot product
        alu_group_width = 1;
        actBlockSize = 1;
        groupSize = 1;
    }

    // Implementing matC = matA * matB

    // Loaded in WMEM
    LData matA(
        {{m, a_row_stride * in_element_size}, // M
         div_ceil(k, groupSize),              // K
         groupSize},
        inputType
    );
    // Loaded in IMEM
    LData matB(
        {
            div_ceil(k, groupSize),
            {groupSize, b_row_stride * in_element_size}, // K
            div_ceil(n, alu_group_width),
            alu_group_width // N
        },
        inputType
    );
    LData matC(
        {
            m,                                                                                  // M
            div_ceil(n, alu_group_width), div_ceil(alu_group_width, actBlockSize), actBlockSize // N
        },
        outputType
    );
    LData biasScale({2}, DType::int32);

    BData bdata = slice.bram.load(biasScale);
    For(auto im = slice.iterate(matA.dim(0))) {      // rows in matA
        For(auto ing = slice.iterate(matB.dim(2))) { // column groups in matB
            PData pdata;
            For(auto ikg = slice.iterate(matA.dim(1))) { // k groups in matA
                WData a = slice.wram.load(matA[im][ikg]);
                For(auto ik = slice.iterate(matA.dim(2))) { // each k in group
                    IData b = slice.iram.load(matB[ikg][ik][ing]);
                    pdata = slice.alu.scalarProductAccumulate(b, a[ik]);
                }
            }

            For(auto a = slice.iterate(matC.dim(-2))) {
                QData res = slice.act.rescaleClamp(
                    pdata[a], bdata, op.getShift(), op.getOutputZp(), act_clip_min, act_clip_max
                );
                slice.store(matC[im][ing][a], res);
            }
        }
    }

    auto newOp = rewriter.create<torq_hw::SliceTaskOp>(
        op->getLoc(), "matmul",
        op.getInput2(),                          // Input tensor
        op.getInput1(),                          // Weights
        op.getScaleBias(),                       // BiasScale tensor
        op.getInit(),                            // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );
    rewriter.replaceOp(op, newOp.getOperation());

    return success();
}

} // namespace mlir::syna::torq
