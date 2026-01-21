// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-mul"

namespace mlir::syna::torq {

template <>
LogicalResult MulPattern::transform(torq_hl::MulOp op, PatternRewriter &rewriter) const {

    // input
    MemRefType dataType, weightType;

    auto type1 = llvm::dyn_cast<MemRefType>(op.getInput1().getType());
    auto rank1 = type1.getRank();
    auto elType1 = type1.getElementType();
    auto input1_element_size = elType1.getIntOrFloatBitWidth() / 8;
    dataType = type1;

    auto type2 = llvm::dyn_cast<MemRefType>(op.getInput2().getType());
    auto elType2 = type2.getElementType();
    auto input2_element_size = elType2.getIntOrFloatBitWidth() / 8;
    auto rank2 = type2.getRank();
    weightType = type2;

    bool hasScalar = false;
    bool needsBroadcast = false;
    // scalar rank is 0, just right now we partially support scalar in different module
    // so when lowering to torq_hl::mulOp we create a dim=1 tensor, rank becomes 1

    // TODO: refactor beblow condition check
    if (rank1 == 0 || rank2 == 0) {
        hasScalar = true;
        if (rank1 == 0 && rank2 == 0) {
            dataType = type1;
            weightType = type2;
        }
        else if (rank1 == 0) {
            dataType = type2;
            weightType = type1;
        }
        else if (rank2 == 0) {
            dataType = type1;
            weightType = type2;
        }
    }
    else if (rank1 == 1 || rank2 == 1) {
        if (rank1 == 1 && rank2 == 1) {
            if (type1.getShape()[0] > type2.getShape()[0]) {
                hasScalar = true;
                dataType = type1;
                weightType = type2;
            }
            else if (type1.getShape()[0] < type2.getShape()[0]) {
                hasScalar = true;
                dataType = type2;
                weightType = type1;
            }
            else {
                if (type1.getShape()[0] == 1) {
                    hasScalar = true;
                }
            }
        }
        else if (rank1 == 1) {
            // Only treat as scalar if shape is [1], not for larger 1D tensors like [288]
            if (type1.getShape()[0] == 1) {
                hasScalar = true;
            }
            else {
                // Broadcasting case: 1D weight tensor needs to be broadcast across higher-dim data
                needsBroadcast = true;
            }
            weightType = type1;
            dataType = type2;
        }
        else if (rank2 == 1) {
            // Only treat as scalar if shape is [1], not for larger 1D tensors like [288]
            if (type2.getShape()[0] == 1) {
                hasScalar = true;
            }
            else {
                // Broadcasting case: 1D weight tensor needs to be broadcast across higher-dim data
                needsBroadcast = true;
            }
            weightType = type2;
            dataType = type1;
        }
    }

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto outElType = output_type.getElementType();

    // alu and act process dbus/wbus two 8/16-bit elements size at a time
    uint32_t data_width = input1_element_size * input2_element_size;

    int32_t act_clip_min = std::numeric_limits<int32_t>::min();
    int32_t act_clip_max = std::numeric_limits<int32_t>::max();

    if (elType1.isBF16() && elType2.isBF16()) {
        data_width = input1_element_size;
        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }

    const uint32_t alu_group_width = 32;
    const uint32_t blockSize = alu_group_width / input1_element_size;
    // for int16 data_width=4, divided by 4 means we need 4 partial 32bit to compute each 48bit
    // result with the act_lsh for int8/bf16 use natual data_width
    const uint32_t actWidth = HwInfo::act_width / data_width;
    const uint32_t actBlockCount = div_ceil(blockSize, actWidth);

    LData biasScale({2}, elType1.isBF16() ? DType::bf16 : DType::int32);

    Slice slice;
    BData bdata = slice.bram.load(biasScale);

    if (hasScalar) {
        // Case 1: Scalar multiplication - one weight value applied to all elements
        const uint32_t frame_size = getEncodedTotalSizeBytes(dataType) / input1_element_size;
        assert(frame_size > 0);

        LData weights({1}, getDType(weightType.getElementType()));
        WData wdata = slice.wram.load(weights);

        LData input(op.getInput1());
        LData output(op.getInit());

        // Dimensions of the input data for processing
        struct In : Vectorized {
            enum {
                NonDenseDims, // First (non-dense) data dimension if any
            };
        };
        int denseDims = std::min(input.denseDims(), output.denseDims());
        int vectorSize = slice.alu.iWidth(input.elementType(), weights.elementType());
        input.fuse(denseDims).vectorize(vectorSize);

        For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                IData idata = slice.iram.load(input[ndd][iv]);
                PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);
                For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.rescaleClamp(
                        pdata[av], bdata, op.getShift(), 0, act_clip_min, act_clip_max
                    );
                    slice.append(output[ndd], res);
                }
            }
        }
    }
    else if (needsBroadcast) {
        // Case 2: Broadcasting - 1D weight tensor broadcast across higher-dim data tensor
        // e.g., [1,207,288] * [288] = [1,207,288]
        //
        // Strategy: The weight tensor has shape [W] (e.g., [288]).
        // The data tensor has shape [..., W] (e.g., [1, 207, 288]).
        // We need to iterate over the "rows" (outer dimensions collapsed) and for each row,
        // perform elementwise multiplication with the same weights.
        //
        // For [1,207,288] * [288]:
        // - data has 1*207 = 207 "rows" of 288 elements each
        // - weights has 288 elements
        // - For each row, we multiply 288 elements by 288 weights

        // Calculate dimensions
        const uint32_t weight_size = getEncodedTotalSizeBytes(weightType) / input1_element_size;
        const uint32_t data_size = getEncodedTotalSizeBytes(dataType) / input1_element_size;
        const uint32_t num_rows = data_size / weight_size; // 59616 / 288 = 207
        const uint32_t weightBlockCount = div_ceil(weight_size, blockSize);

        assert(weight_size > 0);
        assert(num_rows > 0);

        // Create LData for weights with block structure
        LData weights({weightBlockCount, blockSize}, getDType(weightType.getElementType()));

        // Create LData for input with row and block structure
        // Shape: [num_rows, weightBlockCount, blockSize]
        LData input({num_rows, weightBlockCount, blockSize}, getDType(dataType.getElementType()));

        // Create LData for output with row, block, and act structure
        LData output({num_rows, weightBlockCount, actBlockCount, actWidth}, getDType(outElType));

        // Outer loop: iterate over rows
        For(auto row = slice.iterate(num_rows)) {
            // Inner loop: iterate over weight blocks
            For(auto b = slice.iterate(weightBlockCount)) {
                IData idata = slice.iram.load(input[row][b]);
                WData wdata = slice.wram.load(weights[b]); // Same weights for each row
                PData pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);

                For(auto a = slice.iterate(actBlockCount)) {
                    QData res = slice.act.rescaleClamp(
                        pdata[a], bdata, op.getShift(), 0, act_clip_min, act_clip_max
                    );
                    slice.store(output[row][b][a], res);
                }
            }
        }
    }
    else {
        // Case 3: Same-shape elementwise multiplication
        const uint32_t frame_size = getEncodedTotalSizeBytes(dataType) / input1_element_size;
        assert(frame_size > 0);
        const uint32_t blockCount = div_ceil(frame_size, blockSize);

        LData weights({blockCount, blockSize}, getDType(weightType.getElementType()));
        LData input({blockCount, blockSize}, getDType(dataType.getElementType()));
        LData output({blockCount, actBlockCount, actWidth}, getDType(outElType));

        For(auto b = slice.iterate(blockCount)) {
            IData idata = slice.iram.load(input[b]);
            WData wdata = slice.wram.load(weights[b]);
            PData pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);

            For(auto a = slice.iterate(actBlockCount)) {
                QData res = slice.act.rescaleClamp(
                    pdata[a], bdata, op.getShift(), 0, act_clip_min, act_clip_max
                );
                slice.store(output[b][a], res);
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, "mul", op.getInput1(), op.getInput2(), op.getScaleBias(), op.getInit(),
        slice.getCfgAttr(rewriter.getContext()), slice.getNdls()
    );

    return success();
}

} // namespace mlir::syna::torq
