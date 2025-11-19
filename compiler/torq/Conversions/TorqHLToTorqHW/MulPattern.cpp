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
            hasScalar = true;
            weightType = type1;
            dataType = type2;
        }
        else if (rank2 == 1) {
            hasScalar = true;
            weightType = type2;
            dataType = type1;
        }
    }

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto outElType = output_type.getElementType();

    // alu and act process dbus/wbus two 8/16-bit elements size at a time
    uint32_t data_width = input1_element_size * input2_element_size;

    int32_t act_clip_min = op.getOutputMin();
    int32_t act_clip_max = op.getOutputMax();

    if (elType1.isBF16() && elType2.isBF16()) {
        data_width = input1_element_size;
        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }

    const uint32_t frame_size = getEncodedTotalSizeBytes(dataType) / input1_element_size;
    assert(frame_size > 0);

    const uint32_t alu_group_width = 32;
    const uint32_t blockSize = alu_group_width / input1_element_size;
    const uint32_t blockCount = div_ceil(frame_size, blockSize);
    // for int16 data_width=4, divided by 4 means we need 4 partial 32bit to compute each 48bit
    // result with the act_lsh for int8/bf16 use natual data_width
    const uint32_t actWidth = HwInfo::act_width / data_width;
    const uint32_t actBlockCount = div_ceil(blockSize, actWidth);

    LData weights({1}, getDType(weightType.getElementType()));
    if (!hasScalar) {
        weights = LData({blockCount, blockSize}, getDType(weightType.getElementType()));
    }

    LData biasScale({2}, elType1.isBF16() ? DType::bf16 : DType::int32);

    Slice slice;
    BData bdata = slice.bram.load(biasScale);

    if (hasScalar) {

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
    else {
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
