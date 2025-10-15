// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult ScatterPattern::transform(torq_hl::ScatterOp op, PatternRewriter &rewriter) const {
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());

    auto indices_type = llvm::cast<MemRefType>(op.getIndices().getType());
    auto indices_shape = indices_type.getShape();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());

    auto input_strides = getEncodedStridesElements(input_type);
    auto input_shape = input_type.getShape();
    int channelCount = input_shape.size() == 3 ? input_shape[1] : 1;
    int channelStride = input_shape.size() == 3 ? input_strides[1] : 0;
    // FIXME: will work only for i8, not for i16 or bf16
    // one should use getEncodedStridesBytes() instead
    int outChannelStride = input_shape.size() == 3 ? getEncodedStridesElements(output_type)[1] : 0;

    auto output_strides = getEncodedStridesElements(output_type);

    uint32_t entry_size = 1;
    uint32_t idx_size = 2;
    int indicesCount = indices_type.getNumElements();
    const uint32_t group_size = 1; // Only group size of 1 is supported for now
    LLVM_DEBUG({
        llvm::dbgs() << "\nindicesCount: " << indicesCount << " group_size: " << group_size << "\n";
    });

    uint32_t alu_width = alu_group_width / group_size;
    uint32_t max_input = 256 / alu_width;

    uint32_t max_entries = indices_shape[0] / group_size;

    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0xf,                                // alu_d_unsigned
        0,                                  // alu_w_unsigned
        ACTMode::ACT,                       // act_mode
        {0, 0, 0, 0},                       // act left shift
        0,                                  // shift_factor
        std::numeric_limits<int8_t>::min(), // output_min
        std::numeric_limits<int8_t>::max(), // output_max
        0,                                  // act_zero_point
        0,                                  // no_p_clear
        0,                                  // no_p_output
        0, 0, 0, 0,                         // Kernel
        0, 0, 0, 0,                         // Pad
        0,                                  // pad_value
        1,                                  // stride
        0                                   // stride_offset

    );

    Ndls ndls;

    MemNdlDimsData ref{{DimType::H, MemDimTag::X, group_size * max_entries}};
    ndls.add(NdlType::REF, ref);

    // dedr reads the replacement values
    MemNdlDimsData dedr{
        {DimType::L, MemDimTag::X, entry_size, 1},
        {DimType::L, MemDimTag::G, group_size, 0},
        {DimType::H, MemDimTag::X, max_entries, entry_size},
        {DimType::H, MemDimTag::O, channelCount, channelStride}
    };
    ndls.add(NdlType::DEDR, dedr);

    // deqw replaces the value in LRAM address with the dedr read value
    MemNdlDimsData deqw{
        {DimType::L, MemDimTag::B, 1, 1},
        {DimType::L, MemDimTag::D, entry_size, 1},
        {DimType::H, MemDimTag::X, group_size, 0},
        {DimType::H, MemDimTag::X, max_entries, 0},
        {DimType::H, MemDimTag::O, channelCount, outChannelStride}
    };
    ndls.add(NdlType::DEQW, deqw);

    // debr set-0 reads dummy bias values
    ndls.add(
        NdlType::DEBR,
        {{DimType::L, MemDimTag::B, HwInfo::bdat_width, 1},
         {DimType::H, MemDimTag::X, group_size, 0},
         {DimType::H, MemDimTag::X, max_entries, 0},
         {DimType::H, MemDimTag::O, channelCount, 0}},
        0, 0, 'R', 1
    );

    // debr set-1 reads the indices pointing to data to be updated
    ndls.add(
        NdlType::DEBR,
        {{DimType::L, MemDimTag::B, idx_size, 1},
         {DimType::H, MemDimTag::X, group_size, idx_size},
         {DimType::H, MemDimTag::X, max_entries, idx_size * group_size},
         {DimType::H, MemDimTag::O, channelCount, 0}},
        0, 1, 'R', 1
    );

    RegNdlDimsData acbw{
        {DimType::L, RegDimTag::B, HwInfo::bdat_width, 1},
        {DimType::H, RegDimTag::T, 0, HwInfo::bbus_width}
    };
    ndls.add(NdlType::ACBW, acbw);

    RegNdlDimsData acbr{
        {DimType::L, RegDimTag::B, HwInfo::bdat_width, 1},
        {DimType::H, RegDimTag::T, 0, HwInfo::bbus_width}
    };
    ndls.add(NdlType::ACBR, acbr);

    RegNdlDimsData cedw{
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / group_size, 1},
        {DimType::L, RegDimTag::G, 1, HwInfo::iram_seg_width / group_size},
        {DimType::H, RegDimTag::S, 1, 1},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
        {DimType::H, RegDimTag::N, 1, 0},
        {DimType::H, RegDimTag::T, max_entries * channelCount, HwInfo::iram_width}
    };
    ndls.add(NdlType::CEDW, cedw);

    RegNdlDimsData cedr{
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, alu_width, 1},
        {DimType::L, RegDimTag::G, max_input, HwInfo::iram_seg_width / group_size},
        {DimType::H, RegDimTag::S, 1, 1},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
        {DimType::H, RegDimTag::N, 1, 0},
        {DimType::H, RegDimTag::T, max_entries * channelCount, HwInfo::iram_width}
    };
    ndls.add(NdlType::CEDR, cedr);

    RegNdlDimsData cepr{
        {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
        {DimType::L, RegDimTag::D, 64, HwInfo::pdat_width},
        {DimType::H, RegDimTag::S, 1, HwInfo::pdat_width * 64},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::pram_width},
        {DimType::H, RegDimTag::N, 1, 0},
        {DimType::H, RegDimTag::T, max_entries * channelCount,
         HwInfo::pram_width * HwInfo::pram_depth}
    };
    ndls.add(NdlType::CEPR, cepr);

    RegNdlDimsData acpr{
        {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pdat_width},
        {DimType::H, RegDimTag::S, group_size, HwInfo::pdat_width * alu_width},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::pram_width},
        {DimType::H, RegDimTag::N, 1, 0},
        {DimType::H, RegDimTag::T, max_entries * channelCount,
         HwInfo::pram_width * HwInfo::pram_depth}
    };
    ndls.add(NdlType::ACPR, acpr);

    RegNdlDimsData acpw;
    ndls.add(NdlType::ACPW, acpw);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op, // Operation to replace
        "scatter",
        // Task name
        op.getInput(), ValueRange{}, op.getScaleBias(), op.getInit(), ValueRange{}, op.getIndices(),

        ValueRange{}, slice_cfg_attr, ndls
    );
    return success();
}

} // namespace mlir::syna::torq
