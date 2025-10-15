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

// This is an example of an independent table lookup operation.
// - The table contains 512 values for lookup.
// - Input is an int16 9.7 fixed-point value, and the output is an interpolated int32 result.
// - The upper signed 9 bits of the input are used to look up the index in the range [0, 511], while
//   the lower 7 bits represent fractional values and are used to calculate the slope during lookup.
// - Since this is an independent operation, the ALU will operate in bypass mode, and the ACT will
//   function in its default mode.
// - ACT can output only 2 lookup values at a time.
template <>
LogicalResult TablePattern::transform(torq_hl::TableOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    Type elementType = input_type.getElementType();
    const uint8_t data_bytes = elementType.getIntOrFloatBitWidth() / 8;

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    Type outputElementType = output_type.getElementType();
    const uint32_t output_data_bytes = outputElementType.getIntOrFloatBitWidth() / 8;

    uint32_t total_px = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_px *= input_shape[i];
    }

    const int32_t total_px_block = div_ceil(total_px, HwInfo::table_lookup_count);
    ArrayRef<int32_t> table = op.getTableAttr().asArrayRef();

    uint32_t act_sum_bits = 0;
    SmallVector<uint32_t> act_lsh = {0, 0, 0, 0};
    int32_t alu_d_unsigned = 0x0;

    if (data_bytes == 2) {
        act_lsh = {0, 8, 0, 8};
        act_sum_bits = 48;
        alu_d_unsigned = 0x5; // treat low 8b of 16b as unsigned
    }

    auto sliceCfgAttr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC},
        alu_d_unsigned,                      // alu_d_unsigned
        0,                                   // alu_w_unsigned
        ACTMode::ACT,                        // act_mode
        act_lsh,                             // act left shift
        0,                                   // shift_factor
        std::numeric_limits<int32_t>::min(), // output_min
        std::numeric_limits<int32_t>::max(), // output_max
        0,                                   // output_zp
        false,                               // no_p_clear
        false,                               // no_p_output
        0, 0, 0, 0,                          // kernel lrtb
        0, 0, 0, 0,                          // pad lrtb
        0,                                   // pad_value
        1,                                   // stride
        0,                                   // stride_offset
        act_sum_bits,                        // act_sum_bits
        table                                // tabel
    );

    MemNdlDimsData ref = {
        {DimType::H, MemDimTag::X, total_px_block, 0}, {DimType::H, MemDimTag::Y, data_bytes, 0}
    };

    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::A, HwInfo::table_lookup_count * data_bytes, 1},
        {DimType::H, MemDimTag::A, total_px_block, HwInfo::table_lookup_count * data_bytes}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, output_data_bytes, 1},
        {DimType::L, MemDimTag::D, HwInfo::table_lookup_count, output_data_bytes},
        {DimType::H, MemDimTag::A, total_px_block, HwInfo::table_lookup_count * output_data_bytes}
    };

    MemNdlDimsData debr;

    if (data_bytes == 1) {
        debr = {
            {DimType::L, MemDimTag::B, HwInfo::breg_width, 1},
            {DimType::H, MemDimTag::A, 1, HwInfo::breg_width}
        };
    }

    RegNdlDimsData acbw;
    RegNdlDimsData acbr;

    if (data_bytes == 1) {

        acbw = {
            {DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
            {DimType::H, RegDimTag::T, 1, HwInfo::bbus_width}
        };

        acbr = {
            {DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
            {DimType::H, RegDimTag::M, total_px_block, 0}
        };
    }

    RegNdlDimsData acpw;
    if (data_bytes == 2) {
        acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
            {DimType::L, RegDimTag::D, data_bytes, HwInfo::pdat_width},
            {DimType::L, RegDimTag::G, HwInfo::act_width / data_bytes,
             HwInfo::pdat_width * data_bytes},
            {DimType::H, RegDimTag::T}
        };
    }

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, data_bytes, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / data_bytes, data_bytes},
        {DimType::H, RegDimTag::S, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::B, data_bytes, 1},
        {DimType::L, RegDimTag::D, 8 / data_bytes, data_bytes},
        {DimType::L, RegDimTag::G, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::N, total_px_block, 0}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::mac_count},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
        {DimType::L, RegDimTag::D, HwInfo::table_lookup_count * data_bytes, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::table_lookup_count * data_bytes},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    Ndls ndls;
    ndls.add(NdlType::REF, ref);
    ndls.add(NdlType::DEDR, dedr);
    ndls.add(NdlType::DEBR, debr);
    ndls.add(NdlType::DEQW, deqw);
    ndls.add(NdlType::CEDW, cedw);
    ndls.add(NdlType::CEDR, cedr);
    ndls.add(NdlType::CEWR, cewr);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACBW, acbw);
    ndls.add(NdlType::ACBR, acbr);
    ndls.add(NdlType::ACPR, acpr);
    ndls.add(NdlType::ACPW, acpw);

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        "table",                       // Operation to replace
        ValueRange{op.getInput()},     // Input tensor
        ValueRange{},                  // Weights
        ValueRange{op.getScaleBias()}, // BiasScale tensor,
        ValueRange{op.getInit()},      // Output tensor initializer
        ValueRange{},                  // Symbols,
        sliceCfgAttr, ndls
    );

    return success();
}

} // namespace mlir::syna::torq
