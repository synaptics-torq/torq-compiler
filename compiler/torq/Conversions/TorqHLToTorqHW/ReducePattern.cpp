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
LogicalResult ReducePattern::transform(torq_hl::ReduceOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    int axis = op.getAxis();
    Type elementType = input_type.getElementType();
    const uint32_t data_bytes = elementType.getIntOrFloatBitWidth() / 8;

    // reduce dim is always the first shape element
    assert(axis == 0 && "Only support reduce axis 0");

    const uint32_t reduce_dim = input_shape[axis];

    uint32_t retain_dim = 1;
    for (int i = 1; i < input_shape.size(); ++i) {
        retain_dim *= input_shape[i];
    }

    // Since this kernel supports multiple types of reductions, we don't always have 16 output
    // values per cycle. For example, in the case of reduce_sum with 32-bit data, we only get 4
    // values per cycle.
    uint8_t output_elements_per_cycle_divisor = 1;
    const int32_t total_px_block = div_ceil(retain_dim, HwInfo::max_input);
    assert(total_px_block > 0);

    torq_hw::ALUOp1Mode hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
    std::string opName = op.getName().str();

    int32_t alu_d_unsigned;
    if (data_bytes == 1) {
        alu_d_unsigned = 0x0;
    }
    else if (data_bytes == 2) {
        alu_d_unsigned = 0x5;
    }
    else {
        alu_d_unsigned = 0x7;
    }

    uint32_t act_sum_bits = 32;
    SmallVector<uint32_t> act_lsh = {0, 0, 0, 0};
    NumberFormat alu_format = NumberFormat::I;
    NumberFormat act_format = NumberFormat::I;
    bool act_is_float = false;
    if (elementType.isBF16()) {
        alu_format = NumberFormat::BF;
        act_sum_bits = 0;
    }

    if (opName == "reduce_sum") {
        hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
    }
    else if (opName == "reduce_min") {
        hwOp1Mode = torq_hw::ALUOp1Mode::MIN;
    }
    else if (opName == "reduce_max") {
        hwOp1Mode = torq_hw::ALUOp1Mode::MAX;
    }
    else if (opName == "reduce_and") {
        act_sum_bits = 0;
        hwOp1Mode = torq_hw::ALUOp1Mode::BAND;
    }
    else if (opName == "reduce_or") {
        act_sum_bits = 0;
        hwOp1Mode = torq_hw::ALUOp1Mode::BOR;
    }
    else if (opName == "reduce_xor") {
        act_sum_bits = 0;
        hwOp1Mode = torq_hw::ALUOp1Mode::BXOR;
    }
    else if (opName == "reduce_mul") {
        act_sum_bits = 16;
        act_format = NumberFormat::BF;
        hwOp1Mode = torq_hw::ALUOp1Mode::MUL;
        act_is_float = true;
        output_elements_per_cycle_divisor = 2;
    }
    else {
        llvm::errs() << "Unsupported reduce op: " << op.getName() << "\n";
        return failure();
    }

    // only for reduceSum 32-bit data
    if (data_bytes == 4 && (hwOp1Mode == torq_hw::ALUOp1Mode::ACC)) {
        act_lsh = {0, 8, 16, 24};
        output_elements_per_cycle_divisor = 4;
        act_sum_bits = 48;
    }
    // int32 min/max
    int32_t output_min_i = std::numeric_limits<int32_t>::min();
    int32_t output_max_i = std::numeric_limits<int32_t>::max();

    // special handling for float min/max
    float min_f = std::numeric_limits<float>::lowest();
    int32_t output_min_f = *reinterpret_cast<int32_t *>(&min_f);
    float max_f = std::numeric_limits<float>::max();
    int32_t output_max_f = *reinterpret_cast<int32_t *>(&max_f);

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {hwOp1Mode, hwOp1Mode, hwOp1Mode, hwOp1Mode},
        alu_d_unsigned,                             // alu_d_unsigned
        0,                                          // alu_w_unsigned
        ACTMode::ACT,                               // act_mode
        act_lsh,                                    // act left shift
        0,                                          // shift_factor
        act_is_float ? output_min_f : output_min_i, // output_min
        act_is_float ? output_max_f : output_max_i, // output_max
        op.getOutputZp(),                           // output_zp
        0x0,                                        // alu_disable
        0x0,                                        // act_disable
        alu_format,                                 // alu_format
        act_format,                                 // act_format
        act_sum_bits                                // act_sum_bits
    );

    MemNdlDimsData ref = {
        {DimType::H, MemDimTag::A, retain_dim}, {DimType::H, MemDimTag::U, reduce_dim}
    };

    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::A, HwInfo::max_input, 1},
        {DimType::H, MemDimTag::U, reduce_dim, retain_dim * data_bytes},
        {DimType::H, MemDimTag::A, data_bytes, HwInfo::max_input},
        {DimType::H, MemDimTag::A, total_px_block, HwInfo::max_input * data_bytes}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, data_bytes, 1},
        {DimType::L, MemDimTag::A, HwInfo::act_width / output_elements_per_cycle_divisor, data_bytes
        },
        {DimType::H, MemDimTag::A,
         (HwInfo::max_input * output_elements_per_cycle_divisor) / HwInfo::act_width,
         (16 / output_elements_per_cycle_divisor) * data_bytes},
        {DimType::H, MemDimTag::A, total_px_block, HwInfo::max_input * data_bytes}
    };

    // only M.n matters, how many times to repeat bias
    RegNdlDimsData acbr = {
        {DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
        {DimType::L, RegDimTag::D, 1, HwInfo::breg_width},
        {DimType::L, RegDimTag::G, 1, 0},
        {DimType::H, RegDimTag::S, 1, HwInfo::breg_width},
        {DimType::H, RegDimTag::M, HwInfo::max_input / HwInfo::act_width, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    // only S.n matters, number of output step within a conceptional word
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
        {DimType::L, RegDimTag::D, HwInfo::max_input / data_bytes, data_bytes},
        {DimType::H, RegDimTag::G, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::S, 1},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };

    // Handle special case for auto-generated int8 weights (Ws):
    // - For ACC mode:  pattern is [1, 1, 1, 1, ...]
    // - For SACC mode: pattern is [1, -1, 1, -1, ...]
    // These weights are prefilled as 16-bit values (low byte + high byte).
    // Load the low byte first, then the high byte in two steps to reconstruct correctly.
    RegNdlDimsData cewr;
    if (hwOp1Mode == torq_hw::ALUOp1Mode::ACC || hwOp1Mode == torq_hw::ALUOp1Mode::SACC) {
        if (elementType.isBF16()) {
            cewr = {{DimType::L, RegDimTag::B, 2, 1}, {DimType::H, RegDimTag::L, 2, 0}};
        }
        else {
            cewr = {{DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::S, 2, 1}};
        }
        cewr.push_back(
            {DimType::H, RegDimTag::N, (total_px_block * data_bytes) * (reduce_dim / 2), 0}
        );
        cewr.push_back({DimType::H, RegDimTag::T, 1, HwInfo::wram_width});
    }

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::max_input, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::max_input},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::pram_dsize * HwInfo::max_input},
        {DimType::H, RegDimTag::N, reduce_dim, 0},
        {DimType::H, RegDimTag::T, total_px_block * data_bytes,
         HwInfo::pram_dsize * HwInfo::max_input * HwInfo::pram_depth}
    };

    // Explanation of the formula:
    // (HwInfo::max_input / HwInfo::act_width) / (data_bytes / output_elements_per_cycle_divisor)
    //
    // In the case of reduce_sum:
    // Each 32-bit value is represented using 4 words (each word is 1 byte).
    // All 64 word must be passed to the ACT unit to correctly form each word.
    // Therefore, we configure `acpw` to group 4 words by performing a left shift.
    //
    // In other reduction cases, such as reduce_max or reduce_min:
    // The ALU unit only generates 16 int32 values out of 64 input int32 values.
    // So, `acpr` needs only 1 cycle, and this is adjusted by dividing with `data_bytes`.
    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, data_bytes, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S,
         (HwInfo::max_input / HwInfo::act_width) / (data_bytes / output_elements_per_cycle_divisor),
         HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * data_bytes,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    RegNdlDimsData acpw;
    if (data_bytes == 4 &&
        (hwOp1Mode == torq_hw::ALUOp1Mode::ACC || hwOp1Mode == torq_hw::ALUOp1Mode::SACC)) {
        acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
            {DimType::L, RegDimTag::D, data_bytes, HwInfo::pdat_width},
            {DimType::L, RegDimTag::G, HwInfo::act_width / data_bytes,
             HwInfo::pdat_width * data_bytes},
            {DimType::H, RegDimTag::T, total_px_block * data_bytes,
             HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
        };
    }
    else if (act_format == NumberFormat::I) {
        acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
            {DimType::L, RegDimTag::D, 1, HwInfo::pdat_width},
            {DimType::L, RegDimTag::G, HwInfo::act_width, HwInfo::pdat_width},
            {DimType::H, RegDimTag::T, total_px_block * data_bytes,
             HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
        };
    }
    else {
        acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
            {DimType::L, RegDimTag::D, 1, HwInfo::pdat_width},
            {DimType::L, RegDimTag::G, HwInfo::act_width, HwInfo::pdat_width},
            {DimType::H, RegDimTag::T, total_px_block * (8 / data_bytes),
             HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
        };
    }

    Ndls ndls;
    ndls.add(NdlType::REF, ref);
    ndls.add(NdlType::DEDR, dedr);
    ndls.add(NdlType::DEQW, deqw);
    ndls.add(NdlType::CEDW, cedw);
    ndls.add(NdlType::CEDR, cedr);
    ndls.add(NdlType::CEWW, ceww);
    ndls.add(NdlType::CEWR, cewr);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACBR, acbr);
    ndls.add(NdlType::ACPR, acpr);
    ndls.add(NdlType::ACPW, acpw);

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                            // Operation to replace
        op.getName(),                  // Task name
        ValueRange{op.getInput()},     // Input tensor
        ValueRange{},                  // Weights
        ValueRange{op.getScaleBias()}, // BiasScale tensor,
        ValueRange{op.getInit()},      // Output tensor initializer
        ValueRange{},                  // Symbols
        slice_cfg_attr,                // Slice configuration
        ndls                           // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
