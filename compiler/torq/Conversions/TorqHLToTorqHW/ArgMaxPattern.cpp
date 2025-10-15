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
LogicalResult ArgMaxPattern::transform(torq_hl::ArgMaxOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    int axis = op.getAxis();

    /*
        In the output, TOSA argmax removes dimension.
        So the output tensor will have a rank of (rank(input_shape) - 1).
        Currently, we cannot run a 2D argmax due to a compiler limitation
        where the input and output channel counts must be the same.
        As a result, 3D input is the minimum required at the moment.

        This limitation is defined in : Conversions / TorqHLToTorqHW / Patterns.cpp
        (class StoreOpPattern : public OpRewritePattern<torq_hl::StoreOp>)
    */
    // Ensure input is 3D
    assert(input_shape.size() == 3 && "ArgMax operation requires a 3D input tensor.");

    // Ensure batch size is 1 (0th dimension should be 1)
    assert(input_shape[0] == 1 && "ArgMax currently only supports batch size of 1.");

    const uint32_t rows = input_shape[1];
    const uint32_t row_offset = input_shape[2];

    // Ensure reduction axis is 1 and row_offset is a multiple of HwInfo::max_input
    assert(axis == 1 && (row_offset % HwInfo::max_input == 0));

    const int32_t total_px_block = div_ceil(row_offset, HwInfo::max_input);
    assert(total_px_block > 0);

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::AMAX, ALUOp1Mode::AMAX, ALUOp1Mode::AMAX, ALUOp1Mode::AMAX},
        0,                                   // alu_d_unsigned
        0,                                   // alu_w_unsigned
        ACTMode::ACT,                        // act_mode
        {0, 0, 0, 0},                        // act left shift
        16,                                  // shift_factor
        std::numeric_limits<int32_t>::min(), // output_min
        std::numeric_limits<int32_t>::max(), // output_max
        0,                                   // output_zp
        false,                               // no_p_clear
        false,                               // no_p_output
        0, 0, 0, 0,                          // kernel lrtb
        0, 0, 0, 0,                          // pad lrtb
        0,                                   // pad_value
        1                                    // stride
    );

    // Not used in this kernel: DEWR, ACPW
    Ndls ndls;

    ndls.add(
        NdlType::REF, {{DimType::H, MemDimTag::A, row_offset}, {DimType::H, MemDimTag::U, rows}}
    );

    ndls.add(
        NdlType::DEDR, {{DimType::L, MemDimTag::A, HwInfo::max_input, 1},
                        {DimType::H, MemDimTag::U, rows, row_offset},
                        {DimType::H, MemDimTag::A, total_px_block, HwInfo::max_input}}
    );

    ndls.add(
        NdlType::DEQW, {{DimType::L, MemDimTag::B, 4, 1},
                        {DimType::L, MemDimTag::A, HwInfo::act_width, 1 * 4},
                        {DimType::H, MemDimTag::A, HwInfo::max_input / HwInfo::act_width, 16 * 4},
                        {DimType::H, MemDimTag::A, total_px_block, HwInfo::max_input * 4}}
    );

    ndls.add(
        NdlType::DEBR, {{DimType::L, MemDimTag::B, HwInfo::breg_width, 1},
                        {DimType::H, MemDimTag::O, total_px_block, 0}}
    );

    // ({B8Dn}S1M1W1N1)T*
    ndls.add(
        NdlType::ACBW, {{DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
                        {DimType::L, RegDimTag::D, 1, HwInfo::breg_width},
                        {DimType::H, RegDimTag::S, 1, HwInfo::breg_width},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::bbus_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, total_px_block, HwInfo::bbus_width}}
    );

    // ({B8DnGn}S1MnW1N1)T*
    ndls.add(
        NdlType::ACBR, {{DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
                        {DimType::L, RegDimTag::D, 1, HwInfo::breg_width},
                        {DimType::L, RegDimTag::G, 1, 0},
                        {DimType::H, RegDimTag::S, 1, HwInfo::breg_width},
                        {DimType::H, RegDimTag::M, HwInfo::max_input / HwInfo::act_width, 0}, // 4
                        {DimType::H, RegDimTag::W, 1, HwInfo::bbus_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, total_px_block, HwInfo::bbus_width}}
    );

    // ({B1D72}SnM1WnN1)T*
    ndls.add(
        NdlType::CEDW, {{DimType::L, RegDimTag::B, 1, 1},
                        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width, 1},
                        {DimType::H, RegDimTag::S, 1, HwInfo::iram_seg_width},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, rows * total_px_block, HwInfo::iram_width}}
    );

    // ({B1DnGn}SnMnWnNn)T*
    ndls.add(
        NdlType::CEDR, {{DimType::L, RegDimTag::B, 1, 1},
                        {DimType::L, RegDimTag::D, HwInfo::max_input, 1},
                        {DimType::L, RegDimTag::G, 1, HwInfo::iram_seg_width},
                        {DimType::H, RegDimTag::S, 1, 1},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, rows * total_px_block, HwInfo::iram_width}}
    );

    // ({B1D32}S1M1WnN1)T*, Ss
    ndls.add(
        NdlType::CEWW,
        {{DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}}
    );

    // ({B1DnGn}SnMnWnNn)T*, Ss
    ndls.add(
        NdlType::CEWR,
        {{DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}}
    );

    constexpr uint32_t pram_width_bytes = HwInfo::pram_dsize * HwInfo::pram_width;
    // ({B4D256}S1M1WnNn)Tn
    ndls.add(
        NdlType::CEPR,
        {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
         {DimType::L, RegDimTag::D, HwInfo::max_input, HwInfo::pram_dsize},
         {DimType::H, RegDimTag::S, 1, pram_width_bytes},
         {DimType::H, RegDimTag::M, 1, 0},
         {DimType::H, RegDimTag::W, 1, pram_width_bytes},
         {DimType::H, RegDimTag::N, rows, 0},
         {DimType::H, RegDimTag::T, total_px_block, pram_width_bytes * HwInfo::pram_depth}}
    ); // 1

    // ({BnDn}SnRnM1WnN1)Tn, DsSs
    ndls.add(
        NdlType::ACPR,
        {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
         {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
         {DimType::H, RegDimTag::S, HwInfo::max_input / HwInfo::act_width,
          HwInfo::pram_dsize * HwInfo::act_width},
         {DimType::H, RegDimTag::M, 1, 0},
         {DimType::H, RegDimTag::W, 1, pram_width_bytes},
         {DimType::H, RegDimTag::N, 1, 0},
         {DimType::H, RegDimTag::T, total_px_block, pram_width_bytes * HwInfo::pram_depth}}
    );

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                            // Operation to replace
        "arg_max",                     // Task name
        ValueRange{op.getInput()},     // Input tensor
        ValueRange{},                  // Weights
        ValueRange{op.getScaleBias()}, // BiasScale tensor,
        op.getInit(),                  // Output tensor initializer
        ValueRange{},                  // Symbols
        slice_cfg_attr,                // Slice configuration
        ndls
    );

    return success();
}

} // namespace mlir::syna::torq
