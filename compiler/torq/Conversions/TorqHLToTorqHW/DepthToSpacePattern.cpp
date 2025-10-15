// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl-gather"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult
DepthToSpacePattern::transform(torq_hl::DepthToSpaceOp op, PatternRewriter &rewriter) const {

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    // weight
    // auto weight_type = llvm::cast<MemRefType>(op.getWeights().getType());
    // auto weight_shape = weight_type.getShape();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();
    auto output_strides = getEncodedStridesElements(output_type);

    int32_t ddat_width = input_type.getElementType().getIntOrFloatBitWidth() / 8;
    int32_t wdat_width = ddat_width;
    NumberFormat alu_format = NumberFormat::I;
    NumberFormat act_format = NumberFormat::I;

    auto input_strides = getEncodedStridesElements(input_type);

    const int block_size = op.getBlockSize();
    const auto block_skip_offset = op.getModeType() == torq_hl::DepthToSpaceModeEnum::CRD
                                       ? input_strides[1] * ddat_width
                                       : input_strides[1] * ddat_width * output_shape[1];
    const auto channel_skip_offset = op.getModeType() == torq_hl::DepthToSpaceModeEnum::CRD
                                         ? input_strides[1] * ddat_width * block_size * block_size
                                         : input_strides[1] * ddat_width;

    SmallVector<uint32_t> act_lsh = {0, 8, 0, 8};
    int32_t alu_d_unsigned = 0xf;
    int32_t alu_w_unsigned = 0xf;
    uint32_t act_sum_bits = 32;
    auto rounding_mode = RoundingMode::OFF;
    const int num_byte_repeat = 2;
    const int interleave_weight_size = HwInfo::wram_seg_width / wdat_width;
    const int num_data = std::min(
        static_cast<long>(input_shape[3]),
        static_cast<long>(div_ceil(interleave_weight_size, num_byte_repeat))
    );

    if (ddat_width == 2) {
        act_lsh[0] = 0;
        act_lsh[1] = 8;
        act_lsh[2] = 16;
        act_lsh[3] = 24;

        alu_d_unsigned = 0xf;
        alu_w_unsigned = 0xf;
        act_sum_bits = 48;
    }
    const int64_t total_data_block =
        input_shape[2] * div_ceil(input_shape[3], num_data) * output_shape[1];
    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL},
        {ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC},
        alu_d_unsigned,                      // alu_d_unsigned
        alu_w_unsigned,                      // alu_w_unsigned
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
        0,                                   // stride
        0,                                   // stride_offset
        rounding_mode,                       // rounding_mode
        WeightFormat::SI,                    // weight_format
        0, 0,                                // alu_disable, act_disable
        alu_format,                          // alu_format
        act_format,                          // act_format
        act_sum_bits,                        // act_sum_bits
        {}                                   // table
    );

    Ndls ndls;

    // ref
    MemNdlDimsData ref = {
        {DimType::H, MemDimTag::U, input_shape[1], 0},
        {DimType::H, MemDimTag::X, output_shape[3], 0},
        {DimType::H, MemDimTag::Y, output_shape[2], 0}
    };
    ndls.add(NdlType::REF, ref);

    ndls.add(
        NdlType::DEDR,
        {
            {DimType::L, MemDimTag::B, 1, 1},
            {DimType::L, MemDimTag::A, interleave_weight_size, 1}, // 16 for int16, 8 for int8
            {DimType::H, MemDimTag::O, block_size, interleave_weight_size},
            {DimType::H, MemDimTag::O, block_size, 0},
            {DimType::H, MemDimTag::A, total_data_block, 0},
        },
        0
    );

    MemNdlDimsData dewr = {
        {DimType::L, MemDimTag::B, ddat_width, 1},
        // Take block of data from columns
        {DimType::L, MemDimTag::A, num_data, ddat_width},
        // Take first 2 channel data
        {DimType::H, MemDimTag::O, block_size, block_skip_offset},
        // Iterate till 1 row completed
        {DimType::H, MemDimTag::O, div_ceil(input_shape[3], num_data), num_data * ddat_width},
        // Take next 2 channel data and repeat
        {DimType::H, MemDimTag::O, block_size, block_size * block_skip_offset},
        // Iterate for all rows
        {DimType::H, MemDimTag::A, input_shape[2], input_shape[3] * ddat_width},
        // Repeat for next set of 4 channels
        {DimType::H, MemDimTag::V, output_shape[1], channel_skip_offset}
    };
    ndls.add(NdlType::DEWR, dewr);

    MemNdlDimsData debr;

    auto act_writes = HwInfo::act_width / (ddat_width * block_size);
    auto data_repeats = div_ceil(num_data, act_writes);
    MemNdlDimsData deqw = {
        // Write the fused data
        {DimType::L, MemDimTag::B, ddat_width * block_size, 1},
        {DimType::L, MemDimTag::A, act_writes, ddat_width * block_size},
        // Iterate till 1 alu processing completed
        {DimType::H, MemDimTag::O, data_repeats, act_writes * ddat_width * block_size},
        // Iterate till 1 row completed
        {DimType::H, MemDimTag::O, div_ceil(input_shape[3], num_data),
         data_repeats * act_writes * ddat_width * block_size},
        // Take next 2 channel data and repeat
        {DimType::H, MemDimTag::O, block_size, output_shape[3] * ddat_width},
        // Jump 2 rows after writing row channels
        {DimType::H, MemDimTag::O, input_shape[2], output_shape[3] * block_size * ddat_width},
        // Repeat for next set of 4 channels
        {DimType::H, MemDimTag::V, output_shape[1], output_strides[1] * ddat_width}
    };
    ndls.add(NdlType::DEQW, deqw);

    ndls.add(
        NdlType::CEDW, {{DimType::L, RegDimTag::B},
                        {DimType::L, RegDimTag::D, 36, wdat_width},
                        {DimType::L, RegDimTag::G},
                        {DimType::H, RegDimTag::S, 1, HwInfo::iram_seg_width},
                        {DimType::H, RegDimTag::M},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, total_data_block * block_size * block_size,
                         HwInfo::iram_width * wdat_width}}
    );

    ndls.add(
        NdlType::CEDR, {{DimType::L, RegDimTag::I, ddat_width},
                        {DimType::L, RegDimTag::B, 1, 1},
                        {DimType::L, RegDimTag::D, interleave_weight_size, 1},
                        {DimType::H, RegDimTag::S},
                        {DimType::H, RegDimTag::M},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, total_data_block * block_size * block_size,
                         HwInfo::iram_width * wdat_width}}
    );

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::wram_seg_width, ddat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_data_block * block_size * block_size,
         HwInfo::wram_seg_width * ddat_width}
    };
    ndls.add(NdlType::CEWW, ceww);

    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, ddat_width, 1},
        {DimType::L, RegDimTag::J, num_byte_repeat, 0},
        {DimType::L, RegDimTag::D, num_data, ddat_width},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_data_block * block_size * block_size,
         HwInfo::wram_seg_width * ddat_width}
    };
    ndls.add(NdlType::CEWR, cewr);

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, HwInfo::pram_depth, HwInfo::mac_count * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::N, block_size, 0},
        {DimType::H, RegDimTag::T, total_data_block * block_size,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::CEPR, cepr);

    RegNdlDimsData acbw;

    RegNdlDimsData acbr;
    auto partial_d_group = HwInfo::act_width / (block_size * ddat_width);
    RegNdlDimsData acpw = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, ddat_width * block_size, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G, partial_d_group, ddat_width * block_size * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::T, total_data_block * block_size, 1}
    };
    ndls.add(NdlType::ACPW, acpw);

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S, data_repeats, HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_data_block * block_size,
         HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,              // Operation to replace
        "depthtospace",  // Task name
        op.getWeights(), // Input tensor
        op.getInput(),   // Weights
        nullptr,         // BiasScale tensor,
        op.getInit(),    // Output tensor initializer
        slice_cfg_attr,  // Slice configuration
        ndls             // NDLs
    );

    return success();
}
} // namespace mlir::syna::torq
