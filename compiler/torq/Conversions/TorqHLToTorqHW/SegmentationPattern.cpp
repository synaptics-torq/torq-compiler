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

// To program the hardware for segmentation operation we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// The ALU will be on BYP mode for segmentation operation.
// The ALU will be working on 64 pixels each.
// The flow of data is as follows:
// max_mode_pixels : 64
// ALU info 64x4: A(64)U(u)A(a/64)
// 1. Read the input data in blocks of max_mode_pixels pixels.
//    max_mode_pixel is based on the vectorization mode 64x4, 32x8, 16x16.
// 2. Feed the input data to ALU and bypass the ALU operation
// 3. The ALU will output max_model_pixels of data.
// 4. The output is written back to LRAM using the special descriptor(S-tags)
//    to segment the data to 4 parts based on the position of the pixels(Even-even,
//    Even-odd, Odd-even, Odd-Odd). Each part is at a distance of frame_size/4.
// 5. Take the next block of 64 pixels and repeat the process.
// 6. Repeat the process for the next input channel
template <>
LogicalResult
SegmentationPattern::transform(torq_hl::SegmentationOp op, PatternRewriter &rewriter) const {
    // input
    auto input_type = llvm::dyn_cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());

    // relax input strides check
    // clang-format off
    // %alloc_14 = memref.alloc() : memref<1x224x28x28xi8, #synpu_hl<enc mem = lram>>
    // synpu_hl.load %subview_11 : memref<1x224x28x28xi8, strided<[401408, 784, 28, 1], offset: 175616>>
    //    to %alloc_14 : memref<1x224x28x28xi8, #synpu_hl<enc mem = lram>>
    // %alloc_15 = memref.alloc() : memref<1x224x28x28xi8, strided<[186368, 832, 28, 1]>, #synpu_hl<enc align = [0, 0, 64, 0, 0] mem = lram>>
    // "synpu_hl.segmentation"(%alloc_15, %alloc_12, %alloc_13, %alloc_14) <{input_zp = 0 : i32, output_max = 0 : i32, output_min = 0 : i32, output_zp = 0 : i32}> : (memref<1x224x28x28xi8, strided<[186368, 832, 28, 1]>, #synpu_hl<enc align = [0, 0, 64, 0, 0] mem = lram>>, memref<1x1x1x1xi8, #synpu_hl<enc mem = lram>>, memref<2xi32, #synpu_hl<enc mem = lram>>, memref<1x224x28x28xi8, #synpu_hl<enc mem = lram>>) -> ()
    // clang-format on
    //
    // alloc_14 is from subview which is not from customize encoding used for segmentationOp input
    // alloc_15 is from customize encoding used for segmentationOp output
    // we just need to make sure alloc_15 is padding for alignment

    auto inputStrides = getEncodedStridesElements(input_type);

    auto outputStrides = getEncodedStridesElements(output_type);

    int32_t ddat_width = 1;
    int32_t act_width = HwInfo::act_width;
    int32_t qdat_width = 1;

    if (input_type.getElementType().isInteger(16)) {
        ddat_width = 2;
        qdat_width = 2;
        act_width = HwInfo::act_width / qdat_width; // Activation reduces to half for int16
    }

    uint32_t max_input = HwInfo::max_input;
    max_input = max_input /
                ddat_width; // The total pixel that can be processed reduces based on the data width

    const uint32_t in_channel_offset = inputStrides[1];
    const uint32_t out_channel_offset = outputStrides[1];

    const uint32_t rows = input_shape[2];
    const uint32_t row_offset = input_shape[3];
    const uint32_t input_channels = input_shape[1];
    const uint32_t max_out_channels = findExactMultiple(input_channels, 4);
    assert(max_out_channels > 0);

    const uint32_t out_ch_split = div_ceil(input_channels, max_out_channels);
    const uint32_t in_frame_size = align_ceil(rows * row_offset, max_input);
    assert(in_frame_size > 0);
    const uint32_t total_px_block = div_ceil(in_frame_size, max_input);

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0,                                   // alu_d_unsigned
        0,                                   // alu_w_unsigned
        torq_hw::ACTMode::ACT,               // act_mode
        {0, 0, 0, 0},                        // act left shift
        0,                                   // ShiftFactor
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

    MemNdlDimsData ref = {
        {DimType::H, MemDimTag::X, input_shape[3], 0},
        {DimType::H, MemDimTag::Y, input_shape[2], 0},
        {DimType::H, MemDimTag::U, input_shape[1], 0}
    };

    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::B, ddat_width, 1},
        {DimType::L, MemDimTag::A, max_input, ddat_width},
        {DimType::H, MemDimTag::U, max_out_channels, in_channel_offset * ddat_width},
        {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
        {DimType::H, MemDimTag::U, out_ch_split, in_channel_offset * max_out_channels * ddat_width}
    };

    const uint32_t sub_frame_size = out_channel_offset / 4;
    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, qdat_width, 1},
        {DimType::L, MemDimTag::A, act_width * qdat_width, qdat_width},
        {DimType::H, MemDimTag::A, max_input / (act_width * qdat_width), 0},
        {DimType::H, MemDimTag::V, max_out_channels, sub_frame_size * 4 * qdat_width},
        {DimType::H, MemDimTag::A, total_px_block, 0},
        {DimType::H, MemDimTag::V, out_ch_split, max_out_channels * sub_frame_size * 4 * qdat_width
        },
        {DimType::S, MemDimTag::B, qdat_width, 1},
        {DimType::S, MemDimTag::X, 2, sub_frame_size * qdat_width},
        {DimType::S, MemDimTag::X, input_shape[3] / 2, qdat_width},
        {DimType::S, MemDimTag::Y, 2, sub_frame_size * 2 * qdat_width},
        {DimType::S, MemDimTag::Y, input_shape[2] / 2, (input_shape[3] / 2) * qdat_width}
    };

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, ddat_width, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / ddat_width, ddat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
         HwInfo::iram_width}
    };

    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::B, ddat_width, 1},
        {DimType::L, RegDimTag::D, max_input, ddat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
         HwInfo::iram_width}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S, (max_input / (act_width * qdat_width)),
         HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
         HwInfo::mac_count * HwInfo::pram_dsize}
    };

    Ndls ndls;
    ndls.add(NdlType::REF, ref);
    ndls.add(NdlType::DEDR, dedr);
    ndls.add(NdlType::DEQW, deqw);
    ndls.add(NdlType::CEDW, cedw);
    ndls.add(NdlType::CEDR, cedr);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                // Operation to replace
        "segmentation",    // Task name
        op.getInput(),     // Input tensor
        op.getWeights(),   // Weights
        op.getScaleBias(), // BiasScale tensor,
        op.getInit(),      // Output tensor initializer
        slice_cfg_attr,    // Slice configuration
        ndls               // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
