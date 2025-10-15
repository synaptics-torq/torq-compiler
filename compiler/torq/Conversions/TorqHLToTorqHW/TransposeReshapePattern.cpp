// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-transpose-reshape"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

void groupContinuousTransposeReshapeDim(
    torq_hl::TransposeReshapeOp op, llvm::ArrayRef<int64_t> inputShape,
    llvm::ArrayRef<int64_t> outputShape, int32_t &leftDimSize, int32_t &transposeDimSize,
    int32_t &rightDimSize
) {
    int32_t kernel_size = op.getKernel()[1];
    leftDimSize = inputShape[0] * inputShape[1];
    transposeDimSize = kernel_size;
    rightDimSize = outputShape[3];
}

// To program the hardware for TransposeReshape we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// This pattern converts, input 1*1*1*W for conv1d having stride S and Filtersize K
//   to K * (W/S). W/S is the output_width
// The DEDR is taken viewing 1*1*1*W data as output_width * S
// For transposeReshape the ALU will be working on BYP.
// The ALU will be working on 256 pixels each.
// The flow of data is as follows:
// ALU info : G(8)A(32)G(4)A(a/32)G(g/32) - Here G corresponds to the input channels
// 1. Read the input data as blocks of 32x8 pixels, starting from the 4 channel to reverse.
//    The data is fetched in reverse order so that accumulating to 32-bit partials will be easier.
// 2. Feed the input data to ALU transposing the data to 8x32 pixels
// 3. Use the PRAM to accumulate the ALU output by shifting and placing inside 32 bit datatype
// 4. Repeat the steps for another 4 iteration starting the next 4 channels,
//    this makes it transposing the data of 32x(8*4) pixels-> (8*4)x32 pixels
template <>
LogicalResult TransposeReshapePattern::transform(
    torq_hl::TransposeReshapeOp op, PatternRewriter &rewriter
) const {
    // op constants
    const uint32_t max_out_channels = 32;
    const uint32_t addr_group = 2;
    const uint32_t out_ch_group = 4;
    const uint32_t reverse_iter_group = 4;
    int32_t stride = op.getStride()[1]; // 64

    // input
    auto input_type = llvm::dyn_cast<MemRefType>(op.getInput().getType());

    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto outShape = output_type.getShape();

    auto inputStrides = getEncodedStridesElements(input_type);

    // transpose has no channel concept, it is the inputShape[1] to be transposed from current
    // implementation if transpose the the first dim, input_channels here is the rest dims
    // multiplied together
    uint32_t input_channels = 0;
    int32_t in_ch_offset = 0;

    uint32_t frame_size = 0;

    int32_t leftDimSize = 0;
    int32_t transposeDimSize = 0;
    int32_t rightDimSize = 0;

    auto inputShape = input_type.getShape();

    groupContinuousTransposeReshapeDim(
        op, inputShape, outShape, leftDimSize, transposeDimSize, rightDimSize
    );
    LLVM_DEBUG(llvm::dbgs() << "Identified (leftDimSize, transposeDimSize, rightDimSize) -> ("
                            << leftDimSize << "," << transposeDimSize << ",`" << rightDimSize
                            << ")\n";);

    frame_size = transposeDimSize;
    input_channels = rightDimSize;
    in_ch_offset = stride;

    uint32_t max_px_block = 32;

    input_channels = align_ceil(input_channels, 32);
    const uint32_t out_ch_offset = input_channels;
    const uint32_t output_channels = input_channels;

    const uint32_t out_ch_split = div_ceil(output_channels, max_out_channels);
    const uint32_t total_px_block = div_ceil(frame_size, max_px_block);

    auto sliceCfgAttr = SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BYP, ALUOp1Mode::BYP, ALUOp1Mode::BYP, ALUOp1Mode::BYP},
        0b1111,                              // alu_d_unsigned
        0,                                   // alu_w_unsigned
        ACTMode::ACT,                        // act_mode
        {0, 0, 0, 0},                        // act left shift
        0,                                   // shift_factor
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

    // Not used as we are in bypass mode: acbw, acbr, dewr, debr
    Ndls ndls;
    ndls.add(
        NdlType::REF,
        {{DimType::H, MemDimTag::X, frame_size, 0}, {DimType::H, MemDimTag::Y, input_channels, 0}}
    );

    ndls.add(
        NdlType::DEDR, {{DimType::L, MemDimTag::X, max_px_block, 1},
                        {DimType::L, MemDimTag::Y, addr_group, in_ch_offset},
                        {DimType::H, MemDimTag::Y, addr_group, 2 * in_ch_offset},
                        {DimType::H, MemDimTag::Y, addr_group, 16 * in_ch_offset},
                        {DimType::H, MemDimTag::Y, out_ch_group, 4 * in_ch_offset},
                        {DimType::H, MemDimTag::X, total_px_block, max_px_block},
                        {DimType::H, MemDimTag::Y, out_ch_split, in_ch_offset * max_out_channels}}
    );

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, 4, 1},
        {DimType::L, MemDimTag::D, max_out_channels / 4, 4},
        {DimType::L, MemDimTag::G, addr_group, out_ch_offset},
        {DimType::H, MemDimTag::X, max_px_block / addr_group, addr_group * out_ch_offset},
        {DimType::H, MemDimTag::X, total_px_block, max_px_block * out_ch_offset},
        {DimType::H, MemDimTag::Y, out_ch_split, max_out_channels}
    };
    ndls.add(NdlType::DEQW, deqw);

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width, 1},
        {DimType::H, RegDimTag::S, out_ch_group, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * reverse_iter_group,
         HwInfo::iram_width}
    };
    ndls.add(NdlType::CEDW, cedw);

    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, addr_group * out_ch_group, 36},
        {DimType::L, RegDimTag::G, max_px_block, 1},
        {DimType::H, RegDimTag::S, 1, 1},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * reverse_iter_group,
         HwInfo::iram_width}
    };
    ndls.add(NdlType::CEDR, cedr);

    // we are in bypass mode
    // for ceww, torq_api there is no check for weight_bypass, keep it as cewr as hw request
    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };
    ndls.add(NdlType::CEWW, ceww);

    // we are in bypass mode
    // torq_api use T tag size to check if weight bypass or not, weight_bypass = !T.size
    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };
    ndls.add(NdlType::CEWR, cewr);

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::mac_count},
        {DimType::H, RegDimTag::M, reverse_iter_group, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::mac_count * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::CEPR, cepr);

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, size_t(HwInfo::mac_count / HwInfo::act_width),
         HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, 1, HwInfo::mac_count * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split,
         HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                        // Operation to replace
        "transpose_reshape",       // Task name
        ValueRange{op.getInput()}, // Input tensor
        ValueRange{},              // Weights
        ValueRange{},              // BiasScale tensor,
        ValueRange{op.getInit()},  // Output tensor initializer
        ValueRange{},              // Symbols
        sliceCfgAttr,
        ndls // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
