// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-transpose"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

LogicalResult convertToHw(torq_hl::TransposeOp op, PatternRewriter &rewriter);

void validateContinuousDim(
    torq_hl::TransposeOp op, int32_t &leftDim, int32_t &transposeDim, int32_t &rightDim,
    int32_t &leftDimSize, int32_t &transposeDimSize, int32_t &rightDimSize
) {
    auto inputType = llvm::dyn_cast<MemRefType>(op.getInput().getType());
    auto inputShape = inputType.getShape();
    auto inputStrides = getEncodedStridesElements(inputType);

    int32_t rank = inputShape.size();
    // A dimension is continuous if the stride of that dimension is the multiple of previous stride
    // and the size of the previous dimension
    for (int i = transposeDim; i < rank - 1; ++i) {
        if (inputStrides[i] != inputShape[i + 1] * inputStrides[i + 1]) {
            op.emitError("Op transpose strides are not continuous");
        }
    }
    for (int i = rightDim; i < transposeDim - 1; ++i) {
        if (inputStrides[i] != inputShape[i + 1] * inputStrides[i + 1]) {
            op.emitError("Op transpose right dimension strides are not continuous");
        }
    }
}

// To program the hardware for transpose we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// For transpose the ALU will be working on BYP.
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
LogicalResult
TransposePattern::transform(torq_hl::TransposeOp op, PatternRewriter &rewriter) const {
    if (convertToHw(op, rewriter).succeeded()) {
        return success();
    }

    if (clUseNewKernels) {
        return op.emitError("New kernel failed");
    }

    // op constants
    const uint32_t max_out_channels = 32;
    const uint32_t addr_group = 2;
    const uint32_t out_ch_group = 4;
    const uint32_t reverse_iter_group = 4;

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
    int32_t transposeDim = -1;
    uint32_t frame_size = 0;
    int32_t leftDim = 0;
    int32_t rightDim = 1;

    // Identify the permutation and define it in terms of leftDim, transposeDim, rightDim
    identifyTransposeDim(op, outShape, transposeDim, leftDim, rightDim);
    LLVM_DEBUG({
        op.dump();
        llvm::dbgs() << "Identified (leftDim, transposeDim, rightDim) -> (" << leftDim << ","
                     << transposeDim << "," << rightDim << ")\n";
    });

    // Group dimensions that are consecutive
    int32_t leftDimSize = 0;
    int32_t transposeDimSize = 0;
    int32_t rightDimSize = 0;

    auto inputShape = input_type.getShape();

    groupContinuousDim(
        inputShape, leftDim, transposeDim, rightDim, leftDimSize, transposeDimSize, rightDimSize
    );
    LLVM_DEBUG(llvm::dbgs() << "Identified (leftDimSize, transposeDimSize, rightDimSize) -> ("
                            << leftDimSize << "," << transposeDimSize << "," << rightDimSize
                            << ")\n";);

    // Validate whether the identified dims are continuous w.r.t the stride
    validateContinuousDim(
        op, leftDim, transposeDim, rightDim, leftDimSize, transposeDimSize, rightDimSize
    );
    frame_size = transposeDimSize;
    input_channels = rightDimSize;
    in_ch_offset = inputStrides[transposeDim - 1];
    auto leftStride = leftDim < 0 ? 0 : inputStrides[leftDim];

    const uint32_t out_ch_offset = input_channels;

    const uint32_t max_px_block = 32;

    input_channels = align_ceil(input_channels, 32);
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
                        {DimType::H, MemDimTag::Y, out_ch_split, in_ch_offset * max_out_channels},
                        {DimType::H, MemDimTag::X, leftDimSize, leftStride}}
    );

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, 4, 1},
        {DimType::L, MemDimTag::D, max_out_channels / 4, 4},
        {DimType::L, MemDimTag::G, addr_group, out_ch_offset},
        {DimType::H, MemDimTag::X, max_px_block / addr_group, addr_group * out_ch_offset},
        {DimType::H, MemDimTag::X, total_px_block, max_px_block * out_ch_offset},
        {DimType::H, MemDimTag::Y, out_ch_split, max_out_channels},
        {DimType::H, MemDimTag::X, leftDimSize,
         max_out_channels * out_ch_split * max_px_block * total_px_block}
    };
    ndls.add(NdlType::DEQW, deqw);

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width, 1},
        {DimType::H, RegDimTag::S, out_ch_group, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * reverse_iter_group * leftDimSize,
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
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * reverse_iter_group * leftDimSize,
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
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * leftDimSize,
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
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * leftDimSize,
         HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                        // Operation to replace
        "transpose",               // Task name
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
