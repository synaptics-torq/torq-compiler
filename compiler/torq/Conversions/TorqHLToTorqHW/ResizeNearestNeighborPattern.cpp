
// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl-resizenn"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult ResizeNearestNeighborPattern::transform(
    torq_hl::ResizeNearestNeighborOp op, PatternRewriter &rewriter
) const {
    // TODO Current support is only for scaleUp of 2.
    // The input is taken in bypass mode and horizontal pixel is repeated before sending to ALU
    // The input row is taken twice to scale the height by 2
    // Current only 16 values are taken at a time
    auto inputType = llvm::dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> input_shape(inputShape.begin(), inputShape.end());
    auto input_strides = getEncodedStridesElements(inputType);

    auto outputShape = outputType.getShape();
    llvm::SmallVector<int64_t> output_shape(outputShape.begin(), outputShape.end());
    auto output_strides = getEncodedStridesElements(outputType);

    auto max_input = 16; // Only tested for vector batches of 16. Higher sizes would need to be
                         // compute with input dims
    auto scaleUp = 2;    // Only support scale of 2 for now

    uint32_t total_px = 1;
    for (int i = 0; i < input_shape.size() - 1; ++i) {
        total_px *= input_shape[i];
    }
    total_px *= align_ceil(static_cast<int32_t>(input_shape.back()), max_input);
    auto total_px_block = div_ceil(total_px, max_input);

    auto sliceCfgAttr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0,                                   // alu_d_unsigned
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

    MemNdlDimsData ref = {{DimType::H, MemDimTag::X, total_px, 0}};

    MemNdlDimsData dedr;
    dedr = {
        {DimType::L, MemDimTag::A, max_input, 1},
        {DimType::H, MemDimTag::A, div_ceil(input_shape.back(), max_input), max_input},
        {DimType::H, MemDimTag::O, scaleUp, 0}, // Repeat the rows 2 times
        {DimType::H, MemDimTag::A, input_shape[2], input_strides[2]},
        {DimType::H, MemDimTag::V, input_shape[1], input_strides[1]},
        {DimType::H, MemDimTag::V, input_shape[0], input_strides[0]}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, 1, 1},
        {DimType::L, MemDimTag::D, HwInfo::act_width, 1},
        {DimType::H, MemDimTag::A, div_ceil(input_shape.back(), HwInfo::act_width) * scaleUp,
         HwInfo::act_width},
        {DimType::H, MemDimTag::A, output_shape[2], output_strides[2]},
        {DimType::H, MemDimTag::V, output_shape[1], output_strides[1]},
        {DimType::H, MemDimTag::V, output_shape[0], output_strides[0]},
    };

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / 4, 1},
        {DimType::H, RegDimTag::S, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::I, scaleUp}, // Repeat the cols pixel twice
        {DimType::L, RegDimTag::B, 1, 1},    {DimType::L, RegDimTag::D, max_input, 1},
        {DimType::H, RegDimTag::S},          {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},          {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T}
    };

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };

    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::mac_count},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * scaleUp,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, 1, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S, scaleUp, HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * scaleUp,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    // Not used: debr, dewr, acbw, acbr, acpw, aldw

    Ndls ndls;
    ndls.add(NdlType::REF, ref);
    ndls.add(NdlType::DEDR, dedr);
    ndls.add(NdlType::DEQW, deqw);
    ndls.add(NdlType::CEDW, cedw);
    ndls.add(NdlType::CEDR, cedr);
    ndls.add(NdlType::CEWW, ceww);
    ndls.add(NdlType::CEWR, cewr);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,
        "ResizeNearestNeighbor",   // Operation to replace
        ValueRange{op.getInput()}, // Input tensor
        ValueRange{},              // Weights
        ValueRange{},              // BiasScale tensor,
        ValueRange{op.getInit()},  // Output tensor initializer
        ValueRange{},              // Symbols
        sliceCfgAttr,              // Slice configuration
        ndls                       // NDLs
    );

    return success();
}
} // namespace mlir::syna::torq
