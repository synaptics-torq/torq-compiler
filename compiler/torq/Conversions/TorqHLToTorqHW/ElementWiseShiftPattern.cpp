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

static llvm::cl::opt<bool> clRelaxInpuStrideCheck(
    "torq-ews-relax-input-stride-check",
    llvm::cl::desc("Relax ElementWiseShift input stride check"), llvm::cl::init(false)
);

template <>
LogicalResult ElementWiseShiftPattern::transform(
    torq_hl::ElementWiseShiftOp op, PatternRewriter &rewriter
) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    torq_hl::ShiftModeEnum opType = op.getOpType();

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput1().getType());
    auto input_shape = input_type.getShape();
    Type elementType = input_type.getElementType();
    const uint32_t data_bytes = elementType.getIntOrFloatBitWidth() / 8;

    uint32_t total_elements = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_elements *= input_shape[i];
    }

    // only halve of the full throughput
    bool act_halve = data_bytes == 4;
    const int32_t max_input = std::min(
        total_elements * data_bytes,
        static_cast<uint32_t>((HwInfo::act_width >> act_halve) * data_bytes)
    );

    const int32_t total_px_block = div_ceil(total_elements * data_bytes, max_input);
    assert(total_px_block > 0);

    // Inputs
    auto type1 = llvm::dyn_cast<MemRefType>(op.getInput1().getType());
    auto type2 = llvm::dyn_cast<MemRefType>(op.getInput2().getType());
    assert(type1.getElementType() == type2.getElementType() && "Input types must match");
    assert(type1.getShape() == type2.getShape() && "Input shapes must match");

    auto inputStrides = getEncodedStridesElements(type1);
    if (!clRelaxInpuStrideCheck) {
        // TODO: relax check input strides for logical left shift
        if (inputStrides != getEncodedStridesElements(type2)) {
            return op.emitError() << "Input strides must match: Input1: " << type1
                                  << ", Input2: " << type2;
        }
    }

    // The difference between the two input addresses is used as offset in DEDR to fetch the data
    auto input1Address = GetAddressOp::create(rewriter, loc, op.getInput1()).getAddress();
    auto input2Address = GetAddressOp::create(rewriter, loc, op.getInput2()).getAddress();
    auto inputDiff = getAffineDimExpr(1, ctx) - getAffineDimExpr(0, ctx);

    torq_hw::ACTMode actMode = torq_hw::ACTMode::ACT;
    torq_hw::RoundingMode rounding_mode = RoundingMode::OFF;
    const uint8_t act_sum_bits = data_bytes * 8;
    const uint8_t alu_group_width = 32;
    if (opType == torq_hl::ShiftModeEnum::ASR) {
        actMode = torq_hw::ACTMode::ASR;
    }
    else if (opType == torq_hl::ShiftModeEnum::LSR) {
        actMode = torq_hw::ACTMode::LSR;
    }
    else if (opType == torq_hl::ShiftModeEnum::LSL) {
        actMode = torq_hw::ACTMode::LSL;
    }
    else {
        llvm::errs() << "Unsupported elementwise shift op: " << opType << "\n";
        return failure();
    }

    if ((op.getRound())) {
        rounding_mode = RoundingMode::NTP;
    }

    auto slice_cfg_attr = SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0x0,                                 // alu_d_unsigned
        0,                                   // alu_w_unsigned
        actMode,                             // act_mode
        {0, 0, 0, 0},                        // act left shift
        0,                                   // shift_factor
        std::numeric_limits<int32_t>::min(), // output_min
        std::numeric_limits<int32_t>::max(), // output_max
        0,                                   // output_zp
        0,                                   // no_p_clear
        0,                                   // no_p_output
        0, 0, 0, 0,                          // kernel lrtb
        0, 0, 0, 0,                          // pad lrtb
        0,                                   // pad_value
        1,                                   // stride
        0,                                   // stride_offset
        rounding_mode,                       // rounding_mode
        WeightFormat::SI,                    // weight_format
        0, 0,                                // alu_disable, act_disable
        NumberFormat::I,                     // alu_format
        NumberFormat::I,                     // act_format
        act_sum_bits,                        // act_sum_bits
        {}                                   // table
    );

    MemNdlDimsData ref = {{DimType::H, MemDimTag::A, total_px_block}};

    // depthwise mode (G2) to load 2 addresses per cycle
    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::D, max_input, 1},
        {DimType::L, MemDimTag::G, 2, inputDiff},
        {DimType::H, MemDimTag::A, total_px_block, max_input}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, data_bytes, 1},
        {DimType::L, MemDimTag::D, max_input / data_bytes, data_bytes},
        {DimType::H, MemDimTag::A, total_px_block, max_input}
    };

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, 4, 1}, // 4 bytes
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / 4, 4},
        {DimType::H, RegDimTag::T, total_px_block, HwInfo::iram_width}
    };

    // depthwise mode (G8) to match with DEDR (G2): {D32K1G8}
    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::B, 4, 1}, // 4 bytes
        {DimType::L, RegDimTag::D, alu_group_width / 4, 4},
        {DimType::L, RegDimTag::G, 8, HwInfo::iram_seg_width / 2},
        {DimType::H, RegDimTag::T, total_px_block, HwInfo::iram_width}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::T, total_px_block, HwInfo::pram_dsize * HwInfo::mac_count}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::T, total_px_block,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    Ndls ndls;
    ndls.add(NdlType::REF, ref);
    ndls.add(NdlType::DEDR, dedr);
    ndls.add(NdlType::DEQW, deqw);
    ndls.add(NdlType::CEDW, cedw);
    ndls.add(NdlType::CEDR, cedr);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                                         // Operation to replace
        stringifyShiftModeEnum(opType),             // Task name
        ValueRange{op.getInput1(), op.getInput2()}, // Input tensor
        ValueRange{},                               // Weights
        ValueRange{},                               // BiasScale tensor
        ValueRange{op.getInit()},                   // Output tensor initializer
        ValueRange{input1Address, input2Address},   // Symbols used to compute the NDLs
        slice_cfg_attr,                             // Slice configuration
        ndls
    );

    return success();
}

} // namespace mlir::syna::torq
