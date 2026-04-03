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
LogicalResult ElementWiseBinaryPattern::transform(
    torq_hl::ElementWiseBinaryOp op, PatternRewriter &rewriter
) const {
    auto ctx = rewriter.getContext();
    auto loc = op.getLoc();
    torq_hl::ElementwiseOpEnum opType = op.getOpType();

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput1().getType());
    auto input_shape = input_type.getShape();
    Type elementType = input_type.getElementType();
    const uint32_t data_bytes =
        elementType.isInteger(1) ? 1 : elementType.getIntOrFloatBitWidth() / 8;

    // output
    auto output_type = llvm::cast<MemRefType>(op.getInit().getType());
    auto output_element_type = output_type.getElementType();
    uint32_t output_data_bytes =
        output_element_type.isInteger(1) ? 1 : output_element_type.getIntOrFloatBitWidth() / 8;

    uint32_t total_elements = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_elements *= input_shape[i];
    }

    const int32_t total_px_block = div_ceil(total_elements, HwInfo::max_input);
    assert(total_px_block > 0);

    // Inputs
    auto type1 = llvm::dyn_cast<MemRefType>(op.getInput1().getType());
    auto type2 = llvm::dyn_cast<MemRefType>(op.getInput2().getType());
    assert(type1.getElementType() == type2.getElementType() && "Input types must match");
    auto inputStrides = getEncodedStridesElements(type1);
    if (inputStrides != getEncodedStridesElements(type2)) {
        return op.emitError("Input strides must match");
    }

    // Output
    auto outputType = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    // auto outputShape = outputType.getShape();
    auto outputStrides = getEncodedStridesElements(outputType);
    if (inputStrides != outputStrides) {
        return op.emitError("Input and output strides must match");
    }

    // The difference between the two input addresses is used as offset in DEDR to fetch the data
    auto input1Address = GetAddressOp::create(rewriter, loc, op.getInput1()).getAddress();
    auto input2Address = GetAddressOp::create(rewriter, loc, op.getInput2()).getAddress();
    auto inputDiff = getAffineDimExpr(1, ctx) - getAffineDimExpr(0, ctx);

    int32_t alu_d_unsigned;
    if (data_bytes == 1) {
        alu_d_unsigned = 0x0;
    }
    else if (data_bytes == 2) {
        if (op.getIsUnsigned()) {
            alu_d_unsigned = 0xF;
        }
        else {
            alu_d_unsigned = 0x5;
        }
    }
    else {
        alu_d_unsigned = 0x7;
    }

    torq_hw::ALUOp1Mode hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
    NumberFormat alu_format = NumberFormat::I;
    NumberFormat act_format = NumberFormat::I;
    uint8_t acr_rsh = 0;
    SmallVector<uint32_t> act_lsh = {0, 0, 0, 0};
    uint32_t act_sum_bits = 32;
    uint32_t act_ratio = 1;

    if (opType == torq_hl::ElementwiseOpEnum::BITWISE_AND) {
        hwOp1Mode = torq_hw::ALUOp1Mode::BAND;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::BITWISE_OR) {
        hwOp1Mode = torq_hw::ALUOp1Mode::BOR;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::BITWISE_XOR) {
        hwOp1Mode = torq_hw::ALUOp1Mode::BXOR;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::MINIMUM) {
        hwOp1Mode = torq_hw::ALUOp1Mode::MIN;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::MAXIMUM) {
        hwOp1Mode = torq_hw::ALUOp1Mode::MAX;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::GREATER) {
        hwOp1Mode = torq_hw::ALUOp1Mode::GT;
        acr_rsh = 16;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::GREATER_EQUAL) {
        hwOp1Mode = torq_hw::ALUOp1Mode::GE;
        acr_rsh = 16;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::EQUAL) {
        hwOp1Mode = torq_hw::ALUOp1Mode::EQ;
        acr_rsh = 16;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::LOGICAL_AND) {
        hwOp1Mode = torq_hw::ALUOp1Mode::AND;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::LOGICAL_OR) {
        hwOp1Mode = torq_hw::ALUOp1Mode::OR;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::LOGICAL_XOR) {
        hwOp1Mode = torq_hw::ALUOp1Mode::XOR;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::SUB) {
        assert(
            elementType.isSignlessInteger(32) && "Only signed 32-bit integer type supported Sub"
        );
        hwOp1Mode = torq_hw::ALUOp1Mode::SACC;
        act_lsh = {0, 8, 16, 24};
        act_sum_bits = 48;
        act_ratio = 4;
    }
    else if (opType == torq_hl::ElementwiseOpEnum::ADD) {
        assert(
            elementType.isSignlessInteger(32) && "Only signed 32-bit integer type supported Add"
        );
        hwOp1Mode = torq_hw::ALUOp1Mode::ACC;
        act_lsh = {0, 8, 16, 24};
        act_sum_bits = 48;
        act_ratio = 4;
    }
    else {
        llvm::errs() << "Unsupported elementwise binary op: " << opType << "\n";
        return failure();
    }

    if (elementType.isBF16()) {
        alu_format = NumberFormat::BF;
    }

    auto slice_cfg_attr = SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {hwOp1Mode, hwOp1Mode, hwOp1Mode, hwOp1Mode},
        alu_d_unsigned,                      // alu_d_unsigned
        0,                                   // alu_w_unsigned
        ACTMode::ACT,                        // act_mode
        act_lsh,                             // act left shift
        acr_rsh,                             // shift_factor
        std::numeric_limits<int32_t>::min(), // output_min
        std::numeric_limits<int32_t>::max(), // output_max
        0,                                   // output_zp
        0x0,                                 // alu_disable
        0x0,                                 // act_disable
        alu_format,                          // alu_format
        act_format,                          // act_format
        act_sum_bits                         // act_sum_bits
    );

    // REF is just a descriptor of the input data
    MemNdlDimsData ref = {{DimType::H, MemDimTag::A, total_px_block}};

    // DEDR fetches data from lram one block at a time
    // This is repeated for 2 different inputs at 2 different addresses
    // We use the address offset that was precomputed to fetch the data from the second address
    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::D, HwInfo::max_input, 1},
        {DimType::H, MemDimTag::G, 2, inputDiff},
        {DimType::H, MemDimTag::A, total_px_block * data_bytes, HwInfo::max_input}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, output_data_bytes, 1},
        {DimType::L, MemDimTag::A, HwInfo::act_width / act_ratio, output_data_bytes},
        {DimType::H, MemDimTag::A, HwInfo::max_input / HwInfo::act_width,
         (HwInfo::act_width * output_data_bytes) / act_ratio},
        {DimType::H, MemDimTag::A, total_px_block * act_ratio,
         (HwInfo::max_input * output_data_bytes) / act_ratio}
    };

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

    // Handle special case for auto-generated int8 weights (Ws):
    // - For ACC mode:  pattern is [1, 1, 1, 1, ...]
    // - For SACC mode: pattern is [1, -1, 1, -1, ...]
    // These weights are prefilled as 16-bit values (low byte + high byte).
    // Load the low byte first, then the high byte in two steps to reconstruct correctly.
    RegNdlDimsData cewr;
    if (hwOp1Mode == torq_hw::ALUOp1Mode::SACC || hwOp1Mode == torq_hw::ALUOp1Mode::ACC) {
        cewr = {
            {DimType::L, RegDimTag::B, 1, 1},
            {DimType::H, RegDimTag::S, 2, 1},
            {DimType::H, RegDimTag::N, (total_px_block * data_bytes), 0},
            {DimType::H, RegDimTag::T, 1, HwInfo::wram_width}
        };
    }

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, 1, 1}, {DimType::H, RegDimTag::T, 0, HwInfo::wram_width}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::max_input, HwInfo::pram_dsize},
        {DimType::H, RegDimTag::S, 1, HwInfo::pram_dsize * HwInfo::max_input},
        {DimType::H, RegDimTag::M, 1, 0},
        {DimType::H, RegDimTag::W, 1, HwInfo::pram_dsize * HwInfo::max_input},
        {DimType::H, RegDimTag::N, 2, 0},
        {DimType::H, RegDimTag::T, total_px_block * data_bytes,
         HwInfo::pram_dsize * HwInfo::max_input * HwInfo::pram_depth}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, data_bytes, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S,
         (HwInfo::max_input / HwInfo::act_width) / (data_bytes / act_ratio),
         HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * data_bytes,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };

    RegNdlDimsData acpw;
    if (alu_format == NumberFormat::I &&
        (hwOp1Mode == torq_hw::ALUOp1Mode::ACC || hwOp1Mode == torq_hw::ALUOp1Mode::SACC)) {
        acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
            {DimType::L, RegDimTag::D, data_bytes, HwInfo::pdat_width},
            {DimType::L, RegDimTag::G, HwInfo::act_width / data_bytes,
             HwInfo::pdat_width * data_bytes},
            {DimType::H, RegDimTag::T}
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
    ndls.add(NdlType::CEWR, cewr);
    ndls.add(NdlType::CEWW, ceww);
    ndls.add(NdlType::CEPR, cepr);
    ndls.add(NdlType::ACBR, acbr);
    ndls.add(NdlType::ACPR, acpr);
    ndls.add(NdlType::ACPW, acpw);

    rewriter.replaceOpWithNewOp<SliceTaskOp>(
        op,                                         // Operation to replace
        stringifyElementwiseOpEnum(opType),         // Task name
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
