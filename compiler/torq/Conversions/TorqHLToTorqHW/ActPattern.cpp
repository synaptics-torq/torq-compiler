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
LogicalResult ActPattern::transform(torq_hl::ActOp op, PatternRewriter &rewriter) const {

    const uint32_t act_width = 16;

    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    auto input_element_type = input_type.getElementType();
    auto input_element_size = input_element_type.getIntOrFloatBitWidth() / 8;

    // when dypte is boolean, the size is 1
    input_element_size = input_element_type.isInteger(1) ? 1 : input_element_size;

    auto output_type = llvm::cast<MemRefType>(op.getInit().getType());
    auto output_element_type = output_type.getElementType();
    auto output_element_size = output_element_type.getIntOrFloatBitWidth() / 8;
    auto output_rank = output_type.getRank();

    torq_hw::NumberFormat alu_format = torq_hw::NumberFormat::I;
    torq_hw::NumberFormat act_format = torq_hw::NumberFormat::I;
    uint32_t act_sum_bits = 8 * input_element_size;

    // TODO: refactor
    uint32_t is_float = 0;
    int32_t act_clip_min = 0;
    int32_t act_clip_max = 0;
    if (output_element_type.isInteger(32)) {
        act_clip_min = std::numeric_limits<int32_t>::min();
        act_clip_max = std::numeric_limits<int32_t>::max();
    }
    else if (output_element_type.isInteger(16)) {
        act_clip_min = std::numeric_limits<int16_t>::min();
        act_clip_max = std::numeric_limits<int16_t>::max();
    }
    else if (output_element_type.isInteger(8)) {
        act_clip_min = std::numeric_limits<int8_t>::min();
        act_clip_max = std::numeric_limits<int8_t>::max();
    }
    else if (output_element_type.isInteger(1)) {
        act_clip_min = 0;
        act_clip_max = 1;
        output_element_size = 1;
    }
    else if (output_element_type.isBF16()) {
        alu_format = torq_hw::NumberFormat::BF;
        act_format = torq_hw::NumberFormat::BF;
        is_float = 1;

        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }
    else if (output_element_type.isF32()) {
        alu_format = torq_hw::NumberFormat::BF;
        act_format = torq_hw::NumberFormat::BF;
        is_float = 1;

        // hw use bf16 min/max for f32 min/max
        // act_clip_min = 0x80000000;
        // act_clip_max = 0x7fffffff;

        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }
    else {
        op.emitError() << "Unsupported output element type: " << output_element_type << "\n";
        return failure();
    }

    torq_hw::ACTMode act_mode = torq_hw::ACTMode::ACT;
    auto op_name = op.getName().str();
    if (op_name == "abs") {
        act_mode = torq_hw::ACTMode::ABS;
    }
    else if (op_name == "negate") {
        act_mode = torq_hw::ACTMode::NEG;
    }
    else if (op_name == "clz") { // clz int8/int16 has issue
        act_mode = torq_hw::ACTMode::CLZ;
    }
    else if (op_name == "ceil") {
        act_mode = torq_hw::ACTMode::CEL;

        if (!input_element_type.isBF16() && !input_element_type.isF32()) {
            op.emitError() << "Unsupported input element type for ceil: " << input_element_type
                           << "\n";
            return failure();
        }
        if (input_element_type != output_element_type) {
            op.emitError() << "Input and output element types must be the same for ceil: "
                           << input_element_type << " vs " << output_element_type << "\n";
            return failure();
        }
    }
    else if (op_name == "floor") {
        act_mode = torq_hw::ACTMode::FLR;

        if (!input_element_type.isBF16() && !input_element_type.isF32()) {
            op.emitError() << "Unsupported input element type for floor: " << input_element_type
                           << "\n";
            return failure();
        }
        if (input_element_type != output_element_type) {
            op.emitError() << "Input and output element types must be the same for floor: "
                           << input_element_type << " vs " << output_element_type << "\n";
            return failure();
        }
    }
    else if (op_name == "i2f") {
        act_mode = torq_hw::ACTMode::I2F;

        alu_format = torq_hw::NumberFormat::I;
        act_format = torq_hw::NumberFormat::I;
        is_float = 1;
    }
    else if (op_name == "f2i") {
        act_mode = torq_hw::ACTMode::F2I;

        alu_format = torq_hw::NumberFormat::BF;
        act_format = torq_hw::NumberFormat::BF;
        is_float = 1;

        act_clip_min = 0xff800000;
        act_clip_max = 0x7f800000;
    }
    else if (op_name == "i2i" || op_name == "f2f") {
        act_mode = torq_hw::ACTMode::ACT;
    }
    else if (op_name == "clamp") {

        act_mode = torq_hw::ACTMode::ACT;

        // TODO: min/max add check to be in the scope of data type
        if (input_element_type.isInteger()) {
            act_clip_min = op.getMinInt();
            act_clip_max = op.getMaxInt();
        }
        else if (input_element_type.isF32() || input_element_type.isBF16()) {

            union {
                float f;
                uint32_t i;
            } float_to_int;

            float_to_int.f = op.getMinFp().convertToFloat();
            act_clip_min = float_to_int.i;
            float_to_int.f = op.getMaxFp().convertToFloat();
            act_clip_max = float_to_int.i;
        }
        else {
            op.emitError() << "Clamp Op unsupported input element type for clamp: "
                           << input_element_type << "\n";
            return failure();
        }
    }

    uint32_t total_px_size = input_element_size;
    for (int i = 0; i < input_shape.size(); ++i) {
        total_px_size *= input_shape[i];
    }

    const uint32_t actual_alu_group_width = (act_width >> is_float) * input_element_size;

    const uint32_t alu_throughput =
        total_px_size < actual_alu_group_width ? total_px_size : actual_alu_group_width;

    const uint32_t alu_px_block = div_ceil(total_px_size, actual_alu_group_width);

    const uint32_t pbus_throughput =
        output_rank > 0 ? actual_alu_group_width / input_element_size * output_element_size
                        : output_element_size;

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        rewriter.getContext(),
        {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0,                // alu_d_unsigned
        0,                // alu_w_unsigned
        act_mode,         // act_mode
        {0, 0, 0, 0},     // act left shift
        0,                // shift_factor
        act_clip_min,     // output_min
        act_clip_max,     // output_max
        op.getOutputZp(), // output_zp
        0x0,              // alu_disable
        0x0,              // act_disable
        alu_format, act_format, act_sum_bits
    );

    MemNdlDimsData ref = {{DimType::H, MemDimTag::A, total_px_size}};

    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::D, alu_throughput, 1},
        {DimType::H, MemDimTag::A, alu_px_block, alu_throughput}
    };

    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::B, output_element_size, 1},
        {DimType::L, MemDimTag::D, pbus_throughput / output_element_size, output_element_size},
        {DimType::H, MemDimTag::A, alu_px_block,
         alu_throughput * output_element_size / input_element_size}
    };

    RegNdlDimsData cedw = {
        {DimType::L, RegDimTag::B, input_element_size, 1},
        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / input_element_size, input_element_size},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, alu_px_block, HwInfo::iram_width}
    };

    RegNdlDimsData cedr = {
        {DimType::L, RegDimTag::B, input_element_size},
        {DimType::L, RegDimTag::D, alu_group_width / input_element_size, input_element_size},
        {DimType::L, RegDimTag::G, 1, HwInfo::iram_seg_width},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, alu_px_block, HwInfo::iram_width}
    };

    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
        {DimType::L, RegDimTag::D, alu_group_width * alu_groups, HwInfo::pdat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, alu_px_block, 1}
    };

    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, input_element_size},
        {DimType::L, RegDimTag::D, act_width, HwInfo::pdat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, alu_px_block, alu_group_width * alu_groups * HwInfo::pdat_width}
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
        op,                        // Operation to replace
        op.getName(),              // Task name
        ValueRange{op.getInput()}, // Input tensor
        ValueRange{},              // Weights
        ValueRange{},              // BiasScale tensor,
        ValueRange{op.getInit()},  // Output tensor initializer
        ValueRange{},              // Symbols
        slice_cfg_attr,            // Slice configuration
        ndls                       // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq