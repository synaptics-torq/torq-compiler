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

// To program the hardware for FC we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// For fully connected the ALU will be working on MUL and ACC(add) operations.
// The ALU will be working on 1 group of 64 pixels each.
// The data and weight will be read in blocks of iram_width pixels and max_mode_pixels weights.
// In the case of FC the data and weights are interchanged compared to conv2d.
// The data is fed through the dewr route and the weights are fed through the dedr route.
// The flow of data is as follows:
// max_mode_pixels : 64
// ALU info 64x1: V(64)U(u)V(v/64)
// 1. Read the input data in blocks of wram_width pixels.
// 2. Read the weights in blocks of max_mode_pixels weights.
// 3. Feed the data and weights to the ALU.
// 4. To complete the FC accumulation the full input channels has to be processed.
//    The weights and data are changed to cycle through to complete 64 output channel worth of data
//    and weights.
// 5. The ALU will output 64 channels outputs.
// 7. Read the bias & scale in blocks of output_channels.
// 8. The output is rescaled to int8 in blocks of 16 pixels
// 9. The output is written back to LRAM
// 10. Take the next block of 64 output weights and repeat the process.
//
//
template <>
LogicalResult FCPattern::transform(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) const {
    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();

    uint32_t input_channel = input_shape[1];

    // FullyConnected inputs are often 2D: [N, IC], but when fused from Conv2D,
    // the input can still be 4D: [N, H, W, IC], where H=W=1
    // Example: [1, 1, 1, 1280] is a common degenerate Conv2D input
    if (input_shape.size() == 4) {
        // Assert that dims 0, 1, 2 are 1: [N=1, H=1, W=1, IC]
        assert(input_shape[0] == 1 && "FC input dim[0] (batch) must be 1");
        assert(input_shape[1] == 1 && "FC input dim[1] (height) must be 1");
        assert(input_shape[2] == 1 && "FC input dim[2] (width) must be 1");
        input_channel = input_shape[3];
    }
    assert(input_channel > 0 && "input channel must be greater than 0");

    const uint32_t output_channel = output_shape[1];
    const uint32_t max_out_channels = std::min<uint32_t>(output_channel, HwInfo::max_input);
    assert(max_out_channels > 0 && "max output channel must be greater than 0");

    const uint32_t out_ch_split = div_ceil(output_channel, max_out_channels);
    assert(out_ch_split > 0 && "output channel split must be greater than 0");

    std::vector<uint32_t> weight_shape{input_channel};
    std::vector<uint32_t> weight_dims = prepareWeightDims(weight_shape, input_channel, alu_groups);
    assert(!weight_dims.empty());

    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL},
        {ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC},
        0,                   // alu_d_unsigned
        0,                   // alu_w_unsigned
        ACTMode::ACT,        // act_mode
        {0, 0, 0, 0},        // act left shift
        op.getShiftFactor(), // shift_factor
        op.getOutputMin(),   // output_min
        op.getOutputMax(),   // output_max
        op.getOutputZp(),    // output_zp
        false,               // no_p_clear
        false,               // no_p_output
        0, 0, 0, 0,          // kernel lrtb
        0, 0, 0, 0,          // pad lrtb
        op.getInputZp(),     // pad_value
        1                    // stride
    );

    Ndls ndls;

    // Ref tag has only U and V dimensions
    // This is enough to represent the FC input which is of shape [batch, input_channel] and weights
    // [output_channel, input_channels] The output is of shape [batch, output_channel] The U
    // dimension is the input_channel and V dimension is the output_channel
    ndls.add(
        NdlType::REF, {{DimType::H, MemDimTag::U, input_channel, 0},
                       {DimType::H, MemDimTag::V, output_channel, 0}}
    );

    // In FC the dedr agent takes the weight data in blocks of 64 output channel worth
    MemNdlDimsData dedr = {
        {DimType::L, MemDimTag::V, alu_group_width, 1},
        {DimType::H, MemDimTag::U, input_channel, max_out_channels}
    };
    if (out_ch_split > 1) {
        dedr.push_back({DimType::H, MemDimTag::V, out_ch_split, max_out_channels * input_channel});
    }
    ndls.add(NdlType::DEDR, dedr);

    // In FC the dewr agent takes the input data in blocks of wram_width pixels
    MemNdlDimsData dewr = {{DimType::L, MemDimTag::U, weight_dims[0], 1}};
    uint32_t weight_idx = weight_dims[0];
    for (size_t i = 1; i < weight_dims.size(); ++i) {
        if (weight_dims[i] == 1) {
            continue;
        }
        dewr.push_back({DimType::H, MemDimTag::U, weight_dims[i], weight_idx});
        weight_idx *= weight_dims[i];
    }
    if (out_ch_split > 1) {
        dewr.push_back({DimType::H, MemDimTag::V, out_ch_split, 0});
    }
    ndls.add(NdlType::DEWR, dewr);

    // In FC the debr agent reads the bias and scale in blocks of 16
    MemNdlDimsData debr = {
        {DimType::L, MemDimTag::O, HwInfo::breg_width, 1},
        {DimType::L, MemDimTag::V, HwInfo::act_limit, HwInfo::breg_width},
        {DimType::H, MemDimTag::V, max_out_channels / HwInfo::act_limit,
         HwInfo::act_limit * HwInfo::breg_width}
    };
    if (out_ch_split > 1) {
        debr.push_back(
            {DimType::H, MemDimTag::V, out_ch_split, max_out_channels * HwInfo::breg_width}
        );
    }
    ndls.add(NdlType::DEBR, debr);

    // In FC the deqw agent writes the output data in blocks of 16
    MemNdlDimsData deqw = {
        {DimType::L, MemDimTag::V, HwInfo::act_limit, 1},
        {DimType::H, MemDimTag::V, max_out_channels / HwInfo::act_limit, HwInfo::act_limit}
    };
    if (out_ch_split > 1) {
        deqw.push_back({DimType::H, MemDimTag::V, out_ch_split, max_out_channels});
    }
    ndls.add(NdlType::DEQW, deqw);

    // In FC CEDW is taking the weights and writing to the IRAM
    ndls.add(
        NdlType::CEDW, {{DimType::L, RegDimTag::B},
                        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width, 1},
                        {DimType::L, RegDimTag::G},
                        {DimType::H, RegDimTag::S, div_ceil(max_out_channels, alu_group_width),
                         HwInfo::iram_seg_width},
                        {DimType::H, RegDimTag::M},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, out_ch_split, HwInfo::iram_width}}
    );

    // In FC the CEDR takes the weights from IRAM and load it to ALU
    ndls.add(
        NdlType::CEDR,
        {{DimType::L, RegDimTag::B},
         {DimType::L, RegDimTag::D, HwInfo::max_input, 1},
         {DimType::L, RegDimTag::G, div_ceil(max_out_channels, alu_group_width), alu_group_width},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, input_channel * out_ch_split, HwInfo::iram_width}}
    );

    const auto repeats = out_ch_split;
    const auto ce_repeats = weight_dims.size() < 3 ? repeats : weight_dims[2] * repeats;
    ndls.add(
        NdlType::CEWW,
        {{DimType::L, RegDimTag::B},
         {DimType::L, RegDimTag::D, HwInfo::wram_seg_width, 1},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, ce_repeats * weight_dims[1], HwInfo::wram_seg_width}}
    );

    ndls.add(
        NdlType::CEWR,
        {{DimType::L, RegDimTag::B},
         {DimType::L, RegDimTag::D},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S, weight_dims[0], 1},
         {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, ce_repeats * weight_dims[1], HwInfo::wram_seg_width}}
    );

    ndls.add(
        NdlType::CEPR,
        {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
         {DimType::L, RegDimTag::D, max_out_channels, HwInfo::pram_dsize},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W, HwInfo::pram_depth, max_out_channels * HwInfo::pram_dsize},
         {DimType::H, RegDimTag::N, input_channel, 0},
         {DimType::H, RegDimTag::T, out_ch_split,
          HwInfo::pram_depth * max_out_channels * HwInfo::pram_dsize}}
    );

    ndls.add(
        NdlType::ACBW,
        {{DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
         {DimType::L, RegDimTag::D, HwInfo::act_limit, HwInfo::breg_width},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, out_ch_split * max_out_channels / HwInfo::act_limit,
          HwInfo::act_limit * HwInfo::breg_width}}
    );

    ndls.add(
        NdlType::ACBR,
        {{DimType::L, RegDimTag::B, HwInfo::breg_width, 1},
         {DimType::L, RegDimTag::D, HwInfo::act_limit, HwInfo::breg_width},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, max_out_channels / HwInfo::act_limit * out_ch_split,
          HwInfo::act_limit * HwInfo::breg_width}}
    );

    ndls.add(
        NdlType::ACPR,
        {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
         {DimType::L, RegDimTag::D, HwInfo::act_limit, HwInfo::pram_dsize},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S, max_out_channels / HwInfo::act_limit,
          HwInfo::pram_dsize * HwInfo::act_limit},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, out_ch_split, max_out_channels * HwInfo::pram_dsize}}
    );

    // NOTE: data and weight memory are exchanged
    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                // Operation to replace
        "fully_connected", // Task name
        op.getWeights(),   // Input tensor
        op.getInput(),     // Weights
        op.getScaleBias(), // BiasScale tensor,
        op.getInit(),      // Output tensor initializer
        slice_cfg_attr,    // Slice configuration
        ndls               // NDLs
    );

    return success();
}

} // namespace mlir::syna::torq
