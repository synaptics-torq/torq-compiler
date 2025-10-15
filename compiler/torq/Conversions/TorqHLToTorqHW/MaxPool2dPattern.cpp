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
LogicalResult
MaxPool2dPattern::transform(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) const {
    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();

    auto weight_shape = op.getKernel();

    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();

    auto input_strides = getEncodedStridesElements(input_type);

    auto output_strides = getEncodedStridesElements(output_type);

    uint32_t act_width = HwInfo::act_width;
    int32_t ddat_width = 1;
    int32_t qdat_width = 1;
    NumberFormat alu_format = NumberFormat::I;
    NumberFormat act_format = NumberFormat::I;
    auto rounding_mode = RoundingMode::NTP;

    SmallVector<uint32_t> act_lsh = {0, 0, 0, 0};
    int32_t alu_d_unsigned = 0;
    int32_t alu_w_unsigned = 0;
    uint32_t act_sum_bits = 32;

    if (input_type.getElementType().isInteger(16)) {
        ddat_width = 2;
        qdat_width = 2;

        alu_d_unsigned = 5;
    }

    // input/output data layout is NCHW
    const uint32_t input_channel = input_shape[1];
    const uint32_t out_h = output_shape[2];
    const uint32_t out_w = output_shape[3];

    const uint32_t ksize_x = weight_shape[0];
    const uint32_t ksize_y = weight_shape[1];

    uint32_t max_input = alu_group_width / ddat_width;
    uint32_t max_channels = alu_groups;

    uint32_t max_out_channels = output_shape[1] < max_channels ? 1 : max_channels;

    const uint32_t out_ch_split = div_ceil(input_channel, max_out_channels);

    const uint32_t out_frame_size = align_ceil(out_h * out_w, max_input);
    assert(out_frame_size > 0 && "frame size must be greater than 0");

    const uint32_t total_px_block = div_ceil(out_frame_size, max_input);

    int32_t stride = op.getStride()[0];
    assert(stride <= 2 && "stride must be less than or equal to 2");

    int32_t pad_left = op.getPad()[0];
    int32_t pad_right = op.getPad()[1];
    int32_t pad_top = op.getPad()[2];
    int32_t pad_bottom = op.getPad()[3];
    int32_t stride_offset = (stride == 2) ? 1 : 0;

    if (stride == 2 && pad_left == pad_top && pad_left == (weight_shape[1] - 1) / 2) {
        stride_offset = 0;
    }
    if (stride == 2) {
        pad_left = 1;
        pad_right = 1;
        pad_top = 1;
        pad_bottom = 1;
    }

    int32_t kernel_left = (weight_shape[1] - 1) / 2;
    int32_t kernel_right = weight_shape[1] - kernel_left - 1;
    int32_t kernel_top = (weight_shape[0] - 1) / 2;
    int32_t kernel_bottom = weight_shape[0] - kernel_top - 1;

    const int32_t ksize_x_parts = (stride == 2) ? div_ceil(ksize_x, stride) : ksize_x;
    const int32_t ksize_x_max = (ksize_x_parts < 3) ? ksize_x_parts : 3;
    const int32_t ksize_x_split = div_ceil(ksize_x_parts, ksize_x_max);
    const int32_t ksize_y_parts = div_ceil(ksize_y, stride);

    if (ksize_x > 1 || ksize_y > 1) {
        pad_left = 1;
        pad_right = 1;
        pad_top = kernel_top;
        pad_bottom = kernel_bottom;
    }

    int32_t step_adj = div_ceil(kernel_left + kernel_right, stride);
    step_adj = step_adj > 2 ? 2 : step_adj;

    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::MAX, ALUOp1Mode::MAX, ALUOp1Mode::MAX, ALUOp1Mode::MAX},
        alu_d_unsigned,                                       // alu_d_unsigned
        alu_w_unsigned,                                       // alu_w_unsigned
        ACTMode::ACT,                                         // act_mode
        act_lsh,                                              // act left shift
        0,                                                    // shift_factor
        std::numeric_limits<int32_t>::min(),                  // output_min
        std::numeric_limits<int32_t>::max(),                  // output_max
        0,                                                    // act_zero_point
        0,                                                    // no_p_clear
        0,                                                    // no_p_output
        kernel_left, kernel_right, kernel_top, kernel_bottom, // Kernel
        pad_left, pad_right, pad_top, pad_bottom,             // Pad
        op.getInputZp(),                                      // pad_value
        stride,                                               // stride
        stride_offset,                                        // stride_offset
        rounding_mode,                                        // rounding_mode
        WeightFormat::SI, 0, 0, alu_format, act_format,
        act_sum_bits, // alu_format, act_format, act_sum_bits
        {}
    );

    Ndls ndls;

    ndls.add(
        NdlType::REF, {{DimType::H, MemDimTag::I, ksize_x, 0},
                       {DimType::H, MemDimTag::J, ksize_y, 0},
                       {DimType::H, MemDimTag::X, output_shape[3], 0},
                       {DimType::H, MemDimTag::Y, output_shape[2], 0},
                       {DimType::H, MemDimTag::G, input_channel, 0}}
    );

    if (stride == 1) {
        ndls.add(
            NdlType::DEDR,
            {{DimType::L, MemDimTag::B, ddat_width, 1},
             {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
             {DimType::H, MemDimTag::I, ksize_x_split, ksize_x_max},
             {DimType::H, MemDimTag::J, ksize_y, input_shape[3] * ddat_width},
             {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
             {DimType::H, MemDimTag::G, max_out_channels, input_strides[1] * ddat_width},
             {DimType::H, MemDimTag::G, out_ch_split,
              max_out_channels * input_strides[1] * ddat_width}},
            0
        );
    }
    else if (stride == 2) {
        const uint32_t data_part_size = input_strides[1] / 4;
        int32_t start_pos_x = (-kernel_left + stride_offset) & 1;
        int32_t start_pos_y = (-kernel_top + stride_offset) & 1;
        int32_t kernel_left_even = (kernel_left - stride_offset + 1) >> 1;
        int32_t kernel_left_odd = kernel_left - stride_offset - kernel_left_even;
        int32_t kernel_top_even = (kernel_top - stride_offset + 1) >> 1;
        int32_t kernel_top_odd = kernel_top - stride_offset - kernel_top_even;

        ndls.add(
            NdlType::DEDR,
            {{DimType::L, MemDimTag::B, ddat_width, 1},
             {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
             {DimType::H, MemDimTag::I, ksize_x_split, ksize_x_max * ddat_width},
             {DimType::H, MemDimTag::I, 2,
              ((start_pos_x == 0 ? data_part_size : -data_part_size) +
               (kernel_left_even - kernel_left_odd)) *
                  ddat_width},
             {DimType::H, MemDimTag::J, 2,
              ((start_pos_y == 0 ? 2 * data_part_size : -2 * data_part_size) +
               output_shape[3] * (kernel_top_even - kernel_top_odd)) *
                  ddat_width},
             {DimType::H, MemDimTag::J, ksize_y_parts, output_shape[3] * ddat_width},
             {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
             {DimType::H, MemDimTag::G, max_out_channels, input_strides[1] * ddat_width},
             {DimType::H, MemDimTag::G, out_ch_split,
              max_out_channels * input_strides[1] * ddat_width}},
            (stride == 2 ? (2 * start_pos_y + start_pos_x) * data_part_size * ddat_width : 0)
        );
    }
    else {
        assert(false && "unsupported stride");
    }

    MemNdlDimsData deqw;
    auto segment_output = op.getSegmentOutput();
    if (segment_output) {
        const uint32_t sub_frame_size = output_strides[1] / 4;
        deqw.insert(
            // FIXME: segment_output int16 is not supported yet
            deqw.end(), {{DimType::L, MemDimTag::A, act_width, 1},
                         {DimType::H, MemDimTag::A, max_input / act_width, 0},
                         {DimType::H, MemDimTag::A, total_px_block, 0},
                         {DimType::H, MemDimTag::V, max_out_channels, sub_frame_size * 4}}
        );
        if (out_ch_split > 1) {
            deqw.push_back(
                {DimType::H, MemDimTag::V, out_ch_split, max_out_channels * sub_frame_size * 4}
            );
        }
        deqw.push_back({DimType::S, MemDimTag::X, 2, sub_frame_size});
        deqw.push_back({DimType::S, MemDimTag::X, output_shape[3] / 2, 1});
        deqw.push_back({DimType::S, MemDimTag::Y, 2, sub_frame_size * 2});
        deqw.push_back({DimType::S, MemDimTag::Y, output_shape[2] / 2, output_shape[3] / 2});
    }
    else {
        deqw.insert(
            deqw.end(),
            {{DimType::L, MemDimTag::B, qdat_width, 1},
             {DimType::L, MemDimTag::A, act_width, qdat_width},
             {DimType::H, MemDimTag::A, max_input / act_width, act_width * qdat_width},
             {DimType::H, MemDimTag::A, total_px_block, max_input * qdat_width},
             {DimType::H, MemDimTag::V, max_out_channels, output_strides[1] * qdat_width}}
        );
        if (out_ch_split > 1) {
            deqw.push_back(
                {DimType::H, MemDimTag::V, out_ch_split,
                 max_out_channels * output_strides[1] * qdat_width}
            );
        }
    }
    ndls.add(NdlType::DEQW, deqw);

    uint32_t cedw_cedr_t_size = 0;
    if (stride == 1) {
        cedw_cedr_t_size =
            total_px_block * out_ch_split * ksize_y * ksize_x_split * max_out_channels;
    }
    else if (stride == 2) {
        cedw_cedr_t_size = 4 * total_px_block * out_ch_split * ksize_y * max_out_channels;
    }
    else {
        assert(false && "unsupported stride");
    }
    ndls.add(
        NdlType::CEDW, {{DimType::L, RegDimTag::B, ddat_width, 1},
                        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / ddat_width, ddat_width},
                        {DimType::L, RegDimTag::G},
                        {DimType::H, RegDimTag::S, 1, HwInfo::iram_seg_width},
                        {DimType::H, RegDimTag::M},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, cedw_cedr_t_size, HwInfo::iram_width}}
    );

    ndls.add(
        NdlType::CEDR, {{DimType::L, RegDimTag::B, ddat_width, 1},
                        {DimType::L, RegDimTag::D, max_input, ddat_width},
                        {DimType::L, RegDimTag::G, 1, 1},
                        {DimType::H, RegDimTag::S, ksize_x_max, ddat_width},
                        {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, cedw_cedr_t_size, HwInfo::iram_width}}
    );

    ndls.add(
        NdlType::CEPR,
        {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
         {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S},
         {DimType::H, RegDimTag::M},
         {DimType::H, RegDimTag::W, HwInfo::pram_depth, HwInfo::mac_count * HwInfo::pram_dsize},
         {DimType::H, RegDimTag::N, ksize_y * ksize_x, 0},
         {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
          HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}}
    );

    ndls.add(
        NdlType::ACPR, {{DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
                        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
                        {DimType::L, RegDimTag::G},
                        {DimType::H, RegDimTag::S, (max_input / act_width),
                         HwInfo::pram_dsize * HwInfo::act_width},
                        {DimType::H, RegDimTag::M},
                        {DimType::H, RegDimTag::W},
                        {DimType::H, RegDimTag::N},
                        {DimType::H, RegDimTag::T, total_px_block * out_ch_split * max_out_channels,
                         HwInfo::mac_count * HwInfo::pram_dsize}}
    );

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                // Operation to replace
        "maxpool2d",       // Task name
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
