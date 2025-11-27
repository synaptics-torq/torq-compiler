// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// To program the conv2d in hardware we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// For conv2d the ALU will be working on MUL and ACC(add) operations.
// The ALU will be working on 4 groups of 64 pixels each.
// This can be modified to 8 groups or 16 groups of 32 pixels or 16 pixels.
// The data and weight will be read in blocks of 64 pixels and 32 weights.
// This is fed into the ALU in 64 pixels replicated to form 256 pixels
// and 4 out of 32 is used to form output channels for the 64 pixels.
// To generate the 64 pixel outputs the kernel size has to be taken into account.
// The kernel size of 1 means we need only 64 pixels to output 64 pixels.
// The kernel size of 3 means we need 66 pixels to output 64 pixels.
// The kernel size of 5 means we need 68 pixels to output 64 pixels and so on.
// The kernel size of 1 & 3 is the most common and is used in most cases.
// The flow of data is as follows:
// max_mode_pixels : 64, 32, 16
// max_mode_channels : 4, 8, 16
// ALU info 64x4: A(64)V(4)I(3)J(3)U(u)A(a/64)V(v/4)
// ALU info 32x8: A(32)V(8)I(3)J(3)U(u)A(a/32)V(v/8)
// ALU info 16x16: A(32)V(16)I(3)J(3)U(u)A(a/16)V(v/16)
// 1. Read the input data in blocks of max_mode_pixels+kernel_size-1 pixels.
//    The kernel_size-1 is for additional data to be used by kernel.
//    max_mode_pixel is based on the vectorization mode 64x4, 32x8, 16x16.
// 2. Read the weights in blocks of wram_width weights.
// 3. Feed the data and weights to the ALU.
// 4. If the kernel size in x-direction is more than 1 then shift the data to the right to get next
// pixel
//    If the kernel size in y-direction is more than 1 then fetch the next row of the data and
//    process
// 5. To complete the conv2d accumulation the full input channels at the same location has to be
// processed.
// 6. The ALU will output 256 pixels.
// 7. Read the bias & scale in blocks of 4.
// 8. The output is rescaled to int8 in blocks of 16 pixels
// 9. The output is written back to LRAM
// 10. Take the next block of 64 pixels and repeat the process.

// Note:
// For stride 2 the same concept applies but the data is taken from 4 parts of the input.
// The input is already divided into 4 parts based on the position of the pixels(Even-even,
// Even-odd, Odd-even, Odd-Odd). Each part is at a distance of frame_size/4.
// In the case of conv1x1 and stride 2, only the first part(even-even) is taken for processing,
// and processing done as stride 1 case.
//

LogicalResult convertToHw(torq_hl::Conv2DOp op, PatternRewriter &rewriter);

template <>
LogicalResult Conv2DPattern::transform(torq_hl::Conv2DOp op, PatternRewriter &rewriter) const {

    if (convertToHw(op, rewriter).succeeded()) {
        return success();
    }
    if (clUseNewKernels) {
        return op.emitError("New kernel failed");
    }

    // input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto weight_type = llvm::cast<MemRefType>(op.getWeights().getType());
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto input_shape = input_type.getShape();
    bool isBF16 = false;

    // Profile for the kernel
    // I:I8_t W:I8_t O:I8_t B:I64_t
    // I:I16_t W:I8_t O:I8_t B:I64_t
    // I:BF16_t W:BF16_t O:BF16_t B:I32_t
    int32_t ddat_width = 1;
    int32_t wdat_width = 1;
    int32_t bdat_width = HwInfo::breg_width;
    int32_t act_width = HwInfo::act_width;
    int32_t qdat_width = 1;
    NumberFormat alu_format = NumberFormat::I;
    NumberFormat act_format = NumberFormat::I;
    auto rounding_mode = RoundingMode::NTP;
    if (input_type.getElementType().isBF16() && weight_type.getElementType().isBF16()) {
        isBF16 = true;
        ddat_width = 2;
        wdat_width = 2;
        bdat_width = 4; // No scale for bf16 so 4 byte bias is enough
        qdat_width = 2;
        act_width = HwInfo::act_width / qdat_width; // Activation reduces to half for bf16
        alu_format = NumberFormat::BF;
        act_format = NumberFormat::BF;
        rounding_mode = RoundingMode::OFF; // No rounding needed for bf16
    }
    else if (input_type.getElementType().isInteger(16) &&
             weight_type.getElementType().isInteger(8)) {
        ddat_width = 2;
        qdat_width = 2;
        act_width = HwInfo::act_width / qdat_width; // Activation reduces to half for int16
    }
    else if (!input_type.getElementType().isInteger(8) ||
             !weight_type.getElementType().isInteger(8)) {
        return op.emitError("Compute profile not supported");
    }

    // weight
    auto weight_shape = weight_type.getShape();

    // output
    auto output_shape = output_type.getShape();

    auto input_strides = getEncodedStridesElements(input_type);
    auto output_strides = getEncodedStridesElements(output_type);

    // input/output data layout is NCHW
    const uint32_t out_h = output_shape[2];
    const uint32_t out_w = output_shape[3];

    const uint32_t ksize_x = weight_shape[K_SIZE_X];
    const uint32_t ksize_y = weight_shape[K_SIZE_Y];

    uint32_t max_input = alu_group_width;
    uint32_t max_channels = alu_groups;
    switch (op.getVectorizationMode()) {
    case torq_hl::VectorizationModeEnum::_32x8:
        max_input = 32;
        max_channels = 8;
        break;
    case torq_hl::VectorizationModeEnum::_16x16:
        max_input = 16;
        max_channels = 16;
        break;
    default:
        max_input = 64;
        max_channels = 4;
        break;
    }

    max_input = max_input /
                ddat_width; // The total pixel that can be processed reduces based on the data width
    // We want to maximize the ALU usage and process "max_channels" output channels at a time.
    const uint32_t max_out_channels = output_shape[1] >= 4 ? max_channels : 1;
    const uint32_t out_ch_split = div_ceil(output_shape[1], max_out_channels);
    const uint32_t max_in_channels = std::min<uint32_t>(input_shape[1], HwInfo::iram_depth);
    assert(max_in_channels > 0);
    const uint32_t in_ch_split = div_ceil(input_shape[1], max_in_channels);

    const std::vector<uint32_t> w_shape{
        static_cast<uint32_t>(input_shape[1]), ksize_y, ksize_x, max_out_channels
    };
    uint32_t wram_seg_width =
        HwInfo::wram_seg_width /
        wdat_width; // If int16 and bf16 only half the number of weights can be stored

    const std::vector<uint32_t> weight_dims =
        prepareWeightDims(w_shape, max_out_channels, max_channels, wram_seg_width);
    assert(!weight_dims.empty());

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

    if (stride == 2 && pad_left == pad_top && pad_left == (weight_shape[3] - 1) / 2) {
        stride_offset = 0;
    }

    if (stride == 2 && (ksize_x == 1 && ksize_y == 1)) {
        // apply conv only on top-left segment (EE)
        // since we take out dim it's ok
        stride = 1;
        stride_offset = 0;
    }
    if (stride == 2) {
        // Not clear why we have to overwrite the padding values here.
        // TODO: support cases where pad_top and/or pad_bottom is 0
        // not sure how we can handle XY tiling which requires different pad_top/pad_bottom config?
        pad_left = 1;
        pad_right = 1;
        pad_top = 1;
        pad_bottom = 1;
    }

    int32_t kernel_left = (weight_shape[3] - 1) / 2;
    int32_t kernel_right = weight_shape[3] - kernel_left - 1;
    int32_t kernel_top = (weight_shape[2] - 1) / 2;
    int32_t kernel_bottom = weight_shape[2] - kernel_top - 1;

    int baseOffset = 0;
    if (ksize_x > 1 || ksize_y > 1) {
        // Not clear why we have to overwrite the padding values here.
        // Don't overwrite pad_left and/or pad_top if padding disabled.
        pad_left = 1;
        pad_right = 1;
        if (pad_top) {
            pad_top = kernel_top;
        }
        if (pad_bottom) {
            pad_bottom = kernel_bottom;
        }
        if (pad_top == 0) {
            // NPU always starts fetching data kernel_top rows before the beginning of the data.
            // If no top padding, we need to add an offset to start fetching from the beginning
            // of the frame.
            baseOffset = kernel_top * input_strides[2];
        }
    }

    SmallVector<uint32_t> act_lsh = {0, 0, 0, 0};
    int32_t alu_d_unsigned = 0;
    int32_t alu_w_unsigned = 0;
    uint32_t act_sum_bits = 32;
    if (!isBF16 && ddat_width == 2 && wdat_width == 1) {
        // int16 processing is done by separately getting partials of MSB and LSB and accumulate at
        // activation stage
        act_lsh[0] = 0;
        act_lsh[1] = 8;
        act_lsh[2] = 0;
        act_lsh[3] = 8;

        alu_d_unsigned = 5;
        alu_w_unsigned = 0;
        act_sum_bits =
            32; // Activation needs higher precision for int16. FIXME TOSA spec recommends 48bit
    }

    // The ALU can process up to 3 filter elements (width-wise) per cycle.
    // `ksize_x_max` represents the maximum number of elements processed per cycle.
    // `ksize_x_split` is the total number of cycles required to process the full filter width
    // (`ksize_x`).
    //
    // In stride-2 mode with a 3x3 filter, the hardware first processes the EE quadrant using
    // weights w0 and w2, followed by the EO quadrant using w1 in the next cycle. Similar even-odd
    // splitting of weights is applied for 5x5 and 7x7 filters.
    //
    // Therefore, `ksize_x_parts` is adjusted by dividing the kernel width by the stride (for
    // stride-2 cases), to reflect how many positions will actually be processed.
    const int32_t ksize_x_parts = (stride == 2) ? div_ceil(ksize_x, stride) : ksize_x;
    const int32_t ksize_x_max = (ksize_x_parts < 3) ? ksize_x_parts : 3;
    const int32_t ksize_x_split = div_ceil(ksize_x_parts, ksize_x_max);

    const int32_t ksize_y_parts = div_ceil(ksize_y, stride);

    int32_t step_adj = div_ceil(kernel_left + kernel_right, stride);
    step_adj = step_adj > 2 ? 2 : step_adj;

    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL, ALUOp0Mode::MUL},
        {ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC, ALUOp1Mode::ACC},
        alu_d_unsigned,                                       // alu_d_unsigned
        alu_w_unsigned,                                       // alu_w_unsigned
        ACTMode::ACT,                                         // act_mode
        act_lsh,                                              // act left shift
        op.getShiftFactor(),                                  // shift_factor
        op.getOutputMin(),                                    // output_min
        op.getOutputMax(),                                    // output_max
        op.getOutputZp(),                                     // output_zp
        false,                                                // no_p_clear
        false,                                                // no_p_output
        kernel_left, kernel_right, kernel_top, kernel_bottom, // kernel lrtb
        pad_left, pad_right, pad_top, pad_bottom,             // pad lrtb
        op.getInputZp(),                                      // pad_value
        stride,                                               // stride
        stride_offset,                                        // stride_offset
        rounding_mode,                                        // rounding_mode
        WeightFormat::SI,                                     // weight_format
        0, 0,                                                 // alu_disable, act_disable
        alu_format,                                           // alu_format
        act_format,                                           // act_format
        act_sum_bits,                                         // act_sum_bits
        {}                                                    // table
    );

    Ndls ndls;

    // ref
    // Ref descriptor is used to describe all the dimensions of the data
    // I - Kernel X dim
    // J - Kernel Y dim
    // U - Input channel
    // V - Output channel
    // X - Input width
    // Y - Input height
    MemNdlDimsData ref = {
        {DimType::H, MemDimTag::I, ksize_x, 0},
        {DimType::H, MemDimTag::J, ksize_y, 0},
        {DimType::H, MemDimTag::U, input_shape[1], 0},
        {DimType::H, MemDimTag::V, output_shape[1], 0}
    };

    ref.push_back({DimType::H, MemDimTag::X, output_shape[3], 0});
    ref.push_back({DimType::H, MemDimTag::Y, output_shape[2], 0});
    ref.push_back({DimType::H, MemDimTag::G, input_shape[1], 0});
    ndls.add(NdlType::REF, ref);

    // DEDR agent reads the input data which is in NCHW format
    // The input data is read in blocks of 64 pixels and 64+kernel_size-1 pixels
    // The kernel_size-1 is for additional data to be used by kernel.
    // The tags A, J, U, A, V doesn't have any significance.
    // The tags are used for development purposes
    // A - Input data height and width dimension combined
    // J - Kernel Y dimension
    // U - Input channel
    // V - Output channel
    // For stride 2 case the input data is read in 4 parts based on the position of the pixels
    // The input data is already divided into 4 parts based on the position of the pixels(Even-even,
    // Even-odd, Odd-even, Odd-Odd). Each part is at a distance of frame_size/4.
    // Refer original HW code for more details

    if (stride == 1) {
        if (ksize_x_split > 1) {
            ndls.add(
                NdlType::DEDR,
                {{DimType::L, MemDimTag::B, ddat_width, 1},
                 {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
                 {DimType::H, MemDimTag::I, ksize_x_split, ksize_x_max},
                 {DimType::H, MemDimTag::J, ksize_y, output_shape[3] * ddat_width},
                 {DimType::H, MemDimTag::U, input_shape[1], input_strides[1] * ddat_width},
                 {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
                 {DimType::H, MemDimTag::V, out_ch_split, 0}},
                baseOffset
            );
        }
        else {
            ndls.add(
                NdlType::DEDR,
                {{DimType::L, MemDimTag::B, ddat_width, 1},
                 {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
                 {DimType::H, MemDimTag::J, ksize_y, output_shape[3] * ddat_width},
                 {DimType::H, MemDimTag::U, input_shape[1], input_strides[1] * ddat_width},
                 {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
                 {DimType::H, MemDimTag::V, out_ch_split, 0}},
                baseOffset
            );
        }
    }
    else if (stride == 2) {
        int32_t data_part_size = (input_strides[1] / 4) * ddat_width;
        int32_t start_pos_x = (-kernel_left + stride_offset) & 1;
        int32_t start_pos_y = (-kernel_top + stride_offset) & 1;
        int32_t kernel_left_even = (kernel_left - stride_offset + 1) >> 1;
        int32_t kernel_left_odd = kernel_left - stride_offset - kernel_left_even;
        int32_t kernel_top_even = (kernel_top - stride_offset + 1) >> 1;
        int32_t kernel_top_odd = kernel_top - stride_offset - kernel_top_even;

        // Determine the starting quadrant based on the starting position.
        // For 7x7 kernels, processing starts from the EE quadrant.
        // For 5x5 kernels, processing starts from the EO quadrant in the X direction and the OE
        // quadrant in the Y direction.
        //
        // The `if` condition handles cases where the kernel spans more than 3 values in the X
        // direction within a quadrant, requiring multiple cycles for processing. If the span is
        // less than or equal to 3, it can be processed within a single cycle.
        if (ksize_x_split > 1) {
            ndls.add(
                NdlType::DEDR,
                {{DimType::L, MemDimTag::B, ddat_width, 1},
                 {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
                 {DimType::H, MemDimTag::I, ksize_x_split, ksize_x_max},
                 {DimType::H, MemDimTag::I, 2,
                  (start_pos_x == 0 ? data_part_size : -data_part_size) +
                      (kernel_left_even - kernel_left_odd) * ddat_width},
                 {DimType::H, MemDimTag::J, 2,
                  (start_pos_y == 0 ? 2 * data_part_size : -2 * data_part_size) +
                      output_shape[3] * (kernel_top_even - kernel_top_odd) * ddat_width},
                 {DimType::H, MemDimTag::J, ksize_y_parts, output_shape[3] * ddat_width},
                 {DimType::H, MemDimTag::U, input_shape[1], input_strides[1] * ddat_width},
                 {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
                 {DimType::H, MemDimTag::V, out_ch_split, 0}},
                (stride == 2 ? (2 * start_pos_y + start_pos_x) * data_part_size : 0)
            );
        }
        else {
            ndls.add(
                NdlType::DEDR,
                {{DimType::L, MemDimTag::B, ddat_width, 1},
                 {DimType::L, MemDimTag::A, max_input + step_adj, ddat_width},
                 {DimType::H, MemDimTag::I, 2,
                  (start_pos_x == 0 ? data_part_size : -data_part_size) +
                      (kernel_left_even - kernel_left_odd) * ddat_width},
                 {DimType::H, MemDimTag::J, 2,
                  (start_pos_y == 0 ? 2 * data_part_size : -2 * data_part_size) +
                      (output_shape[3] * (kernel_top_even - kernel_top_odd)) * ddat_width},
                 {DimType::H, MemDimTag::J, ksize_y_parts, output_shape[3] * ddat_width},
                 {DimType::H, MemDimTag::U, input_shape[1], input_strides[1] * ddat_width},
                 {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width},
                 {DimType::H, MemDimTag::V, out_ch_split, 0}},
                (stride == 2 ? (2 * start_pos_y + start_pos_x) * data_part_size : 0)
            );
        }
    }
    else {
        assert(false && "unsupported stride");
    }

    // DEWR agent reads the weights which is in OIHWO format
    // The weights are read in blocks of 32 weights
    // The tags A, O, V doesn't have any significance.
    // The tags are used for development purposes
    MemNdlDimsData dewr = {
        {DimType::L, MemDimTag::B, wdat_width, 1},
        {DimType::L, MemDimTag::D, weight_dims[0], wdat_width}
    };

    uint32_t weight_idx = weight_dims[0];
    for (size_t i = 1; i < weight_dims.size(); ++i) {
        if (weight_dims[i] == 1) {
            continue;
        }
        dewr.push_back({DimType::H, MemDimTag::O, weight_dims[i], weight_idx * wdat_width});
        weight_idx *= weight_dims[i];
    }
    dewr.push_back({DimType::H, MemDimTag::A, total_px_block, 0});
    if (out_ch_split > 1) {
        auto per_oc_weights =
            std::accumulate(weight_dims.begin(), weight_dims.end(), 1, std::multiplies<size_t>());
        dewr.push_back({DimType::H, MemDimTag::V, out_ch_split, per_oc_weights * wdat_width});
    }
    ndls.add(NdlType::DEWR, dewr);

    // DEBR agent reads the bias and scale which is in O format
    // The bias and scale are grouped and are read in blocks of 4
    // The tags A, O, V doesn't have any significance.
    // The tags are used for development purposes
    MemNdlDimsData debr = {
        {DimType::L, MemDimTag::O, bdat_width, 1},
        {DimType::H, MemDimTag::V, max_out_channels, bdat_width},
        {DimType::H, MemDimTag::A, total_px_block, 0}
    };

    if (out_ch_split > 1) {
        debr.push_back({DimType::H, MemDimTag::V, out_ch_split, max_out_channels * bdat_width});
    }
    ndls.add(NdlType::DEBR, debr);

    // DEQW agent writes the output data which is in NCHW format
    // The output data is written in blocks of max_activation_pixels,
    // this is repeated total_alu_mac/max_activation_pixels times.
    // The tags A, V doesn't have any significance, but in special cases such as stride 2 the A-tag
    // has significance. The tags are used for development purposes For stride 2 case the previous
    // layer can contribute to optimize the segmentation process If the current layer is the parent
    // of stride 2 layer then the segmentation is done by this agent. The data is split into 4 parts
    // based on the position of the pixels(Even-even, Even-odd, Odd-even, Odd-Odd). Each part is at
    // a distance of frame_size/4. The S-tag is used to split the data into 4 parts. If the S-tag is
    // used then the A-tag should be 0.
    MemNdlDimsData deqw;
    auto segment_output = op.getSegmentOutput();
    if (segment_output) {
        uint32_t sub_frame_size = output_strides[1] / 4;
        deqw.insert(
            deqw.end(), {{DimType::L, MemDimTag::A, act_width, 1},
                         {DimType::H, MemDimTag::A, max_input / act_width, 0},
                         {DimType::H, MemDimTag::V, max_out_channels, sub_frame_size * 4},
                         {DimType::H, MemDimTag::A, total_px_block, 0}}
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
             {DimType::L, MemDimTag::D, act_width, qdat_width},
             {DimType::H, MemDimTag::A, max_input / act_width, act_width * qdat_width},
             {DimType::H, MemDimTag::V, max_out_channels, output_strides[1] * qdat_width},
             {DimType::H, MemDimTag::A, total_px_block, max_input * qdat_width}}
        );
        if (out_ch_split > 1) {
            deqw.push_back(
                {DimType::H, MemDimTag::V, out_ch_split,
                 max_out_channels * output_strides[1] * qdat_width}
            );
        }
    }
    ndls.add(NdlType::DEQW, deqw);

    // CEDW agent fetches the data loaded by DEDR and writes to IRAM.
    // The data is read in blocks of max_mode_pixels+kernel_size-1 pixels and written
    // in block of iram_seg_width pixels.
    // With multiple cycles it can write iram_width pixels to IRAM if needed.
    // The tags B, D, G, S, M, W, N, T have specific meanings
    // B - Data width
    // D - Number of data in a word
    // G - Number of groups of word
    // S - Steps to move to next data within word
    // M - Number of times to repeat a word
    // W - Number of words
    // N - Number of times to repeat a block
    // T - Total number of blocks
    // Refer HW doc for more information
    const uint32_t total_cedw_block =
        total_px_block * out_ch_split * in_ch_split * ksize_y *
        ksize_x_split; // This is to complete the full frame size taken 64 at a time
    const uint32_t total_cedw_block_stride2 =
        4 * total_px_block * out_ch_split *
        ksize_y; //  This is to complete the full frame size taken 64 at a time,
                 // in this case we have 4 parts due to stride 2 optimization.
    if (stride == 1) {
        ndls.add(
            NdlType::CEDW,
            {{DimType::L, RegDimTag::B, ddat_width, 1},
             {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / ddat_width, ddat_width},
             {DimType::L, RegDimTag::G},
             {DimType::H, RegDimTag::S},
             {DimType::H, RegDimTag::M},
             {DimType::H, RegDimTag::W},
             {DimType::H, RegDimTag::N},
             {DimType::H, RegDimTag::T, total_cedw_block, HwInfo::iram_width * ddat_width}}
        );
    }
    else if (stride == 2) {
        ndls.add(
            NdlType::CEDW,
            {{DimType::L, RegDimTag::B, ddat_width},
             {DimType::L, RegDimTag::D, HwInfo::iram_seg_width, ddat_width},
             {DimType::L, RegDimTag::G},
             {DimType::H, RegDimTag::S},
             {DimType::H, RegDimTag::M},
             {DimType::H, RegDimTag::W},
             {DimType::H, RegDimTag::N},
             {DimType::H, RegDimTag::T, total_cedw_block_stride2, HwInfo::iram_width * ddat_width}}
        );
    }
    else {
        assert(false && "unsupported stride");
    }

    // CEDR agent fetches the data in IRAM and writes to ALU.
    // The data is read in blocks of 64 and can be repeated 4 times to complete 256 pixels.
    // In convolution case the data is read in blocks of 64 pixels,
    // and repeated kernel_size-1 times each time shifting by 1 pixel location in the IRAM data.
    // The tags B, D, G, S, M, W, N, T have specific meanings
    // B - Data width
    // D - Number of data in a word
    // G - Number of groups of word
    // S - Steps to move to next data within word
    // M - Number of times to repeat a word
    // W - Number of words
    // N - Number of times to repeat a block
    // T - Total number of blocks
    // In stride 2 case we need to repeat this operation for all parts once,
    // plus an additional 1 more time for first 2 parts
    // Refer HW doc for more information

    const uint32_t total_cedr_block =
        total_px_block * in_ch_split * out_ch_split * ksize_y *
        ksize_x_split; // This is to complete the full frame size taken 64 at a time
    const uint32_t total_cedr_block_stride2 =
        4 * total_px_block * out_ch_split *
        ksize_y; // This is to complete the full frame size taken 64 at a time,
                 // in this case we have 4 parts due to stride 2 optimization.
    if (stride == 1) {
        ndls.add(
            NdlType::CEDR,
            {{DimType::L, RegDimTag::B, ddat_width, 1},
             {DimType::L, RegDimTag::D, max_input, ddat_width},
             {DimType::L, RegDimTag::G},
             {DimType::H, RegDimTag::S, ksize_x_max, ddat_width},
             {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
             {DimType::H, RegDimTag::W},
             {DimType::H, RegDimTag::N},
             {DimType::H, RegDimTag::T, total_cedr_block, HwInfo::iram_width * ddat_width}}
        );
    }
    else if (stride == 2) {
        ndls.add(
            NdlType::CEDR,
            {{DimType::L, RegDimTag::B, ddat_width, 1},
             {DimType::L, RegDimTag::D, max_input, ddat_width},
             {DimType::L, RegDimTag::G},
             {DimType::H, RegDimTag::S, 2, 1},
             {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
             {DimType::H, RegDimTag::W},
             {DimType::H, RegDimTag::N},
             {DimType::H, RegDimTag::T, total_cedr_block_stride2, HwInfo::iram_width * ddat_width}}
        );
    }
    else {
        assert(false && "unsupported stride");
    }

    // CEWW agent fetches data from DEWR and writes the weight data to WRAM.
    // The weight data is read in blocks of 32 weights.
    // The tags B, D, G, S, M, W, N, T have specific meanings, please refer HW doc.
    // For now the weight data is just dumped into the WRAM without any processing.
    const auto repeats = total_px_block * out_ch_split;
    const auto ce_repeats = weight_dims.size() < 3 ? repeats : weight_dims[2] * repeats;

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, wdat_width, 1},
        {DimType::L, RegDimTag::D, wram_seg_width, wdat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, ce_repeats * weight_dims[1], HwInfo::wram_seg_width * wdat_width}
    };
    ndls.add(NdlType::CEWW, ceww);

    // CEWR agent fetches the data from WRAM and writes to ALU.
    // The data is read in blocks of 4 and repeated 8 times to complete 32 weights
    // The tags B, D, G, S, M, W, N, T have specific meanings, please refer HW doc.
    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, wdat_width},
        {DimType::L, RegDimTag::D, 1, wdat_width},
        {DimType::L, RegDimTag::G, max_out_channels, wdat_width},
        {DimType::H, RegDimTag::S, div_ceil(weight_dims[0], max_out_channels),
         max_out_channels * wdat_width},
        {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, ce_repeats * weight_dims[1], HwInfo::wram_seg_width * wdat_width}
    };
    ndls.add(NdlType::CEWR, cewr);

    // CEPR agent helps in accumulation of the result from ALU.
    // After each ALU operation the results are send to PRAM,
    // which is later loaded for further accumulation by ALU
    // It takes the full mac_count ALU data and stores in PRAM.
    // This is repeated for the total number of accumulation that has to be done.
    // In the convolution case the total accumulation is (kernel_y * input_channels).
    // This accumulation cycle is repeated for the total number of data to process.
    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, HwInfo::pram_depth, HwInfo::mac_count * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::N, ksize_y * input_shape[1] * ksize_x, 0},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split,
         HwInfo::pram_depth * HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::CEPR, cepr);

    // ACBW agent fetches the bias-scale data from DEBR and stores in BREG
    // The data is read in groups of 8 bytes, 4-byte bias & 4-byte scale.
    RegNdlDimsData acbw = {
        {DimType::L, RegDimTag::B, bdat_width, 1},
        {DimType::L, RegDimTag::D},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, out_ch_split * max_out_channels * total_px_block, bdat_width}
    };
    ndls.add(NdlType::ACBW, acbw);

    // ACBR agent fetches the bias-scale data from BREG and writes to the ACT.
    // The data is read in groups of 8 bytes, 4-byte bias & 4-byte scale.
    // The act works on 16 pixels at a time so needs 16 repeats to finish 256 mac.
    // In the 16 repeats, each 4 repeat is for a single output channel.
    // Each bias scale is repeated 4 times to complete the pixels of a single output channel.
    // The M-tag shows the repetition needed.
    RegNdlDimsData acbr = {
        {DimType::L, RegDimTag::B, bdat_width, 1},
        {DimType::L, RegDimTag::D},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M, (max_input / act_width), 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, out_ch_split * max_out_channels * total_px_block, bdat_width}
    };
    ndls.add(NdlType::ACBR, acbr);

    if (!isBF16) {
        RegNdlDimsData acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
            {DimType::L, RegDimTag::D, ddat_width, HwInfo::pram_dsize},
            {DimType::L, RegDimTag::G, act_width, HwInfo::pram_dsize * ddat_width},
            {DimType::H, RegDimTag::T, act_width * out_ch_split * total_px_block, 1}
        };
        ndls.add(NdlType::ACPW, acpw);
    }

    // ACPR agent fetches the data in PRAM and sends to ACT.
    // The data is 4-byte long.
    // The data is taken 16 at a time since ACT can only process 16 values in parallel.
    // The tag B here will have size as 4 to denote the data size
    // The tag D is 16 here
    // The tag S denotes the number of repetition needed to complete the data in PRAM,
    // in this case it is 256/16.
    // Since the 256 values are present in a single word/block the S-tag is used to fetch the data
    RegNdlDimsData acpr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S, (max_input / act_width) * max_out_channels,
         HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block * out_ch_split,
         HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                // Operation to replace
        "conv2d",          // Task name
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
