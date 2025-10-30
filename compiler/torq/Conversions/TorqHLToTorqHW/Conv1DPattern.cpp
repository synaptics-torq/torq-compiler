// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/TorqUtils.h"

#include "torq/Utils/Kernel.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

LogicalResult transformWithReduce(torq_hl::Conv1DOp op, PatternRewriter &rewriter) {
    struct DDim {
        enum { N, C, H, W };
    };
    struct KDim {
        enum { O, I, H, W };
    };
    auto inputMemrefType = llvm::cast<MemRefType>(op.getInput().getType());
    auto weightMemrefType = llvm::cast<MemRefType>(op.getWeights().getType());
    auto biasMemrefType = llvm::cast<MemRefType>(op.getScaleBias().getType());
    auto outputMemrefType = llvm::dyn_cast<MemRefType>(op.getInit().getType());

    auto inputShape = inputMemrefType.getShape();
    auto weightShape = weightMemrefType.getShape();
    auto outputShape = outputMemrefType.getShape();
    auto biasShape = biasMemrefType.getShape();
    auto inputStrides = getEncodedStridesElements(inputMemrefType);
    auto outputStrides = getEncodedStridesElements(outputMemrefType);
    const auto dataType = getDType(inputMemrefType.getElementType());
    const auto outputType = getDType(outputMemrefType.getElementType());
    const auto weightType = getDType(weightMemrefType.getElementType());
    const auto biasType = getDType(biasMemrefType.getElementType());

    const int32_t batchCount = inputShape[DDim::N];
    const int32_t inputChannels = inputShape[DDim::C];
    const int32_t outputChannels = outputShape[DDim::C];
    const int32_t outputWidth = outputShape[DDim::W];
    const int32_t kernelWidth = weightShape[KDim::W];
    const int32_t stride = op.getStride()[0];

    assert(
        inputShape.size() == 4 && weightShape.size() == 4 && outputShape.size() == 5 &&
        biasShape.size() == 1
    );
    assert(inputShape[DDim::H] == 1 && outputShape[DDim::H] == 1);
    assert(biasShape[0] == outputChannels);
    assert(dataType == DType::bf16 && weightType == DType::bf16 && biasType == DType::fp32);

    Slice slice;
    const int32_t inBlockSize = slice.alu.iWidth(dataType, weightType);
    const int32_t wBlockSize = slice.alu.wWidth(dataType);
    const int32_t blockSize = std::min(std::min(inBlockSize, wBlockSize), kernelWidth);
    const int32_t blockCount = div_ceil(kernelWidth, blockSize);
    const int32_t actBlockSize = std::min(slice.act.width(dataType, weightType), blockSize);
    const int32_t actBlockCount = div_ceil(blockSize, actBlockSize);

    LData input(
        {{batchCount, inputStrides[DDim::N] * sizeofType(dataType)},
         {inputChannels, inputStrides[DDim::C] * sizeofType(dataType)},
         {outputWidth, stride * sizeofType(dataType)},
         blockCount,
         blockSize},
        dataType
    );
    LData output(
        {{batchCount, outputStrides[DDim::N] * sizeofType(outputType)},
         {outputChannels, outputStrides[DDim::C] * sizeofType(outputType)},
         {outputWidth, kernelWidth * sizeofType(outputType)},
         blockCount,
         actBlockCount,
         actBlockSize},
        outputType
    );
    LData weight(
        {outputChannels,
         {inputChannels, kernelWidth * sizeofType(weightType)},
         blockCount,
         blockSize},
        weightType
    );
    LData biasScale({outputChannels}, biasType);

    // Note: outputs are generated in order, so no need to pad each channel
    // we just need a padding area at the end of the output tensor

    For(auto batch = slice.iterate(batchCount)) {
        For(auto oc = slice.iterate(outputChannels)) {
            BData bdata = slice.bram.load(biasScale[oc]);
            For(auto ow = slice.iterate(outputWidth)) {
                PData pdata;
                For(auto b = slice.iterate(blockCount)) {
                    For(auto ic = slice.iterate(inputChannels)) {
                        IData idata = slice.iram.load(input[batch][ic][ow][b]);
                        WData wdata = slice.wram.load(weight[oc][ic][b]);
                        pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);
                    }
                    For(auto a = slice.iterate(actBlockCount)) {
                        QData res = slice.act.rescaleClamp(
                            pdata[a], bdata, op.getShiftFactor(), op.getOutputZp(),
                            op.getOutputMin(), op.getOutputMax()
                        );
                        slice.store(output[batch][oc][ow][b][a], res);
                    }
                }
            }
        }
    }

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                                      // Operation to replace
        "Conv1DWithReduce",                      // Task name
        op.getInput(),                           // Input tensor
        op.getWeights(),                         // Weights
        op.getScaleBias(),                       // BiasScale tensor,
        op.getInit(),                            // Output tensor initializer
        slice.getCfgAttr(rewriter.getContext()), // Slice configuration
        slice.getNdls()                          // NDLs
    );

    return success();
}

// To program the Conv1D in hardware we need to generate the following descriptors:
// ref, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acpr, acbw, acbr
// For Conv1D the ALU will be working on MUL and ACC(add) operations.
// The conv1d is padded before he pattern using fill and insert slice operation
// The inptu data is alread resahped to filter_size * op_width using TransposeReshape Kernel
// Read as many as input data that can hold in alu group width(64 for int8 data) to process the
// inouts paralalley The weights of conv1d will be in format 1x1x1xW.Each weight is loaded one at a
// time and multiplied to all input parralely Feed the data and weights to the ALU.
// 5. To complete the Conv1D accumulation the full weights need to be loaded at the same location
// has to be processed.
// 6. The ALU will output alu group width pixels.
// 7. Read the bias & scale in blocks of 4.
// 8. The output is rescaled to int8 in blocks of 16 pixels
// 9. The output is written back to LRAM
// 10. Take the next block of 64 pixels and repeat the process.
LogicalResult transformWithTransposeReshape(torq_hl::Conv1DOp op, PatternRewriter &rewriter) {
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

    uint32_t max_input = HwInfo::max_input;

    max_input = max_input /
                ddat_width; // The total pixel that can be processed reduces based on the data width

    uint32_t wram_seg_width =
        HwInfo::wram_seg_width /
        wdat_width; // If int16 and bf16 only half the number of weights can be stored

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

    int32_t kernel_left = 0;
    int32_t kernel_right = 0;
    int32_t kernel_top = 0;
    int32_t kernel_bottom = 0;

    int baseOffset = 0;

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

    // DEDR agent reads the input data which is in HW format
    // The input data is read in blocks of 64 pixels and 64+kernel_size-1 pixels
    // The kernel_size-1 is for additional data to be used by kernel.
    // The tags A, J, U, A, V doesn't have any significance.
    // The tags are used for development purposes
    // A - Input data width
    // J - Kernel X dimension
    // U - Input channel
    // A - Input width pixel block
    ndls.add(
        NdlType::DEDR,
        {{DimType::L, MemDimTag::B, ddat_width, 1},
         {DimType::L, MemDimTag::A, max_input, ddat_width},
         {DimType::H, MemDimTag::J, ksize_x, input_strides[2]},
         {DimType::H, MemDimTag::U, input_shape[1], input_strides[1] * ddat_width},
         {DimType::H, MemDimTag::A, total_px_block, max_input * ddat_width}},
        baseOffset
    );

    // DEWR agent reads the weights which is in 1*1*1*W format
    // Only 1 weight read at a time
    // 0 - Kernel X dimension
    // A - Input width pixel block
    MemNdlDimsData dewr = {
        {DimType::L, MemDimTag::B, wdat_width, 1},
        {DimType::L, MemDimTag::D, 1, wdat_width},
        {DimType::H, MemDimTag::O, ksize_x, 1},
        {DimType::H, MemDimTag::A, total_px_block, 0}
    };

    ndls.add(NdlType::DEWR, dewr);

    // DEBR agent reads the bias and scale which is in O format
    // The bias and scale are grouped and are read in blocks of 4
    // The tags A, O, V doesn't have any significance.
    // The tags are used for development purposes
    MemNdlDimsData debr = {
        {DimType::L, MemDimTag::O, bdat_width, 1}, {DimType::H, MemDimTag::A, total_px_block, 0}
    };

    ndls.add(NdlType::DEBR, debr);

    // DEQW agent writes the output data which is in 1*W format
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
                         {DimType::H, MemDimTag::A, total_px_block, 0}}
        );
        deqw.push_back({DimType::S, MemDimTag::X, 2, sub_frame_size});
        deqw.push_back({DimType::S, MemDimTag::X, output_shape[3] / 2, 1});
        deqw.push_back({DimType::S, MemDimTag::Y, 2, sub_frame_size * 2});
        deqw.push_back({DimType::S, MemDimTag::Y, output_shape[2] / 2, output_shape[3] / 2});
    }
    else {
        deqw.insert(
            deqw.end(), {{DimType::L, MemDimTag::B, qdat_width, 1},
                         {DimType::L, MemDimTag::D, act_width, qdat_width},
                         {DimType::H, MemDimTag::A, max_input / act_width, act_width * qdat_width},
                         {DimType::H, MemDimTag::A, total_px_block, max_input * qdat_width}}
        );
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
        total_px_block * ksize_x; // This is to complete the full frame size taken 64 at a time
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
    // plus an additional 1 more time for first 2 parts
    // Refer HW doc for more information

    const uint32_t total_cedr_block =
        total_px_block * ksize_y; // This is to complete the full frame size taken 64 at a time

    ndls.add(
        NdlType::CEDR,
        {{DimType::L, RegDimTag::B, ddat_width, 1},
         {DimType::L, RegDimTag::D, max_input, ddat_width},
         {DimType::L, RegDimTag::G},
         {DimType::H, RegDimTag::S, 1, 0},
         {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
         {DimType::H, RegDimTag::W},
         {DimType::H, RegDimTag::N},
         {DimType::H, RegDimTag::T, total_cedr_block, HwInfo::iram_width * ddat_width}}
    );

    // CEWW agent fetches data from DEWR and writes the weight data to WRAM.
    // The data is read 1 at a time and its repeated ksize_x*total_pixel_block times
    // The tags B, D, G, S, M, W, N, T have specific meanings, please refer HW doc.
    // For now the weight data is just dumped into the WRAM without any processing.
    const auto repeats = total_px_block * ksize_x;

    RegNdlDimsData ceww = {
        {DimType::L, RegDimTag::B, wdat_width, 1},              // 1,1
        {DimType::L, RegDimTag::D, wram_seg_width, wdat_width}, // 1,1
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, repeats, 0}
    }; // 508,0
    ndls.add(NdlType::CEWW, ceww);

    // CEWR agent fetches the data from WRAM and writes to ALU.
    // The data is read 1 at a time and its repeated ksize_x*total_pixel_block times
    // The tags B, D, G, S, M, W, N, T have specific meanings, please refer HW doc.
    RegNdlDimsData cewr = {
        {DimType::L, RegDimTag::B, wdat_width},
        {DimType::L, RegDimTag::D, 1, wdat_width},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M, HwInfo::pram_depth, 0},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, repeats}
    };
    ndls.add(NdlType::CEWR, cewr);

    // CEPR agent helps in accumulation of the result from ALU.
    // After each ALU operation the results are send to PRAM,
    // which is later loaded for further accumulation by ALU
    // It takes the full mac_count ALU data and stores in PRAM.
    // This is repeated for the total number of accumulation that has to be done.
    // In the convolution case the total accumulation is (ksize_x).
    // This accumulation cycle is repeated for the total number of data to process.
    RegNdlDimsData cepr = {
        {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
        {DimType::L, RegDimTag::D, HwInfo::mac_count, HwInfo::pram_dsize},
        {DimType::L, RegDimTag::G},
        {DimType::H, RegDimTag::S},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W, HwInfo::pram_depth, HwInfo::mac_count * HwInfo::pram_dsize},
        {DimType::H, RegDimTag::N, ksize_y * input_shape[1] * ksize_x, 0},
        {DimType::H, RegDimTag::T, total_px_block,
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
        {DimType::H, RegDimTag::T, total_px_block, bdat_width}
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
        {DimType::H, RegDimTag::T, total_px_block, bdat_width}
    };
    ndls.add(NdlType::ACBR, acbr);

    if (!isBF16) {
        RegNdlDimsData acpw = {
            {DimType::L, RegDimTag::B, HwInfo::pram_dsize, 1},
            {DimType::L, RegDimTag::D, ddat_width, HwInfo::pram_dsize},
            {DimType::L, RegDimTag::G, act_width, HwInfo::pram_dsize * ddat_width},
            {DimType::H, RegDimTag::T, act_width * total_px_block, 1}
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
        {DimType::H, RegDimTag::S, (max_input / act_width), HwInfo::pram_dsize * HwInfo::act_width},
        {DimType::H, RegDimTag::M},
        {DimType::H, RegDimTag::W},
        {DimType::H, RegDimTag::N},
        {DimType::H, RegDimTag::T, total_px_block, HwInfo::mac_count * HwInfo::pram_dsize}
    };
    ndls.add(NdlType::ACPR, acpr);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                // Operation to replace
        "Conv1D",          // Task name
        op.getInput(),     // Input tensor
        op.getWeights(),   // Weights
        op.getScaleBias(), // BiasScale tensor,
        op.getInit(),      // Output tensor initializer
        slice_cfg_attr,    // Slice configuration
        ndls               // NDLs
    );

    return success();
}

template <>
LogicalResult Conv1DPattern::transform(torq_hl::Conv1DOp op, PatternRewriter &rewriter) const {
    auto outputMemrefType = llvm::dyn_cast<MemRefType>(op.getInit().getType());
    auto outputShape = outputMemrefType.getShape();
    switch (outputShape.size()) {
    case 5:
        return transformWithReduce(op, rewriter);
    case 4:
        return transformWithTransposeReshape(op, rewriter);
    }
    return op.emitError("unsupported output shape for Conv1D");
}

} // namespace mlir::syna::torq
