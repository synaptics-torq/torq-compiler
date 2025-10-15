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

#define DEBUG_TYPE "torq-lower-torqhl-gather"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

template <>
LogicalResult GatherPattern::transform(torq_hl::GatherOp op, PatternRewriter &rewriter) const {
    auto input_type = llvm::cast<MemRefType>(op.getValues().getType());

    auto indices_type = llvm::cast<MemRefType>(op.getIndices().getType());
    auto indicesShape = indices_type.getShape();
    llvm::SmallVector<int64_t> indices_shape(indicesShape.begin(), indicesShape.end());

    uint32_t entry_size = input_type.getElementType().getIntOrFloatBitWidth() / 8;
    uint32_t idx_size = indices_type.getElementType().getIntOrFloatBitWidth() / 8;
    // output
    auto output_type = llvm::dyn_cast<MemRefType>(op.getInit().getType());

    auto input_strides = getEncodedStridesElements(input_type);
    auto indices_strides = getEncodedStridesElements(indices_type);
    auto input_shape = input_type.getShape();
    int channelCount = input_shape.size() == 3 ? input_shape[1] : 1;
    int channelStride = input_shape.size() == 3 ? input_strides[1] * entry_size : 0;
    // FIXME: will work only for i8, not for i16 or bf16
    // one should use getEncodedStridesBytes() instead
    int outChannelStride =
        input_shape.size() == 3 ? getEncodedStridesElements(output_type)[1] * entry_size : 0;

    auto output_strides = getEncodedStridesElements(output_type);
    // FIXME: This will produce dense ouput but is less efficient than using group size 4
    int indicesCount = indices_type.getNumElements();
    const uint32_t group_size = (indicesCount % 4 == 0) ? 4 : (indicesCount % 2 == 0) ? 2 : 1;
    LLVM_DEBUG({
        llvm::dbgs() << "\nindicesCount: " << indicesCount << " group_size: " << group_size << "\n";
    });

    uint32_t alu_width = alu_group_width / group_size;
    uint32_t max_input = 256 / alu_width;

    LLVM_DEBUG(llvm::dbgs() << "Indices strides:\n";
               for (int i = 0; i < indices_strides.size(); ++i) llvm::dbgs()
               << indices_strides[i] << " ";
               llvm::dbgs() << "\nchannelCount: " << channelCount << "\n";);

    // Create split of max_entries since idx dimension maybe aligned to 64 in activation cases.
    auto dense_splits = groupAsDenseDims(indices_shape, indices_strides);

    // Find the indices max entries based on the alignment
    dense_splits.front().size = div_ceil(dense_splits.front().size, group_size);
    dense_splits.front().stride = group_size;

    int64_t max_entries = std::accumulate(
        dense_splits.begin(), dense_splits.end(), 1,
        [&](int64_t x, ShapeStrideTuple y) { return x * y.size; }
    );

    LLVM_DEBUG({
        llvm::dbgs() << "Max entry splits:\n";
        for (int i = 0; i < dense_splits.size(); ++i) {
            llvm::dbgs() << dense_splits[i].size << " " << dense_splits[i].stride << "\n";
        }
        llvm::dbgs() << "Max entries:" << max_entries << "\n";
    });

    auto ctx = rewriter.getContext();

    auto slice_cfg_attr = torq_hw::SliceCFGAttr::get(
        ctx, {ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP, ALUOp0Mode::DBYP},
        {ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR, ALUOp1Mode::BOR},
        0xf,                                 // alu_d_unsigned
        0,                                   // alu_w_unsigned
        ACTMode::ACT,                        // act_mode
        {0, 0, 0, 0},                        // act left shift
        0,                                   // shift_factor
        std::numeric_limits<int32_t>::min(), // output_min
        std::numeric_limits<int32_t>::max(), // output_max
        0,                                   // act_zero_point
        0,                                   // no_p_clear
        0,                                   // no_p_output
        0, 0, 0, 0,                          // Kernel
        0, 0, 0, 0,                          // Pad
        0,                                   // pad_value
        1,                                   // stride
        0                                    // stride_offset

    );

    Ndls ndls;
    ndls.add(NdlType::REF, {{DimType::H, MemDimTag::X, group_size}});

    ndls.add(
        NdlType::DEDR,
        {{DimType::L, MemDimTag::X, entry_size, 1},
         {DimType::L, MemDimTag::G, group_size, 0},
         {DimType::H, MemDimTag::X, 1, 0},
         {DimType::H, MemDimTag::X, max_entries, 0},
         {DimType::H, MemDimTag::O, channelCount, channelStride}

        },
        0, // offset
        0, // set-id
        0, // sync_mode: consumer agent
        1  // sync_nhd: number of HDIM
    );

    MemNdlDimsData dedr_dims = {
        {DimType::L, MemDimTag::X, idx_size * group_size, 1},
        {DimType::H, MemDimTag::X, 1, idx_size * group_size}
    };

    int num_splits = dense_splits.size();
    for (int i = 0; i < num_splits; ++i) {
        auto s = dense_splits[i];
        dedr_dims.push_back({DimType::H, MemDimTag::X, s.size, s.stride * idx_size});
    }
    dedr_dims.push_back({DimType::H, MemDimTag::O, channelCount, 0});

    ndls.add(
        NdlType::DEDR, dedr_dims,
        0,   // offset
        1,   // set-id
        'P', // sync_mode: producer agent
        1    // sync_nhd: number of HDIM
    );

    MemNdlDimsData deqw_dims = {
        {DimType::L, MemDimTag::B, 1, 1},
        {DimType::L, MemDimTag::D, entry_size, 1},
        {DimType::H, MemDimTag::X, group_size, entry_size},
    };

    for (int i = 0; i < num_splits; ++i) {
        auto s = dense_splits[i];
        deqw_dims.push_back({DimType::H, MemDimTag::X, s.size, s.stride * entry_size});
    }
    deqw_dims.push_back({DimType::H, MemDimTag::O, channelCount, outChannelStride});
    ndls.add(NdlType::DEQW, deqw_dims);

    ndls.add(
        NdlType::CEDW, {{DimType::L, RegDimTag::B, 1, 1},
                        {DimType::L, RegDimTag::D, HwInfo::iram_seg_width / group_size, 1},
                        {DimType::L, RegDimTag::G, 1, HwInfo::iram_seg_width / group_size},
                        {DimType::H, RegDimTag::S, 1, 1},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, max_entries * channelCount, HwInfo::iram_width}}
    );

    ndls.add(
        NdlType::CEDR, {{DimType::L, RegDimTag::B, 1, 1},
                        {DimType::L, RegDimTag::D, alu_width, 1},
                        {DimType::L, RegDimTag::G, max_input, HwInfo::iram_seg_width / group_size},
                        {DimType::H, RegDimTag::S, 1, 1},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::iram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, max_entries * channelCount, HwInfo::iram_width}}
    );

    ndls.add(
        NdlType::CEPR, {{DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
                        {DimType::L, RegDimTag::D, 64, HwInfo::pdat_width},
                        {DimType::H, RegDimTag::S, 1, HwInfo::pdat_width * 64},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::pram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, max_entries * channelCount,
                         HwInfo::pram_width * HwInfo::pram_depth}}
    );

    ndls.add(
        NdlType::ACPR, {{DimType::L, RegDimTag::B, HwInfo::pdat_width, 1},
                        {DimType::L, RegDimTag::D, HwInfo::act_width, HwInfo::pdat_width},
                        {DimType::H, RegDimTag::S, group_size, HwInfo::pdat_width * alu_width},
                        {DimType::H, RegDimTag::M, 1, 0},
                        {DimType::H, RegDimTag::W, 1, HwInfo::pram_width},
                        {DimType::H, RegDimTag::N, 1, 0},
                        {DimType::H, RegDimTag::T, max_entries * channelCount,
                         HwInfo::pram_width * HwInfo::pram_depth}}
    );

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,       // Operation to replace
        "gather", // Task name
        op.getValues(), ValueRange{}, ValueRange{}, op.getInit(), op.getIndices(), ValueRange{},
        ValueRange{},
        slice_cfg_attr, // Slice configuration
        ndls            // NDLs
    );
    return success();
}

} // namespace mlir::syna::torq
