// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/TorqHLToTorqHW/Passes.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "llvm/Support/CommandLine.h"

const uint32_t constexpr K_SIZE_X = 3;
const uint32_t constexpr K_SIZE_Y = 2;

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clUseNewKernels;

template <class HlOp> class OpPattern : public OpRewritePattern<HlOp> {
  public:
    using OpRewritePattern<HlOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(HlOp hlOp, PatternRewriter &rewriter) const override {
        return transform(hlOp, rewriter);
    }

  private:
    LogicalResult transform(HlOp hlOp, PatternRewriter &rewriter) const;

    /// ALU can be used in 3 different modes: 64x4, 32x8 and 16x16
    const uint32_t alu_group_width = HwInfo::max_input;
    const uint32_t alu_groups = HwInfo::mac_count / HwInfo::max_input;
};

typedef OpPattern<torq_hl::AddOp> AddPattern;
typedef OpPattern<torq_hl::AvgPool2DOp> Avgpool2DPattern;
typedef OpPattern<torq_hl::FullyConnectedOp> FCPattern;
typedef OpPattern<torq_hl::SegmentationOp> SegmentationPattern;
typedef OpPattern<torq_hl::TransposeOp> TransposePattern;
typedef OpPattern<torq_hl::DepthwiseConv2DOp> DWPattern;
typedef OpPattern<torq_hl::Conv2DOp> Conv2DPattern;
typedef OpPattern<torq_hl::MaxPool2dOp> MaxPool2dPattern;
typedef OpPattern<torq_hl::MatMulOp> MatMulPattern;
typedef OpPattern<torq_hl::GatherOp> GatherPattern;
typedef OpPattern<torq_hl::IdentityOp> IdentityPattern;
typedef OpPattern<torq_hl::ConvertOp> ConvertPattern;
typedef OpPattern<torq_hl::MulOp> MulPattern;
typedef OpPattern<torq_hl::TableOp> TablePattern;
typedef OpPattern<torq_hl::ArgMaxOp> ArgMaxPattern;
typedef OpPattern<torq_hl::TransposeReshapeOp> TransposeReshapePattern;
typedef OpPattern<torq_hl::Conv1DOp> Conv1DPattern;

typedef OpPattern<torq_hl::FMAOp> FMAPattern;
typedef OpPattern<torq_hl::FillOp> FillPattern;
typedef OpPattern<torq_hl::ReduceOp> ReducePattern;
typedef OpPattern<torq_hl::ScatterOp> ScatterPattern;
typedef OpPattern<torq_hl::ResizeNearestNeighborOp> ResizeNearestNeighborPattern;

typedef OpPattern<torq_hl::ActOp> ActPattern;
typedef OpPattern<torq_hl::BroadcastOp> BroadcastPattern;

typedef OpPattern<torq_hl::ElementWiseBinaryOp> ElementWiseBinaryPattern;
typedef OpPattern<torq_hl::ElementWiseUnaryOp> ElementWiseUnaryPattern;
typedef OpPattern<torq_hl::ElementWiseShiftOp> ElementWiseShiftPattern;

typedef OpPattern<torq_hl::DepthToSpaceOp> DepthToSpacePattern;
typedef OpPattern<torq_hl::ReduceMeanOp> ReduceMeanPattern;
typedef OpPattern<torq_hl::InterleavedInsertOp> InterleavedInsertPattern;

typedef OpPattern<torq_hl::SelectOp> SelectPattern;

void populateNssTaskPatterns(MLIRContext *ctx, RewritePatternSet &patterns);

} // namespace mlir::syna::torq
