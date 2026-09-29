// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#ifndef TORQ_EXPERIMENTAL_LINALG_CONV_TILING
// If != 0 enable experimental linalg conv tiling (1: per channel tiling, 2: y-axis tiling)
#define TORQ_EXPERIMENTAL_LINALG_CONV_TILING 0
#endif

extern llvm::cl::list<std::string> clExecuteOnHost;
extern llvm::cl::list<std::string> clExecuteOnCSS;

namespace mlir::syna {

namespace torq {

/// Per-output-channel fp32 bias folded into linalg.reduce_sum (OptimizeConv1DPattern).
inline constexpr llvm::StringLiteral kReducePerChannelBiasAttr{"torq.reduce_per_channel_bias"};

/// @return dividend / divisor rounded upward
constexpr uint32_t div_ceil(uint32_t dividend, uint32_t divisor) {
    return (dividend + divisor - 1) / divisor;
}

/// @return val rounded upward to alignment a
constexpr uint32_t align_ceil(uint32_t val, uint32_t a) { return ((val + a - 1) / a) * a; }

// C++20 std::midpoint: computes average without overflow
inline int64_t midpoint(int64_t min, int64_t max) { return min + ((max - min) / 2); }

template <typename IntegerT> IntegerT doubleToInt(double value) {
    constexpr int intMin = std::numeric_limits<IntegerT>::min();
    constexpr int intMax = std::numeric_limits<IntegerT>::max();
    assert(!std::isnan(value) && !std::isinf(value));
    assert(value >= intMin);
    assert(value <= intMax);
    return static_cast<IntegerT>(value);
}

std::vector<uint32_t> prepareWeightDims(
    std::vector<uint32_t> weight_shapes, size_t alu_groups_check, size_t alu_groups,
    uint32_t wram_seg_width = HwInfo::wram_seg_width
);

std::pair<bool, LogicalResult>
setTargetExecutorIfForced(Operation *op, PatternRewriter &rewriter, std::string opName);

} // namespace torq

} // namespace mlir::syna
