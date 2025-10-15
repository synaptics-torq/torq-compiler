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

#ifndef TORQ_EXPERIMENTAL_LINALG_CONV_TILING
// If != 0 enable experimental linalg conv tiling (1: per channel tiling, 2: y-axis tiling)
#define TORQ_EXPERIMENTAL_LINALG_CONV_TILING 0
#endif

namespace mlir::syna {

void printOperation(Operation *op);
void printRegion(Region &region);
void printBlock(Block &block);

namespace torq {

/// @return dividend / divisor rounded upward
constexpr uint32_t div_ceil(uint32_t dividend, uint32_t divisor) {
    return (dividend + divisor - 1) / divisor;
}

/// @return dividend / divisor rounded downward
constexpr uint32_t div_floor(uint32_t dividend, uint32_t divisor) { return dividend / divisor; }

/// @return val rounded upward to alignment a
constexpr uint32_t align_ceil(uint32_t val, uint32_t a) { return ((val + a - 1) / a) * a; }

/// @return val rounded downward to alignment a
constexpr uint32_t align_floor(uint32_t val, uint32_t a) { return (val / a) * a; }

template <typename IntegerT> IntegerT doubleToInt(double value) {
    constexpr int intMin = std::numeric_limits<IntegerT>::min();
    constexpr int intMax = std::numeric_limits<IntegerT>::max();
    assert(!std::isnan(value) && !std::isinf(value));
    assert(value >= intMin);
    assert(value <= intMax);
    return static_cast<IntegerT>(value);
}

/// @return a multiple of @p value which is less than @p max_range.
constexpr size_t findExactMultiple(size_t value, size_t max_range) {
    for (size_t i = max_range; i > 1; --i) {
        if (value % i == 0)
            return i;
    }
    return 1;
}

std::vector<uint32_t> prepareWeightDims(
    std::vector<uint32_t> weight_shapes, size_t alu_groups_check, size_t alu_groups,
    uint32_t wram_seg_width = HwInfo::wram_seg_width
);

} // namespace torq

} // namespace mlir::syna
