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
#include "llvm/Support/CommandLine.h"

#ifndef TORQ_EXPERIMENTAL_LINALG_CONV_TILING
// If != 0 enable experimental linalg conv tiling (1: per channel tiling, 2: y-axis tiling)
#define TORQ_EXPERIMENTAL_LINALG_CONV_TILING 0
#endif

extern llvm::cl::list<std::string> clExecuteOnHost;
extern llvm::cl::list<std::string> clExecuteOnCSS;

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

std::pair<bool, LogicalResult>
setTargetExecutorIfForced(Operation *op, PatternRewriter &rewriter, std::string opName);

// A RAII guard to ensure an Operation is erased when it goes out of scope.
// The `release()` method can be used to explicitly transfer ownership of the
// `Operation` away from the guard, preventing it from being erased upon
// destruction.
class OpEraseGuard {
  public:
    explicit OpEraseGuard(Operation *op = nullptr) : op_(op) {}

    // Destructor: Erase the operation if it still exists.
    ~OpEraseGuard() {
        if (op_) {
            op_->erase();
        }
    }

    // Release the operation from the guard without erasing it.
    Operation *release() {
        Operation *temp = op_;
        op_ = nullptr;
        return temp;
    }

    // Allow implicit conversion to Operation* for convenience
    operator Operation *() const { return op_; }

    OpEraseGuard(const OpEraseGuard &) = delete;
    OpEraseGuard(OpEraseGuard &&other) = delete;
    OpEraseGuard &operator=(const OpEraseGuard &) = delete;
    OpEraseGuard &operator=(OpEraseGuard &&other) = delete;

  private:
    Operation *op_;
};

} // namespace torq

} // namespace mlir::syna
