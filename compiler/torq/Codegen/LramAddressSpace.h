// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/IntervalMap.h"

#include <cstdint>

namespace mlir::syna::torq {

// A two-ended address space over [0, usable): blocks smaller than largeBufferBytes place
// bottom-up and the rest top-down (the Pool allocator's policy), so the small buffers
// churning between the large ones never fragment the space the large ones need.
class LramAddressSpace {
    static constexpr int64_t largeBufferBytes = 8 * 1024;

    // the free ranges; IntervalMap merges adjacent ones on insert, so release() never has to
    using FreeRanges = llvm::IntervalMap<int64_t, char, 16, llvm::IntervalMapHalfOpenInfo<int64_t>>;
    FreeRanges::Allocator allocator_;
    FreeRanges free_;

    // Remove [offset, offset + size), which lies at one end of `range`, from it.
    static void claim(FreeRanges::iterator &range, int64_t offset, int64_t size);

  public:
    explicit LramAddressSpace(int64_t usable);

    // Returns the placed offset, or -1 if no free range fits `size`.
    int64_t allocate(int64_t size);

    void release(int64_t offset, int64_t size);
};

} // namespace mlir::syna::torq
