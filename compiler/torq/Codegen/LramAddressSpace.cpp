// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Codegen/LramAddressSpace.h"

namespace mlir::syna::torq {

LramAddressSpace::LramAddressSpace(int64_t usable) : free_(allocator_) {
    free_.insert(0, usable, 0);
}

void LramAddressSpace::claim(FreeRanges::iterator &range, int64_t offset, int64_t size) {
    if (range.stop() - range.start() == size) {
        range.erase();
    }
    else if (offset == range.start()) {
        range.setStart(offset + size);
    }
    else {
        range.setStop(offset);
    }
}

int64_t LramAddressSpace::allocate(int64_t size) {
    if (size < largeBufferBytes) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it.stop() - it.start() >= size) {
                int64_t offset = it.start();
                claim(it, offset, size);
                return offset;
            }
        }
    }
    else {
        for (auto it = free_.end(); it != free_.begin();) {
            --it;
            if (it.stop() - it.start() >= size) {
                int64_t offset = it.stop() - size;
                claim(it, offset, size);
                return offset;
            }
        }
    }
    return -1;
}

void LramAddressSpace::release(int64_t offset, int64_t size) {
    free_.insert(offset, offset + size, 0);
}

} // namespace mlir::syna::torq
