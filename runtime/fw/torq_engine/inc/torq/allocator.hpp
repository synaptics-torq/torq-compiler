// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

///
/// Torq allocator.
///

#pragma once

#include <cstddef>
#include <cstdint>

namespace torq {

/// Buffer allocator.
/// Allows to allocate aligned memory from different areas.
/// Memory must be allocated such that it completely includes all the cache-lines used for the
/// actual data. This ensures that no cache-line used for data is also used for something else.
class Allocator {
public:
    /// Required memory alignment and padding, in bytes.
    static constexpr size_t alignment = 64;

    virtual ~Allocator() = default;
    /// Allocate memory.
    /// The allocation must include complete cache lines covering the requested data.
    /// @param size: required memory size in bytes
    /// @return aligned memory address, or nullptr if allocation fails
    virtual void *allocate(size_t size) = 0;

    /// Deallocate memory.
    /// @param data: memory address returned by this allocator, or nullptr
    virtual void deallocate(void *data) = 0;

    /// Round a value upward to the specified alignment.
    /// @param val: value to align; adding num - 1 must not overflow
    /// @param num: nonzero power-of-two alignment
    /// @return val rounded upward to alignment num
    static uintptr_t align(uintptr_t val, uint32_t num) {
        return (val + num - 1) & ~(static_cast<uintptr_t>(num) - 1);
    }
};

/// Get the global platform allocator.
/// Linux NPU builds prefer the cached DMA heap and fall back to aligned malloc
/// if the heap cannot be opened. CPU-only Linux builds use aligned malloc.
/// Allocation errors after opening the heap are reported, not silently retried.
/// @return allocator with static lifetime
Allocator &default_allocator();

}
