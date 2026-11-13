// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "allocator_private.hpp"

#include <cstdint>
#include <limits>

#include <zephyr/dt-bindings/memory-attr/memory-attr-sr2xx.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/mem_mgmt/mem_attr_heap.h>
#include <zephyr/sys/util.h>

LOG_MODULE_REGISTER(torq_allocator, LOG_LEVEL_INF);

namespace torq {
namespace {

class PlatformAllocator : public Allocator {
public:
    /// @copydoc Allocator::allocate
    void *allocate(size_t size) override {
        if (!size || size > std::numeric_limits<size_t>::max() - (buffer_alignment - 1)) return nullptr;
        const size_t aligned_size = ROUND_UP(size, buffer_alignment);
        void *data = mem_attr_heap_aligned_alloc(DT_MEM_SW_APSRAM | DT_MEM_SW_CACHED,
            buffer_alignment, aligned_size);
        if (!data) {
            LOG_ERR("%s allocation failed: requested=%zu aligned=%zu",
                "cached APSRAM", size, aligned_size);
        }
        return data;
    }

    /// @copydoc Allocator::deallocate
    void deallocate(void *data) override {
        if (!data) return;
        mem_attr_heap_free(data);
    }
};

}

Allocator &default_allocator()
{
    static PlatformAllocator allocator;
    return allocator;
}

}