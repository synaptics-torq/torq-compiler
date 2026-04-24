// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/base/api.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TORQ_HW_INVALID_FD (-1)

#define TORQ_HW_DMA_HEAP_NODE_CACHED "/dev/dma_heap/system"
#define TORQ_HW_DMA_HEAP_NODE_UNCACHED "/dev/dma_heap/system-cust-uncached"
#define DMABUF_NODE TORQ_HW_DMA_HEAP_NODE_CACHED
#define DMABUF_NODE_UNCACHED TORQ_HW_DMA_HEAP_NODE_UNCACHED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef enum torq_hw_device_buffer_mode_e {
    TORQ_HW_DEVICE_BUFFER_MODE_MALLOC = 1,
    TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP = 2,
    TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP_UNCACHED = 3,
} torq_hw_device_buffer_mode_t;

typedef struct torq_hw_device_buffer_t {
    int handle;           // Backend-specific handle; TORQ_HW_INVALID_FD when unused
    void *mapped;         // Userspace-mapped pointer
    size_t requestedSize; // Size originally requested by the caller
    size_t allocatedSize; // Actual allocation size (may be rounded up for alignment)
    size_t dataOffset;    // Start of payload within allocation
    torq_hw_device_buffer_mode_t mode;
    bool hostCached;
} torq_hw_device_buffer_t;

// Allocates a device buffer according to ``mode`` and maps it in userspace.
iree_status_t torq_hw_device_buffer_allocate(
    torq_hw_device_buffer_mode_t mode, size_t size, torq_hw_device_buffer_t *out_buffer
);

// Frees a previously allocated device buffer.
// Returns a non-OK status if backend cleanup fails.
iree_status_t torq_hw_device_buffer_free(torq_hw_device_buffer_t *buffer);

// Retains a process-global DMA heap device node handle.
// The first successful retain opens `DMABUF_NODE`; subsequent retains share it.
// This is a no-op on non-Astra builds.
iree_status_t torq_hw_dma_heap_node_acquire(void);

// Releases a previously retained process-global DMA heap device node handle.
// The node is closed when the last retain is released.
// This is a no-op on non-Astra builds.
void torq_hw_dma_heap_node_release(void);

// Invalidates a host-visible device buffer range before CPU reads.
// Buffers backed by malloc or uncached DMA heaps complete as no-ops.
iree_status_t torq_hw_device_buffer_invalidate_range(
    torq_hw_device_buffer_t *buffer, size_t offset, size_t length
);

// Flushes a host-visible device buffer range after CPU writes.
// Buffers backed by malloc or uncached DMA heaps complete as no-ops.
iree_status_t torq_hw_device_buffer_flush_range(
    torq_hw_device_buffer_t *buffer, size_t offset, size_t length
);

static inline void *torq_hw_device_buffer_data(const torq_hw_device_buffer_t *buffer) {
    return buffer ? (void *)((uint8_t *)buffer->mapped + buffer->dataOffset) : NULL;
}

static inline bool torq_hw_device_buffer_is_dmabuf(const torq_hw_device_buffer_t *buffer) {
    return buffer && buffer->handle != TORQ_HW_INVALID_FD;
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
