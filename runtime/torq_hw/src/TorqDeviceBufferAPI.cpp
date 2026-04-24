// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqDeviceBufferAPI.h"

#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <mutex>

#include "iree/base/internal/atomics.h"

#ifdef ENABLE_ASTRA_MACHINA
#include <fcntl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif // ENABLE_ASTRA_MACHINA

namespace {

constexpr size_t kHeapAlignment = 64;

#ifdef ENABLE_ASTRA_MACHINA
constexpr size_t kPageSize = 4096;

typedef struct torq_hw_dma_heap_node_state_t {
    const char *path;
    int fd;
    size_t ref_count;
} torq_hw_dma_heap_node_state_t;

std::mutex g_dma_heap_node_mutex;
torq_hw_dma_heap_node_state_t g_cached_dma_heap_dev_node = {
    TORQ_HW_DMA_HEAP_NODE_CACHED, TORQ_HW_INVALID_FD, 0,
};
torq_hw_dma_heap_node_state_t g_uncached_dma_heap_dev_node = {
    TORQ_HW_DMA_HEAP_NODE_UNCACHED, TORQ_HW_INVALID_FD, 0,
};

static torq_hw_dma_heap_node_state_t *
torq_hw_dma_heap_node_state_for_mode(torq_hw_device_buffer_mode_t mode) {
    switch (mode) {
    case TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP:
        return &g_cached_dma_heap_dev_node;
    case TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP_UNCACHED:
        return &g_uncached_dma_heap_dev_node;
    default:
        return nullptr;
    }
}

static iree_status_t torq_hw_dma_heap_node_open_locked(torq_hw_dma_heap_node_state_t *node_state) {
    if (node_state->fd != TORQ_HW_INVALID_FD) {
        return iree_ok_status();
    }
    const int dev_node = ::open(node_state->path, O_RDWR);
    if (dev_node == TORQ_HW_INVALID_FD) {
        const int err = errno;
        return iree_make_status(
            iree_status_code_from_errno(err), "failed to open dma heap node %s", node_state->path
        );
    }
    node_state->fd = dev_node;
    return iree_ok_status();
}

static iree_status_t
torq_hw_dma_heap_node_dup(torq_hw_device_buffer_mode_t mode, int *out_dup_fd) {
    if (!out_dup_fd) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_dup_fd must be non-null");
    }

    std::lock_guard<std::mutex> lock(g_dma_heap_node_mutex);
    torq_hw_dma_heap_node_state_t *node_state = torq_hw_dma_heap_node_state_for_mode(mode);
    if (!node_state || node_state->fd == TORQ_HW_INVALID_FD) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION, "dma heap node for mode %d has not been acquired",
            (int)mode
        );
    }

    const int dup_fd = ::dup(node_state->fd);
    if (dup_fd == TORQ_HW_INVALID_FD) {
        const int err = errno;
        return iree_make_status(iree_status_code_from_errno(err), "failed to dup dma heap node fd");
    }
    *out_dup_fd = dup_fd;
    return iree_ok_status();
}

static iree_status_t
torq_hw_try_alloc_dma_heap(
    torq_hw_device_buffer_mode_t mode, size_t requested_size, torq_hw_device_buffer_t *out_buffer
) {
    size_t alloc_size = requested_size;
    // clamping to 1 since that seems to be common practice in other IREE allocators
    if (alloc_size == 0)
        alloc_size = 1;
    const size_t aligned_size = (alloc_size + kPageSize - 1) & ~(kPageSize - 1);

    int dev_node = TORQ_HW_INVALID_FD;
    iree_status_t status = torq_hw_dma_heap_node_dup(mode, &dev_node);
    if (!iree_status_is_ok(status)) {
        return status;
    }

    struct dma_heap_allocation_data req = {0};
    req.len = aligned_size;
    req.fd_flags = O_RDWR;
    if (::ioctl(dev_node, DMA_HEAP_IOCTL_ALLOC, &req) == TORQ_HW_INVALID_FD) {
        const int err = errno;
        ::close(dev_node);
        return iree_make_status(iree_status_code_from_errno(err), "DMA_HEAP_IOCTL_ALLOC failed");
    }
    ::close(dev_node);

    const int dmabuf_fd = req.fd;
    if (dmabuf_fd == TORQ_HW_INVALID_FD) {
        return iree_make_status(IREE_STATUS_INTERNAL, "dma heap returned invalid dmabuf fd");
    }
    void *mapped = ::mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    if (mapped == MAP_FAILED) {
        const int err = errno;
        ::close(dmabuf_fd);
        return iree_make_status(
            iree_status_code_from_errno(err), "failed to mmap dma heap allocation"
        );
    }

    out_buffer->handle = dmabuf_fd;
    out_buffer->mapped = mapped;
    out_buffer->requestedSize = requested_size;
    out_buffer->allocatedSize = aligned_size;
    out_buffer->mode = mode;
    out_buffer->hostCached = mode == TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP;
    return iree_ok_status();
}

#endif // ENABLE_ASTRA_MACHINA

static iree_status_t
torq_hw_alloc_malloc(size_t requested_size, torq_hw_device_buffer_t *out_buffer) {
    size_t alloc_size = requested_size;
    if (alloc_size == 0) {
        alloc_size = 1;
    }
    const size_t aligned_size = (alloc_size + kHeapAlignment - 1) & ~(kHeapAlignment - 1);

    void *ptr = nullptr;
    if (posix_memalign(&ptr, kHeapAlignment, aligned_size) != 0 || !ptr) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED, "failed to allocate aligned host buffer"
        );
    }

    out_buffer->handle = TORQ_HW_INVALID_FD;
    out_buffer->mapped = ptr;
    out_buffer->requestedSize = requested_size;
    out_buffer->allocatedSize = aligned_size;
    out_buffer->mode = TORQ_HW_DEVICE_BUFFER_MODE_MALLOC;
    out_buffer->hostCached = true;
    return iree_ok_status();
}

} // namespace

extern "C" {

iree_status_t torq_hw_dma_heap_node_acquire(void) {
#ifdef ENABLE_ASTRA_MACHINA
    std::lock_guard<std::mutex> lock(g_dma_heap_node_mutex);
    iree_status_t status;
    IREE_RETURN_IF_ERROR(torq_hw_dma_heap_node_open_locked(&g_cached_dma_heap_dev_node));
    ++g_cached_dma_heap_dev_node.ref_count;
    status = torq_hw_dma_heap_node_open_locked(&g_uncached_dma_heap_dev_node);
    if (!iree_status_is_ok(status)) {
        --g_cached_dma_heap_dev_node.ref_count;
        if (g_cached_dma_heap_dev_node.ref_count == 0 &&
            g_cached_dma_heap_dev_node.fd != TORQ_HW_INVALID_FD) {
            (void)::close(g_cached_dma_heap_dev_node.fd);
            g_cached_dma_heap_dev_node.fd = TORQ_HW_INVALID_FD;
        }
        return status;
    }
    ++g_uncached_dma_heap_dev_node.ref_count;
#endif // ENABLE_ASTRA_MACHINA
    return iree_ok_status();
}

void torq_hw_dma_heap_node_release(void) {
#ifdef ENABLE_ASTRA_MACHINA
    std::lock_guard<std::mutex> lock(g_dma_heap_node_mutex);
    if (g_uncached_dma_heap_dev_node.ref_count) {
        --g_uncached_dma_heap_dev_node.ref_count;
        if (g_uncached_dma_heap_dev_node.ref_count == 0 &&
            g_uncached_dma_heap_dev_node.fd != TORQ_HW_INVALID_FD) {
            (void)::close(g_uncached_dma_heap_dev_node.fd);
            g_uncached_dma_heap_dev_node.fd = TORQ_HW_INVALID_FD;
        }
    }
    if (g_cached_dma_heap_dev_node.ref_count) {
        --g_cached_dma_heap_dev_node.ref_count;
        if (g_cached_dma_heap_dev_node.ref_count == 0 &&
            g_cached_dma_heap_dev_node.fd != TORQ_HW_INVALID_FD) {
            (void)::close(g_cached_dma_heap_dev_node.fd);
            g_cached_dma_heap_dev_node.fd = TORQ_HW_INVALID_FD;
        }
    }
#endif // ENABLE_ASTRA_MACHINA
}

iree_status_t torq_hw_device_buffer_allocate(
    torq_hw_device_buffer_mode_t mode, size_t size, torq_hw_device_buffer_t *out_buffer
) {
    if (!out_buffer) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_buffer must be non-null");
    }
    memset(out_buffer, 0, sizeof(*out_buffer));
    out_buffer->handle = TORQ_HW_INVALID_FD;

    iree_status_t status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid device buffer mode %d", (int)mode);

    switch (mode) {
    case TORQ_HW_DEVICE_BUFFER_MODE_MALLOC:
        status = torq_hw_alloc_malloc(size, out_buffer);
        break;
    case TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP:
    case TORQ_HW_DEVICE_BUFFER_MODE_DMA_HEAP_UNCACHED:
#ifdef ENABLE_ASTRA_MACHINA
        status = torq_hw_try_alloc_dma_heap(mode, size, out_buffer);
#else
        status = iree_make_status(
            IREE_STATUS_UNAVAILABLE, "DMA heap mode not available without ENABLE_ASTRA_MACHINA"
        );
#endif // ENABLE_ASTRA_MACHINA
        break;
    default:
        break;
    }
    return status;
}

static iree_status_t torq_hw_device_buffer_validate_range(
    const torq_hw_device_buffer_t *buffer, size_t offset, size_t length
) {
    if (!buffer || !buffer->mapped) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "buffer must be mapped");
    }
    if (offset > buffer->requestedSize || length > buffer->requestedSize - offset) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "buffer range is out of bounds");
    }
    return iree_ok_status();
}

#ifdef ENABLE_ASTRA_MACHINA
static iree_status_t torq_hw_device_buffer_sync(torq_hw_device_buffer_t *buffer, uint64_t flags) {
    struct dma_buf_sync sync = {};
    sync.flags = flags;
    if (::ioctl(buffer->handle, DMA_BUF_IOCTL_SYNC, &sync) != 0) {
        const int err = errno;
        return iree_make_status(iree_status_code_from_errno(err), "DMA_BUF_IOCTL_SYNC failed");
    }
    return iree_ok_status();
}
#endif // ENABLE_ASTRA_MACHINA

iree_status_t torq_hw_device_buffer_invalidate_range(
    torq_hw_device_buffer_t *buffer, size_t offset, size_t length
) {
    IREE_RETURN_IF_ERROR(torq_hw_device_buffer_validate_range(buffer, offset, length));
    if (buffer->handle == TORQ_HW_INVALID_FD || !buffer->hostCached) {
        iree_atomic_thread_fence(iree_memory_order_acquire);
        return iree_ok_status();
    }
#ifdef ENABLE_ASTRA_MACHINA
    IREE_RETURN_IF_ERROR(torq_hw_device_buffer_sync(buffer, DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ));
    IREE_RETURN_IF_ERROR(torq_hw_device_buffer_sync(buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ));
#endif // ENABLE_ASTRA_MACHINA
    iree_atomic_thread_fence(iree_memory_order_acquire);
    return iree_ok_status();
}

iree_status_t torq_hw_device_buffer_flush_range(
    torq_hw_device_buffer_t *buffer, size_t offset, size_t length
) {
    IREE_RETURN_IF_ERROR(torq_hw_device_buffer_validate_range(buffer, offset, length));
    iree_atomic_thread_fence(iree_memory_order_release);
    if (buffer->handle == TORQ_HW_INVALID_FD || !buffer->hostCached) {
        return iree_ok_status();
    }
#ifdef ENABLE_ASTRA_MACHINA
    IREE_RETURN_IF_ERROR(
        torq_hw_device_buffer_sync(buffer, DMA_BUF_SYNC_START | DMA_BUF_SYNC_WRITE)
    );
    IREE_RETURN_IF_ERROR(torq_hw_device_buffer_sync(buffer, DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE));
#endif // ENABLE_ASTRA_MACHINA
    return iree_ok_status();
}

iree_status_t torq_hw_device_buffer_free(torq_hw_device_buffer_t *buffer) {
    if (!buffer) {
        return iree_ok_status();
    }
    iree_status_t status = iree_ok_status();
    if (!buffer->mapped) {
        *buffer = {};
        buffer->handle = TORQ_HW_INVALID_FD;
        return status;
    }
#ifdef ENABLE_ASTRA_MACHINA
    if (buffer->handle != TORQ_HW_INVALID_FD) {
        if (buffer->mapped != MAP_FAILED) {
            if (::munmap(buffer->mapped, buffer->allocatedSize) != 0) {
                const int err = errno;
                status = iree_status_join(
                    status,
                    iree_make_status(
                        iree_status_code_from_errno(err), "munmap failed for fd %d", buffer->handle
                    )
                );
            }
        }
        if (::close(buffer->handle) != 0) {
            const int err = errno;
            status = iree_status_join(
                status,
                iree_make_status(
                    iree_status_code_from_errno(err), "close failed for fd %d", buffer->handle
                )
            );
        }
    }
    else {
        free(buffer->mapped);
    }
#else
    free(buffer->mapped);
#endif // ENABLE_ASTRA_MACHINA
    *buffer = {};
    buffer->handle = TORQ_HW_INVALID_FD;
    return status;
}

} // extern "C"
