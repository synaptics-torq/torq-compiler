// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include "torq/allocator.hpp"
#include "allocator_linux.hpp"
#include "logging_linux.hpp"
#include "driver/torq_allocator.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <linux/dma-buf.h>
#include <map>
#include <mutex>
#include <sys/ioctl.h>

#ifndef TORQ_LINUX_DEVICE
#define TORQ_LINUX_DEVICE "torq"
#endif

namespace torq {
namespace {

/// Allocate cache-line-aligned host storage for CPU use and DMA-heap fallback.
class MallocAllocator final : public Allocator {
public:
    /// @copydoc Allocator::allocate
    void *allocate(size_t size) override {
        if (!size || size > std::numeric_limits<size_t>::max() - (alignment - 1)) return nullptr;
        void* ptr = std::aligned_alloc(64, Allocator::align(size, 64));
        return ptr;
    }

    /// @copydoc Allocator::deallocate
    void deallocate(void *data) override { std::free(data); }
};

class HalAllocator final : public Allocator {
public:
    HalAllocator() {
        iree_status_t status = iree_hal_torq_allocator_create(
            iree_make_cstring_view("torq-engine"), iree_allocator_system(), &allocator_);
        if (!iree_status_is_ok(status)) {
            LOGW << "TORQ HAL allocator unavailable, using malloc: "
                 << iree_status_code_string(iree_status_code(status));
        }
        iree_status_ignore(status);
    }

    ~HalAllocator() override { iree_hal_allocator_release(allocator_); }

    bool available() const { return allocator_ != nullptr; }

    /// @copydoc Allocator::allocate
    void *allocate(size_t size) override {
        if (!size || !available() || size > std::numeric_limits<size_t>::max() - (4096 - 1))
            return nullptr;
        iree_hal_buffer_params_t parameters = {};
        parameters.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
        parameters.access = IREE_HAL_MEMORY_ACCESS_ALL;
        parameters.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                           IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
        parameters.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
        iree_hal_buffer_t *buffer = nullptr;
        iree_status_t status = iree_hal_allocator_allocate_buffer(allocator_, parameters, size, &buffer);
        if (!iree_status_is_ok(status)) {
            LOGE << "TORQ HAL allocation failed: "
                 << iree_status_code_string(iree_status_code(status));
            iree_status_ignore(status);
            return nullptr;
        }
        iree_hal_buffer_mapping_t mapping = {};
        status = iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, 0, size, &mapping);
        if (!iree_status_is_ok(status)) {
            LOGE << "TORQ HAL mapping failed: " << iree_status_code_string(iree_status_code(status));
            iree_status_ignore(status);
            iree_hal_buffer_release(buffer);
            return nullptr;
        }
        const auto *device_buffer = iree_hal_torq_buffer_device_buffer(buffer);
        const int fd = device_buffer ? device_buffer->handle : -1;
        const int sync_handle = device_buffer && device_buffer->hostCached ? fd : -1;
        void *address = mapping.contents.data;
        if (!sync_fd(sync_handle, DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW)) {
            iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
            iree_hal_buffer_release(buffer);
            return nullptr;
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            allocations_.emplace(address, Allocation{buffer, mapping, fd, sync_handle, true, next_generation_++});
        } catch (...) {
            sync_fd(sync_handle, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
            iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
            iree_hal_buffer_release(buffer);
            return nullptr;
        }
        LOGI << "AllocatorHAL: size=" << size << " fd=" << fd << " address=" << address;
        return address;
    }

    /// Release a DMA-BUF mapping and descriptor owned by this allocator.
    /// @param address: allocation base address, or nullptr
    void deallocate(void *address) override {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = allocations_.find(address);
        if (found == allocations_.end()) return;
        auto &allocation = found->second;
        if (allocation.cpu_access) sync_fd(allocation.sync_handle, DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW);
        iree_status_ignore(iree_hal_buffer_unmap_range(&allocation.mapping));
        iree_hal_buffer_release(allocation.buffer);
        allocations_.erase(found);
    }

    iree_hal_buffer_t *retain_buffer(const void *address) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = allocations_.find(address);
        if (found == allocations_.end()) return nullptr;
        iree_hal_buffer_retain(found->second.buffer);
        return found->second.buffer;
    }

    int fd(const void *address) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = allocations_.find(address);
        return found == allocations_.end() ? -1 : found->second.fd;
    }

    uint64_t generation(const void *address) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = allocations_.find(address);
        return found == allocations_.end() ? 0 : found->second.generation;
    }

    bool cpu_access(const void *address, bool begin) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = allocations_.find(address);
        if (found == allocations_.end() || found->second.cpu_access == begin) return true;
        if (!sync_fd(found->second.sync_handle,
                (begin ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END) | DMA_BUF_SYNC_RW)) return false;
        found->second.cpu_access = begin;
        return true;
    }

private:
    struct Allocation {
        iree_hal_buffer_t *buffer;
        iree_hal_buffer_mapping_t mapping;
        int fd;
        int sync_handle;
        bool cpu_access;
        uint64_t generation;
    };

    static bool sync_fd(int fd, uint64_t flags) {
        if (fd < 0) return true;
        dma_buf_sync sync = {};
        sync.flags = flags;
        int result;
        do { result = ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync); }
        while (result < 0 && errno == EINTR);
        if (result < 0) LOGE << "DMA_BUF_IOCTL_SYNC failed for fd=" << fd << ": " << std::strerror(errno);
        return result == 0;
    }

    iree_hal_allocator_t *allocator_ = nullptr;
    uint64_t next_generation_ = 1;
    std::mutex mutex_;
    std::map<const void *, Allocation> allocations_;
};

HalAllocator &hal_allocator() {
    static HalAllocator allocator;
    return allocator;
}

}

Allocator &default_allocator()
{
    if (std::strcmp(TORQ_LINUX_DEVICE, "torq") == 0 && hal_allocator().available())
        return hal_allocator();
    static MallocAllocator allocator;
    return allocator;
}

int linux_buffer_fd(const void *address)
{
    return std::strcmp(TORQ_LINUX_DEVICE, "torq") == 0 ? hal_allocator().fd(address) : -1;
}

iree_hal_buffer_t *linux_buffer_retain_hal_buffer(const void *address)
{
    return std::strcmp(TORQ_LINUX_DEVICE, "torq") == 0 ? hal_allocator().retain_buffer(address) : nullptr;
}

uint64_t linux_buffer_generation(const void *address)
{
    return std::strcmp(TORQ_LINUX_DEVICE, "torq") == 0 ? hal_allocator().generation(address) : 0;
}

bool linux_buffer_begin_cpu_access(const void *address)
{
    return std::strcmp(TORQ_LINUX_DEVICE, "torq") != 0 || hal_allocator().cpu_access(address, true);
}

bool linux_buffer_end_cpu_access(const void *address)
{
    return std::strcmp(TORQ_LINUX_DEVICE, "torq") != 0 || hal_allocator().cpu_access(address, false);
}

}