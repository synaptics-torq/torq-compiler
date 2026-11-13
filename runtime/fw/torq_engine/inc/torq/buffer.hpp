// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

///
/// Torq data buffer.
///

#pragma once

#include <cstddef>
#include <cstring>
#include <utility>

#include "torq/allocator.hpp"

namespace torq {

/// Torq data buffer.
class Buffer {
public:
    /// Create an empty data buffer.
    /// @param allocator: allocator to be used (nullptr selects the platform default)
    /// The allocator must outlive the buffer. Linux NPU builds prefer DMA-BUF;
    /// CPU builds and systems without an available DMA heap use aligned malloc.
    explicit Buffer(Allocator *allocator = nullptr)
        : allocator_(allocator ? allocator : &default_allocator()) {}

    /// Create and allocate a data buffer.
    /// Allocation failure leaves the buffer empty.
    /// @param size: buffer size in bytes
    /// @param allocator: allocator to be used (nullptr selects the platform default)
    explicit Buffer(size_t size, Allocator *allocator = nullptr) : Buffer(allocator) {
        resize(size);
    }

    /// Create a data buffer to refer to an existing memory area.
    /// The user must make sure that the provided memory is correctly aligned and padded.
    /// The specified memory area will *not* be deallocated when the buffer is destroyed.
    /// @param data: pointer to the existing memory area
    /// @param size: size of the actual data
    Buffer(void *data, size_t size) : data_(data), size_(data ? size : 0) {}

    /// Create a data buffer that refers to a part of the memory area of an existing buffer.
    /// The memory of the provided buffer must have already been allocated.
    /// To avoid referring to released memory, the existing buffer memory must *not* be deallocated
    /// before this buffer is destroyed.
    /// @param source: an existing Buffer
    /// @param offset: offset of the desired data inside the Buffer memory area
    /// @param size: size of the desired data
    Buffer(const Buffer &source, size_t offset, size_t size) {
        if (source.data_ && offset <= source.size_ && size <= source.size_ - offset) {
            data_ = static_cast<unsigned char *>(source.data_) + offset;
            size_ = size;
        }
    }

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;

    /// Transfer the allocation or borrowed memory reference from another buffer.
    /// The source becomes empty. Do not move buffers attached to a network.
    /// @param other: buffer whose contents are transferred
    Buffer(Buffer &&other) noexcept { swap(other); }

    /// Release the current allocation and transfer another buffer's contents.
    /// The source becomes empty. Do not move buffers attached to a network.
    /// @param other: buffer whose contents are transferred
    /// @return this buffer
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            Buffer previous(std::move(other));
            swap(previous);
        }
        return *this;
    }

    /// Release owned storage. Borrowed storage is never deallocated.
    ~Buffer() {
        if (data_ && allocator_) allocator_->deallocate(data_);
    }

    /// Resize buffer.
    /// Only possible if an allocator was provided. Any previous content is lost.
    /// Allocation failure preserves the current data and size.
    /// Do not resize a buffer while a network is using it for inference.
    /// Linux permits resizing between predictions; changed storage is rebound
    /// by the predictor before the next invocation.
    ///
    /// @param size: new buffer size
    /// @return true if success
    bool resize(size_t size) {
        if (size == size_) return true;
        if (!allocator_) return false;
        void *replacement = size ? allocator_->allocate(size) : nullptr;
        if (size && !replacement) return false;
        if (data_) allocator_->deallocate(data_);
        data_ = replacement;
        size_ = size;
        return true;
    }

    /// Copy data in buffer.
    /// Always successful if the input data size is the same as current buffer size,
    /// otherwise the buffer is resized if possible.
    ///
    /// @param data: pointer to data to be copied
    /// @param size: size of data to be copied
    /// @return true if success, false for null data, zero size, or allocation failure
    bool assign(const void *data, size_t size) {
        if (!data || !size) return false;
        if (size != size_) {
            if (!allocator_) return false;
            Buffer replacement(allocator_);
            if (!replacement.resize(size)) return false;
            std::memcpy(replacement.data_, data, size);
            swap(replacement);
            return true;
        }
        std::memmove(data_, data, size);
        return true;
    }

    /// Actual data
    void *data() { return data_; }
    /// @copydoc Buffer::data
    const void *data() const { return data_; }
    /// Actual data size
    size_t size() const { return size_; }

    /// Change the allocator.
    /// Can only be done if the buffer is empty.
    /// On Linux, selecting the current allocator is also allowed for a nonempty buffer.
    /// @param allocator: allocator (nullptr selects the platform default)
    /// @return true if success
    bool set_allocator(Allocator *allocator) {
#if TORQ_OS_LINUX
        if (allocator_ == (allocator ? allocator : &default_allocator())) return true;
#endif
        if (data_ || size_) return false;
        allocator_ = allocator ? allocator : &default_allocator();
        return true;
    }

private:
    void swap(Buffer &other) noexcept {
        std::swap(allocator_, other.allocator_);
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }

    Allocator *allocator_ = nullptr;
    void *data_ = nullptr;
    size_t size_ = 0;
};

}  // namespace torq