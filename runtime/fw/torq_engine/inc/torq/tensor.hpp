// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

///
/// Torq data tensor.
///

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "torq/buffer.hpp"
#include "torq/types.hpp"

namespace torq {

/// Storage type of each tensor element; no normalization or quantization is implied.
enum class DataType : uint8_t {
    invalid, int8, uint8, int16, uint16, int32, uint32, int64, uint64,
    float16, float32, float64
};

/// Tensor description obtained from VMFB ABI reflection or supplied by the caller.
/// Layout is not inferred from shape; preprocessing must supply its own layout.
struct TensorAttributes {
    std::string name;
    Shape shape;
    DataType dtype = DataType::invalid;
    Layout layout = Layout::none;

    /// @return element size in bytes, or 0 for an unsupported data type
    size_t item_size() const {
        switch (dtype) {
        case DataType::int8: case DataType::uint8: return sizeof(int8_t);
        case DataType::int16: case DataType::uint16: case DataType::float16: return sizeof(int16_t);
        case DataType::int32: case DataType::uint32: case DataType::float32: return sizeof(int32_t);
        case DataType::int64: case DataType::uint64: case DataType::float64: return sizeof(int64_t);
        default: return 0;
        }
    }

    /// Compute storage size with overflow and zero-dimension checks.
    /// An empty shape describes a rank-zero tensor with one element.
    /// @return tensor size in bytes, or 0 for invalid or overflowing dimensions
    size_t size() const {
        size_t bytes = item_size();
        if (!bytes) return 0;
        for (uint32_t dimension : shape) {
            if (!dimension || dimension > std::numeric_limits<size_t>::max() / bytes) return 0;
            bytes *= static_cast<size_t>(dimension);
        }
        return bytes;
    }
};

/// Torq data tensor.
/// Raw bytes are stored without implicit conversion. Network-owned tensors must
/// not be moved or replaced by applications while the network is in use.
class Tensor {
public:
    /// Create a tensor with an empty, lazily allocated default buffer.
    /// @param attributes: tensor name, shape, storage type, and optional layout
    explicit Tensor(TensorAttributes attributes)
        : attributes_(std::move(attributes)), default_buffer_(std::make_shared<Buffer>()),
          buffer_(default_buffer_) {}

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

    /// Get the name of the tensor.
    /// Can be useful in case of networks with multiple inputs or outputs
    /// to identify a tensor with a string instead of a positional index.
    /// @return tensor name
    const std::string &name() const { return attributes_.name; }
    /// Get the Shape of the Tensor, that is the number of elements in each dimension.
    /// The order of the dimensions is specified by the tensor layout.
    /// @return tensor shape
    const Shape &shape() const { return attributes_.shape; }
    /// Get the layout of the Tensor, that is how data are organized in memory.
    /// TORQ supports two layouts: `NCHW` and `NHWC`.
    /// The *N* dimension (number of samples) is present for
    /// compatibility with standard conventions but must always be one.
    /// @return tensor layout
    Layout layout() const { return attributes_.layout; }
    /// Get the Dimensions of the Tensor, that is the number of elements in each dimension.
    /// The returned values are independent of the tensor layout.
    /// @return tensor dimensions (all 0s if the rank of the tensor is not 4)
    Dimensions dimensions() const { return Dimensions(shape(), layout()); }
    /// Get tensor data type.
    /// @return the type of each item in the tensor.
    DataType data_type() const { return attributes_.dtype; }
    /// @return size *in bytes* of the tensor data
    size_t size() const { return attributes_.size(); }
    /// Get the number of items in the tensor.
    /// A tensor `size()` is always equal to `item_count()` multiplied by the size of
    /// the tensor data type.
    /// @return number of data items in the tensor
    size_t item_count() const { return attributes_.item_size() ? size() / attributes_.item_size() : 0; }
    /// @return true if this is a scalar tensor, that is it contains a single element.
    /// (the shape of a scalar tensor has one dimension, equal to 1)
    bool is_scalar() const { return shape().size() == 1 && shape()[0] == 1; }
    /// Get pointer to the tensor's current data Buffer if any.
    /// This will be the default buffer of the tensor unless the user has assigned a different
    /// buffer using set_buffer()
    /// @return current data buffer, or nullptr if none
    Buffer *buffer() { return buffer_.get(); }
    const Buffer *buffer() const { return buffer_.get(); }

    /// Get a pointer to the beginning of the data inside the tensor data buffer, if any.
    /// The method returns a `void` pointer since the actual data type is what returned by
    /// the `data_type()` method.
    /// Empty storage is allocated and zero-initialized by the non-const overload.
    /// The const overload never allocates storage.
    /// @return pointer to the raw data inside the data buffer, nullptr if none.
    void *data() {
        if (!size() || !buffer_) return nullptr;
        if (!buffer_->size()) {
            if (!buffer_->resize(size())) return nullptr;
            std::memset(buffer_->data(), 0, size());
        }
        return buffer_->size() == size() ? buffer_->data() : nullptr;
    }

    /// @copydoc Tensor::data
    const void *data() const {
        return buffer_ && buffer_->size() == size() ? buffer_->data() : nullptr;
    }

    /// Copy raw data to the tensor data buffer.
    /// The data is considered as raw data so no normalization or conversion is done whatever
    /// the actual data-type of the tensor. The data size must be equal to the `size()`
    /// of the tensor.
    /// @param source: pointer to data to be copied
    /// @param bytes: size of data to be copied
    /// @return true if success
    bool assign(const void *source, size_t bytes) {
        if (!source || bytes != size()) return false;
        void *destination = data();
        if (!destination) return false;
        std::memmove(destination, source, bytes);
        return true;
    }

    /// Copy the content of a tensor to the tensor data buffer.
    /// No normalization or conversion is done, the data type and size of the two tensors must match
    /// @param source: source tensor containing the data to be copied.
    /// @return true if success, false if type or size mismatch
    bool assign(const Tensor &source) {
        return data_type() == source.data_type() && assign(source.data(), source.size());
    }

    /// Set the tensor's current data buffer.
    /// The buffer size must be 0 or match the tensor size otherwise it will be rejected
    /// (empty buffers are automatically resized to the the tensor size).
    /// The tensor retains shared ownership of the Buffer object.
    /// Borrowed memory and custom allocators must outlive every retained reference.
    /// Empty buffers are allocated and zero-initialized. Rejected buffers do not
    /// replace the current one. Linux HAL attachment is deferred until predict().
    /// Passing nullptr restores allocated default storage rather than detaching
    /// the tensor. A successful call does not guarantee that HAL attachment succeeds.
    /// @param buffer: buffer to be used for this tensor (nullptr restores the default buffer).
    /// @return true if success
    bool set_buffer(std::shared_ptr<Buffer> buffer) {
        if (!buffer) buffer = default_buffer_;
        if (!size() || (buffer->size() && buffer->size() != size())) return false;
        if (!buffer->size()) {
            if (!buffer->resize(size())) return false;
            std::memset(buffer->data(), 0, size());
        }
        buffer_ = std::move(buffer);
        return true;
    }

private:
    friend class PredictorTORQ;
    TensorAttributes attributes_;
    std::shared_ptr<Buffer> default_buffer_;
    std::shared_ptr<Buffer> buffer_;
};

/// Tensor collection
class Tensors {
public:
    /// @return number of tensors in the collection
    size_t size() const { return tensors_.size(); }
    /// @return true if the collection contains no tensors
    bool empty() const { return tensors_.empty(); }
    /// Access a tensor by its positional index.
    /// @param index: tensor index
    /// @return tensor at the specified index
    /// @pre index must be less than size().
    Tensor &operator[](size_t index) { return tensors_[index]; }
    /// @copydoc Tensors::operator[]
    const Tensor &operator[](size_t index) const { return tensors_[index]; }
    /// @return iterator to the first tensor
    auto begin() { return tensors_.begin(); }
    /// @return iterator past the last tensor
    auto end() { return tensors_.end(); }
    /// @copydoc Tensors::begin
    auto begin() const { return tensors_.begin(); }
    /// @copydoc Tensors::end
    auto end() const { return tensors_.end(); }

private:
    friend class Network;
    std::vector<Tensor> tensors_;
};

}  // namespace torq