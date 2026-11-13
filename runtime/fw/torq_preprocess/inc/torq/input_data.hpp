// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_INPUT_DATA_HPP_
#define TORQ_INPUT_DATA_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "torq/types.hpp"

namespace torq {

/// Input representation: prepared tensor bytes or an unsigned 8-bit image.
enum class InputType : uint8_t { invalid, raw, image_8bits };

/// Input bytes and optional image attributes, with owned or borrowed storage.
class InputData {
public:
    /// Create empty input data.
    InputData() = default;
    /// Refer to caller-owned bytes, which must remain valid while used.
    /// @param data: input byte address
    /// @param size: input size in bytes
    /// @param type: input representation
    /// @param shape: image extents, required for image input
    /// @param layout: image dimension order
    InputData(const void *data, size_t size, InputType type = InputType::raw,
              Shape shape = {}, Layout layout = Layout::none)
        : data_(data), size_(size), type_(type), shape_(std::move(shape)), layout_(layout) {}
    /// Take ownership of input bytes.
    /// @param data: byte vector whose storage is transferred
    /// @param type: input representation
    /// @param shape: image extents, required for image input
    /// @param layout: image dimension order
    InputData(std::vector<uint8_t> &&data, InputType type = InputType::raw,
              Shape shape = {}, Layout layout = Layout::none)
        : storage_(std::move(data)), size_(storage_.size()), type_(type),
          shape_(std::move(shape)), layout_(layout) {}

    InputData(const InputData &) = delete;
    InputData &operator=(const InputData &) = delete;
    InputData(InputData &&) noexcept = default;
    InputData &operator=(InputData &&) noexcept = default;

    /// @return true if no bytes are available
    bool empty() const { return !data() || !size_; }
    /// @return address of the owned or borrowed input bytes
    const void *data() const { return storage_.empty() ? data_ : storage_.data(); }
    /// @return input size in bytes
    size_t size() const { return size_; }
    /// @return input representation
    InputType type() const { return type_; }
    /// @return image extents in layout order
    const Shape &shape() const { return shape_; }
    /// @return image dimension order
    Layout layout() const { return layout_; }
    /// @return named image extents, or zero extents for unsupported shapes/layouts
    Dimensions dimensions() const { return Dimensions(shape_, layout_); }

private:
    std::vector<uint8_t> storage_;
    const void *data_ = nullptr;
    size_t size_ = 0;
    InputType type_ = InputType::invalid;
    Shape shape_;
    Layout layout_ = Layout::none;
};

}

#endif