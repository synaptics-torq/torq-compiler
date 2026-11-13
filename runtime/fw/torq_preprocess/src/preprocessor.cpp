// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    preprocessor.cpp
 * \brief   Image preprocessor implementation for int8 models.
 * \details Converts uint8 RGB image data to int8 in-place by XOR with 0x80.
 *          Uses Arm M-Profile Vector Extension (MVE) when available, or
 *          falls back to a 32-bit word-level XOR loop.
 */

#include "torq/preprocessor.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#ifdef __ARM_FEATURE_MVE
#include <arm_mve.h>
#endif

namespace torq {

static constexpr uint8_t  kUint8ToInt8Offset = 0x80u;

static void convert_to_int8(void *data, size_t total)
{
    auto *bytes = static_cast<uint8_t *>(data);

#ifdef __ARM_FEATURE_MVE
    constexpr size_t kMveLanes = 16;
    const uint8x16_t mask = vdupq_n_u8(kUint8ToInt8Offset);
    size_t remaining = total;
    while (remaining) {
        const auto active_lanes = static_cast<uint32_t>(remaining < kMveLanes ? remaining : kMveLanes);
        const mve_pred16_t predicate = vctp8q(active_lanes);
        vstrbq_p_u8(bytes, veorq_u8(vldrbq_z_u8(bytes, predicate), mask), predicate);
        bytes += active_lanes;
        remaining -= active_lanes;
    }
#else
    constexpr size_t kWordSize = sizeof(uint32_t);
    constexpr uint32_t kUint8ToInt8Word = 0x80808080u;
    const size_t word_count = total / kWordSize;
    for (size_t index = 0; index < word_count; ++index) {
        uint32_t word;
        std::memcpy(&word, bytes + index * kWordSize, sizeof(word));
        word ^= kUint8ToInt8Word;
        std::memcpy(bytes + index * kWordSize, &word, sizeof(word));
    }
    for (size_t index = word_count * kWordSize; index < total; ++index)
        bytes[index] ^= kUint8ToInt8Offset;
#endif
}

bool Preprocessor::assign(Tensor &tensor, const InputData &input, Rect *assigned_rect) const
{
    if (input.empty()) return false;
    if (input.type() == InputType::raw) {
        if (!tensor.assign(input.data(), input.size())) return false;
        if (assigned_rect) *assigned_rect = {};
        return true;
    }
    const Layout tensor_layout = tensor.layout() == Layout::none ? tensor_layout_ : tensor.layout();
    const Dimensions input_dimensions = input.dimensions();
    const Dimensions tensor_dimensions(tensor.shape(), tensor_layout);
    if (input.type() != InputType::image_8bits || input_dimensions.empty() || tensor_dimensions.empty() ||
        input_dimensions.n != 1 || tensor_dimensions.n != 1) return false;
    if (tensor.data_type() != DataType::int8 && tensor.data_type() != DataType::uint8) return false;

    const size_t height = input_dimensions.h;
    const size_t width = input_dimensions.w;
    const size_t channels = input_dimensions.c;
    if (!width || !height || (channels != 1 && channels != 3 && channels != 4) ||
        width > std::numeric_limits<size_t>::max() / height / channels ||
        width * height * channels != input.size() || tensor.item_count() != input.size() ||
        tensor_dimensions.h != height || tensor_dimensions.w != width ||
        tensor_dimensions.c != channels) return false;

    if (input.layout() == tensor_layout) {
        if (!tensor.assign(input.data(), input.size())) return false;
        if (tensor.data_type() == DataType::int8) convert_to_int8(tensor.data(), tensor.size());
    } else {
        Buffer converted;
        if (!converted.resize(input.size())) return false;
        const auto *source = static_cast<const uint8_t *>(input.data());
        auto *destination = static_cast<uint8_t *>(converted.data());
        const size_t pixels = width * height;
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
            for (size_t channel = 0; channel < channels; ++channel) {
                const size_t interleaved = pixel * channels + channel;
                const size_t planar = channel * pixels + pixel;
                uint8_t value = source[input.layout() == Layout::nhwc ? interleaved : planar];
                if (tensor.data_type() == DataType::int8) value ^= kUint8ToInt8Offset;
                destination[tensor_layout == Layout::nhwc ? interleaved : planar] = value;
            }
        }
        if (!tensor.assign(converted.data(), converted.size())) return false;
    }
    if (assigned_rect) *assigned_rect = {{0.0f, 0.0f}, {static_cast<float>(width), static_cast<float>(height)}};
    return true;
}

bool Preprocessor::assign(Tensors &tensors, const InputData &input, size_t start_index,
                          Rect *assigned_rect) const
{
    if (start_index >= tensors.size()) return false;
    for (size_t index = start_index; index < tensors.size(); ++index) {
        if (!assign(tensors[index], input, assigned_rect)) return false;
    }
    return true;
}

} // namespace torq
