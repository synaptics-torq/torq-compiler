// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    classifier.cpp
 * \brief   Int8 classification postprocessor implementation.
 * \details Dequantises int8 logits from the model output tensor and
 *          returns the top-K class predictions sorted by descending confidence.
 */

#include "torq/classifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace torq {

Classifier::Classifier(size_t top_count, float scale, int32_t zero_point)
    : top_count_(top_count), scale_(scale), zero_point_(zero_point) {}

Classifier::Result Classifier::process(const Tensors &tensors) const
{
    if (tensors.size() != 1 || !tensors[0].data() || !tensors[0].size()) return {};
    const auto &tensor = tensors[0];
    const auto type = tensor.data_type();
    if (type != DataType::int8 && type != DataType::uint8 && type != DataType::float32) return {};
    if (type != DataType::float32 && (!std::isfinite(scale_) || scale_ <= 0.0f ||
        (type == DataType::int8 && (zero_point_ < std::numeric_limits<int8_t>::min() ||
                                   zero_point_ > std::numeric_limits<int8_t>::max())) ||
        (type == DataType::uint8 && (zero_point_ < 0 ||
                                    zero_point_ > std::numeric_limits<uint8_t>::max())))) return {};

    const size_t count = tensor.item_count();
    if (!count || count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) return {};
    const size_t keep = top_count_ ? std::min(top_count_, count) : count;
    Result result;
    result.items.reserve(keep);
    const auto better = [](const Result::Item &left, const Result::Item &right) {
        return left.confidence > right.confidence ||
               (left.confidence == right.confidence && left.class_index < right.class_index);
    };
    for (size_t index = 0; index < count; ++index) {
        float confidence;
        if (type == DataType::float32) {
            std::memcpy(&confidence, static_cast<const uint8_t *>(tensor.data()) + index * sizeof(float),
                        sizeof(confidence));
        } else {
            const int32_t value = type == DataType::int8
                ? static_cast<const int8_t *>(tensor.data())[index]
                : static_cast<const uint8_t *>(tensor.data())[index];
            confidence = static_cast<float>(value - zero_point_) * scale_;
        }
        if (!std::isfinite(confidence)) return {};
    Result::Item item{static_cast<int32_t>(index), confidence};
        if (keep == 1) {
            if (result.items.empty()) result.items.push_back(item);
            else if (better(item, result.items.front())) result.items.front() = item;
            continue;
        }
        if (result.items.size() < keep) {
            result.items.push_back(item);
        } else {
            if (!better(item, result.items.front())) continue;
            std::pop_heap(result.items.begin(), result.items.end(), better);
            result.items.back() = item;
        }
        std::push_heap(result.items.begin(), result.items.end(), better);
    }
    std::sort_heap(result.items.begin(), result.items.end(), better);
    result.success = true;
    return result;
}

} // namespace torq
