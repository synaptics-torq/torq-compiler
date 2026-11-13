// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    postprocessor.hpp
 * \brief   Common detection result types.
 */

#ifndef TORQ_POSTPROCESSOR_HPP_
#define TORQ_POSTPROCESSOR_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "torq/types.hpp"

namespace torq {

/// Decoded detections and processing status.
struct DetectionResult {
    /// Class, confidence, and bounding box of a detected object.
    struct Item {
        int32_t class_index = 0;
        float confidence = 0.0f;
        Rect bounding_box;
    };

    bool success = false;
    std::vector<Item> items;
};

namespace detail {

/// Compute rectangle overlap relative to union area or the smaller box area.
/// @param left: first bounding box
/// @param right: second bounding box
/// @param with_min: use the smaller area as denominator
/// @return overlap ratio, or 0 for a nonpositive denominator
inline float intersection_over_union(const Rect &left, const Rect &right, bool with_min = false)
{
    const float width = std::max(0.0f,
        std::min(left.origin.x + left.size.x, right.origin.x + right.size.x) -
        std::max(left.origin.x, right.origin.x));
    const float height = std::max(0.0f,
        std::min(left.origin.y + left.size.y, right.origin.y + right.size.y) -
        std::max(left.origin.y, right.origin.y));
    const float intersection = width * height;
    const float left_area = left.size.x * left.size.y;
    const float right_area = right.size.x * right.size.y;
    const float denominator = with_min ? std::min(left_area, right_area)
                                      : left_area + right_area - intersection;
    return denominator > 0.0f ? intersection / denominator : 0.0f;
}

}

} // namespace torq

#endif /* TORQ_POSTPROCESSOR_HPP_ */
