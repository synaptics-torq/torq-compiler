// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_TYPES_HPP_
#define TORQ_TYPES_HPP_

#include <cstdint>
#include <vector>

#if TORQ_OS_RTOS == TORQ_OS_LINUX
#error "Exactly one of TORQ_OS_RTOS and TORQ_OS_LINUX must be enabled."
#endif

namespace torq {

/// Tensor dimension order; none indicates an unspecified layout.
enum class Layout : uint8_t { none, nhwc, nchw };

/// Tensor extents in layout order.
using Shape = std::vector<uint32_t>;

/// Named extents of a rank-four tensor.
struct Dimensions {
    uint32_t n = 0;
    uint32_t h = 0;
    uint32_t w = 0;
    uint32_t c = 0;

    Dimensions() = default;
    /// Extract extents; unsupported shapes or layouts leave all extents zero.
    /// @param shape: rank-four tensor shape
    /// @param layout: NHWC or NCHW dimension order
    Dimensions(const Shape &shape, Layout layout) {
        if (shape.size() != 4 || (layout != Layout::nhwc && layout != Layout::nchw)) return;
        n = shape[0];
        h = shape[layout == Layout::nhwc ? 1 : 2];
        w = shape[layout == Layout::nhwc ? 2 : 3];
        c = shape[layout == Layout::nhwc ? 3 : 1];
    }

    /// @return true if any extent is zero
    bool empty() const { return !n || !h || !w || !c; }
};

/// Two-dimensional coordinates or extents.
struct Dim2d {
    float x = 0.0f;
    float y = 0.0f;
};

/// Rectangle described by an origin and width/height extents.
struct Rect {
    Dim2d origin;
    Dim2d size;

    /// @return true if either extent is nonpositive
    bool empty() const { return size.x <= 0.0f || size.y <= 0.0f; }
};

}

#endif