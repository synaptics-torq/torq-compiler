// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    preprocessor.hpp
 * \brief   Image preprocessor interfaces used by the TORQ engine.
 */

#ifndef TORQ_PREPROCESSOR_HPP_
#define TORQ_PREPROCESSOR_HPP_

#include <cstddef>

#include "torq/input_data.hpp"
#include "torq/tensor.hpp"

namespace torq {

/// Assign prepared bytes or matching-size 8-bit images to input tensors.
class Preprocessor {
public:
    /// @param tensor_layout: fallback layout for tensors with unspecified layout
    explicit Preprocessor(Layout tensor_layout = Layout::nhwc) : tensor_layout_(tensor_layout) {}

    /// Copy raw bytes or reorder image layout and convert unsigned pixels for int8.
    /// @param tensor: destination tensor; images must match its dimensions
    /// @param input: prepared bytes or image data
    /// @param assigned_rect: optional image bounds, or an empty rectangle for raw data
    /// @return true if assignment succeeds
    bool assign(Tensor &tensor, const InputData &input, Rect *assigned_rect = nullptr) const;
    /// Assign the same input to every tensor from start_index onward.
    /// @param tensors: destination collection
    /// @param input: prepared bytes or image data
    /// @param start_index: first destination tensor
    /// @param assigned_rect: optional bounds of the last successfully assigned input
    /// @return true if every assignment succeeds
    bool assign(Tensors &tensors, const InputData &input, size_t start_index = 0,
                Rect *assigned_rect = nullptr) const;

private:
    Layout tensor_layout_;
};

} // namespace torq

#endif /* TORQ_PREPROCESSOR_HPP_ */
