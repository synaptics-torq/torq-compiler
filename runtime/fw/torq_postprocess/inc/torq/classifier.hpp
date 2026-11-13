// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    classifier.hpp
 * \brief   Classification postprocessor interface.
 * \details Provides a wrapper that converts quantized outputs to floating-point
 *          confidences and returns the top-K predictions.
 */

#ifndef TORQ_CLASSIFIER_HPP_
#define TORQ_CLASSIFIER_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "torq/tensor.hpp"

namespace torq {

/** \brief Default dequantisation scale (1/256). */
static constexpr float kDefaultScale     = 1.0f / 256.0f;
/** \brief Default dequantisation zero-point. */
static constexpr int   kDefaultZeroPoint = -128;
/** \brief Default number of top-K results to return. */
static constexpr size_t kDefaultTopK     = 1;

/**
 * \brief Classification postprocessor for int8, uint8, or float32 outputs.
 *
 * Converts quantized outputs to float confidences and returns the top-ranked
 * class predictions. Float32 values are used directly.
 */
class Classifier {
public:
    /// Ranked predictions and processing status.
    struct Result {
        /// Class index and its decoded confidence.
        struct Item {
            int32_t class_index;
            float confidence;
        };

        bool success = false;
        std::vector<Item> items;
    };

    /// Configure ranking and output dequantization.
    /// @param top_count: maximum number of predictions to return
    /// @param scale: quantized output scale
    /// @param zero_point: quantized output zero point
    explicit Classifier(size_t top_count = kDefaultTopK, float scale = kDefaultScale,
                        int32_t zero_point = kDefaultZeroPoint);

    /// Decode and rank a single output tensor.
    /// @param tensors: model outputs containing one supported tensor
    /// @return predictions with success set, or an unsuccessful result for invalid input
    Result process(const Tensors &tensors) const;

private:
    size_t top_count_;
    float scale_;
    int32_t zero_point_;
};

} // namespace torq

#endif /* TORQ_CLASSIFIER_HPP_ */
