// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    detector.hpp
 * \brief   Postprocessing for generic object detection models.
 */

#ifndef TORQ_DETECTOR_HPP_
#define TORQ_DETECTOR_HPP_

#include <cstdint>
#include <memory>

#include "torq/postprocessor.hpp"
#include "torq/tensor.hpp"

namespace torq {

/**
 * \brief Generic object detection postprocessor.
 *
 * Selects the person-detection or object-detection decoder from output tensors,
 * applies a confidence threshold, and runs greedy NMS.
 */
class Detector {
public:
    using Result = DetectionResult;

    /// Configure detection filtering and output dequantization.
    /// @param score_threshold: minimum retained confidence
    /// @param n_max: maximum number of detections
    /// @param nms: enable non-maximum suppression
    /// @param iou_threshold: overlap threshold for suppression
    /// @param iou_with_min: use the smaller box area instead of the union area
    /// @param scale: quantized output scale
    /// @param zero_point: quantized output zero point
    explicit Detector(float score_threshold = 0.25f, int n_max = 6,
                      bool nms = true, float iou_threshold = 0.6f, bool iou_with_min = false,
                      float scale = 0.004194467328488827f, int32_t zero_point = -128);
    ~Detector();

    Detector(const Detector &) = delete;
    Detector &operator=(const Detector &) = delete;

    /// Select a decoder compatible with the model outputs.
    /// @param tensors: output tensor descriptions
    /// @return true if a supported decoder is initialized
    bool init(const Tensors &tensors);
    /// Decode and filter detections.
    /// @param tensors: model output tensors
    /// @param input_rect: optional input region used to map bounding boxes
    /// @return detections with success set, or an unsuccessful result on failure
    Result process(const Tensors &tensors, const Rect &input_rect = {});

    class Impl;

private:
    float score_threshold_;
    int max_detections_;
    bool nms_;
    float iou_threshold_;
    bool iou_with_min_;
    float scale_;
    int32_t zero_point_;
    std::unique_ptr<Impl> d;
};

} // namespace torq

#endif /* TORQ_DETECTOR_HPP_ */
