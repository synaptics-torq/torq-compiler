// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file    detector.cpp
 * \brief   Object detection postprocessor implementation.
 * \details Dequantises three int8 output tensors (box1, scores, box2),
 *          applies a confidence threshold, and runs greedy IoU-based NMS
 *          to produce a filtered list of bounding-box detections.
 */

#include "torq/detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#endif

namespace torq {

class Detector::Impl {
public:
    using Result = Detector::Result;

    Impl(float score_threshold, int n_max, bool nms, float iou_threshold, bool iou_with_min)
        : score_threshold_(score_threshold), max_detections_(n_max), nms_(nms),
          iou_threshold_(iou_threshold), iou_with_min_(iou_with_min) {}
    virtual ~Impl() = default;
    virtual bool init(const Tensors &tensors) const = 0;
    virtual Result process(const Tensors &tensors, const Rect &input_rect) = 0;

protected:
    bool valid_options() const {
        return max_detections_ >= 0 && std::isfinite(score_threshold_) &&
            score_threshold_ >= 0.0f && score_threshold_ <= 1.0f &&
            std::isfinite(iou_threshold_) && iou_threshold_ >= 0.0f && iou_threshold_ <= 1.0f;
    }

    bool keep_detection(const Result::Item &candidate, const Result &selected) const {
        return !nms_ || std::none_of(selected.items.begin(), selected.items.end(),
            [&](const Result::Item &kept) {
                return detail::intersection_over_union(candidate.bounding_box, kept.bounding_box,
                    iou_with_min_) > iou_threshold_;
            });
    }

    float score_threshold_;
    int max_detections_;
    bool nms_;
    float iou_threshold_;
    bool iou_with_min_;
};

namespace {

class Yolov8nPersonDetection final : public Detector::Impl {
public:
    using Impl::Impl;
    bool init(const Tensors &tensors) const override;
    Result process(const Tensors &tensors, const Rect &) override;
};

class Yolov8nOD final : public Detector::Impl {
public:
    Yolov8nOD(float score_threshold, int n_max, bool nms, float iou_threshold,
              bool iou_with_min, float scale, int32_t zero_point)
        : Impl(score_threshold, n_max, nms, iou_threshold, iou_with_min),
          scale_(scale), zero_point_(zero_point) {}

    bool init(const Tensors &tensors) const override;
    Result process(const Tensors &tensors, const Rect &input_rect) override;

private:
    float scale_;
    int32_t zero_point_;
    std::vector<int8_t> best_scores_;
    std::vector<uint8_t> best_classes_;
};

static constexpr size_t kNumAnchors       = 600;
static constexpr size_t kRequiredOutputs  = 3;
/** Number of (x, y) values per anchor in the box1/box2 tensors. */
static constexpr size_t kBoxTensorStride  = 2;

static_assert(kNumAnchors - 1 <= std::numeric_limits<uint16_t>::max());

static constexpr float SCALE_X2Y2   = 2.2298152446746826f;
static constexpr int   ZP_X2Y2      = -84;
static constexpr float SCALE_X1Y1   = 2.4782917499542236f;
static constexpr int   ZP_X1Y1      = -128;
static constexpr float SCALE_SCORES = 0.00390625f;
static constexpr int   ZP_SCORES    = -128;
static constexpr float PD_MIN_BOX_DIM = 1.0f;

bool Yolov8nPersonDetection::init(const Tensors &tensors) const
{
    if (!valid_options() || tensors.size() != kRequiredOutputs) return false;
    for (const auto &tensor : tensors) if (tensor.data_type() != DataType::int8) return false;
    return tensors[0].size() == kNumAnchors * kBoxTensorStride &&
           tensors[1].size() == kNumAnchors && tensors[2].size() == kNumAnchors * kBoxTensorStride;
}

Detector::Result Yolov8nPersonDetection::process(const Tensors &tensors, const Rect &)
{
    if (!init(tensors)) return {};
    for (const auto &tensor : tensors) if (!tensor.data()) return {};
    Result result;
    const auto *second_corner = static_cast<const int8_t *>(tensors[0].data());
    const auto *scores = static_cast<const int8_t *>(tensors[1].data());
    const auto *first_corner = static_cast<const int8_t *>(tensors[2].data());
    std::vector<uint16_t> candidates;
    candidates.reserve(kNumAnchors);
    for (size_t index = 0; index < kNumAnchors; ++index) {
        if (static_cast<float>(scores[index] - ZP_SCORES) * SCALE_SCORES >= score_threshold_)
            candidates.push_back(static_cast<uint16_t>(index));
    }

    result.success = true;
    if (candidates.empty()) return result;

    const auto compare_score = [&](uint16_t left, uint16_t right) {
        return scores[left] < scores[right] || (scores[left] == scores[right] && left > right);
    };
    std::make_heap(candidates.begin(), candidates.end(), compare_score);

    const size_t limit = max_detections_ ? std::min(static_cast<size_t>(max_detections_), candidates.size())
                                       : candidates.size();
    result.items.reserve(limit);
    while (!candidates.empty() && result.items.size() < limit) {
        const uint16_t index = candidates.front();
        std::pop_heap(candidates.begin(), candidates.end(), compare_score);
        candidates.pop_back();
        const size_t offset = static_cast<size_t>(index) * kBoxTensorStride;
        const float first_x = static_cast<float>(first_corner[offset] - ZP_X1Y1) * SCALE_X1Y1;
        const float first_y = static_cast<float>(first_corner[offset + 1] - ZP_X1Y1) * SCALE_X1Y1;
        const float second_x = static_cast<float>(second_corner[offset] - ZP_X2Y2) * SCALE_X2Y2;
        const float second_y = static_cast<float>(second_corner[offset + 1] - ZP_X2Y2) * SCALE_X2Y2;
        Result::Item detection;
        detection.confidence = static_cast<float>(scores[index] - ZP_SCORES) * SCALE_SCORES;
        auto &box = detection.bounding_box;
        box.origin = {std::min(first_x, second_x), std::min(first_y, second_y)};
        box.size = {std::max(first_x, second_x) - box.origin.x,
                    std::max(first_y, second_y) - box.origin.y};
        if (box.size.x <= PD_MIN_BOX_DIM || box.size.y <= PD_MIN_BOX_DIM) continue;
        if (keep_detection(detection, result)) result.items.push_back(detection);
    }
    return result;
}

constexpr size_t kYoloBoxParams = 4;
constexpr float kYoloHalf = 0.5f;

bool Yolov8nOD::init(const Tensors &tensors) const
{
    if (!valid_options() || tensors.size() != 1 || tensors[0].data_type() != DataType::int8 ||
        !tensors[0].size() || !std::isfinite(scale_) || scale_ <= 0.0f ||
        zero_point_ < std::numeric_limits<int8_t>::min() ||
        zero_point_ > std::numeric_limits<int8_t>::max()) return false;
    const auto &shape = tensors[0].shape();
    constexpr size_t output_rank = 3;
    return shape.size() == output_rank && shape[0] == 1 && shape[1] > kYoloBoxParams &&
        shape[1] - kYoloBoxParams <= static_cast<uint32_t>(std::numeric_limits<uint8_t>::max()) + 1 &&
        shape[2] <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

Detector::Result Yolov8nOD::process(const Tensors &tensors, const Rect &input_rect)
{
    if (!init(tensors) || !tensors[0].data() ||
        !std::isfinite(input_rect.origin.x) || !std::isfinite(input_rect.origin.y) ||
        !std::isfinite(input_rect.size.x) || !std::isfinite(input_rect.size.y) ||
        input_rect.size.x <= 0.0f || input_rect.size.y <= 0.0f) return {};

    Result result;
    result.success = true;
    const size_t num_classes = tensors[0].shape()[1] - kYoloBoxParams;
    const size_t num_anchors = tensors[0].shape()[2];
    const auto *raw = static_cast<const int8_t *>(tensors[0].data());
    const float raw_threshold = std::ceil(score_threshold_ / scale_) + static_cast<float>(zero_point_);
    if (raw_threshold > std::numeric_limits<int8_t>::max()) return result;
    const int8_t confidence_threshold = static_cast<int8_t>(
        std::max(static_cast<float>(std::numeric_limits<int8_t>::min()), raw_threshold));

    best_scores_.resize(num_anchors);
    best_classes_.assign(num_anchors, 0);
    std::memcpy(best_scores_.data(), raw + kYoloBoxParams * num_anchors, num_anchors);
    for (size_t class_index = 1; class_index < num_classes; ++class_index) {
        const int8_t *class_scores = raw + (kYoloBoxParams + class_index) * num_anchors;
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
        constexpr size_t simd_lanes = 16;
        const uint8x16_t class_ids = vdupq_n_u8(static_cast<uint8_t>(class_index));
        for (size_t offset = 0; offset < num_anchors;) {
            const size_t active_lanes = std::min(num_anchors - offset, simd_lanes);
            const mve_pred16_t predicate = vctp8q(static_cast<uint32_t>(active_lanes));
            const int8x16_t scores = vldrbq_z_s8(class_scores + offset, predicate);
            const int8x16_t best = vldrbq_z_s8(best_scores_.data() + offset, predicate);
            const uint8x16_t classes = vldrbq_z_u8(best_classes_.data() + offset, predicate);
            const mve_pred16_t greater = vcmpgtq_m_s8(scores, best, predicate);
            vstrbq_p_s8(best_scores_.data() + offset, vpselq_s8(scores, best, greater), predicate);
            vstrbq_p_u8(best_classes_.data() + offset, vpselq_u8(class_ids, classes, greater), predicate);
            offset += active_lanes;
        }
#else
        for (size_t anchor = 0; anchor < num_anchors; ++anchor) {
            if (class_scores[anchor] > best_scores_[anchor]) {
                best_scores_[anchor] = class_scores[anchor];
                best_classes_[anchor] = static_cast<uint8_t>(class_index);
            }
        }
#endif
    }

    std::vector<Result::Item> candidates;
    for (size_t anchor = 0; anchor < num_anchors; ++anchor) {
        if (best_scores_[anchor] < confidence_threshold) continue;
        Result::Item item;
        item.class_index = best_classes_[anchor];
        item.confidence = static_cast<float>(best_scores_[anchor] - zero_point_) * scale_;
        const float center_x = static_cast<float>(raw[anchor] - zero_point_) * scale_ * input_rect.size.x;
        const float center_y = static_cast<float>(raw[num_anchors + anchor] - zero_point_) * scale_ * input_rect.size.y;
        const float width = static_cast<float>(raw[2 * num_anchors + anchor] - zero_point_) *
            scale_ * input_rect.size.x;
        const float height = static_cast<float>(raw[3 * num_anchors + anchor] - zero_point_) *
            scale_ * input_rect.size.y;
        item.bounding_box = {{center_x - width * kYoloHalf, center_y - height * kYoloHalf}, {width, height}};
        if (!std::isfinite(item.confidence) || !std::isfinite(item.bounding_box.origin.x) ||
            !std::isfinite(item.bounding_box.origin.y) || !std::isfinite(width) ||
            !std::isfinite(height)) return {};
        if (width <= 0.0f || height <= 0.0f) continue;
        candidates.push_back(item);
    }
    const auto compare_score = [](const Result::Item &left, const Result::Item &right) {
        return left.confidence < right.confidence;
    };
    std::make_heap(candidates.begin(), candidates.end(), compare_score);
    const size_t limit = max_detections_ ? std::min(static_cast<size_t>(max_detections_), candidates.size())
                                       : candidates.size();
    result.items.reserve(limit);
    while (!candidates.empty() && result.items.size() < limit) {
        const Result::Item candidate = candidates.front();
        std::pop_heap(candidates.begin(), candidates.end(), compare_score);
        candidates.pop_back();
        if (keep_detection(candidate, result)) result.items.push_back(candidate);
    }
    return result;
}

}

Detector::Detector(float score_threshold, int n_max, bool nms, float iou_threshold,
                   bool iou_with_min, float scale, int32_t zero_point)
    : score_threshold_(score_threshold), max_detections_(n_max), nms_(nms),
      iou_threshold_(iou_threshold), iou_with_min_(iou_with_min), scale_(scale), zero_point_(zero_point) {}

Detector::~Detector() = default;

bool Detector::init(const Tensors &tensors)
{
    if (d && d->init(tensors)) return true;
    std::unique_ptr<Impl> implementation;
    if (tensors.size() == kRequiredOutputs) {
        implementation = std::make_unique<Yolov8nPersonDetection>(
            score_threshold_, max_detections_, nms_, iou_threshold_, iou_with_min_);
    } else if (tensors.size() == 1) {
        implementation = std::make_unique<Yolov8nOD>(
            score_threshold_, max_detections_, nms_, iou_threshold_, iou_with_min_, scale_, zero_point_);
    }
    if (!implementation || !implementation->init(tensors)) {
        d.reset();
        return false;
    }
    d = std::move(implementation);
    return true;
}

Detector::Result Detector::process(const Tensors &tensors, const Rect &input_rect)
{
    if (!init(tensors) || !std::isfinite(input_rect.origin.x) || !std::isfinite(input_rect.origin.y) ||
        !std::isfinite(input_rect.size.x) || !std::isfinite(input_rect.size.y) ||
        input_rect.size.x < 0.0f || input_rect.size.y < 0.0f ||
        ((input_rect.size.x == 0.0f) != (input_rect.size.y == 0.0f))) return {};
    Result result = d->process(tensors, input_rect);
    if (!result.success || input_rect.size.x == 0.0f) return result;
    for (auto &item : result.items) {
        auto &box = item.bounding_box;
        const float right = std::clamp(std::round(box.origin.x + box.size.x), 0.0f, input_rect.size.x);
        const float bottom = std::clamp(std::round(box.origin.y + box.size.y), 0.0f, input_rect.size.y);
        const float left = std::clamp(std::round(box.origin.x), 0.0f, input_rect.size.x);
        const float top = std::clamp(std::round(box.origin.y), 0.0f, input_rect.size.y);
        box.origin = {left + input_rect.origin.x, top + input_rect.origin.y};
        box.size = {right - left, bottom - top};
    }
    return result;
}

} // namespace torq
