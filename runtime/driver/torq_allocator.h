// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TorqDeviceBufferAPI.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Creates a Torq device buffer allocator.
//
// The caller selects allocation mode explicitly:
// - DMA heap on Astra Machina builds.
// - Aligned host memory on other builds.
iree_status_t iree_hal_torq_allocator_create(
    iree_string_view_t identifier, iree_allocator_t host_allocator,
    iree_hal_allocator_t **out_allocator
);

bool iree_hal_torq_buffer_isa(const iree_hal_buffer_t *buffer);

const torq_hw_device_buffer_t *iree_hal_torq_buffer_device_buffer(const iree_hal_buffer_t *buffer);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
