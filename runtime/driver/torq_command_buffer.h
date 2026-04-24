// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_torq_dispatch_state_v0_t {
    // Keep the standard IREE dispatch state first so executable entry points can
    // continue treating this as iree_hal_executable_dispatch_state_v0_t.
    iree_alignas(64) iree_hal_executable_dispatch_state_v0_t dispatch_state;
    void *binding_ptr_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    size_t binding_length_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    iree_hal_buffer_t *binding_buffer_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    iree_device_size_t binding_offset_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    iree_device_size_t binding_range_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    uint8_t binding_flags_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
} iree_hal_torq_dispatch_state_v0_t;

#define IREE_HAL_TORQ_BINDING_FLAG_ZERO_COPY_ELIGIBLE 0x1u
#define IREE_HAL_TORQ_BINDING_FLAG_ZERO_COPY_ATTACHED 0x2u

static inline iree_hal_torq_dispatch_state_v0_t *iree_hal_torq_dispatch_state_cast(
    iree_hal_executable_dispatch_state_v0_t *state
) {
    return (iree_hal_torq_dispatch_state_v0_t *)state;
}

static inline const iree_hal_torq_dispatch_state_v0_t *iree_hal_torq_dispatch_state_const_cast(
    const iree_hal_executable_dispatch_state_v0_t *state
) {
    return (const iree_hal_torq_dispatch_state_v0_t *)state;
}

// Creates an inline synchronous one-shot single-threaded command "buffer".
// This is designed for ultra-low latency situations where we know the command
// buffer is going to be submitted with no wait semaphores indicating that it
// can begin execution immediately. No inter-command-buffer scheduling will be
// performed and all barriers and events are ignored.
//
// Executes all work on the calling thread synchronously (today).
//
// Must have IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION set.
iree_status_t iree_hal_torq_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is an inline command buffer.
bool iree_hal_torq_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

iree_host_size_t iree_hal_torq_command_buffer_size(
    iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity);

iree_status_t iree_hal_torq_command_buffer_initialize(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, iree_byte_span_t storage,
    iree_hal_command_buffer_t** out_command_buffer);

void iree_hal_torq_command_buffer_deinitialize(
    iree_hal_command_buffer_t* base_command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
