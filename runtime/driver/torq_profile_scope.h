// Copyright 2024 Synaptics
// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hal_torq_profile_event_type_t {
  IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_WAIT,
  IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_APPLY_DEFERRED,
  IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_SIGNAL,
  IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_COPY_BUFFER,
  IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET,
} iree_hal_torq_profile_event_type_t;

/*--- pure C API to manage HAL profiling scopes ---*/

void* iree_hal_torq_profile_scope_begin(const char* scope_name);

void iree_hal_torq_profile_scope_add_event(
    void* scope, int32_t event_type, int32_t time_tag, int32_t action_index);

void iree_hal_torq_profile_scope_end(void* scope);

// If |status| is OK, emits begin/end events around |expr| and stores the
// result back to |status|. If |status| is already failed, this is a no-op.
#define IREE_HAL_TORQ_PROFILE_STAGE_IF_OK_WITH_ACTION(                      \
    scope, status, event, action_index, expr)              \
  do {                                                                       \
    if (iree_status_is_ok((status))) {                                       \
      iree_hal_torq_profile_scope_add_event((scope), (event), 0,          \
                                            (action_index));                 \
      (status) = (expr);                                                     \
      iree_hal_torq_profile_scope_add_event((scope), (event), 1,          \
                                            (action_index));                 \
    }                                                                        \
  } while (false)

// Convenience wrapper for scope-level stages that use action id = -1.
#define IREE_HAL_TORQ_PROFILE_STAGE_IF_OK(scope, status, event, expr)        \
  IREE_HAL_TORQ_PROFILE_STAGE_IF_OK_WITH_ACTION(                             \
      (scope), (status), (event), -1, (expr))

/* ----------------------------------------------- */

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
