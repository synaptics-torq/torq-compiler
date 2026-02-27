// Copyright 2024 Synaptics
// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq_profile_scope.h"

#include "TorqEventLog.h"

#include <optional>

using namespace synaptics;

namespace {

static std::optional<EventType> to_profile_event_type(int32_t event_type) {
  switch (event_type) {
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_WAIT_BEGIN:
      return EventType::HAL_QUEUE_WAIT_BEGIN;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_WAIT_END:
      return EventType::HAL_QUEUE_WAIT_END;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_APPLY_DEFERRED_BEGIN:
      return EventType::HAL_QUEUE_APPLY_DEFERRED_BEGIN;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_APPLY_DEFERRED_END:
      return EventType::HAL_QUEUE_APPLY_DEFERRED_END;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_SIGNAL_BEGIN:
      return EventType::HAL_QUEUE_SIGNAL_BEGIN;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_SIGNAL_END:
      return EventType::HAL_QUEUE_SIGNAL_END;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_COPY_BUFFER_BEGIN:
      return EventType::HAL_COMMAND_BUFFER_COPY_BUFFER_BEGIN;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_COPY_BUFFER_END:
      return EventType::HAL_COMMAND_BUFFER_COPY_BUFFER_END;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_BEGIN:
      return EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_BEGIN;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_END:
      return EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET_END;
    default:
      return std::nullopt;
  }
}

}  // namespace

extern "C" {

void* iree_hal_torq_profile_scope_begin(const char* scope_name) {
  if (!scope_name || !scope_name[0] || !TorqEventLog::isProfilingEnabled()) {
    return nullptr;
  }
  return TorqEventLog::get().startDispatch(scope_name, EventType::HAL_CALL_BEGIN, EventType::HAL_CALL_END);
}

void iree_hal_torq_profile_scope_add_event(
    void* scope, int32_t event_type, int32_t action_index) {
  if (!scope) {
    return;
  }
  auto mapped_event_type = to_profile_event_type(event_type);
  if (!mapped_event_type) {
    return;
  }
  auto* event_log = static_cast<TorqDispatchEventLog*>(scope);
  event_log->addEvent(*mapped_event_type, action_index);
}

void iree_hal_torq_profile_scope_end(void* scope) {
  if (!scope) {
    return;
  }
  delete static_cast<TorqDispatchEventLog*>(scope);
}

}  // extern "C"
