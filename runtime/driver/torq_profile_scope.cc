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
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_WAIT:
      return EventType::HAL_QUEUE_WAIT;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_APPLY_DEFERRED:
      return EventType::HAL_QUEUE_APPLY_DEFERRED;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_QUEUE_SIGNAL:
      return EventType::HAL_QUEUE_SIGNAL;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_COPY_BUFFER:
      return EventType::HAL_COMMAND_BUFFER_COPY_BUFFER;
    case IREE_HAL_TORQ_PROFILE_EVENT_HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET:
      return EventType::HAL_COMMAND_BUFFER_PUSH_DESCRIPTOR_SET;
  }
  return std::nullopt;
}

}  // namespace

extern "C" {

void* iree_hal_torq_profile_scope_begin(const char* scope_name) {
  if (!scope_name || !scope_name[0] || !TorqEventLog::isProfilingEnabled()) {
    return nullptr;
  }
  return TorqEventLog::get().startDispatch(scope_name, EventType::HAL_CALL);
}

void iree_hal_torq_profile_scope_add_event(
    void* scope, int32_t event_type, int32_t time_tag, int32_t action_index) {
  if (!scope) {
    return;
  }
  auto mapped_event_type = to_profile_event_type(event_type);
  if (!mapped_event_type) {
    return;
  }
  auto* event_log = static_cast<TorqDispatchEventLog*>(scope);
  event_log->addEvent(*mapped_event_type, (Event::TimeTag)time_tag, action_index);
}

void iree_hal_torq_profile_scope_end(void* scope) {
  if (!scope) {
    return;
  }
  delete static_cast<TorqDispatchEventLog*>(scope);
}

}  // extern "C"
