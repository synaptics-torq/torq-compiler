// Copyright 2024 Synaptics
// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "native_executable.h"
#include "TorqExecutable.h"

#include "TorqHw.h"
#include "TorqUtils.h"
#include "TorqEventLog.h"
#include "iree/base/api.h"

#include "iree/base/internal/flatcc/parsing.h"

#include "torq_executable_def_reader.h"
#include "torq_executable_def_verifier.h"
#include "torq_device.h"

#include "iree/hal/pipeline_layout.h"
#include "iree/base/internal/flags.h"

#include <unistd.h>

namespace {

extern const iree_hal_executable_vtable_t
    iree_hal_torq_native_executable_vtable;
}  // namespace

iree_hal_torq_native_executable_t*
iree_hal_torq_native_executable_cast(iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_torq_native_executable_vtable);
  return (iree_hal_torq_native_executable_t*)base_value;
}

iree_status_t iree_hal_torq_native_executable_create(
    iree_hal_device_t* device,
    iree_allocator_t allocator,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  iree_allocator_t host_allocator = allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_torq_native_executable_t* executable = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*executable) +
      executable_params->pipeline_layout_count * sizeof(*executable->pipeline_layouts) +
      executable_params->constant_count * sizeof(*executable_params->constants),
      (void**)&executable);

  executable->torq_executable = nullptr;
  executable->program = nullptr;

  executable->program = malloc(executable_params->executable_data.data_length);

  if (executable->program == nullptr) {    
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  }
  
  memcpy(executable->program, executable_params->executable_data.data, executable_params->executable_data.data_length);

  executable->pipeline_layout_count = executable_params->pipeline_layout_count;

  for (iree_host_size_t i = 0; i <  executable_params->pipeline_layout_count; ++i) {
    executable->pipeline_layouts[i] = executable_params->pipeline_layouts[i];
    iree_hal_pipeline_layout_retain(executable->pipeline_layouts[i]);
  }

  if (executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable) +
                    executable_params->pipeline_layout_count *
                        sizeof(*executable->pipeline_layouts));
    memcpy(target_constants, executable_params->constants,
           executable_params->constant_count *
               sizeof(*executable_params->constants));
    executable->constants = target_constants;
  } else {
    executable->constants = NULL;
  }

  synaptics::TorqExecutable* torq_executable = nullptr;

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_torq_native_executable_vtable,
                                 &executable->resource);
    executable->allocator = allocator;
    executable->entry_point_count = 0;
    
    executable->device = device;
  
    torq_executable = new synaptics::TorqExecutable(executable);

    executable->torq_executable = torq_executable;

    if (!executable->torq_executable) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }    

  }

  if (iree_status_is_ok(status)) {
    status = torq_executable->initialize();   
  }  

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_torq_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_torq_native_executable_t* executable =
      iree_hal_torq_native_executable_cast(base_executable);
  iree_allocator_t host_allocator =
      executable->allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable->torq_executable) {
    delete static_cast<synaptics::TorqExecutable*>(executable->torq_executable);   
  }

  if (executable->program) {
    free(executable->program);
  }

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static void printDispatchInvocationInfo(iree_hal_torq_native_executable_t* executable, iree_hal_executable_dispatch_state_v0_t* dispatch_state) {

  iree_hal_torq_ExecutableDef_table_t executable_def =
    iree_hal_torq_ExecutableDef_as_root(executable->program);

  std::string executable_name = iree_hal_torq_ExecutableDef_executable_name_get(executable_def);
  
  LOGD << "Dispatch State:";
  LOGD << "  executable=" << executable;
  LOGD << "  workgroup_count x=" << dispatch_state->workgroup_count_x << " y=" << dispatch_state->workgroup_count_y << " z=" << dispatch_state->workgroup_count_z;
  LOGD << "  pipeline_layout_count=" << executable->pipeline_layout_count;
  LOGD << "  binding count=" << std::to_string(dispatch_state->binding_count);
  LOGD << "  constants count=" << dispatch_state->push_constant_count;
  
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
      LOGD << "  binding[" << i << "]: buffer=" << dispatch_state->binding_ptrs[i] << " length=" << dispatch_state->binding_lengths[i];
  }

}

iree_status_t iree_hal_torq_native_executable_run(iree_hal_executable_t* base_value, iree_hal_executable_dispatch_state_v0_t* dispatch_state) {

    iree_hal_torq_native_executable_t* executable = iree_hal_torq_native_executable_cast(base_value);

    if (TorqLogger::enabled(TORQ_LOG_DEBUG)) {
      printDispatchInvocationInfo(executable, dispatch_state);
    }
    
    synaptics::TorqExecutable* torq_executable = static_cast<synaptics::TorqExecutable*>(executable->torq_executable);

    return torq_executable->executeDispatch(dispatch_state);
    
}

namespace {
const iree_hal_executable_vtable_t iree_hal_torq_native_executable_vtable = {
    /*.destroy=*/iree_hal_torq_native_executable_destroy,
};
}  // namespace
