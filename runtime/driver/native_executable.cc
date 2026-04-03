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

#include "iree/hal/utils/executable_header.h"

#include "torq_executable_def_reader.h"
#include "torq_executable_def_verifier.h"
#include "torq_device.h"

//#include "iree/hal/pipeline_layout.h"
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
      executable_params->constant_count * sizeof(*executable_params->constants),
      (void**)&executable);

  executable->torq_executable = nullptr;
  executable->program = nullptr;

  executable->program = malloc(executable_params->executable_data.data_length);
  memcpy(executable->program, executable_params->executable_data.data, executable_params->executable_data.data_length);

  if (executable_params->constant_count > 0) {
    uint32_t* target_constants =
        (uint32_t*)((uint8_t*)executable + sizeof(*executable));    
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
  LOGD << "  binding count=" << std::to_string(dispatch_state->binding_count);
  LOGD << "  constants count=" << dispatch_state->constant_count;
  
  for (iree_host_size_t i = 0; i < dispatch_state->binding_count; ++i) {
      LOGD << "  binding[" << i << "]: buffer=" << dispatch_state->binding_ptrs[i] << " length=" << dispatch_state->binding_lengths[i];
  }

}

iree_status_t iree_hal_torq_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
        
  // Read the header prefix (with unsafe inference if size is unknown).
  const bool unsafe_infer_size = (executable_data.data_length == 0);
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
      executable_data, unsafe_infer_size,
      iree_hal_torq_ExecutableDef_file_identifier, &flatbuffer_data));

  // Verify the flatbuffer structure.
  if (!iree_hal_torq_ExecutableDef_verify_as_root(
          flatbuffer_data.data, flatbuffer_data.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "failed to verify executable flatbuffer structure");
  }

  std::string torq_file_format = "torq-fb";
  memcpy(executable_format, torq_file_format.c_str(), torq_file_format.length() + /*NUL*/ 1);

  // Return the total size (header + flatbuffer).
  *out_inferred_size =
      sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
  return iree_ok_status();
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
