// Copyright 2024 Synaptics
// Copyright 2019 The IREE Authors
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


typedef struct iree_hal_torq_native_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t allocator;
  iree_hal_device_t *device;
  iree_host_size_t entry_point_count;
  //iree_hal_torq_ExecutableDef_table_t executable_def;
  uint32_t* constants;
  iree_host_size_t pipeline_layout_count;
  void *program;
  iree_hal_pipeline_layout_t* pipeline_layouts[];
} iree_hal_torq_native_executable_t;


typedef struct iree_hal_torq_source_location_t {
  iree_string_view_t file_name;
  int line;
  iree_string_view_t func_name;
} iree_hal_torq_source_location_t;

iree_status_t iree_hal_torq_native_executable_create(
    iree_hal_device_t* device,
    iree_allocator_t allocator,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable);

void iree_hal_torq_native_executable_entry_point_source_location(
    iree_hal_executable_t* executable, iree_host_size_t entry_ordinal,
    iree_hal_torq_source_location_t* out_source_location);

iree_status_t iree_hal_torq_native_executable_run(
    iree_hal_executable_t* base_value, iree_hal_executable_dispatch_state_v0_t* dispatch_state);

iree_hal_torq_native_executable_t* iree_hal_torq_native_executable_cast(
    iree_hal_executable_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
