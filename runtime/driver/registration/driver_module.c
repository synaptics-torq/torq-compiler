// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "driver_module.h"

#include "driver/torq_allocator.h"
#include "driver/torq_driver.h"

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/hal/local/plugins/registration/init.h"
#include "iree/hal/utils/caching_allocator.h"

#include <stddef.h>

IREE_FLAG(
    string, torq_device_allocator, "cpu",
    "Allocator backing Torq device buffers: ['cpu', 'dmabuf']"
);

/// default caching parameters, can be fine-tuned for better performance
static const iree_device_size_t kTorqCachingMaxAllocationSize =
    32ull * 1024ull * 1024ull;  // 32 MiB
static const iree_device_size_t kTorqCachingMaxPoolCapacity =
    128ull * 1024ull * 1024ull;  // 128 MiB
static const iree_host_size_t kTorqCachingMaxFreeAllocationCount = 32;

static iree_status_t iree_hal_torq_create_allocator_from_flags(
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  iree_string_view_t allocator_name =
      iree_make_cstring_view(FLAG_torq_device_allocator);
  if (iree_string_view_equal(allocator_name, IREE_SV("cpu"))) {
    return iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
                                          host_allocator, host_allocator,
                                          out_allocator);
  }
  if (iree_string_view_equal(allocator_name, IREE_SV("dmabuf"))) {
    return iree_hal_torq_allocator_create(iree_make_cstring_view("local"),
                                          host_allocator, out_allocator);
  }

  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "unsupported --torq_device_allocator value '%.*s'; expected 'cpu' or "
      "'dmabuf'",
      (int)allocator_name.size, allocator_name.data);
}

static iree_status_t iree_hal_torq_wrap_allocator_with_cache(
    iree_hal_allocator_t* base_allocator, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(base_allocator);
  IREE_ASSERT_ARGUMENT(out_allocator);
  *out_allocator = NULL;

  iree_hal_allocator_memory_heap_t heaps[8];
  iree_host_size_t heap_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_query_memory_heaps(
      base_allocator, IREE_ARRAYSIZE(heaps), heaps, &heap_count));

  iree_hal_caching_allocator_pool_params_t pool_params[8];
  for (iree_host_size_t i = 0; i < heap_count; ++i) {
    iree_hal_caching_allocator_pool_params_initialize(heaps[i], &pool_params[i]);

    if (pool_params[i].max_allocation_size > kTorqCachingMaxAllocationSize) {
      pool_params[i].max_allocation_size = kTorqCachingMaxAllocationSize;
    }
    pool_params[i].max_allocation_capacity = kTorqCachingMaxPoolCapacity;
    pool_params[i].max_free_allocation_count = kTorqCachingMaxFreeAllocationCount;
  }

  return iree_hal_caching_allocator_create_with_pools(
      heap_count, pool_params, base_allocator, host_allocator, out_allocator);
}

static iree_status_t iree_hal_torq_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t default_driver_info = {
      .driver_name = IREE_SVL("torq"),
      .full_name = IREE_SVL("Execute using Torq hardware accelerator"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = &default_driver_info;
  return iree_ok_status();
}

static iree_status_t iree_hal_torq_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  if (!iree_string_view_equal(driver_name, IREE_SV("torq"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  iree_hal_torq_device_params_t default_params;
  iree_hal_torq_device_params_initialize(&default_params);

  iree_hal_executable_plugin_manager_t* plugin_manager = NULL;
  iree_status_t status = iree_hal_executable_plugin_manager_create_from_flags(
      host_allocator, &plugin_manager);

  iree_hal_executable_loader_t* loaders[8] = {NULL};
  iree_host_size_t loader_count = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_all_available_executable_loaders(
        plugin_manager, IREE_ARRAYSIZE(loaders), &loader_count, loaders,
        host_allocator);
  }

  iree_hal_allocator_t* device_allocator = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_torq_create_allocator_from_flags(host_allocator,
                                                  &device_allocator);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_allocator_t* caching_allocator = NULL;
    iree_status_t caching_status = iree_hal_torq_wrap_allocator_with_cache(
        device_allocator, host_allocator, &caching_allocator);
    if (iree_status_is_ok(caching_status)) {
      iree_hal_allocator_release(device_allocator);
      device_allocator = caching_allocator;
    } else {
      iree_status_ignore(caching_status);
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_torq_driver_create(
        driver_name, &default_params, loader_count, loaders, device_allocator,
        host_allocator, out_driver);
  }

  iree_hal_allocator_release(device_allocator);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  iree_hal_executable_plugin_manager_release(plugin_manager);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_torq_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_torq_driver_factory_enumerate,
      .try_create = iree_hal_torq_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
