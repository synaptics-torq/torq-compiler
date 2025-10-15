// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_torq_nop_executable_cache_create(
    iree_hal_device_t* device,
    iree_string_view_t identifier,
    iree_allocator_t allocator,
    iree_hal_executable_cache_t** out_executable_cache);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
