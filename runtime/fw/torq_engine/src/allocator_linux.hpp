// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>

typedef struct iree_hal_buffer_t iree_hal_buffer_t;

namespace torq {

iree_hal_buffer_t *linux_buffer_retain_hal_buffer(const void *address);

/// Get the DMA-BUF descriptor for an allocation made by the Linux allocator.
/// The descriptor remains owned by the allocator; callers must not close it.
/// @param address: base address of an allocated buffer
/// @return descriptor, or -1 for host, borrowed, or unrecognized memory
int linux_buffer_fd(const void *address);

/// Get the identity of a Linux DMA-BUF allocation.
/// A new allocation gets a new identity even if its address and fd are reused.
/// @param address: base address of an allocated buffer
/// @return allocation identity, or 0 for unrecognized memory
uint64_t linux_buffer_generation(const void *address);

/// Enable CPU access to a Linux DMA-BUF allocation and invalidate cached data.
/// Repeated calls and calls for ordinary host memory are successful no-ops.
/// @param address: base address of an allocated buffer
/// @return true if success
bool linux_buffer_begin_cpu_access(const void *address);

/// End CPU access to a Linux DMA-BUF allocation and flush cached data.
/// Repeated calls and calls for ordinary host memory are successful no-ops.
/// @param address: base address of an allocated buffer
/// @return true if success
bool linux_buffer_end_cpu_access(const void *address);

}