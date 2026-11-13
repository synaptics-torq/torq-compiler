// Copyright 2026 Synaptics
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TORQ_ALLOCATOR_PRIVATE_HPP_
#define TORQ_ALLOCATOR_PRIVATE_HPP_

#include "torq/allocator.hpp"

namespace torq {

/// Alignment shared by platform buffers and IREE host allocations.
inline constexpr size_t buffer_alignment = Allocator::alignment;

}

#endif