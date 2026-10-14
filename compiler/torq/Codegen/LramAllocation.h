// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Interfaces/FunctionInterfaces.h"

#include <cstdint>
#include <optional>

namespace mlir::syna::torq {

// The bytes the swaps of `funcOp`'s allocation move, counting each swap-out and each
// swap-in, or nullopt when the allocation fails. The trial does not change `funcOp`.
std::optional<int64_t> lramSwapBytes(FunctionOpInterface funcOp);

// True when the LRAM planner places `funcOp` within `usableBytes` without any swap.
// `funcOp` must already contain the deallocations that define the buffer lifetimes.
bool lramPlacementFitsWithoutSwaps(FunctionOpInterface funcOp, int64_t usableBytes);

} // namespace mlir::syna::torq
