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

// The bytes the swaps of `funcOp`'s LRAM plan transfer, or nullopt when no placement
// exists. Plans as the real pipeline does, within `poolSize` (the whole LRAM when 0),
// without changing `funcOp`, which must already contain its deallocations.
std::optional<int64_t> lramSwapBytes(FunctionOpInterface funcOp, int64_t poolSize = 0);

} // namespace mlir::syna::torq
