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

// The bytes of the LRAM buffers that the allocation of `funcOp` spills or relocates,
// or nullopt when the allocation fails. The trial does not change `funcOp`.
std::optional<int64_t> lramSwapBytes(FunctionOpInterface funcOp);

} // namespace mlir::syna::torq
