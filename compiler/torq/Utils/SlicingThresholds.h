// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>

namespace mlir::syna::torq {

// LinalgSlicingPass splits an op across the NPU slices only when one of its
// operands has at least this many elements.
inline constexpr int64_t kMinElementsForSlicing = 2000;

} // namespace mlir::syna::torq
