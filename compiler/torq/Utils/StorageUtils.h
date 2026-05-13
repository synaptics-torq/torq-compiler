// Copyright 2024 Synaptics Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>

namespace mlir::syna {

// Sub-byte integer types are currently stored as one byte per element, except
// i4, which is packed at 4 bits per element.
// TODO: Packing i1 values at 1 bit per element currently regresses because the
// lowerings/kernels expect i1 values to occupy one byte each.
int64_t getStorageBitWidth(int64_t elementBitWidth);

int64_t getStorageSizeBytes(int64_t elementCount, int64_t elementBitWidth);

} // namespace mlir::syna
