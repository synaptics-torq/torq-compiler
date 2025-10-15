// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/BuiltinAttributes.h"

#include "torq/Dialect/TorqHL/TorqHLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "torq/Dialect/TorqHL/TorqHLAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "torq/Dialect/TorqHL/TorqHLTypes.h.inc"

namespace mlir::syna::torq_hl {

constexpr int32_t ACT_MIN = -(1 << 15);
constexpr int32_t ACT_MAX = (1 << 15) - 1;

} // namespace mlir::syna::torq_hl
