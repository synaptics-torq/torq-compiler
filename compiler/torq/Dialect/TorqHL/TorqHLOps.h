// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLTraits.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::syna::torq_hl {

void getLayerOpEffects(
    Operation *op, SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
);

} // namespace mlir::syna::torq_hl

#define GET_OP_CLASSES
#include "torq/Dialect/TorqHL/TorqHLOps.h.inc"
