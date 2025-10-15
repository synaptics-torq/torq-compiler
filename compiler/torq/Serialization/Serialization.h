// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma

#include "torq/Dialect/TorqHW/TorqHWOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "torq_executable_def_builder.h"

namespace mlir::syna::torq {

LogicalResult serializeTorqHW(mlir::ModuleOp moduleOp, DenseIntElementsAttr &binaryAttr);

} // namespace mlir::syna::torq
