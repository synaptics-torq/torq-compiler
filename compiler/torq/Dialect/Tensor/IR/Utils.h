// Copyright 2025 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::syna::torq::tensor {

std::optional<linalg::SliceParameters> computeCollapseSliceParameters(
    OpBuilder &builder, ::mlir::tensor::CollapseShapeOp collapseOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
);

std::optional<linalg::SliceParameters> computeExpandSliceParameters(
    OpBuilder &builder, ::mlir::tensor::ExpandShapeOp expandOp, ArrayRef<OpFoldResult> ivs,
    ArrayRef<OpFoldResult> tileSizes, ArrayRef<OpFoldResult> sizeBounds, bool omitPartialTileCheck
);

} // namespace mlir::syna::torq::tensor
