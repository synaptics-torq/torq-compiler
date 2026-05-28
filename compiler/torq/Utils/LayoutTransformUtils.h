// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Layout Transformation Utilities
//===----------------------------------------------------------------------===//
//
// Shared utilities for NHWC↔NCHW layout transformations.
// Used by: ConvertNhwcOpToNchwPass, OptimizeTransposeLayoutPass
//
//===----------------------------------------------------------------------===//

#ifndef TORQ_UTILS_LAYOUTTRANSFORMUTILS_H
#define TORQ_UTILS_LAYOUTTRANSFORMUTILS_H

#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Permutation Utilities
//===----------------------------------------------------------------------===//

/// Permute shape from NHWC to NCHW.
/// Supports 3D (HWC→CHW) and 4D (NHWC→NCHW) tensors.
SmallVector<int64_t> nhwcToNchwShape(ArrayRef<int64_t> shape);

/// Adapt 4D layout permutation to arbitrary rank (handles 3D/5D tensors).
/// Keeps leading batch dims unchanged, applies layout swap to last 3 dims.
/// Examples: rank=3: [2,0,1] (HWC→CHW), rank=4: [0,3,1,2], rank=5: [0,1,4,2,3]
SmallVector<int64_t> adaptPermToRank(ArrayRef<int64_t> perm4D, int64_t rank);

//===----------------------------------------------------------------------===//
// Layout Conversion Checks
//===----------------------------------------------------------------------===//

/// Check if permutation is NCHW→NHWC [0,2,3,1].
inline bool isNchwToNhwcTranspose(ArrayRef<int64_t> perm) {
    return Permutation(perm.begin(), perm.end()) == Permutation::nchw2nhwc();
}

/// Check if permutation is NHWC→NCHW [0,3,1,2].
inline bool isNhwcToNchwTranspose(ArrayRef<int64_t> perm) {
    return Permutation(perm.begin(), perm.end()) == Permutation::nhwc2nchw();
}

/// Check if transpose op is a 4D NCHW↔NHWC layout conversion.
bool isLayoutConversionTranspose(linalg::TransposeOp transposeOp);

//===----------------------------------------------------------------------===//
// Tensor Utilities
//===----------------------------------------------------------------------===//

/// Create a zero-filled tensor of the given shape and element type.
Value createZeroFilledTensor(
    OpBuilder &rewriter, Location loc, ArrayRef<int64_t> shape, Type elemType
);

/// Create a min-filled tensor of the given shape and element type.
Value createMinFilledTensor(
    OpBuilder &rewriter, Location loc, ArrayRef<int64_t> shape, Type elemType
);

/// Rebuild a linalg.generic with new inputs, output init, and indexing maps.
/// The body region is cloned verbatim (element types must be unchanged).
Value rebuildGenericWithNewLayout(
    OpBuilder &rewriter, linalg::GenericOp origOp, ValueRange newInputs, Value newOutputInit,
    ArrayRef<AffineMap> newMaps
);

} // namespace mlir::syna::torq

#endif // TORQ_UTILS_LAYOUTTRANSFORMUTILS_H
