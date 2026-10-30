// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::syna::torq {

// Return a collection of all the Values with dynamic shapes in region.
SmallVector<mlir::Value> collectDynamicShapes(mlir::Region &region);

/// Walk forward from val within scopeOp, propagating strictly through scalar
/// or index results. Returns true if any reachable operation produces a dynamic
/// shape tensor result.
bool hasReachableDynamicShape(mlir::Value val, mlir::Operation *scopeOp);

LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, SmallVectorImpl<Value> &inputs, PatternRewriter &rewriter);

/// Returns true if all indexing maps in op are identity-like
/// (identity, or broadcast from a size-1 dimension via constant-zero expr).
bool checkIdentityLikeMaps(linalg::GenericOp op);

/// Emits a value-preserving reshape from `src` to `dstTy` (both static ranked
/// tensor types, same element type and numel):
///   - `src` itself, when the types are equal;
///   - a single tensor.collapse_shape / tensor.expand_shape when a
///     reassociation exists;
///   - a collapse+expand pair otherwise (mid-shape), through the reduced
///     (non-unit) shape when both endpoints reduce identically, else rank-1.
Value emitValuePreservingReshape(
    PatternRewriter &rewriter, Location loc, Value src, RankedTensorType srcTy,
    RankedTensorType dstTy
);

/// Split a keepdims output map into kept positions (AffineDimExpr) and dropped ones
/// (AffineConstantExpr(0) at a unit dim). Fails on any other result form, and when nothing is
/// dropped (the map is already a projected permutation) or nothing is kept.
LogicalResult splitKeepdimsOutputMap(
    AffineMap outMap, RankedTensorType outType, SmallVector<int64_t> &keptPos,
    SmallVector<AffineExpr> &keptExprs
);

/// Reassociation between the keepdims shape and the reduced shape: each kept dim starts a group;
/// dropped unit dims join the current group (or the first, if they precede any kept dim), so
/// every group holds exactly one kept dim.
SmallVector<ReassociationIndices> keepdimsReassociation(ArrayRef<int64_t> keptPos, int64_t rank);

} // namespace mlir::syna::torq
