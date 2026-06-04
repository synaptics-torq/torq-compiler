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

LogicalResult collapseShapeWithDim(Value &input, int dim, PatternRewriter &rewriter);

LogicalResult collapseValue(
    Value &input, SmallVector<int64_t> &dims, int outputShapeSize, PatternRewriter &rewriter
);

LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, SmallVectorImpl<Value> &inputs, PatternRewriter &rewriter);

LogicalResult promoteScalar(linalg::LinalgOp srcOp, Value &input, PatternRewriter &rewriter);

/// Returns true if all indexing maps in op are identity-like
/// (identity, or broadcast from a size-1 dimension via constant-zero expr).
bool checkIdentityLikeMaps(linalg::GenericOp op);

} // namespace mlir::syna::torq
