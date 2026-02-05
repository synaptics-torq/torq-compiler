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

LogicalResult collapseShapeWithDim(Value &input, int dim, PatternRewriter &rewriter);

LogicalResult collapseValue(
    Value &input, SmallVector<int64_t> &dims, int outputShapeSize, PatternRewriter &rewriter
);

LogicalResult
broadcastInputs(linalg::LinalgOp srcOp, SmallVectorImpl<Value> &inputs, PatternRewriter &rewriter);

LogicalResult promoteScalar(linalg::LinalgOp srcOp, Value &input, PatternRewriter &rewriter);

/// @brief Promote scalar operands to 1D tensors via a symbolic reshape
/// @note Only matches elementwise N-ary ops for now
///
/// The purpose of this promotion is to ensure that ops with scalar operands
/// are picked up by broadcast patterns
///
/// Simlpe example: `tensor<i16>` -> `tensor<1xi16>`
struct PromoteScalarsTo1D : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override;
};

struct ReshapeToCollapseExpand : public OpRewritePattern<tensor::ReshapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::ReshapeOp op, PatternRewriter &rewriter) const override;
};

} // namespace mlir::syna::torq
