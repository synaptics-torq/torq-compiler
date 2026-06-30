// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-tensor-concat"

namespace mlir::syna::torq {

namespace {

class DecomposeTensorConcatPass
    : public impl::DecomposeTensorConcatBase<DecomposeTensorConcatPass> {
  public:
    using DecomposeTensorConcatBase::DecomposeTensorConcatBase;

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        tensor::populateDecomposeTensorConcatPatterns(patterns);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDecomposeTensorConcatPass() {
    return std::make_unique<DecomposeTensorConcatPass>();
}

} // namespace mlir::syna::torq
