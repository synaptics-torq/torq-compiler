// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-convert-tensor-pad-to-linalg"

namespace mlir::syna::torq {

namespace {

class ConvertTensorPadToLinalgPass
    : public impl::ConvertTensorPadToLinalgBase<ConvertTensorPadToLinalgPass> {
  public:
    using ConvertTensorPadToLinalgBase::ConvertTensorPadToLinalgBase;

    void runOnOperation() override {
        func::FuncOp func = getOperation();
        SmallVector<tensor::PadOp> pads;
        func.walk([&](tensor::PadOp padOp) { pads.push_back(padOp); });

        IRRewriter rewriter(&getContext());
        for (tensor::PadOp padOp : pads) {
            rewriter.setInsertionPoint(padOp);
            if (failed(linalg::rewriteInDestinationPassingStyle(rewriter, padOp))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createConvertTensorPadToLinalgPass() {
    return std::make_unique<ConvertTensorPadToLinalgPass>();
}

} // namespace mlir::syna::torq
