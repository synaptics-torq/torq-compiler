// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-scalars-to-tensors"

namespace mlir::syna::torq {

namespace {

// replace insert(extract(x)) with x for scalar (0-dim) tensors
class TensorizeExtractInsertOp : public OpRewritePattern<tensor::InsertOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::InsertOp op, PatternRewriter &rewriter) const {

        if (op.getType().getRank() != 0) {
            return failure();
        }

        auto sourceExtractOp = op.getScalar().getDefiningOp<tensor::ExtractOp>();

        if (!sourceExtractOp) {
            return failure();
        }

        if (sourceExtractOp.getTensor().getType().getRank() != 0) {
            return failure();
        }

        rewriter.replaceOp(op, sourceExtractOp.getTensor());

        return success();
    }
};

class ScalarsToTensorsPass : public ScalarsToTensorsBase<ScalarsToTensorsPass> {
  public:
    using ScalarsToTensorsBase<ScalarsToTensorsPass>::ScalarsToTensorsBase;
    void runOnOperation() override;
};

void ScalarsToTensorsPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<TensorizeExtractInsertOp>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createScalarsToTensorsPass() {
    return std::make_unique<ScalarsToTensorsPass>();
}

} // namespace mlir::syna::torq
