// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-compute-const"

namespace mlir::syna::torq {

namespace {

class OpToConstOpRewriter : public RewritePattern {
  public:
    OpToConstOpRewriter(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit*/ 11, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!isCompileTimeConst(op)) {
            return failure();
        }

        auto constAttr = computeValue(op->getResults()[0], true, {});
        if (failed(constAttr)) {
            op->emitError() << "Failed to compute compile-time constant";
            return failure();
        }

        auto constOp = rewriter.create<arith::ConstantOp>(op->getLoc(), *constAttr);
        rewriter.replaceOp(op, constOp);

        return success();
    }
};

} // namespace

// The pass looks for operations marked as compile-time-const and
// replaces them with constant operations computed at compile time.
class CompileTimeConstComputePass
    : public CompileTimeConstComputeBase<CompileTimeConstComputePass> {
  public:
    using CompileTimeConstComputeBase::CompileTimeConstComputeBase;

    void runOnOperation() override {
        SmallVector<Operation *> opsToProcess;
        auto funcOp = getOperation();
        funcOp->walk([&](Operation *op) {
            if (!isCompileTimeConst(op)) {
                return WalkResult::advance();
            }
            opsToProcess.push_back(op);
            return WalkResult::advance();
        });
        RewritePatternSet patterns(&getContext());
        patterns.add<OpToConstOpRewriter>(&getContext());

        if (failed(applyOpPatternsAndFold(opsToProcess, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstComputePass() {
    return std::make_unique<CompileTimeConstComputePass>();
}
} // namespace mlir::syna::torq
