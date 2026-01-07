// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-mark-host-executor"

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clDisableHost;

llvm::cl::opt<bool> clFallbackF32ToHost(
    "torq-fallback-f32-to-host",
    llvm::cl::desc("Fallback to host execution of any operation that uses f32"),
    llvm::cl::init(true)
);

namespace {

class LinalgOpPattern : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  public:
    LinalgOpPattern(MLIRContext *context) : OpInterfaceRewritePattern<linalg::LinalgOp>(context) {}

    LogicalResult
    matchAndRewrite(linalg::LinalgOp srcOp, PatternRewriter &rewriter) const override {

        if (!clFallbackF32ToHost || clDisableHost) {
            return failure();
        }

        if (getTargetExecutor(srcOp) == torq_hl::Executor::Host) {
            return failure();
        }

        if (clFallbackF32ToHost && !isa<linalg::TransposeOp>(srcOp) &&
            !isa<linalg::FillOp>(srcOp)) {

            auto isFp32Op = false;

            for (auto operand : srcOp->getOperands()) {
                if (auto rankedType = dyn_cast<RankedTensorType>(operand.getType())) {
                    if (rankedType.getElementType().isF32()) {
                        isFp32Op = true;
                        break;
                    }
                }
                else if (operand.getType().isF32()) {
                    isFp32Op = true;
                    break;
                }
            }

            if (!isFp32Op) {
                return failure();
            }

            std::string failReason;
            std::string opName;
            int32_t minIntValue = 0;
            int32_t maxIntValue = 0;
            float minFloatValue = 0.0f;
            float maxFloatValue = 0.0f;

            bool canExecuteOnTorq = false;

            // check if the operation can be lowered to a torq kernel
            if (isTorqCastOp(srcOp, opName, failReason)) {
                canExecuteOnTorq = true;
            }
            else if (isTorqNegateOp(srcOp, failReason)) {
                canExecuteOnTorq = true;
            }
            else if (isTorqAbsOp(srcOp, failReason)) {
                canExecuteOnTorq = true;
            }
            else if (isTorqCeilOp(srcOp, failReason)) {
                canExecuteOnTorq = true;
            }
            else if (isTorqClampOp(
                         srcOp, minIntValue, maxIntValue, minFloatValue, maxFloatValue, failReason
                     )) {
                canExecuteOnTorq = true;
            }
            else if (isTorqFloorOp(srcOp, failReason)) {
                canExecuteOnTorq = true;
            }
            else if (isTorqMatMul(srcOp, failReason)) {
                canExecuteOnTorq = true;
            }

            if (canExecuteOnTorq) {
                return failure();
            }

            rewriter.modifyOpInPlace(srcOp, [&]() {
                setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
            });
            return success();
        }

        return failure();
    }
};

class MarkHostExecutorPass : public MarkHostExecutorBase<MarkHostExecutorPass> {
  public:
    MarkHostExecutorPass() = default;
    MarkHostExecutorPass(const MarkHostExecutorPass &pass) {}

    void runOnOperation() override;
};

void MarkHostExecutorPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<LinalgOpPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkHostExecutorPass() {
    return std::make_unique<MarkHostExecutorPass>();
}

} // namespace mlir::syna::torq