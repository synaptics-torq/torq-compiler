// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/EncodingRequirements.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-encode-tensors"

namespace mlir::syna::torq {

namespace {

class EncodeKernel : public OpInterfaceRewritePattern<torq_hl::KernelInterface> {
  public:
    using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::KernelInterface op, PatternRewriter &rewriter) const {

        // never encode converts themselves
        if (isa<torq_hl::ConvertOp>(op)) {
            return failure();
        }

        auto kernelEncoding = op.getKernelEncoding();

        auto maybeChanged = torq_hl::encodeKernelInputOutputs(op, kernelEncoding, rewriter);

        if (failed(maybeChanged)) {
            llvm::report_fatal_error("cannot encode kernel");
        }

        if (*maybeChanged) {
            rewriter.modifyOpInPlace(op, [&]() {
                EncodingRequirements encoding =
                    toTensorEncodingRequirementsAttr(kernelEncoding.outputEncoding);
                op->setAttr("torq-output-encoding", encoding.toAttr(op.getContext()));
            });
        }

        return *maybeChanged ? success() : failure();
    }
};

class EncodeTensorsPass : public EncodeTensorsBase<EncodeTensorsPass> {
  public:
    using EncodeTensorsBase<EncodeTensorsPass>::EncodeTensorsBase;
    void runOnOperation() override;
};

void EncodeTensorsPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<EncodeKernel>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createEncodeTensorsPass() {
    return std::make_unique<EncodeTensorsPass>();
}

} // namespace mlir::syna::torq
