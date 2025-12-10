// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#ifdef ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/GenericOp.h"
#endif
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-arith-constants"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class ArithConstPattern : public OpRewritePattern<arith::ConstantOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(arith::ConstantOp constOp, PatternRewriter &rewriter) const override {

        auto origType = mlir::dyn_cast<mlir::TensorType>(constOp.getResult().getType());

        // we don't rewrite the index constants or other constants that are not
        // tensors these will be removed from the IR by later passes
        if (!origType) {
            return failure();
        }

        if (origType.getElementType().isIndex()) {
            // This is an index constant, probably coming from a reshape op, don't convert it
            return failure();
        }

#ifdef ENABLE_TORQ_GENERIC
        // don't convert Pvalues
        if (torq_hl::usedOnlyAsPValue(constOp)) {
            return failure();
        }
#else
        // FIXME@Lorenzo: Is this correct?
        if (constOp->getUses().empty()) {
            return failure();
        }
#endif // ENABLE_TORQ_GENERIC

        // create the type for the torq_hl::ConstOp that matches the shape and
        // element type of the arith::ConstantOp
        auto outputType = MemRefType::get(origType.getShape(), origType.getElementType());

        // create a new torq_hl::ConstOp that stores the value, this op returns
        // a memref so we can't use it to substitute the arith::ConstantOp that
        // returns a tensor
        auto torqConstOp = rewriter.create<syna::torq_hl::ConstOp>(
            constOp.getLoc(), outputType, constOp.getValue()
        );

        // replace the arith::ConstantOp with a bufferization::ToTensorOp that
        // transforms the torq_hl::ConstOp to a tensor
        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
            constOp, origType, torqConstOp.getResult(),
            /*restrict=*/true, /*writable=*/false
        );

        return success();
    }
};

class LowerArithConstantsPass : public LowerArithConstantsBase<LowerArithConstantsPass> {
  public:
    LowerArithConstantsPass() = default;
    LowerArithConstantsPass(const LowerArithConstantsPass &pass) {}

    void runOnOperation() override;
};

void LowerArithConstantsPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<ArithConstPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerArithConstantsPass() {
    return std::make_unique<LowerArithConstantsPass>();
}

} // namespace mlir::syna::torq
