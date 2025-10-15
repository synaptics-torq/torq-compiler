// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"
#ifdef ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/GenericOp.h"
#endif // ENABLE_TORQ_GENERIC

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-fold-pvalue-inits"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class PInitPattern : public OpRewritePattern<linalg::FillOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::FillOp fillOp, PatternRewriter &rewriter) const override {

        if (fillOp.getNumResults() != 1) {
            return failure();
        }

#ifdef ENABLE_TORQ_GENERIC
        if (!torq_hl::usedOnlyAsPValue(fillOp.getResult(0))) {
            return failure();
        }
#else
        // FIXME@Lorenzo: What to do here?
        // if (fillOp.getResult(0).getUses().empty()) {
        //     return success();
        // }

#endif // ENABLE_TORQ_GENERIC

        auto fillOpType = dyn_cast<RankedTensorType>(fillOp.getType(0));

        if (!fillOpType) {
            return failure();
        }

        auto fillConst = fillOp.value().getDefiningOp<arith::ConstantOp>();

        if (!fillConst) {
            return failure();
        }

        auto fillValue = dyn_cast<IntegerAttr>(fillConst.getValue());

        if (!fillValue) {
            return failure();
        }

        auto pType = RankedTensorType::get(fillOpType.getShape(), rewriter.getI32Type());

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            fillOp, SplatElementsAttr::get(pType, fillValue)
        );

        return success();
    }
};

class FoldPValueInitsPass : public FoldPValueInitsBase<FoldPValueInitsPass> {
  public:
    FoldPValueInitsPass() = default;
    FoldPValueInitsPass(const FoldPValueInitsPass &pass) {}

    void runOnOperation() override;
};

void FoldPValueInitsPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<PInitPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldPValueInitsPass() {
    return std::make_unique<FoldPValueInitsPass>();
}

} // namespace mlir::syna::torq
