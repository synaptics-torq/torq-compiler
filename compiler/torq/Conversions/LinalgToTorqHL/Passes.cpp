// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <tuple>

namespace mlir::syna::torq {

class LinalgToTorqHLConversionPass
    : public LinalgToTorqHLConversionBase<LinalgToTorqHLConversionPass> {
  public:
    using LinalgToTorqHLConversionBase::LinalgToTorqHLConversionBase;

    void runOnOperation() override {

        auto *context = &getContext();

        ConversionTarget conversionTargetLinalg(getContext());

        conversionTargetLinalg.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect>(
        );

        conversionTargetLinalg.addDynamicallyLegalDialect<linalg::LinalgDialect>(
            [](Operation *op) -> std::optional<bool> {
                // Linalg will add linalg::YieldOp before doing linalg::TransposeOp
                if (isa<linalg::YieldOp>(op))
                    return true;

                // we do not need to legalize operations that will be executed on the host
                if (getTargetExecutor(op, torq_hl::Executor::Slice) == torq_hl::Executor::Host) {
                    return true;
                }

                // for other operations we don't say anything, so that if a pattern matches we
                // convert, if it doesn't we leave it as is
                return std::nullopt;
            }
        );

        // conversionTargetLinalg.addLegalOp<linalg::YieldOp>();

        // Make sure all tensor::PadOp are converted to linalg::FillOp so that they can be converted
        // to torq_hl::FillOp
        conversionTargetLinalg.addIllegalOp<tensor::PadOp>();

        RewritePatternSet patternsLinalg(&getContext());

        populateLinalgToTorqHLPatterns(context, patternsLinalg, false);

        if (failed(applyPartialConversion(
                getOperation(), conversionTargetLinalg, std::move(patternsLinalg)
            ))) {
            getOperation().emitError() << "conversion failed";
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLConversionPass() {
    return std::make_unique<LinalgToTorqHLConversionPass>();
}

class LinalgToTorqHLPreConversionPass
    : public LinalgToTorqHLPreConversionBase<LinalgToTorqHLPreConversionPass> {
  public:
    using LinalgToTorqHLPreConversionBase::LinalgToTorqHLPreConversionBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();
        RewritePatternSet patterns(ctx);
        populateLinalgToTorqHLPrePatterns(ctx, patterns, false);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
            return signalPassFailure();
        }

        RewritePatternSet lowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, lowPrioPatterns, false);

        auto lowPrioFrozenPatterns =
            FrozenRewritePatternSet(std::move(lowPrioPatterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), lowPrioFrozenPatterns))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLPreConversionPass() {
    return std::make_unique<LinalgToTorqHLPreConversionPass>();
}

// DEADCODE:
class TensorToLinalgPass : public TensorToLinalgBase<TensorToLinalgPass> {
  public:
    using TensorToLinalgBase::TensorToLinalgBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        RewritePatternSet patterns(ctx);
        populateTensorToLinalgPatterns(ctx, patterns);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns))) {
            return signalPassFailure();
        }
    }
};

// DEADCODE:
std::unique_ptr<InterfacePass<FunctionOpInterface>> createTensorToLinalgPass() {
    return std::make_unique<TensorToLinalgPass>();
}

class MarkPatternsForTileAndFusePass
    : public MarkPatternsForTileAndFuseBase<MarkPatternsForTileAndFusePass> {
  public:
    using MarkPatternsForTileAndFuseBase::MarkPatternsForTileAndFuseBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        // Assign UIDs to TilingInterface operations
        OpBuilder builder(funcOp);
        int64_t nextId = 0;
        funcOp->walk([&](TilingInterface op) {
            op->setAttr(mlir::syna::torq::TORQ_FUSE_GROUP_ID, builder.getI64IntegerAttr(nextId++));
        });

        RewritePatternSet prePatterns(ctx);
        populateLinalgToTorqHLPrePatterns(ctx, prePatterns, true);

        auto preFrozenPatterns =
            FrozenRewritePatternSet(std::move(prePatterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), preFrozenPatterns))) {
            return signalPassFailure();
        }

        RewritePatternSet preLowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, preLowPrioPatterns, true);

        auto preLowPrioFrozenPatterns = FrozenRewritePatternSet(
            std::move(preLowPrioPatterns), disabledPatterns, enabledPatterns
        );

        if (failed(applyPatternsAndFoldGreedily(getOperation(), preLowPrioFrozenPatterns))) {
            return signalPassFailure();
        }

        RewritePatternSet linalgPatterns(ctx);
        populateLinalgToTorqHLPatterns(ctx, linalgPatterns, true);

        auto linalgFrozenPatterns =
            FrozenRewritePatternSet(std::move(linalgPatterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), linalgFrozenPatterns))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkPatternsForTileAndFusePass() {
    return std::make_unique<MarkPatternsForTileAndFusePass>();
}

class ArithToTorqHLConversionPass
    : public ArithToTorqHLConversionBase<ArithToTorqHLConversionPass> {
  public:
    using ArithToTorqHLConversionBase::ArithToTorqHLConversionBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();
        RewritePatternSet patterns(ctx);
        populateArithToTorqHLPatterns(ctx, patterns);
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createArithToTorqHLConversionPass() {
    return std::make_unique<ArithToTorqHLConversionPass>();
}

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Conversions/LinalgToTorqHL/Passes.h.inc"
} // namespace

void registerLinalgToTorqHLPasses() {
    // Generated.
    registerPasses();
}

} // namespace mlir::syna::torq
