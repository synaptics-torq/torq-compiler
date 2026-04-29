// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "OpPatternOptions.h"
#include "PassesDetail.h"
#include "PatternUtils.h"
#include "Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <tuple>

namespace mlir::syna::torq {

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {

    populateLinalgConv2DToTorqHLConv1DPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLConv2DPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLConv2DMatmulPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLPoolingPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLReduceMeanPatterns(context, patterns, markFuseGroups);
}

void populateLinalgToTorqHLPrePatternsLowPrio(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    populateLinalgToTorqHLEWBinaryPatterns(context, patterns, markFuseGroups);
}

static bool isPoolingNhwcSumLegal(Operation *op) {
    auto poolOp = dyn_cast<linalg::PoolingNhwcSumOp>(op);
    if (!poolOp)
        return true;

    auto outType = dyn_cast<RankedTensorType>(poolOp.getResultTensors()[0].getType());
    if (!outType || !outType.getElementType().isInteger())
        return true;

    auto inputType = dyn_cast<RankedTensorType>(poolOp.getInputs()[0].getType());
    auto kernelType = dyn_cast<RankedTensorType>(poolOp.getInputs()[1].getType());
    if (!inputType || !kernelType || kernelType.getRank() != 2)
        return true;

    auto inputShape = inputType.getShape();
    auto kernelShape = kernelType.getShape();
    return kernelShape[0] != inputShape[1] || kernelShape[1] != inputShape[2];
}

static bool isHostOrCssExecutor(Operation *op) {
    auto executor = getTargetExecutor(op, torq_hl::Executor::Slice);
    return executor == torq_hl::Executor::Host || executor == torq_hl::Executor::CSS;
}

class LinalgToTorqHLConversionPass
    : public impl::LinalgToTorqHLConversionBase<LinalgToTorqHLConversionPass> {
  public:
    using LinalgToTorqHLConversionBase::LinalgToTorqHLConversionBase;

    void runOnOperation() override {

        auto *context = &getContext();

        ConversionTarget conversionTarget(getContext());

        conversionTarget.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect>(
        );

        conversionTarget.addDynamicallyLegalDialect<linalg::LinalgDialect>(
            [](Operation *op) -> std::optional<bool> {
                // Linalg will add linalg::YieldOp before doing linalg::TransposeOp
                if (isa<linalg::YieldOp>(op))
                    return true;

                // we do not need to legalize operations that will be executed on the host or css
                if (isHostOrCssExecutor(op))
                    return true;

                // for other operations we don't say anything, so that if a pattern matches we
                // convert, if it doesn't we leave it as is
                return std::nullopt;
            }
        );

        // conversionTargetLinalg.addLegalOp<linalg::YieldOp>();

        // Convert tensor::PadOp to torq_hl::FillOp, but keep pads marked for
        // Host execution so they stay in the tensor dialect and can be included
        // in the Host CPU program by AssignOperationsToCpuProgramsPass.
        conversionTarget.addDynamicallyLegalOp<tensor::PadOp>([](tensor::PadOp padOp) -> bool {
            return isHostOrCssExecutor(padOp);
        });

        RewritePatternSet patterns(&getContext());

        populateLinalgToTorqHLPatterns(context, patterns, false);

        if (failed(applyPartialConversion(getOperation(), conversionTarget, std::move(patterns)))) {
            getOperation().emitError() << "conversion failed";
            return signalPassFailure();
        }

        RewritePatternSet patternsEpilogue(&getContext());

        populateLinalgToTorqHLClampPatterns(context, patternsEpilogue, false);

        if (failed(applyPartialConversion(
                getOperation(), conversionTarget, std::move(patternsEpilogue)
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
    : public impl::LinalgToTorqHLPreConversionBase<LinalgToTorqHLPreConversionPass> {
  public:
    using LinalgToTorqHLPreConversionBase::LinalgToTorqHLPreConversionBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        ConversionTarget target(*ctx);
        target.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect,
            math::MathDialect>();
        target.addLegalOp<linalg::TransposeOp>();
        target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
            [](Operation *op) -> std::optional<bool> {
                if (isa<linalg::YieldOp, linalg::IndexOp>(op))
                    return true;

                // we do not need to legalize operations that will be executed on the host or css
                if (isHostOrCssExecutor(op))
                    return true;

                return std::nullopt;
            }
        );
        target.addDynamicallyLegalOp<linalg::PoolingNhwcSumOp>(
            [](Operation *op) -> std::optional<bool> {
                if (isPoolingNhwcSumLegal(op))
                    return true;
                return std::nullopt;
            }
        );

        // Identity linalg.generic ops (e.g. generalized broadcasts that just
        // yield their input) are legal. Patterns such as EltwiseBinaryConvert
        // create these as intermediate ops during rewriting; under
        // applyPartialConversion they would otherwise fail legalization and
        // cause the entire pattern to be rolled back.
        target.addDynamicallyLegalOp<linalg::GenericOp>(
            [](linalg::GenericOp op) -> std::optional<bool> {
                if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
                    !op.getRegion().hasOneBlock())
                    return std::nullopt;
                auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
                if (!yieldOp || yieldOp.getNumOperands() != 1)
                    return std::nullopt;
                auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
                if (!blockArg || blockArg.getArgNumber() != 0 ||
                    blockArg.getOwner() != op.getBody())
                    return std::nullopt;
                return true;
            }
        );

        RewritePatternSet patterns(ctx);
        populateLinalgToTorqHLPrePatterns(ctx, patterns, false);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }

        RewritePatternSet lowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, lowPrioPatterns, false);

        if (failed(applyPartialConversion(getOperation(), target, std::move(lowPrioPatterns)))) {
            return signalPassFailure();
        }

        mlir::OpPassManager localPm;
        localPm.addPass(mlir::createCanonicalizerPass());
        if (failed(runPipeline(localPm, funcOp))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLPreConversionPass() {
    return std::make_unique<LinalgToTorqHLPreConversionPass>();
}

// DEADCODE:
class TensorToLinalgPass : public impl::TensorToLinalgBase<TensorToLinalgPass> {
  public:
    using TensorToLinalgBase::TensorToLinalgBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        RewritePatternSet patterns(ctx);
        populateTensorToLinalgPatterns(ctx, patterns);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsGreedily(getOperation(), frozenPatterns))) {
            return signalPassFailure();
        }
    }
};

// DEADCODE:
std::unique_ptr<InterfacePass<FunctionOpInterface>> createTensorToLinalgPass() {
    return std::make_unique<TensorToLinalgPass>();
}

class MarkPatternsForTileAndFusePass
    : public impl::MarkPatternsForTileAndFuseBase<MarkPatternsForTileAndFusePass> {
  public:
    using MarkPatternsForTileAndFuseBase::MarkPatternsForTileAndFuseBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = &getContext();

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

        if (failed(applyPatternsGreedily(getOperation(), preFrozenPatterns))) {
            return signalPassFailure();
        }

        RewritePatternSet preLowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, preLowPrioPatterns, true);

        auto preLowPrioFrozenPatterns = FrozenRewritePatternSet(
            std::move(preLowPrioPatterns), disabledPatterns, enabledPatterns
        );

        if (failed(applyPatternsGreedily(getOperation(), preLowPrioFrozenPatterns))) {
            return signalPassFailure();
        }

        RewritePatternSet linalgPatterns(ctx);
        populateLinalgToTorqHLPatterns(ctx, linalgPatterns, true);

        if (failed(applyPatternsGreedily(getOperation(), std::move(linalgPatterns)))) {
            return signalPassFailure();
        }

        RewritePatternSet linalgPatternsEpilogue(&getContext());
        populateLinalgToTorqHLClampPatterns(ctx, linalgPatternsEpilogue, true);

        if (failed(applyPatternsGreedily(getOperation(), std::move(linalgPatternsEpilogue)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkPatternsForTileAndFusePass() {
    return std::make_unique<MarkPatternsForTileAndFusePass>();
}

static bool isArithToTorqHLTarget(linalg::GenericOp op) {
    auto resultType = mlir::dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType)
        return false;
    auto resultElementType = resultType.getElementType();
    if (resultElementType.isF64() || resultElementType.isInteger(64))
        return false;

    for (Value input : op.getInputs()) {
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType)
            continue;
        if (inputType.getElementType().isF64() || inputType.getElementType().isInteger(64))
            return false;
    }

    if (auto binaryOp = getElementwiseBinaryOp(op, /*allowConstants=*/true)) {
        if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>(binaryOp))
            return true;
        if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MinimumFOp, arith::MinSIOp,
                arith::MinUIOp, arith::MinNumFOp, arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp,
                arith::MaxNumFOp, arith::CmpFOp, arith::CmpIOp>(binaryOp))
            return true;
        if (clACTBasedSub && isa<arith::SubIOp>(binaryOp))
            return true;
        if (clACTBasedAdd && isa<arith::AddIOp>(binaryOp))
            return true;
    }

    if (auto ternaryOp = getElementwiseTernaryOp(op, /*allowConstants=*/true)) {
        if (isa<arith::SelectOp>(ternaryOp))
            return true;
    }

    arith::ShRSIOp dummy;
    if (isRoundingRightShiftOp(op, dummy))
        return true;

    // Unary logical not
    if (op.getNumDpsInputs() == 1 && op.getNumDpsInits() == 1) {
        auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
        if (yieldOp && yieldOp.getNumOperands() == 1) {
            if (auto xorOp =
                    dyn_cast_or_null<arith::XOrIOp>(yieldOp.getValues()[0].getDefiningOp())) {
                std::string failReason;
                if (isLogicNotOp(xorOp, failReason))
                    return true;
            }
        }
    }

    return false;
}

class ArithToTorqHLConversionPass
    : public impl::ArithToTorqHLConversionBase<ArithToTorqHLConversionPass> {
  public:
    using ArithToTorqHLConversionBase::ArithToTorqHLConversionBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        ConversionTarget target(*ctx);
        target.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect,
            math::MathDialect>();
        target.addLegalOp<linalg::YieldOp, linalg::IndexOp>();
        target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
            return !isArithToTorqHLTarget(op);
        });

        RewritePatternSet patterns(ctx);
        populateArithToTorqHLPatterns(ctx, patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
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
