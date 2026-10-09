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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

/// Rewrites a ranked tensor.bitcast into a torq_hl.bitcast (same source and
/// result types). The torq op carries the bufferization interface that turns
/// it into a free torq_hl.reinterpret_cast view.
class TensorBitcastToTorqBitcastPattern : public OpRewritePattern<tensor::BitcastOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor::BitcastOp op, PatternRewriter &rewriter) const override {
        auto resultType = dyn_cast<RankedTensorType>(op.getType());
        if (!resultType)
            return rewriter.notifyMatchFailure(op, "expected a ranked tensor.bitcast");
        rewriter.replaceOpWithNewOp<torq_hl::BitcastOp>(op, resultType, op.getSource());
        return success();
    }
};

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {

    populateLinalgConv2DToTorqHLConv1DPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLConv2DPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLQConv2DPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQMatmulPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQEWBinaryPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQPoolingPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQMulDivPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQSigmoidPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQReducePatterns(context, patterns, markFuseGroups);

    // Mixed-quant (block-dequant) matmuls convert before the Q patterns can
    // anchor on the dequant generic.
    populateLinalgToTorqHLMixedQuantMatmulPatterns(context, patterns, markFuseGroups);

    // Conv1D-as-matmul → fully_connected with absorbed bias. Registered before
    // the generic Conv2DMatmulOpConversion; its higher pattern benefit lets it
    // win when the matmul has the Conv1D transpose/expand layout.
    populateLinalgToTorqHLConv1DMatmulPatterns(context, patterns, markFuseGroups);

    // Claim the raised batch matmul and its epilogue before standalone cast/mul marking.
    populateLinalgToTorqHLBatchedRaisedMatmulPatterns(context, patterns, markFuseGroups);

    // The raised MatMulInteger's i32 zero-point correction plus its int-to-float cast.
    populateLinalgToTorqHLMatMulIntegerCorrectPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLConv2DMatmulPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLPoolingPatterns(context, patterns, markFuseGroups);

    populateLinalgToTorqHLReduceMeanPatterns(context, patterns, markFuseGroups);
}

void populateLinalgToTorqHLPrePatternsLowPrio(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    populateLinalgToTorqHLEWBinaryPatterns(context, patterns, markFuseGroups);
}

// TODO: When PoolingNhwcSumOp supports BF16, the AllowBF16 template parameter
// can be removed and BF16 can always be allowed.
template <typename PoolingOpT, bool AllowBF16> static bool isPoolingSumLegal(Operation *op) {
    auto poolOp = dyn_cast<PoolingOpT>(op);
    if (!poolOp)
        return true;

    auto outType = dyn_cast<RankedTensorType>(poolOp.getResultTensors()[0].getType());
    if (!outType)
        return true;
    bool typeOk = outType.getElementType().isInteger();
    if constexpr (AllowBF16)
        typeOk = typeOk || outType.getElementType().isBF16();

    // Quantized AveragePool/GlobalAveragePool chains lower the pool itself to
    // f32 because it sits between DQ and Q generics.  Let the Q fusion patterns
    // handle NCHW f32 pooling for both global and local cases.
    if constexpr (std::is_same_v<PoolingOpT, linalg::PoolingNchwSumOp>) {
        if (outType.getElementType().isF32())
            typeOk = true;
    }

    if (!typeOk)
        return true;

    auto inputType = dyn_cast<RankedTensorType>(poolOp.getInputs()[0].getType());
    auto kernelType = dyn_cast<RankedTensorType>(poolOp.getInputs()[1].getType());
    if (!inputType || !kernelType || kernelType.getRank() != 2)
        return true;

    auto inputShape = inputType.getShape();
    auto kernelShape = kernelType.getShape();

    // NHWC: [N, H, W, C] : 1, 2
    // NCHW: [N, C, H, W] : 2, 3
    size_t hIdx, wIdx;
    if constexpr (std::is_same_v<PoolingOpT, linalg::PoolingNhwcSumOp>) {
        hIdx = 1;
        wIdx = 2;
    }
    else {
        hIdx = 2;
        wIdx = 3;
    }

    // For BF16, always let patterns attempt conversion (global pooling is handled
    // by PoolingNchwSumOpConversion, local pooling by PoolingNchwSumOpToDW2DConversion).
    if (outType.getElementType().isBF16())
        return false;

    // Quantized AveragePool/GlobalAveragePool chains lower the pool itself to
    // f32 because it sits between DQ and Q generics.  Mark NCHW f32 pooling
    // illegal so the Q fusion patterns can convert both global and local cases.
    if constexpr (std::is_same_v<PoolingOpT, linalg::PoolingNchwSumOp>) {
        if (outType.getElementType().isF32())
            return false;
    }

    // For integer types, only global pooling (kernel == whole frame) is supported.
    return kernelShape[0] != inputShape[hIdx] || kernelShape[1] != inputShape[wIdx];
}

static bool isHostOrCssExecutor(Operation *op) {
    auto executor = getTargetExecutor(op, torq_hl::Executor::Slice);
    return executor == torq_hl::Executor::Host || executor == torq_hl::Executor::CSS;
}

static LogicalResult validateRaisedMatmulAttributeContract(FunctionOpInterface funcOp) {
    // The raise emits both linalg.matmul and linalg.batch_matmul (the per-channel depthwise
    // GEMV keeps its batch dimension), so both must honour the contract.
    WalkResult result = funcOp.walk([](Operation *op) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(op))
            return WalkResult::advance();
        Attribute raisedAttr = op->getAttr(TORQ_RAISED_MATMUL_INTEGER);
        Attribute unsignedAttr = op->getAttr(TORQ_MATMUL_LHS_UNSIGNED);
        if (!raisedAttr && !unsignedAttr)
            return WalkResult::advance();
        if (isa_and_nonnull<UnitAttr>(raisedAttr) && isa_and_nonnull<UnitAttr>(unsignedAttr))
            return WalkResult::advance();

        op->emitOpError() << "raised MatMulInteger attribute contract violated: '"
                          << TORQ_RAISED_MATMUL_INTEGER << "' and '" << TORQ_MATMUL_LHS_UNSIGNED
                          << "' must both be present as UnitAttr attributes";
        return WalkResult::interrupt();
    });
    return result.wasInterrupted() ? failure() : success();
}

class LinalgToTorqHLConversionPass
    : public impl::LinalgToTorqHLConversionBase<LinalgToTorqHLConversionPass> {
  public:
    using LinalgToTorqHLConversionBase::LinalgToTorqHLConversionBase;

    void runOnOperation() override {

        if (failed(validateRaisedMatmulAttributeContract(getOperation()))) {
            return signalPassFailure();
        }

        auto *context = &getContext();

        ConversionTarget conversionTarget(getContext());

        conversionTarget.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect,
            affine::AffineDialect>();

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

        // Keep any op routed to Host/CSS legal in this conversion. These ops
        // will be outlined to CPU programs and don't need TorqHL lowering.
        conversionTarget.markUnknownOpDynamicallyLegal([](Operation *op) -> std::optional<bool> {
            if (!op->hasAttr("torq-executor"))
                return std::nullopt;
            if (isHostOrCssExecutor(op))
                return true;
            return std::nullopt;
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

        // Any remaining tensor.bitcast (from the linalg decompositions or the
        // frontend) becomes a torq_hl.bitcast here, so it bufferizes as a free
        // torq_hl.reinterpret_cast view instead of a data-movement slice. This
        // runs after tiling, where tensor.bitcast survives untouched.
        {
            RewritePatternSet bitcastPatterns(ctx);
            bitcastPatterns.add<TensorBitcastToTorqBitcastPattern>(ctx);
            if (failed(applyPatternsGreedily(funcOp, std::move(bitcastPatterns)))) {
                return signalPassFailure();
            }
        }

        if (failed(validateRaisedMatmulAttributeContract(funcOp))) {
            return signalPassFailure();
        }

        ConversionTarget target(*ctx);
        target.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect,
            math::MathDialect, affine::AffineDialect>();
        target.addLegalOp<linalg::TransposeOp>();
        // Index arithmetic on tile offsets (same form the tiling passes emit);
        // patterns composing slice offsets may create these.
        target.addLegalOp<affine::AffineApplyOp>();
        target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
            [](Operation *op) -> std::optional<bool> {
                // we do not need to legalize operations that will be executed on the host or css
                if (isHostOrCssExecutor(op))
                    return true;

                return llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
                    .Case<linalg::YieldOp, linalg::IndexOp>([](auto) { return true; })
                    .Case<linalg::PoolingNhwcSumOp>([&](auto) -> std::optional<bool> {
                        if (isPoolingSumLegal<linalg::PoolingNhwcSumOp, false>(op))
                            return true;
                        return std::nullopt;
                    })
                    .Case<linalg::PoolingNchwSumOp>([&](auto) -> std::optional<bool> {
                        if (isPoolingSumLegal<linalg::PoolingNchwSumOp, true>(op))
                            return true;
                        return std::nullopt;
                    })
                    .Default([](Operation *) { return std::nullopt; });
            }
        );

        // Identity linalg.generic ops (e.g. generalized broadcasts that just
        // yield their input) are legal. Patterns such as EltwiseBinaryConvert
        // create these as intermediate ops during rewriting; under
        // applyPartialConversion they would otherwise fail legalization and
        // cause the entire pattern to be rolled back.
        target.addDynamicallyLegalOp<linalg::GenericOp>(
            [](linalg::GenericOp op) -> std::optional<bool> {
                // QConv2D synthesizes Host-routed generics (zp-correction
                // reduce/mul, bias+correction add) that are resolved later by
                // CompileTimeConstComputePass.  These are the only Host-marked
                // linalg.generic ops in the pre-conversion target.
                if (isHostOrCssExecutor(op))
                    return true;
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

class MarkPatternsForTileAndFusePass
    : public impl::MarkPatternsForTileAndFuseBase<MarkPatternsForTileAndFusePass> {
  public:
    using MarkPatternsForTileAndFuseBase::MarkPatternsForTileAndFuseBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = &getContext();

        if (failed(validateRaisedMatmulAttributeContract(funcOp))) {
            return signalPassFailure();
        }

        // Assign UIDs to TilingInterface operations
        OpBuilder builder(funcOp);
        int64_t nextId = 0;
        funcOp->walk([&](TilingInterface op) {
            op->setAttr(mlir::syna::torq::TORQ_FUSE_GROUP_ID, builder.getI64IntegerAttr(nextId++));
        });

        // Large models (e.g. NNNR3) mark many fuse groups; allow more than the
        // default 10 greedy iterations while still failing on true rewrite loops.
        constexpr unsigned kMaxMarkIterations = 50;
        GreedyRewriteConfig config;
        config.setMaxIterations(kMaxMarkIterations);

        RewritePatternSet prePatterns(ctx);
        populateLinalgToTorqHLPrePatterns(ctx, prePatterns, true);

        auto preFrozenPatterns =
            FrozenRewritePatternSet(std::move(prePatterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsGreedily(getOperation(), preFrozenPatterns, config))) {
            getOperation()->emitError()
                << "torq-mark-patterns-for-tile-and-fuse: pre-patterns did not converge";
            return signalPassFailure();
        }

        RewritePatternSet preLowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, preLowPrioPatterns, true);

        auto preLowPrioFrozenPatterns = FrozenRewritePatternSet(
            std::move(preLowPrioPatterns), disabledPatterns, enabledPatterns
        );

        if (failed(applyPatternsGreedily(getOperation(), preLowPrioFrozenPatterns, config))) {
            getOperation()->emitError()
                << "torq-mark-patterns-for-tile-and-fuse: low-prio patterns did not converge";
            return signalPassFailure();
        }

        RewritePatternSet linalgPatterns(ctx);
        populateLinalgToTorqHLPatterns(ctx, linalgPatterns, true);

        if (failed(applyPatternsGreedily(getOperation(), std::move(linalgPatterns), config))) {
            getOperation()->emitError()
                << "torq-mark-patterns-for-tile-and-fuse: linalg patterns did not converge";
            return signalPassFailure();
        }

        RewritePatternSet linalgPatternsEpilogue(&getContext());
        populateLinalgToTorqHLClampPatterns(ctx, linalgPatternsEpilogue, true);

        if (failed(applyPatternsGreedily(getOperation(), std::move(linalgPatternsEpilogue), config)
            )) {
            getOperation()->emitError()
                << "torq-mark-patterns-for-tile-and-fuse: clamp patterns did not converge";
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
