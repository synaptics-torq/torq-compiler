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

        RewritePatternSet cpuOverridePatterns(ctx);
        // CPU/NPU Dispatch - MUST be called FIRST
        populateCPUNPUDispatchPatterns(ctx, cpuOverridePatterns);

        auto cpuOverrideFrozenPatterns = FrozenRewritePatternSet(
            std::move(cpuOverridePatterns), disabledPatterns, enabledPatterns
        );

        if (failed(applyPatternsAndFoldGreedily(getOperation(), cpuOverrideFrozenPatterns))) {
            return signalPassFailure();
        }

        llvm::SmallVector<Operation *> torqOps;
        auto identifyTorqOps = [&](Operation *op) {
            if (locHas(op->getLoc(), "affinity_torq")) {
                torqOps.push_back(op);
            }
        };

        getOperation().walk(identifyTorqOps);
        RewritePatternSet patterns(ctx);
        populateLinalgToTorqHLPrePatterns(ctx, patterns, false);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyOpPatternsAndFold(torqOps, frozenPatterns))) {
            return signalPassFailure();
        }
        torqOps.clear();
        getOperation().walk(identifyTorqOps);

        RewritePatternSet lowPrioPatterns(ctx);
        populateLinalgToTorqHLPrePatternsLowPrio(ctx, lowPrioPatterns, false);

        auto lowPrioFrozenPatterns =
            FrozenRewritePatternSet(std::move(lowPrioPatterns), disabledPatterns, enabledPatterns);

        if (failed(applyOpPatternsAndFold(torqOps, lowPrioFrozenPatterns))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgToTorqHLPreConversionPass() {
    return std::make_unique<LinalgToTorqHLPreConversionPass>();
}

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

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTensorToLinalgPass() {
    return std::make_unique<TensorToLinalgPass>();
}

class MarkPatternsForSuperTilingPass
    : public MarkPatternsForSuperTilingBase<MarkPatternsForSuperTilingPass> {
  public:
    using MarkPatternsForSuperTilingBase::MarkPatternsForSuperTilingBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        // TODO(sflur): move to a separate pass?
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

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMarkPatternsForSuperTilingPass() {
    return std::make_unique<MarkPatternsForSuperTilingPass>();
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

class DeviceMapManualOverridePass
    : public DeviceMapManualOverrideBase<DeviceMapManualOverridePass> {
  public:
    using DeviceMapManualOverrideBase::DeviceMapManualOverrideBase;

    static void addTag(Operation *op, StringRef tag) {
        if (locHas(op->getLoc(), tag))
            return; // already tagged
        MLIRContext *ctx = op->getContext();
        auto tagLoc = NameLoc::get(StringAttr::get(ctx, tag), op->getLoc());
        op->setLoc(FusedLoc::get(ctx, {op->getLoc(), tagLoc}));
    }

    void runOnOperation() override {
        auto f = getOperation();
        f.walk([&](Operation *op) {
            auto name = op->getName().getStringRef();
            auto nameAttr = op->getAttrOfType<StringAttr>("name");
            if (nameAttr) {
                StringRef qname = nameAttr.getValue(); // "onnx.Constant"
                name = qname;
            }
            if (name == "onnx.MatMulInteger" || name == "onnx.Sqrt" ||
                name == "onnx.DynamicQuantizeLinear" ||
                name == "onnx.InstanceNormalization" /*|| name == "onnx.Add"*/) {
                addTag(op, "affinity_cpu");
                return;
            }
            else {
                addTag(op, "affinity_torq");
                return;
            }
        });
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createArithToTorqHLConversionPass() {
    return std::make_unique<ArithToTorqHLConversionPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDeviceMapManualOverridePass() {
    return std::make_unique<DeviceMapManualOverridePass>();
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
