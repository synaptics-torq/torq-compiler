// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

llvm::cl::opt<bool> clUseNewKernels(
    "torq-use-new-kernels", llvm::cl::desc("Use new kernels for TorqHL to TorqHW conversion"),
    llvm::cl::init(false)
);

namespace {

class ConvertSliceProgramToTorqHwPass
    : public ConvertSliceProgramToTorqHwBase<ConvertSliceProgramToTorqHwPass> {
  public:
    using ConvertSliceProgramToTorqHwBase::ConvertSliceProgramToTorqHwBase;
    void runOnOperation() override;
};

void ConvertSliceProgramToTorqHwPass::runOnOperation() {
    auto programOp = getOperation();

    if (programOp.getProgram().getType().getExecutor() != torq_hl::Executor::Slice) {
        return;
    }

    MLIRContext *ctx = programOp.getContext();

    ConversionTarget conversionTarget(*ctx);
    conversionTarget.addLegalDialect<torq_hw::TorqHWDialect>();
    conversionTarget.addLegalOp<torq_hl::ProgramOp, torq_hl::ReturnOp>();

    RewritePatternSet patterns(ctx);

    patterns.add<AddPattern>(ctx);
    patterns.add<Conv2DPattern>(ctx);
    patterns.add<Avgpool2DPattern>(ctx);
    patterns.add<MulPattern>(ctx);
    patterns.add<DWPattern>(ctx);
    patterns.add<FCPattern>(ctx);
    patterns.add<SegmentationPattern>(ctx);
    patterns.add<TransposePattern>(ctx);
    patterns.add<MaxPool2dPattern>(ctx);
    patterns.add<MatMulPattern>(ctx);
    patterns.add<GatherPattern>(ctx);
    patterns.add<IdentityPattern>(ctx);
    patterns.add<TablePattern>(ctx);
    patterns.add<ArgMaxPattern>(ctx);
    patterns.add<TransposeReshapePattern>(ctx);
    patterns.add<Conv1DPattern>(ctx);

    patterns.add<ConvertPattern>(ctx);
    patterns.add<FMAPattern>(ctx);
    patterns.add<FillPattern>(ctx);
    patterns.add<ReducePattern>(ctx);
    patterns.add<ScatterPattern>(ctx);

    patterns.add<ActPattern>(ctx);
    patterns.add<BroadcastPattern>(ctx);
    patterns.add<ResizeNearestNeighborPattern>(ctx);
    patterns.add<ElementWiseBinaryPattern>(ctx);
    patterns.add<ElementWiseUnaryPattern>(ctx);
    patterns.add<ElementWiseShiftPattern>(ctx);
    patterns.add<DepthToSpacePattern>(ctx);
    patterns.add<ReduceMeanPattern>(ctx);
    patterns.add<SelectPattern>(ctx);

    if (failed(applyFullConversion(getOperation(), conversionTarget, std::move(patterns)))) {
        getOperation().emitError() << "conversion failed";
        return signalPassFailure();
    }
}

class ConvertNssProgramToTorqHwPass
    : public ConvertNssProgramToTorqHwBase<ConvertNssProgramToTorqHwPass> {
  public:
    using ConvertNssProgramToTorqHwBase::ConvertNssProgramToTorqHwBase;
    void runOnOperation() override;
};

void ConvertNssProgramToTorqHwPass::runOnOperation() {
    auto programOp = getOperation();

    if (programOp.getProgram().getType().getExecutor() != torq_hl::Executor::NSS) {
        return;
    }

    MLIRContext *ctx = programOp.getContext();

    ConversionTarget conversionTarget(*ctx);
    conversionTarget.addLegalDialect<torq_hw::TorqHWDialect>();
    conversionTarget
        .addLegalOp<torq_hl::ProgramOp, torq_hl::ReturnOp, memref::AllocOp, memref::DeallocOp>();

    RewritePatternSet patterns(ctx);

    populateNssTaskPatterns(ctx, patterns);

    if (failed(applyFullConversion(getOperation(), conversionTarget, std::move(patterns)))) {
        getOperation().emitError() << "conversion failed";
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<OperationPass<torq_hl::ProgramOp>> createConvertSliceProgramToTorqHwPass() {
    return std::make_unique<ConvertSliceProgramToTorqHwPass>();
}

std::unique_ptr<OperationPass<torq_hl::ProgramOp>> createConvertNssProgramToTorqHwPass() {
    return std::make_unique<ConvertNssProgramToTorqHwPass>();
}

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Conversions/TorqHLToTorqHW/Passes.h.inc"
} // namespace

void registerTorqHLToTorqHWPasses() {
    // Generated.
    registerPasses();
}

} // namespace mlir::syna::torq
