// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-to-torq-generic"

#include <memory>
#include <tuple>

namespace mlir::syna::torq {

#define GEN_PASS_DECL
#define GEN_PASS_CLASSES
#include "torq/Conversions/LinalgToTorqHL/LinalgToTorqGeneric.h.inc"

namespace {

class SpecializeLinalgGenericOpPass
    : public SpecializeLinalgGenericOpBase<SpecializeLinalgGenericOpPass> {
  public:
    using SpecializeLinalgGenericOpBase<
        SpecializeLinalgGenericOpPass>::SpecializeLinalgGenericOpBase;

    void runOnOperation() override;
};

void SpecializeLinalgGenericOpPass::runOnOperation() {
    auto funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    linalg::populateLinalgGenericOpsSpecializationPatterns(patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
        signalPassFailure();
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSpecializeLinalgGenericOpPass() {
    return std::make_unique<SpecializeLinalgGenericOpPass>();
}

class LinalgToTorqHLGenericPass : public LinalgToTorqHLGenericBase<LinalgToTorqHLGenericPass> {
  public:
    LinalgToTorqHLGenericPass() = default;
    LinalgToTorqHLGenericPass(const LinalgToTorqHLGenericPass &pass) {}
    LinalgToTorqHLGenericPass(const LinalgToTorqHLGenericOptions &options) {
        specializeConstantComputations = options.specializeConstantComputations;
    }

    void runOnOperation() override {

        auto &context = getContext();

        ConversionTarget target(context);

        target.addLegalOp<linalg::YieldOp>();

        target.addLegalDialect<
            torq_hl::TorqHLDialect, func::FuncDialect, tensor::TensorDialect, arith::ArithDialect>(
        );

        RewritePatternSet patterns(&context);

        populateLinalgToAluPatterns(&context, patterns);
        populateLinalgToActPatterns(&context, patterns);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            getOperation().emitError() << "conversion failed";
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgToTorqHLGenericPass(bool specializeConstantComputations) {
    return std::make_unique<LinalgToTorqHLGenericPass>(
        LinalgToTorqHLGenericOptions{specializeConstantComputations}
    );
}

} // namespace mlir::syna::torq
