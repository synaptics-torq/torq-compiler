// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include <memory>
#include <tuple>

namespace mlir::syna::torq {

class TosaToLinalgConversionPass
    : public impl::TosaToLinalgConversionBase<TosaToLinalgConversionPass> {
  public:
    using TosaToLinalgConversionBase::TosaToLinalgConversionBase;

    void runOnOperation() override {

        auto *context = &getContext();

        ConversionTarget conversionTarget(getContext());

        // do not try to convert these operations, the conversion patterns can generate these
        // types of operations
        conversionTarget.addLegalDialect<
            func::FuncDialect, tensor::TensorDialect, arith::ArithDialect, linalg::LinalgDialect>();

        RewritePatternSet patterns(&getContext());
        populateTOSAToLinalgPatterns(context, patterns);

        // try to convert as many tosa operations (not marked legal) to linalg operations,
        // if an operation is not legal and no pattern matches it, it will be left as is
        if (failed(applyPartialConversion(getOperation(), conversionTarget, std::move(patterns)))) {
            getOperation().emitError() << "conversion failed";
            return signalPassFailure();
        }
    }
};

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Conversions/TosaToLinalg/Passes.h.inc"
} // namespace

void registerTosaToLinalgPasses() {
    // Generated.
    registerPasses();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTosaToLinalgConversionPass() {
    return std::make_unique<TosaToLinalgConversionPass>();
}

} // namespace mlir::syna::torq
