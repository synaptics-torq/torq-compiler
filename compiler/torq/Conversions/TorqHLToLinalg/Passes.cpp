// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Passes.h"
#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"

namespace mlir::syna::torq {

class TorqHLToLinalgConversionPass
    : public impl::TorqHLToLinalgConversionBase<TorqHLToLinalgConversionPass> {
  public:
    using TorqHLToLinalgConversionBase::TorqHLToLinalgConversionBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        TypeConverter typeConverter;
        typeConverter.addConversion([](Type type) { return type; });

        ConversionTarget target(*ctx);
        target.addLegalDialect<linalg::LinalgDialect>();
        target.addIllegalDialect<torq_hl::TorqHLDialect>();

        RewritePatternSet patterns(ctx);
        populateTorqHLToLinalgPatterns(ctx, patterns, typeConverter);

        if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLToLinalgConversionPass() {
    return std::make_unique<TorqHLToLinalgConversionPass>();
}

namespace {
#define GEN_PASS_REGISTRATION
#include "torq/Conversions/TorqHLToLinalg/Passes.h.inc"
} // namespace

void registerTorqHLToLinalgPasses() { registerPasses(); }

} // namespace mlir::syna::torq
