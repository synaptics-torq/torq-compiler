// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::syna::torq {

namespace {

class OptimizeTosaForTorqPass : public impl::OptimizeTosaForTorqBase<OptimizeTosaForTorqPass> {
  public:
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateTransposeConvPhaseConvPatterns(&getContext(), patterns);
        // Only apply the patterns: do not fold, CSE constants or simplify regions in the rest of
        // the function.
        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps)
            .enableFolding(false)
            .enableConstantCSE(false)
            .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
            getOperation().emitError() << "torq-optimize-tosa-for-torq did not converge";
            return signalPassFailure();
        }
    }
};

#define GEN_PASS_REGISTRATION
#include "torq/Transforms/Tosa/Passes.h.inc"

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeTosaForTorqPass() {
    return std::make_unique<OptimizeTosaForTorqPass>();
}

void registerTosaForTorqPasses() { registerPasses(); }

} // namespace mlir::syna::torq
