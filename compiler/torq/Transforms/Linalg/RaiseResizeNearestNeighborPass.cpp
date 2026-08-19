// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"

#include "PassesDetail.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::syna::torq {

namespace {

class RaiseResizeNearestNeighborPass
    : public impl::RaiseResizeNearestNeighborBase<RaiseResizeNearestNeighborPass> {
  public:
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        populateResizeNearestNeighborRaisingPatterns(&getContext(), patterns);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

} // anonymous namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRaiseResizeNearestNeighborPass() {
    return std::make_unique<RaiseResizeNearestNeighborPass>();
}

} // namespace mlir::syna::torq
