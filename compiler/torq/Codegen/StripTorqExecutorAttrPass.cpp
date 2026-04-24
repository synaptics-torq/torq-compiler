// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"

#define DEBUG_TYPE "torq-strip-executor-attr"

namespace mlir::syna::torq {

namespace {

struct StripTorqExecutorAttrPass
    : public impl::StripTorqExecutorAttrBase<StripTorqExecutorAttrPass> {

    void runOnOperation() override {
        getOperation()->walk([](Operation *op) {
            if (isa<torq_hl::TorqHLDialect>(op->getDialect())) {
                op->removeAttr("torq-executor");
            }
        });
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createStripTorqExecutorAttrPass() {
    return std::make_unique<StripTorqExecutorAttrPass>();
}

} // namespace mlir::syna::torq
