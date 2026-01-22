// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

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

#define DEBUG_TYPE "torq-unroll-loop"

namespace mlir::syna::torq {

namespace {

static void unrollLoops(scf::ForOp forOp) {

    auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

    if (!lbCstOp || !ubCstOp || !stepCstOp || lbCstOp.value() < 0 || ubCstOp.value() < 0 ||
        stepCstOp.value() < 0) {
        return;
    }

    int64_t tripCount = llvm::divideCeil(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());

    (void)loopUnrollByFactor(forOp, tripCount);
}

class UnrollLoopPass : public UnrollLoopBase<UnrollLoopPass> {
  public:
    using UnrollLoopBase<UnrollLoopPass>::UnrollLoopBase;

    void runOnOperation() override;
};

void UnrollLoopPass::runOnOperation() {
    auto funcOp = getOperation();

    scf::ForOp innermostForOp;

    while (1) {
        funcOp->walk([&](scf::ForOp forOp) {
            auto nestedForOps = forOp.getOps<scf::ForOp>();
            if (nestedForOps.empty()) {
                innermostForOp = forOp;
            }
        });

        if (!innermostForOp) {
            break;
        }

        unrollLoops(innermostForOp);
        innermostForOp = nullptr;
    }

    return;
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollLoopPass() {
    return std::make_unique<UnrollLoopPass>();
}

} // namespace mlir::syna::torq