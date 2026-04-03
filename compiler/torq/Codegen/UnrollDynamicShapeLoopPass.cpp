// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "torq-unroll-dynamic-shape-loop"

namespace mlir::syna::torq {

namespace {

class UnrollDynamicShapeLoopPass
    : public impl::UnrollDynamicShapeLoopBase<UnrollDynamicShapeLoopPass> {
  public:
    using UnrollDynamicShapeLoopBase<UnrollDynamicShapeLoopPass>::UnrollDynamicShapeLoopBase;

    void runOnOperation() override;
};

static void unrollLoop(scf::ForOp forOp) {
    SmallVector<Value> dynamicShapes = torq::collectDynamicShapes(forOp.getRegion());
    if (dynamicShapes.empty()) {
        return;
    }
    LLVM_DEBUG({ llvm::dbgs() << "found dynamic shapes, unrolling loop.\n"; });

    auto lb = forOp.getLoopLowerBounds();
    auto ub = forOp.getLoopUpperBounds();
    auto step = forOp.getLoopSteps();

    if (!lb || lb->size() != 1 || !ub || ub->size() != 1 || !step || step->size() != 1) {
        forOp->emitWarning("scf.for with dynamic loop");
        return;
    }

    std::optional<llvm::APInt> tripCount = constantTripCount(
        lb->back(), ub->back(), step->back(), /*isSigned=*/true, scf::computeUbMinusLb
    );
    if (!tripCount) {
        forOp->emitWarning("scf.for with dynamic loop");
    }

    (void)loopUnrollByFactor(forOp, tripCount->getSExtValue());
}

void UnrollDynamicShapeLoopPass::runOnOperation() {
    auto funcOp = getOperation();

    funcOp->walk(unrollLoop);

    LLVM_DEBUG({ llvm::dbgs() << "Unroll Dynamic Shape Loop - DONE\n"; });
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollDynamicShapeLoopPass() {
    return std::make_unique<UnrollDynamicShapeLoopPass>();
}

} // namespace mlir::syna::torq
