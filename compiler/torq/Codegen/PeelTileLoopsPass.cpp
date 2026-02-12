// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Utils/ShapeUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/Debug.h"

#include <cassert>

#define DEBUG_TYPE "torq-peel-tile-loops"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class PeelTileLoopsPass : public PeelTileLoopsBase<PeelTileLoopsPass> {
  public:
    PeelTileLoopsPass() = default;
    PeelTileLoopsPass(const PeelTileLoopsPass &pass) {}

    void runOnOperation() override;
};

void PeelLoop(IRRewriter &rewriter, mlir::scf::ForOp forOp) {
    SmallVector<Value> dynamicShapes = torq::collectDynamicShapes(forOp.getRegion());
    if (dynamicShapes.empty()) {
        return;
    }
    LLVM_DEBUG({ llvm::dbgs() << "found dynamic shapes, peeling loop.\n"; });

    // TODO(SF): I suspect we might need to peel the first iteration(s) in
    // somecases, but let's wait until we find an example.
    // One will need to write and call something similar to rewriteAffineOpAfterPeeling
    // but with previousLowerBound instead of upper bound.
    // scf::ForOp firstIteration;
    // if(failed(mlir::scf::peelForLoopFirstIteration(rewriter, forOp, firstIteration))) {
    //     llvm::dbgs() << "peelForLoopFirstIteration failed\n";
    // }

    scf::ForOp lastIteration;
    if (failed(mlir::scf::peelForLoopAndSimplifyBounds(rewriter, forOp, lastIteration))) {
        if (failed(mlir::scf::peelForLoopLastIteration(rewriter, forOp, lastIteration))) {
            forOp->emitWarning("can not peel an scf.for with dynamic shapes.");
        }
    }
}

void PeelTileLoopsPass::runOnOperation() {
    auto funcOp = getOperation();
    // MLIRContext *context = &getContext();
    IRRewriter rewriter(&getContext());

    funcOp->walk([&](mlir::scf::ForOp forOp) { PeelLoop(rewriter, forOp); });

    LLVM_DEBUG({ llvm::dbgs() << "Peel Tile Loops - DONE\n"; });
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPeelTileLoopsPass() {
    return std::make_unique<PeelTileLoopsPass>();
}

} // namespace mlir::syna::torq
