// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>

#define DEBUG_TYPE "torq-peel-tile-loops"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// Taken from mlir/lib/Dialect/SCF/Utils/AffineCanonicalizationUtils.cpp, as it
// is not publicly visible there.
// Given some affine constraints, simplify the min/max op, or return failure.
static FailureOr<affine::AffineApplyOp> canonicalizeMinMaxOp(
    RewriterBase &rewriter, Operation *op, affine::FlatAffineValueConstraints constraints
) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    FailureOr<affine::AffineValueMap> simplified =
        affine::simplifyConstrainedMinMaxOp(op, std::move(constraints));
    if (failed(simplified))
        return failure();
    return rewriter.replaceOpWithNewOp<affine::AffineApplyOp>(
        op, simplified->getAffineMap(), simplified->getOperands()
    );
}

// Simplify each affine min/max op in forOp, using the loop bounds.
static void rewriteAffineOpInLoop(RewriterBase &rewriter, scf::ForOp forOp) {

    std::optional<APInt> tripCount = constantTripCount(
        forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), /*isSigned=*/true,
        scf::computeUbMinusLb
    );
    if (!tripCount || tripCount->getSExtValue() < 2)
        return;

    int64_t lb = *getConstIntValue(forOp.getLowerBound());
    int64_t step = *getConstIntValue(forOp.getStep());

    Value iv = forOp.getInductionVar();

    affine::FlatAffineValueConstraints constraints;
    constraints.appendDimVar({iv});
    // iv <= lb + step*(tripCount - 1) ==> -iv + (lb + step*(tripCount - 1)) >= 0
    constraints.addInequality({-1, (lb + step * (tripCount->getSExtValue() - 1))});
    // iv >= lb ==> iv - lb >= 0
    constraints.addInequality({1, -lb});

    forOp.walk([&](Operation *affineOp) {
        if (!isa<affine::AffineMinOp, affine::AffineMaxOp>(affineOp))
            return;

        (void)canonicalizeMinMaxOp(rewriter, affineOp, constraints);
    });
}

// If forOp has dynamic shapes:
// - if the upper bound is not aligned with step, peel the last iteration;
// - otherwise, peel one iteration from the start and one from the end.
// - simplify the min/max ops in the loop, after the peeling.
bool peelLoop(IRRewriter &rewriter, mlir::scf::ForOp forOp) {
    SmallVector<Value> dynamicShapes = torq::collectDynamicShapes(forOp.getRegion());
    if (dynamicShapes.empty()) {
        return false;
    }
    LLVM_DEBUG({ llvm::dbgs() << "found dynamic shapes, peeling loop.\n"; });

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();

    std::optional<APInt> tripCount =
        constantTripCount(lb, ub, step, /*isSigned=*/true, scf::computeUbMinusLb);
    if (!tripCount || tripCount->getSExtValue() < 2)
        return false;

    scf::ForOp lastIteration;
    if (llvm::succeeded(mlir::scf::peelForLoopAndSimplifyBounds(rewriter, forOp, lastIteration))) {
        // The last iteration was an incomplete iteration. Don't do anymore
        // peeling for now, in case this is enough.
        return true;
    }

    scf::ForOp firstIteration;
    if (failed(mlir::scf::peelForLoopFirstIteration(rewriter, forOp, firstIteration))) {
        forOp->emitWarning("can not peel the first iteration of an scf.for with dynamic shapes.");
    }

    if (tripCount->getSExtValue() > 2) {
        if (failed(mlir::scf::peelForLoopLastIteration(rewriter, forOp, lastIteration))) {
            forOp->emitWarning("can not peel an scf.for with dynamic shapes.");
        }
    }

    // Simplify the affine min/max in the loop using the loop bounds
    rewriteAffineOpInLoop(rewriter, forOp);

    return true;
}

struct PeelTileLoopsPass : public impl::PeelTileLoopsBase<PeelTileLoopsPass> {
    mlir::OpPassManager localPm_;

    PeelTileLoopsPass() { localPm_.addPass(mlir::createCanonicalizerPass()); }

    PeelTileLoopsPass(const PeelTileLoopsPass &pass) : localPm_(pass.localPm_) {}

    void runOnOperation() override {
        auto funcOp = getOperation();
        IRRewriter rewriter(&getContext());

        // First run the affine simplification on all for loops (tiling of
        // tensor::PadOp adds an scf::IfOp that is superfluous in some cases).
        funcOp->walk([&](mlir::scf::ForOp forOp) { rewriteAffineOpInLoop(rewriter, forOp); });
        (void)runPipeline(localPm_, funcOp);

        // Keep peeling iterations from for loops with dynamic shapes.
        bool changed;
        do {
            changed = false;
            funcOp->walk([&](mlir::scf::ForOp forOp) { changed |= peelLoop(rewriter, forOp); });

            // We have to run full blown canonicalizer to see if the dynamic
            // shapes disappeared.
            (void)runPipeline(localPm_, funcOp);
        } while (changed);

        LLVM_DEBUG({ llvm::dbgs() << "Peel Tile Loops - DONE\n"; });
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPeelTileLoopsPass() {
    return std::make_unique<PeelTileLoopsPass>();
}

} // namespace mlir::syna::torq
