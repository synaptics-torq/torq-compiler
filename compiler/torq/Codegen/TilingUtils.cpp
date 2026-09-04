// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TilingUtils.h"

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-tiling-utils"

namespace mlir::syna::torq {

std::optional<size_t> getSlicingIterationDomainIndex(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<size_t>>(op)
        .Case<linalg::Conv2DNhwcHwcfOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::Conv2DNhwcHwcfOp;
        })
        .Case<linalg::Conv2DNchwFchwOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::Conv2DNchwFchwOp;
        })
        .Case<linalg::DepthwiseConv2DNhwcHwcOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::DepthwiseConv2DNhwcHwcOp;
        })
        .Case<linalg::DepthwiseConv2DNchwChwOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::DepthwiseConv2DNchwChwOp;
        })
        .Case<linalg::PoolingNhwcMaxOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::PoolingNhwcMaxOp;
        })
        .Case<linalg::PoolingNchwMaxOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::PoolingNchwMaxOp;
        })
        .Case<linalg::PoolingNcwMaxOp>([](auto) -> std::optional<size_t> {
            return SlicingIterationDomainIndex::PoolingNcwMaxOp;
        })
        .Default([](auto) -> std::optional<size_t> { return std::nullopt; });
}

namespace {

// Replace the untiled `op` with its tiled version from `tiledResults.
void replaceTiledOp(
    RewriterBase &rewriter, Operation *op, const scf::SCFTileAndFuseResult &tiledResults
) {
    for (OpResult res : op->getResults()) {
        if (Value replacement = tiledResults.replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, replacement);
        }
    }
}

// If ivValue is the induction variable of scf::ForOp, return the lower bound of
// the loop if it's a constant.
llvm::FailureOr<Attribute> getConstLowerBoundOfIv(Value ivValue) {
    auto blockArg = dyn_cast<mlir::BlockArgument>(ivValue);
    if (!blockArg)
        return failure();

    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
        return failure();

    arith::ConstantOp lowerBoundOp = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    if (!lowerBoundOp)
        return failure();

    return lowerBoundOp.getValueAttr();
}

llvm::FailureOr<SmallVector<Attribute>> computeAffineMapAtFirstIteration(
    AffineMap &map, SmallVector<Value> &operands, llvm::DenseMap<Value, Attribute> &computedValues
) {
    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    assert(map.getNumInputs() == operands.size());

    SmallVector<Attribute> computedOperands;
    computedOperands.reserve(operands.size());
    for (Value operand : operands) {
        llvm::FailureOr<Attribute> computedOperand =
            computeValueAtFirstIteration(operand, computedValues);
        if (failed(computedOperand))
            return failure();
        computedOperands.push_back(*computedOperand);
    }

    SmallVector<Attribute> results;
    if (failed(map.constantFold(computedOperands, results)))
        return llvm::failure();

    return results;
}

llvm::FailureOr<Attribute> computeAffineMinMaxAtFirstIteration(
    Operation *op, llvm::DenseMap<Value, Attribute> &computedValues
) {
    bool isMin;
    AffineMap map;
    SmallVector<Value> operands;
    if (auto minOp = dyn_cast<affine::AffineMinOp>(op)) {
        isMin = true;
        map = minOp.getMap();
        operands = minOp.getOperands();
    }
    else {
        auto maxOp = cast<affine::AffineMaxOp>(op);
        isMin = false;
        map = maxOp.getMap();
        operands = maxOp.getOperands();
    }

    auto results = computeAffineMapAtFirstIteration(map, operands, computedValues);
    if (failed(results))
        return llvm::failure();

    Attribute *result = nullptr;
    if (isMin) {
        result = llvm::min_element(*results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    else {
        result = llvm::max_element(*results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    if (result == nullptr)
        return llvm::failure();

    return *result;
}

llvm::FailureOr<Attribute> computeAffineApplyAtFirstIteration(
    affine::AffineApplyOp applyOp, llvm::DenseMap<Value, Attribute> &computedValues
) {
    AffineMap map = applyOp.getMap();
    SmallVector<Value> operands = applyOp.getOperands();

    auto results = computeAffineMapAtFirstIteration(map, operands, computedValues);
    if (failed(results))
        return llvm::failure();

    assert(results->size() == 1 && "AffineApplyOp should have exactly one reault");

    return results->front();
}

// Taken from mlir/lib/Dialect/SCF/Utils/AffineCanonicalizationUtils.cpp, as it
// is not publicly visible there.
// Given some affine constraints, simplify the min/max op, or return failure.
FailureOr<affine::AffineApplyOp> canonicalizeMinMaxOp(
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

} // namespace

void rewriteAffineOpInLoop(RewriterBase &rewriter, LoopLikeOpInterface loopOp) {
    affine::FlatAffineValueConstraints constraints;

    // Assuming LoopLikeOpInterface or helpers provide these methods for all dimensions
    auto ivs = *loopOp.getLoopInductionVars();
    auto lbs = *loopOp.getLoopLowerBounds();
    auto ubs = *loopOp.getLoopUpperBounds();
    auto steps = *loopOp.getLoopSteps();

    for (auto [iv, lb, ub, step] : llvm::zip_equal(ivs, lbs, ubs, steps)) {
        std::optional<APInt> tripCount =
            constantTripCount(lb, ub, step, /*isSigned=*/true, scf::computeUbMinusLb);
        if (!tripCount)
            continue;

        int64_t lbVal = *getConstantIntValue(lb);
        int64_t stepVal = *getConstantIntValue(step);

        if (tripCount->getSExtValue() == 1) {
            // Single-trip loop: IV is always exactly lb.
            // Express as two inequalities (lb ≤ iv and iv ≤ lb) so that
            // canonicalizeMinMaxOp can prove affine.min/max expressions are
            // constant across the loop body.
            constraints.appendDimVar({iv});
            constraints.addInequality({1, -lbVal}); // iv - lb >= 0
            constraints.addInequality({-1, lbVal}); // lb - iv >= 0
            continue;
        }

        if (tripCount->getSExtValue() < 2)
            continue;

        constraints.appendDimVar({iv});
        constraints.addInequality({-1, (lbVal + stepVal * (tripCount->getSExtValue() - 1))});
        constraints.addInequality({1, -lbVal});
    }

    loopOp.getOperation()->walk([&](Operation *affineOp) {
        if (!isa<affine::AffineMinOp, affine::AffineMaxOp>(affineOp))
            return;

        (void)canonicalizeMinMaxOp(rewriter, affineOp, constraints);
    });
}

void applyTiledResults(
    RewriterBase &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
) {
    // Replace the root with its tiled result
    replaceTiledOp(rewriter, op, tiledResults);

    // In general, fused producers can declare that they want to be yielded.
    // Here we replace the untiled producers with the yielded result.
    for (Operation *prodOp : tiledResults.fusedProducers) {
        replaceTiledOp(rewriter, prodOp, tiledResults);

        // Erase the extract_slice tiling add, so we can see if the producer has
        // no users (and not tile it again).
        for (Operation *user : llvm::make_early_inc_range(prodOp->getUsers())) {
            if (isa<tensor::ExtractSliceOp>(user) && user->use_empty()) {
                rewriter.eraseOp(user);
            }
        }
    }
}

void eraseBackward(RewriterBase &rewriter, Operation *op) {
    llvm::DenseSet<Operation *> workset{op};
    while (!workset.empty()) {
        auto iter = workset.begin();
        Operation *op = *iter;
        workset.erase(iter);

        if (!op->use_empty())
            continue;

        for (Value operand : op->getOperands()) {
            if (Operation *defOp = operand.getDefiningOp())
                workset.insert(defOp);
        }
        rewriter.eraseOp(op);
    }
}

llvm::FailureOr<Attribute>
computeValueAtFirstIteration(Value val, llvm::DenseMap<Value, Attribute> &computedValues) {
    if (computedValues.contains(val))
        return computedValues[val];

    auto valOp = val.getDefiningOp();
    if (!valOp) {
        // This must be a block argument; if it's the induction variable of
        // scf::ForOp, return the lower bound of the loop.
        return getConstLowerBoundOfIv(val);
    }

    auto result = llvm::TypeSwitch<Operation *, llvm::FailureOr<Attribute>>(valOp)
                      .Case<affine::AffineApplyOp>([&](auto applyOp) {
                          return computeAffineApplyAtFirstIteration(applyOp, computedValues);
                      })
                      .Case<affine::AffineMinOp, affine::AffineMaxOp>([&](auto minMaxOp) {
                          return computeAffineMinMaxAtFirstIteration(minMaxOp, computedValues);
                      })
                      .Default([](auto) { return failure(); });

    if (succeeded(result))
        computedValues[val] = *result;

    return result;
}

} // namespace mlir::syna::torq
