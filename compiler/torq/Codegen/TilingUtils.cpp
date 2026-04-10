// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TilingUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-tiling-utils"

namespace mlir::syna::torq {

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
    assert(map.getNumDims() == operands.size());

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

} // namespace

void applyTiledResults(
    RewriterBase &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
) {
    // Replace the root with its tiled result
    replaceTiledOp(rewriter, op, tiledResults);

    // In general, fused producers can declare that they want to be yielded.
    // Here we replace the untiled producers with the yielded result.
    for (Operation *prodOp : tiledResults.fusedProducers) {
        replaceTiledOp(rewriter, prodOp, tiledResults);
    }
}

void eraseForward(RewriterBase &rewriter, Operation *op) {
    while (!op->getUsers().empty())
        eraseForward(rewriter, *op->getUsers().begin());
    rewriter.eraseOp(op);
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
