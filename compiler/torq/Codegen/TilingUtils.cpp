// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TilingUtils.h"

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

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

} // namespace mlir::syna::torq
