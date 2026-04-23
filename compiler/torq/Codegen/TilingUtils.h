#pragma once

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

// Replace the untiled consumer `op` with its tiled version from `tiledResults`,
// and do the same for any producer that was yielded as well.
void applyTiledResults(
    RewriterBase &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
);

// Earase `op` and its sources recursively, as long as they don't have other
// users.
void eraseBackward(RewriterBase &rewriter, Operation *op);

// Computes `val` in the first iteration of the surrounding loops.
// `computedValues` is used to cache intermediate results.
llvm::FailureOr<Attribute>
computeValueAtFirstIteration(Value val, llvm::DenseMap<Value, Attribute> &computedValues);

} // namespace mlir::syna::torq
