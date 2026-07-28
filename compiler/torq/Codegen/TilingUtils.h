#pragma once

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

// Iteration domain slicing dimension index.
enum SlicingIterationDomainIndex : size_t {
    Conv2DNhwcHwcfOp = 3,
    Conv2DNchwFchwOp = 1,

    DepthwiseConv2DNhwcHwcOp = 3,
    DepthwiseConv2DNchwChwOp = 1,

    PoolingNhwcMaxOp = 3,
    PoolingNchwMaxOp = 1,
    PoolingNcwMaxOp = 1,
};

// Slicing granularity
constexpr int64_t kGrouping = 4;

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

// Simplify each affine min/max op in loopOp, using the loop bounds.
void rewriteAffineOpInLoop(RewriterBase &rewriter, LoopLikeOpInterface loopOp);

} // namespace mlir::syna::torq
