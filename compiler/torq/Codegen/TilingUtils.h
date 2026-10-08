#pragma once

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <optional>

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

// The iteration-domain dimension LinalgSlicing slices `op` on, or nullopt when
// `op` has no slicing pattern. Keep in sync with the pattern registration in
// LinalgSlicingPass.
std::optional<size_t> getSlicingIterationDomainIndex(Operation *op);

// The dimension LinalgSlicing slices `op` on when `op` is the principal of an
// elementwise generic or reciprocal group that it slices, or nullopt (with the
// reason) when it leaves `op` on one slice. TileAndFusePass uses it to round tile
// sizes only for ops that are sliced. Defined in LinalgSlicingPass.cpp.
std::optional<int64_t> getElementwiseSlicingDim(Operation *op, StringRef *reason = nullptr);

// Slicing granularity
constexpr int64_t kGrouping = 4;

// Output channels of a conv that one hardware tile computes on one slice.
constexpr int64_t kConvOutputChannelTile = 64;

// Set by TileAndFuse on a conv that LinalgSlicing must keep on one slice.
constexpr llvm::StringLiteral kSingleSliceAttr = "torq-single-slice";

// True when `op` is a Conv2DNchwFchwOp whose packed weight is too large to keep
// `sliceCount` copies live within half of LRAM. Dual slicing such a conv can
// force smaller tiles that each re-stream the weights, so TileAndFuse probes it
// both ways and sets kSingleSliceAttr when one slice needs fewer tiles.
//
// The weight is sized for one 64-output-channel hardware tile per slice, not the
// whole filter. This only selects the convs that get the second probe; one it
// misses keeps dual slicing, as on main. Depthwise and NHWC convs are not probed.
bool isDualSliceWeightHeavyConv(Operation *op, int64_t sliceCount);

// True when every tile (or slice) reads the whole source of `sliceOp`. Fusing its
// producer into the loop would recompute the full result once per tile, so
// TileAndFuse leaves it before the loop, even when it is a pattern fuse group
// member. LinalgSlicing does the same for the slices of a weight-heavy conv.
bool readsWholeSourceEveryTile(tensor::ExtractSliceOp sliceOp);

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

// A canonicalizer pass that keeps the torq-fuse-group attr of tensor.collapse_shape and
// tensor.expand_shape when a pattern replaces them with new reshapes. Some upstream patterns
// (FoldCollapseOfCastOp, ConvertToStaticExpandShape) rebuild the reshape and drop this
// discardable attr. Only reshapes that had the attr pass it on.
std::unique_ptr<Pass> createFuseGroupPreservingCanonicalizerPass();

} // namespace mlir::syna::torq
