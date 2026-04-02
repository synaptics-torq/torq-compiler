#pragma once

#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

// Replace the untiled consumer `op` with its tiled version from `tiledResults`,
// and do the same for any producer that was yielded as well.
void applyTiledResults(
    RewriterBase &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
);

// Earase `op` and all it's users recursively.
void eraseForward(RewriterBase &rewriter, Operation *op);

} // namespace mlir::syna::torq
