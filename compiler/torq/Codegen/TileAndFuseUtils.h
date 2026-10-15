// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/OpDefinition.h"

// Helpers shared by TileAndFuse and its preparatory tiling passes.
//
// These deliberately live here rather than in TilingUtils.h: TilingUtils is a
// shared, multi-consumer file for generic tiling/slicing helpers used by both
// TileAndFusePass and LinalgSlicingPass (see its commit history), whereas
// everything below is TileAndFuse-specific machinery or prepares IR for that
// pass. Keeping that boundary explicit avoids widening TilingUtils.h's
// interface for consumers that do not participate in T&F.

namespace mlir {
class IRRewriter;
class Operation;
namespace func {
class FuncOp;
} // namespace func
namespace linalg {
class GenericOp;
class LinalgOp;
} // namespace linalg
} // namespace mlir

namespace mlir::syna::torq {

// Iteration domains that can be tiled, in the order they should be tiled.
struct TilingInfo {
    llvm::SmallSetVector<int64_t, 4> tilingOrder;

    // Minimal tile size per iter domain index. This is a best effort constraint:
    // we only use it for the root (consumer), and if after shrinking all dims
    // the op still does not fit, we will ignore minSize and shrink further.
    llvm::SmallVector<int64_t, 4> minSize;

    // Downward size adjustments per iter domain index. May return 0 to signal
    // that the requested size is too small to be a valid (byte-aligned) tile.
    llvm::SmallVector<std::function<int64_t(int64_t)>, 4> adjustSize;
};

// Compose a downward size adjustment onto an existing one, iterating to a
// fixpoint so neither constraint can undo the other. Both must be shrink-only.
inline std::function<int64_t(int64_t)>
composeSizeAdjustments(std::function<int64_t(int64_t)> f1, std::function<int64_t(int64_t)> f2) {
    return [=](int64_t size) {
        int64_t newSize = f2(f1(size));
        while (newSize != size) {
            assert(newSize <= size && "size adjustments must not grow the tile");
            size = newSize;
            newSize = f2(f1(size));
        }
        return newSize;
    };
}

// The smallest tile size a domain can be cut down to (always > 0).
int64_t getSmallestTileSize(const TilingInfo &tilingInfo, int64_t domain, int64_t domainSize);

// Try to compute the int value of `sizeFoldResult`. If it is not a constant,
// evaluate it at the first iteration of the surrounding loops.
llvm::FailureOr<int64_t> computeSizeAtFirstIteration(mlir::OpFoldResult sizeFoldResult);

// Return an order with the weight's unique parallel domain first. The
// principal domain is mapped to the tiling root by its static extent.
std::optional<llvm::SmallVector<int64_t>> computeChannelFirstShrinkOrder(
    mlir::Operation *consumerOp, const llvm::SetVector<mlir::Operation *> &producerOps,
    const TilingInfo &tilingInfo, llvm::ArrayRef<int64_t> iterDomainSizes
);

// The root domain of the output channels of the group's weight-bearing principal op,
// matched by its static extent. nullopt when the group has no such op, or when no root
// domain has the channel extent. Failure when the channel is ambiguous: several
// parallel domains index the weight, or several root domains have the channel extent.
llvm::FailureOr<std::optional<int64_t>> findPrincipalChannelDomain(
    mlir::Operation *consumerOp, const llvm::SetVector<mlir::Operation *> &producerOps,
    const TilingInfo &tilingInfo, llvm::ArrayRef<int64_t> iterDomainSizes
);

// Compute the order in which fitTileToMemory's shrink pass should cut domains.
// Weight-bearing operations prioritize domains indexing the larger of the
// principal op's activation and weight operands. Other matmuls prioritize the
// domains that remove the most bytes across the fuse group. Returns nullopt
// when no heuristic applies, in which case the caller keeps tilingOrder.
std::optional<llvm::SmallVector<int64_t>> computeShrinkOrderByReduction(
    mlir::Operation *consumerOp, const llvm::SetVector<mlir::Operation *> &producerOps,
    const TilingInfo &tilingInfo, llvm::ArrayRef<int64_t> iterDomainSizes,
    llvm::ArrayRef<mlir::OpFoldResult> sizes, bool fallback
);

// K-chunk LRAM-oversized matmuls (linalg.matmul / linalg.batch_matmul) into a
// chain of accumulating chunk matmuls, before T&F's parallel fit. The split is
// generated as an scf.for reduction loop and then immediately fully unrolled,
// so T&F's loop-free fit-check keeps working while the splitting logic stays
// expressed in terms of standard SCF tiling utilities. Matmuls that still
// cannot fit are marked for Host execution.
void splitOversizedMatmulsAlongK(func::FuncOp funcOp);

// LRAM budget available to a tile: total LRAM minus descriptor headroom.
int64_t getLramTilingBudget();

// Estimate the total operand bytes of a linalg op at the given iteration-domain tile sizes.
llvm::FailureOr<int64_t>
estimateTiledOperandBytes(linalg::LinalgOp linalgOp, llvm::ArrayRef<int64_t> tileSizes);

// Shrink tileSize down until both the common tile and the final (possibly
// smaller) tail tile are SDIM-compatible with the encoding in Utils/Kernel.cpp.
int64_t makeSdimFriendlyTileSize(int64_t iterDomainSize, int64_t tileSize);

// Pick the largest reduction tile at most requestedTileSize that is
// SDIM-compatible, allowing a (SDIM-friendly) tail when requestedTileSize does
// not divide reductionSize evenly.
int64_t largestReductionTileAtMost(int64_t reductionSize, int64_t requestedTileSize);

// Rewrite a marked keepdims reduction into a tileable non-keepdims reduction plus reshape.
// Return the replacement reduction, or nullptr if the op was not rewritten.
linalg::GenericOp
rewriteKeepdimsFuseGroupReduction(mlir::IRRewriter &rewriter, linalg::GenericOp op);

} // namespace mlir::syna::torq
