// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/OpDefinition.h"

// Helpers specific to the TileAndFuse pass's fit-to-memory logic.
//
// These deliberately live here rather than in TilingUtils.h: TilingUtils is a
// shared, multi-consumer file for generic tiling/slicing helpers used by both
// TileAndFusePass and LinalgSlicingPass (see its commit history), whereas
// everything below is single-consumer, TileAndFuse-specific machinery -- the
// memory-fit shrink/grow search and the types that support it. Keeping that
// boundary explicit avoids widening TilingUtils.h's interface for consumers
// that never need T&F's fit-to-memory model.

namespace mlir {
class Operation;
namespace func {
class FuncOp;
} // namespace func
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

// The smallest tile size a domain can be cut down to (always > 0).
int64_t getSmallestTileSize(const TilingInfo &tilingInfo, int64_t domain, int64_t domainSize);

// Try to compute the int value of `sizeFoldResult`. If it is not a constant,
// evaluate it at the first iteration of the surrounding loops.
llvm::FailureOr<int64_t> computeSizeAtFirstIteration(mlir::OpFoldResult sizeFoldResult);

// Compute the order in which fitTileToMemory's shrink pass should cut domains:
// greedily by the number of operand bytes each shrink removes across the whole
// fuse group, ties broken by the legacy tilingOrder. Returns nullopt when the
// group is not a matmul family or the extent correspondence is ambiguous, in
// which case the caller keeps the legacy tilingOrder.
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

} // namespace mlir::syna::torq
