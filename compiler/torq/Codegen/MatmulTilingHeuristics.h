// Copyright 2025 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "TileAndFuseUtils.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

// Matmul-specific steering of the TileAndFuse tile search. A GEMM that does not
// fit LRAM trades DMA (each M tile re-streams the weight, each N tile the input),
// ALU utilization (N is the vectorized axis) and a fixed per-tile cost, none of
// which the generic shrink/grow search models. Everything here is shrink-only or
// a preference, so the search's fit-to-memory invariants are untouched, and it
// applies only to tiles whose kernel computes a matmul.

namespace mlir::syna::torq {

// The matmul the tile anchored on `op` computes: `op` itself, its fuse group's
// principal op, or a member of `fusedProducers` (the producers the tiling fuses
// into the tile). Null when the tile computes no matmul.
mlir::Operation *
findTiledMatmul(mlir::Operation *op, const llvm::SetVector<mlir::Operation *> &fusedProducers);

// Steer `tilingInfo` for the tile anchored on `op`: consider its parallel
// domains largest-first, keep the N tile a whole number of ALU vector columns
// and prefer at least one. No-op when the tile computes no matmul.
void applyMatmulTilingHeuristics(
    TilingInfo &tilingInfo, mlir::TilingInterface op, llvm::ArrayRef<int64_t> iterDomainSizes,
    const llvm::SetVector<mlir::Operation *> &fusedProducers
);

// The loop interchange that visits the tiles of `op` n-outer (for(n)for(m))
// when that streams fewer bytes than the default m-outer order; empty to keep
// the default.
llvm::SmallVector<int64_t> chooseMatmulLoopInterchange(
    mlir::Operation *op, llvm::ArrayRef<mlir::OpFoldResult> tileSizes,
    const llvm::SetVector<mlir::Operation *> &fusedProducers
);

} // namespace mlir::syna::torq
