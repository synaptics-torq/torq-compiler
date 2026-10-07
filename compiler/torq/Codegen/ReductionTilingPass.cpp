// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TileAndFuseUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>

#define DEBUG_TYPE "torq-reduction-tiling"

namespace mlir::syna::torq {
namespace {

// The reduce write descriptor (CEPW n_size) can encode at most 1<<16 elements per reduction.
// See third_party/torq-hw/ct/torq_api.c, _reg_ndl_desc_gen.
constexpr int64_t kMaxReduceTileElements = 1 << 16;

std::optional<std::pair<unsigned, int64_t>>
getSupportedReductionDimAndSize(linalg::GenericOp genericOp) {
    SmallVector<unsigned> reductionDims;
    genericOp.getReductionDims(reductionDims);
    if (reductionDims.empty())
        return std::nullopt;
    if (reductionDims.size() != 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "reduction-tiling: expected one reduction dimension, found "
                         << reductionDims.size() << "\n"
        );
        return std::nullopt;
    }

    unsigned reductionDim = reductionDims.front();
    int64_t reductionSize = genericOp.getStaticLoopRanges()[reductionDim];
    if (reductionSize == ShapedType::kDynamic || reductionSize <= 1)
        return std::nullopt;
    return std::make_pair(reductionDim, reductionSize);
}

bool supportsTileLocalPartialReduction(linalg::GenericOp genericOp) {
    for (OpOperand &initOperand : genericOp.getDpsInitsMutable()) {
        AffineMap map = genericOp.getMatchingIndexingMap(&initOperand);
        if (!map.isProjectedPermutation())
            return false;
    }
    return true;
}

FailureOr<SmallVector<int64_t>>
chooseTileSizes(linalg::GenericOp genericOp, unsigned reductionDim, int64_t reductionSize) {
    SmallVector<int64_t> loopRanges = genericOp.getStaticLoopRanges();
    if (llvm::is_contained(loopRanges, ShapedType::kDynamic))
        return failure();

    const int64_t lramBudget = getLramTilingBudget();
    FailureOr<int64_t> untiledBytes = estimateTiledOperandBytes(genericOp, loopRanges);
    if (failed(untiledBytes))
        return failure();
    if (*untiledBytes <= lramBudget && reductionSize <= kMaxReduceTileElements)
        return SmallVector<int64_t>{};

    SmallVector<int64_t> unitTileSizes(loopRanges.size(), 1);
    FailureOr<int64_t> floorBytes = estimateTiledOperandBytes(genericOp, unitTileSizes);
    if (failed(floorBytes) || *floorBytes > lramBudget)
        return failure();

    SmallVector<int64_t> tileSizes(loopRanges);
    int64_t initialReductionTile =
        largestReductionTileAtMost(reductionSize, std::min(reductionSize, kMaxReduceTileElements));
    tileSizes[reductionDim] = initialReductionTile;

    // Prefer reduction-only tiling. This preserves the parallel tile shape and keeps the
    // transformation independent from tile-and-fuse's later parallel fit.
    FailureOr<int64_t> estimatedBytes = estimateTiledOperandBytes(genericOp, tileSizes);
    if (failed(estimatedBytes))
        return failure();
    while (*estimatedBytes > lramBudget && tileSizes[reductionDim] > 1) {
        tileSizes[reductionDim] =
            largestReductionTileAtMost(reductionSize, tileSizes[reductionDim] / 2);
        estimatedBytes = estimateTiledOperandBytes(genericOp, tileSizes);
        if (failed(estimatedBytes))
            return failure();
    }

    // If reduction-only tiling cannot fit, restore the largest legal reduction tile and shrink the
    // kept dimensions. This avoids changing the accumulation order when parallel tiling is enough.
    if (*estimatedBytes > lramBudget) {
        tileSizes.assign(loopRanges);
        tileSizes[reductionDim] = initialReductionTile;
        estimatedBytes = estimateTiledOperandBytes(genericOp, tileSizes);
        if (failed(estimatedBytes))
            return failure();
    }

    while (*estimatedBytes > lramBudget) {
        int64_t largestSize = 1;
        std::optional<unsigned> largestParallelDim;
        for (auto [dim, iteratorType] : llvm::enumerate(genericOp.getIteratorTypesArray())) {
            if (iteratorType == utils::IteratorType::parallel && tileSizes[dim] > largestSize) {
                largestSize = tileSizes[dim];
                largestParallelDim = dim;
            }
        }
        if (largestParallelDim) {
            tileSizes[*largestParallelDim] = largestSize / 2;
        }
        else if (tileSizes[reductionDim] > 1) {
            tileSizes[reductionDim] =
                largestReductionTileAtMost(reductionSize, tileSizes[reductionDim] / 2);
        }
        else {
            return failure();
        }
        estimatedBytes = estimateTiledOperandBytes(genericOp, tileSizes);
        if (failed(estimatedBytes))
            return failure();
    }

    return tileSizes;
}

LogicalResult
tileReduction(IRRewriter &rewriter, linalg::GenericOp genericOp, bool longReductionsOnly) {
    auto reductionDimAndSize = getSupportedReductionDimAndSize(genericOp);
    if (!reductionDimAndSize)
        return success();
    auto [reductionDim, reductionSize] = *reductionDimAndSize;
    if (longReductionsOnly && reductionSize <= kMaxReduceTileElements)
        return success();

    FailureOr<SmallVector<int64_t>> tileSizes =
        chooseTileSizes(genericOp, reductionDim, reductionSize);
    if (failed(tileSizes))
        return failure();
    if (tileSizes->empty())
        return success();

    // Rewrite only reductions that need tiling, so fitting groups keep their accumulation order.
    if (isMarkedFuseGroup(genericOp)) {
        if (linalg::GenericOp rewrittenOp =
                rewriteKeepdimsFuseGroupReduction(rewriter, genericOp)) {
            genericOp = rewrittenOp;
            reductionDimAndSize = getSupportedReductionDimAndSize(genericOp);
            if (!reductionDimAndSize)
                return failure();
            std::tie(reductionDim, reductionSize) = *reductionDimAndSize;
            tileSizes = chooseTileSizes(genericOp, reductionDim, reductionSize);
            if (failed(tileSizes) || tileSizes->empty())
                return failure();
        }
    }

    SmallVector<OpFoldResult> tileSizesOfr(genericOp.getNumLoops(), rewriter.getIndexAttr(0));
    SmallVector<int64_t> loopRanges = genericOp.getStaticLoopRanges();
    for (unsigned dim = 0; dim < tileSizes->size(); ++dim)
        if ((*tileSizes)[dim] < loopRanges[dim])
            tileSizesOfr[dim] = rewriter.getIndexAttr((*tileSizes)[dim]);

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizesOfr);
    if ((*tileSizes)[reductionDim] < reductionSize) {
        if (!supportsTileLocalPartialReduction(genericOp))
            return failure();
        options.setReductionTilingStrategy(ReductionTilingStrategy::PartialReductionOuterParallel);
        options.setReductionDims({reductionDim});
    }

    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(genericOp.getOperation()), options);
    if (failed(tilingResult))
        return failure();

    for (LoopLikeOpInterface loop : tilingResult->loops)
        loop->setAttr(TORQ_REDUCTION_TILING_LOOP, rewriter.getUnitAttr());

    rewriter.replaceOp(genericOp, tilingResult->replacements);

    SmallVector<scf::ForOp> forOps;
    for (LoopLikeOpInterface loop : tilingResult->loops)
        if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
            forOps.push_back(forOp);

    // Partial-reduction loops write independent tiles, so leave them for the post-T&F pipeline.
    if ((*tileSizes)[reductionDim] == reductionSize)
        linalg::peelLoops(rewriter, forOps);
    return success();
}

class ReductionTilingPass : public impl::ReductionTilingBase<ReductionTilingPass> {
  public:
    using ReductionTilingBase::ReductionTilingBase;
    explicit ReductionTilingPass(bool longReductionsOnly)
        : longReductionsOnly(longReductionsOnly) {}

    void runOnOperation() override {
        SmallVector<linalg::GenericOp> reductions;
        getOperation()->walk([&](linalg::GenericOp genericOp) {
            if (getTargetExecutor(genericOp) != torq_hl::Executor::Host &&
                getSupportedReductionDimAndSize(genericOp))
                reductions.push_back(genericOp);
        });

        IRRewriter rewriter(&getContext());
        for (linalg::GenericOp genericOp : reductions) {
            if (failed(tileReduction(rewriter, genericOp, longReductionsOnly))) {
                genericOp.emitError("failed to tile reduction for NSS constraints");
                signalPassFailure();
                return;
            }
        }
    }

  private:
    bool longReductionsOnly = false;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createReductionTilingPass(bool longReductionsOnly) {
    return std::make_unique<ReductionTilingPass>(longReductionsOnly);
}

} // namespace mlir::syna::torq
