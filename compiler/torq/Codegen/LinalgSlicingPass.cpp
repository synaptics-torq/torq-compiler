// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TilingUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>

#define DEBUG_TYPE "torq-linalg-slicing"

namespace mlir::syna::torq {

namespace {

// A unit attribute, used to track the main operation over the multiple tiling
// and unrolling the pass does.
const std::string TORQ_LINALG_SLICING = "torq-linalg-slicing";

const int64_t kGrouping = 4;

// Return tile size for peeling.
int64_t getPeelingTileSize(int64_t domainSize, int64_t grouping) {
    assert(domainSize > grouping);

    // We want to split the domain into two or three tiles, the first tile
    // should be divisible by grouping so it can utilize multiple slices at the
    // same time, and the second tile is whatever is left.
    int64_t remainder;
    if (domainSize >= TorqHw::get().getSliceCount() * grouping) {
        // If we have enough rows, the first tile will utilize all the
        // slices, and we might need to sub-tile the second tile, later.
        remainder = domainSize % (TorqHw::get().getSliceCount() * grouping);
    }
    else {
        // If there are not enough rows to utilize all the slices, the
        // first tile will utilize as many slices as possible, and the
        // second tile will be the remainder.
        remainder = domainSize % grouping;
    }

    return domainSize - remainder;
}

// Return tile size for slicing.
int64_t getSlicingTileSize(int64_t domainSize, int64_t grouping) {
    assert(domainSize % grouping == 0);
    assert(domainSize >= grouping);

    // We need to tile the rows evenly between the slices. We assume the peeling
    // was done to guarantee that now the rows can be properly distributed
    // between the slices without a remainder.

    if (domainSize < TorqHw::get().getSliceCount() * grouping) {
        // Not enough rows to utilize all slices, so only some of them will
        // get a tile of size grouping.
        return grouping;
    }

    // We have enough rows to utilize all slices.

    assert(domainSize % (TorqHw::get().getSliceCount() * grouping) == 0);

    // This tile size will be a multiple of grouping.
    return domainSize / TorqHw::get().getSliceCount();
}

scf::SCFTileAndFuseOptions::ControlFnTy getControlFn(IntegerAttr fuseGroup) {
    return [fuseGroup](
               mlir::tensor::ExtractSliceOp, OpResult producerOpResult, bool
           ) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        if (!fuseGroup)
            return std::nullopt;

        auto producerGroups =
            producerOpResult.getOwner()->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (producerGroups && llvm::is_contained(producerGroups, fuseGroup)) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{false};
        }

        return std::nullopt;
    };
}

FailureOr<scf::SCFTileAndFuseResult> tileAndFuse(
    PatternRewriter &rewriter, Operation *op, int64_t tileSize, size_t slicingIter,
    scf::SCFTilingOptions::LoopType loopType
) {
    LLVM_DEBUG(
        llvm::dbgs() << "tiling size: " << tileSize << ", "
                     << (loopType == scf::SCFTilingOptions::LoopType::ForOp ? "ForOp" : "ForallOp")
                     << "\n"
    );

    IntegerAttr fuseGroup = isFuseGroupPrincipalOp(op);

    // Tiling will start from the output of the pattern fuse group.
    Operation *rootOp = fuseGroup ? getFuseGroupOutputOp(op, fuseGroup) : op;

    // We can only slice TilingInterface ops
    TilingInterface rootTi = dyn_cast<TilingInterface>(rootOp);
    assert(rootTi && "pattern fuse group ops are expected to be TilingInterface");

    size_t iterDomCount = rootTi.getIterationDomain(rewriter).size();

    SmallVector<OpFoldResult> tileSizes(iterDomCount, rewriter.getIndexAttr(0));
    tileSizes[slicingIter] = rewriter.getIndexAttr(tileSize);

    scf::SCFTileAndFuseOptions options;
    options.tilingOptions.setTileSizes(tileSizes);
    options.tilingOptions.setLoopType(loopType);
    options.setFusionControlFn(getControlFn(fuseGroup));

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootTi, options);
    if (failed(tiledResults)) {
        LLVM_DEBUG(assert(false));
        rootTi->emitError("failed to do slicing");
        return failure();
    }
    applyTiledResults(rewriter, rootOp, *tiledResults);

    assert(rootOp->use_empty() && "expected rootOp to have no users after tiling");
    eraseBackward(rewriter, rootOp);

    return tiledResults;
}

void sliceToSize(
    PatternRewriter &rewriter, Operation *op, int64_t domainSize, size_t iterDomCount,
    size_t slicingIter, int64_t grouping
) {
    int64_t tileSize = getSlicingTileSize(domainSize, grouping);
    if (tileSize == domainSize) {
        return;
    }

    (void
    )tileAndFuse(rewriter, op, tileSize, slicingIter, scf::SCFTilingOptions::LoopType::ForallOp);
}

void peelAndSliceToSize(
    PatternRewriter &rewriter, Operation *op, int64_t domainSize, size_t iterDomCount,
    size_t slicingIter, int64_t grouping
) {
    if (domainSize < 2 * grouping)
        return;

    int64_t tileSize = getPeelingTileSize(domainSize, grouping);

    if (tileSize == domainSize) {
        sliceToSize(rewriter, op, domainSize, iterDomCount, slicingIter, grouping);
        return;
    }

    op->setAttr(TORQ_LINALG_SLICING, rewriter.getUnitAttr());

    auto tiledResults =
        tileAndFuse(rewriter, op, tileSize, slicingIter, scf::SCFTilingOptions::LoopType::ForOp);
    if (failed(tiledResults)) {
        return;
    }
    assert(tiledResults->loops.size() == 1);

    SmallVector<Operation *, 2> clonedOps{2, nullptr};

    if (failed(loopUnrollByFactor(
            cast<scf::ForOp>(tiledResults->loops[0]), 2,
            [&](unsigned i, Operation *clonedOp, OpBuilder) {
                assert(0 <= i && i < 2);
                if (clonedOp->hasAttr(TORQ_LINALG_SLICING)) {
                    clonedOp->removeAttr(TORQ_LINALG_SLICING);
                    clonedOps[i] = clonedOp;
                }
            }
        ))) {
        LLVM_DEBUG(assert(false));
        op->emitError("failed to peel");
        return;
    }

    assert(llvm::all_of(clonedOps, [](Operation *op) { return op; }) && "expected two clones");

    // Slice the first tile.
    sliceToSize(rewriter, clonedOps[0], tileSize, iterDomCount, slicingIter, grouping);

    // Peel and slice the second tile.
    peelAndSliceToSize(
        rewriter, clonedOps[1], domainSize - tileSize, iterDomCount, slicingIter, grouping
    );
}

LogicalResult
peelAndSlice(PatternRewriter &rewriter, Operation *op, size_t slicingIter, int64_t grouping) {
    // Do nothing if the containing operation is a forall (already sliced)
    if (op->getParentOfType<scf::ForallOp>()) {
        return rewriter.notifyMatchFailure(op, "already sliced");
    }

    // If fuseGroup is not nullptr, fuse all the members of the pattern fuse group
    // (op is assumed to be the principal op of the group); otherwise, tile op only.
    IntegerAttr fuseGroup = isFuseGroupPrincipalOp(op);

    // If op is a principal op of a pattern fuse group, it is not a member of any
    // other group.
    assert(!fuseGroup || op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP).size() == 1);

    // If op is not a principal op of a pattern fuse group, we assume it's not
    // marked at all (otherwise we need to tile the whole group).
    assert(fuseGroup || !isMarkedFuseGroup(op));

    // Tiling will start from the output of the pattern fuse group.
    Operation *rootOp = fuseGroup ? getFuseGroupOutputOp(op, fuseGroup) : op;

    // We can only slice TilingInterface ops
    TilingInterface rootTi = dyn_cast<TilingInterface>(rootOp);
    assert(rootTi && "pattern fuse group ops are expected to be TilingInterface");

    if (!cast<ShapedType>(rootOp->getResult(0).getType()).hasStaticShape()) {
        return rewriter.notifyMatchFailure(op, "output shape is not static");
    }

    // The untiled domain sizes
    auto [iterDomainOffsets, iterDomainSizes, iterDomainStrides] =
        getOffsetsSizesAndStrides(rootTi.getIterationDomain(rewriter));

    std::optional<SmallVector<int64_t>> iterDomainConstSizes =
        getConstantIntValues(iterDomainSizes);
    assert(iterDomainConstSizes && "iteration domain sizes are not constants");
    // Can't return failure() here as we already used the rewriter,
    // which will cause the driver to spin.

    assert(slicingIter < iterDomainConstSizes->size());

    int64_t domainSize = (*iterDomainConstSizes)[slicingIter];
    if (domainSize < 2 * grouping) {
        // There are not enough rows to utilize even two slices at the same time.
        return rewriter.notifyMatchFailure(op, "dimension is too small to slice");
    }

    peelAndSliceToSize(
        rewriter, op, domainSize, iterDomainConstSizes->size(), slicingIter, grouping
    );

    return success();
}

template <class Conv2DOp> struct Conv2DPattern : public OpRewritePattern<Conv2DOp> {
    size_t slicingIter_;

    Conv2DPattern(MLIRContext *context, size_t slicingIter)
        : OpRewritePattern<Conv2DOp>(context), slicingIter_(slicingIter) {}

    LogicalResult matchAndRewrite(Conv2DOp conv2DOp, PatternRewriter &rewriter) const override {
        // TODO: not all conv2ds are part of a pattern fuse group.
        if (!isFuseGroupPrincipalOp(conv2DOp)) {
            return rewriter.notifyMatchFailure(conv2DOp, "not the principal operation");
        }

        return peelAndSlice(rewriter, conv2DOp, slicingIter_, kGrouping);
    }
}; // class Conv2DPattern

// Elementwise pattern for linalg.generic ops. Picks the leftmost non-unit
// dimension to slice on, keeping inner dimensions contiguous in memory.
struct ElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        if (!isFuseGroupPrincipalOp(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "not the principal operation");
        }

        // Only operate on elementwise (all-parallel) generic ops.
        if (!linalg::isElementwise(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "Not an elementwise operation");
        }

        // Resolve the slicing dimension: leftmost non-unit.
        std::optional<int64_t> slicingIter;
        ArrayRef<int64_t> shape = cast<ShapedType>(genericOp->getResultTypes()[0]).getShape();
        for (auto [i, size] : llvm::enumerate(shape)) {
            if (size > 1) {
                slicingIter = i;
                break;
            }
        }
        if (!slicingIter)
            return rewriter.notifyMatchFailure(genericOp, "no dimension to slice");

        return peelAndSlice(rewriter, genericOp, *slicingIter, kGrouping);
    }
}; // class ElementwisePattern

struct LinalgSlicingPass : public impl::LinalgSlicingBase<LinalgSlicingPass> {
    mlir::OpPassManager canonicalizer_;

    LinalgSlicingPass() { canonicalizer_.addPass(mlir::createCanonicalizerPass()); }

    void runOnOperation() override {
        if (TorqHw::get().getSliceCount() < 2)
            return;

        LLVM_DEBUG(llvm::dbgs() << "Linalg Slicing - START\n");

        auto funcOp = getOperation();
        MLIRContext *context = funcOp.getContext();

        RewritePatternSet patterns(context);

        patterns.add<Conv2DPattern<linalg::Conv2DNhwcHwcfOp>>(context, 3);
        patterns.add<Conv2DPattern<linalg::Conv2DNchwFchwOp>>(context, 1);

        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNhwcHwcOp>>(context, 3);
        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNchwChwOp>>(context, 1);

        // Elementwise generic ops: dynamically slice on leftmost non-unit dim
        patterns.add<ElementwisePattern>(context);

        // patterns.add<FullyConnectedPattern>(ctx);

        FrozenRewritePatternSet frozenPatterns(std::move(patterns));

        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps);

        while (true) {
            bool changed = false;
            if (failed(applyPatternsGreedily(getOperation(), frozenPatterns, config, &changed))) {
                LLVM_DEBUG(assert(false && "failed"));
                return signalPassFailure();
            }
            if (!changed)
                break;

            if (failed(runPipeline(canonicalizer_, funcOp))) {
                LLVM_DEBUG(assert(false && "failed"));
                return signalPassFailure();
            }

            IRRewriter rewriter(context);
            funcOp->walk([&](mlir::scf::ForallOp forallOp) {
                rewriteAffineOpInLoop(rewriter, forallOp);
            });
        }

        LLVM_DEBUG(llvm::dbgs() << "Linalg Slicing - DONE\n");
    }
}; // class LinalgSlicingPass

} // namespace

// For S available slices, and grouping of G channels, we might need to split
// the work into at most 3 different chunks: the first chunk will be sliced evenly
// between all S slices; the second chunk will split evenly between some of the
// S slices (strictly less than S); and the last chunk will be smaller than G.
// For example, on HW with 4 slices, and grouping of 8, doing a conv2d with 92
// channels: first chunk will be 64 channels that will be sliced into 4 slices of
// 16 channels (64 = 4*8*2 = S*G*2), second chunk will be 24 channels that will
// be sliced into 3 slices of 8 channels (24 = 3*8 = k*G, where 0 < k < S), and
// the last chunk will be 4 channels (4 = 92-64-24).
std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgSlicingPass() {
    return std::make_unique<LinalgSlicingPass>();
}

} // namespace mlir::syna::torq
