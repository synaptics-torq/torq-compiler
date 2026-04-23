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

// Return the required tile size in the peeling phase.
int64_t getPeelingTileSize(int64_t domainSize, int64_t grouping) {
    assert(domainSize > grouping);

    // We want to split the domain into two tiles, the first tile
    // should be divisible by grouping so it can utilize multiple
    // slices at the same time (they can only be utilized when they all
    // have exactly the same number of rows, and this number is a
    // multiple of grouping), and the second tile is whatever is left.
    int64_t remainder;
    if (domainSize >= TorqHw::get().getSliceCount() * grouping) {
        // If we have enough rows, the first tile will utilize all the
        // slices, and we might need to sub-tile the second tile later.
        remainder = domainSize % (TorqHw::get().getSliceCount() * grouping);
    }
    else {
        // If there are not enough rows to utilize all the slices, the
        // first tile will utilize as many slices as possible, and the
        // second tile will be the remainder.
        remainder = domainSize % grouping;
    }

    // Return the (first) tile size.
    return domainSize - remainder;
}

// Return the required tile size in the slicing phase.
int64_t getSlicingTileSize(int64_t domainSize, int64_t grouping) {
    assert(domainSize > grouping);

    // We need to tile the rows evenly between the slices. We assume the
    // pass has already been executed in the peelGrouping mode twice to
    // guarantee that now the rows can be properly distributed between the
    // slices without a remainder.

    if (domainSize < TorqHw::get().getSliceCount() * grouping) {
        // Not enough rows to utilize all slices, so only some of them will
        // get a tile of size grouping.
        assert(domainSize % grouping == 0);
        return grouping;
    }

    // We have enough rows to utilize all slices.

    assert(domainSize % (TorqHw::get().getSliceCount() * grouping) == 0);

    // This tile size will be a multiple of grouping.
    return domainSize / TorqHw::get().getSliceCount();
}

// Do the actual tiling, for both the peeling and slicing phases.
LogicalResult tile(
    PatternRewriter &rewriter, Operation *op, bool peelGrouping, int64_t grouping,
    size_t slicingIter
) {
    // If fuseGroup is not nullptr, fuse all the members of the pattern fuse group
    // (op is assumed to be the principal op of the group); otherwise, tile op only.
    IntegerAttr fuseGroup = isFuseGroupPrincipalOp(op);

    Operation *rootOp = op;
    if (fuseGroup) {
        // Tiling will start from the output of the pattern fuse group.
        // TODO: this is a bit fragile; we assume tiling did not introduce
        // operations (e.g. tensor.extract_slice) in the middle of the already
        // marked pattern, and that it preserved the marking attributes. If this
        // doesn't work, we can either:
        // - clear the marking and run the marking pass again; or
        // - strengthen getFuseGroupOutputOp to look over tensor.extract_slice.
        rootOp = getFuseGroupOutputOp(op, fuseGroup);
    }

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
    if (domainSize <= 2 * grouping) {
        // There are not enough rows to utilize even two slices at the same time.
        return rewriter.notifyMatchFailure(op, "dimension is too small to slice");
    }

    // Initially set all tile sizes to 0 (don't tile).
    SmallVector<OpFoldResult> tileSizes(iterDomainSizes.size(), rewriter.getIndexAttr(0));
    int64_t tileSize;
    if (peelGrouping)
        tileSize = getPeelingTileSize(domainSize, grouping);
    else
        tileSize = getSlicingTileSize(domainSize, grouping);
    if (tileSize == domainSize)
        return rewriter.notifyMatchFailure(op, "tile size is the same as the dimension size");
    tileSizes[slicingIter] = rewriter.getIndexAttr(tileSize);

    scf::SCFTileAndFuseOptions options;
    options.tilingOptions.setTileSizes(tileSizes);
    options.tilingOptions.setLoopType(
        peelGrouping ? scf::SCFTilingOptions::LoopType::ForOp
                     : scf::SCFTilingOptions::LoopType::ForallOp
    );
    if (fuseGroup) {
        options.setFusionControlFn(
            [fuseGroup](
                mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
                bool isDestinationOperand
            ) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
                auto producerGroups =
                    producerOpResult.getOwner()->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
                bool shouldFuse = producerGroups && llvm::is_contained(producerGroups, fuseGroup);

                if (shouldFuse) {
                    return scf::SCFTileAndFuseOptions::ControlFnResult{false};
                }

                return std::nullopt;
            }
        );
    }
    else {
        // Don't do any fusing
        options.setFusionControlFn(
            [](mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
               bool isDestinationOperand
            ) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> { return std::nullopt; }
        );
    }

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootTi, options);
    if (failed(tiledResults)) {
        LLVM_DEBUG(assert(false));
        return rootTi->emitError("failed to do slicing");
    }
    applyTiledResults(rewriter, rootOp, *tiledResults);

    // This should erase (the untiled) op, so we don't try to slice it
    // again.
    assert(rootOp->use_empty() && "expected rootOp to have no users after tiling");
    eraseBackward(rewriter, rootOp);

    if (peelGrouping) {
        assert(tiledResults->loops.size() == 1);
        auto unrollResult = loopUnrollByFactor(cast<scf::ForOp>(tiledResults->loops[0]), 2);
        assert(llvm::succeeded(unrollResult));
    }

    return success();
}

template <class Conv2DOp> struct Conv2DPattern : public OpRewritePattern<Conv2DOp> {
    bool peelGrouping_;
    size_t slicingIter_;

    Conv2DPattern(MLIRContext *context, bool peelGrouping, size_t slicingIter)
        : OpRewritePattern<Conv2DOp>(context), peelGrouping_(peelGrouping),
          slicingIter_(slicingIter) {}

    LogicalResult matchAndRewrite(Conv2DOp conv2DOp, PatternRewriter &rewriter) const override {
        // Do nothing if the containing operation is a forall (already sliced)
        if (conv2DOp->template getParentOfType<scf::ForallOp>()) {
            return rewriter.notifyMatchFailure(conv2DOp, "already sliced");
        }

        return tile(rewriter, conv2DOp, peelGrouping_, 4, slicingIter_);
    }
}; // class Conv2DPattern

// Elementwise pattern for linalg.generic ops. Picks the leftmost non-unit
// dimension to slice on, keeping inner dimensions contiguous in memory.
struct ElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    bool peelGrouping_;

    ElementwisePattern(MLIRContext *context, bool peelGrouping)
        : OpRewritePattern<linalg::GenericOp>(context), peelGrouping_(peelGrouping) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        // Do nothing if the containing operation is a forall (already sliced)
        if (genericOp->template getParentOfType<scf::ForallOp>()) {
            return rewriter.notifyMatchFailure(genericOp, "already sliced");
        }

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

        return tile(rewriter, genericOp, peelGrouping_, 4, *slicingIter);
    }
}; // class ElementwisePattern

struct LinalgSlicingPass : public impl::LinalgSlicingBase<LinalgSlicingPass> {
    using LinalgSlicingBase<LinalgSlicingPass>::LinalgSlicingBase;

    LinalgSlicingOptions options_;

    LinalgSlicingPass(const LinalgSlicingOptions &options) : options_(options) {}

    void runOnOperation() override {
        if (TorqHw::get().getSliceCount() < 2)
            return;

        LLVM_DEBUG(llvm::dbgs() << "Linalg Slicing - START\n");

        auto funcOp = getOperation();
        MLIRContext *context = funcOp.getContext();

        RewritePatternSet patterns(context);

        patterns.add<Conv2DPattern<linalg::Conv2DNhwcHwcfOp>>(context, options_.peelGrouping, 3);
        patterns.add<Conv2DPattern<linalg::Conv2DNchwFchwOp>>(context, options_.peelGrouping, 1);

        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNhwcHwcOp>>(
            context, options_.peelGrouping, 3
        );
        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNchwChwOp>>(
            context, options_.peelGrouping, 1
        );

        // Elementwise generic ops: dynamically slice on leftmost non-unit dim
        patterns.add<ElementwisePattern>(context, options_.peelGrouping);

        // patterns.add<FullyConnectedPattern>(ctx);

        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps);

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
            assert(false && "failed");
            return signalPassFailure();
        }

        LLVM_DEBUG(llvm::dbgs() << "Linalg Slicing - DONE\n");
    }
}; // class LinalgSlicingPass

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLinalgSlicingPass(const LinalgSlicingOptions &options) {
    return std::make_unique<LinalgSlicingPass>(options);
}

// The first two runs split the work into 3 chunks, and the last run does the
// slicing. All splitting is done using scf tiling, but the chunks are
// scf::ForOp loops with exactly two iterations, that are immediately unrolled,
// and the slicing uses scf::ForallOp loops, and is not unrolled here.
//
// For S available slices, and grouping of G channels, we might need to split
// the work into at most 3 different chunks: the first chunk will be sliced evenly
// between all S slices; the second chunk will split evenly between some of the
// S slices (strictly less than S); and the last chunk will be smaller than G.
// For example, on HW with 4 slices, and grouping of 8, doing a conv2d with 92
// channels: first chunk will be 64 channels that will be sliced into 4 slices of
// 16 channels (64 = 4*8*2 = S*G*2), second chunk will be 24 channels that will
// be sliced into 3 slices of 8 channels (24 = 3*8 = k*G, where 0 < k < S), and
// the last chunk will be 4 channels (4 = 92-64-24).
void addLinalgSlicingPasses(OpPassManager &funcPm) {
    for (size_t i = 0; i < 2; ++i) {
        funcPm.addPass(createLinalgSlicingPass({.peelGrouping = true}));
        funcPm.addPass(createCanonicalizerPass());
    }
    funcPm.addPass(createLinalgSlicingPass({.peelGrouping = false}));
    funcPm.addPass(createCanonicalizerPass());
}

} // namespace mlir::syna::torq
