// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TilingUtils.h"

#include "torq/Utils/TorqHw.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

#define DEBUG_TYPE "torq-linalg-slicing"

namespace mlir::syna::torq {

namespace {

template <class LinalgOp> struct LinalgPattern : public OpRewritePattern<LinalgOp> {
    bool peelGrouping_;
    int64_t grouping_;
    size_t slicingIter_;

    LinalgPattern(MLIRContext *context, bool peelGrouping, int64_t grouping, size_t slicingIter)
        : OpRewritePattern<LinalgOp>(context), peelGrouping_(peelGrouping), grouping_(grouping),
          slicingIter_(slicingIter) {}

    // return the required tile size at the current phase.
    int64_t getTileSize(int64_t domainSize) const {
        assert(domainSize > grouping_);

        if (peelGrouping_) {
            // We want to split the domain in to two tiles, the first tile
            // should be divisible by grouping_ so it can utilize multiple
            // slices at the same time (they can only be utilized when they all
            // have exactly the same number of rows, and this number is a
            // multiple of grouping_), and the second tile is whatever is left.
            int64_t remainder;
            if (domainSize >= TorqHw::get().getSliceCount() * grouping_) {
                // If we have enough rows, the first tile will utilize all the
                // slices, and we might need to sub-tile the second tile later.
                remainder = domainSize % (TorqHw::get().getSliceCount() * grouping_);
            }
            else {
                // If there are not enough rows to utilize all the slices, the
                // first tile will utilize as many slices as possible, and the
                // second tile will be the remainder.
                remainder = domainSize % grouping_;
            }

            // Return the (first) tile size.
            return domainSize - remainder;
        }

        // We need to tile the rows evenly between the slices. We assume the
        // pass has already been executed in the peelGrouping_ mode twice to
        // guarantee that now the rows can be properly distributed between the
        // slices without a remainder.

        if (domainSize < TorqHw::get().getSliceCount() * grouping_) {
            // Not enough rows to utilize all slices, so only some of them will
            // get a tile of size grouping_.
            assert(domainSize % grouping_ == 0);
            return grouping_;
        }

        // We have enough rows to utilize all slices.

        assert(domainSize % (TorqHw::get().getSliceCount() * grouping_) == 0);

        // this tile size will be a multiple of grouping_
        return domainSize / TorqHw::get().getSliceCount();
    }

    LogicalResult matchAndRewrite(LinalgOp linalgOp, PatternRewriter &rewriter) const override {
        // We only operate on the principal operation of pattern fuse groups.
        IntegerAttr fuseGroup = isFuseGroupPrincipalOp(linalgOp);
        if (!fuseGroup)
            return failure();

        // Do nothing if the containing operation is a forall (already sliced)
        if (llvm::isa_and_nonnull<scf::ForallOp>(linalgOp->getParentOp())) {
            LLVM_DEBUG(llvm::dbgs() << "already forall\n");
            return failure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << (peelGrouping_ ? "Peeling" : "Slicing") << " [";
            linalgOp->print(llvm::dbgs(), OpPrintingFlags().elideLargeElementsAttrs());
            llvm::dbgs() << "]\n";
        });

        // Tiling will start from the output of the pattern fuse group.
        Operation *rootOp = getFuseGroupOutputOp(linalgOp, fuseGroup);

        // We can only slice TilingInterface ops
        TilingInterface rootTi = dyn_cast<TilingInterface>(rootOp);
        assert(rootTi && "pattern fuse group ops are expected to be TilingInterface");

        if (cast<ShapedType>(rootOp->getResultTypes()[0]).getNumDynamicDims() != 0) {
            LLVM_DEBUG(llvm::dbgs() << "output shape is not static\n");
            return failure();
        }

        // The untiled domain sizes
        auto [iterDomainOffsets, iterDomainSizes, iterDomainStrides] =
            getOffsetsSizesAndStrides(rootTi.getIterationDomain(rewriter));

        std::optional<SmallVector<int64_t>> iterDomainConstSizes =
            getConstantIntValues(iterDomainSizes);
        assert(iterDomainConstSizes && "iteration domain sizes are not constants");
        // Can't return failure() here as we already used the rewriter,
        // which will cause the driver to spin.

        LLVM_DEBUG({
            llvm::dbgs() << "Iter Domains: ";
            llvm::interleave(*iterDomainConstSizes, llvm::dbgs(), "x");
            llvm::dbgs() << "\n";
        });

        assert(slicingIter_ < iterDomainConstSizes->size());

        int64_t domainSize = (*iterDomainConstSizes)[slicingIter_];
        if (domainSize <= 2 * grouping_) {
            // There are not enough rows to utilize even two slices at the same time.
            LLVM_DEBUG(llvm::dbgs() << "domain size too small\n");
            return failure();
        }

        scf::SCFTileAndFuseOptions options;

        // Initially set all tile sizes to 0 (don't tile).
        SmallVector<OpFoldResult> tileSizes(iterDomainSizes.size(), rewriter.getIndexAttr(0));

        int64_t tileSize = getTileSize(domainSize);
        if (tileSize == domainSize)
            return failure();

        tileSizes[slicingIter_] = getAsIndexOpFoldResult(rewriter.getContext(), tileSize);
        options.tilingOptions.setTileSizes(tileSizes);

        options.tilingOptions.setLoopType(
            peelGrouping_ ? scf::SCFTilingOptions::LoopType::ForOp
                          : scf::SCFTilingOptions::LoopType::ForallOp
        );

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

        FailureOr<scf::SCFTileAndFuseResult> tiledResults =
            scf::tileConsumerAndFuseProducersUsingSCF(rewriter, rootTi, options);

        if (failed(tiledResults)) {
            rootTi->emitError("failed to slice a pattern fuse group");
            LLVM_DEBUG(assert(false));
            return failure();
        }

        applyTiledResults(rewriter, rootOp, *tiledResults);

        eraseForward(rewriter, linalgOp);

        if (peelGrouping_) {
            assert(tiledResults->loops.size() == 1);
            auto unrollResult = loopUnrollByFactor(cast<scf::ForOp>(tiledResults->loops[0]), 2);
            assert(llvm::succeeded(unrollResult));
        }

        LLVM_DEBUG(llvm::dbgs() << "success!!!\n");

        return success();
    }
}; // class LinalgPattern

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

        patterns.add<LinalgPattern<linalg::Conv2DNhwcHwcfOp>>(context, options_.peelGrouping, 4, 3);
        patterns.add<LinalgPattern<linalg::Conv2DNchwFchwOp>>(context, options_.peelGrouping, 4, 1);

        patterns.add<LinalgPattern<linalg::DepthwiseConv2DNhwcHwcOp>>(
            context, options_.peelGrouping, 4, 3
        );
        patterns.add<LinalgPattern<linalg::DepthwiseConv2DNchwChwOp>>(
            context, options_.peelGrouping, 4, 1
        );

        // patterns.add<FullyConnectedPattern>(ctx);

        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns), config))) {
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

// The first two runs split the work to 3 chunks, and the last run does the
// slicing. All splitting is done using scf tiling, but the chunks are
// scf::ForOp loops with exactly two iterations, that are immediately unrolled,
// and the slicing uses scf::ForallOp loops, and is not unrolled here.
//
// For S available slices, and grouping of G channels, we might need to split
// the work to at most 3 different chunks: the first chunk will be sliced evenly
// between all S slices; the second chunk will split evenly between some of the
// S slices (strictly less than S); and the last chunk will be smaller than G.
// For example, on HW with 4 slices, and grouping of 8, doing a conv2d with 92
// channels: first chunk will be 64 channels that will be sliced to 4 slices of
// 16 channels (64 = 4*8*2 = S*G*2), second chunk will be 24 channels that will
// be sliced to 3 slices of 8 channels (24 = 3*8 = k*G, where 0 < k < S), and
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
