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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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
const int64_t kMinElementsForSlicing = 2000;

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
    if (tileSize == domainSize)
        return;

    if (failed(tileAndFuse(
            rewriter, op, tileSize, slicingIter, scf::SCFTilingOptions::LoopType::ForallOp
        ))) {
        op->emitError("failed to slice op to size");
        return;
    }
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
        op->emitError("failed to peel op");
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
    if (op->getParentOfType<scf::ForallOp>())
        return rewriter.notifyMatchFailure(op, "already sliced");

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

    if (!cast<ShapedType>(rootOp->getResult(0).getType()).hasStaticShape())
        return rewriter.notifyMatchFailure(op, "output shape is not static");

    // Skip ops where all operands are too small to benefit from slicing.
    bool anyLargeOperand = llvm::any_of(op->getOperandTypes(), [](Type t) {
        auto shaped = dyn_cast<ShapedType>(t);
        return shaped && shaped.hasStaticShape() &&
               shaped.getNumElements() >= kMinElementsForSlicing;
    });
    if (!anyLargeOperand)
        return rewriter.notifyMatchFailure(op, "tensor too small to benefit from slicing");

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

// BatchMatmul pattern. Slices on the N dimension (last dim of input B /
// output columns), distributing rows of the second input across hardware slices.
// For batch_matmul A[batch, M, K] x B[batch, K, N] -> C[batch, M, N],
// we slice B along N (and correspondingly C along N).
struct BatchMatmulPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
    using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::BatchMatmulOp matmulOp, PatternRewriter &rewriter) const override {
        if (isMarkedFuseGroup(matmulOp) && !isFuseGroupPrincipalOp(matmulOp))
            return rewriter.notifyMatchFailure(matmulOp, "not the principal operation");

        // Check that the output shape is fully static before calling getIterationDomain;
        // that call mutates the IR and would cause the rewrite driver to spin if invoked
        // on a dynamic-shaped op.
        if (!cast<ShapedType>(matmulOp->getResult(0).getType()).hasStaticShape())
            return rewriter.notifyMatchFailure(matmulOp, "output shape is not static");

        // Scan the output shape to find the slicing dimension. Using the output
        // shape (parallel dims only) avoids accidentally picking a reduction dim,
        // which must not be sliced. Output shape indices map directly to
        // iteration-domain indices because linalg orders parallel dims first.
        ArrayRef<int64_t> outputShape =
            cast<ShapedType>(matmulOp->getResult(0).getType()).getShape();

        std::optional<size_t> slicingIter;
        for (auto [i, size] : llvm::enumerate(outputShape)) {
            if (size > 1 && size >= 2 * kGrouping) {
                slicingIter = i;
                break;
            }
        }

        if (!slicingIter)
            return rewriter.notifyMatchFailure(matmulOp, "no suitable output dimension to slice");

        return peelAndSlice(rewriter, matmulOp, *slicingIter, kGrouping);
    }
}; // class BatchMatmulPattern

// True iff `op` is a pure-elementwise linalg.generic that is a legal member of a
// coalesced elementwise slicing region: all-parallel, projected-permutation
// indexing (linalg::isElementwise) and static shape.
//
// NOTE on narrowing trunc: the standalone narrowing-trunc guard declines to slice
// a STANDALONE narrowing-trunc generic (an i32->i16 rescale/quantize that must not
// be tiled on its own). That guard lives in ElementwisePattern::matchAndRewrite and
// is unchanged. Here we do NOT reject trunc-bearing generics, because (a) coalescing
// only acts on chains of >=2 ops with a single escaping leaf, so a standalone trunc
// op forms a size-1 region and is skipped; and (b) interior trunc ops are legitimate
// chain members -- e.g. tanh's float-exponent range-reduction block
// (bitcast/subi/trunci/index_cast/extui/addi/bitcast) sits in the middle of the
// polynomial chain. Excluding it would split the chain and defeat coalescing. When
// fused into one forall the trunc is tiled per-element, never sliced standalone, so
// the standalone-trunc concern does not arise.
bool isCoalescableElementwise(Operation *op) {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
        return false;
    if (!linalg::isElementwise(genericOp))
        return false;
    if (!cast<ShapedType>(genericOp->getResult(0).getType()).hasStaticShape())
        return false;
    return true;
}

// Map each fuse-group id to whether EVERY op carrying it is a coalescable
// pure-elementwise generic, computed in ONE walk of the function. A singleton
// (1 member) and a pure-elementwise pattern group (e.g. mul+clamp -- arith.mulf +
// cmpf/select, all elementwise) map to true; a group with ANY non-elementwise
// member (e.g. a conv/matmul fused group) maps to false -> that group is a hard
// region boundary. Built once so isDissolvableElementwise is an O(1) lookup per
// query instead of walking the whole function each time (the seed scan + BFS call
// it O(N) times).
llvm::DenseMap<int64_t, bool> buildGroupAllCoalescableMap(FunctionOpInterface funcOp) {
    llvm::DenseMap<int64_t, bool> groupAllCoalescable;
    funcOp->walk([&](Operation *op) {
        auto arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (!arr)
            return;
        bool coalescable = isCoalescableElementwise(op);
        for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>()) {
            auto [it, inserted] =
                groupAllCoalescable.try_emplace(fuseGroupAttr.getInt(), coalescable);
            if (!inserted)
                it->second = it->second && coalescable;
        }
    });
    return groupAllCoalescable;
}

// A legal coalescing-region member: a coalescable pure-elementwise generic that is
// EITHER ungrouped OR belongs to a fuse group ALL of whose members are themselves
// coalescable pure-elementwise generics. That admits both a length-1 SINGLETON
// (each polynomial step) AND a pure-elementwise PATTERN group such as the final
// mul+clamp (so the whole erf/tanh tail joins one sliced region). A group
// containing ANY non-elementwise member (conv/matmul fused groups) is rejected ->
// hard region boundary, and its size()==1 invariant is never disturbed.
bool isDissolvableElementwise(
    Operation *op, const llvm::DenseMap<int64_t, bool> &groupAllCoalescable
) {
    if (!isCoalescableElementwise(op))
        return false;
    if (!isMarkedFuseGroup(op))
        return true; // ungrouped
    ArrayAttr arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    if (!arr)
        return true;
    for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>())
        if (!groupAllCoalescable.lookup(fuseGroupAttr.getInt()))
            return false;
    return true;
}

// Coalesce a maximal connected region of pure-elementwise generics
// (the erf/tanh/mul polynomial chain) into ONE pattern fuse group so the whole
// region is sliced into a SINGLE scf.forall, keeping every intermediate
// LRAM-resident (one DRAM load -> LRAM chain -> one DRAM store per slice),
// matching the slicing-OFF behaviour. Without this, each polynomial step is a
// standalone principal -> its own forall -> a full-size DRAM round-trip per step
// (torq_hl.store explosion 1 -> 36 for erf).
//
// Region membership (isDissolvableElementwise):
//   - a coalescable pure-elementwise generic that is EITHER ungrouped OR in a
//     fuse group whose members are ALL coalescable pure-elementwise. This admits
//     both a length-1 singleton (each polynomial step) AND a pure-elementwise
//     pattern group such as the final mul+clamp, so the whole erf/tanh tail
//     joins one sliced region;
//   - grown over def-use edges to producer/consumer ops that are also
//     dissolvable. Multi-use values ARE allowed inside the region (the erf x^2
//     term feeds 5 muls); a use only matters if it escapes the region (handled
//     below as the leaf's external use).
// Boundary stops (never crossed):
//   - any non-coalescable op (reduction / matmul / conv / transpose / reshape) --
//     linalg::isElementwise is false for these;
//   - any fuse group with a non-elementwise member (a conv/matmul fused group):
//     dissolving it would break the assert(size()==1) invariant in peelAndSlice,
//     so it is left untouched and its result is a region input/output edge.
// Each member's existing fuseGroup (a singleton UID, or a pure-elementwise pattern
// group like mul+clamp) is stripped before the single shared id is set.
// The region's principal is its unique leaf (the op whose result is used outside
// the region or is the dispatch store operand). The leaf's own
// TORQ_FUSE_GROUP_ID becomes the shared group id, so isFuseGroupPrincipalOp
// returns it for the leaf automatically and every interior op hits the existing
// "not the principal" skip in ElementwisePattern.
void coalesceElementwiseChains(FunctionOpInterface funcOp) {
    MLIRContext *context = funcOp.getContext();
    // Precompute "are all members of this fuse group coalescable?" once, so the
    // membership test below is an O(1) lookup rather than a per-call walk.
    llvm::DenseMap<int64_t, bool> groupAllCoalescable = buildGroupAllCoalescableMap(funcOp);

    // Collect all candidate seeds in deterministic order.
    SmallVector<linalg::GenericOp> seeds;
    funcOp->walk([&](linalg::GenericOp g) {
        if (isDissolvableElementwise(g, groupAllCoalescable))
            seeds.push_back(g);
    });

    llvm::DenseSet<Operation *> assigned;
    for (linalg::GenericOp seed : seeds) {
        if (assigned.contains(seed.getOperation()))
            continue;

        // BFS the maximal connected coalescable region around `seed`.
        llvm::SetVector<Operation *> region;
        SmallVector<Operation *> work{seed.getOperation()};
        region.insert(seed.getOperation());
        while (!work.empty()) {
            Operation *cur = work.pop_back_val();

            // Walk to producers (defining ops of operands).
            for (Value operand : cur->getOperands()) {
                Operation *def = operand.getDefiningOp();
                if (!def || region.contains(def))
                    continue;
                if (!isDissolvableElementwise(def, groupAllCoalescable))
                    continue;
                region.insert(def);
                work.push_back(def);
            }
            // Walk to consumers (users of results).
            for (Value result : cur->getResults()) {
                for (Operation *user : result.getUsers()) {
                    if (region.contains(user))
                        continue;
                    if (!isDissolvableElementwise(user, groupAllCoalescable))
                        continue;
                    region.insert(user);
                    work.push_back(user);
                }
            }
        }

        // A single op is no different from the standalone path; skip.
        if (region.size() < 2)
            continue;

        // Find the region leaf: the op whose result is consumed only OUTSIDE the
        // region (escapes to the dispatch store or to an already-grouped op). A
        // well-formed elementwise chain has exactly one such leaf; if there are
        // several (the region forks to multiple external consumers) we bail out
        // for safety -- coalescing a fork could change materialization semantics.
        Operation *leaf = nullptr;
        bool multipleLeaves = false;
        for (Operation *op : region) {
            bool escapes = false;
            for (Value result : op->getResults()) {
                for (Operation *user : result.getUsers()) {
                    if (!region.contains(user)) {
                        escapes = true;
                        break;
                    }
                }
                if (escapes)
                    break;
            }
            if (escapes) {
                if (leaf) {
                    multipleLeaves = true;
                    break;
                }
                leaf = op;
            }
        }
        if (multipleLeaves || !leaf)
            continue;

        // Use the leaf's own per-op UID as the shared group id so the leaf is the
        // principal (isFuseGroupPrincipalOp(leaf) == leafFuseGroupAttr).
        auto leafFuseGroupAttr = leaf->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
        if (!leafFuseGroupAttr)
            continue;

        // Region members may carry a fuse group: a length-1 SINGLETON (own UID) or
        // a pure-elementwise PATTERN group (e.g. mul+clamp, whose members share
        // the principal's UID as fuseGroup). Strip every fuseGroup each member carries FIRST
        // (removeFuseGroupMarkingBackwards removes that fuseGroup from the backward cone;
        // those fuseGroups live only inside this region, so nothing outside is touched),
        // so all members are ungrouped before we assign the single shared [leafFuseGroupAttr].
        // This keeps the assert(size()==1) invariant in peelAndSlice intact.
        for (Operation *op : region) {
            if (auto arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
                SmallVector<int64_t> fuseGroups;
                for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>())
                    fuseGroups.push_back(fuseGroupAttr.getInt());
                for (int64_t fuseGroup : fuseGroups)
                    removeFuseGroupMarkingBackwards(op, fuseGroup);
            }
        }
        ArrayAttr groupAttr = ArrayAttr::get(context, {leafFuseGroupAttr});
        for (Operation *op : region) {
            op->setAttr(TORQ_FUSE_GROUP, groupAttr);
            assigned.insert(op);
        }
    }
}

// Elementwise pattern for linalg.generic ops. Picks the leftmost non-unit
// dimension to slice on, keeping inner dimensions contiguous in memory.
struct ElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        // If this op is part of a fuse group but is NOT the principal op, skip
        // it — the principal op drives tiling for the whole group.
        // Standalone elementwise ops (no fuse-group attr) are allowed through.
        if (isMarkedFuseGroup(genericOp) && !isFuseGroupPrincipalOp(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "not the principal operation");
        }

        // Only operate on elementwise (all-parallel) generic ops.
        if (!linalg::isElementwise(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "Not an elementwise operation");
        }

        // Downstream NSS lowering for standalone elementwise linalg.generic ops
        // only recognizes unary, binary, ternary/select, and a few named
        // special cases. Wider n-ary epilogues, such as quantized matmul
        // zero-point correction, stay on CSS/host and should not be sliced.
        if (genericOp.getNumDpsInputs() > 3) {
            return rewriter.notifyMatchFailure(
                genericOp, "n-ary elementwise generic has no NSS lowering, skipping slicing"
            );
        }

        // Skip linalg.generic ops that contain a narrowing integer/float
        // truncation (arith.trunci / arith.truncf) and are not part of any
        // fuse group (no "torq-fuse-group" array attr).  These are standalone
        // rescale/quantize ops (e.g. pooling: apply_scale -> clamp -> trunci,
        // i32->i16) that must not be independently sliced.  If the op does
        // carry "torq-fuse-group" it belongs to a pattern fuse group and the
        // principal op drives tiling for the whole group — allow it through.
        bool hasNarrowingTrunc = genericOp.getBody()
                                     ->walk([](Operation *op) -> WalkResult {
                                         if (isa<arith::TruncIOp, arith::TruncFOp>(op))
                                             return WalkResult::interrupt();
                                         return WalkResult::advance();
                                     })
                                     .wasInterrupted();
        if (hasNarrowingTrunc && !isMarkedFuseGroup(genericOp))
            return rewriter.notifyMatchFailure(
                genericOp,
                "standalone op with trunci/truncf not in any fuse group, skipping slicing"
            );

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

        // Coalesce pure-elementwise polynomial chains (erf/tanh/mul) into one fuse
        // group BEFORE the patterns below run, so each chain slices into a single
        // scf.forall (LRAM-resident) instead of one forall per op (DRAM round-trip
        // per step).
        coalesceElementwiseChains(funcOp);

        RewritePatternSet patterns(context);

        patterns.add<Conv2DPattern<linalg::Conv2DNhwcHwcfOp>>(context, 3);
        patterns.add<Conv2DPattern<linalg::Conv2DNchwFchwOp>>(context, 1);

        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNhwcHwcOp>>(context, 3);
        patterns.add<Conv2DPattern<linalg::DepthwiseConv2DNchwChwOp>>(context, 1);

        // Batch matmul: slice on the largest parallel output dimension
        patterns.add<BatchMatmulPattern>(context);

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
