// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TilingUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include <type_traits>

#define DEBUG_TYPE "torq-linalg-slicing"

namespace mlir::syna::torq {

namespace {

const int64_t kMinElementsForSlicing = 2000;

// Return tile size for peeling.
int64_t getPeelingTileSize(int64_t domainSize, int64_t grouping, int64_t sliceCount) {
    assert(domainSize > grouping);
    assert(sliceCount > 0);

    // We want to split the domain into two or three tiles, the first tile
    // should be divisible by grouping so it can utilize multiple slices at the
    // same time, and the second tile is whatever is left.
    int64_t remainder;
    if (domainSize >= sliceCount * grouping) {
        // If we have enough rows, the first tile will utilize all the
        // slices, and we might need to sub-tile the second tile, later.
        remainder = domainSize % (sliceCount * grouping);
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
int64_t getSlicingTileSize(int64_t domainSize, int64_t grouping, int64_t sliceCount) {
    assert(domainSize % grouping == 0);
    assert(domainSize >= grouping);
    assert(sliceCount > 0);

    // We need to tile the rows evenly between the slices. We assume the peeling
    // was done to guarantee that now the rows can be properly distributed
    // between the slices without a remainder.

    if (domainSize < sliceCount * grouping) {
        // Not enough rows to utilize all slices, so only some of them will
        // get a tile of size grouping.
        return grouping;
    }

    // We have enough rows to utilize all slices.

    assert(domainSize % (sliceCount * grouping) == 0);

    // This tile size will be a multiple of grouping.
    return domainSize / sliceCount;
}

scf::SCFTileAndFuseOptions::ControlFnTy getControlFn(IntegerAttr fuseGroup, size_t slicingIter) {
    return [fuseGroup, slicingIter](
               mlir::tensor::ExtractSliceOp sliceOp, OpResult producerOpResult,
               bool isDestinationOperand
           ) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        auto producerGroups =
            producerOpResult.getOwner()->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (producerGroups && llvm::is_contained(producerGroups, fuseGroup)) {
            return scf::SCFTileAndFuseOptions::ControlFnResult{false};
        }

        if (auto padOp = dyn_cast<tensor::PadOp>(producerOpResult.getOwner())) {
            auto staticOffsets = sliceOp.getStaticOffsets();
            if (slicingIter < staticOffsets.size() &&
                ShapedType::isDynamic(staticOffsets[slicingIter])) {
                auto lowPads = padOp.getStaticLow();
                auto highPads = padOp.getStaticHigh();
                if (slicingIter < lowPads.size() && slicingIter < highPads.size() &&
                    lowPads[slicingIter] == 0 && highPads[slicingIter] == 0) {
                    return scf::SCFTileAndFuseOptions::ControlFnResult{false};
                }
            }
        }

        if (!fuseGroup)
            return std::nullopt;

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
    options.setFusionControlFn(getControlFn(fuseGroup, slicingIter));

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
    size_t slicingIter, int64_t grouping, int64_t sliceCount
) {
    int64_t tileSize = getSlicingTileSize(domainSize, grouping, sliceCount);
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
    size_t slicingIter, int64_t grouping, int64_t sliceCount
) {
    if (domainSize < 2 * grouping)
        return;

    int64_t tileSize = getPeelingTileSize(domainSize, grouping, sliceCount);

    if (tileSize == domainSize) {
        sliceToSize(rewriter, op, domainSize, iterDomCount, slicingIter, grouping, sliceCount);
        return;
    }

    auto tiledResults =
        tileAndFuse(rewriter, op, tileSize, slicingIter, scf::SCFTilingOptions::LoopType::ForOp);
    if (failed(tiledResults)) {
        op->emitError("failed to peel op");
        return;
    }
    assert(tiledResults->loops.size() == 1);

    if (failed(loopUnrollByFactor(cast<scf::ForOp>(tiledResults->loops[0]), 2))) {
        LLVM_DEBUG(assert(false));
        op->emitError("failed to peel");
        return;
    }
}

// Follows `value` back through slices and elementwise ops with a single live
// input and returns the value the chain starts from.
static Value elementwiseChainSource(Value value) {
    while (Operation *def = value.getDefiningOp()) {
        if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(def)) {
            value = sliceOp.getSource();
            continue;
        }
        auto linalgOp = dyn_cast<linalg::LinalgOp>(def);
        if (!linalgOp || !linalg::isElementwise(linalgOp))
            break;
        SmallVector<Value> liveInputs;
        for (Value input : linalgOp.getDpsInputs()) {
            if (!isConstantLikeProducer(input))
                liveInputs.push_back(input);
        }
        if (liveInputs.size() != 1)
            break;
        value = liveInputs.front();
    }
    return value;
}

LogicalResult peelAndSlice(
    PatternRewriter &rewriter, Operation *op, size_t slicingIter, int64_t grouping,
    int64_t sliceCount
) {
    // Do nothing if the containing operation is a forall (already sliced)
    if (op->getParentOfType<scf::ForallOp>())
        return rewriter.notifyMatchFailure(op, "already sliced");

    // If fuseGroup is not nullptr, fuse all the members of the pattern fuse group
    // (op is assumed to be the principal op of the group); otherwise, tile op only.
    IntegerAttr fuseGroup = isFuseGroupPrincipalOp(op);

    // Skip slicing fuse groups that contain a residual add (two live activations
    // from different branches). Bias adds (one live input plus a constant-like
    // broadcast) are safe to slice once constant pipelines are kept out of the
    // group, and so are gates like SiLU's x * sigmoid(x), whose live inputs are
    // elementwise functions of one value. Residual groups still break dominance
    // if pulled into scf.forall (ConvTranspose skip-connects).
    if (fuseGroup) {
        bool hasResidualAdd = false;
        op->getParentOp()->walk([&](Operation *candidate) {
            auto groups = candidate->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            if (!groups || !llvm::is_contained(groups, fuseGroup))
                return WalkResult::advance();
            if (auto linalgOp = dyn_cast<linalg::LinalgOp>(candidate)) {
                if (linalg::isElementwise(linalgOp) && linalgOp.getNumDpsInputs() > 1) {
                    SmallVector<Value> sources;
                    for (Value input : linalgOp.getDpsInputs()) {
                        if (!isConstantLikeProducer(input))
                            sources.push_back(elementwiseChainSource(input));
                    }
                    if (sources.size() > 1 && !llvm::all_equal(sources)) {
                        hasResidualAdd = true;
                        return WalkResult::interrupt();
                    }
                }
            }
            return WalkResult::advance();
        });
        if (hasResidualAdd)
            return rewriter.notifyMatchFailure(
                op, "skip slicing fuse group that includes a residual add"
            );
    }

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
        rewriter, op, domainSize, iterDomainConstSizes->size(), slicingIter, grouping, sliceCount
    );

    return success();
}

// True when this conv stays on one slice: TileAndFuse marked it (see
// kSingleSliceAttr), or its weight is heavy and its output channels do not split
// evenly across the slices, so the peeled remainder would load the weights once
// more. ConvPattern skips slicing such a conv; ElementwisePattern mirrors it so
// a rescale that is the conv's sole consumer stays full, preserving the conv's
// single-use i32 output that lets it fuse to an i8 torq_hl.conv2d.
static bool convKeptSingleSlice(Operation *op, int64_t sliceCount) {
    auto conv = dyn_cast_if_present<linalg::Conv2DNchwFchwOp>(op);
    if (!conv)
        return false;
    if (conv->hasAttr(kSingleSliceAttr))
        return true;
    int64_t outChannels = cast<ShapedType>(conv->getResult(0).getType()).getDimSize(1);
    return isDualSliceWeightHeavyConv(conv, sliceCount) &&
           (ShapedType::isDynamic(outChannels) || outChannels % (sliceCount * kGrouping) != 0);
}

template <class ConvOp> struct ConvPattern : public OpRewritePattern<ConvOp> {
    size_t slicingIter_;
    int64_t sliceCount_;

    ConvPattern(MLIRContext *context, size_t slicingIter, int64_t sliceCount)
        : OpRewritePattern<ConvOp>(context), slicingIter_(slicingIter), sliceCount_(sliceCount) {}

    LogicalResult matchAndRewrite(ConvOp conv2DOp, PatternRewriter &rewriter) const override {
        // TODO: not all conv2ds are part of a pattern fuse group.
        if (!isFuseGroupPrincipalOp(conv2DOp)) {
            return rewriter.notifyMatchFailure(conv2DOp, "not the principal operation");
        }

        if (convKeptSingleSlice(conv2DOp, sliceCount_))
            return rewriter.notifyMatchFailure(conv2DOp, "kept on one slice");

        return peelAndSlice(rewriter, conv2DOp, slicingIter_, kGrouping, sliceCount_);
    }
}; // class Conv2DPattern

// BatchMatmul pattern. Slices on the N dimension (last dim of input B /
// output columns), distributing rows of the second input across hardware slices.
// For batch_matmul A[batch, M, K] x B[batch, K, N] -> C[batch, M, N],
// we slice B along N (and correspondingly C along N).
struct BatchMatmulPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
    int64_t sliceCount_;

    BatchMatmulPattern(MLIRContext *context, int64_t sliceCount)
        : OpRewritePattern<linalg::BatchMatmulOp>(context), sliceCount_(sliceCount) {}

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

        return peelAndSlice(rewriter, matmulOp, *slicingIter, kGrouping, sliceCount_);
    }
}; // class BatchMatmulPattern

// Minimum contiguous elements that must remain inside one transpose slice chunk.
constexpr int64_t kMinTransposeSliceRun = 64;

// Transpose pattern. Slices a standalone linalg.transpose on the outermost
// dimension that splits evenly and leaves a long contiguous run in each chunk.
// A transpose adopted by FuseTransposeIntoGroupPass reaches this pattern as a
// group member, not a principal, and is skipped.
struct TransposePattern : public OpRewritePattern<linalg::TransposeOp> {
    int64_t sliceCount_;

    TransposePattern(MLIRContext *context, int64_t sliceCount)
        : OpRewritePattern<linalg::TransposeOp>(context), sliceCount_(sliceCount) {}

    LogicalResult
    matchAndRewrite(linalg::TransposeOp transposeOp, PatternRewriter &rewriter) const override {
        if (getTargetExecutor(transposeOp) == torq_hl::Executor::Host)
            return rewriter.notifyMatchFailure(transposeOp, "host-executed operation");

        if (isMarkedFuseGroup(transposeOp) && !isFuseGroupPrincipalOp(transposeOp))
            return rewriter.notifyMatchFailure(transposeOp, "not the principal operation");

        auto shaped = dyn_cast<ShapedType>(transposeOp->getResult(0).getType());
        if (!shaped || !shaped.hasStaticShape())
            return rewriter.notifyMatchFailure(transposeOp, "output shape is not static");

        // Take the outermost dimension that splits evenly and still leaves a long
        // contiguous run inside each chunk.
        const int64_t chunk = sliceCount_ * kGrouping;
        ArrayRef<int64_t> shape = shaped.getShape();
        std::optional<size_t> slicingIter;
        for (size_t dim = 0; dim < shape.size(); ++dim) {
            if (shape[dim] < 2 * kGrouping || shape[dim] % chunk != 0)
                continue;
            int64_t run = 1;
            for (size_t inner = dim + 1; inner < shape.size(); ++inner)
                run *= shape[inner];
            if (run < kMinTransposeSliceRun)
                continue;
            slicingIter = dim;
            break;
        }
        if (!slicingIter)
            return rewriter.notifyMatchFailure(transposeOp, "no dimension splits into long runs");

        return peelAndSlice(rewriter, transposeOp, *slicingIter, kGrouping, sliceCount_);
    }
}; // struct TransposePattern

// The producer conv of this rescale group, when it is a Conv2DNchwFchwOp kept
// full by convKeptSingleSlice and read by the group as its single live
// entry. Slicing the group would turn that single-use conv output into a
// per-slice extract_slice fan-out, which stops Conv2dConvert from fusing the
// epilogue into an i8 torq_hl.conv2d and forces the full i32 conv activation to
// be pinned in LRAM. A conv that IS dual-sliced is not returned here, so its
// rescale is sliced with it; the decision depends only on the conv, so it is
// consistent with ConvPattern regardless of pattern firing order.
//
// ElementwisePattern already skips a rescale that is a non-principal member of
// the conv's own group, which is how conv marking places it. This covers the
// other case: a rescale that is the principal of its own group, or in no group.
static Operation *
findBlockedConvProducer(Operation *op, IntegerAttr fuseGroup, int64_t sliceCount) {
    SmallVector<Operation *, 8> groupOps;
    if (fuseGroup) {
        // Walk the whole parent: group members are not always adjacent. The marking
        // (markFuseGroupBackward) goes through ops it does not mark, such as a skipped
        // pad or a compute op with a side-branch input.
        op->getParentOp()->walk([&](Operation *candidate) {
            auto groups = candidate->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
            if (groups && llvm::is_contained(groups, fuseGroup))
                groupOps.push_back(candidate);
            return WalkResult::advance();
        });
    }
    else {
        groupOps.push_back(op);
    }

    for (Operation *member : groupOps) {
        for (Value operand : member->getOperands()) {
            Operation *def = operand.getDefiningOp();
            if (!def || llvm::is_contained(groupOps, def))
                continue;
            if (!operand.hasOneUse())
                continue;
            if (convKeptSingleSlice(def, sliceCount))
                return def;
        }
    }
    return nullptr;
}

// Elementwise pattern for linalg.generic ops. Picks the leftmost non-unit
// dimension to slice on, keeping inner dimensions contiguous in memory.
struct ElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    int64_t sliceCount_;

    ElementwisePattern(MLIRContext *context, int64_t sliceCount)
        : OpRewritePattern<linalg::GenericOp>(context), sliceCount_(sliceCount) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        // Host-executed elementwise ops are compile-time helper/rescale paths
        // and should not be tiled/sliced.
        if (getTargetExecutor(genericOp) == torq_hl::Executor::Host)
            return rewriter.notifyMatchFailure(genericOp, "host-executed operation");

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

        // Skip gather / table-lookup ops: their body reads a tensor via
        // tensor.extract, which is a hidden data-dependent operand that is NOT
        // sliced when the output tile is sliced. linalg::isElementwise is true
        // for them (projected-permutation maps, all-parallel iterators), but
        // tiling here would read the whole extracted tensor per tile. Leave them
        // un-tiled so LinalgToTorqHL lowers them to torq_hl.gather/torq_hl.table,
        // which the downstream NSS lowering tiles to the LRAM/DTCM budget.
        bool hasExtract =
            genericOp.getBody()->walk([](tensor::ExtractOp) { return WalkResult::interrupt(); }
            ).wasInterrupted();
        if (hasExtract)
            return rewriter.notifyMatchFailure(
                genericOp, "gather/table extract is not sliceable as elementwise, skipping slicing"
            );

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

        // A conv kept full by the dual-slice weight guard must not have its
        // single-use i32 rescale epilogue sliced, or the conv can no longer
        // fuse to an i8 torq_hl.conv2d and the full i32 activation is pinned.
        if (findBlockedConvProducer(genericOp, isFuseGroupPrincipalOp(genericOp), sliceCount_)) {
            return rewriter.notifyMatchFailure(
                genericOp, "rescale of a non-dual-sliceable conv, keep full so the conv fuses to i8"
            );
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

        return peelAndSlice(rewriter, genericOp, *slicingIter, kGrouping, sliceCount_);
    }
}; // class ElementwisePattern

// Reciprocal pattern. LinalgToTorqHL lowers linalg.reciprocal directly to a table
// lookup, so it reaches slicing as a named op that ElementwisePattern (which only sees
// linalg.generic) would leave on one slice. Slice it the same way: on the leftmost
// non-unit dimension.
struct ReciprocalPattern : public OpRewritePattern<linalg::ReciprocalOp> {
    int64_t sliceCount_;

    ReciprocalPattern(MLIRContext *context, int64_t sliceCount)
        : OpRewritePattern<linalg::ReciprocalOp>(context), sliceCount_(sliceCount) {}

    LogicalResult
    matchAndRewrite(linalg::ReciprocalOp reciprocalOp, PatternRewriter &rewriter) const override {
        if (getTargetExecutor(reciprocalOp) == torq_hl::Executor::Host)
            return rewriter.notifyMatchFailure(reciprocalOp, "host-executed operation");

        if (isMarkedFuseGroup(reciprocalOp) && !isFuseGroupPrincipalOp(reciprocalOp))
            return rewriter.notifyMatchFailure(reciprocalOp, "not the principal operation");

        std::optional<int64_t> slicingIter;
        ArrayRef<int64_t> shape = cast<ShapedType>(reciprocalOp->getResultTypes()[0]).getShape();
        for (auto [i, size] : llvm::enumerate(shape)) {
            if (size > 1) {
                slicingIter = i;
                break;
            }
        }
        if (!slicingIter)
            return rewriter.notifyMatchFailure(reciprocalOp, "no dimension to slice");

        return peelAndSlice(rewriter, reciprocalOp, *slicingIter, kGrouping, sliceCount_);
    }
}; // struct ReciprocalPattern

struct LinalgSlicingPass : public impl::LinalgSlicingBase<LinalgSlicingPass> {
    mlir::OpPassManager canonicalizer_;

    LinalgSlicingPass() { canonicalizer_.addPass(createFuseGroupPreservingCanonicalizerPass()); }

    explicit LinalgSlicingPass(const LinalgSlicingOptions &options) {
        this->sliceCount = options.sliceCount;
        canonicalizer_.addPass(createFuseGroupPreservingCanonicalizerPass());
    }

    LinalgSlicingPass(const LinalgSlicingPass &pass)
        : impl::LinalgSlicingBase<LinalgSlicingPass>(pass) {
        this->sliceCount = pass.sliceCount;
        canonicalizer_.addPass(createFuseGroupPreservingCanonicalizerPass());
    }

    void runOnOperation() override {
        if (this->sliceCount < 2)
            return;

        LLVM_DEBUG(llvm::dbgs() << "Linalg Slicing - START\n");

        auto funcOp = getOperation();
        MLIRContext *context = funcOp.getContext();

        RewritePatternSet patterns(context);

        patterns.add<ConvPattern<linalg::Conv2DNhwcHwcfOp>>(
            context, SlicingIterationDomainIndex::Conv2DNhwcHwcfOp, this->sliceCount
        );
        patterns.add<ConvPattern<linalg::Conv2DNchwFchwOp>>(
            context, SlicingIterationDomainIndex::Conv2DNchwFchwOp, this->sliceCount
        );

        patterns.add<ConvPattern<linalg::DepthwiseConv2DNhwcHwcOp>>(
            context, SlicingIterationDomainIndex::DepthwiseConv2DNhwcHwcOp, this->sliceCount
        );
        patterns.add<ConvPattern<linalg::DepthwiseConv2DNchwChwOp>>(
            context, SlicingIterationDomainIndex::DepthwiseConv2DNchwChwOp, this->sliceCount
        );

        patterns.add<ConvPattern<linalg::PoolingNhwcMaxOp>>(
            context, SlicingIterationDomainIndex::PoolingNhwcMaxOp, this->sliceCount
        );
        patterns.add<ConvPattern<linalg::PoolingNchwMaxOp>>(
            context, SlicingIterationDomainIndex::PoolingNchwMaxOp, this->sliceCount
        );
        patterns.add<ConvPattern<linalg::PoolingNcwMaxOp>>(
            context, SlicingIterationDomainIndex::PoolingNcwMaxOp, this->sliceCount
        );

        // Batch matmul: slice on the largest parallel output dimension
        patterns.add<BatchMatmulPattern>(context, this->sliceCount);

        // TODO: support matmul too
        // Elementwise generic ops: dynamically slice on leftmost non-unit dim
        patterns.add<TransposePattern>(context, this->sliceCount);
        patterns.add<ElementwisePattern>(context, this->sliceCount);
        patterns.add<ReciprocalPattern>(context, this->sliceCount);

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
std::unique_ptr<InterfacePass<FunctionOpInterface>> createLinalgSlicingPass(int64_t sliceCount) {
    return std::make_unique<LinalgSlicingPass>(LinalgSlicingOptions{sliceCount});
}

} // namespace mlir::syna::torq
