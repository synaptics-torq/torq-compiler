// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// FoldUnitExtentWrapper Pass
//===----------------------------------------------------------------------===//
//
// Fixes #1412.
//
// A depthwise conv carries a channel-multiplier dimension M, so in linalg it
// starts as a rank-5 linalg.depthwise_conv_2d_nhwc_hwcm. When M=1, IREE 3.10's
// LinalgQuantizedConvToConvPass rewrites it into the plain rank-4
// linalg.depthwise_conv_2d_nhwc_hwc, dropping the M dimension. To leave the ops
// after it unchanged, it re-adds the size-1 M dim right away, leaving a reshape
// round-trip around the output cast:
//
//   %conv  = linalg.depthwise_conv_2d_nhwc_hwc ...  : tensor<1x16x16x2xf32>    // rank 4
//   %big   = tensor.expand_shape %conv ...          : tensor<1x16x16x2x1xf32>  // rank 5
//   %trunc = linalg.generic {truncf} ins(%big) ...  : tensor<1x16x16x2x1xbf16> // rank 5
//   %out   = tensor.collapse_shape %trunc ...       : tensor<1x16x16x2xbf16>   // rank 4
//
// With M=1 the size-1 dim holds no data and the expand/collapse undo each
// other, so this round-trip is leftover scaffolding. Upstream
// fold-unit-extent-dims / canonicalization would normally remove it, but in the
// torq pipeline that does not run before ConvertNhwcOpToNchwPass. That pass
// only handles rank-4 ops: it pulls the rank-5 generic into a conv group and
// then crashes when it builds a 4-entry transpose for a 5-D value (transposeType
// assert in ConversionUtils.cpp, exit 134).
//
// This pass runs just before ConvertNhwcOpToNchwPass and folds the reshape
// round-trip back into one lower-rank generic, so the next pass only ever sees
// rank 4. The match is rank-agnostic (see matchWrapperChain): it fires on any
// inverse expand/collapse pair that only inserts size-1 dims around an
// identity-map generic, not just the rank-4->5 shape above; every other op is
// left untouched. Because only size-1 dims are dropped, a real multiplier
// (M>1, a non-1 dim) is never folded.
//
// The fix lives in its own pass on purpose: ConvertNhwcOpToNchwPass keeps its
// original two-phase code, with no extra per-conv state to track.
//
// Before dispatch formation the pass also runs with rank restoration, folding
// the mirrored collapse -> generic -> expand round trip the importer leaves
// around a singleton-axis Gather -> unary -> Unsqueeze chain, so the op runs at
// the rank of its surrounding tensors.
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-fold-unit-extent-wrapper"

namespace mlir::syna::torq {

namespace {

// A reshape round trip around an elementwise generic, in data-flow order:
// `inner` reshapes the input by adding or removing unit (size-1) dims,
// `wrappedGeneric` is the wrapped op (e.g. a truncf), and `outer` restores the
// original shape. Two orders occur:
//   - expand -> generic -> collapse: the IREE depthwise wrapper described above;
//   - collapse -> generic -> expand: the importer's singleton-axis Gather ->
//     unary -> Unsqueeze chain, which runs the op at a reduced rank. Folding it
//     restores the rank, so it is only enabled before dispatch formation (see
//     createFoldUnitExtentWrapperPass): afterwards the per-dispatch collapses
//     are IREE's own unit-dim folding, which must not be undone.
struct WrapperChain {
    Operation *inner;
    linalg::GenericOp wrappedGeneric;
    Operation *outer;
};

// True if every indexing map of `generic` is the identity map. We only fold
// when this holds: an identity map reads each element in place, so dropping the
// size-1 dimension keeps the same result. A broadcast or permutation map could
// change the result, so those are left alone. (The looser linalg::isElementwise
// / LinalgOp::hasOnlyProjectedPermutations accept permutations, so they cannot
// stand in here -- the fold rebuilds with fresh identity maps.)
bool hasIdentityIndexingMaps(linalg::GenericOp generic) {
    return llvm::all_of(generic.getIndexingMapsArray(), [](AffineMap map) {
        return map.isIdentity();
    });
}

// True if the expand/collapse `reshape` only inserts or removes size-1
// dimensions (no real dimension split or merge). computeRankReductionMask
// returns the dropped indices iff the smaller shape is the larger one with only
// size-1 entries erased. A real split (e.g. 6 -> 2x3) or a multiplier dim (M>1)
// leaves a non-1 difference, so we reject -- we only ever move dims that hold
// no data.
bool movesOnlyUnitDims(Operation *reshape) {
    auto src = cast<RankedTensorType>(reshape->getOperand(0).getType());
    auto dst = cast<RankedTensorType>(reshape->getResult(0).getType());
    if (src.getRank() < dst.getRank())
        std::swap(src, dst);
    return computeRankReductionMask(src.getShape(), dst.getShape()).has_value();
}

// Decide whether `generic` is wrapped in a unit-dim reshape round trip. On a
// match, return the three ops; otherwise return nullopt. The checks are ordered
// cheapest-first so unrelated ops are rejected quickly. The match is
// rank-agnostic: it only requires the two reshapes to move unit dims and the
// outer one to restore the original shape.
std::optional<WrapperChain> matchWrapperChain(linalg::GenericOp generic, bool restoreRank) {
    // One ranked result, and its maps must be identity so moving the size-1
    // dimensions is safe (a broadcast/permutation map could change it). The
    // rebuilt op gets a fresh init and loses its iteration rank, so the body
    // must not read the init or the loop indices.
    if (generic.getNumResults() != 1)
        return std::nullopt;
    if (!isa<RankedTensorType>(generic.getResult(0).getType()))
        return std::nullopt;
    if (!hasIdentityIndexingMaps(generic))
        return std::nullopt;
    if (generic.getNumDpsInputs() != 1 || generic.hasIndexSemantics() ||
        generic.payloadUsesValueFromOperand(generic.getDpsInitOperand(0)))
        return std::nullopt;

    // It reads exactly one input, and that input is an expand_shape (or, when
    // restoring rank, a collapse_shape).
    Operation *inner = generic.getDpsInputs()[0].getDefiningOp();
    if (!isa_and_nonnull<tensor::ExpandShapeOp>(inner) &&
        !(restoreRank && isa_and_nonnull<tensor::CollapseShapeOp>(inner)))
        return std::nullopt;

    // It has exactly one user, and that user is the opposite reshape.
    if (!generic.getResult(0).hasOneUse())
        return std::nullopt;
    Operation *outer = *generic.getResult(0).getUsers().begin();
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(outer) ||
        outer->getName() == inner->getName())
        return std::nullopt;

    // The outer reshape returns to the inner one's source shape and both only
    // add/remove size-1 dims, so the round trip is a value identity. A real
    // multiplier dim (M>1) is non-1 and is rejected here.
    if (cast<RankedTensorType>(outer->getResult(0).getType()).getShape() !=
        cast<RankedTensorType>(inner->getOperand(0).getType()).getShape())
        return std::nullopt;
    if (!movesOnlyUnitDims(inner) || !movesOnlyUnitDims(outer))
        return std::nullopt;

    return WrapperChain{inner, generic, outer};
}

// Build the folded replacement generic. It reads the value that feeds the inner
// reshape, writes a fresh tensor the type of the outer reshape result, and
// reuses the body of the original generic unchanged.
linalg::GenericOp buildFoldedGeneric(WrapperChain chain, OpBuilder &builder) {
    Location loc = chain.wrappedGeneric.getLoc();
    MLIRContext *ctx = builder.getContext();

    auto resultType = cast<RankedTensorType>(chain.outer->getResult(0).getType());
    int64_t rank = resultType.getRank();
    Value init =
        tensor::EmptyOp::create(builder, loc, resultType.getShape(), resultType.getElementType())
            .getResult();

    // One identity map for the input and one for the output; all dims parallel.
    SmallVector<AffineMap, 2> maps(2, AffineMap::getMultiDimIdentityMap(rank, ctx));
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

    auto folded = linalg::GenericOp::create(
        builder, loc, /*resultTensorTypes=*/TypeRange{resultType},
        /*inputs=*/ValueRange{chain.inner->getOperand(0)}, /*outputs=*/ValueRange{init}, maps,
        iterators
    );

    // Copy the original op's body (e.g. the truncf) into the new op.
    IRMapping mapping;
    chain.wrappedGeneric.getRegion().cloneInto(&folded.getRegion(), mapping);
    return folded;
}

// Rewrite one matched wrapper into a single generic at the outer shape. Each
// match is handled on its own; the narrow match in matchWrapperChain is the
// only gate, so no group-level bookkeeping is needed.
struct FoldUnitExtentWrapperPattern : OpRewritePattern<linalg::GenericOp> {
    FoldUnitExtentWrapperPattern(MLIRContext *ctx, bool restoreRank)
        : OpRewritePattern(ctx), restoreRank(restoreRank) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp generic, PatternRewriter &rewriter) const override {
        std::optional<WrapperChain> chain = matchWrapperChain(generic, restoreRank);
        if (!chain)
            return failure();

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(chain->wrappedGeneric);
        linalg::GenericOp folded = buildFoldedGeneric(*chain, rewriter);

        // Point the outer reshape's users at the new op, then delete the
        // wrapper ops. The inner reshape may feed other ops too, so erase it
        // only when nothing reads it anymore.
        rewriter.replaceOp(chain->outer, folded.getResult(0));
        rewriter.eraseOp(chain->wrappedGeneric);
        if (chain->inner->use_empty())
            rewriter.eraseOp(chain->inner);

        LLVM_DEBUG(
            llvm::dbgs() << "[fold-unit-extent-wrapper] folded wrapper to "
                         << folded.getResult(0).getType() << "\n"
        );
        return success();
    }

    bool restoreRank;
};

struct FoldUnitExtentWrapperPass : impl::FoldUnitExtentWrapperBase<FoldUnitExtentWrapperPass> {
    explicit FoldUnitExtentWrapperPass(bool restoreRank) : restoreRank(restoreRank) {}

    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<FoldUnitExtentWrapperPattern>(&getContext(), restoreRank);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }

    bool restoreRank;
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldUnitExtentWrapperPass(bool restoreRank
) {
    return std::make_unique<FoldUnitExtentWrapperPass>(restoreRank);
}

} // namespace mlir::syna::torq
