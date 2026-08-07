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

// The three ops IREE adds around a depthwise conv, in data-flow order:
// `expand` grows the conv result by inserting unit (size-1) dims, `wrappedGeneric`
// is the wrapped op (e.g. a truncf) that runs at the higher rank, and `collapse`
// removes those same dims to return to the original rank.
struct WrapperChain {
    tensor::ExpandShapeOp expand;
    linalg::GenericOp wrappedGeneric;
    tensor::CollapseShapeOp collapse;
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

// True if `expand` only inserts size-1 dimensions (no real dimension split).
// computeRankReductionMask returns the dropped indices iff the src shape is the
// result shape with only size-1 entries erased, and nullopt otherwise. A real
// split (e.g. 6 -> 2x3) or a multiplier dim (M>1) leaves a non-1 difference, so
// it returns nullopt and we reject -- we only ever drop dims that hold no data.
bool expandOnlyInsertsUnitDims(tensor::ExpandShapeOp expand) {
    return computeRankReductionMask(
               expand.getResultType().getShape(), expand.getSrcType().getShape()
    )
        .has_value();
}

// Decide whether `generic` is the wrapper described at the top of this file. On
// a match, return the three ops that form the wrapper; otherwise return nullopt.
// The checks are ordered cheapest-first so unrelated ops are rejected quickly.
// The match is rank-agnostic: it does not hard-code rank 4/5, only that the
// expand/collapse pair adds and removes unit dims around an identity-map generic.
std::optional<WrapperChain> matchWrapperChain(linalg::GenericOp generic) {
    // One ranked result, and its maps must be identity so dropping the inserted
    // size-1 dimensions is safe (a broadcast/permutation map could change it).
    if (generic.getNumResults() != 1)
        return std::nullopt;
    if (!isa<RankedTensorType>(generic.getResult(0).getType()))
        return std::nullopt;
    if (!hasIdentityIndexingMaps(generic))
        return std::nullopt;

    // It reads exactly one input, and that input is an expand_shape.
    if (generic.getNumDpsInputs() != 1)
        return std::nullopt;
    auto expand = generic.getDpsInputs()[0].getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand)
        return std::nullopt;

    // It has exactly one user, and that user is a collapse_shape.
    if (!generic.getResult(0).hasOneUse())
        return std::nullopt;
    auto collapse = dyn_cast<tensor::CollapseShapeOp>(*generic.getResult(0).getUsers().begin());
    if (!collapse)
        return std::nullopt;

    // The expand and collapse must be exact inverses: same grouping, and the
    // collapse returns to the expand's original rank. Then the only thing the
    // pair does is add dimensions and remove them again.
    if (expand.getReassociationIndices() != collapse.getReassociationIndices())
        return std::nullopt;
    if (cast<RankedTensorType>(collapse.getResult().getType()).getRank() !=
        cast<RankedTensorType>(expand.getSrc().getType()).getRank())
        return std::nullopt;

    // Every dimension the expand adds must be size 1, so dropping them keeps
    // every value. A real multiplier dim (M>1) is non-1 and is rejected here.
    if (!expandOnlyInsertsUnitDims(expand))
        return std::nullopt;

    return WrapperChain{expand, generic, collapse};
}

// Build the folded replacement generic. It reads the lower-rank value that
// feeds the expand_shape, writes a fresh tensor the size of the collapse result
// (same rank), and reuses the body of the original generic unchanged.
linalg::GenericOp buildFoldedGeneric(WrapperChain chain, OpBuilder &builder) {
    Location loc = chain.wrappedGeneric.getLoc();
    MLIRContext *ctx = builder.getContext();

    auto resultType = cast<RankedTensorType>(chain.collapse.getResult().getType());
    int64_t rank = resultType.getRank();
    Value init =
        tensor::EmptyOp::create(builder, loc, resultType.getShape(), resultType.getElementType())
            .getResult();

    // One identity map for the input and one for the output; all dims parallel.
    SmallVector<AffineMap, 2> maps(2, AffineMap::getMultiDimIdentityMap(rank, ctx));
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

    auto folded = linalg::GenericOp::create(
        builder, loc, /*resultTensorTypes=*/TypeRange{resultType},
        /*inputs=*/ValueRange{chain.expand.getSrc()}, /*outputs=*/ValueRange{init}, maps, iterators
    );

    // Copy the original op's body (e.g. the truncf) into the new op.
    IRMapping mapping;
    chain.wrappedGeneric.getRegion().cloneInto(&folded.getRegion(), mapping);
    return folded;
}

// Rewrite one matched wrapper into a single lower-rank generic. Each match is
// handled on its own; the narrow match in matchWrapperChain is the only gate,
// so no group-level bookkeeping is needed.
struct FoldUnitExtentWrapperPattern : OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp generic, PatternRewriter &rewriter) const override {
        std::optional<WrapperChain> chain = matchWrapperChain(generic);
        if (!chain)
            return failure();

        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(chain->wrappedGeneric);
        linalg::GenericOp folded = buildFoldedGeneric(*chain, rewriter);

        // Point the collapse's users at the new lower-rank op, then delete the
        // wrapper ops. The expand may feed other ops too, so erase it only when
        // nothing reads it anymore.
        rewriter.replaceOp(chain->collapse, folded.getResult(0));
        rewriter.eraseOp(chain->wrappedGeneric);
        if (chain->expand.getResult().use_empty())
            rewriter.eraseOp(chain->expand);

        LLVM_DEBUG(
            llvm::dbgs() << "[fold-unit-extent-wrapper] folded wrapper to lower-rank generic\n"
        );
        return success();
    }
};

struct FoldUnitExtentWrapperPass : impl::FoldUnitExtentWrapperBase<FoldUnitExtentWrapperPass> {
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<FoldUnitExtentWrapperPattern>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldUnitExtentWrapperPass() {
    return std::make_unique<FoldUnitExtentWrapperPass>();
}

} // namespace mlir::syna::torq
