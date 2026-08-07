// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-outline-compile-time-const"

namespace mlir::syna::torq {

namespace {

// Recursively collect all ops in the def-chain of `op` that live inside
// `clonedOutermost` (the cloned outermost loop op).  Ops whose defining op is
// outside `clonedOutermost` are treated as external constants and skipped.
// Returns failure if any operand is defined by a runtime-input op (dispatch
// tensor load / HAL binding).
LogicalResult
collectDefChain(Operation *op, Operation *clonedOutermost, SetVector<Operation *> &visited) {
    if (!visited.insert(op))
        return success();

    for (Value operand : op->getOperands()) {
        if (isa<BlockArgument>(operand))
            continue; // block args (IVs / iter-args) are fine
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
            continue;
        // External to the cloned nest — treat as constant, skip.
        if (!clonedOutermost->isAncestor(defOp))
            continue;
        // the value depends on the input, we cannot compute this
        if (isa<iree_compiler::IREE::TensorExt::DispatchTensorLoadOp,
                iree_compiler::IREE::HAL::InterfaceBindingSubspanOp>(defOp)) {
            LLVM_DEBUG({ llvm::dbgs() << "Value depends on inputs, cannot compute statically\n"; });
            return failure();
        }
        if (failed(collectDefChain(defOp, clonedOutermost, visited)))
            return failure();
    }
    return success();
}

// Build an affine map: iv -> floor_div(iv - lb, step).
AffineMap buildIvMap(int64_t lb, int64_t step, MLIRContext *ctx) {
    AffineExpr d0 = getAffineDimExpr(0, ctx);
    return AffineMap::get(
        1, 0, (d0 - getAffineConstantExpr(lb, ctx)).floorDiv(getAffineConstantExpr(step, ctx)), ctx
    );
}

// Returns the trip count for each IV dimension of `loop`, or nullopt if any is
// non-constant.
std::optional<SmallVector<int64_t>> getTripCounts(LoopLikeOpInterface loop) {
    auto maybeLbs = loop.getLoopLowerBounds();
    auto maybeUbs = loop.getLoopUpperBounds();
    auto maybeSteps = loop.getLoopSteps();
    if (!maybeLbs || !maybeUbs || !maybeSteps)
        return std::nullopt;
    SmallVector<int64_t> counts;
    for (auto [lb, ub, step] : llvm::zip_equal(*maybeLbs, *maybeUbs, *maybeSteps)) {
        auto tc = constantTripCount(lb, ub, step, true, scf::computeUbMinusLb);
        if (!tc)
            return std::nullopt;
        counts.push_back(tc->getSExtValue());
    }
    return counts;
}

// Returns IV affine maps for `loop`, or nullopt if any bound/step is
// non-constant.
std::optional<SmallVector<AffineMap>> getIvMaps(LoopLikeOpInterface loop, MLIRContext *ctx) {
    auto maybeLbs = loop.getLoopLowerBounds();
    auto maybeSteps = loop.getLoopSteps();
    if (!maybeLbs || !maybeSteps)
        return std::nullopt;
    SmallVector<AffineMap> maps;
    for (auto [lb, step] : llvm::zip_equal(*maybeLbs, *maybeSteps)) {
        auto maybeLb = getConstantIntValue(lb);
        auto maybeStep = getConstantIntValue(step);
        if (!maybeLb || !maybeStep)
            return std::nullopt;
        maps.push_back(buildIvMap(*maybeLb, *maybeStep, ctx));
    }
    return maps;
}

// `op`'s operands plus any values its regions capture from above. collectDefChain does not walk
// region bodies, so the invariance and hoistability analyses must collect captures themselves.
void collectExternallyReadValues(Operation *op, SmallVectorImpl<Value> &values) {
    llvm::append_range(values, op->getOperands());
    for (Region &region : op->getRegions()) {
        SetVector<Value> capturedAbove;
        getUsedValuesDefinedAbove(region, capturedAbove);
        llvm::append_range(values, capturedAbove);
    }
}

// Drop the levels of `levels` whose IVs `defChain` never reads: each would only add an axis of
// identical copies to the outlined constant (e.g. a matmul tiled over M and N re-materializes the
// whole weight per M tile). Chain ops in a dropped level's body get cloned once ahead of the
// shell nest instead. Bounds cannot re-introduce a dropped IV (collectLoopNest requires constant
// bounds), and every level may drop -- then the chain hoists whole and needs no nest. Granularity
// is per level: a multi-IV forall is kept whole if any of its IVs is read.
void dropLoopLevelsChainIsInvariantTo(
    const SetVector<Operation *> &defChain, SmallVectorImpl<LoopLikeOpInterface> &levels
) {
    DenseSet<Value> chainReads;
    SmallVector<Value> reads;
    for (Operation *chainOp : defChain) {
        reads.clear();
        collectExternallyReadValues(chainOp, reads);
        chainReads.insert(reads.begin(), reads.end());
    }

    llvm::erase_if(levels, [&](LoopLikeOpInterface level) {
        SmallVector<Value> ivs = *level.getLoopInductionVars();
        return llvm::none_of(ivs, [&](Value iv) { return chainReads.contains(iv); });
    });
}

// Whether the chain ops in dropped levels can be rebuilt ahead of the nest. Every value they read
// (operands and region captures) must dominate the insertion point before the outermost loop:
// defined outside the nest, a loop-carried destination (remapped there to a tensor.empty), or
// produced by an op hoisted earlier. A value from a *retained* level does not qualify -- it only
// exists once its shell loop is built -- so the caller then keeps every level.
bool canHoistChainOutOfDroppedLevels(
    const SetVector<Operation *> &orderedChain, ArrayRef<LoopLikeOpInterface> retainedLevels,
    Operation *outermostLoop
) {
    DenseSet<Block *> retainedBodies;
    for (auto level : retainedLevels)
        retainedBodies.insert(&level.getLoopRegions()[0]->front());

    DenseSet<Operation *> hoisted;
    SmallVector<Value> reads;
    for (Operation *chainOp : orderedChain) {
        if (retainedBodies.contains(chainOp->getBlock()))
            continue;
        reads.clear();
        collectExternallyReadValues(chainOp, reads);
        for (Value operand : reads) {
            if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                // Defined outside the nest (e.g. a func arg): dominates.
                Operation *argParent = blockArg.getOwner()->getParentOp();
                if (argParent && !outermostLoop->isAncestor(argParent))
                    continue;
                // IVs have no counterpart outside the nest; loop-carried destinations do.
                auto owner = dyn_cast_or_null<LoopLikeOpInterface>(argParent);
                if (!owner || blockArg.getArgNumber() < owner.getLoopInductionVars()->size())
                    return false;
                continue;
            }
            Operation *def = operand.getDefiningOp();
            if (!def || !outermostLoop->isAncestor(def))
                continue; // defined outside the nest, so it already dominates
            if (!hoisted.contains(def))
                return false;
        }
        hoisted.insert(chainOp);
    }
    return true;
}

// Collect the enclosing scf.for / scf.forall chain for `op`, ordered from
// innermost to outermost.  Returns failure if any loop has a non-constant trip
// count for any dimension.
LogicalResult
collectLoopNest(Operation *op, MLIRContext *ctx, SmallVectorImpl<LoopLikeOpInterface> &levels) {
    Operation *cursor = op->getParentOp();
    while (cursor) {
        if (auto loopOp = dyn_cast<LoopLikeOpInterface>(cursor)) {
            if (!getTripCounts(loopOp) || !getIvMaps(loopOp, ctx))
                return failure();
            levels.push_back(loopOp);
        }
        else if (isa<func::FuncOp>(cursor)) {
            break;
        }
        cursor = cursor->getParentOp();
    }
    return success();
}

// Append offset=0, size=s, stride=1 for each dimension in `shape`.
void appendTensorShapeDims(
    OpBuilder &b, ArrayRef<int64_t> shape, SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<OpFoldResult> &sizes, SmallVectorImpl<OpFoldResult> &strides
) {
    offsets.append(shape.size(), b.getIndexAttr(0));
    llvm::append_range(sizes, getAsIndexOpFoldResult(b.getContext(), shape));
    strides.append(shape.size(), b.getIndexAttr(1));
}

// Append IV-based offsets/sizes/strides (all size=1, stride=1) for every clone
// level (outermost first) to the provided vectors.
void appendLoopDimSliceParams(
    OpBuilder &b, Location loc, ArrayRef<LoopLikeOpInterface> cloneLoops,
    SmallVectorImpl<OpFoldResult> &offsets, SmallVectorImpl<OpFoldResult> &sizes,
    SmallVectorImpl<OpFoldResult> &strides
) {
    for (auto loop : cloneLoops) {
        SmallVector<Value> ivs = *loop.getLoopInductionVars();
        SmallVector<AffineMap> ivMaps = *getIvMaps(loop, b.getContext());
        llvm::append_range(
            offsets,
            llvm::map_range(
                llvm::zip_equal(ivMaps, ivs),
                [&](auto &&pair) -> OpFoldResult {
                    auto [map, iv] = pair;
                    return affine::AffineApplyOp::create(b, loc, map, ValueRange{iv}).getResult();
                }
            )
        );
        sizes.append(ivMaps.size(), b.getIndexAttr(1));
        strides.append(ivMaps.size(), b.getIndexAttr(1));
    }
}

// Emit ExpandShapeOp + InsertSliceOp (or ParallelInsertSliceOp for forall) at
// the innermost clone loop using all loop IVs as multi-dim offsets.
void emitInsertSliceAtInnermost(
    PatternRewriter &rewriter, Location loc, ArrayRef<LoopLikeOpInterface> cloneLoops, Value lastV,
    ArrayRef<int64_t> expandShape, ArrayRef<int64_t> origShape, Type elemTy,
    const SmallVector<ReassociationIndices> &reassoc
) {
    LoopLikeOpInterface innermostLoop = cloneLoops.back();
    OpBuilder::InsertionGuard gInsert(rewriter);
    if (auto forallOp = dyn_cast<scf::ForallOp>(innermostLoop.getOperation()))
        rewriter.setInsertionPoint(forallOp.getTerminator());
    else
        rewriter.setInsertionPointToEnd(&innermostLoop.getLoopRegions()[0]->front());

    SmallVector<OpFoldResult> offsets, sizes, strides;
    appendLoopDimSliceParams(rewriter, loc, cloneLoops, offsets, sizes, strides);
    appendTensorShapeDims(rewriter, origShape, offsets, sizes, strides);

    auto expandV = tensor::ExpandShapeOp::create(
                       rewriter, loc, RankedTensorType::get(expandShape, elemTy), lastV, reassoc
    )
                       .getResult();

    if (auto forallOp = dyn_cast<scf::ForallOp>(innermostLoop.getOperation())) {
        OpBuilder::InsertionGuard gInP(rewriter);
        rewriter.setInsertionPointToStart(forallOp.getTerminator().getBody());
        tensor::ParallelInsertSliceOp::create(
            rewriter, loc, expandV, innermostLoop.getRegionIterArgs()[0], offsets, sizes, strides
        );
    }
    else {
        auto iOp = tensor::InsertSliceOp::create(
            rewriter, loc, expandV, innermostLoop.getRegionIterArgs()[0], offsets, sizes, strides
        );
        scf::YieldOp::create(rewriter, loc, iOp.getResult());
    }
}

// For each (outer, inner) clone loop pair, propagate the inner result upward
// via scf.yield (scf.for outer) or tensor.parallel_insert_slice (scf.forall outer).
void propagateResultsUpward(
    PatternRewriter &rewriter, Location loc, ArrayRef<LoopLikeOpInterface> cloneLoops
) {
    for (int li = (int)cloneLoops.size() - 2; li >= 0; --li) {
        LoopLikeOpInterface outer = cloneLoops[li];
        LoopLikeOpInterface inner = cloneLoops[li + 1];
        Value innerResult = inner.getOperation()->getResult(0);

        if (auto outerForall = dyn_cast<scf::ForallOp>(outer.getOperation())) {
            SmallVector<OpFoldResult> pInsOffsets, pInsSizes, pInsStrides;
            SmallVector<Value> outerIvs = *outer.getLoopInductionVars();
            SmallVector<AffineMap> outerIvMaps = *getIvMaps(outer, rewriter.getContext());
            llvm::append_range(
                pInsOffsets,
                llvm::map_range(
                    llvm::zip_equal(outerIvMaps, outerIvs),
                    [&](auto &&pair) -> OpFoldResult {
                        auto [map, iv] = pair;
                        return affine::AffineApplyOp::create(rewriter, loc, map, ValueRange{iv})
                            .getResult();
                    }
                )
            );
            pInsSizes.append(outerIvs.size(), rewriter.getIndexAttr(1));
            pInsStrides.append(outerIvs.size(), rewriter.getIndexAttr(1));
            appendTensorShapeDims(
                rewriter, cast<RankedTensorType>(innerResult.getType()).getShape(), pInsOffsets,
                pInsSizes, pInsStrides
            );
            OpBuilder::InsertionGuard gInP(rewriter);
            rewriter.setInsertionPointToStart(outerForall.getTerminator().getBody());
            tensor::ParallelInsertSliceOp::create(
                rewriter, loc, innerResult, outerForall.getRegionIterArgs()[0], pInsOffsets,
                pInsSizes, pInsStrides
            );
        }
        else {
            OpBuilder::InsertionGuard gOuter(rewriter);
            rewriter.setInsertionPointAfter(inner.getOperation());
            scf::YieldOp::create(rewriter, loc, innerResult);
        }
    }
}

// Emit tensor.extract_slice + tensor.collapse_shape at the current insertion
// point (inside the original loop nest) and replace `op` with the result.
void emitExtractAndReplace(
    PatternRewriter &rewriter, Operation *op, Value constV, RankedTensorType opTy,
    ArrayRef<LoopLikeOpInterface> loopLevels, MLIRContext *ctx,
    const SmallVector<ReassociationIndices> &reassoc
) {
    auto shape = opTy.getShape();
    SmallVector<OpFoldResult> extractOffsets, extractSizes, extractStrides;

    // loopLevels is innermost-first; iterate reversed for outermost-first dim ordering.
    for (auto level : llvm::reverse(loopLevels)) {
        SmallVector<Value> ivs = *level.getLoopInductionVars();
        SmallVector<AffineMap> ivMaps = *getIvMaps(level, ctx);
        llvm::append_range(
            extractOffsets, llvm::map_range(
                                llvm::zip_equal(ivMaps, ivs),
                                [&](auto &&pair) -> OpFoldResult {
                                    auto [map, iv] = pair;
                                    return affine::AffineApplyOp::create(
                                               rewriter, op->getLoc(), map, ValueRange{iv}
                                    )
                                        .getResult();
                                }
                            )
        );
        extractSizes.append(ivMaps.size(), rewriter.getIndexAttr(1));
        extractStrides.append(ivMaps.size(), rewriter.getIndexAttr(1));
    }
    appendTensorShapeDims(rewriter, shape, extractOffsets, extractSizes, extractStrides);

    auto extractOp = tensor::ExtractSliceOp::create(
        rewriter, op->getLoc(), constV, extractOffsets, extractSizes, extractStrides
    );
    auto collapseV = tensor::CollapseShapeOp::create(
                         rewriter, op->getLoc(), opTy, extractOp.getResult(), reassoc
    )
                         .getResult();
    LLVM_DEBUG(llvm::dbgs() << "Replacing op with computed const value\n";
               llvm::dbgs() << "Original op:\n";
               op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
               llvm::dbgs() << "\n"; llvm::dbgs() << "Computed const value:\n";
               collapseV.print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
               llvm::dbgs() << "\n";);
    rewriter.replaceOp(op, collapseV);
}

class ConvertOpInsideForOpRewriter : public RewritePattern {
  public:
    ConvertOpInsideForOpRewriter(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit*/ 12, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Outline compile-time-const ops from inside a nest of scf.for/scf.forall
        // into a precomputed tensor result.  The approach:
        //   1. Clone the full outermost loop.
        //   2. Locate the cloned op of interest inside the clone.
        //   3. Collect its def-chain within the clone; validate no runtime inputs.
        //   4. Prune the cloned nest to retain only the def-chain ops.
        //   5. Build new empty-shell loop nest (with initTensor iter-arg) and move
        //      the pruned ops into it.
        //   6. Wire insert-slice at innermost, propagate upward, extract in original.
        if (!isCompileTimeConstAttr(op)) {
            LLVM_DEBUG(llvm::dbgs() << "Op is not marked as compile-time-const: "; op->dump(););
            return failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "Op to compute inside ForOp: "; op->dump(););

        // Collect the full loop nest (innermost first).
        SmallVector<LoopLikeOpInterface> loopLevels;
        if (failed(collectLoopNest(op, rewriter.getContext(), loopLevels)) || loopLevels.empty()) {
            LLVM_DEBUG(llvm::dbgs() << "No suitable parent loop nest found for op: "; op->dump(););
            return failure();
        }

        Operation *outermostLoop = loopLevels.back();

        // 1. Collect the def-chain of `op` within the outermost loop.
        SetVector<Operation *> origDefChain;
        if (failed(collectDefChain(op, outermostLoop, origDefChain))) {
            LLVM_DEBUG(llvm::dbgs() << "Def-chain depends on runtime inputs for op: "; op->dump(););
            return failure();
        }

        // Keep only the levels the chain varies over; back out if what that leaves in dropped
        // bodies cannot be hoisted ahead of the nest.
        SetVector<Operation *> orderedChain = topologicalSort(origDefChain);
        SmallVector<LoopLikeOpInterface> allLoopLevels = loopLevels;
        dropLoopLevelsChainIsInvariantTo(origDefChain, loopLevels);
        if (loopLevels.size() != allLoopLevels.size() &&
            !canHoistChainOutOfDroppedLevels(orderedChain, loopLevels, outermostLoop)) {
            LLVM_DEBUG(llvm::dbgs() << "Cannot hoist dropped-level chain, keeping every level\n");
            loopLevels = allLoopLevels;
        }

        auto opTy = cast<RankedTensorType>(op->getResult(0).getType());
        auto shape = opTy.getShape();

        // Build the full precomputed tensor shape:
        //   [outermost_dim0, ..., innermost_dimN, *original_shape]
        SmallVector<int64_t> fullShape;
        for (auto level : llvm::reverse(loopLevels))
            llvm::append_range(fullShape, *getTripCounts(level));
        size_t numLoopDims = fullShape.size();
        llvm::append_range(fullShape, shape);

        // expandShape: fullShape with all leading loop dims set to 1.
        SmallVector<int64_t> expandShape(fullShape);
        for (size_t i = 0; i < numLoopDims; ++i)
            expandShape[i] = 1;
        auto reassoc = getReassociationIndicesForCollapse(expandShape, shape);

        // -------------------------------------------------------------------
        // 2. Build new empty-shell loop nest (with initTensor threaded through),
        //    clone the original def-chain ops into the new shells, wire insert.
        // -------------------------------------------------------------------
        // Remove compile-time-const attrs before building to avoid re-matching.
        Value constV;
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(outermostLoop);
            Location loc = outermostLoop->getLoc();

            // An original loop's block args split into induction variables (which
            // lead the list) and loop-carried destinations (scf.for iter_args /
            // scf.forall shared_outs). The const-accumulator shell loop threads a
            // single, differently-typed iter-arg, so the destinations have no
            // positional counterpart. Def-chain ops only read such destinations to
            // size a write (their values are fully overwritten downstream), so
            // substitute a fresh tensor.empty of the same type. Materialize it here,
            // outside the shell nest, so it dominates every clone. Mapping these onto
            // the accumulator iter-arg positionally instead would splice a
            // wrong-typed source into cloned slice/insert ops and emit invalid IR.
            IRMapping moveMapping;
            for (auto level : allLoopLevels) {
                Block *origBody = &level.getLoopRegions()[0]->front();
                unsigned numIvs = level.getLoopInductionVars()->size();
                for (unsigned i = numIvs, e = origBody->getNumArguments(); i < e; ++i) {
                    BlockArgument destArg = origBody->getArgument(i);
                    if (destArg.use_empty())
                        continue;
                    auto destTy = dyn_cast<RankedTensorType>(destArg.getType());
                    if (!destTy || !destTy.hasStaticShape())
                        return failure();
                    moveMapping.map(
                        destArg, tensor::EmptyOp::create(
                                     rewriter, loc, destTy.getShape(), destTy.getElementType()
                                 )
                                     .getResult()
                    );
                }
            }

            // Chain ops from dropped levels are invariant to them: clone once ahead of the shell
            // nest. canHoistChainOutOfDroppedLevels already proved their operands dominate here.
            DenseSet<Block *> retainedBodies;
            for (auto level : loopLevels)
                retainedBodies.insert(&level.getLoopRegions()[0]->front());
            for (auto *chainOp : orderedChain) {
                if (retainedBodies.contains(chainOp->getBlock()))
                    continue;
                removeCompileTimeConstAttr(rewriter.clone(*chainOp, moveMapping));
            }

            // Chain invariant to every level: hoisted whole, no nest to build. No new const
            // boundary needed -- the surviving CompileInputToConstOp consumer still lets
            // CompileTimeConstComputePass fold the hoisted chain.
            if (loopLevels.empty()) {
                Value hoisted = moveMapping.lookup(op->getResult(0));
                rewriter.replaceOp(op, hoisted);
                return success();
            }

            Value initTensor =
                tensor::EmptyOp::create(rewriter, loc, fullShape, opTy.getElementType())
                    .getResult();

            // Build the shell loops (outermost-first) using the original loop bounds.
            SmallVector<LoopLikeOpInterface> newCloneLoops;
            {
                Value currentIterArg = initTensor;
                // loopLevels is innermost-first; reverse to build outermost-first.
                for (auto level : llvm::reverse(loopLevels)) {
                    LoopLikeOpInterface newLoop;
                    if (isa<scf::ForallOp>(level)) {
                        newLoop = scf::ForallOp::create(
                            rewriter, loc, *level.getLoopLowerBounds(), *level.getLoopUpperBounds(),
                            *level.getLoopSteps(), llvm::ArrayRef<Value>{currentIterArg},
                            std::nullopt
                        );
                    }
                    else {
                        auto forOp = cast<scf::ForOp>(level);
                        newLoop = scf::ForOp::create(
                            rewriter, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                            forOp.getStep(), llvm::ArrayRef<Value>{currentIterArg}
                        );
                    }
                    currentIterArg = newLoop.getRegionIterArgs()[0];
                    rewriter.setInsertionPointToStart(&newLoop.getLoopRegions()[0]->front());
                    newCloneLoops.push_back(newLoop);
                }
            }

            // Map induction variables positionally: they lead the block-argument
            // list of both scf.for and scf.forall, so index i matches. Loop-carried
            // destinations were already remapped to fresh tensor.empty values above.
            // loopLevels[0] = innermost original → newCloneLoops.back() = innermost shell.
            for (auto [level, cloneInfo] :
                 llvm::zip_equal(loopLevels, llvm::reverse(newCloneLoops))) {
                Block *origBody = &level.getLoopRegions()[0]->front();
                Block *newBody = &cloneInfo.getLoopRegions()[0]->front();

                unsigned numIvs = level.getLoopInductionVars()->size();
                for (unsigned i = 0; i < numIvs; ++i)
                    moveMapping.map(origBody->getArgument(i), newBody->getArgument(i));
            }

            // Clone original chain ops level-by-level into the new shell loops.
            for (auto [level, newCI] : llvm::zip_equal(llvm::reverse(loopLevels), newCloneLoops)) {
                Block *origBody = &level.getLoopRegions()[0]->front();
                Block &newBody = newCI.getLoopRegions()[0]->front();

                OpBuilder::InsertionGuard gLevel(rewriter);
                rewriter.setInsertionPointToStart(&newBody);

                for (auto *chainOp : orderedChain) {
                    if (chainOp->getBlock() != origBody)
                        continue;
                    auto cloneOp = rewriter.clone(*chainOp, moveMapping);
                    removeCompileTimeConstAttr(cloneOp
                    ); // Cloning keeps the original attr, remove it to avoid re-matching.
                }
            }
            // Cannot miss: if any level is retained, `op` sits in a retained body -- a chain path
            // from the retained IV read to an op in a dropped body would have failed the hoist
            // guard and reverted to the full nest.
            Value lastV = moveMapping.lookup(op->getResult(0));

            emitInsertSliceAtInnermost(
                rewriter, loc, newCloneLoops, lastV, expandShape, shape, opTy.getElementType(),
                *reassoc
            );

            propagateResultsUpward(rewriter, loc, newCloneLoops);

            LoopLikeOpInterface outermostClone = newCloneLoops[0];
            constV = createCompileTimeConstOp(outermostClone.getOperation(), rewriter)
                         .value_or(outermostClone.getOperation()->getResult(0));
            assert(
                succeeded(verify(constV.getDefiningOp())) && "Expected defining op for const result"
            );
        }

        emitExtractAndReplace(
            rewriter, op, constV, opTy, loopLevels, rewriter.getContext(), *reassoc
        );
        return success();
    }
};

} // namespace

// The pass looks for operations marked as compile-time-const and
// replaces them with constant operations computed at compile time.
class CompileTimeConstOutlinePass
    : public impl::CompileTimeConstOutlineBase<CompileTimeConstOutlinePass> {
  public:
    using CompileTimeConstOutlineBase::CompileTimeConstOutlineBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        LLVM_DEBUG(llvm::dbgs() << "Running CompileTimeConstOutlinePass on function: "
                                << funcOp.getName() << "\n";);
        auto valuesToProcess = collectAllCompileTimeConstOps(funcOp);
        SmallVector<Operation *> opsToProcess;
        for (auto val : valuesToProcess) {
            if (auto defOp = val.getDefiningOp()) {
                opsToProcess.push_back(defOp);
                setCompileTimeConstAttr(defOp);
                LLVM_DEBUG({
                    llvm::dbgs() << "Collecting op to compute: ";
                    defOp->dump();
                });
            }
        }

        RewritePatternSet patterns(&getContext());
        patterns.add<ConvertOpInsideForOpRewriter>(&getContext());

        if (failed(applyOpPatternsGreedily(opsToProcess, std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileTimeConstOutlinePass() {
    return std::make_unique<CompileTimeConstOutlinePass>();
}
} // namespace mlir::syna::torq
