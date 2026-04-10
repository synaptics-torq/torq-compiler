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

        if (!isCompileTimeConst(op)) {
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

        // 1. Collect the def-chain of `op` within the outermost loop.
        SetVector<Operation *> origDefChain;
        if (failed(collectDefChain(op, outermostLoop, origDefChain))) {
            LLVM_DEBUG(llvm::dbgs() << "Def-chain depends on runtime inputs for op: "; op->dump(););
            return failure();
        }

        // -------------------------------------------------------------------
        // 2. Build new empty-shell loop nest (with initTensor threaded through),
        //    clone the original def-chain ops into the new shells, wire insert.
        // -------------------------------------------------------------------
        // Remove compile-time-const attrs before building to avoid re-matching.
        for (auto *origOp : origDefChain)
            removeCompileTimeConstAttr(origOp);

        Value constV;
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(outermostLoop);
            Location loc = outermostLoop->getLoc();

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

            // Build IRMapping: original loop block args → new shell loop block args.
            // loopLevels[0] = innermost original → newCloneLoops.back() = innermost shell.
            IRMapping moveMapping;
            for (auto [level, cloneInfo] :
                 llvm::zip_equal(loopLevels, llvm::reverse(newCloneLoops))) {
                Block *origBody = &level.getLoopRegions()[0]->front();
                Block *newBody = &cloneInfo.getLoopRegions()[0]->front();
                moveMapping.map(origBody->getArguments(), newBody->getArguments());
            }

            // Collect origDefChain ops in topological order across all loop levels.
            SetVector<Operation *> orderedChain = topologicalSort(origDefChain);

            // Clone original chain ops level-by-level into the new shell loops.
            for (auto [level, newCI] : llvm::zip_equal(llvm::reverse(loopLevels), newCloneLoops)) {
                Block *origBody = &level.getLoopRegions()[0]->front();

                OpBuilder::InsertionGuard gLevel(rewriter);
                if (auto forallOp = dyn_cast<scf::ForallOp>(newCI.getOperation()))
                    rewriter.setInsertionPoint(forallOp.getTerminator());
                else
                    rewriter.setInsertionPointToEnd(&newCI.getLoopRegions()[0]->front());

                for (auto *chainOp : orderedChain) {
                    if (chainOp->getBlock() != origBody)
                        continue;
                    rewriter.clone(*chainOp, moveMapping);
                }
            }
            Value lastV = moveMapping.lookup(op->getResult(0));

            emitInsertSliceAtInnermost(
                rewriter, loc, newCloneLoops, lastV, expandShape, shape, opTy.getElementType(),
                *reassoc
            );

            propagateResultsUpward(rewriter, loc, newCloneLoops);

            LoopLikeOpInterface outermostClone = newCloneLoops[0];
            setCompileTimeConstAttr(outermostClone.getOperation());
            constV = outermostClone.getOperation()->getResult(0);
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
        SmallVector<Operation *> opsToProcess;
        auto funcOp = getOperation();
        funcOp->walk([&](Operation *op) {
            if (!isCompileTimeConst(op)) {
                return WalkResult::advance();
            }
            opsToProcess.push_back(op);
            return WalkResult::advance();
        });
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
