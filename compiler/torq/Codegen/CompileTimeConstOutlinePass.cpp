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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-outline-compile-time-const"

namespace mlir::syna::torq {

namespace {

LogicalResult walkDefChainContained(
    Operation *op, Operation *parent, SmallVectorImpl<Operation *> &opsToMoveStack
) {
    // Walk and collect the operand-def chains and verify every dependency stays within the loop.
    // If the depended op is outside the loop, consider it as a constant for now and do not move it
    // Even if after the outline if the op is not constant it shouldn't affect the overall flow
    if (llvm::is_contained(opsToMoveStack, op)) {
        return success();
    }
    opsToMoveStack.push_back(op);

    for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        LLVM_DEBUG(
            llvm::dbgs() << "Visiting operand: " << operand << " defined by op: "; if (defOp) {
                defOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            } else { llvm::dbgs() << "BlockArgument"; } llvm::dbgs() << "\n";
        );
        if (auto bArg = dyn_cast<BlockArgument>(operand)) {
            if (bArg.getOwner() != op->getBlock()) {
                LLVM_DEBUG(llvm::dbgs()
                               << "Reached nested block argument, stopping walk at operand: "
                               << operand << "\n";);
                return failure();
            }
            continue;
        }
        if (!parent->isProperAncestor(defOp)) {
            LLVM_DEBUG(llvm::dbgs() << "Reached ForOp ancestor, stopping walk at op: ";
                       defOp->dump(););
            continue;
        }
        // the value depends on the input, we cannot compute this
        if (isa<iree_compiler::IREE::TensorExt::DispatchTensorLoadOp,
                iree_compiler::IREE::HAL::InterfaceBindingSubspanOp>(defOp)) {
            LLVM_DEBUG({ llvm::dbgs() << "Value depends on inputs, cannot compute statically\n"; });

            return failure();
        }

        if (failed(walkDefChainContained(defOp, parent, opsToMoveStack))) {
            return failure();
        }
    }
    return success();
}

class ConvertOpInsideForOpRewriter : public RewritePattern {
  public:
    ConvertOpInsideForOpRewriter(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit*/ 12, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        // Outline compile-time-const ops from inside scf.for into a precomputed loop result.
        // If op tagged with compile-time-constant inside a scf.for loop,
        // create a new loop that computes only the value and add additional insert/extract slice to
        // access the computed data

        if (!isCompileTimeConst(op)) {
            return failure();
        }
        LLVM_DEBUG(llvm::dbgs() << "Op to compute inside ForOp: "; op->dump(););
        auto forOp = op->getParentOfType<scf::ForOp>();
        if (!forOp) {
            LLVM_DEBUG(llvm::dbgs() << "No parent ForOp found for op: "; op->dump(););
            return failure();
        }
        SmallVector<Operation *, 4> opsToMoveStack;
        if (failed(walkDefChainContained(op, forOp.getOperation(), opsToMoveStack))) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to walk def chain for op: "; op->dump(););
            return failure();
        }

        std::sort(opsToMoveStack.begin(), opsToMoveStack.end(), [](Operation *a, Operation *b) {
            return b->isBeforeInBlock(a);
        });
        LLVM_DEBUG(llvm::dbgs() << "Ops to move for const computation:\n";
                   for (auto [i, o]
                        : llvm::enumerate(opsToMoveStack)) {
                       llvm::dbgs() << "  [" << i << "]: ";
                       o->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
                       llvm::dbgs() << "\n";
                   });

        FailureOr<Value> maybeBias;

        auto lb = forOp.getLowerBound();
        auto ub = forOp.getUpperBound();
        auto step = forOp.getStep();
        auto maybeUpperBound =
            constantTripCount(lb, ub, step, /*isSigned=*/true, scf::computeUbMinusLb);
        if (!maybeUpperBound) {
            LLVM_DEBUG(llvm::dbgs() << "Failed to compute upper bound for ForOp: "; forOp->dump(););
            return failure();
        }
        int64_t upperBound = maybeUpperBound->getSExtValue();

        auto opTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        auto shape = opTy.getShape();
        SmallVector<int64_t, 4> newShape(shape.begin(), shape.end());
        newShape.insert(newShape.begin(), upperBound);

        auto expandShape = newShape;
        expandShape[0] = 1;
        auto reassoc = getReassociationIndicesForCollapse(expandShape, shape);

        auto stepAttr = returnAttr(step);
        auto lbAttr = returnAttr(lb);
        auto stepInt = dyn_cast<IntegerAttr>(stepAttr).getInt();
        auto lbInt = dyn_cast<IntegerAttr>(lbAttr).getInt();
        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr cStep = rewriter.getAffineConstantExpr(stepInt);
        AffineExpr cLowerBound = rewriter.getAffineConstantExpr(lbInt);
        AffineExpr divExpr = (d0 - cLowerBound).floorDiv(cStep);

        AffineMap newIvMap = AffineMap::get(1, 0, divExpr, rewriter.getContext());

        for (auto op : opsToMoveStack) {
            // After new loop outlined for const computation,
            // remove the compile-time-const attribute to avoid re-matching.
            removeCompileTimeConstAttr(op);
        }

        Value constV;
        {
            OpBuilder::InsertionGuard g1(rewriter);
            rewriter.setInsertionPoint(forOp);
            Value fullConstResult =
                tensor::EmptyOp::create(rewriter, forOp.getLoc(), newShape, opTy.getElementType())
                    .getResult();

            scf::ForOp cloneForOp = scf::ForOp::create(
                rewriter, forOp.getLoc(), lb, ub, step, llvm::ArrayRef<Value>{fullConstResult}
            );

            {
                OpBuilder::InsertionGuard g2(rewriter);
                rewriter.setInsertionPointToStart(cloneForOp.getBody());
                SmallVector<Operation *, 8> opsToMove(
                    opsToMoveStack.rbegin(), opsToMoveStack.rend()
                );
                IRMapping mapping;
                auto lastV = cloneAndReplaceToBody(
                    rewriter, opsToMove, mapping, forOp.getBody(), cloneForOp.getBody()
                );
                for (auto moveOp : opsToMoveStack) {
                    if (moveOp->use_empty())
                        rewriter.eraseOp(moveOp);
                }
                auto iv = cloneForOp.getInductionVar();

                auto affineOp =
                    affine::AffineApplyOp::create(rewriter, op->getLoc(), newIvMap, ValueRange{iv});
                SmallVector<OpFoldResult, 4> sliceOffsets;
                SmallVector<OpFoldResult, 4> sliceSizes;
                SmallVector<OpFoldResult, 4> sliceStrides;

                sliceOffsets.push_back(affineOp.getResult());
                sliceSizes.push_back(rewriter.getIndexAttr(1));
                sliceStrides.push_back(rewriter.getIndexAttr(1));
                for (size_t i = 0; i < shape.size(); ++i) {
                    sliceOffsets.push_back(rewriter.getIndexAttr(0));
                    sliceSizes.push_back(rewriter.getIndexAttr(shape[i]));
                    sliceStrides.push_back(rewriter.getIndexAttr(1));
                }
                auto expandV =
                    tensor::ExpandShapeOp::create(
                        rewriter, op->getLoc(),
                        RankedTensorType::get(expandShape, opTy.getElementType()), lastV, *reassoc
                    )
                        .getResult();
                auto iOp = tensor::InsertSliceOp::create(
                    rewriter, op->getLoc(), expandV, cloneForOp.getRegionIterArgs()[0],
                    sliceOffsets, sliceSizes, sliceStrides
                );
                scf::YieldOp::create(rewriter, op->getLoc(), iOp.getResult());
            }
            setCompileTimeConstAttr(cloneForOp.getOperation());
            constV = cloneForOp.getResult(0);
            assert(
                succeeded(verify(constV.getDefiningOp())) && "Expected defining op for const result"
            );
        }

        {
            // Add the extract slice op to get the original for loop
            auto origIv = forOp.getInductionVar();
            SmallVector<OpFoldResult, 4> extractOffsets;
            SmallVector<OpFoldResult, 4> extractSizes;
            SmallVector<OpFoldResult, 4> extractStrides;

            auto affineOp =
                affine::AffineApplyOp::create(rewriter, op->getLoc(), newIvMap, ValueRange{origIv});
            extractOffsets.push_back(affineOp.getResult());
            extractSizes.push_back(rewriter.getIndexAttr(1));
            extractStrides.push_back(rewriter.getIndexAttr(1));
            for (size_t i = 0; i < shape.size(); ++i) {
                extractOffsets.push_back(rewriter.getIndexAttr(0));
                extractSizes.push_back(rewriter.getIndexAttr(shape[i]));
                extractStrides.push_back(rewriter.getIndexAttr(1));
            }
            auto extractOp = tensor::ExtractSliceOp::create(
                rewriter, op->getLoc(), constV, extractOffsets, extractSizes, extractStrides
            );
            auto collapseV = tensor::CollapseShapeOp::create(
                                 rewriter, op->getLoc(), opTy, extractOp.getResult(), *reassoc
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
