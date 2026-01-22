// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

// #include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
// #include "torq/Conversions/LinalgToTorqHL/Patterns.h"

#include "torq/Utils/ComputeConstants.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <memory>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "torq-fold-constants"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::syna::torq {

using ControlFoldingFn = std::function<bool(Operation *operation)>;

namespace {

class FoldConstant : public OpInterfaceRewritePattern<LinalgOp> {
  public:
    FoldConstant(
        MLIRContext *context, const ControlFoldingFn &controlFn, PatternBenefit benefit = 1
    )
        : OpInterfaceRewritePattern<LinalgOp>(context, benefit), controlFn(controlFn) {}

    LogicalResult matchAndRewrite(LinalgOp linalgOp, PatternRewriter &rewriter) const override {

        // We need a control function to understand what we should fold
        if (!controlFn(linalgOp))
            return failure();

        DenseElementsAttr outputAttr = computeConstant(linalgOp, false);
        if (!outputAttr) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(linalgOp, outputAttr);

        return success();
    }

  private:
    ControlFoldingFn controlFn;
};

class FoldFillWithConstantPattern : public OpInterfaceRewritePattern<linalg::LinalgOp> {

  public:
    FoldFillWithConstantPattern(
        MLIRContext *context, const ControlFoldingFn &controlFn, PatternBenefit benefit = 1
    )
        : OpInterfaceRewritePattern<linalg::LinalgOp>(context, benefit), controlFn(controlFn) {}

    LogicalResult
    matchAndRewrite(linalg::LinalgOp linalgOp, PatternRewriter &rewriter) const override {

        if (linalgOp.getNumDpsInits() != 1) {
            return failure();
        }

        if (linalgOp.getNumDpsInputs() != 1) {
            return failure();
        }

        // we only consider fill-like operations
        if (linalgOp.getBlock()->getOperations().size() > 1) {
            return failure();
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(linalgOp.getBlock()->getTerminator());

        if (!yieldOp || yieldOp.getNumOperands() != 1) {
            return failure();
        }

        auto yieldVal = dyn_cast<BlockArgument>(yieldOp.getOperand(0));

        if (!yieldVal) {
            return failure();
        }

        auto fillValue = linalgOp.getMatchingOpOperand(yieldVal)->get();

        if (fillValue.getType() != rewriter.getI32Type()) {
            return failure();
        }

        auto fillConstOp = fillValue.getDefiningOp<arith::ConstantOp>();

        if (!fillConstOp) {
            return failure();
        }

        auto intAttr = dyn_cast<IntegerAttr>(fillConstOp.getValue());

        if (!intAttr) {
            return failure();
        }

        if (!controlFn(linalgOp)) {
            return failure();
        }

        auto outputType = dyn_cast<ShapedType>(linalgOp->getResult(0).getType());

        if (!outputType) {
            return failure();
        }

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            linalgOp, SplatElementsAttr::get(outputType, intAttr)
        );

        return success();
    }

  private:
    ControlFoldingFn controlFn;
};

class FoldInsertSlicePattern : public OpRewritePattern<tensor::InsertSliceOp> {

  public:
    FoldInsertSlicePattern(
        MLIRContext *context, const ControlFoldingFn &controlFn, PatternBenefit benefit = 1
    )
        : OpRewritePattern<tensor::InsertSliceOp>(context, benefit), controlFn(controlFn) {}

    LogicalResult
    matchAndRewrite(tensor::InsertSliceOp insertSliceOp, PatternRewriter &rewriter) const override {

        if (!controlFn(insertSliceOp)) {
            return failure();
        }

        for (int i = 0; i < insertSliceOp.getResultType().getRank(); i++) {
            if (insertSliceOp.isDynamicOffset(i) || insertSliceOp.isDynamicSize(i) ||
                insertSliceOp.isDynamicStride(i)) {
                return failure();
            }
        }

        ShapedType destType = cast<ShapedType>(insertSliceOp.getDest().getType());
        ShapedType sourceType = cast<ShapedType>(insertSliceOp.getSource().getType());

        SmallVector<APInt> intOutputValues;

        auto destConst = insertSliceOp.getDest().getDefiningOp<arith::ConstantOp>();

        if (!destConst) {
            return failure();
        }

        auto sourceConst = insertSliceOp.getSource().getDefiningOp<arith::ConstantOp>();

        if (!sourceConst) {
            return failure();
        }

        auto destData = cast<DenseIntElementsAttr>(destConst.getValue());

        if (!destData) {
            return failure();
        }

        auto sourceData = cast<DenseIntElementsAttr>(sourceConst.getValue());

        if (!sourceData) {
            return failure();
        }

        auto destValue = destData.getValues<APInt>();
        auto sourceValue = sourceData.getValues<APInt>();

        auto maybeRankReductionMask =
            computeRankReductionMask(insertSliceOp.getStaticSizes(), sourceType.getShape());

        if (!maybeRankReductionMask) {
            return failure();
        }

        auto destStrides = computeStrides(destType.getShape());
        auto srcStrides = computeStrides(sourceType.getShape());

        for (int linearIndex = 0; linearIndex < destType.getNumElements(); linearIndex++) {

            SmallVector<int64_t> destIdx{delinearize(linearIndex, destStrides)};

            bool inSource = true;
            SmallVector<int64_t> sourceIdx;

            for (int i = 0; i < destType.getRank(); i++) {

                if (destIdx[i] < insertSliceOp.getStaticOffset(i) ||
                    destIdx[i] >=
                        insertSliceOp.getStaticOffset(i) + insertSliceOp.getStaticSize(i)) {
                    inSource = false;
                    break;
                }

                // add the index if this dimension is not one of the dimensions that doesn't exist
                // in the source
                if (!maybeRankReductionMask->contains(i)) {
                    sourceIdx.push_back(destIdx[i] - insertSliceOp.getStaticOffset(i));
                }
            }

            if (inSource) {

                auto sourceLinearIndex = linearize(sourceIdx, srcStrides);
                intOutputValues.push_back(sourceValue[sourceLinearIndex]);
            }
            else {
                intOutputValues.push_back(destValue[linearIndex]);
            }
        }

        DenseElementsAttr outputAttr =
            DenseElementsAttr::get(insertSliceOp.getResultType(), intOutputValues);

        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            insertSliceOp, insertSliceOp.getResultType(), outputAttr
        );

        return success();
    }

  private:
    ControlFoldingFn controlFn;
};

} // namespace

void populateTorqConstantFoldLinalgOperations(
    RewritePatternSet &patterns, const ControlFoldingFn &controlFn
) {
    MLIRContext *context = patterns.getContext();

    patterns.insert<FoldConstant>(context, controlFn);
    patterns.insert<FoldFillWithConstantPattern>(context, controlFn);
    patterns.insert<FoldInsertSlicePattern>(context, controlFn);
}

class FoldConstantsPass : public FoldConstantsBase<FoldConstantsPass> {
  public:
    using FoldConstantsBase::FoldConstantsBase;

    int computeMemorySize(Type type) {
        auto rankedTensorType = dyn_cast<RankedTensorType>(type);

        if (!rankedTensorType)
            return type.getIntOrFloatBitWidth();

        return rankedTensorType.getNumElements() * rankedTensorType.getElementTypeBitWidth();
    }

    // find all the linalg op or tensor ops that can potentially be folded
    static DenseSet<Operation *> getFoldableOperations(FunctionOpInterface func) {

        // FIXME: this function is overly optimistic estimating what we can fold
        // our patterns cannot yet fold all the operations we declare foldable here
        // so we may end up assuming we can fold something that we actually can't
        // so we may miscalulate the size of the code we can save

        DenseSet<Operation *> opsFromConstants;

        for (auto &op : func.getFunctionBody().getOps()) {

            // we fold only operations that support the linalgop interface or
            // some selected tensor operations
            if (!isa<linalg::LinalgOp, tensor::InsertSliceOp>(op)) {
                continue;
            }

            bool isConstant = true;

            for (auto operand : op.getOperands()) {

                // the operand is not constant, we cannot fold it
                if (!operand.getDefiningOp()) {
                    isConstant = false;
                    break;
                }

                if (operand.getDefiningOp<arith::ConstantOp>() ||
                    operand.getDefiningOp<tensor::EmptyOp>()) {
                    continue;
                }

                if (!opsFromConstants.contains(operand.getDefiningOp())) {
                    isConstant = false;
                    break;
                }
            }

            if (isConstant) {
                opsFromConstants.insert(&op);
            }
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Found " << opsFromConstants.size() << " foldable operations\n";
            llvm::dbgs() << "Foldable operations:\n";
            for (auto op : opsFromConstants) {
                op->dump();
            }
        });

        return opsFromConstants;
    }

    // divide the foldable operations into clusters of operations that are connected in the use-def
    // graph
    static SmallVector<DenseSet<Operation *>>
    computeFoldingClusters(DenseSet<Operation *> foldableOperations) {

        SmallVector<DenseSet<Operation *>> clusters;

        while (!foldableOperations.empty()) {

            DenseSet<Operation *> cluster;

            Operation *op = *foldableOperations.begin();
            foldableOperations.erase(op);

            DenseSet<Operation *> opsToProcess;
            opsToProcess.insert(op);

            while (!opsToProcess.empty()) {
                Operation *op1 = *opsToProcess.begin();

                opsToProcess.erase(op1);

                cluster.insert(op1);

                // find all the constant predecessors of the operation and put them in the same
                // cluster
                for (auto operand : op1->getOperands()) {
                    if (foldableOperations.contains(operand.getDefiningOp())) {
                        foldableOperations.erase(operand.getDefiningOp());
                        opsToProcess.insert(operand.getDefiningOp());
                    }
                }

                // find all the constant successors of the operation and put them it the same
                // cluster
                for (auto user : op1->getUsers()) {
                    if (foldableOperations.contains(user)) {
                        foldableOperations.erase(user);
                        opsToProcess.insert(user);
                    }
                }
            }

            clusters.push_back(cluster);
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Found " << clusters.size() << " clusters\n";
            for (auto cluster : clusters) {
                llvm::dbgs() << "\n";
                llvm::dbgs() << "Cluster:\n";
                for (auto op : cluster) {
                    op->dump();
                }
            }

            llvm::dbgs() << "\n";
        });

        return clusters;
    }

    // Compute the memory used before and after folding a given cluster and add to the list of
    // foldable operations all the operations that are in a cluster that makes sense to fold
    DenseSet<Operation *> findFoldingChoices(SmallVector<DenseSet<Operation *>> clusters) {

        DenseSet<Operation *> foldableOperations;

        for (auto cluster : clusters) {

            DenseSet<Value> constantsBefore;
            DenseSet<Value> constantsAfter;

            for (auto op : cluster) {

                for (auto operand : op->getOperands()) {
                    if (operand.getDefiningOp<arith::ConstantOp>()) {

                        constantsBefore.insert(operand);

                        if (constantsAfter.contains(operand)) {
                            continue;
                        }

                        // add the constants in the constants after if the constant is used by
                        // another operation not in the cluster
                        for (auto user : operand.getUsers()) {
                            if (!cluster.contains(user)) {
                                constantsAfter.insert(operand);
                                break;
                            }
                        }
                    }
                }

                for (auto result : op->getResults()) {
                    if (constantsAfter.contains(result)) {
                        continue;
                    }

                    for (auto user : result.getUsers()) {
                        if (!cluster.contains(user)) {
                            constantsAfter.insert(result);
                            break;
                        }
                    }
                }
            }

            int memoryBefore = 0;
            for (auto constant : constantsBefore) {
                memoryBefore += computeMemorySize(constant.getType());
            }

            int memoryAfter = 0;
            for (auto constant : constantsAfter) {
                memoryAfter += computeMemorySize(constant.getType());
            }

            if (memoryBefore >= memoryAfter) {
                foldableOperations.insert(cluster.begin(), cluster.end());
            }

            LLVM_DEBUG({
                llvm::dbgs() << "\n";
                llvm::dbgs() << "Cluster with following ops, memory before: " << memoryBefore
                             << " memory after: " << memoryAfter << ":\n";
                for (auto op : cluster) {
                    op->dump();
                }
                llvm::dbgs() << "\n";
            });
        }

        return foldableOperations;
    }

    DenseSet<Operation *> analyzeFunction(FunctionOpInterface func) {

        // find all the operations we could fold, these are operators that
        // implement the linalg interface and
        // depend only on constant inputs (which may be in turn other
        // generic ops with constant inputs or actual constant ops)
        auto foldableOperations = getFoldableOperations(func);

        // create groups of operations that we want to fold together
        auto clusters = computeFoldingClusters(foldableOperations);

        // find all operation in groups that ensure after we fold
        // the code size is smaller or the same as the current code size
        return findFoldingChoices(clusters);
    }

    void runOnOperation() override {

        auto &context = getContext();

        RewritePatternSet patterns(&context);

        auto toFold = analyzeFunction(cast<FunctionOpInterface>(getOperation()));

        auto controlFn = [&toFold](Operation *op) { return toFold.contains(op); };

        torq::populateTorqConstantFoldLinalgOperations(patterns, controlFn);

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            getOperation().emitError() << "pass failed";
            return signalPassFailure();
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFoldConstantsPass() {
    return std::make_unique<FoldConstantsPass>();
}

} // namespace mlir::syna::torq
