// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GenericOpPassesDetail.h"

#include "torq/Dialect/TorqHL/GenericOp.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-merge-bias-scale"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq_hl {

namespace {

class MergeBiasScalePattern : public OpRewritePattern<torq_hl::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    AffineMap extendIndexingMap(PatternRewriter &rewriter, AffineMap orig, int dimValue) const {
        SmallVector<AffineExpr> exprs;

        exprs.append(orig.getResults().begin(), orig.getResults().end());

        exprs.push_back(getAffineConstantExpr(dimValue, orig.getContext()));

        return AffineMap::get(
            orig.getNumDims(), orig.getNumSymbols(), exprs, rewriter.getContext()
        );
    }

    LogicalResult
    matchAndRewrite(torq_hl::GenericOp genericOp, PatternRewriter &rewriter) const override {

        auto scale = genericOp.getScale();
        auto bias = genericOp.getBias();

        if (!scale || !bias) {
            return failure();
        }

        // check if the inputs are already merged
        if (scale == bias) {
            return failure();
        }

        if (scale.getType() != bias.getType()) {
            return failure();
        }

        // create a empty tensor to store the merged tensor

        auto origShape = cast<RankedTensorType>(scale.getType()).getShape();

        SmallVector<int64_t> mergedShape{origShape.begin(), origShape.end()};
        mergedShape.push_back(2);

        RankedTensorType mergedType = RankedTensorType::get(
            mergedShape, cast<RankedTensorType>(scale.getType()).getElementType()
        );

        auto emptyMerged =
            rewriter.create<tensor::EmptyOp>(genericOp.getLoc(), mergedType, ValueRange{});

        // fill the empty tensor with zeros, this will make folding easier later

        auto zero = rewriter.create<arith::ConstantOp>(
            genericOp.getLoc(), rewriter.getZeroAttr(mergedType.getElementType())
        );

        auto zeroMerged = rewriter
                              .create<linalg::FillOp>(
                                  genericOp.getLoc(), ValueRange(zero), ValueRange(emptyMerged)
                              )
                              .getResult(0);

        // insert the scale tensor into the merged tensor

        SmallVector<OpFoldResult> insertOffset{mergedShape.size(), rewriter.getIndexAttr(0)};

        SmallVector<OpFoldResult> insertSizes;
        for (auto i : mergedShape) {
            insertSizes.push_back(rewriter.getIndexAttr(i));
        }
        insertSizes[insertSizes.size() - 1] = rewriter.getIndexAttr(1);

        SmallVector<OpFoldResult> insertStrides{mergedShape.size(), rewriter.getIndexAttr(1)};

        auto scaleMerged = rewriter.create<tensor::InsertSliceOp>(
            genericOp.getLoc(), genericOp.getScale(), zeroMerged, insertOffset, insertSizes,
            insertStrides
        );

        // insert the bias tensor into the merged tensor

        insertOffset[insertOffset.size() - 1] = rewriter.getIndexAttr(1);

        auto merged = rewriter.create<tensor::InsertSliceOp>(
            genericOp.getLoc(), genericOp.getBias(), scaleMerged, insertOffset, insertSizes,
            insertStrides
        );

        // rewrite the generic op to use the merged tensor

        GenericOpConfig config = GenericOpConfig::fromOperation(genericOp);

        config.scale =
            GenericOpParam(merged, extendIndexingMap(rewriter, genericOp.getScaleMap().value(), 0));
        config.bias =
            GenericOpParam(merged, extendIndexingMap(rewriter, genericOp.getBiasMap().value(), 1));

        rewriter.replaceOpWithNewOp<torq_hl::GenericOp>(genericOp, config);

        return success();
    }
};

class MergeBiasScaleTensorsPass : public MergeBiasScaleTensorsBase<MergeBiasScaleTensorsPass> {
  public:
    using MergeBiasScaleTensorsBase<MergeBiasScaleTensorsPass>::MergeBiasScaleTensorsBase;

    void runOnOperation() override;
};

void MergeBiasScaleTensorsPass::runOnOperation() {

    auto &context = getContext();

    RewritePatternSet patterns(&context);

    patterns.add<MergeBiasScalePattern>(&context);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        getOperation().emitError() << "pass failed";
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMergeBiasScaleTensorsPass() {
    return std::make_unique<MergeBiasScaleTensorsPass>();
}

} // namespace mlir::syna::torq_hl
