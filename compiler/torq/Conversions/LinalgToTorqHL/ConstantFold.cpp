// Copyright 2024 Synaptics Inc.
// derived from
//===- ConstantFold.cpp - Implementation of constant folding on Linalg ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements constant folding on Linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "torq-fold-constants"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir::syna::torq {

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

} // namespace mlir::syna::torq
