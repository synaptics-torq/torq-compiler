// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tensor-to-torq-or-linalg-pattern"

namespace mlir::syna::torq {

// DEADCODE:
// Converts a `tensor.collapse_shape` operation into a `linalg.generic`
// operation that explicitly materializes the collapsed tensor via a copy.
// Note that linalg implementation of TilingInterface requires that the indexing_map of the output
// satisfies AffineMap::isProjectedPermutation. For this reason we start from the output map being
// the identity map, and construct the input map using floorDiv and mod.
// Example:
// source:
//   tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<10x20x30x40xi8> into tensor<6000x40xi8>
// target:
//   %2 = tensor.empty() : tensor<6000x40xi8>
//   linalg.generic {
//     indexing_maps = [
//       (d0, d1) -> ((d0 / 20*30) mod 10, (d0 / 30) mod 20, d0 mod 30, d1),
//       (d0, d1) -> (d0, d1)
//     ],
//     iterator_types = ["parallel", "parallel"]
//   }
//   ins(%1, tensor<10x20x30x40xi8>)
//   outs(%2 : tensor<6000x40xi8>) {
//   ^bb0(%in: i8, %out: i8) :
//     linalg.yield %in : i8
//   } -> tensor<6000x40xi8>
struct CollapseShapeOpToLinalgRewrite : public OpRewritePattern<tensor::CollapseShapeOp> {
    using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;
    CollapseShapeOpToLinalgRewrite(MLIRContext *context)
        : OpRewritePattern<tensor::CollapseShapeOp>(context) {}
    LogicalResult
    matchAndRewrite(tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {

        // We only convert the operation if it's part of a pattern that will be converted to torqHL.
        if (!isMarkedFuseGroup(collapseOp))
            return failure();

        Value inputTensor = collapseOp.getSrc();
        TensorType inputTensorType = cast<TensorType>(inputTensor.getType());
        int64_t numInputDims = inputTensorType.getRank();

        // TODO: should this be RankedTensorType?
        TensorType outputTensorType = cast<TensorType>(collapseOp.getResult().getType());
        int64_t numOutputDims = outputTensorType.getRank();

        Value emptyOutput = rewriter.create<tensor::EmptyOp>(
            collapseOp.getLoc(), collapseOp.getResult().getType(), mlir::ValueRange()
        );

        // identity map (d0, d1, ...) -> (d0, d1, ...).
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(numOutputDims);

        SmallVector<AffineExpr, 4> inputMapExprs(numInputDims);

        // Iterate through the reassociation groups defined by `tensor.collapse_shape`.
        // Each group specifies a set of original dimensions that collapse into a single
        // new output dimension.
        for (auto [groupIndex, groupAttr] : llvm::enumerate(collapseOp.getReassociation())) {
            ArrayAttr reassocGroup = cast<ArrayAttr>(groupAttr);

            int64_t sizeProd = 1;

            // Iterate through the original dimensions *within the group* from right to left
            // (i.e., from the innermost dimension to the outermost dimension that contributed
            // to the current collapsed dimension).
            for (int i = reassocGroup.size() - 1; i >= 0; --i) {
                int64_t originalDimIdx = cast<IntegerAttr>(reassocGroup[i]).getInt();
                int64_t originalDimSize = inputTensorType.getDimSize(originalDimIdx);

                inputMapExprs[originalDimIdx] =
                    rewriter.getAffineDimExpr(groupIndex).floorDiv(sizeProd);
                if (i > 0) {
                    // The mod in the last index (left most) is redundant. This
                    // also eliminates the mod from singletons.
                    inputMapExprs[originalDimIdx] = inputMapExprs[originalDimIdx] % originalDimSize;
                }
                sizeProd = sizeProd * originalDimSize;
            }
        }

        // TODO: are we allowed to bypass the rewriter like this?
        AffineMap inputMap = AffineMap::get(numOutputDims, 0, inputMapExprs, rewriter.getContext());

        SmallVector<utils::IteratorType, 4> iteratorTypes(
            numOutputDims, utils::IteratorType::parallel
        );

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            collapseOp,
            /*resultTypes=*/TypeRange(outputTensorType),
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{emptyOutput},
            /*indexingMaps=*/llvm::ArrayRef({inputMap, outputMap}),
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                // The body of the linalg.generic: simply yield the input value.
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            },
            // It's important that we clone the torq-fuse-group attribute
            collapseOp->getAttrs()
        );

        return success();
    }
};

// DEADCODE:
// Converts a `tensor.expand_shape` operation into a `linalg.generic`
// operation that explicitly materializes the expand tensor via a copy.
// Example:
// source:
//   tensor.expand_shape %1 [[0, 1, 2], [3]] output_shape [10, 20, 30, 40] : tensor<6000x40xi8> into
//   tensor<10x20x30x40xi8>
// target:
//   %2 = tensor.empty() : tensor<10x20x30x40xi8>
//   linalg.generic {
//     indexing_maps = [
//       (d0, d1, d2, d3) -> (d0*20*30 + d1*30 + d2, d3)
//       (d0, d1, d2, d3) -> (d0, d1, d2, d3),
//     ],
//     iterator_types = ["parallel", "parallel"]
//   }
//   ins(%1 : tensor<6000x40xi8>)
//   outs(%2, tensor<10x20x30x40xi8>) {
//   ^bb0(%in: i8, %out: i8) :
//     linalg.yield %in : i8
//   } -> tensor<10x20x30x40xi8>
struct ExpandShapeOpToLinalgRewrite : public OpRewritePattern<tensor::ExpandShapeOp> {
    using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;
    ExpandShapeOpToLinalgRewrite(MLIRContext *context)
        : OpRewritePattern<tensor::ExpandShapeOp>(context) {}
    LogicalResult
    matchAndRewrite(tensor::ExpandShapeOp expandOp, PatternRewriter &rewriter) const override {

        // We only convert the operation if it's part of a pattern that will be converted to torqHL.
        if (!isMarkedFuseGroup(expandOp))
            return failure();

        Value inputTensor = expandOp.getSrc();
        TensorType inputTensorType = cast<TensorType>(inputTensor.getType());
        int64_t numInputDims = inputTensorType.getRank();

        // TODO: should this be RankedTensorType?
        TensorType outputTensorType = cast<TensorType>(expandOp.getResult().getType());
        int64_t numOutputDims = outputTensorType.getRank();

        Value emptyOutput = rewriter.create<tensor::EmptyOp>(
            expandOp.getLoc(), expandOp.getResult().getType(), mlir::ValueRange()
        );

        // identity map (d0, d1, ...) -> (d0, d1, ...).
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(numOutputDims);

        SmallVector<AffineExpr, 4> inputMapExprs(numInputDims);

        // Iterate through the reassociation groups defined by `tensor.collapse_shape`.
        // Each group specifies a set of original dimensions that collapse into a single
        // new output dimension.
        for (auto [groupIndex, groupAttr] : llvm::enumerate(expandOp.getReassociation())) {
            ArrayAttr reassocGroup = cast<ArrayAttr>(groupAttr);

            AffineExpr expr = rewriter.getAffineConstantExpr(0);
            AffineExpr exprDimSize = rewriter.getAffineConstantExpr(1);

            // Iterate through the original dimensions *within the group* from right to left
            // (i.e., from the innermost dimension to the outermost dimension that contributed
            // to the current expanded dimension).
            for (size_t i = reassocGroup.size(); i > 0; --i) {
                int64_t originalDimIdx = cast<IntegerAttr>(reassocGroup[i - 1]).getInt();
                int64_t originalDimSize = outputTensorType.getDimSize(originalDimIdx);
                AffineExpr originalDimExpr = rewriter.getAffineDimExpr(originalDimIdx);

                expr = expr + (originalDimExpr * exprDimSize);
                exprDimSize = exprDimSize * originalDimSize;
            }
            inputMapExprs[groupIndex] = expr;
        }

        // TODO: are we allowed to bypass the rewriter like this?
        AffineMap inputMap = AffineMap::get(numOutputDims, 0, inputMapExprs, rewriter.getContext());

        SmallVector<utils::IteratorType, 4> iteratorTypes(
            numOutputDims, utils::IteratorType::parallel
        );

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            expandOp,
            /*resultTypes=*/TypeRange(outputTensorType),
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{emptyOutput},
            /*indexingMaps=*/llvm::ArrayRef({inputMap, outputMap}),
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                // The body of the linalg.generic: simply yield the input value.
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            },
            // It's important that we clone the torq-fuse-group attribute
            expandOp->getAttrs()
        );

        return success();
    }
};

void populateTensorToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<CollapseShapeOpToLinalgRewrite>(context);
    patterns.insert<ExpandShapeOpToLinalgRewrite>(context);
}

} // namespace mlir::syna::torq
