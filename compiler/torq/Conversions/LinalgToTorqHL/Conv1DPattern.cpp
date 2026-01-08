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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-conv1d-pattern"

namespace mlir::syna::torq {

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgMatmul\n";
        auto loc = convOp.getLoc();

        // Extract tensors and shapes
        Value input = convOp.getInputs()[0];   // Input tensor [N,C,W]
        Value filter = convOp.getInputs()[1];  // Filter tensor [F,C,Kw]
        Value output = convOp.getOutputs()[0]; // Output tensor [N,F,Ow]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Extract dimensions
        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        if (inputShape.size() != 3 || filterShape.size() != 3 || outputShape.size() != 3) {
            return rewriter.notifyMatchFailure(convOp, "Expected 3D tensors for Conv1D");
        }

        // Extract convolution parameters
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );

        int64_t N = inputShape[0];       // Batch size
        int64_t C = inputShape[1];       // Input channels
        int64_t F = filterShape[0];      // Output channels/filters
        int64_t Kw = filterShape[2];     // Kernel width
        int64_t Ow = outputShape[2];     // Output width
        int64_t stride = strides[0];     // Stride value
        int64_t dilation = dilations[0]; // Dilation value

        // Step 1: Unfold the input tensor using im2col approach
        // Each position in the output corresponds to a patch of the input
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        // Create a tensor to hold the unfolded input
        // Shape: [Ow, C*Kw] - each row contains a full patch for one output position
        SmallVector<int64_t> unfoldedShape = {Ow, C * Kw};
        auto unfoldedType = RankedTensorType::get(unfoldedShape, elemType);
        auto unfoldedInit = rewriter.create<tensor::EmptyOp>(loc, unfoldedShape, elemType);

        // Create the im2col transformation using a linalg.generic
        SmallVector<AffineExpr> unfoldIndexExprs;
        auto dim0 = rewriter.getAffineDimExpr(0); // Output position (Ow dimension)
        auto dim1 = rewriter.getAffineDimExpr(1); // Input channel and kernel position

        // dim1 / Kw gives us the channel index
        auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
        // dim1 % Kw gives us the kernel position
        auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
        // Calculate input position: outputPos * stride + kernelIdx * dilation
        auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                            kernelIdx * rewriter.getAffineConstantExpr(dilation);

        unfoldIndexExprs.push_back(rewriter.getAffineConstantExpr(0)); // N dimension (batch)
        unfoldIndexExprs.push_back(channelIdx);                        // C dimension (channels)
        unfoldIndexExprs.push_back(inputPosExpr);                      // W dimension (width)

        auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
        auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

        // Create the generic op for unfolding with explicit iterator types
        SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

        auto im2col = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{unfoldedType}, ValueRange{input}, ValueRange{unfoldedInit},
            ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap}, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            }
        );

        // Set torq.im2col attribute so that we can easily recognize this op during tiling
        im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
        auto unfoldedInput = im2col.getResult(0);

        // Step 2: Reshape the filter tensor from [F, C, Kw] to [F, C*Kw]
        SmallVector<int64_t> reshapedFilterShape = {F, C * Kw};
        auto reshapedFilterType =
            RankedTensorType::get(reshapedFilterShape, filterType.getElementType());
        auto reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );

        // Step 3: Create the matmul operation
        // We'll do: [F, C*Kw] @ [Ow, C*Kw]^T -> [F, Ow]
        // First, we need to transpose the unfolded input
        SmallVector<int64_t> transposedUnfoldedShape = {C * Kw, Ow};
        // auto transposedUnfoldedType = RankedTensorType::get(transposedUnfoldedShape, elemType);
        auto transposedUnfoldedInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedUnfoldedShape, elemType);

        auto transposedUnfolded = rewriter.create<linalg::TransposeOp>(
            loc, unfoldedInput, transposedUnfoldedInit, ArrayRef<int64_t>{1, 0}
        );

        // Create the matmul output tensor [F, Ow]
        SmallVector<int64_t> matmulResultShape = {F, Ow};
        auto matmulResultType = RankedTensorType::get(matmulResultShape, outputElemType);
        auto matmulInit = rewriter.create<tensor::EmptyOp>(loc, matmulResultShape, outputElemType);

        // Perform the actual matmul
        // Perform the actual matmul
        SmallVector<Value> inputs;
        inputs.push_back(reshapedFilter.getResult());
        inputs.push_back(transposedUnfolded.getResults()[0]);

        SmallVector<Value> outputs;
        outputs.push_back(matmulInit.getResult());

        auto matmulOp =
            rewriter.create<linalg::MatmulOp>(loc, TypeRange{matmulResultType}, inputs, outputs);

        // Step 4: Reshape the result back to [N, F, Ow]
        if (N == 1) {
            // Simply reshape to add the batch dimension
            auto finalResult = rewriter.create<tensor::ExpandShapeOp>(
                loc, matmulResultType, matmulOp.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, finalResult);
        }
        else {
            return rewriter.notifyMatchFailure(
                convOp, "Batched Conv1D not supported in this pattern"
            );
        }

        return success();
    }
};

struct Conv1DNcwFcwToLinalgConv2DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        // Guard: Prevent infinite loop with Conv2dConvert pattern.
        // This pattern converts Conv1D -> Conv2D with height=1 (NHWC format, dim 1 = 1).
        // Conv2dConvert detects Conv2D with height=1 and converts it back to Conv1D,
        // causing an infinite loop. To prevent this, we check if the operation has
        // already been processed or if we're in a context where the loop would occur.
        //
        // Check if this operation was created by a previous conversion attempt
        // (indicated by a marker attribute or by checking the defining operation)
        if (convOp->hasAttr("torq.conv1d_to_conv2d_processed")) {
            return rewriter.notifyMatchFailure(
                convOp, "Conv1D already processed, skipping to prevent infinite loop"
            );
        }

        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Add height dimension (1) to input: [N,C,W] -> [N,C,1,W]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], 1, inputType.getShape()[2]},
            inputType.getElementType()
        );

        auto expandedInput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedInputType, input, inputReassoc);

        // Transpose to NHWC format: [N,C,1,W] -> [N,1,W,C]
        SmallVector<int64_t> inputPerm = {0, 2, 3, 1};
        Value nhwcInput = transposeValue(expandedInput, inputPerm, loc, rewriter);

        // Add height dimension to filter: [F,C,W] -> [F,C,1,W]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], 1, filterType.getShape()[2]},
            filterType.getElementType()
        );

        auto expandedFilter =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedFilterType, filter, filterReassoc);

        // Transpose to HWCF format: [F,C,1,W] -> [1,W,C,F]
        SmallVector<int64_t> filterPerm = {2, 3, 1, 0};
        Value hwcfFilter = transposeValue(expandedFilter, filterPerm, loc, rewriter);

        // Add height dimension to output: [N,F,W] -> [N,F,1,W]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], 1, outputType.getShape()[2]},
            outputType.getElementType()
        );

        auto expandedOutput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedOutputType, output, outputReassoc);

        // Transpose to NHWC format: [N,F,1,W] -> [N,1,W,F]
        SmallVector<int64_t> outputPerm = {0, 2, 3, 1};
        Value nhwcOutput = transposeValue(expandedOutput, outputPerm, loc, rewriter);

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Convert 1D strides/dilations to 2D (add height dimension)
        SmallVector<int64_t> strides2d = {1};
        strides2d.push_back(stridesAttr.getValues<int64_t>()[0]);
        SmallVector<int64_t> dilations2d = {1};
        dilations2d.push_back(dilationsAttr.getValues<int64_t>()[0]);

        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        // Create Conv2D
        auto conv2d = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc, nhwcOutput.getType(), ValueRange{nhwcInput, hwcfFilter}, ValueRange{nhwcOutput},
            stridesAttr2d, dilationsAttr2d
        );

        // Mark the Conv2D to prevent Conv2dConvert from converting it back to Conv1D,
        // which would cause an infinite loop. Conv2dConvert should check for this
        // attribute and skip conversion if present.
        conv2d->setAttr("torq.skip_conv2d_to_conv1d", rewriter.getBoolAttr(true));

        // Transpose result back: [N,1,W,F] -> [N,F,1,W]
        Value transposedResult = transposeValue(conv2d.getResult(0), {0, 3, 1, 2}, loc, rewriter);

        // Collapse height dimension: [N,F,1,W] -> [N,F,W]
        auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
            loc, outputType, transposedResult, outputReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());
        return success();
    }
};

void populateLinalgToTorqHLConv1DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    // Conv1DNcwFcwToLinalgConv2DPattern now has a guard to prevent infinite loop:
    // It converts Conv1D to Conv2D with height=1, and marks the Conv2D with
    // "torq.skip_conv2d_to_conv1d" attribute. Conv2dConvert checks for this
    // attribute and skips conversion to prevent the infinite loop.
    else {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }
}

} // namespace mlir::syna::torq
