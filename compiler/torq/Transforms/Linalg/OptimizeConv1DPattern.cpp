// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-conv1d-pattern"

namespace mlir::syna::torq {

llvm::cl::opt<bool> clConv1dAsMatmul(
    "torq-convert-conv1d-to-matmul", llvm::cl::desc("Convert conv1d to imToCol + matmul"),
    llvm::cl::init(true)
);

llvm::cl::opt<bool> clConv1dToGenericConv1D(
    "torq-convert-conv1d-to-generic",
    llvm::cl::desc("Convert conv1d to generic conv1d (5D output with preserved kernel dim)"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clConv1dTruncateForReduce(
    "torq-conv1d-truncate-for-reduce",
    llvm::cl::desc("Truncate conv1d output to bf16 before reduce sum to save memory bandwidth. "
                   "When false (default): conv1d outputs f32 directly to reduce (accurate). "
                   "When true: insert truncf between conv1d and reduce (memory efficient)."),
    llvm::cl::init(false)
);

/// Optimization pattern for linalg.conv_1d operation.

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
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

        // Step 1: im2col - unfold input patches into [Ow, C*Kw]
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();

        // Collapse [N, C, W] -> [C, W] so the indexing map is rank-2 with no constant dims.
        // Constant dims in linalg.generic indexing maps get folded by canonicalization,
        // causing an operand-rank vs map-result-rank mismatch in downstream passes.
        SmallVector<int64_t> collapsedInputShape = {C, inputShape[2]};
        auto collapsedInputType = RankedTensorType::get(collapsedInputShape, elemType);
        auto collapsedInput = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1}, {2}}
        );

        SmallVector<int64_t> unfoldedShape = {Ow, C * Kw};
        auto unfoldedType = RankedTensorType::get(unfoldedShape, elemType);
        auto unfoldedInit = tensor::EmptyOp::create(rewriter, loc, unfoldedShape, elemType);

        // (d0=Ow, d1=C*Kw) -> (d1/Kw, d0*stride + (d1%Kw)*dilation)
        auto dim0 = rewriter.getAffineDimExpr(0);
        auto dim1 = rewriter.getAffineDimExpr(1);
        auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
        auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
        auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                            kernelIdx * rewriter.getAffineConstantExpr(dilation);

        SmallVector<AffineExpr> unfoldIndexExprs = {channelIdx, inputPosExpr};
        auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
        auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

        // Create the generic op for unfolding with explicit iterator types
        SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

        auto im2col = linalg::GenericOp::create(
            rewriter, loc, TypeRange{unfoldedType}, ValueRange{collapsedInput},
            ValueRange{unfoldedInit}, ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap},
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                linalg::YieldOp::create(nestedBuilder, nestedLoc, blockArgs[0]);
            }
        );

        // Set torq.im2col attribute so that we can easily recognize this op during tiling
        im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
        auto unfoldedInput = im2col.getResult(0);

        // Step 2: Reshape the filter tensor from [F, C, Kw] to [F, C*Kw]
        SmallVector<int64_t> reshapedFilterShape = {F, C * Kw};
        auto reshapedFilterType =
            RankedTensorType::get(reshapedFilterShape, filterType.getElementType());
        auto reshapedFilter = tensor::CollapseShapeOp::create(
            rewriter, loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );

        // Step 3: Create the matmul operation
        // We'll do: [Ow, C*Kw] @ [C*Kw, F] -> [Ow, F]
        // Transpose the filter [F, C*Kw] -> [C*Kw, F] instead of transposing im2col output
        SmallVector<int64_t> transposedFilterShape = {C * Kw, F};
        auto transposedFilterInit = tensor::EmptyOp::create(
            rewriter, loc, transposedFilterShape, filterType.getElementType()
        );

        auto transposedFilter = linalg::TransposeOp::create(
            rewriter, loc, reshapedFilter.getResult(), transposedFilterInit, ArrayRef<int64_t>{1, 0}
        );

        // [Ow, C*Kw] x [C*Kw, F] -> [Ow, F]; accumulate into output element type (f32)
        SmallVector<int64_t> matmulResultShape = {Ow, F};
        auto matmulResultType = RankedTensorType::get(matmulResultShape, outputElemType);
        auto matmulInit = tensor::EmptyOp::create(rewriter, loc, matmulResultShape, outputElemType);

        SmallVector<Value> inputs;
        inputs.push_back(unfoldedInput);
        inputs.push_back(transposedFilter.getResults()[0]);

        SmallVector<Value> outputs;
        outputs.push_back(matmulInit.getResult());

        auto matmulOp =
            linalg::MatmulOp::create(rewriter, loc, TypeRange{matmulResultType}, inputs, outputs);

        // Transpose result [Ow, F] -> [F, Ow]
        SmallVector<int64_t> transposedResultShape = {F, Ow};
        auto transposedResultInit =
            tensor::EmptyOp::create(rewriter, loc, transposedResultShape, outputElemType);
        auto transposedResult = linalg::TransposeOp::create(
            rewriter, loc, matmulOp.getResults()[0], transposedResultInit, ArrayRef<int64_t>{1, 0}
        );

        // Step 4: expand [F, Ow] -> [N, F, Ow]
        if (N == 1) {
            SmallVector<int64_t> expandedShape = {N, F, Ow};
            auto expandedType = RankedTensorType::get(expandedShape, outputElemType);
            auto expandedResult = tensor::ExpandShapeOp::create(
                rewriter, loc, expandedType, transposedResult.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, expandedResult.getResult());
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
            tensor::ExpandShapeOp::create(rewriter, loc, expandedInputType, input, inputReassoc);

        // Add height dimension to filter: [F,C,W] -> [F,C,1,W]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], 1, filterType.getShape()[2]},
            filterType.getElementType()
        );

        auto expandedFilter =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedFilterType, filter, filterReassoc);

        // Add height dimension to output: [N,F,W] -> [N,F,1,W]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], 1, outputType.getShape()[2]},
            outputType.getElementType()
        );

        auto expandedOutput =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedOutputType, output, outputReassoc);

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

        auto conv2d = linalg::Conv2DNchwFchwOp::create(
            rewriter, loc, expandedOutput.getType(), ValueRange{expandedInput, expandedFilter},
            ValueRange{expandedOutput}, stridesAttr2d, dilationsAttr2d
        );

        // check if there is linalg.genericOp does truncf for the conv1d result, if so, we need to
        // create a new linalg.genericOp to do the truncf for new conv2d result, and remove the
        // old one

        auto conv2dResult = conv2d.getResult(0);
        auto currentResult = conv2dResult;

        linalg::GenericOp truncfGenericOp = nullptr;

        auto conv1dResult = convOp.getResult(0);

        // if convOp has only one use
        if (conv1dResult.hasOneUse()) {
            auto userOp = conv1dResult.use_begin()->getOwner();
            if (auto genericOp = llvm::dyn_cast<linalg::GenericOp>(userOp)) {

                if (genericOp.getNumResults() == 1 &&
                    dyn_cast<RankedTensorType>(genericOp.getResult(0).getType())
                        .getElementType()
                        .isBF16() &&
                    dyn_cast<RankedTensorType>(conv1dResult.getType()).getElementType().isF32()) {

                    // This is likely a truncf op, we need to create a new one for the new conv2d
                    // result new truncf result type is the same as the conv2d result type but with
                    // float16 element type
                    auto truncfResultType = RankedTensorType::get(
                        cast<RankedTensorType>(conv2dResult.getType()).getShape(),
                        rewriter.getBF16Type()
                    );
                    auto truncfInit = tensor::EmptyOp::create(
                        rewriter, loc, cast<RankedTensorType>(conv2dResult.getType()).getShape(),
                        rewriter.getBF16Type()
                    );

                    // TODO: Rank
                    auto newUnfoldIndexMap =
                        AffineMap::getMultiDimIdentityMap(4, rewriter.getContext());
                    auto newOutputIndexMap =
                        AffineMap::getMultiDimIdentityMap(4, rewriter.getContext());
                    // new linalg.generic op need to be parallel as it is just doing elementwise
                    // truncation
                    SmallVector<utils::IteratorType> newIteratorTypes(
                        4, utils::IteratorType::parallel
                    );

                    auto newTruncf = linalg::GenericOp::create(
                        rewriter, loc, TypeRange{truncfResultType}, ValueRange{conv2dResult},
                        ValueRange{truncfInit},
                        ArrayRef<AffineMap>{newUnfoldIndexMap, newOutputIndexMap}, newIteratorTypes,
                        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                            auto truncf = arith::TruncFOp::create(
                                nestedBuilder, nestedLoc, rewriter.getBF16Type(), blockArgs[0]
                            );
                            linalg::YieldOp::create(nestedBuilder, nestedLoc, truncf.getResult());
                        }
                    );

                    currentResult = newTruncf.getResult(0);

                    // assign as late as possible
                    truncfGenericOp = genericOp;
                }
            }
        }

        // Collapse height dimension: [N,F,1,W] -> [N,F,W]
        // create a new reassociation indices for collapsing the height dimension
        SmallVector<ReassociationIndices> collapseReassoc = {{0}, {1}, {2, 3}};
        auto collapsedResultType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], outputType.getShape()[2]},
            cast<RankedTensorType>(currentResult.getType()).getElementType()
        );

        auto collapsedResult = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedResultType, currentResult, collapseReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());

        if (truncfGenericOp) {
            currentResult = collapsedResult.getResult();
            truncfGenericOp->replaceAllUsesWith(ValueRange{currentResult});
            rewriter.eraseOp(truncfGenericOp);
        }

        return success();
    }
};

/// Converts linalg::Conv1DNcwFcwOp to a linalg::GenericOp with expanded dimensions.
///
/// This pattern transforms a 1D convolution into a 5D generic operation that explicitly
/// computes the convolution while preserving the kernel dimension for later reduction.
/// The transformation maintains NCHW layout throughout.
///
/// Input/Output Shapes:
///   - Input: [N, C, W] (3D) -> expand -> [N, C, 1, W] (4D NCHW)
///   - Filter: [F, C, Kw] (3D) -> expand -> [F, C, 1, Kw] (4D FCHW)
///   - Output: [N, F, Ow] (3D) -> becomes intermediate [N, F, 1, Ow, Kw] (5D NCHW)
///
/// Algorithm:
///   1. Expand input and filter tensors by adding height=1 dimension (NCHW layout)
///   2. Create 5D generic op with all-parallel iterators:
///      - Output indexing: (n, f, kh, ow, kw) -> (n, f, kh, ow, kw)
///      - Input indexing:  (n, f, kh, ow, kw) -> (n, 0, kh, ow*stride + kw*dilation)
///      - Filter indexing: (n, f, kh, ow, kw) -> (f, 0, kh, kw)
///   3. Element-wise multiply input and filter in f32, accumulate into 5D output
///   4. Optionally insert truncf (f32 -> bf16) BEFORE reduce sum based on mode:
///      - Accurate mode (default): Keep f32 input for best accuracy
///      - Memory-optimized mode (--torq-conv1d-truncate-for-reduce): Insert truncf to save
///      bandwidth
///   5. Reduce over kernel dimension (kw) to get [N, F, 1, Ow]. Output is f32 for
///      accurate accumulation, regardless of input type.
///   6. Collapse height dimension to get [N, F, Ow] (f32 type)
///   7. Insert truncf (f32 -> bf16) AFTER reduce sum if final output should be bf16.
///
/// Memory Optimization (--torq-conv1d-truncate-for-reduce):
///   - false (default, accurate): Conv1d(f32) -> Reduce(f32->f32) -> truncf
///     Best accuracy with full f32 reduce input.
///   - true (memory-optimized): Conv1d(f32) -> truncf -> Reduce(bf16->f32) -> truncf
///     Lower memory bandwidth, slight precision loss from bf16 intermediate.
///
/// The preserved kernel dimension (Kw) in the 5D output allows the downstream
/// LinalgGenericConv1DToTorqHLConv1DPattern to directly lower to torq_hl.conv1d.
struct Conv1DNcwFcwToGenericConv1DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];   // [N, C, W]
        Value filter = convOp.getInputs()[1];  // [F, C, Kw]
        Value output = convOp.getOutputs()[0]; // [N, F, Ow]

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        if (!inputType.getElementType().isFloat()) {
            return rewriter.notifyMatchFailure(
                convOp, "Only floating point element type supported at the moment"
            );
        }

        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();
        ArrayRef<int64_t> outputShape = outputType.getShape();

        int64_t N = inputShape[0];
        int64_t C = inputShape[1];
        int64_t F = filterShape[0];
        int64_t Kw = filterShape[2];
        int64_t Ow = outputShape[2];

        // Expand input from [N, C, W] to [N, C, 1, W] (add height dimension)
        SmallVector<int64_t> input4DShape = {N, C, 1, inputShape[2]};
        auto input4DType = RankedTensorType::get(input4DShape, inputType.getElementType());
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto input4D =
            tensor::ExpandShapeOp::create(rewriter, loc, input4DType, input, inputReassoc);

        // Expand filter from [F, C, Kw] to [F, C, 1, Kw] (add height dimension)
        SmallVector<int64_t> filter4DShape = {F, C, 1, Kw};
        auto filter4DType = RankedTensorType::get(filter4DShape, filterType.getElementType());
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto filter4D =
            tensor::ExpandShapeOp::create(rewriter, loc, filter4DType, filter, filterReassoc);

        // Get convolution parameters
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );

        int64_t stride = strides[0];
        int64_t dilation = dilations[0];

        auto outputElemType = outputType.getElementType();

        // Create 5D output tensor: [N, F, 1, Ow, Kw]
        // The extra dimension preserves the kernel for later processing
        // Layout is NCHW: [batch, filters, height=1, output_width, kernel_width]
        SmallVector<int64_t> output5DShape = {N, F, 1, Ow, Kw};

        // Create indexing maps for linalg.generic
        // Input:  (n, f, kh, ow, kw) -> (n, 0, kh, ow * stride + kw * dilation)
        // Filter: (n, f, kh, ow, kw) -> (f, 0, kh, kw)
        // Output: (n, f, kh, ow, kw) -> (n, f, kh, ow, kw)

        SmallVector<AffineExpr> inputMapExprs;
        auto n = rewriter.getAffineDimExpr(0);
        auto f = rewriter.getAffineDimExpr(1);
        auto kh = rewriter.getAffineDimExpr(2);
        auto ow = rewriter.getAffineDimExpr(3);
        auto kw = rewriter.getAffineDimExpr(4);

        // Input indexing: (n, c=0, kh, ow * stride + kw * dilation)
        // 4D input [N, C, H, W] expanded from 3D [N, C, W]
        inputMapExprs.push_back(n);
        inputMapExprs.push_back(rewriter.getAffineConstantExpr(0)); // channel
        inputMapExprs.push_back(kh);                                // height dimension
        inputMapExprs.push_back(
            ow * rewriter.getAffineConstantExpr(stride) +
            kw * rewriter.getAffineConstantExpr(dilation)
        );

        auto inputMap = AffineMap::get(5, 0, inputMapExprs, rewriter.getContext());

        // Filter indexing: (f, c=0, kh, kw)
        // 4D filter [F, C, H, Kw] expanded from 3D [F, C, Kw]
        SmallVector<AffineExpr> filterMapExprs;
        filterMapExprs.push_back(f);
        filterMapExprs.push_back(rewriter.getAffineConstantExpr(0)); // channel
        filterMapExprs.push_back(kh);                                // height dimension
        filterMapExprs.push_back(kw);

        auto filterMap = AffineMap::get(5, 0, filterMapExprs, rewriter.getContext());

        // Output indexing: [n, f, kh, ow, kw]
        auto outputMap = AffineMap::getMultiDimIdentityMap(5, rewriter.getContext());

        // Iterator types: all parallel - reduction happens in ReduceOp
        SmallVector<utils::IteratorType> iteratorTypes = {
            utils::IteratorType::parallel, // n (batch)
            utils::IteratorType::parallel, // f (filters)
            utils::IteratorType::parallel, // kh (height dimension, always 1)
            utils::IteratorType::parallel, // ow (output width)
            utils::IteratorType::parallel  // kw (kernel width - will be reduced later)
        };

        // Detect if there's a truncf generic following the conv1d.
        // If so, we'll insert truncf BEFORE the reduce sum (not after conv1d).
        // The conv1d still outputs f32 for computation accuracy.
        auto convResult = convOp.getResult(0);
        linalg::GenericOp truncfGenericOp = nullptr;
        Type computeElemType = rewriter.getF32Type(); // f32 for computation
        Type resultElemType = rewriter.getF32Type();  // f32 output (don't fuse truncf)

        // Detect truncfGenericOp to determine if output should be bf16
        if (convResult.hasOneUse()) {
            auto userOp = convResult.use_begin()->getOwner();
            if (auto truncfOp = llvm::dyn_cast<linalg::GenericOp>(userOp)) {
                // Check if it's a truncf generic: f32 input → bf16 output
                if (truncfOp.getNumResults() == 1 &&
                    cast<RankedTensorType>(truncfOp.getResult(0).getType())
                        .getElementType()
                        .isBF16() &&
                    cast<RankedTensorType>(convResult.getType()).getElementType().isF32()) {

                    // Found truncf: f32 -> bf16. We'll insert this BEFORE reduce sum.
                    truncfGenericOp = truncfOp;
                }
            }
        }

        // Create output type (may be bf16 if truncf is fused)
        auto finalOutput5DType = RankedTensorType::get(output5DShape, resultElemType);
        auto finalOutput5DInit =
            linalg::FillOp::create(
                rewriter, loc, ValueRange{createZeroConstant(rewriter, loc, resultElemType)},
                ValueRange{tensor::EmptyOp::create(rewriter, loc, output5DShape, resultElemType)}
            )
                .result();

        // Create the generic operation with f32 output for computation accuracy.
        // Truncf to bf16 will be inserted before reduce sum if needed.
        auto generic = linalg::GenericOp::create(
            rewriter, loc, TypeRange{finalOutput5DType},
            ValueRange{input4D.getResult(), filter4D.getResult()}, ValueRange{finalOutput5DInit},
            ArrayRef<AffineMap>{inputMap, filterMap, outputMap}, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                Value inputVal = blockArgs[0];
                Value filterVal = blockArgs[1];
                Value accum = blockArgs[2];

                // Convert inputs to f32 for computation
                if (inputVal.getType() != computeElemType) {
                    inputVal =
                        arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, inputVal);
                }
                if (filterVal.getType() != computeElemType) {
                    filterVal =
                        arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, filterVal);
                }
                // Accumulator is f32 (resultElemType = f32)
                if (accum.getType() != computeElemType) {
                    accum = arith::ExtFOp::create(nestedBuilder, nestedLoc, computeElemType, accum);
                }

                // Multiply input and filter
                auto mul = arith::MulFOp::create(nestedBuilder, nestedLoc, inputVal, filterVal);

                // Accumulate
                auto add = arith::AddFOp::create(nestedBuilder, nestedLoc, mul, accum);

                // Output f32 - truncf to bf16 will be done before reduce sum if needed
                linalg::YieldOp::create(nestedBuilder, nestedLoc, add.getResult());
            }
        );

        auto genericResult = generic.getResult(0);

        // Reduce over the kernel dimension [4] to get back to 4D output [N, F, 1, Ow]
        // This ensures compatibility with existing consumers expecting 4D output
        SmallVector<int64_t> reducedShape = {N, F, 1, Ow};

        // Determine reduce input type based on optimization mode:
        // - Accurate mode (clConv1dTruncateForReduce=false): conv1d -> f32 -> reduce (f32 input)
        // - Memory-optimized mode (clConv1dTruncateForReduce=true): conv1d -> truncf -> bf16 ->
        // reduce
        Value reduceInput = genericResult;
        Type reduceInputType = rewriter.getF32Type();

        // Remember if we need bf16 output (either from conv output type or from following truncf)
        bool needsBF16Output = outputElemType.isBF16() || truncfGenericOp;

        // In accurate mode (default), keep f32 input to reduce for best accuracy.
        // In memory-optimized mode, insert truncf before reduce to save bandwidth.
        if (clConv1dTruncateForReduce && truncfGenericOp) {
            // Memory-optimized mode: Insert truncf before reduce (f32 -> bf16)
            // This reduces memory bandwidth but may lose some precision.
            auto bf16Type = rewriter.getBF16Type();
            auto truncfShape = cast<RankedTensorType>(genericResult.getType()).getShape();
            auto truncfResultType = RankedTensorType::get(truncfShape, bf16Type);
            auto truncfInit = tensor::EmptyOp::create(rewriter, loc, truncfShape, bf16Type);

            SmallVector<AffineExpr> truncfExprs;
            for (unsigned i = 0; i < 5; i++) {
                truncfExprs.push_back(rewriter.getAffineDimExpr(i));
            }
            auto truncfMap = AffineMap::get(5, 0, truncfExprs, rewriter.getContext());
            SmallVector<utils::IteratorType> truncfIterators(5, utils::IteratorType::parallel);

            auto truncfGeneric = linalg::GenericOp::create(
                rewriter, loc, TypeRange{truncfResultType}, ValueRange{genericResult},
                ValueRange{truncfInit}, ArrayRef<AffineMap>{truncfMap, truncfMap}, truncfIterators,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    auto truncf = arith::TruncFOp::create(b, l, bf16Type, args[0]);
                    linalg::YieldOp::create(b, l, truncf.getResult());
                }
            );
            reduceInput = truncfGeneric.getResult(0);
            reduceInputType = bf16Type;

            // Erase the original truncf since we've inserted a new one before reduce
            truncfGenericOp->replaceAllUsesWith(ValueRange{truncfGenericOp->getOperand(0)});
            rewriter.eraseOp(truncfGenericOp);
            truncfGenericOp = nullptr;
        }
        else if (!clConv1dTruncateForReduce && truncfGenericOp) {
            // Accurate mode: Keep f32 input to reduce, just remove the original truncf
            // The reduce will process f32 values directly for best accuracy.
            truncfGenericOp->replaceAllUsesWith(ValueRange{truncfGenericOp->getOperand(0)});
            rewriter.eraseOp(truncfGenericOp);
            truncfGenericOp = nullptr;
        }

        // Reduce sum: bf16 input -> f32 output for accurate accumulation.
        // The reduce sum uses fp32 accumulation internally for accuracy.
        Type reduceOutputType = rewriter.getF32Type();
        Value zeroValue = createZeroConstant(rewriter, loc, reduceOutputType);
        auto reduceInit = tensor::EmptyOp::create(rewriter, loc, reducedShape, reduceOutputType);
        Value zeroTensor =
            linalg::FillOp::create(rewriter, loc, ValueRange{zeroValue}, ValueRange{reduceInit})
                .result();

        // Reduce over dimension 4 (Kw - the kernel width dimension)
        auto reduceOp = linalg::ReduceOp::create(
            rewriter, loc, ValueRange{reduceInput}, ValueRange{zeroTensor}, 4,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // Extend bf16 to f32 for accumulation if needed
                Value lhs = args[0];
                Value rhs = args[1];
                if (lhs.getType() != reduceOutputType) {
                    lhs = arith::ExtFOp::create(b, l, reduceOutputType, lhs);
                }
                if (rhs.getType() != reduceOutputType) {
                    rhs = arith::ExtFOp::create(b, l, reduceOutputType, rhs);
                }
                auto sum = arith::AddFOp::create(b, l, lhs, rhs);
                linalg::YieldOp::create(b, l, ValueRange{sum});
            }
        );

        // Collapse the height dimension [2] from [N, F, 1, Ow] to [N, F, Ow]
        SmallVector<int64_t> collapsedShape = {N, F, Ow};
        auto collapsedType = RankedTensorType::get(collapsedShape, reduceOutputType);

        // Collapse dimension 2 (the height=1 dimension) into dimension 3 (width)
        SmallVector<ReassociationIndices> collapseReassoc = {{0}, {1}, {2, 3}};
        Value collapsedResult =
            tensor::CollapseShapeOp::create(
                rewriter, loc, collapsedType, reduceOp.getResults()[0], collapseReassoc
            )
                .getResult();

        // If original output should be bf16, insert truncf after reduce
        Value finalResult = collapsedResult;
        if (needsBF16Output) {
            // Create truncf generic: f32 -> bf16
            auto bf16Type = rewriter.getBF16Type();
            auto truncfShape = collapsedShape;
            auto truncfResultType = RankedTensorType::get(truncfShape, bf16Type);
            auto truncfInit = tensor::EmptyOp::create(rewriter, loc, truncfShape, bf16Type);

            SmallVector<AffineExpr> truncfExprs;
            for (unsigned i = 0; i < 3; i++) {
                truncfExprs.push_back(rewriter.getAffineDimExpr(i));
            }
            auto truncfMap = AffineMap::get(3, 0, truncfExprs, rewriter.getContext());
            SmallVector<utils::IteratorType> truncfIterators(3, utils::IteratorType::parallel);

            auto truncfGeneric = linalg::GenericOp::create(
                rewriter, loc, TypeRange{truncfResultType}, ValueRange{collapsedResult},
                ValueRange{truncfInit}, ArrayRef<AffineMap>{truncfMap, truncfMap}, truncfIterators,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    auto truncf = arith::TruncFOp::create(b, l, bf16Type, args[0]);
                    linalg::YieldOp::create(b, l, truncf.getResult());
                }
            );
            finalResult = truncfGeneric.getResult(0);
        }

        rewriter.replaceOp(convOp, finalResult);

        return success();
    }
};

void populateOptimizeConv1DPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    else if (clConv1dToGenericConv1D) {
        patterns.insert<Conv1DNcwFcwToGenericConv1DPattern>(context);
    }
    else {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }
}

} // namespace mlir::syna::torq
