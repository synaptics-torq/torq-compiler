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

#include "algorithm"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-conv1d-pattern"

namespace mlir::syna::torq {

// Elementwise f32 -> bf16 truncf as a parallel linalg.generic; rank is taken from `value`.
static Value emitTruncfToBF16(PatternRewriter &rewriter, Location loc, Value value) {
    auto srcType = cast<RankedTensorType>(value.getType());
    int64_t rank = srcType.getRank();
    auto bf16Type = rewriter.getBF16Type();
    auto resultType = RankedTensorType::get(srcType.getShape(), bf16Type);
    Value init = tensor::EmptyOp::create(rewriter, loc, srcType.getShape(), bf16Type).getResult();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{value}, ValueRange{init},
        ArrayRef<AffineMap>{identityMap, identityMap}, iterators,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto truncf = arith::TruncFOp::create(b, l, bf16Type, args[0]);
            linalg::YieldOp::create(b, l, truncf.getResult());
        }
    );
    return generic.getResult(0);
}

// Emit the im2col unfold ([Ow, C*Kw]) as an affine-map linalg.generic marked torq.im2col.
// Used when the NDL im2col op is too large to stay LRAM-resident: MarkHostExecutorPass routes
// the marked generic to the host (torq_hl.im2col is not tileable).
static Value emitIm2Col(
    PatternRewriter &rewriter, Location loc, Value input, int64_t C, int64_t W, int64_t Kw,
    int64_t Ow, int64_t stride, int64_t dilation, RankedTensorType unfoldedType
) {
    auto elemType = unfoldedType.getElementType();

    // Collapse [N,C,W] -> [C,W] so the indexing map is rank-2 with no constant dims (constant
    // dims get folded by canonicalization, causing an operand-rank vs map-result-rank mismatch
    // later).
    auto collapsedInputType = RankedTensorType::get({C, W}, elemType);
    auto collapsedInput = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1}, {2}}
    );
    // if (stride == 1 && dilation == 1) {
    //     return collapsedInput.getResult();
    // }

    // (d0=Ow, d1=C*Kw) -> (d1/Kw, d0*stride + (d1%Kw)*dilation)
    auto unfoldedInit = tensor::EmptyOp::create(rewriter, loc, unfoldedType.getShape(), elemType);

    auto [d0, d1] = std::make_tuple(rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1));
    auto [KwExpr, strideExpr, dilationExpr] = std::make_tuple(
        rewriter.getAffineConstantExpr(Kw), rewriter.getAffineConstantExpr(stride),
        rewriter.getAffineConstantExpr(dilation)
    );

    auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    // (d0=Ow, d1=C*Kw) -> (d1/Kw, d0*stride + (d1%Kw)*dilation)
    auto inputIndexMap = AffineMap::get(
        2, 0, {d1.floorDiv(KwExpr), d1 % KwExpr * dilationExpr + d0 * strideExpr},
        rewriter.getContext()
    );
    SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

    auto im2col = linalg::GenericOp::create(
        rewriter, loc, TypeRange{unfoldedType}, ValueRange{collapsedInput.getResult()},
        ValueRange{unfoldedInit}, ArrayRef<AffineMap>{inputIndexMap, outputIndexMap}, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
            linalg::YieldOp::create(nestedBuilder, nestedLoc, blockArgs[0]);
        }
    );

    return im2col.getResult(0);
}

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

        if (!isAllZerosTensor(output)) {
            return rewriter.notifyMatchFailure(convOp, "init is not all zeros");
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

        if (stride <= 2 && dilation == 1) {
            return rewriter.notifyMatchFailure(convOp, "Converting to Conv2d is better");
        }
        // Steps 1-3: im2col (unfold input patches into [Ow, C*Kw]) + matmul -> [Ow, F].
        //
        // The unfold is torq_hl.im2col (affine NDL window read -> unfolded buffer). The op is not
        // tileable, so when the full unfold does not fit in LRAM we tile im2col + matmul along
        // the output width. Only a pathological conv whose single-column unfold still overflows
        // LRAM falls back to an affine-map generic marked torq.im2col on the host.
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        int64_t W = inputShape[2];

        // Filter: [F, C, Kw] -> [F, C*Kw] -> transpose -> [C*Kw, F].
        auto reshapedFilterType = RankedTensorType::get({F, C * Kw}, filterType.getElementType());
        auto reshapedFilter = tensor::CollapseShapeOp::create(
            rewriter, loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );
        // Transpose filter [F, K] -> [K, F] so the matmul is im2col[Ow,K] x filterT[K,F] -> [Ow,F].
        // The transpose folds into the constant weight at canonicalization time.
        auto transposedFilterInit = tensor::EmptyOp::create(
            rewriter, loc, ArrayRef<int64_t>{C * Kw, F}, filterType.getElementType()
        );
        Value transposedFilter = linalg::TransposeOp::create(
                                     rewriter, loc, reshapedFilter.getResult(),
                                     transposedFilterInit, ArrayRef<int64_t>{1, 0}
        )
                                     .getResults()[0];

        // [Ow, C*Kw] x [C*Kw, F] -> [Ow, F]; accumulate into the output element type (f32).
        auto emitMatmul = [&](Value unfolded) -> Value {
            auto matmulResultType = RankedTensorType::get({Ow, F}, outputElemType);
            auto matmulInit = linalg::FillOp::create(
                rewriter, loc, createZeroConstant(rewriter, loc, outputElemType),
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{Ow, F}, outputElemType)
                    .getResult()
            );
            return linalg::MatmulOp::create(
                       rewriter, loc, TypeRange{matmulResultType},
                       ValueRange{unfolded, transposedFilter}, ValueRange{matmulInit.getResult(0)}
            )
                .getResults()[0];
        };

        auto unfoldedType = RankedTensorType::get({Ow, C * Kw}, elemType);

        Value matmulResult = emitMatmul(
            emitIm2Col(rewriter, loc, input, C, W, Kw, Ow, stride, dilation, unfoldedType)
        );
        // Transpose [Ow, F] -> [F, Ow] so Conv1DMatmulToTorqHlFCPattern detects the transpose
        // and correctly emits is_batch_scaled=false (per-output-channel bias on OC=F).
        auto transposedMatmulInit =
            tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{F, Ow}, outputElemType);
        Value transposedMatmul =
            linalg::TransposeOp::create(
                rewriter, loc, matmulResult, transposedMatmulInit, ArrayRef<int64_t>{1, 0}
            )
                .getResults()[0];

        // Step 4: expand [F, Ow] -> [N, F, Ow]
        if (N == 1) {
            SmallVector<int64_t> expandedShape = {N, F, Ow};
            auto expandedType = RankedTensorType::get(expandedShape, outputElemType);
            auto expandedResult = tensor::ExpandShapeOp::create(
                rewriter, loc, expandedType, transposedMatmul,
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

// True if `v` traces back (through layout/cast ops) to a compile-time constant,
// i.e. a normal conv weight rather than a runtime activation.
//
// A one-level check is NOT enough: by this pass both a constant weight and a
// runtime-activation filter arrive as a linalg.transpose (transpose(const) vs
// transpose(activation)), so we must look THROUGH the value-preserving
// layout/cast ops to the root to tell them apart. The real chain is 1-2 ops
// (a transpose, sometimes an i8->i16 cast generic); kMaxDepth keeps the walk
// short and bounds it so a long/cyclic def chain can never spin.
static bool tracesToConstantWeight(Value v) {
    constexpr int kMaxDepth = 4;
    for (int i = 0; i < kMaxDepth && v; ++i) {
        Operation *def = v.getDefiningOp();
        if (!def)
            return false; // block argument / runtime input -> activation
        if (isa<arith::ConstantOp>(def))
            return true;
        // Look through pure layout/cast ops that just reshape or retype weights.
        // extract_slice appears when a grouped conv splits its constant filter
        // into per-group channel slices before the transpose.
        if (isa<linalg::TransposeOp, tensor::CollapseShapeOp, tensor::ExpandShapeOp,
                tensor::ExtractSliceOp>(def)) {
            v = def->getOperand(0);
            continue;
        }
        if (auto g = dyn_cast<linalg::GenericOp>(def)) {
            if (g.getNumDpsInputs() >= 1) {
                v = g.getDpsInputOperand(0)->get();
                continue;
            }
        }
        return false;
    }
    // Hit the depth bound without resolving -> treat as non-constant (activation)
    // so we don't raise something we couldn't prove is a weight.
    return false;
}

// Raise a 1x1 linalg.conv_2d_nhwc_hwcf (matmul-as-conv) into a linalg.matmul
// before the NHWC->NCHW pass runs, but ONLY when both conv inputs are runtime
// activations (e.g. a TFLite FULLY_CONNECTED with two activation inputs, as in
// attention). For a normal conv with a constant weight the existing conv path
// already handles it correctly, so we leave those untouched.
//
// The NHWC->NCHW pass cannot correctly relayout the quantized epilogue of an
// activation x activation matmul-as-conv; raising to matmul avoids the layout
// boundary and lets the matmul lowering handle the quant params.
//   input  [N,1,1,C]  -> collapse -> [N,C]
//   filter [1,1,C,F]  -> collapse -> [C,F]
//   matmul [N,C]x[C,F] -> [N,F]   -> expand -> [N,1,1,F]
struct Conv2D1x1NhwcHwcfToMatmulPattern : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        Value input = convOp.getInputs()[0];   // [N, H, W, C]
        Value filter = convOp.getInputs()[1];  // [Kh, Kw, C, F]
        Value output = convOp.getOutputs()[0]; // [N, Ho, Wo, F]

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        ArrayRef<int64_t> inputShape = inputType.getShape();
        ArrayRef<int64_t> filterShape = filterType.getShape();

        // Only handle the 1x1 pointwise kernel (matmul-as-conv).
        if (filterShape[0] != 1 || filterShape[1] != 1) {
            return rewriter.notifyMatchFailure(convOp, "not a 1x1 kernel");
        }

        // Only raise the activation x activation case. A constant-weight conv is
        // handled correctly by the existing conv path; raising it regresses the
        // full model (downstream rank mismatch).
        if (tracesToConstantWeight(filter)) {
            return rewriter.notifyMatchFailure(convOp, "constant weight; keep as conv");
        }

        // Unit stride/dilation only — a 1x1 stride-1 conv is a pure matmul.
        SmallVector<int64_t> strides = llvm::to_vector<4>(
            llvm::map_range(convOp.getStrides(), [](APInt v) { return v.getSExtValue(); })
        );
        SmallVector<int64_t> dilations = llvm::to_vector<4>(
            llvm::map_range(convOp.getDilations(), [](APInt v) { return v.getSExtValue(); })
        );
        if (strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1) {
            return rewriter.notifyMatchFailure(convOp, "non-unit stride/dilation");
        }

        // The conv accumulator must start at zero so the matmul reproduces it.
        if (!isAllZerosTensor(output)) {
            return rewriter.notifyMatchFailure(convOp, "conv init is not all zeros");
        }

        int64_t N = inputShape[0];
        int64_t C = inputShape[3];
        int64_t F = filterShape[3];
        auto inElemType = inputType.getElementType();
        auto outElemType = outputType.getElementType();

        // input [N,1,1,C] -> [N,C]
        auto collapsedInputType = RankedTensorType::get({N, C}, inElemType);
        auto collapsedInput = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedInputType, input, ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        // filter [1,1,C,F] -> [C,F] (no transpose: matmul contracts [N,C]x[C,F]).
        auto collapsedFilterType = RankedTensorType::get({C, F}, filterType.getElementType());
        auto collapsedFilter = tensor::CollapseShapeOp::create(
            rewriter, loc, collapsedFilterType, filter,
            ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        // [N,C] x [C,F] -> [N,F], accumulating into the conv output element type.
        SmallVector<int64_t> matmulShape = {N, F};
        auto matmulType = RankedTensorType::get(matmulShape, outElemType);
        auto matmulInit = linalg::FillOp::create(
            rewriter, loc, createZeroConstant(rewriter, loc, outElemType),
            tensor::EmptyOp::create(rewriter, loc, matmulShape, outElemType).getResult()
        );
        auto matmulOp = linalg::MatmulOp::create(
            rewriter, loc, TypeRange{matmulType},
            ValueRange{collapsedInput.getResult(), collapsedFilter.getResult()},
            ValueRange{matmulInit.getResult(0)}
        );

        // [N,F] -> [N,1,1,F]
        auto expandedResult = tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, matmulOp.getResult(0),
            ArrayRef<ReassociationIndices>{{0, 1, 2}, {3}}
        );

        rewriter.replaceOp(convOp, expandedResult.getResult());
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

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        if (stridesAttr.getValues<int64_t>()[0] > 2 || dilationsAttr.getValues<int64_t>()[0] > 1) {
            return rewriter.notifyMatchFailure(convOp, "strides/dilations must be 1D");
        }

        // Add extra dimension (1) to input as W: [N,C,W] -> [N,C,W,1]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], inputType.getShape()[2], 1},
            inputType.getElementType()
        );

        auto expandedInput =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedInputType, input, inputReassoc);

        // Add extra dimension (1) to filter: [F,C,W] -> [F,C,W,1]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], filterType.getShape()[2], 1},
            filterType.getElementType()
        );

        auto expandedFilter =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedFilterType, filter, filterReassoc);

        // Add extra dimension (1) to output: [N,F,W] -> [N,F,W,1]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], outputType.getShape()[2], 1},
            outputType.getElementType()
        );

        auto expandedOutput =
            tensor::ExpandShapeOp::create(rewriter, loc, expandedOutputType, output, outputReassoc);

        // Convert 1D strides/dilations to 2D (add 1 as width dimension)
        SmallVector<int64_t> strides2d = {stridesAttr.getValues<int64_t>()[0]};
        strides2d.push_back(1);
        SmallVector<int64_t> dilations2d = {dilationsAttr.getValues<int64_t>()[0]};
        dilations2d.push_back(1);

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

                    // Re-emit the conv1d's f32->bf16 truncf on the new conv2d result; the
                    // original generic is dropped below once the collapse is wired up.
                    currentResult = emitTruncfToBF16(rewriter, loc, conv2dResult);
                    truncfGenericOp = genericOp;
                }
            }
        }

        // Collapsing the width dimension ([N,F,Ow,1] -> [N,F,Ow])
        // Create a new reassociation indices for collapsing the last dimension
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

/// Sink a static tensor.extract_slice past an elementwise linalg.generic whose every
/// input is an identically-parameterised extract_slice, i.e. rewrite
///   elementwise(slice(a), slice(b), ...) -> slice(elementwise(a, b, ...)).
///
/// A strided ConvTranspose lowers to a conv whose output is a few columns wider than the real
/// result; the epilogue is `conv -> extract_slice(Ow+k -> Ow) -> addf(per-channel bias) -> truncf`.
/// The crop drops the extra ConvTranspose columns and may also absorb a downstream slice the
/// compiler folded in here, so it can shrink the width by more than one element. That slice
/// between the conv result and the bias add stops the bias/truncf from fusing into the conv
/// epilogue (matmul or conv2d), leaving an fp32 add with no NPU lowering that is illegal
/// under --torq-disable-host. Slicing commutes with positional elementwise ops, so moving it
/// below the add/truncf makes the conv->add->truncf chain contiguous again for the downstream
/// epilogue fusion.
struct SinkExtractSliceThroughElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    struct SliceInputsInfo {
        tensor::ExtractSliceOp firstSlice;
        RankedTensorType srcType;
        RankedTensorType resultType;
        SmallVector<int64_t> staticSizes;
        SmallVector<Value> sources;
    };

    LogicalResult
    checkElementwiseSignature(linalg::GenericOp genericOp, PatternRewriter &rewriter) const {
        // Target the ConvTranspose bias epilogue elementwise(slice(a), slice(b), ...): a binary
        // (or wider) op whose operands are all the same crop. A single-input crop is just a
        // rescale/cast/copy on a sliced tensor (very common in conv backbones, e.g. per-channel
        // requantize after a channel split); sinking those does nothing for bias fusion and only
        // risks reshaping unrelated epilogues, so require at least two inputs.
        if (genericOp.getNumResults() != 1 || genericOp.getNumDpsInputs() < 2) {
            return rewriter.notifyMatchFailure(genericOp, "need single result and >=2 inputs");
        }

        // Elementwise: every iterator parallel and every indexing map an identity.
        if (!llvm::all_of(genericOp.getIteratorTypesArray(), [](utils::IteratorType t) {
                return t == utils::IteratorType::parallel;
            })) {
            return rewriter.notifyMatchFailure(genericOp, "not all-parallel");
        }
        for (AffineMap m : genericOp.getIndexingMapsArray()) {
            if (!m.isIdentity()) {
                return rewriter.notifyMatchFailure(genericOp, "non-identity indexing map");
            }
        }
        return success();
    }

    LogicalResult collectSliceInputs(
        linalg::GenericOp genericOp, PatternRewriter &rewriter, SliceInputsInfo &sliceInfo
    ) const {
        // Every input must be an extract_slice with identical, fully-static slice parameters
        // taken from an equally-shaped source: slicing the same region from each operand commutes
        // with the positional elementwise body. Collect the slice sources as we validate.
        auto firstSlice = genericOp.getDpsInputs().front().getDefiningOp<tensor::ExtractSliceOp>();
        if (!firstSlice) {
            return rewriter.notifyMatchFailure(genericOp, "inputs are not extract_slice");
        }
        auto srcType = cast<RankedTensorType>(firstSlice.getSource().getType());
        if (!srcType.hasStaticShape() ||
            llvm::any_of(firstSlice.getStaticOffsets(), ShapedType::isDynamic) ||
            llvm::any_of(firstSlice.getStaticSizes(), ShapedType::isDynamic) ||
            llvm::any_of(firstSlice.getStaticStrides(), ShapedType::isDynamic)) {
            return rewriter.notifyMatchFailure(genericOp, "dynamic slice");
        }

        SmallVector<Value> sources;
        sources.reserve(genericOp.getNumDpsInputs());
        for (Value in : genericOp.getDpsInputs()) {
            auto s = in.getDefiningOp<tensor::ExtractSliceOp>();
            if (!s) {
                return rewriter.notifyMatchFailure(genericOp, "input is not extract_slice");
            }
            if (s.getSource().getType() != srcType ||
                s.getStaticOffsets() != firstSlice.getStaticOffsets() ||
                s.getStaticSizes() != firstSlice.getStaticSizes() ||
                s.getStaticStrides() != firstSlice.getStaticStrides()) {
                return rewriter.notifyMatchFailure(genericOp, "mismatched slice params");
            }
            sources.push_back(s.getSource());
        }

        sliceInfo.firstSlice = firstSlice;
        sliceInfo.srcType = srcType;
        sliceInfo.resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
        sliceInfo.staticSizes.assign(
            firstSlice.getStaticSizes().begin(), firstSlice.getStaticSizes().end()
        );
        sliceInfo.sources = std::move(sources);
        return success();
    }

    LogicalResult checkConvEpilogueSliceShapeAndSources(
        linalg::GenericOp genericOp, PatternRewriter &rewriter, SliceInputsInfo &sliceInfo
    ) const {
        // Only sink genuine convolution epilogues: the crop must keep every source dimension
        // except exactly one (the spatial width), and at least one sliced source must be a
        // convolution result. The dropped-column count is left unbounded because a strided
        // ConvTranspose's extra columns plus any folded-in downstream slice can remove more than
        // one element. Requiring a convolution source keeps this scoped to conv+bias epilogues
        // (where the bias fuses into the conv) and, together with the identical slice-parameter
        // check above, excludes unrelated backbone crops such as channel splits.
        // getStaticSizes() is indexed by source dimension (rank reduction only drops unit result
        // dims), so it lines up with the source shape here.
        int64_t shrunkDims = 0;
        for (auto [sliceSize, srcDim] :
             llvm::zip_equal(sliceInfo.staticSizes, sliceInfo.srcType.getShape())) {
            if (sliceSize != srcDim) {
                ++shrunkDims;
            }
        }
        if (shrunkDims != 1) {
            return rewriter.notifyMatchFailure(
                genericOp, "slice does not crop exactly one dimension"
            );
        }
        if (llvm::none_of(sliceInfo.sources, [](Value src) {
                return src.getDefiningOp<linalg::ConvolutionOpInterface>() != nullptr;
            })) {
            return rewriter.notifyMatchFailure(
                genericOp, "no convolution source to fuse the bias into"
            );
        }
        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        if (failed(checkElementwiseSignature(genericOp, rewriter))) {
            return failure();
        }

        SliceInputsInfo sliceInfo;
        if (failed(collectSliceInputs(genericOp, rewriter, sliceInfo))) {
            return failure();
        }
        if (failed(checkConvEpilogueSliceShapeAndSources(genericOp, rewriter, sliceInfo))) {
            return failure();
        }

        auto loc = genericOp.getLoc();
        Type outElemType = sliceInfo.resultType.getElementType();
        auto newResultType = RankedTensorType::get(sliceInfo.srcType.getShape(), outElemType);

        Value newInit =
            tensor::EmptyOp::create(rewriter, loc, sliceInfo.srcType.getShape(), outElemType)
                .getResult();

        int64_t rank = sliceInfo.srcType.getRank();
        auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
        SmallVector<AffineMap> maps(sliceInfo.sources.size() + 1, identityMap);
        SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

        auto newGeneric = linalg::GenericOp::create(
            rewriter, loc, TypeRange{newResultType}, ValueRange(sliceInfo.sources),
            ValueRange{newInit}, maps, iterators
        );
        rewriter.cloneRegionBefore(
            genericOp.getRegion(), newGeneric.getRegion(), newGeneric.getRegion().end()
        );

        // Re-apply the original (possibly rank-reducing) slice on the elementwise result.
        Value newSlice = tensor::ExtractSliceOp::create(
            rewriter, loc, sliceInfo.resultType, newGeneric.getResult(0),
            sliceInfo.firstSlice.getMixedOffsets(), sliceInfo.firstSlice.getMixedSizes(),
            sliceInfo.firstSlice.getMixedStrides()
        );
        rewriter.replaceOp(genericOp, newSlice);
        return success();
    }
};

void populateOptimizeConv1DPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    // Move the ConvTranspose trailing-column slice below its elementwise bias/truncf epilogue
    // so the bias fuses into the conv.
    patterns.insert<SinkExtractSliceThroughElementwisePattern>(context);

    // Raise 1x1 matmul-as-conv to linalg.matmul before the NHWC->NCHW pass.
    patterns.insert<Conv2D1x1NhwcHwcfToMatmulPattern>(context);

    // Conv1d lowering strategy. Three algorithms were tried:
    //   1. rank-4 conv2d form (height-1 conv2d; stride <= 2, dilation 1): the common
    //      case; carries a conv bias in the init.
    //   2. im2col + matmul (fully-connected form): what the conv2d form cannot express --
    //      stride >= 3 or dilation > 1, with an all-zero init. The im2col unfold is
    //      torq_hl.im2col (NDL window read); when it does not fit LRAM, im2col + matmul
    //      are tiled along the output width, and only a pathological conv whose
    //      single-column unfold still overflows LRAM falls back to a host-marked generic.
    //   3. native 5D linalg.generic with the kernel dim preserved, lowered to
    //      torq_hl.conv1d (see LinalgGenericConv1DToTorqHLConv1DPattern), with Ow-tiling
    //      to fit LRAM and the per-channel bias folded into the following reduce
    //      (FoldConvBiasIntoReducePattern). It matched bf16 data/weights only and was an
    //      NPU-only escape hatch for the convs (2) bails on (e.g. biased convs); gated
    //      behind --torq-convert-conv1d-to-generic since 54c9058d and never enabled by
    //      any test, script or model, so it is removed.
    patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
}

} // namespace mlir::syna::torq
