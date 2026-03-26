// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Analysis/DataFlowFramework.h"
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
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-conv1d-pattern"

namespace mlir::syna::torq {

template <typename LinalgConv>
struct Conv2DToTorqHlConv1DPattern : public OpRewritePattern<LinalgConv> {

  private:
    const Permutation _dataPerm;    // Dim permutation for data transpose
    const Permutation _weightsPerm; // Weights permutation for weight transpose
    const int _shift8b;             // Scale shift for 8-bit integer operations
    const int _shift16b;            // Scale shift for 16-bit integer operations
    const bool _markFuseGroups;     // When true, mark the TI operations, don't convert.

  private:
    Value createConv1DOutput(
        Value output, Value weights, Permutation dataPerm, bool useConv1dWithReduce,
        PatternRewriter &rewriter
    ) const {
        auto outTy = transposeType(output.getType(), dataPerm);
        if (useConv1dWithReduce) {
            auto weightType = cast<RankedTensorType>(weights.getType());
            int64_t filter_len = weightType.getShape()[1];
            outTy = mlir::cast<RankedTensorType>(output.getType());
            llvm::SmallVector<int64_t> outSh(outTy.getShape().begin(), outTy.getShape().end());
            outSh.push_back(filter_len);
            outTy = RankedTensorType::get(outSh, outTy.getElementType());
        }
        return createInitTensor(output, rewriter, outTy);
    }

    bool checkConv1dWithReduce(Value input, Value output) const {
        auto inputType = mlir::cast<RankedTensorType>(input.getType());
        auto outputType = mlir::cast<RankedTensorType>(output.getType());
        auto channels = inputType.getShape()[3];
        auto outputChannels = outputType.getShape()[3];
        // Check if input and output types are compatible with Conv1D + Reduce
        bool compatible = channels > 1 || outputChannels > 1;
        return compatible;
    }

    // template <typename LinalgConv>
    Value preConversionConv1D(
        LinalgConv convOp, Value input, Value weights, Value output, bool useConv1dWithReduce,
        Permutation dataPerm, PatternRewriter &rewriter
    ) const {
        if (useConv1dWithReduce) {
            input = transposeValue(input, dataPerm, input.getLoc(), rewriter);
            // Create type for Conv1D output with an extra dimension at the end.
            // This will be reduced later with linalg.reduce.
        }
        else {
            auto outTy = mlir::cast<RankedTensorType>(output.getType());
            int64_t out_len = outTy.getShape()[2];

            auto weightType = cast<RankedTensorType>(weights.getType());
            int64_t filter_len = weightType.getShape()[1];
            int64_t op_rows = filter_len;
            int64_t op_cols = out_len;

            // Note: op is Conv2DNhwcHwcfOp
            auto inputType = mlir::cast<RankedTensorType>(input.getType());
            int64_t batch = inputType.getShape()[0];
            int64_t channels = inputType.getShape()[3];

            llvm::SmallVector<int64_t> transposedShape = {batch, channels, op_rows, op_cols};
            RankedTensorType transposedType =
                RankedTensorType::get(transposedShape, inputType.getElementType());

            llvm::SmallVector<int64_t> permVals = {1, 0};
            auto permAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), permVals);

            auto transposeReshape = rewriter.create<torq_hl::TransposeReshapeOp>(
                input.getLoc(), transposedType, createInitTensor(convOp, rewriter, transposedType),
                attrValuesAsVec(convOp.getStrides()), weightType.getShape(), permAttr, input
            );
            input = transposeReshape.getOutput();
        }
        return input;
    }

    Value finalizeConv1DConversion(
        Value torqOut, bool useConv1dWithReduce, PatternRewriter &rewriter
    ) const {
        auto loc = torqOut.getLoc();
        auto torqOutType = mlir::cast<RankedTensorType>(torqOut.getType());
        if (useConv1dWithReduce) {
            auto reducedShape = torqOutType.getShape().drop_back();
            auto elemType = torqOutType.getElementType();
            Type reduceElemType = elemType;
            Value reduceInput = torqOut;

            Value zeroValue = createZeroConstant(rewriter, loc, reduceElemType);
            auto cEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedShape, reduceElemType);
            Value zeroTensor =
                rewriter.create<linalg::FillOp>(loc, ValueRange{zeroValue}, ValueRange{cEmpty})
                    .result();
            linalg::ReduceOp reduceOp = rewriter.create<linalg::ReduceOp>(
                loc, ValueRange{reduceInput}, ValueRange{zeroTensor}, 4,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                    );
                }
            );

            torqOut = reduceOp->getResult(0);
        }

        // // Overwrite torqOut with init tensor for debugging
        // torqOut = createInitTensor(convOp, rewriter, cast<RankedTensorType>(torqOut.getType()));
        // // Fill input with 1s for debugging
        // torqOut = rewriter.create<torq_hl::FillOp>(
        //     loc, cast<RankedTensorType>(torqOut.getType()), torqOut,
        //     rewriter.getI32IntegerAttr(/*0x3f800000*//*0x00003f80*/0)
        // ).getOutput();

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);
        return torqOut;
    }

  public:
    using OpRewritePattern<LinalgConv>::OpRewritePattern;

    Conv2DToTorqHlConv1DPattern(
        MLIRContext *context, const Permutation &dataPerm, const Permutation &weightsPerm,
        int shift8b, int shift16b, bool markFuseGroups
    )
        : OpRewritePattern<LinalgConv>(context), _dataPerm(dataPerm), _weightsPerm(weightsPerm),
          _shift8b(shift8b), _shift16b(shift16b), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgConv convOp, PatternRewriter &rewriter) const override {

        if (!isa<linalg::Conv2DNhwcHwcfOp>(convOp) && !isa<linalg::Conv2DNchwFchwOp>(convOp)) {
            return rewriter.notifyMatchFailure(
                convOp, "Only linalg::Conv2DNhwcHwcfOp and linalg::Conv2DNchwChwOp can be "
                        "rewritten as Conv1D"
            );
            return success();
        }
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        // TODO:
        const int _channelDim = isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 3 : 1;

        auto loc = convOp.getLoc();
        constexpr int weightZp = 0;
        constexpr int groups = 1;
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];
        ::mlir::DenseIntElementsAttr stridesAttr = convOp.getStrides();
        auto strideValue = stridesAttr.getValues<int64_t>()[1];

        auto inputType = mlir::cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        auto outputType = cast<RankedTensorType>(output.getType());
        auto outElemType = outputType.getElementType();
        bool isInt = outElemType.isInteger();
        int outChannels = outputType.getShape()[_channelDim];

        // For 1D convolution detection, we need input height (H) to be 1
        // NCHW format: [N, C, H, W], H is at index 2
        // NHWC format: [N, H, W, C], H is at index 1
        const int inputHDim = isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 1 : 2;
        if (!(inputType.getRank() == 4 && inputShape[inputHDim] == 1)) {
            return rewriter.notifyMatchFailure(convOp, "not conv1d");
        }

        // Get dimensions based on layout format
        int64_t batch = inputType.getShape()[0];
        // For NCHW: channels at index 1, out_len (W) at index 3
        // For NHWC: channels at index 3, out_len (H) at index 1
        int64_t channels = inputType.getShape()[isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 3 : 1];
        int64_t out_len = outputType.getShape()[isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 2 : 3];
        int64_t outputChannels =
            outputType.getShape()[isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 3 : 1];

        auto weightType = cast<RankedTensorType>(weights.getType());
        // For FCHW: [F, C, H, W], W is at index 3
        // For HWCF: [H, W, C, F], W is at index 1
        int64_t filter_len = weightType.getShape()[isa<linalg::Conv2DNhwcHwcfOp>(convOp) ? 1 : 3];

        int64_t op_rows = filter_len;
        int64_t op_cols = out_len;

        VectorIntOrFloat bias(outChannels, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannels, _shift8b, _shift16b);
        if (!scInfo && isInt) {
            return failure();
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto torqWeights = transposeValue(weights, _weightsPerm, weights.getLoc(), rewriter);
        setCompileTimeConstAttr(torqWeights.getDefiningOp());

        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        llvm::SmallVector<int64_t> transposedShape = {batch, channels, op_rows, op_cols};

        RankedTensorType transposedType =
            RankedTensorType::get(transposedShape, inputType.getElementType());

        llvm::SmallVector<int64_t> permVals = {1, 0};
        auto permAttr = mlir::DenseI64ArrayAttr::get(rewriter.getContext(), permVals);

        auto torqOutType = transposeType(output.getType(), _dataPerm);

        // Decide whether to use Conv1D with reduction or TransposeReshape + Conv1D
        // The former is completely generic but probably less efficient for single-channel cases
        // The latter is more efficient but only works for single-channel input and outputs.
        bool useConv1dWithReduce = channels > 1 || outputChannels > 1;
        if (useConv1dWithReduce) {

            input = transposeValue(input, _dataPerm, loc, rewriter);
            // Create type for Conv1D output with an extra dimension at the end.
            // This will be reduced later with linalg.reduce.
            llvm::SmallVector<int64_t> torqOutShape(
                torqOutType.getShape().begin(), torqOutType.getShape().end()
            );
            torqOutShape.push_back(filter_len);
            torqOutType = RankedTensorType::get(torqOutShape, torqOutType.getElementType());
        }
        else {
            auto transposeReshape = rewriter.create<torq_hl::TransposeReshapeOp>(
                loc, transposedType, createInitTensor(convOp, rewriter, transposedType),
                attrValuesAsVec(convOp.getStrides()), weightType.getShape(), permAttr, input
            );
            input = transposeReshape.getOutput();
            // Reset stride to 1 for Conv1DOp as the actual stride is handled in TransposeReshape
            strideValue = 1;
        }

        llvm::SmallVector<int64_t> zeroPad(4, 0);
        llvm::SmallVector<int64_t> stride = {strideValue};

        auto torqConv1Op = rewriter.create<torq_hl::Conv1DOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), 0, weightZp,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, zeroPad, stride,
            attrValuesAsVec(convOp.getDilations()), torq_hl::VectorizationModeEnum::None,
            torqWeights, biasScale, input
        );
        Value torqOut = torqConv1Op.getOutput();

        if (useConv1dWithReduce) {
            auto reducedShape = torqOutType.getShape().drop_back();
            auto elemType = torqOutType.getElementType();
            Type reduceElemType = elemType;

            Value zeroValue = createZeroConstant(rewriter, loc, reduceElemType);
            auto cEmpty = rewriter.create<tensor::EmptyOp>(loc, reducedShape, reduceElemType);
            Value zeroTensor =
                rewriter.create<linalg::FillOp>(loc, ValueRange{zeroValue}, ValueRange{cEmpty})
                    .result();
            linalg::ReduceOp reduceOp = rewriter.create<linalg::ReduceOp>(
                loc, ValueRange{torqOut}, ValueRange{zeroTensor}, 4,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                    );
                }
            );

            torqOut = reduceOp->getResult(0);
        }

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);

        return success();
    }
};

/// Lowers linalg::GenericOp (from Conv1D with preserved kernel dim) to torq_hl::Conv1DOp.
///
/// This pattern matches a 5D linalg.generic op produced by Conv1DNcwFcwToGenericConv1DPattern
/// and lowers it directly to torq_hl.conv1d. The 5D structure [N, F, 1, Ow, Kw] is preserved
/// in the output, allowing downstream patterns to handle the kernel reduction.
///
/// Input/Output Shapes (NCHW layout):
///   - Input: [N, C, 1, W] (4D)
///   - Filter: [F, C, 1, Kw] (4D)
///   - Output: [N, F, 1, Ow, Kw] (5D), element type may be bf16 if truncf is fused
///
/// Features:
///   - Extracts stride/dilation from the generic op's affine indexing maps
///   - Folds per-channel bias into the operation
///   - Fuses scale/clamp operations (foldForwardScaleClamp)
///   - Fuses truncf (f32 -> bf16) either from:
///     a) Inside the generic body (legacy pattern), or
///     b) As a separate op following the generic (current pattern with reduce sum)
///   - Supports fuse group marking for backend optimization
///   - Marks filter as compile-time constant
struct LinalgGenericConv1DToTorqHLConv1DPattern : public OpRewritePattern<linalg::GenericOp> {
  private:
    const int _shift8b;  // Scale shift for 8-bit integer operations
    const int _shift16b; // Scale shift for 16-bit integer operations
    const bool _markFuseGroups;

  public:
    LinalgGenericConv1DToTorqHLConv1DPattern(
        MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups
    )
        : OpRewritePattern<linalg::GenericOp>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        auto loc = genericOp.getLoc();

        // Check if already marked as fuse group
        if (_markFuseGroups && isMarkedFuseGroup(genericOp)) {
            return rewriter.notifyMatchFailure(genericOp, "Already marked");
        }

        // Check if this is a 5D generic op (output should be 5D)
        if (genericOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected single result");
        }

        auto resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
        if (resultType.getRank() != 5) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 5D output tensor");
        }

        // Check that we have exactly 2 inputs (input tensor and filter)
        if (genericOp.getInputs().size() != 2) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 2 input operands");
        }

        Value input = genericOp.getInputs()[0];
        Value filter = genericOp.getInputs()[1];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());

        // Expected shapes (NCHW layout):
        // input: [N, C, 1, W] - 4D expanded from [N, C, W]
        // filter: [F, C, 1, Kw] - 4D expanded from [F, C, Kw]
        // output: [N, F, 1, Ow, Kw] - NCHW layout with preserved kernel dim

        if (inputType.getRank() != 4 || filterType.getRank() != 4) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 4D input and filter tensors");
        }

        // Validate input shape: [N, C, 1, W]
        if (inputType.getShape()[2] != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected input shape [N, C, 1, W]");
        }

        // Validate filter shape: [F, C, 1, Kw]
        if (filterType.getShape()[2] != 1) {
            return rewriter.notifyMatchFailure(genericOp, "Expected filter shape [F, C, 1, Kw]");
        }

        auto outShape = resultType.getShape();
        if (outShape.size() != 5 || outShape[2] != 1) {
            return rewriter.notifyMatchFailure(
                genericOp, "Expected output shape [N, F, 1, Ow, Kw]"
            );
        }

        int64_t N = outShape[0];
        int64_t F = outShape[1];
        int64_t Ow = outShape[3];
        int64_t Kw = outShape[4];

        auto elemType = resultType.getElementType();

        // We expect exactly 3 affine maps: 2 inputs + 1 output
        auto maps = genericOp.getIndexingMapsArray();
        if (maps.size() != 3) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 3 indexing maps");
        }

        // Extract stride and dilation from input affine map
        // Input map: (n, f, kh, ow, kw) -> (n, 0, kh, ow * stride + kw * dilation)
        int64_t strideValue = 1;
        int64_t dilationValue = 1;

        auto inputMap = maps[0];
        // The 4th result expr should be: ow * stride + kw * dilation
        if (inputMap.getNumResults() == 4) {
            auto widthExpr = inputMap.getResult(3);
            // Try to extract stride and dilation from the affine expression
            // Expression should be: dim(3) * stride + dim(4) * dilation
            if (auto binOp = llvm::dyn_cast<AffineBinaryOpExpr>(widthExpr)) {
                if (binOp.getKind() == mlir::AffineExprKind::Add) {
                    auto lhs = binOp.getLHS();
                    auto rhs = binOp.getRHS();
                    // Check lhs: ow * stride (ow is dim 3)
                    if (auto mulOp = llvm::dyn_cast<AffineBinaryOpExpr>(lhs)) {
                        if (mulOp.getKind() == mlir::AffineExprKind::Mul) {
                            if (auto constExpr =
                                    llvm::dyn_cast<AffineConstantExpr>(mulOp.getRHS())) {
                                strideValue = constExpr.getValue();
                            }
                        }
                    }
                    // Check rhs: kw * dilation (kw is dim 4)
                    if (auto mulOp = llvm::dyn_cast<AffineBinaryOpExpr>(rhs)) {
                        if (mulOp.getKind() == mlir::AffineExprKind::Mul) {
                            if (auto constExpr =
                                    llvm::dyn_cast<AffineConstantExpr>(mulOp.getRHS())) {
                                dilationValue = constExpr.getValue();
                            }
                        }
                    }
                }
            }
        }

        // Should be all parallel: (parallel, parallel, parallel, parallel, parallel)
        auto iterators = genericOp.getIteratorTypesArray();
        if (iterators.size() != 5) {
            return rewriter.notifyMatchFailure(genericOp, "Expected 5 iterator types");
        }

        // The generic op should perform a multiply operation
        if (genericOp.getBody()->getOperations().size() < 2) {
            return rewriter.notifyMatchFailure(genericOp, "Generic body too small");
        }

        // Find and validate the actual computation operation
        arith::MulFOp mulOp = nullptr;
        for (auto &op : genericOp.getBody()->getOperations()) {
            if (auto mul = llvm::dyn_cast<arith::MulFOp>(&op)) {
                mulOp = mul;
                break;
            }
        }

        if (!mulOp) {
            return rewriter.notifyMatchFailure(genericOp, "Expected arith.mulf operation in body");
        }

        // Get the generic op result
        auto genericResult = genericOp.getResult(0);

        // Determine the output type for torq_hl.conv1d and check for truncf to fuse.
        // Two cases:
        // 1. truncf is inside the generic body (legacy fused pattern) - use bf16 output
        // 2. truncf is a separate linalg.generic following the generic (current pattern) - fuse it
        // here
        Type torqConv1dOutputType = elemType;
        linalg::GenericOp followingTruncfGeneric = nullptr;

        // Case 1: Look for arith.truncf in the generic body
        for (auto &op : genericOp.getBody()->getOperations()) {
            if (auto truncfOp = llvm::dyn_cast<arith::TruncFOp>(&op)) {
                // Found truncf fused in body, use bf16 output type
                torqConv1dOutputType = truncfOp.getResult().getType();
                break;
            }
        }

        // Case 2: Check if there's a truncf generic following the generic (f32 -> bf16)
        // This happens when OptimizeConv1DPattern inserts truncf before reduce sum.
        // The truncf is wrapped in a linalg.generic, not a bare arith.truncf.
        // We can fuse it into torq_hl.conv1d for efficiency.
        if (genericResult.hasOneUse()) {
            auto userOp = genericResult.use_begin()->getOwner();
            if (auto genericUser = llvm::dyn_cast<linalg::GenericOp>(userOp)) {
                // Check if it's a truncf generic: f32 input -> bf16 output
                auto srcType = dyn_cast<RankedTensorType>(genericUser.getInputs()[0].getType());
                auto dstType = dyn_cast<RankedTensorType>(genericUser.getResult(0).getType());
                if (srcType && dstType && srcType.getElementType().isF32() &&
                    dstType.getElementType().isBF16()) {
                    // Verify it contains a truncf operation
                    bool hasTruncf = false;
                    for (auto &op : genericUser.getBody()->getOperations()) {
                        if (llvm::isa<arith::TruncFOp>(&op)) {
                            hasTruncf = true;
                            break;
                        }
                    }
                    if (hasTruncf) {
                        // Found truncf generic following the conv generic, fuse it into conv1d
                        torqConv1dOutputType = dstType.getElementType();
                        followingTruncfGeneric = genericUser;
                    }
                }
            }
        }

        // Create 5D output for torq_hl.conv1d: [N, F, 1, Ow, Kw]
        // This preserves the height dimension structure (NCHW layout)
        SmallVector<int64_t> conv1dOutShape = {N, F, 1, Ow, Kw};
        auto conv1dOutType = RankedTensorType::get(conv1dOutShape, torqConv1dOutputType);
        auto conv1dOutInit =
            rewriter.create<tensor::EmptyOp>(loc, conv1dOutShape, torqConv1dOutputType);

        // Fold forward bias from the output users
        bool isInt = torqConv1dOutputType.isInteger();
        VectorIntOrFloat bias(F, isInt);
        Value outputValue = genericResult;
        while (foldForwardPerChannelAdd(outputValue, 1, bias)) {
        }

        // Fold forward scale and clamp info
        ScaleClampInfo scInfo = foldForwardScaleClamp(outputValue, F, _shift8b, _shift16b);

        // Get weight zero point
        Value filterValue = filter;
        int weightZp = foldForwardWeightZp(filterValue);

        int64_t inputZp = 0;
        int32_t groups = 1;

        SmallVector<int64_t> pad(4, 0);
        SmallVector<int64_t> stride = {strideValue};
        SmallVector<int64_t> dilation = {dilationValue};

        // Create bias tensor
        Value biasValue = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                                : createConst(bias.floats, rewriter, loc);

        // Mark fuse group if requested
        if (_markFuseGroups) {
            markFuseGroupBackward(
                outputValue, {input}, rewriter,
                genericOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Mark filter as compile-time constant if applicable
        if (filter.getDefiningOp()) {
            setCompileTimeConstAttr(filter.getDefiningOp());
        }

        // Create torq_hl.conv1d operation (outputs 5D [N, F, 1, Ow, Kw])
        auto torqConv1dOp = rewriter.create<torq_hl::Conv1DOp>(
            loc, conv1dOutType, conv1dOutInit, inputZp, weightZp, scInfo.zp, scInfo.min, scInfo.max,
            scInfo.scaleShift, groups, pad, stride, dilation, torq_hl::VectorizationModeEnum::None,
            filter, biasValue, input
        );

        // Replace the generic op with torq_hl.conv1d
        rewriter.replaceOp(genericOp, torqConv1dOp.getResults());

        // If we fused a following truncf generic, erase it as well since conv1d now outputs bf16
        if (followingTruncfGeneric) {
            // The truncf generic's users should now use the conv1d result directly
            // Since we already replaced genericOp with conv1d, and truncf was using genericOp,
            // we need to make sure truncf's users are redirected to conv1d's bf16 output
            rewriter.replaceOp(followingTruncfGeneric, torqConv1dOp.getResults());
        }

        return success();
    }
};

void populateLinalgConv2DToTorqHLConv1DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // Conv2D with height=1 can be lowered to Conv1D for efficiency
    patterns.insert<Conv2DToTorqHlConv1DPattern<linalg::Conv2DNhwcHwcfOp>>(
        context, Permutation::nhwc2nchw(), Permutation::hwcf2fchw(), 28, 12, markFuseGroups
    );

    patterns.insert<Conv2DToTorqHlConv1DPattern<linalg::Conv2DNchwFchwOp>>(
        context, Permutation::none(), Permutation::none(), 28, 12, markFuseGroups
    );

    // Generic 5D conv1d pattern (from Conv1DNcwFcwToGenericConv1DPattern)
    patterns.insert<LinalgGenericConv1DToTorqHLConv1DPattern>(context, 28, 12, markFuseGroups);
}

} // namespace mlir::syna::torq
