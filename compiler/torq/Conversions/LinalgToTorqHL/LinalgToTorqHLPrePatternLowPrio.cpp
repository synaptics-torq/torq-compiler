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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-pre-pattern-lowprio"

namespace mlir::syna::torq {

struct EltwiseBinaryConvert : public OpRewritePattern<linalg::GenericOp> {
  public:
    enum OpType {
        ADD_OP,
        SUB_OP,
        // Other operations can be added here if needed
    };
    using LinalgEltOpT = linalg::GenericOp;
    using TorqEltOp = torq_hl::AddOp;
    using OpRewritePattern<LinalgEltOpT>::OpRewritePattern;

    bool foldBackwardScalar(
        Value &input, ScaleInfo &scaleInfo, int32_t *scalarValue, PatternRewriter &rewriter
    ) const {

        auto elOp = input.getDefiningOp<Operation *>();
        bool maybeReplace = false;
        if (isa<tensor::ExpandShapeOp>(elOp) || isa<tensor::CollapseShapeOp>(elOp)) {
            maybeReplace = true;
        }
        if (isa<linalg::GenericOp>(elOp)) {
            auto maybeElemOp = dyn_cast<linalg::GenericOp>(elOp);
            auto &r = maybeElemOp.getRegion();
            if (r.hasOneBlock() && isa<linalg::YieldOp>(r.front().front())) {
                maybeReplace = true;
            }
        }

        // FIXME Replacing expand, collapse or its corresponding linalg.generic op
        // with its input only works when the input is a scalar tensor with dims 1x1x..x1.
        if (maybeReplace) {
            auto elOpInput = elOp->getOperand(0);
            auto elOpInputShape = dyn_cast<RankedTensorType>(elOpInput.getType()).getShape();
            if (mlir::computeProduct(elOpInputShape) == 1) {
                input = elOpInput;

                elOp->getResult(0).replaceAllUsesWith(elOpInput);
                rewriter.eraseOp(elOp);
            }
        }

        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputShape = inputType.getShape();
        auto inputElementType = inputType.getElementType();

        Value scalarInput = input;
        if (inputShape.size() <= 1) {
            while (foldBackwardRescale(scalarInput, scaleInfo)) {
            }

            while (foldScalarRescale(scalarInput, scaleInfo, inputElementType, rewriter)) {
            }
        }

        auto constOp = dyn_cast<arith::ConstantOp>(scalarInput.getDefiningOp());

        if (constOp && getIntegerConstantValue(constOp, scalarValue)) {
            return true;
        }
        return false;
    }

    bool broadcastProcessing(Value &input, ScaleInfo &scaleInfo, PatternRewriter &rewriter) const {
        auto bkpInput = input;
        if (!input.hasOneUse()) {
            return false;
        }
        auto selectInput = [](Value &in, Value subst) {
            if (!in.hasOneUse()) {
                return false;
            }
            in = subst;
            return true;
        };

        linalg::GenericOp bcastOp = input.getDefiningOp<linalg::GenericOp>();
        std::optional<SmallVector<int64_t>> bcastDims;
        if (bcastOp) {
            bcastDims = isaBroadcastOpInterface(bcastOp);
            if (bcastDims) {
                auto success = selectInput(input, bcastOp.getDpsInputOperand(0)->get());
                if (!success) {
                    input = bkpInput;
                    return false;
                }
            }
        }

        auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>();
        if (expandOp) {
            auto success = selectInput(input, expandOp.getSrc());
            if (!success) {
                input = bkpInput;
                return false;
            }
        }

        auto collapseOp = input.getDefiningOp<tensor::CollapseShapeOp>();
        if (collapseOp) {
            auto success = selectInput(input, collapseOp.getSrc());
            if (!success) {
                input = bkpInput;
                return false;
            }
        }

        auto maybeElemOp = input.getDefiningOp<linalg::GenericOp>();
        if (maybeElemOp) {
            auto &r = maybeElemOp.getRegion();
            if (r.hasOneBlock() && isa<linalg::YieldOp>(r.front().front())) {
                auto success = selectInput(input, maybeElemOp.getOperand(0));
                if (!success) {
                    input = bkpInput;
                    return false;
                }
            }
            else {
                maybeElemOp = nullptr;
            }
        }

        if (_markFuseGroups) {
            return true;
        }

        while (foldBackwardRescale(input, scaleInfo)) {
        }

        auto inputElementType = dyn_cast<RankedTensorType>(input.getType()).getElementType();

        // linalg.generic as collapse_shape
        if (maybeElemOp) {
            auto maybeElemOpOutputShape =
                dyn_cast<RankedTensorType>(maybeElemOp.getResult(0).getType()).getShape();

            auto createElOp = [&](PatternRewriter &rewriter, Location loc, Type elementType) {
                auto newOutputType =
                    RankedTensorType::get(maybeElemOpOutputShape, inputElementType);

                auto emptyOp = rewriter.create<tensor::EmptyOp>(
                    maybeElemOp.getLoc(), newOutputType.getShape(), newOutputType.getElementType()
                );

                auto newOp = rewriter.create<linalg::GenericOp>(
                    loc, ArrayRef<Type>{newOutputType}, ValueRange{input}, ValueRange{emptyOp},
                    /*indexingMaps=*/
                    ArrayRef<AffineMap>{
                        maybeElemOp.getIndexingMapsArray().front(),
                        maybeElemOp.getIndexingMapsArray().back()
                    },
                    /*iteratorTypes=*/maybeElemOp.getIteratorTypesArray(),
                    [&](OpBuilder &bBuilder, Location bLoc, ValueRange bArgs) {
                        bBuilder.create<linalg::YieldOp>(bLoc, bArgs[0]);
                    }
                );

                return newOp;
            };

            auto newOp = createElOp(rewriter, maybeElemOp.getLoc(), inputElementType);

            maybeElemOp.getResult(0).replaceAllUsesWith(newOp->getResult(0));
            rewriter.eraseOp(maybeElemOp);
            input = newOp->getResult(0);
        }

        // tensor.collapse_shape
        if (collapseOp) {
            auto collapseShape = collapseOp.getResultType().getShape();
            auto newOutputType = RankedTensorType::get(collapseShape, inputElementType);

            auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
                collapseOp.getLoc(), newOutputType, input, collapseOp.getReassociationIndices()
            );
            collapseOp->getResult(0).replaceAllUsesWith(newCollapseOp.getResult());
            rewriter.eraseOp(collapseOp);
            input = newCollapseOp.getResult();
        }

        // tensor.expand_shape
        if (expandOp) {
            auto expandShape = expandOp.getResultType().getShape();
            auto newOutputType = RankedTensorType::get(expandShape, inputElementType);

            auto newExpandOp = rewriter.create<tensor::ExpandShapeOp>(
                expandOp.getLoc(), newOutputType, input, expandOp.getReassociationIndices()
            );
            expandOp->getResult(0).replaceAllUsesWith(newExpandOp.getResult());
            rewriter.eraseOp(expandOp);
            input = newExpandOp.getResult();
        }

        if (bcastDims) {

            auto dstTy = bcastOp.getDpsInitOperand(0)->get().getType();
            auto src = bcastOp.getDpsInputOperand(0)->get();
            auto bOutputShape = mlir::cast<RankedTensorType>(dstTy).getShape();
            auto bOutputType = RankedTensorType::get(bOutputShape, inputElementType);

            auto op = rewriter.create<linalg::BroadcastOp>(
                bcastOp.getLoc(), src, createInitTensor(bcastOp, rewriter, bOutputType), *bcastDims
            );
            auto gOp = linalg::generalizeNamedOp(rewriter, op);
            rewriter.replaceOp(bcastOp, gOp->getResults()[0]);

            input = gOp->getResults()[0];
        }

        return true;
    }

    EltwiseBinaryConvert(
        MLIRContext *context, int shift8b, int shift16b, OpType opType, bool markFuseGroups
    )
        : OpRewritePattern<LinalgEltOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _opType(opType), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgEltOpT eltOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(eltOp)) {
            return rewriter.notifyMatchFailure(eltOp, "Already marked");
        }

        Operation *binaryOp = getElementwiseBinaryOp(eltOp);
        if (!binaryOp) {
            return rewriter.notifyMatchFailure(eltOp, "Not an elementwise binary op");
        }

        // Get the inputs and output of the original operation
        Value input0 = eltOp.getInputs()[0];
        // Binary ops can actually have only one input if both operands are the same
        Value input1 = eltOp.getInputs()[eltOp.getInputs().size() > 1 ? 1 : 0];
        Value output = eltOp.getResultTensors()[0];
        const auto outType = cast<RankedTensorType>(output.getType());

        const auto loc = eltOp.getLoc();

        // TODO:
        // 1. handle arith.AddFOp and floating point operations
        // 2. handle two inputs are identical, rescale pattern is different

        int sign = 1;
        auto opName = "add"; // Default operation name is "add" (for decoration purposes)
        switch (_opType) {
        case ADD_OP:
            if (clACTBasedAdd) {
                return rewriter.notifyMatchFailure(
                    eltOp, "ACT-based add enabled; ALU-based AddOp not expected"
                );
            }
            if (!dyn_cast<arith::AddIOp>(binaryOp)) {
                return rewriter.notifyMatchFailure(eltOp, "Expected arith.addi");
            }
            break;
        case SUB_OP:
            if (clACTBasedSub) {
                return rewriter.notifyMatchFailure(
                    eltOp, "ACT-based sub enabled; ALU-based SubOp not expected"
                );
            }
            if (!dyn_cast<arith::SubIOp>(binaryOp)) {
                return rewriter.notifyMatchFailure(eltOp, "Expected arith.subi");
            }
            sign = -1; // For subtraction, we need to negate the second operand
            opName = "sub";
            break;
        }

        // For rank4 convert in & out to NCHW so that the transposes can be folded with those of the
        // nearby ops which also work in NCHW. This should be replaced with a pass that is able to
        // fold transposes before and after any elementwise op.
        auto dataPerm = outType.getRank() == 4 ? Permutation::nhwc2nchw() : Permutation::none();

        bool isInt = outType.getElementType().isInteger();

        // check if output has rescaleClamp firstly to quik return if it is not present
        const int outChannelCount = 1; // No channels here, only one single scale
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, true);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(eltOp, "Cannot fold forward scale/clamp");
        }

        // Fold inputs rescale if present
        ScaleInfo scaleInput0;
        while (foldBackwardRescale(input0, scaleInput0)) {
        }
        ScaleInfo scaleInput1;
        while (foldBackwardRescale(input1, scaleInput1)) {
        }

        broadcastProcessing(input0, scaleInput0, rewriter);
        broadcastProcessing(input1, scaleInput1, rewriter);

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input0, input1}, rewriter,
                eltOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Compute scale and bias vectors
        const double outputScale = scInfo.scaleDouble[0];
        double multiplier0 = outputScale * scaleInput0.scale;
        double multiplier1 = outputScale * scaleInput1.scale;
        int scaleFactor = 1 << scInfo.scaleShift;

        auto weight0 = doubleToInt<int16_t>(multiplier0 * scaleFactor);
        auto bias0 = -doubleToInt<int32_t>(multiplier0 * scaleFactor * scaleInput0.zp);
        int16_t weight1 = doubleToInt<int16_t>(multiplier1 * scaleFactor) * sign;
        int32_t bias1 = -doubleToInt<int32_t>(multiplier1 * scaleFactor * scaleInput1.zp) * sign;

        int32_t scalarValue0 = 0;
        bool input0IsScalar = foldBackwardScalar(input0, scaleInput0, &scalarValue0, rewriter);

        int32_t scalarValue1 = 0;
        bool input1IsScalar = foldBackwardScalar(input1, scaleInput1, &scalarValue1, rewriter);

        if (input0IsScalar && input1IsScalar) {
            return rewriter.notifyMatchFailure(eltOp, "don't support both input is scalar for now");
        }

        if (input0IsScalar) {
            double scaleFactor = (1 << scInfo.scaleShift) + 0.5;
            weight0 = 0;
            bias0 = scalarValue0 * scaleFactor * sign * outputScale;

            input0 = input1;
        }
        else if (input1IsScalar) {

            scaleInput1.scale = 0;
            scaleInput1.zp = 0;

            // Compute scale and bias vectors
            double scaleFactor = (1 << scInfo.scaleShift) + 0.5;

            multiplier1 = outputScale * scaleInput1.scale;
            weight1 = doubleToInt<int16_t>(multiplier1 * scaleFactor) * sign;

            // should not use negative bias here as it compute from scale
            // if later issue happens, please check if need to add minus sign here
            bias1 = scalarValue1 * scaleFactor * sign * outputScale;

            // force weigth1 is 0, input0 * weight0 + input1 * 0
            input1 = input0;
        }

        // Generate torq_hl op with input in the expected format
        input0 = transposeValue(input0, dataPerm, loc, rewriter);
        input1 = transposeValue(input1, dataPerm, loc, rewriter);

        // concatenate weight0 and weight1
        std::vector<int16_t> weights = {weight0, weight1};
        auto torqWeights = createConst(weights, rewriter, loc);

        VectorIntOrFloat bias(1, isInt);
        bias.ints[0] = bias0 + bias1; // TODO: handle the case of floating point operation

        // Scale is always 1
        const std::vector<int32_t> scale = {1}; // TODO: handle the case of floating point operation

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scale), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with output in the expected format
        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = rewriter.create<TorqEltOp>(
            loc, torqOutType, createInitTensor(eltOp, rewriter, torqOutType), opName,
            /* input zp not needed */ 0, scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            torqWeights, biasScale, input0, input1
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const OpType _opType;       // Type of the operation (ADD_OP, SUB_OP, etc.)
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

struct PoolingConvert : public OpRewritePattern<linalg::PoolingNhwcSumOp> {
  public:
    using LinalgOpT = linalg::PoolingNhwcSumOp;
    using TorqOp = torq_hl::AvgPool2DOp;
    using OpRewritePattern<LinalgOpT>::OpRewritePattern;

    PoolingConvert(MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups)
        : OpRewritePattern<LinalgOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgOpT linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        // Get the inputs and output of the original operation
        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();
        if (kernelShape.size() != 2 || kernelShape[0] != in_s[1] || kernelShape[1] != in_s[2]) {
            return rewriter.notifyMatchFailure(linalgOp, "Only kernel == whole frame supported");
        }
        int itemsPerChannel = in_s[1] * in_s[2];

        auto dataPerm = Permutation::none();

        bool isInt = outType.getElementType().isInteger();
        if (!isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Only integer pooling supported");
        }

        const int channelDimension = 3;
        int32_t channelCount = in_s[channelDimension];

        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, channelCount, _shift8b, _shift16b, true);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Cannot fold forward scale/clamp");
        }

        // Don't fold input rescale here. Input zp is applied after pooling in the ForwardScale
        ScaleInfo scaleInput0;

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input0, kernel}, rewriter,
                linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Generate torq_hl op with input in the expected format
        input0 = transposeValue(input0, dataPerm, loc, rewriter);

        // Compute scale and bias vectors
        const std::vector<int32_t> scale(
            channelCount, int32_t(scaleInput0.scale * (1 << scInfo.scaleShift) / itemsPerChannel)
        );
        const std::vector<int32_t> bias(channelCount, -scInfo.bias);

        // Prepare bias (and scale for integer ops)
        auto biasScale = createConst(interleave(bias, scale), rewriter, loc);

        // Prepare weights
        auto weights = createI8Const(
            rewriter, linalgOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        // Generate torq_hl op with output in the expected format
        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = rewriter.create<TorqOp>(
            linalgOp.getLoc(), torqOutType, createInitTensor(linalgOp, rewriter, torqOutType),
            scInfo.zp, scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, weights, biasScale,
            input0
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

struct ReduceMeanPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp.getNumDpsInputs() != 1 || srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "ReducemeanPattern Expected 1 input/init");
        }

        auto inType = dyn_cast_or_null<RankedTensorType>(srcOp.getInputs()[0].getType());
        auto output = srcOp.getResult(0);
        auto outType = dyn_cast_or_null<RankedTensorType>(output.getType());

        if (!inType || !outType || !inType.getElementType().isBF16() ||
            !outType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(
                srcOp, "ReducemeanPattern only bf16 supported for now"
            );
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected yield");
        }

        auto divFOp = dyn_cast_or_null<arith::DivFOp>(yieldOp.getValues()[0].getDefiningOp());
        if (!divFOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected div op");
        }

        if (!isa<BlockArgument>(divFOp.getLhs())) {
            return rewriter.notifyMatchFailure(srcOp, "Div lhs must be block arg");
        }

        auto divRhs = divFOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!divRhs) {
            return rewriter.notifyMatchFailure(srcOp, "Div rhs must be constant");
        }

        auto divConstAttr = dyn_cast_or_null<FloatAttr>(divRhs.getValue());
        if (!divConstAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Div constant must be float");
        }

        double divConst = divConstAttr.getValueAsDouble();
        if (divConst <= 0.0) {
            return rewriter.notifyMatchFailure(srcOp, "Div constant must be positive");
        }

        // if output is used by CollapseShape, fold collapseShape op
        if (output.hasOneUse() && (isa<tensor::CollapseShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        auto reducesumOp = srcOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!reducesumOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected sum reduction");
        }

        auto reducesumYield = dyn_cast<linalg::YieldOp>(reducesumOp.getBody()->getTerminator());
        if (!reducesumYield || !isa<arith::AddFOp>(reducesumYield.getValues()[0].getDefiningOp())) {
            return rewriter.notifyMatchFailure(
                reducesumOp, "ReducesumPattern Expected AddFOp reduction"
            );
        }

        SmallVector<unsigned> reductionDims;
        reducesumOp.getReductionDims(reductionDims);
        if (reductionDims.size() < 1) {
            return rewriter.notifyMatchFailure(
                reducesumOp, "ReducesumPattern expected reduction loop > 0"
            );
        }

        SmallVector<unsigned> parallelDims;
        reducesumOp.getParallelDims(parallelDims);

        Value input = reducesumOp.getInputs()[0];

        // reduceMean has batch and its iteratetype is parallel
        SmallVector<uint64_t, 4> permVec;
        permVec.push_back(0);
        permVec.append(reductionDims.begin(), reductionDims.end());
        for (int i = 1; i < parallelDims.size(); i++) {
            permVec.push_back(parallelDims[i]);
        }

        // avgpool kernel request nhwc
        auto loc = srcOp.getLoc();
        Permutation dataPerm(permVec.begin(), permVec.end());
        input = transposeValue(input, dataPerm, loc, rewriter);

        // Scale = 1 / meanConst for mean calculation
        float meanValue = 1.0f / divConst;
        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weightsData(1, llvm::APFloat(bf16, std::to_string(meanValue)));
        auto weights = createConst(weightsData, rewriter, srcOp.getLoc());

        std::vector<float> biasScaleData{0.0};
        auto biasScale = createConst(biasScaleData, rewriter, srcOp.getLoc());

        auto avgpoolOutType = dyn_cast_or_null<RankedTensorType>(output.getType());

        rewriter.replaceOpWithNewOp<torq_hl::AvgPool2DOp>(
            srcOp, avgpoolOutType, createInitTensor(reducesumOp, rewriter, avgpoolOutType), 0, 0,
            0xff800000, 0x7f800000, 0, weights, biasScale, input
        );

        return success();
    }
};

// ReduceMeanConvert: detects a pattern of sum-reduction along the last axis followed by
// division by the reduced axis size, and lowers it to torq_hl::ReduceMeanOp.
// Example matched IR (bf16 shown but pattern supports bf16/f32):
//   %sum = linalg.generic {iterator_types=["parallel","reduction"]}
//            ins(%x: tensor<NxMxt>) outs(%init: tensor<Nxt>) { %r = arith.addf %in, %out; yield %r
//            }
//   %mean = linalg.generic {iterator_types=["parallel"]}
//            ins(%sum: tensor<Nxt>) outs(%out: tensor<Nxt>) { %q = arith.divf %in, %cst_M; yield %q
//            }
struct ReduceMeanConvert : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp meanOp, PatternRewriter &rewriter) const override {
        if (meanOp.getNumDpsInputs() != 1 || meanOp.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(meanOp, "Expected 1 input/init");

        auto iters = meanOp.getIteratorTypesArray();
        if (iters.empty() ||
            llvm::any_of(iters, [](auto t) { return t != mlir::utils::IteratorType::parallel; }))
            return rewriter.notifyMatchFailure(meanOp, "Expected parallel iterators");

        auto yieldOp = dyn_cast<linalg::YieldOp>(meanOp.getBody()->getTerminator());
        if (!yieldOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected yield");

        Operation *divOp = yieldOp.getValues()[0].getDefiningOp();
        if (!divOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected div op");

        // Only bf16 (DivFOp) is supported
        auto divFOp = dyn_cast<arith::DivFOp>(divOp);
        if (!divFOp)
            return rewriter.notifyMatchFailure(meanOp, "Only bf16 (DivFOp) supported");

        Value divLhs = divFOp.getLhs();
        if (!isa<BlockArgument>(divLhs))
            return rewriter.notifyMatchFailure(meanOp, "Div lhs must be block arg");

        auto divRhs = divFOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!divRhs)
            return rewriter.notifyMatchFailure(meanOp, "Div rhs must be constant");

        auto divConstAttr = dyn_cast<FloatAttr>(divRhs.getValue());
        if (!divConstAttr)
            return rewriter.notifyMatchFailure(meanOp, "Div constant must be float");

        double divConst = divConstAttr.getValueAsDouble();
        if (divConst <= 0.0)
            return rewriter.notifyMatchFailure(meanOp, "Div constant must be positive");

        auto sumOp = meanOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!sumOp)
            return rewriter.notifyMatchFailure(meanOp, "Expected sum reduction");

        auto sumIters = sumOp.getIteratorTypesArray();
        if (llvm::count_if(sumIters, [](auto t) {
                return t == mlir::utils::IteratorType::reduction;
            }) != 1)
            return rewriter.notifyMatchFailure(sumOp, "Expected one reduction");

        int reductionAxis = 0;
        for (size_t i = 0; i < sumIters.size(); ++i)
            if (sumIters[i] == mlir::utils::IteratorType::reduction)
                reductionAxis = i;

        auto sumYield = dyn_cast<linalg::YieldOp>(sumOp.getBody()->getTerminator());
        if (!sumYield || !isa<arith::AddFOp>(sumYield.getValues()[0].getDefiningOp()))
            return rewriter.notifyMatchFailure(sumOp, "Expected AddFOp reduction");

        auto loc = meanOp.getLoc();
        Value input = sumOp.getInputs()[0];
        auto inputType = cast<RankedTensorType>(input.getType());
        auto resultType = cast<RankedTensorType>(meanOp.getResult(0).getType());
        int64_t reducedDimSize = static_cast<int64_t>(divConst);

        // Reshape to NCHW rank-4
        Value inputNCHW = input;
        int rank = inputType.getRank();
        if (rank == 2) {
            auto s = inputType.getShape();
            auto nchwType = RankedTensorType::get({1, 1, s[0], s[1]}, inputType.getElementType());
            inputNCHW = rewriter.create<tensor::ExpandShapeOp>(
                loc, nchwType, input, SmallVector<ReassociationIndices>{{0, 1, 2}, {3}}
            );
        }
        else if (rank == 3) {
            auto s = inputType.getShape();
            auto nchwType =
                RankedTensorType::get({s[0], 1, s[1], s[2]}, inputType.getElementType());
            inputNCHW = rewriter.create<tensor::ExpandShapeOp>(
                loc, nchwType, input, SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}}
            );
        }
        else if (rank != 4)
            return rewriter.notifyMatchFailure(meanOp, "Only rank 2-4 supported");

        // Map reduction axis to NCHW and apply H↔W transpose if reducing W (axis=3)
        int nchwAxis = (rank == 2)   ? (reductionAxis == 1 ? 3 : 2)
                       : (rank == 3) ? (reductionAxis == 2 ? 3 : 2)
                                     : reductionAxis;

        Value reduceMeanInput = inputNCHW;
        bool needsPreTranspose = (nchwAxis == 3);
        if (needsPreTranspose)
            reduceMeanInput = transposeValue(inputNCHW, Permutation({0, 1, 3, 2}), loc, rewriter);

        // Scale = 1/reducedDimSize for mean calculation (e.g., 1/288 = 0.003472)
        float scaleValue = 1.0f / static_cast<float>(reducedDimSize);
        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weightsData(1, llvm::APFloat(bf16, std::to_string(scaleValue)));
        auto weights = createConst(weightsData, rewriter, loc);

        // Create f32 bias tensor
        std::vector<float> biasScaleData{0.0};
        auto biasScale = createConst(biasScaleData, rewriter, loc);

        auto inputShape = cast<RankedTensorType>(reduceMeanInput.getType()).getShape();
        SmallVector<int64_t> reduceMeanOutShape = {inputShape[0], inputShape[1], 1, inputShape[3]};
        auto reduceMeanOutType =
            RankedTensorType::get(reduceMeanOutShape, resultType.getElementType());

        // For bf16: use full f32 range (no clipping)
        float min_f = std::numeric_limits<float>::lowest();
        int32_t output_min = *reinterpret_cast<int32_t *>(&min_f);
        float max_f = std::numeric_limits<float>::max();
        int32_t output_max = *reinterpret_cast<int32_t *>(&max_f);

        auto reduceMeanOp = rewriter.create<torq_hl::ReduceMeanOp>(
            loc, reduceMeanOutType, createInitTensor(meanOp, rewriter, reduceMeanOutType), 0, 0,
            output_min, output_max,                // input_zp=0, output_zp=0, min, max
            0, weights, biasScale, reduceMeanInput // shift_factor=0 for bf16
        );

        // Undo H↔W transpose if it was applied (restore original axis positions)
        Value out = reduceMeanOp.getOutput();
        if (needsPreTranspose)
            out = transposeValue(out, Permutation({0, 1, 3, 2}), loc, rewriter);
        if (rank == 2)
            out = rewriter.create<tensor::CollapseShapeOp>(
                loc, resultType, out, SmallVector<ReassociationIndices>{{0, 1, 2, 3}}
            );
        else if (rank == 3)
            out = rewriter.create<tensor::CollapseShapeOp>(
                loc, resultType, out, SmallVector<ReassociationIndices>{{0}, {1, 2, 3}}
            );

        rewriter.replaceOp(meanOp, out);
        return success();
    }
};

Value getSpaceToDepth(Value input, int sh, int sw, PatternRewriter &rewriter) {
    auto loc = input.getLoc();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto shape = inputType.getShape();
    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];

    // Expand [n, c, h, w] -> [n, c, h/sh, sh, w/sw, sw]
    // Map: [0]->[0], [1]->[1], [2]->[2,3], [3]->[4,5]
    SmallVector<int64_t, 6> expandedShape = {n, c, h / sh, sh, w / sw, sw};
    auto expandedType = RankedTensorType::get(expandedShape, elementType);
    Value expanded = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, input, ArrayRef<ReassociationIndices>{{0}, {1}, {2, 3}, {4, 5}}
    );

    // Transpose to [n, c, sh, sw, h/sh, w/sw]
    const SmallVector<int64_t, 6> perm = {0, 1, 3, 5, 2, 4};
    Value transposed = transposeValue(expanded, perm, input.getLoc(), rewriter);

    // Collapse [n, c, sh, sw, h/sh, w/sw] to [n, c*sh*sw, h/sh, w/sw]
    auto outType = RankedTensorType::get({n, c * sh * sw, h / sh, w / sw}, elementType);
    Value out = rewriter.create<tensor::CollapseShapeOp>(
        loc, outType, transposed, ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}, {4}, {5}}
    );

    return out;
}

struct Conv2DOpBigStride : public OpRewritePattern<syna::torq_hl::Conv2DOp> {
  private:
    typedef syna::torq_hl::Conv2DOp ConvOp;

    const bool _markFuseGroups;

  public:
    using OpRewritePattern<ConvOp>::OpRewritePattern;
    Conv2DOpBigStride(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<ConvOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(op)) {
            return rewriter.notifyMatchFailure(op, "Already marked");
        }

        const auto loc = op.getLoc();
        auto strides = op.getStride();
        Value weights = op.getWeights();
        arith::ConstantOp constOp = weights.getDefiningOp<arith::ConstantOp>();
        auto convWeights = cast<DenseElementsAttr>(constOp.getValue());
        auto weightType = llvm::cast<RankedTensorType>(convWeights.getType());
        auto weightShape = weightType.getShape().vec();
        bool supportedStrides = strides.size() == 2 && ((strides[0] == 1 && strides[1] == 1) ||
                                                        (strides[0] == 2 && strides[1] == 2));
        if (supportedStrides) {
            // We can handle these strides directly in our HW kernel, nothing to change
            return failure();
        }

        bool kernelEqStride = strides.size() == 2 && weightShape.size() == 4 &&
                              weightShape[2] == strides[0] && weightShape[3] == strides[1];
        if (!kernelEqStride) {
            // The shape of the conv kernel is different from the strides, we can't do anything
            return rewriter.notifyMatchFailure(op, "Kernel shape different from strides");
        }

        // This is a 2D conv with strides [h,w] with kern size == strides
        // Rewrite it as (space2depth + conv2d(stride1) with weights [O,h*w*I,H/sh,W/sw])
        auto weightElementType = weightType.getElementType();
        auto sh = strides[0];
        auto sw = strides[1];

        if (_markFuseGroups) {
            markOpFuseGroup(
                op, rewriter, op->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Convert input from [N, C, H, W] to [N, C*h*w, H/sh, W/sw]
        Value inToDepth = getSpaceToDepth(op.getInput(), sh, sw, rewriter);

        // Reshape weights from [OIHW] to [O,I*h*w,H/sh,W/sw]
        auto newWeightType = RankedTensorType::get(
            {weightShape[0], weightShape[1] * sh * sw, weightShape[2] / sh, weightShape[2] / sw},
            weightElementType
        );
        auto newWeightAttr = DenseElementsAttr::get(newWeightType, convWeights.getRawData());
        auto torqWeights = rewriter.create<arith::ConstantOp>(loc, newWeightType, newWeightAttr);

        // Update conv2d
        rewriter.modifyOpInPlace(op, [&]() {
            op.setOperand(op.getInputMutable().getOperandNumber(), inToDepth);
            op.setOperand(op.getWeightsMutable().getOperandNumber(), torqWeights);
            op.setStride({1, 1});
        });

        return success();
    }
};

void populateLinalgToTorqHLPrePatternsLowPrio(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // These patterns are applied after the high-priority patterns
    // This allows to apply the most beneficial patterns first (eg conv2d with bias addition)
    // before the remaining patterns (eg addition)
    // Note: using benefit to control the order of application is not enough since this
    // only works for patterns that are applied to the same op

    int sh8b = 12;
    int sh16b = 12; // FIXME 16b shift?
    patterns.insert<EltwiseBinaryConvert>(
        context, sh8b, sh16b, EltwiseBinaryConvert::ADD_OP, markFuseGroups
    );
    patterns.insert<EltwiseBinaryConvert>(
        context, sh8b, sh16b, EltwiseBinaryConvert::SUB_OP, markFuseGroups
    );
    patterns.insert<PoolingConvert>(context, 20, 20 /* FIXME */, markFuseGroups);

    patterns.insert<ReduceMeanConvert>(context);

    // TODO: refactor with ReduceMeanConvert later soon
    patterns.insert<ReduceMeanPattern>(context);

    patterns.insert<Conv2DOpBigStride>(context, markFuseGroups);
}

} // namespace mlir::syna::torq