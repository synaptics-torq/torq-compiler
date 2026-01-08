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

#define DEBUG_TYPE "linalg-torq-elementwsie-binary-pattern"

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

void populateLinalgToTorqHLEWBinaryPatterns(
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
}

} // namespace mlir::syna::torq