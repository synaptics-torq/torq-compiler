// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/GenericOp.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include <numeric>

namespace mlir::syna::torq {

namespace {

/*

Match an operation of the form:

  %c-1_i32 = arith.constant -1 : i32

  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                                        iterator_types = ["parallel", "parallel", "parallel",
"parallel"]} ins(%1, %2 : tensor<?xCxi32>, tensor<Cxi32>) outs(%3 : tensor<?xCxi32>) { ^bb0(%in:
i32, %in_3: i32, %out: i32): %12 = arith.muli %in_3, %c-1_i32 : i32 %13 = arith.subi %in, %12 : i32
    linalg.yield %13 : i32
  } -> tensor<1x224x224x32xi32>


To an operation of the form:

%4 = torq.generic bias (%2: tensor<Cxi32>
                   q (%3 : tensor<?xCxi32>)
                   p (%1 : tensor<?xCxi32>)
                   d_map = affine_map<..>
                   b_map = affine_map<..>
                   q_map = affine_map<..>
                   alu_config = { op0 = DBPY , op1 = ACC }
                   act_config = { shiftFactorDiv4 = 0, outputZeroPoint = 0, outputMin = MININT32,
outputMax = MAXINT32}

*/
class AddBiasPattern : public OpConversionPattern<linalg::GenericOp> {

  public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        linalg::GenericOp genericOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // expects one init, the value of q
        if (genericOp.getNumDpsInits() != 1) {
            return failure();
        }

        // expects two inputs, data and weights
        if (genericOp.getNumDpsInputs() != 2) {
            return failure();
        }

        // this is an element-wise operation
        if (genericOp.getNumReductionLoops() > 0) {
            return failure();
        }

        auto initOpOperand = genericOp.getDpsInitOperand(0);
        auto input0OpOperand = genericOp.getDpsInputOperand(0);
        auto input1OpOperand = genericOp.getDpsInputOperand(1);

        auto initOperand = adaptor.getOperands()[initOpOperand->getOperandNumber()];
        auto input0Operand = adaptor.getOperands()[input0OpOperand->getOperandNumber()];
        auto input1Operand = adaptor.getOperands()[input1OpOperand->getOperandNumber()];

        // the init value for q, should have type i32
        if (!isI32Type(initOperand, rewriter)) {
            return failure();
        }

        // the other two inputs, d and b, should have type i32
        if (!isI32Type(input0Operand, rewriter) || !isI32Type(input1Operand, rewriter)) {
            return failure();
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!yieldOp) {
            return failure();
        }

        torq_hl::GenericOpConfig params;

        auto subOp = yieldOp.getOperand(0).getDefiningOp<arith::SubIOp>();

        if (!subOp) {
            return failure();
        }

        auto subLhs = dyn_cast<BlockArgument>(subOp.getLhs());

        if (!subLhs) {
            return failure();
        }

        if (subLhs.getArgNumber() != 0) {
            return failure();
        }

        auto mulOp = subOp.getRhs().getDefiningOp<arith::MulIOp>();

        if (!mulOp) {
            return failure();
        }

        auto mulRhs = mulOp.getRhs().getDefiningOp<arith::ConstantOp>();

        if (mulRhs.getType() != rewriter.getI32Type()) {
            return failure();
        }

        auto constVal = dyn_cast<IntegerAttr>(mulRhs.getValue());

        if (constVal.getInt() != -1) {
            return failure();
        }

        auto mulLhs = dyn_cast<BlockArgument>(mulOp.getLhs());

        if (!mulLhs) {
            return failure();
        }

        if (mulLhs.getArgNumber() != 1) {
            return failure();
        }

        params.p = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(subLhs), adaptor);

        params.q = torq_hl::getParamFromAdaptor(initOpOperand, adaptor);

        params.bias = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(mulLhs), adaptor);

        params.actConfig = torq_hl::ActConfigAttr::get(
            rewriter.getContext(), 0, 0, torq_hl::ACT_MIN, torq_hl::ACT_MAX
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(genericOp.getLoc(), params);

        rewriter.replaceOp(genericOp, newOp.getResult(1));

        return success();
    }
};

/*

Match an operation of the form:

  %30 = "linalg.generic" ins (%1, %2, %3) outs(%4) <{indexing_maps = [affine_map<(d0, d1, d2, d3) ->
(d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>,
                                                              affine_map<(d0, d1, d2, d3) -> (d0,
d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}> ({ ^bb0(%arg1:
i32, %arg2: i32, %arg3: i8, %arg4: i8): %31 = "tosa.apply_scale"(%arg1, %arg2, %arg3) <{double_round
= true}> : (i32, i32, i8) -> i32 %32 = "arith.addi"(%31, %const3) <{overflowFlags =
#arith.overflow<none>}> : (i32, i32) -> i32 %33 = "arith.maxsi"(%32, %const2) : (i32, i32) -> i32
  %34 = "arith.minsi"(%33, %const1) : (i32, i32) -> i32
  %35 = "arith.trunci"(%34) : (i32) -> i8
  "linalg.yield"(%35) : (i8) -> ()
}) : (tensor<1x224x224x32xi32>, tensor<32xi32>, tensor<32xi8>, tensor<1x224x224x32xi8>) ->
tensor<1x224x224x32xi8>

To an operation of the form:

%p_in = tensor.empty() : tensor<?xi32>

%shift_factor = arith.constant dense <28> : i32

%4 = torq.generic scale (%3: tensor<Cxi32>
                   q (%4 : tensor<?xCxi32>)
                   p (%1 : tensor<?xCxi32>)
                   d_map = affine_map<..>
                   b_map = affine_map<..>
                   q_map = affine_map<..>
                   alu_config = { op0 = DBPY , op1 = ACC }
                   act_config = { a0 = 1, a1 = 1, shiftFactorDiv4 = 0, outputZeroPoint = 0,
outputMin = MININT32, outputMax = MAXINT32}

*/
class ApplyScalePattern : public OpConversionPattern<linalg::GenericOp> {

  public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    torq_hl::GenericOpParam createActScale(
        OpAdaptor adaptor, tosa::ApplyScaleOp applyScaleOp, int32_t shiftFactor,
        PatternRewriter &rewriter
    ) const {

        auto loc = applyScaleOp.getLoc();

        auto genericOp = applyScaleOp->getParentOfType<linalg::GenericOp>();

        if (!genericOp) {
            return {};
        }

        auto tosaMultiplier = applyScaleOp.getMultiplier();
        auto tosaShift = applyScaleOp.getShift();

        auto tosaMultiplierArg = dyn_cast<BlockArgument>(tosaMultiplier);
        auto tosaShiftArg = dyn_cast<BlockArgument>(tosaShift);

        if (!tosaMultiplierArg || !tosaShiftArg) {
            return {};
        }

        auto tosaMultiplierOpOperand = genericOp.getMatchingOpOperand(tosaMultiplierArg);
        auto tosaShiftOpOperand = genericOp.getMatchingOpOperand(tosaShiftArg);

        auto tosaMultiplierValue =
            adaptor.getOperands()[tosaMultiplierOpOperand->getOperandNumber()];
        auto tosaShiftValue = adaptor.getOperands()[tosaShiftOpOperand->getOperandNumber()];

        auto tosaMultiplierType = cast<RankedTensorType>(tosaMultiplierValue.getType());
        auto tosaShiftType = cast<RankedTensorType>(tosaShiftValue.getType());

        if (!tosaMultiplierType && !tosaShiftArg) {
            return {};
        }

        if (tosaMultiplierType.getShape() != tosaShiftType.getShape()) {
            return {};
        }

        auto usageMap = genericOp.getMatchingIndexingMap(tosaMultiplierOpOperand);

        // make sure the two inputs are used in the same way so that we can process them
        // with an identity map which is simpler, this could be extended
        if (usageMap != genericOp.getMatchingIndexingMap(tosaShiftOpOperand)) {
            return {};
        }

        auto scaleType =
            RankedTensorType::get(tosaMultiplierType.getShape(), rewriter.getI32Type());

        auto scaleInitOp = rewriter.create<tensor::EmptyOp>(loc, scaleType, ValueRange{});

        size_t rank = tosaMultiplierType.getRank();

        SmallVector<AffineMap> maps{
            3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };

        SmallVector<utils::IteratorType> iteratorTypes{rank, utils::IteratorType::parallel};

        auto regionBuilderFun = [shiftFactor](OpBuilder &builder, Location loc, ValueRange values) {
            auto tosaMultiplier = values[0];
            auto tosaShift = values[1];

            auto shiftFactorConst =
                builder.create<arith::ConstantIntOp>(loc, shiftFactor, builder.getI32Type());

            // floatShiftMask = static_cast<double>(1 << shift);
            auto onei32 = builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(1));
            auto shiftMask = builder.create<arith::ShLIOp>(loc, onei32, shiftFactorConst);
            auto floatShiftMask =
                builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), shiftMask);

            // double floatScaleFactor = static_cast<double>(tosaMultiplier[i]) / (1ul <<
            // tosaShift[i]);
            auto floatMultiplier =
                builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), tosaMultiplier);
            auto onei64 = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(1));
            auto tosaShifti64 =
                builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), tosaShift);
            auto tosaMask = builder.create<arith::ShLIOp>(loc, onei64, tosaShifti64);
            auto tosaFloatMask =
                builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), tosaMask);
            auto floatScaleFactor =
                builder.create<arith::DivFOp>(loc, floatMultiplier, tosaFloatMask);

            // int_scale_factor = floatScaleFactor * floatShiftMask
            auto floatIntMultiplier =
                builder.create<arith::MulFOp>(loc, floatScaleFactor, floatShiftMask);
            auto intScaleFactor =
                builder.create<arith::FPToSIOp>(loc, builder.getI32Type(), floatIntMultiplier);

            builder.create<linalg::YieldOp>(loc, ValueRange{intScaleFactor});
        };

        auto scaleGenericOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{scaleType}, ValueRange{tosaMultiplierValue, tosaShiftValue},
            ValueRange{scaleInitOp}, maps, iteratorTypes,
            /* doc = */ "", /* library_call = */ "", regionBuilderFun
        );

        auto result = scaleGenericOp.getResult(0);

        return {result, usageMap};
    }

    LogicalResult matchAndRewrite(
        linalg::GenericOp genericOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // expects one init, the value of p
        if (genericOp.getNumDpsInits() != 1) {
            return failure();
        }

        // expects three inputs, data and weights
        if (genericOp.getNumDpsInputs() != 3) {
            return failure();
        }

        // this is an element-wise operation
        if (genericOp.getNumReductionLoops() > 0) {
            return failure();
        }

        auto initOpOperand = genericOp.getDpsInitOperand(0);

        auto initOperand = adaptor.getOperands()[initOpOperand->getOperandNumber()];

        // the output is i8
        if (!isI8Type(initOperand, rewriter)) {
            return failure();
        }

        // match the body

        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!yieldOp) {
            return failure();
        }

        auto truncOp = yieldOp.getValues()[0].getDefiningOp<arith::TruncIOp>();

        if (!truncOp)
            return failure();

        auto minOp = truncOp.getIn().getDefiningOp<arith::MinSIOp>();

        if (!minOp)
            return failure();

        // The max constant is used in the min operation
        auto maybeMaxConst = getConstIntValue(minOp.getRhs());

        if (!maybeMaxConst)
            return failure();

        auto maxConst = *maybeMaxConst;

        auto maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();

        if (!maxOp)
            return failure();

        // The min constant is used in the max operation
        auto maybeMinConst = getConstIntValue(maxOp.getRhs());

        if (!maybeMinConst)
            return failure();

        auto minConst = *maybeMinConst;

        auto addOp = maxOp.getLhs().getDefiningOp<arith::AddIOp>();

        if (!addOp)
            return failure();

        auto maybeZeroPoint = getConstIntValue(addOp.getRhs());

        if (!maybeZeroPoint)
            return failure();

        auto zeroPointConst = *maybeZeroPoint;

        auto applyScaleOp = addOp.getLhs().getDefiningOp<tosa::ApplyScaleOp>();

        if (!applyScaleOp)
            return failure();

        auto dataArg = dyn_cast<BlockArgument>(applyScaleOp.getValue());

        if (!dataArg) {
            return failure();
        }

        torq_hl::GenericOpConfig params;

        int shiftFactor = 28;

        params.scale = createActScale(adaptor, applyScaleOp, shiftFactor, rewriter);

        if (!params.scale) {
            return failure();
        }

        params.p = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(dataArg), adaptor);
        params.q = torq_hl::getParamFromAdaptor(initOpOperand, adaptor);

        params.actConfig = torq_hl::ActConfigAttr::get(
            rewriter.getContext(), shiftFactor / 4, zeroPointConst, maxConst, minConst
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(genericOp.getLoc(), params);

        rewriter.replaceOp(genericOp, newOp.getResult(1));

        return success();
    }
};

class LinalgFillOpPattern : public OpConversionPattern<linalg::FillOp> {
  public:
    using OpConversionPattern<linalg::FillOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        linalg::FillOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        if (op.getNumDpsInits() != 1) {
            return failure();
        }

        if (op.getNumDpsInputs() != 1) {
            return failure();
        }

        auto fillVal = adaptor.getInputs()[0].getDefiningOp<arith::ConstantOp>();

        if (!fillVal) {
            return failure();
        }

        auto fillConst = dyn_cast<IntegerAttr>(fillVal.getValue());

        if (!fillConst) {
            return failure();
        }

        int8_t value = fillConst.getValue().getSExtValue();

        torq_hl::GenericOpConfig config;

        auto qOperand = op.getDpsInitOperand(0);

        auto outputType = dyn_cast<ShapedType>(adaptor.getOutputs()[0].getType());

        if (!outputType) {
            return failure();
        }

        auto pType = RankedTensorType::get(outputType.getShape(), rewriter.getI32Type());

        auto p = rewriter.create<arith::ConstantOp>(
            op.getLoc(), SplatElementsAttr::get(pType, rewriter.getI32IntegerAttr(0))
        );

        auto dType = RankedTensorType::get({1}, fillConst.getType());
        auto dElements = DenseElementsAttr::get(dType, {fillConst.getValue()});
        auto dVal = rewriter.create<arith::ConstantOp>(op.getLoc(), dElements);

        auto bType = RankedTensorType::get({1, 2}, rewriter.getI32Type());
        auto bElements = DenseElementsAttr::get(bType, {APInt(32, 1), APInt(32, 0)});
        auto bVal = rewriter.create<arith::ConstantOp>(op.getLoc(), bElements);

        auto outputMap = op.getMatchingIndexingMap(qOperand);
        auto dMap = AffineMap::get(
            outputMap.getNumDims(), 0, getAffineConstantExpr(0, rewriter.getContext())
        );
        auto scaleMap = AffineMap::get(
            outputMap.getNumDims(), 0, getAffineConstantExprs({0, 0}, rewriter.getContext()),
            rewriter.getContext()
        );
        auto biasMap = AffineMap::get(
            outputMap.getNumDims(), 0, getAffineConstantExprs({0, 1}, rewriter.getContext()),
            rewriter.getContext()
        );

        config.q = torq_hl::GenericOpParam(adaptor.getOutputs()[0], outputMap);
        config.p = torq_hl::GenericOpParam(p, outputMap);
        config.d = torq_hl::GenericOpParam(dVal, dMap);
        config.scale = torq_hl::GenericOpParam(bVal, scaleMap);
        config.bias = torq_hl::GenericOpParam(bVal, biasMap);

        config.actConfig = torq_hl::ActConfigAttr::get(
            op.getContext(), 0, (int32_t)value, torq_hl::ACT_MIN, torq_hl::ACT_MAX
        );
        config.aluConfig = torq_hl::AluConfigAttr::get(
            op.getContext(), torq_hl::ALUOp0Mode::DBYP, torq_hl::ALUOp1Mode::ACC
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(op.getLoc(), config);

        rewriter.replaceOp(op, newOp.getResult(1));

        return success();
    }
};

} // namespace

void populateLinalgToActPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<AddBiasPattern>(context);
    patterns.insert<ApplyScalePattern>(context);
    patterns.insert<LinalgFillOpPattern>(context);
}

} // namespace mlir::syna::torq
