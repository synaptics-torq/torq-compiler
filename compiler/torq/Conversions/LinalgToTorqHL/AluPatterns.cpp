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

  %4 = linalg.generic { indexing_maps = [affine_map<..>,
                                         affine_map<..>,
                                         affine_map<..>],
                        iterator_types = [ ... ]}
    ins(%1, %2 : tensor<?xi8>, tensor<?xi8>) outs(%3 : tensor<?xi32>) {
  ^bb0(%in: i8, %in_3: i8, %out: i32):
    %12 = arith.extsi %in : i8 to i32
    %13 = arith.extsi %in_3 : i8 to i32
    %14 = arith.muli %12, %13 : i32
    %15 = arith.addi %out, %14 : i32
    linalg.yield %15 : i32
  } -> tensor<?xi32>

To an operation of the form:

%p_in = tensor.empty() : tensor<?xi32>

%4 = torq.generic d (%1 : tensor<?xi8>)
                   w (%2: tensor<?xi8>
                   q (%3 : tensor<?xi32>)
                   p (%p_in : tensor<?xi32>)
                   d_map = affine_map<..>
                   w_map = affine_map<..>
                   q_map = affine_map<..>
                   alu_config = { op0 = MUL , op1 = ACC }

*/
class AddMulPattern : public OpConversionPattern<linalg::GenericOp> {

  public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        linalg::GenericOp genericOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // expects one init, the value of p
        if (genericOp.getNumDpsInits() != 1) {
            return failure();
        }

        // expects two inputs, data and weights
        if (genericOp.getNumDpsInputs() != 2) {
            return failure();
        }

        auto initOpOperand = genericOp.getDpsInitOperand(0);
        auto input0OpOperand = genericOp.getDpsInputOperand(0);
        auto input1OpOperand = genericOp.getDpsInputOperand(1);

        auto initOperand = adaptor.getOperands()[initOpOperand->getOperandNumber()];
        auto input0Operand = adaptor.getOperands()[input0OpOperand->getOperandNumber()];
        auto input1Operand = adaptor.getOperands()[input1OpOperand->getOperandNumber()];

        // the init value, p, should have type i32
        if (!isI32Type(initOperand, rewriter)) {
            return failure();
        }

        // the other two inputs, d and w, should have type i8
        if (!isI8Type(input0Operand, rewriter) || !isI8Type(input1Operand, rewriter)) {
            return failure();
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!yieldOp) {
            return failure();
        }

        torq_hl::GenericOpConfig params;

        // the initial value and map of P value is the init operand
        params.p = torq_hl::getParamFromAdaptor(initOpOperand, adaptor);

        // the indexing map of Q is the same as P
        // the initial value of Q is an empty tensor, it will be overwritten with the Q output
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            genericOp.getLoc(), initOperand.getType(), ValueRange{}
        );
        params.q =
            torq_hl::GenericOpParam(emptyOp, genericOp.getMatchingIndexingMap(initOpOperand));

        auto addOp = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();

        if (!addOp) {
            return failure();
        }

        auto mulOp = addOp.getRhs().getDefiningOp<arith::MulIOp>();

        if (!mulOp) {
            return failure();
        }

        auto mulLhs = mulOp.getLhs().getDefiningOp<arith::ExtSIOp>();

        if (!mulLhs || mulLhs.getIn().getType() != rewriter.getI8Type()) {
            return failure();
        }

        auto mulRhs = mulOp.getRhs().getDefiningOp<arith::ExtSIOp>();

        if (!mulRhs || mulRhs.getIn().getType() != rewriter.getI8Type()) {
            return failure();
        }

        auto mulLhsArg = dyn_cast<BlockArgument>(mulLhs.getIn());
        auto mulRhsArg = dyn_cast<BlockArgument>(mulRhs.getIn());

        if (!mulLhsArg || !mulRhsArg) {
            return failure();
        }

        params.d = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(mulLhsArg), adaptor);
        params.w = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(mulRhsArg), adaptor);

        params.aluConfig = torq_hl::AluConfigAttr::get(
            rewriter.getContext(), torq_hl::ALUOp0Mode::MUL, torq_hl::ALUOp1Mode::ACC
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(genericOp.getLoc(), params);

        rewriter.replaceOp(genericOp, newOp.getResult(1));

        return success();
    }
};

/*

Match an operation of the form:

  %4 = linalg.generic { indexing_maps = [affine_map<..>,
                                         affine_map<..>,
                                         affine_map<..>],
                        iterator_types = [ ... ]}
    ins(%1: tensor<?xi8>) outs(%3 : tensor<?xi32>) {
  ^bb0(%in: i8, %in_3: i8, %out: i32):
    %12 = arith.extsi %in : i8 to i32
    %13 = arith.addi %out, %12 : i32
    linalg.yield %14 : i32
  } -> tensor<?xi32>

To an operation of the form:

%p_in = tensor.empty() : tensor<?xi32>

%4 = torq.generic d (%1 : tensor<?xi8>)
                   q (%3 : tensor<?xi32>)
                   p (%p_in : tensor<?xi32>)
                   d_map = affine_map<..>
                   q_map = affine_map<..>
                   alu_config = { op0 = DBYP , op1 = ACC }

*/
class ReduceSumPattern : public OpConversionPattern<linalg::GenericOp> {

  public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        linalg::GenericOp genericOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // expects one init, the value of p
        if (genericOp.getNumDpsInits() != 1) {
            return failure();
        }

        // expects one input
        if (genericOp.getNumDpsInputs() != 1) {
            return failure();
        }

        auto initOpOperand = genericOp.getDpsInitOperand(0);
        auto inputOpOperand = genericOp.getDpsInputOperand(0);

        auto initOperand = adaptor.getOperands()[initOpOperand->getOperandNumber()];
        auto inputOperand = adaptor.getOperands()[inputOpOperand->getOperandNumber()];

        // the init value, p, should have type i32
        if (!isI32Type(initOperand, rewriter)) {
            return failure();
        }

        // the other two inputs, d and w, should have type i8
        if (!isI8Type(inputOperand, rewriter)) {
            return failure();
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!yieldOp) {
            return failure();
        }

        torq_hl::GenericOpConfig params;

        // the initial value and map of P value is the init operand
        params.p = torq_hl::getParamFromAdaptor(initOpOperand, adaptor);

        // the indexing map of Q is the same as P
        // the initial value of Q is an empty tensor, it will be overwritten with the Q output
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            genericOp.getLoc(), initOperand.getType(), ValueRange{}
        );
        params.q =
            torq_hl::GenericOpParam(emptyOp, genericOp.getMatchingIndexingMap(initOpOperand));

        auto addOp = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();

        if (!addOp) {
            return failure();
        }

        auto addLhs = addOp.getLhs().getDefiningOp<arith::ExtSIOp>();

        if (!addLhs || addLhs.getIn().getType() != rewriter.getI8Type()) {
            return failure();
        }

        auto addLhsArg = dyn_cast<BlockArgument>(addLhs.getIn());

        if (!addLhsArg) {
            return failure();
        }

        params.d = torq_hl::getParamFromAdaptor(genericOp.getMatchingOpOperand(addLhsArg), adaptor);

        params.aluConfig = torq_hl::AluConfigAttr::get(
            rewriter.getContext(), torq_hl::ALUOp0Mode::DBYP, torq_hl::ALUOp1Mode::ACC
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(genericOp.getLoc(), params);

        rewriter.replaceOp(genericOp, newOp.getResult(1));

        return success();
    }
};

/*

Match an operation of the form:

  %4 = linalg.generic {...} ins(%1 : tensor<?xi8>) outs(%2 : tensor<?xi8>) {
  ^bb0(%in: i8, %in_3: i8):
    linalg.yield %in : i8
  } -> tensor<?xi32>

To an operation of the form:

    %p_in = tensor.empty() : tensor<?xi32>

    %4 = torq.generic d (%1 : tensor<?xi8>)
                       w (%2: tensor<?xi8>
                       q (%3 : tensor<?xi32>)
                       p (%p_in : tensor<?xi32>)
                       d_map = affine_map<..>
                       w_map = affine_map<..>
                       q_map = affine_map<..>
                       alu_config = { op0 = DBYP , op1 = ACC }

*/
class MemCopyPattern : public OpConversionPattern<linalg::GenericOp> {

  public:
    using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        linalg::GenericOp genericOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // expects one init, the value of p
        if (genericOp.getNumDpsInits() != 1) {
            return failure();
        }

        // expects one input
        if (genericOp.getNumDpsInputs() != 1) {
            return failure();
        }

        auto initOperand =
            adaptor.getOperands()[genericOp.getDpsInitOperand(0)->getOperandNumber()];
        auto inputOperand =
            adaptor.getOperands()[genericOp.getDpsInputOperand(0)->getOperandNumber()];

        // input and output must be of type i8 or i32
        if (!((isI8Type(initOperand, rewriter) && isI8Type(inputOperand, rewriter)) ||
              (isI32Type(initOperand, rewriter) && isI32Type(inputOperand, rewriter)))) {
            return failure();
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());

        if (!yieldOp) {
            return failure();
        }

        auto arg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));

        if (!arg) {
            return failure();
        }

        // the value being copied has to be the first and only input
        if (arg.getArgNumber() != 0) {
            return failure();
        }

        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            genericOp.getLoc(), initOperand.getType(), ValueRange{}
        );

        auto initOpOperand = genericOp.getDpsInitOperand(0);

        torq_hl::GenericOpConfig params;

        // the indexing map of Q is the same as P
        // the initial value of Q is an empty tensor, it will be overwritten with the Q output
        params.d = torq_hl::getParamFromAdaptor(genericOp.getDpsInputOperand(0), adaptor);

        // this operation is using a primitive value, we need to make the operand a tensor
        if (!isa<RankedTensorType>(params.d.value().getType())) {

            auto tensorOp =
                rewriter.create<tensor::FromElementsOp>(genericOp.getLoc(), params.d.value());

            // int64_t rank = tensorOp.getType().getRank();
            auto tensorMap = AffineMap::get(1, 0, getAffineConstantExpr(0, rewriter.getContext()));

            params.d = torq_hl::GenericOpParam(tensorOp, tensorMap);
        }

        params.p = torq_hl::getParamFromAdaptor(initOpOperand, adaptor);
        params.q =
            torq_hl::GenericOpParam(emptyOp, genericOp.getMatchingIndexingMap(initOpOperand));

        params.aluConfig = torq_hl::AluConfigAttr::get(
            genericOp.getContext(), torq_hl::ALUOp0Mode::DBYP, torq_hl::ALUOp1Mode::ACC
        );

        auto newOp = rewriter.create<torq_hl::GenericOp>(genericOp.getLoc(), params);

        rewriter.replaceOp(genericOp, newOp.getResult(1));

        return success();
    }
};

} // namespace

void populateLinalgToAluPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<AddMulPattern>(context);
    patterns.insert<ReduceSumPattern>(context);
    patterns.insert<MemCopyPattern>(context);
}

} // namespace mlir::syna::torq
