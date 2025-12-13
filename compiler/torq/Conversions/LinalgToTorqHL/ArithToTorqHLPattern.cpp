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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arith-to-torq-pattern"

namespace mlir::syna::torq {

class ElementwiseBinaryArithOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        Operation *op = getElementwiseBinaryOp(srcOp, /*allowCmp=*/true);
        if (!op) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        auto resultType = mlir::dyn_cast<RankedTensorType>(srcOp.getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the output type to be a RankedTensorType"
            );
        }

        auto lhs = op->getOperand(0);
        auto rhs = op->getOperand(1);
        torq_hl::ElementwiseOpEnum opType;
        bool isUnsigned = false;
        auto input0 = srcOp.getInputs()[0];
        auto input1 = srcOp.getInputs()[srcOp.getInputs().size() > 1 ? 1 : 0];
        bool swapInputs = false;
        if (isa<arith::AndIOp>(op)) {
            if (resultType.getElementType().isInteger(1)) {
                opType = torq_hl::ElementwiseOpEnum::LOGICAL_AND;
            }
            else {
                opType = torq_hl::ElementwiseOpEnum::BITWISE_AND;
            }
        }
        else if (isa<arith::OrIOp>(op)) {
            if (resultType.getElementType().isInteger(1)) {
                opType = torq_hl::ElementwiseOpEnum::LOGICAL_OR;
            }
            else {
                opType = torq_hl::ElementwiseOpEnum::BITWISE_OR;
            }
        }
        else if (isa<arith::XOrIOp>(op)) {
            if (isa<BlockArgument>(lhs)) {
                Operation *rhsDefOp = rhs.getDefiningOp();
                if (rhsDefOp && isa<arith::ConstantOp>(rhsDefOp)) {
                    auto constOp = cast<arith::ConstantOp>(rhsDefOp);
                    auto intAttr = mlir::dyn_cast<IntegerAttr>(constOp.getValue());
                    // check if rhs is bool value, if so, it's actually a logical not(unary op not
                    // binary op)
                    if (intAttr && intAttr.getValue().isOne()) {
                        return rewriter.notifyMatchFailure(
                            srcOp, "unary XOrIOp(logical not) op can't be considered as binary op"
                        );
                    }
                }
            }
            if (resultType.getElementType().isInteger(1)) {
                opType = torq_hl::ElementwiseOpEnum::LOGICAL_XOR;
            }
            else {
                opType = torq_hl::ElementwiseOpEnum::BITWISE_XOR;
            }
        }
        else if (isa<arith::MinimumFOp>(op) || isa<arith::MinSIOp>(op) || isa<arith::MinUIOp>(op) ||
                 isa<arith::MinNumFOp>(op)) {
            opType = torq_hl::ElementwiseOpEnum::MINIMUM;
        }
        else if (isa<arith::MaximumFOp>(op) || isa<arith::MaxSIOp>(op) || isa<arith::MaxUIOp>(op) ||
                 isa<arith::MaxNumFOp>(op)) {
            opType = torq_hl::ElementwiseOpEnum::MAXIMUM;
        }
        // act based add/sub doesn't support non-alignment, and it should be conditioned by its cmd
        // options
        else if (clACTBasedSub && resultType.getElementType().isInteger(32) &&
                 isa<arith::SubIOp>(op)) {
            opType = torq_hl::ElementwiseOpEnum::SUB;
        }
        else if (clACTBasedAdd && resultType.getElementType().isInteger(32) &&
                 isa<arith::AddIOp>(op)) {
            opType = torq_hl::ElementwiseOpEnum::ADD;
        }
        else if (isa<arith::CmpFOp>(op)) {
            auto cmpFOp = dyn_cast<arith::CmpFOp>(op);

            // TODO: add UNE, OLE

            auto predicate = cmpFOp.getPredicate();

            if (predicate == arith::CmpFPredicate::OGE) {
                opType = torq_hl::ElementwiseOpEnum::GREATER_EQUAL;
            }
            else if (predicate == arith::CmpFPredicate::OGT) {
                opType = torq_hl::ElementwiseOpEnum::GREATER;
            }
            else if (predicate == arith::CmpFPredicate::OLT) {
                // Orderd less than => Reverse inputs and use GREATER
                opType = torq_hl::ElementwiseOpEnum::GREATER;
                swapInputs = true;
            }
            else if (predicate == arith::CmpFPredicate::OEQ) {
                opType = torq_hl::ElementwiseOpEnum::EQUAL;
            }
            else {
                return rewriter.notifyMatchFailure(
                    srcOp, "Unsupported comparison operation in linalg.generic"
                );
            }
        }
        else if (isa<arith::CmpIOp>(op)) {
            auto cmpIOp = dyn_cast<arith::CmpIOp>(op);

            // TODO: add ne

            auto predicate = cmpIOp.getPredicate();
            if (predicate == arith::CmpIPredicate::sge || predicate == arith::CmpIPredicate::uge) {
                opType = torq_hl::ElementwiseOpEnum::GREATER_EQUAL;
            }
            else if (predicate == arith::CmpIPredicate::sgt ||
                     predicate == arith::CmpIPredicate::ugt) {
                opType = torq_hl::ElementwiseOpEnum::GREATER;
            }
            else if (predicate == arith::CmpIPredicate::eq) {
                opType = torq_hl::ElementwiseOpEnum::EQUAL;
            }
            else if (predicate == arith::CmpIPredicate::slt) {
                // Signed less than => Reverse inputs and use GREATER
                opType = torq_hl::ElementwiseOpEnum::GREATER;
                swapInputs = true;
            }
            else if (predicate == arith::CmpIPredicate::ult) {
                // Unsigned less than => Reverse inputs and use GREATER
                opType = torq_hl::ElementwiseOpEnum::GREATER;
                swapInputs = true;
                isUnsigned = true;
            }
            else if (predicate == arith::CmpIPredicate::sle) {
                // Signed less than equal => Reverse inputs and use GREATER_EQUAL
                opType = torq_hl::ElementwiseOpEnum::GREATER_EQUAL;
                swapInputs = true;
            }
            else if (predicate == arith::CmpIPredicate::ule) {
                // Unsigned less than equal => Reverse inputs and use GREATER_EQUAL
                opType = torq_hl::ElementwiseOpEnum::GREATER_EQUAL;
                swapInputs = true;
                isUnsigned = true;
            }
            else {
                return rewriter.notifyMatchFailure(
                    srcOp, "Unsupported comparison operation in linalg.generic"
                );
            }
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Unsupported elementwise arith operation in linalg.generic"
            );
        }

        // Note: IREE may replace the scalar as the second operand, even if in the original input it
        // was the first. We check If the second operand is a scalar, we need to materialize it as a
        // tensor, until scalar constants are directly supported in ElementWiseOp.
        if (auto rhsBlockArg = dyn_cast<BlockArgument>(rhs)) {
            int argIdx = rhsBlockArg.getArgNumber();
            input1 = srcOp.getInputs()[argIdx];
        }
        else if (auto rhsConstOp = dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
            auto inputType = mlir::cast<RankedTensorType>(input0.getType());
            auto constAttr = rhsConstOp.getValue();
            auto constTensor = rewriter.create<arith::ConstantOp>(
                srcOp.getLoc(), inputType, DenseElementsAttr::get(inputType, constAttr)
            );
            input1 = constTensor.getResult();
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported rhs operand type");
        }

        if (swapInputs) {
            std::swap(input0, input1);
        }
        rewriter.replaceOpWithNewOp<torq_hl::ElementWiseBinaryOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opType, input0,
            input1, isUnsigned
        );

        return success();
    }
};

class ElementwiseUnaryArithOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp.getNumDpsInputs() != 1 || srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly 1 inputs and 1 output for elementwise binary arith op"
            );
        }

        auto resultType = mlir::dyn_cast<RankedTensorType>(srcOp.getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the output type to be a RankedTensorType"
            );
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a linalg.yield terminator");
        }

        auto op = yieldOp.getValues()[0].getDefiningOp();
        if (!op) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the yield operand to be defined by an operation"
            );
        }

        if (op->getNumOperands() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the unary arith operation to have exactly 2 operands"
            );
        }

        auto lhsArg = dyn_cast_or_null<BlockArgument>(op->getOperand(0));
        if (!lhsArg) {
            auto lshConstant = getConstIntValue(op->getOperand(0));
            if (!lshConstant) {
                return rewriter.notifyMatchFailure(srcOp, "lhs is neither input nor constant");
            }
            auto rhsArg = dyn_cast<BlockArgument>(op->getOperand(1));
            if (!rhsArg) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected the second operand to be a BlockArgument"
                );
            }
        }
        else {
            auto rhsConstant = getConstIntValue(op->getOperand(1));
            if (!rhsConstant) {
                return rewriter.notifyMatchFailure(srcOp, "rhs is not constant");
            }
        }

        torq_hl::ElementwiseOpEnum opType;

        if (isa<arith::XOrIOp>(op)) {
            if (resultType.getElementType().isInteger(1)) {
                opType = torq_hl::ElementwiseOpEnum::LOGICAL_NOT;
            }
            else {
                opType = torq_hl::ElementwiseOpEnum::BITWISE_NOT;
            }
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Unsupported elementwise unary arith operation in linalg.generic"
            );
        }

        rewriter.replaceOpWithNewOp<torq_hl::ElementWiseUnaryOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opType,
            srcOp.getInputs()[0]
        );

        return success();
    }
};

template <typename OpTy> class ArithCastOpPattern : public OpRewritePattern<OpTy> {
  public:
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy srcOp, PatternRewriter &rewriter) const override {

        auto resultType = mlir::dyn_cast<RankedTensorType>(srcOp.getOut().getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the output type to be a RankedTensorType"
            );
        }

        auto opName = getCastOpName(srcOp.getIn(), srcOp.getOut());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opName, 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getIn()
        );

        return success();
    }
};

// rounding right shift
// y = (x >> 7) + (((x >> 6) & 1) ? 1 : 0)

// clang-format off
// case #1: one input/output, shift is constant
// %4 = linalg.generic {indexing_maps = .... ins(%2 : tensor<1x21x1024xi32>) outs(%3 : tensor<1x21x1024xi32>) {
// ^bb0(%in: i32, %out: i32):
//     %5 = arith.shrsi %in, %c7_i32 : i32
//     %6 = arith.shrsi %in, %c6_i32 : i32
//     %7 = arith.trunci %6 : i32 to i1
//     %8 = arith.extui %7 : i1 to i32
//     %9 = arith.addi %5, %8 : i32
//     linalg.yield %9 : i32
// } -> tensor<1x21x1024xi32>

// case #2: two tensor inputs
// %6 = linalg.generic {indexing_maps = ... ins(%3, %4 : tensor<1x21x1024xi32>, tensor<1x21x1xi32>) outs(%5 : tensor<1x21x1024xi32>) {
//   ^bb0(%in: i32, %in_0: i32, %out: i32):
//     %7 = arith.shrsi %in, %in_0 : i32
//     %8 = arith.cmpi sgt, %in_0, %c0_i32 : i32
//     %9 = arith.subi %in_0, %c1_i32 : i32
//     %10 = arith.shrsi %in, %9 : i32
//     %11 = arith.trunci %10 : i32 to i1
//     %12 = arith.andi %8, %11 : i1
//     %13 = arith.extui %12 : i1 to i32
//     %14 = arith.addi %7, %13 : i32
//     linalg.yield %14 : i32
// } -> tensor<1x21x1024xi32>
// clang-format on

class RoundingRightShiftPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        arith::ShRSIOp shrsiOp1;
        if (!isRoundingRightShiftOp(op, shrsiOp1)) {
            return rewriter.notifyMatchFailure(op, "Not a RoundingRightShiftOp");
        }

        auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(
                op, "Expected result type to be tensor for RoundingRightShiftPattern"
            );
        }

        auto input1 = op.getOperand(0);
        auto input2 = op.getOperand(1);

        // for case #1, we need to get shift constant and create a constant tensor
        if (op.getNumDpsInputs() == 1) {
            auto c1 = shrsiOp1.getRhs().getDefiningOp<arith::ConstantOp>();
            if (!c1) {
                return rewriter.notifyMatchFailure(
                    op, "Expected defining operation for yield operand to be arith.constant for "
                        "RoundingRightShiftPattern"
                );
            }

            auto shift = dyn_cast<IntegerAttr>(c1.getValue()).getInt();
            auto size = resultType.getNumElements();
            SmallVector<int32_t> data(size, static_cast<int32_t>(shift));

            RankedTensorType input2Type =
                RankedTensorType::get(resultType.getShape(), rewriter.getI32Type());
            auto value = DenseIntElementsAttr::get(input2Type, data);
            input2 =
                rewriter
                    .create<arith::ConstantOp>(
                        op.getLoc(),
                        RankedTensorType::get(resultType.getShape(), rewriter.getI32Type()), value
                    )
                    .getResult();
        }

        rewriter.replaceOpWithNewOp<torq_hl::ElementWiseShiftOp>(
            op, op.getResult(0).getType(), createInitTensor(op, rewriter, resultType),
            torq_hl::ShiftModeEnum::ASR, true, input1, input2
        );

        return success();
    }
};

class ElementWiseShiftOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        Operation *binaryOp = getElementwiseBinaryOp(srcOp, /*allowConstants*/ true);
        if (!binaryOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        torq_hl::ShiftModeEnum opName;
        bool round = false;
        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        // Detect which shift op is used
        if (isa<arith::ShLIOp>(binaryOp)) {
            opName = torq_hl::ShiftModeEnum::LSL;
        }
        else if (isa<arith::ShRUIOp>(binaryOp)) {
            opName = torq_hl::ShiftModeEnum::LSR;
        }
        else if (isa<arith::ShRSIOp>(binaryOp)) {
            opName = torq_hl::ShiftModeEnum::ASR;
            round = false;
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a defining operation for yield operand to be arith.shli, "
                       "arith.shrui or arith.shrsi"
            );
        }

        rewriter.replaceOpWithNewOp<torq_hl::ElementWiseShiftOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            opName, round, srcOp.getOperand(0), srcOp.getOperand(1)
        );
        return success();
    }
};

class SelectOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        Operation *ternaryOp = getElementwiseTernaryOp(srcOp, /*allowConstants*/ true);
        if (!ternaryOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise ternary op");
        }
        // Match arith.select
        auto selectOp = dyn_cast<arith::SelectOp>(ternaryOp);
        if (!selectOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an arith.select operation");
        }
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        SmallVector<Value, 3> selectInputs;
        for (int i = 0; i < 3; ++i) {
            Value operand = ternaryOp->getOperand(i);
            if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                int argIdx = blockArg.getArgNumber();
                selectInputs.push_back(srcOp.getInputs()[argIdx]);
            }
            else if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(operand.getDefiningOp())) {
                auto constAttr = constOp.getValue();
                auto constTensor = rewriter.create<arith::ConstantOp>(
                    srcOp.getLoc(), resultType, DenseElementsAttr::get(resultType, constAttr)
                );
                selectInputs.push_back(constTensor.getResult());
            }
            else {
                return rewriter.notifyMatchFailure(
                    srcOp, "Select operand is not block arg or constant"
                );
            }
        }

        rewriter.replaceOpWithNewOp<torq_hl::SelectOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), selectInputs[0],
            selectInputs[1], selectInputs[2]
        );
        return success();
    }
};

void populateArithToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<ArithCastOpPattern<arith::ExtUIOp>>(context);
    patterns.insert<ArithCastOpPattern<arith::TruncIOp>>(context);

    patterns.insert<ElementwiseBinaryArithOpPattern>(context);
    patterns.insert<ElementwiseUnaryArithOpPattern>(context);
    patterns.insert<ElementWiseShiftOpPattern>(context);

    patterns.insert<RoundingRightShiftPattern>(context);

    patterns.insert<SelectOpPattern>(context);
}

} // namespace mlir::syna::torq