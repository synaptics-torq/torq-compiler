// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

#define DEBUG_TYPE "linalg-torq-patterns"

namespace mlir::syna::torq {

bool isTorqCastOp(Operation *op, std::string &opName, std::string &failReason, bool *isUnsigned) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

    // Initialize isUnsigned to false
    if (isUnsigned) {
        *isUnsigned = false;
    }

    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getNumDpsInputs() != 1) {
        LLVM_DEBUG({ llvm::dbgs() << "expect 1 inputs\n"; });
        failReason = "Expect 1 input";
        return false;
    }

    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();
    auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    auto outputElementType = resultType.getElementType();

    if (inputElementType.isF64() || outputElementType.isF64() || inputElementType.isInteger(64) ||
        outputElementType.isInteger(64)) {
        failReason = "Torq CastOp doesn't support 64bit operand";
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    auto castOp = yieldOp.getValues()[0].getDefiningOp();

    if (!castOp) {
        failReason = "Expected a defining operation for yield operand";
        return false;
    }

    if (inputElementType.isF32() && outputElementType.isBF16()) {
        castOp = dyn_cast_or_null<arith::TruncFOp>(castOp);
    }
    else if (inputElementType.isBF16() && outputElementType.isF32()) {
        castOp = dyn_cast_or_null<arith::ExtFOp>(castOp);
    }
    else if (inputElementType.isInteger() && outputElementType.isInteger()) {
        // We should check the cast operation type here, as similar logic applies for ExtSIOp in
        // the following condition.
        if ((inputElementType.isInteger(1) || (inputElementType.isInteger(8))) &&
            (outputElementType.isInteger(8) || outputElementType.isInteger(16)) &&
            isa<arith::ExtUIOp>(castOp)) {
            castOp = dyn_cast_or_null<arith::ExtUIOp>(castOp);
            if (isUnsigned) {
                *isUnsigned = true;
            }
        }
        else if (inputElementType.getIntOrFloatBitWidth() >
                 outputElementType.getIntOrFloatBitWidth()) {
            castOp = dyn_cast_or_null<arith::TruncIOp>(castOp);
        }
        else if (inputElementType.getIntOrFloatBitWidth() <
                 outputElementType.getIntOrFloatBitWidth()) {
            castOp = dyn_cast_or_null<arith::ExtSIOp>(castOp);
        }
        else {
            failReason = "Unsupported cast operation between input and output types";
            return false;
        }
    }
    else if (inputElementType.isInteger() &&
             (outputElementType.isF32() || outputElementType.isBF16())) {
        castOp = dyn_cast_or_null<arith::SIToFPOp>(castOp);
    }
    else if ((inputElementType.isF32() || inputElementType.isBF16()) &&
             outputElementType.isInteger()) {
        auto &block = srcOp.getRegion().front();
        auto &firstOp = block.front();
        if (!isa<math::RoundEvenOp>(firstOp)) {
            failReason = "Expected the first operation in the linalg body to be a "
                         "math.round_even when casting from float to integer";
            return false;
        }

        castOp = &firstOp;

        // Check if the linalg body contains a FPToSIOp or FPToUIOp
        bool hasFPToSI = false;
        for (auto &block : srcOp.getRegion().getBlocks()) {
            for (auto &op : block.getOperations()) {
                if (isa<arith::FPToSIOp>(op) || isa<arith::FPToUIOp>(op)) {
                    hasFPToSI = true;
                    break;
                }
            }
        }
        if (!hasFPToSI) {
            failReason = "Expected the linalg body to contain a FPToSIOp or FPToUIp when "
                         "casting from float to integer";
            return false;
        }
    }
    else {
        failReason = "Unsupported cast operation between input and output types";
        return false;
    }

    if (!castOp) {
        failReason = "Expected a defining operation for yield operand to be a cast operation";
        return false;
    }

    // castop in is not input
    auto arg = dyn_cast<BlockArgument>(castOp->getOperands()[0]);
    if (!arg) {
        failReason = "Expected the input of cast operation to be a BlockArgument";
        return false;
    }

    opName = getCastOpName(input, srcOp.getResult(0));

    if (opName.empty()) {
        failReason = "Unsupported cast operation between input and output types";
        return false;
    }

    return true;
}

bool isTorqAbsOp(Operation *op, std::string &failReason) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
        failReason = "Expected 1 input and 1 output";
        return false;
    }
    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    if (inputElementType.isF64() || inputElementType.isInteger(64)) {
        failReason = "Torq AbsOp doesn't support 64bit operand";
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    if (inputElementType.isInteger()) {
        auto maxOp = yieldOp.getValues()[0].getDefiningOp<arith::MaxSIOp>();
        if (!maxOp) {
            failReason = "Expected a defining operation for yield operand to be arith.maxsi";
            return false;
        }
        auto subOp = maxOp.getRhs().getDefiningOp<arith::SubIOp>();
        if (!subOp) {
            failReason = "Expected a defining operation for maxop rhs operand to be arith.subi";
            return false;
        }

        auto rhsArg = dyn_cast<BlockArgument>(subOp.getRhs());
        if (!rhsArg) {
            failReason = "Expected the rhs input of sub operation to be a BlockArgument";
            return false;
        }
    }
    else if (inputElementType.isF32() || inputElementType.isBF16()) {
        auto absOp = yieldOp.getValues()[0].getDefiningOp<math::AbsFOp>();
        if (!absOp) {
            failReason = "Expected a defining operation for yield operand to be arith.absf";
            return false;
        }

        auto arg = dyn_cast<BlockArgument>(absOp.getOperand());
        if (!arg) {
            failReason = "Expected the input of abs operation to be a BlockArgument";
            return false;
        }
    }
    else {
        failReason = "Unsupported element type";
        return false;
    }

    return true;
}

bool isTorqCeilOp(Operation *op, std::string &failReason) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
        failReason = "Expected 1 input and 1 output";
        return false;
    }

    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    if (!inputElementType.isF32() && !inputElementType.isBF16()) {
        failReason = "Unsupported element type for CeilOp";
        return false;
    }

    auto ceilOp = yieldOp.getValues()[0].getDefiningOp<math::CeilOp>();
    if (!ceilOp) {
        failReason = "Expected a defining operation for yield operand to be math.ceil";
        return false;
    }

    auto arg = dyn_cast<BlockArgument>(ceilOp.getOperand());
    if (!arg) {
        failReason = "Expected the input of ceil operation to be a BlockArgument";
        return false;
    }

    return true;
}

bool isTorqClampOp(
    Operation *op, int32_t &minIntValue, int32_t &maxIntValue, float &minFloatValue,
    float &maxFloatValue, std::string &failReason
) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getInputs().size() != 1 || srcOp.getOutputs().size() != 1) {
        failReason = "Expected 1 input and 1 output";
        return false;
    }

    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    if (inputElementType.isF64() || inputElementType.isInteger(64)) {
        failReason = "Torq ClampOp doesn't support 64bit operand";
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    minIntValue = 0;
    maxIntValue = 0;
    minFloatValue = 0.0f;
    maxFloatValue = 0.0f;

    if (inputElementType.isF32() || inputElementType.isBF16()) {
        auto maxOp = yieldOp.getValues()[0].getDefiningOp<arith::MaximumFOp>();
        if (!maxOp) {
            failReason = "Expected a defining operation for yield operand to be arith.maximumf";
            return false;
        }
        auto minOp = maxOp.getLhs().getDefiningOp<arith::MinimumFOp>();
        if (!minOp) {
            failReason = "Expected a defining operation for yield operand to be arith.minimumf";
            return false;
        }

        auto constOp = minOp.getRhs().getDefiningOp<arith::ConstantOp>();
        auto constOp2 = maxOp.getRhs().getDefiningOp<arith::ConstantOp>();
        if (!constOp || !constOp2) {
            failReason = "Expected both min and max operations to have constant rhs operands";
            return false;
        }
        maxFloatValue = dyn_cast<FloatAttr>(constOp.getValue()).getValue().convertToFloat();
        minFloatValue = dyn_cast<FloatAttr>(constOp2.getValue()).getValue().convertToFloat();

        auto arg = dyn_cast<BlockArgument>(minOp.getLhs());
        if (!arg) {
            failReason = "Expected the lhs of min operation to be a BlockArgument";
            return false;
        }
    }
    else if (inputElementType.isInteger()) {
        arith::MinSIOp minOp = yieldOp.getValues()[0].getDefiningOp<arith::MinSIOp>();
        if (!minOp) {
            failReason = "Expected a defining operation for yield operand to be arith.minsi";
            return false;
        }
        arith::MaxSIOp maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();
        if (!maxOp) {
            failReason = "Expected a defining operation for yield operand to be arith.maxsi";
            return false;
        }

        auto maxInt = getConstIntValue(minOp.getRhs());
        if (!maxInt) {
            failReason = "Expected minOp rhs to be a constant integer";
            return false;
        }
        maxIntValue = *maxInt;

        auto minInt = getConstIntValue(maxOp.getRhs());
        if (!minInt) {
            failReason = "Expected maxOp rhs to be a constant integer";
            return false;
        }
        minIntValue = *minInt;

        auto arg = dyn_cast<BlockArgument>(maxOp.getLhs());
        if (!arg) {
            failReason = "Expected the lhs of max operation to be a BlockArgument";
            return false;
        }
    }
    else {
        failReason = "Unsupported element type for clamp operation";
        return false;
    }

    return true;
}

bool isTorqFloorOp(Operation *op, std::string &failReason) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);
    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getInputs().size() != 1 || srcOp.getOutputs().size() != 1) {
        failReason = "Expects generic op with 1 input and 1 output";
        return false;
    }
    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    if (!inputElementType.isF32() && !inputElementType.isBF16()) {
        failReason = "Unsupported element type for CeilOp";
        return false;
    }

    auto floorOp = yieldOp.getValues()[0].getDefiningOp<math::FloorOp>();
    if (!floorOp) {
        failReason = "Expected a defining operation for yield operand to be math.floor";
        return false;
    }

    auto arg = dyn_cast<BlockArgument>(floorOp.getOperand());
    if (!arg) {
        failReason = "Expected the input of neg operation to be a BlockArgument";
        return false;
    }

    return true;
}

bool isTorqMatMul(Operation *op, std::string &failReason) {

    if (!isa<linalg::BatchMatmulOp, linalg::MatmulOp, linalg::DotOp, linalg::MatvecOp>(op)) {
        failReason = "Not a supported matmul op";
        return false;
    }

    return true;
}

bool isTorqNegateOp(Operation *op, std::string &failReason) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

    if (!srcOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
        failReason = "Expects 1 input and 1 output";
        return false;
    }

    Value input = srcOp.getInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto inputElementType = inputType.getElementType();

    if (inputElementType.isF64() || inputElementType.isInteger(64)) {
        failReason = "Torq NegateOp doesn't support 64bit operand";
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    if (inputElementType.isInteger()) {
        auto subOp = yieldOp.getValues()[0].getDefiningOp<arith::SubIOp>();
        if (!subOp) {
            failReason = "Expected a defining operation for yield operand to be arith.subi";
            return false;
        }

        auto lshConstant = getConstIntValue(subOp.getLhs());
        if (!lshConstant) {
            failReason = "lhs is neither input nor constant";
            return false;
        }
        if (*lshConstant != 0) {
            failReason = "Expected lhs of sub operation to be zero for NegateOp";
            return false;
        }

        auto rhsArg = dyn_cast<BlockArgument>(subOp.getRhs());
        if (!rhsArg) {
            failReason = "Expected both inputs of rhs to be BlockArguments";
            return false;
        }
    }
    else if (inputElementType.isF32() || inputElementType.isBF16()) {
        auto negOp = yieldOp.getValues()[0].getDefiningOp<arith::NegFOp>();
        if (!negOp) {
            failReason = "Expected a defining operation for yield operand to be arith.negf";
            return false;
        }

        auto arg = dyn_cast<BlockArgument>(negOp.getOperand());
        if (!arg) {
            failReason = "Expected the input of neg operation to be a BlockArgument";
            return false;
        }
    }
    else {
        failReason = "Unsupported element type for NegateOp";
        return false;
    }

    return true;
}

bool isTorqReduceSumOp(Operation *op, std::string &failReason) {
    auto reduceOp = dyn_cast<linalg::ReduceOp>(op);
    if (!reduceOp) {
        failReason = "Not a linalg.reduce op";
        return false;
    }

    // Only support single input
    if (reduceOp.getInputs().size() != 1) {
        failReason = "Reduce sum only supports exactly one input";
        return false;
    }

    // Check input element type - bf16, f32, or integers
    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
    if (!inputType) {
        failReason = "Input is not a ranked tensor";
        return false;
    }
    auto inputElementType = inputType.getElementType();

    // Support BF16, F32, or integer types.
    // Both bf16 and f32 inputs are supported by the hardware kernel.
    if (!inputElementType.isBF16() && !inputElementType.isF32() && !inputElementType.isInteger()) {
        failReason = "Reduce sum only supports BF16, F32, or integer input types";
        return false;
    }

    // Check bit width
    if (inputElementType.getIntOrFloatBitWidth() != 8 &&
        inputElementType.getIntOrFloatBitWidth() != 16 &&
        inputElementType.getIntOrFloatBitWidth() != 32) {
        failReason = "Reduce sum only supports 8, 16, or 32 bit element types";
        return false;
    }

    // Check that it's a sum reduction (arith.addf or arith.addi in the body)
    auto yieldOp = dyn_cast<linalg::YieldOp>(reduceOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected linalg::YieldOp as the terminator";
        return false;
    }

    auto reduceBodyOp = yieldOp.getOperand(0).getDefiningOp();
    if (!isa<arith::AddIOp>(reduceBodyOp) && !isa<arith::AddFOp>(reduceBodyOp)) {
        failReason = "Not a sum reduction (expected arith.addf or arith.addi)";
        return false;
    }

    // Check output type - can be same as input or f32 (for bf16 input)
    auto outputType = dyn_cast<RankedTensorType>(reduceOp->getResultTypes().front());
    if (!outputType) {
        failReason = "Output is not a ranked tensor";
        return false;
    }
    auto outputElementType = outputType.getElementType();

    // For bf16 input, output can be bf16 or f32
    if (inputElementType.isBF16()) {
        if (!outputElementType.isBF16() && !outputElementType.isF32()) {
            failReason = "BF16 reduce sum output must be BF16 or F32";
            return false;
        }
    }
    else {
        // For integer inputs, output must match input type
        if (inputElementType != outputElementType) {
            failReason = "Integer reduce sum output type must match input type";
            return false;
        }
    }

    // Only support single dimension reduction
    if (reduceOp.getDimensions().size() != 1) {
        failReason = "Reduce sum only supports single dimension reduction";
        return false;
    }

    return true;
}

namespace {

// NOTE: this struct is duplicated below (TransposeOpConversionRewrite) as a OpRewritePattern.
// Any change here should be reflected there too. Ultimately, we should get rid of
// TransposeOpConversion.
struct TransposeOpConversion : public OpConversionPattern<linalg::TransposeOp> {

    TransposeOpConversion(MLIRContext *context) : OpConversionPattern(context) {
        setHasBoundedRewriteRecursion();
    }
    LogicalResult matchAndRewrite(
        linalg::TransposeOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        if (srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 output");
        }

        if (!failed(foldForwardDepthToSpace(srcOp, rewriter, std::nullopt))) {
            return success();
        }

        rewriter.replaceOpWithNewOp<torq_hl::TransposeOp>(
            srcOp, srcOp.getResult()[0].getType(), adaptor.getInit(), adaptor.getPermutationAttr(),
            adaptor.getInput()
        );

        return success();
    }
};

// NOTE: this struct is duplicated above (TransposeOpConversion) as a OpConversionPattern.
// Any change here should be reflected there too. Ultimately, we should get rid of
// TransposeOpConversion.
struct TransposeOpConversionRewrite : public OpRewritePattern<linalg::TransposeOp> {
  private:
    const bool _markFuseGroups;

  public:
    TransposeOpConversionRewrite(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {
        setHasBoundedRewriteRecursion();
    }

    LogicalResult
    matchAndRewrite(linalg::TransposeOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        std::optional<IntegerAttr> maybeFuseGroupAttr = std::nullopt;
        if (_markFuseGroups) {
            maybeFuseGroupAttr = srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
        }

        if (srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 output");
        }

        if (!failed(foldForwardDepthToSpace(srcOp, rewriter, maybeFuseGroupAttr)) ||
            _markFuseGroups) {
            markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr);
            return success();
        }

        auto trOp = torq_hl::TransposeOp::create(
            rewriter, srcOp.getLoc(), srcOp.getResult()[0].getType(), srcOp.getInit(),
            srcOp.getPermutationAttr(), srcOp.getInput()
        );
        rewriter.replaceOp(srcOp, trOp.getOutput());

        return success();
    }
};

template <typename OpTy> struct MatmulOpConversion final : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpTy srcOp, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(srcResultType.getElementType());

        rewriter.replaceOpWithNewOp<torq_hl::MatMulOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            0, outMin, outMax, 0, createI32Const(rewriter, srcOp, interleave(bias, scale)),
            srcOp.getOperand(0), srcOp.getOperand(1)
        );

        return success();
    }
};

struct FillOpConversion : public OpConversionPattern<linalg::FillOp> {

    FillOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        linalg::FillOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        // fuse fillOp with the op uses fillOp's result as output
        bool fused = false;
        if (Operation *u = *srcOp.getResults()[0].getUsers().begin()) {
            fused = TypeSwitch<Operation *, bool>(u)
                        .Case<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::PoolingNhwcMaxOp>(
                            [&](auto typedOp) {
                                if (typedOp.getOutputs()[0] == srcOp.getResults()[0]) {

                                    srcOp.getResults()[0].replaceAllUsesWith(srcOp.getOutputs()[0]);
                                    rewriter.eraseOp(srcOp);

                                    return true;
                                }
                                return false;
                            }
                        )
                        .Default([&](Operation *op) { return false; });
        }

        if (fused) {
            return success();
        }

        // TODO(sflur): make sure the ConstantOp is still reachable after tiling.
        auto constantOp = srcOp.value().getDefiningOp<mlir::arith::ConstantOp>();
        if (!constantOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a ConstantOp defining the value");
        }

        TypedAttr valueAttr = constantOp.getValue();
        mlir::IntegerAttr bitPatternAttr;
        int fillElementSize = valueAttr.getType().getIntOrFloatBitWidth() / 8;
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(valueAttr)) {
            // Already an integer constant
            int fillValue = intAttr.getInt();
            if (fillElementSize > 2 && fillValue != 0) {
                return rewriter.notifyMatchFailure(srcOp, "Unsupported 32-bits int value");
            }
            bitPatternAttr = rewriter.getI32IntegerAttr(fillValue);
        }
        else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valueAttr)) {
            // Get the bit pattern of the float
            APFloat apf = floatAttr.getValue();
            uint64_t bits;
            if (&apf.getSemantics() == &llvm::APFloat::BFloat()) {
                bits = apf.bitcastToAPInt().getZExtValue(); // 16-bit
                bitPatternAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(bits));
            }
            else if (&apf.getSemantics() == &llvm::APFloat::IEEEsingle()) {
                bits = apf.bitcastToAPInt().getZExtValue(); // 32-bit
                if (fillElementSize > 2 && bits != 0) {
                    return rewriter.notifyMatchFailure(srcOp, "Unsupported 32-bits float value");
                }
                bitPatternAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(bits));
            }
            else {
                return rewriter.notifyMatchFailure(srcOp, "Unsupported float type");
            }
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported constant type");
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<torq_hl::FillOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            bitPatternAttr
        );

        return success();
    }
};

struct FillOpConversionRewrite : public OpRewritePattern<linalg::FillOp> {
  private:
    const bool _markFuseGroups;

  public:
    FillOpConversionRewrite(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(linalg::FillOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        std::optional<IntegerAttr> maybeFuseGroupAttr = std::nullopt;
        if (_markFuseGroups) {
            if (isMarkedFuseGroup(srcOp)) {
                return rewriter.notifyMatchFailure(srcOp, "Already marked");
            }

            maybeFuseGroupAttr = srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
        }

        // fuse fillOp with the op uses fillOp's result as output
        bool fused = false;
        if (Operation *u = *srcOp.getResults()[0].getUsers().begin()) {
            fused = TypeSwitch<Operation *, bool>(u)
                        .Case<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::PoolingNhwcMaxOp>(
                            [&](auto typedOp) {
                                if (typedOp.getOutputs()[0] == srcOp.getResults()[0]) {

                                    if (_markFuseGroups) {
                                        // Use the id of typedOp to mark srcOp, as typedOp is the
                                        // principal operation.
                                        maybeFuseGroupAttr =
                                            typedOp->template getAttrOfType<IntegerAttr>(
                                                TORQ_FUSE_GROUP_ID
                                            );
                                        markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr);
                                        // Make sure typedOp is also marked. We have to do it
                                        // because typedOp might not have a principal pattern (e.g.
                                        // tests/testdata/tosa_ops/matmul-in-int8-out-int32.mlir
                                        // breaks without the marking here).
                                        // NB: this could break the marking of typedOp if it's
                                        // principal pattern has not run yet. Hence the
                                        // FillOpConversionRewrite should run after all other
                                        // patterns.
                                        markOpFuseGroup(typedOp, rewriter, maybeFuseGroupAttr);
                                        return true;
                                    }

                                    srcOp.getResults()[0].replaceAllUsesWith(srcOp.getOutputs()[0]);
                                    rewriter.eraseOp(srcOp);

                                    return true;
                                }
                                return false;
                            }
                        )
                        .Default([&](Operation *op) { return false; });
        }
        if (fused) {
            return success();
        }

        // TODO(sflur): make sure the ConstantOp is still reachable after tiling.
        auto constantOp = srcOp.value().getDefiningOp<mlir::arith::ConstantOp>();
        if (!constantOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a ConstantOp defining the value");
        }

        TypedAttr valueAttr = constantOp.getValue();
        mlir::IntegerAttr bitPatternAttr;
        int fillElementSize = valueAttr.getType().getIntOrFloatBitWidth() / 8;
        if (auto intAttr = mlir::dyn_cast<IntegerAttr>(valueAttr)) {
            // Already an integer constant
            int fillValue = intAttr.getInt();
            if (fillElementSize > 2 && fillValue != 0) {
                return rewriter.notifyMatchFailure(srcOp, "Unsupported 32-bits int value");
            }
            bitPatternAttr = rewriter.getI32IntegerAttr(fillValue);
        }
        else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(valueAttr)) {
            // Get the bit pattern of the float
            APFloat apf = floatAttr.getValue();
            uint64_t bits;
            if (&apf.getSemantics() == &llvm::APFloat::BFloat()) {
                bits = apf.bitcastToAPInt().getZExtValue(); // 16-bit
                bitPatternAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(bits));
            }
            else if (&apf.getSemantics() == &llvm::APFloat::IEEEsingle()) {
                bits = apf.bitcastToAPInt().getZExtValue(); // 32-bit
                if (fillElementSize > 2 && bits != 0) {
                    return rewriter.notifyMatchFailure(srcOp, "Unsupported 32-bits float value");
                }
                bitPatternAttr = rewriter.getI32IntegerAttr(static_cast<int32_t>(bits));
            }
            else {
                return rewriter.notifyMatchFailure(srcOp, "Unsupported float type");
            }
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported constant type");
        }

        if (markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr)) {
            return success();
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        rewriter.replaceOpWithNewOp<torq_hl::FillOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            bitPatternAttr
        );

        return success();
    }
};

struct ReduceOpConversion : public OpRewritePattern<linalg::ReduceOp> {

    ReduceOpConversion(MLIRContext *context) : OpRewritePattern(context) {}
    LogicalResult
    matchAndRewrite(linalg::ReduceOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Supported exactly one input for ReduceOp");
        }

        auto inputType = mlir::cast<RankedTensorType>(srcOp.getInputs()[0].getType());
        auto inputElementType = inputType.getElementType();

        if (!inputElementType.isIntOrFloat()) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support int or float element types for ReduceOp"
            );
        }

        // Support BF16, F32, or integer inputs.
        // Both bf16 and f32 inputs are supported by the hardware kernel.
        if (!inputElementType.isBF16() && !inputElementType.isF32() &&
            !inputElementType.isInteger()) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support BF16, F32, or integer element types for ReduceOp"
            );
        }

        if (inputElementType.getIntOrFloatBitWidth() != 8 &&
            inputElementType.getIntOrFloatBitWidth() != 16 &&
            inputElementType.getIntOrFloatBitWidth() != 32) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support 8, 16, or 32 bit element types for ReduceOp"
            );
        }

        auto srcOutputType = dyn_cast<RankedTensorType>(srcOp->getResultTypes().front());

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());

        if (!yieldOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected linalg::YieldOp as the terminator of the ReduceOp body"
            );
        }

        Value input = srcOp.getInputs()[0];

        ArrayRef<int64_t> dimensions = srcOp.getDimensions();
        if (dimensions.size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support one dimension reduction in ReduceOp"
            );
        }
        int64_t axis = dimensions[0];

        std::string opName = "reduce_sum";
        auto reduceOp = yieldOp.getOperand(0).getDefiningOp();

        if (isa<arith::AddIOp>(reduceOp) || isa<arith::AddFOp>(reduceOp)) {
            opName = "reduce_sum";
        }
        else if (isa<arith::MaxSIOp>(reduceOp) || isa<arith::MaxUIOp>(reduceOp) ||
                 isa<arith::MaximumFOp>(reduceOp)) {
            opName = "reduce_max";
        }
        else if (isa<arith::MinSIOp>(reduceOp) || isa<arith::MinUIOp>(reduceOp) ||
                 isa<arith::MinimumFOp>(reduceOp)) {
            opName = "reduce_min";
        }
        else if (isa<arith::OrIOp>(reduceOp)) {
            opName = "reduce_or";
        }
        else if (isa<arith::AndIOp>(reduceOp)) {
            opName = "reduce_and";
        }
        else if (isa<arith::XOrIOp>(reduceOp)) {
            opName = "reduce_xor";
        }
        else if (isa<arith::MulFOp>(reduceOp)) {
            opName = "reduce_mul";
        }
        else {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported reduction operation");
        }

        constexpr int shift_factor = 12;
        // reduceOp out_min and out_max is defined in kernel, here just initialize

        // TODO: materialize bias/scale weights for rescale precision
        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        std::vector<int16_t> weights = {1, 1};

        // For bf16 reduce_sum with f32 output, preserve f32 type for better precision
        // If output is bf16, keep it as bf16 (user's choice for memory/performance)
        RankedTensorType resultType = srcOutputType;

        rewriter.replaceOpWithNewOp<syna::torq_hl::ReduceOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opName, axis, 0, 0, 0,
            shift_factor, createI16Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{2}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );

        return success();
    }
};

class AddOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    FloatAttr fromConstScalar(arith::ConstantOp constOp) const {
        if (!constOp) {
            return nullptr;
        }
        auto attr = constOp.getValue();
        if (auto intAttr = dyn_cast<FloatAttr>(attr)) {
            return intAttr;
        }
        auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr);
        assert(denseAttr && "Unsupported constant type");
        return denseAttr.getValues<FloatAttr>()[0];
    }

    LogicalResult get2Inputs(
        linalg::GenericOp srcOp, Operation *binaryOp, Value &input0, Value &input1, float &newBias,
        bool &needReverse, bool &rhs_is_scalar, PatternRewriter &rewriter
    ) const {
        const int numLinalgInputs = srcOp.getNumDpsInputs();
        const int numArithOperands = binaryOp->getNumOperands();
        if (numLinalgInputs == 0 || numLinalgInputs > 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "add expects generic op with 2 (or 1) inputs\n"
            );
        }
        LLVM_DEBUG({
            llvm::dbgs() << "Inputs: " << numLinalgInputs << "Operands: " << numArithOperands
                         << "\n";
        });

        if (numArithOperands != 2) {
            return rewriter.notifyMatchFailure(srcOp, "add expects arith op with 2 operands\n");
        }

        if (numLinalgInputs == 1) {
            // No other choice than to take the single input as input0
            input0 = srcOp.getInputs()[0];
        }

        // one input must be constant, the other is the input tensor
        auto lhs = binaryOp->getOperand(0);
        auto rhs = binaryOp->getOperand(1);

        auto constOp = rhs.getDefiningOp<arith::ConstantOp>();
        if (!constOp) {
            constOp = lhs.getDefiningOp<arith::ConstantOp>();
            if (constOp) {
                needReverse = true;
            }
        }
        FloatAttr data = fromConstScalar(constOp);
        if (data) {
            if (needReverse && numLinalgInputs == 2) {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[1];
            }
            else {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[0];
            }

            newBias = data.getValue().convertToFloat();
            rhs_is_scalar = true;
            // When one operand is a scalar constant, we create a 1-element tensor for input1 to
            // preserve add op semantics. This scalar is incorporated into the bias term instead of
            // being used directly as a tensor input.
            auto elemType = mlir::cast<RankedTensorType>(input0.getType()).getElementType();
            RankedTensorType constType = RankedTensorType::get({1}, elemType);
            DenseElementsAttr value = DenseElementsAttr::get(constType, data.getValue());
            input1 =
                arith::ConstantOp::create(rewriter, srcOp.getLoc(), constType, value).getResult();

            return success();
        }

        input0 = srcOp.getInputs()[0];
        if (numLinalgInputs == 1) {
            input1 = srcOp.getInputs()[0];
        }
        else {
            input1 = srcOp.getInputs()[1];
        }

        return success();
    }

    LogicalResult createBf16Add(
        std::string opName, int sign, linalg::GenericOp srcOp, Operation *binaryOp,
        PatternRewriter &rewriter
    ) const {

        bool rhs_is_scalar = false;

        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weights(2, llvm::APFloat(bf16, "1.0"));
        // for sub, set weight of 2nd input to -1.0
        // so it becomes => %input0 + (-1.0) * %input1
        if (opName == "sub") {
            weights[1] = llvm::APFloat(bf16, "-1.0");
        }
        std::vector<float> biasScale{0.0};

        if (srcOp.getNumDpsInits() != 1 && srcOp.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "bf16 addOp doesn't support single input with two different operands\n"
            );
        }

        Value input0;
        Value input1;
        float newBias;
        bool needReverse = false;
        auto res = get2Inputs(
            srcOp, binaryOp, input0, input1, newBias, needReverse, rhs_is_scalar, rewriter
        );
        if (failed(res)) {
            return res;
        }
        if (rhs_is_scalar) {
            // Pattern: %input - cst
            if (opName == "sub" && !needReverse) {
                newBias = -newBias;
            }
            // Pattern: %input + cst
            biasScale[0] = newBias;
        }
        // Pattern: cst - %input
        // Rewrite: (-1 * %input) + cst
        // Action: flip weights to [-1, +1] and treat as "add"; the constant stays in the bias
        // (needReverse is set when the constant was the lhs of a subtraction)
        // Note: scale not supported for bf16 operations so we have to take this approach
        if (needReverse && opName == "sub") {
            weights[0] = llvm::APFloat(bf16, "-1.0");
            weights[1] = llvm::APFloat(bf16, "1.0");
            opName = "add";
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        // special handling for float min/max
        float min_f = std::numeric_limits<float>::lowest();
        int32_t output_min_f = *reinterpret_cast<int32_t *>(&min_f);
        float max_f = std::numeric_limits<float>::max();
        int32_t output_max_f = *reinterpret_cast<int32_t *>(&max_f);
        auto torqWeights = createConst(weights, rewriter, srcOp.getLoc());

        rewriter.replaceOpWithNewOp<torq_hl::AddOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            opName,
            0, // input_zp
            0, // output_zp
            output_min_f, output_max_f,
            0, // shift_factor
            torqWeights, createConst(biasScale, rewriter, srcOp.getLoc()), input0, input1, false,
            rhs_is_scalar
        );
        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        Operation *binaryOp = getElementwiseBinaryOp(srcOp, true);
        if (!binaryOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        Value output = srcOp.getResultTensors()[0];
        const auto outType = cast<RankedTensorType>(output.getType());
        if (outType.getElementType().isF32() || outType.getElementType().isF64()) {
            return rewriter.notifyMatchFailure(
                srcOp, "addOp doesn't support f32 and f64 right now\n"
            );
        }
        if (outType.getElementType().isInteger(64)) {
            return rewriter.notifyMatchFailure(srcOp, "addOp doesn't support i64");
        }

        const bool isBF16 = outType.getElementType().isBF16();

        auto opName = "";
        int sign = 1;
        if (isa<arith::AddIOp>(binaryOp) || isa<arith::AddFOp>(binaryOp)) {
            opName = "add";
        }
        else if (isa<arith::SubIOp>(binaryOp) || isa<arith::SubFOp>(binaryOp)) {
            opName = "sub";
            sign = -1;
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp,
                "Expected a defining operation for yield operand to be arith.addi or arith.subi"
            );
        }

        if (isBF16) {
            // TODO: factorize common code with i8/i16 path
            return createBf16Add(opName, sign, srcOp, binaryOp, rewriter);
        }

        auto srcOpSize = srcOp.getNumDpsInputs();

        // elementwise binary input type should be the same as the output type
        // TODO: need check i8 dtype, for now we don't have these cases
        if (srcOpSize == 1 && (outType.getElementType().isInteger(8))) {
            return rewriter.notifyMatchFailure(
                srcOp, "Unsupported dtype for elementwise binary operation"
            );
        }

        std::vector<int32_t> bias = {0};
        std::vector<int32_t> scale = {1};
        bool rhs_is_scalar = false;

        Value input0 = srcOp.getOperand(0);
        Value input1 = srcOp.getInputs()[srcOp.getInputs().size() > 1 ? 1 : 0];

        if (srcOpSize == 1 &&
            (outType.getElementType().isInteger(16) || outType.getElementType().isInteger(32))) {
            // one input must be constant, the other is the input tensor

            bool inputNeedReverse = false;
            auto lhs = binaryOp->getOperand(0);
            auto rhs = binaryOp->getOperand(1);

            auto constOp = lhs.getDefiningOp<arith::ConstantOp>();
            if (!constOp) {
                constOp = rhs.getDefiningOp<arith::ConstantOp>();
            }
            else {
                inputNeedReverse = true;
            }
            if (constOp) {
                auto attr = constOp.getValue();
                int32_t data = 0;
                if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                    data = intAttr.getInt();
                }
                else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
                    if (denseAttr.isSplat() || denseAttr.getNumElements() == 1) {
                        data = (*denseAttr.begin()).getSExtValue();
                    }
                    else {
                        // TODO: handle non-scalar constant
                        return rewriter.notifyMatchFailure(srcOp, "Constant tensor must be scalar");
                    }
                }
                else {
                    return rewriter.notifyMatchFailure(srcOp, "Unsupported constant type");
                }
                auto elemType = outType.getElementType();
                RankedTensorType constType = RankedTensorType::get({1}, elemType);
                DenseElementsAttr value;
                if (elemType.isInteger(16)) {
                    value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(data));
                }
                else if (elemType.isInteger(32)) {
                    value = DenseIntElementsAttr::get(constType, data);
                }
                else {
                    return rewriter.notifyMatchFailure(srcOp, "Unsupported constant type");
                }
                input1 = arith::ConstantOp::create(rewriter, srcOp.getLoc(), constType, value)
                             .getResult();

                bias = {data * sign};
                rhs_is_scalar = true;

                if (inputNeedReverse) {
                    scale = {sign};
                }
            }
        }

        auto weightsI16 = createI16Const(
            rewriter, srcOp, std::vector<int16_t>{1, static_cast<int16_t>(sign)},
            llvm::ArrayRef<int64_t>{2}
        );
        auto weightsI8 = createI8Const(
            rewriter, srcOp, std::vector<int8_t>{1, static_cast<int8_t>(sign)},
            llvm::ArrayRef<int64_t>{2}
        );

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
        auto [outMin, outMax] = getDTypeRange(srcResultType.getElementType());

        rewriter.replaceOpWithNewOp<torq_hl::AddOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            opName,
            0, // input_zp
            0, // output_zp
            outMin, outMax,
            0, // shift_factor
            outType.getElementType().isInteger(32) ? weightsI8 : weightsI16,
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input0, input1, false,
            rhs_is_scalar
        );

        return success();
    }
};

// Clamp patterns (ClampOpPattern, NaiveClampOpPattern, UnorderedClampOpPattern)
// have been moved to ClampPattern.cpp and consolidated into ClampOpConversion.
// They are registered via populateLinalgToTorqHLClampPatterns().

class AbsOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;

        if (!isTorqAbsOp(srcOp, failReason)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), "abs", 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getInputs()[0],
            /*weights=*/mlir::Value()
        );

        return success();
    }
};

class NegateOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;
        if (!isTorqNegateOp(srcOp, failReason)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        Value input = srcOp.getInputs()[0];
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), "negate", 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input, /*weights=*/mlir::Value()
        );

        return success();
    }
};

class ClzOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }
        Value input = srcOp.getInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputElementType = inputType.getElementType();
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a linalg.yield terminator");
        }

        if (!inputElementType.isInteger()) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported element type for ClzOp");
        }

        auto clzOp = yieldOp.getValues()[0].getDefiningOp<math::CountLeadingZerosOp>();
        if (!clzOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a defining operation for yield operand to be math.ctlz"
            );
        }

        auto arg = dyn_cast<BlockArgument>(clzOp.getOperand());
        if (!arg) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected the input of neg operation to be a BlockArgument"
            );
        }

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), "clz", 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input, /*weights=*/mlir::Value()
        );

        return success();
    }
};

class CeilOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;

        if (!isTorqCeilOp(srcOp, failReason)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), "ceil", 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getInputs()[0],
            /*weights=*/mlir::Value()
        );

        return success();
    }
};

class FloorOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;
        if (!isTorqFloorOp(srcOp, failReason)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        Value input = srcOp.getInputs()[0];
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), "floor", 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input, /*weights=*/mlir::Value()
        );

        return success();
    }
};

class ReinterpretCastOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // Check for 1 input and 1 output
        if (srcOp.getInputs().size() != 1 || srcOp.getOutputs().size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expects generic op with 1 input and 1 output"
            );
        }

        Value input = srcOp.getInputs()[0];
        auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        // Check for linalg.yield terminator
        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp || yieldOp.getValues().size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a linalg.yield terminator with one value"
            );
        }

        // Check for arith.bitcast in the body
        auto bitcastOp = yieldOp.getValues()[0].getDefiningOp<arith::BitcastOp>();
        if (!bitcastOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected arith.bitcast in linalg.generic body"
            );
        }

        // Check types: i16 -> bf16
        auto bitcastInType = bitcastOp.getIn().getType();
        auto bitcastOutType = bitcastOp.getType();

        // currently suppport bf16->i16 and i16->bf16 only
        // can be extended if needed
        if (!((bitcastInType.isInteger(16) && bitcastOutType.isBF16()) ||
              (bitcastInType.isBF16() && bitcastOutType.isInteger(16)))) {
            return rewriter.notifyMatchFailure(srcOp, "Expected bitcast between i16 and bf16");
        }

        // Check that bitcast input is the block argument (input tensor element)
        auto blockArg = dyn_cast<BlockArgument>(bitcastOp.getIn());
        if (!blockArg) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected bitcast input to be block argument"
            );
        }

        rewriter.replaceOpWithNewOp<torq_hl::IdentityOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), input
        );

        return success();
    }
};

class CastOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;
        std::string opName;
        bool isUnsignedOp = false;

        if (!isTorqCastOp(srcOp, opName, failReason, &isUnsignedOp)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        Value input = srcOp.getInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputElementType = inputType.getElementType();
        auto resultType = cast<RankedTensorType>(srcOp.getResult(0).getType());

        // Create weights tensor [1] for unsigned input
        llvm::SmallVector<mlir::Value, 1> weightsVec;
        if (isUnsignedOp) {
            if (inputElementType.isInteger()) {
                int bitWidth = inputElementType.getIntOrFloatBitWidth();
                // For i1, create weights as int8
                if (bitWidth == 1) {
                    weightsVec.push_back(createIConst(rewriter, srcOp, {APInt(8, 1)}));
                }
                else {
                    weightsVec.push_back(createIConst(rewriter, srcOp, {APInt(bitWidth, 1)}));
                }
            }
            else {
                return rewriter.notifyMatchFailure(
                    srcOp, "Only integer types are supported for unsigned Cast operation"
                );
            }
        }
        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opName, 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            /*input=*/srcOp.getInputs()[0],
            /*weights=*/(weightsVec.empty() ? mlir::Value() : weightsVec.front())
        );

        return success();
    }
};

struct BroadcastOpConversion : public OpRewritePattern<linalg::BroadcastOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::BroadcastOp srcOp, PatternRewriter &rewriter) const override {

        auto outputType = cast<RankedTensorType>(srcOp.getInit().getType());

        auto op = torq_hl::BroadcastOp::create(
            rewriter, srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
            srcOp.getDimensionsAttr(), srcOp.getInput()
        );
        rewriter.replaceOp(srcOp, op.getOutput());

        return success();
    }
};

// rescale various cases
// ui8 -> i8
// %5 = arith.extui %in : i8 to i32
// %6 = arith.subi %5, %c128_i32 : i32
// %7 = tosa.apply_scale %6, %c1073741824_i32, %c30_i8 {double_round = false}: (i32, i32, i8) -> i32
// %8 = arith.maxsi %7, %c-128_i32 : i32
// %9 = arith.minsi %8, %c127_i32 : i32
// %10 = arith.trunci %9 : i32 to i8

// i8 -> ui8
// %5 = arith.extsi %in : i8 to i32
// %6 = arith.subi %5, %c-70_i32 : i32
// %7 = tosa.apply_scale %6, %c1073741824_i32, %c30_i8 {double_round = false}: (i32, i32, i8) -> i32
// %8 = arith.addi %7, %c58_i32 : i32
// %9 = arith.maxsi %8, %c0_i32 : i32
// %10 = arith.minsi %9, %c255_i32 : i32
// %11 = arith.trunci %10 : i32 to i8
// linalg.yield %11 : i8

// i8 -> i16, there is input_zp, output_zp is 0, extract ouputMin/outputMax
// %4 = linalg.generic {...} ins(%2 : tensor<16384xi8>) outs(%3 : tensor<16384xi16>) {
//   ^bb0(%in: i8, %out: i16):
//     %5 = arith.extsi %in : i8 to i32
//     %6 = arith.subi %5, %c-128_i32 : i32
//     %7 = tosa.apply_scale %6, %c1073741824_i32, %c30_i8 {double_round = false} : (i32, i32, i8)
//     -> i32 %8 = arith.maxsi %7, %c-32768_i32 : i32 %9 = arith.minsi %8, %c32767_i32 : i32 %10 =
//     arith.trunci %9 : i32 to i16 linalg.yield %10 : i16
//   } -> tensor<16384xi16>

// i8 -> i32, there is input_zp, output_zp is 0, i32 outputMin/outputMax
//   %4 = linalg.generic {...} ins(%2 : tensor<16384xi8>) outs(%3 : tensor<16384xi32>) {
//   ^bb0(%in: i8, %out: i32):
//     %5 = arith.extsi %in : i8 to i32
//     %6 = arith.subi %5, %c-128_i32 : i32
//     %7 = tosa.apply_scale %6, %c1073741824_i32, %c30_i8 {double_round = false} : (i32, i32, i8)
//     -> i32 linalg.yield %7 : i32
//   } -> tensor<16384xi32>

// i16 -> i8 and i32 -> i8, no input_zp, extract outputMin/outputMax and outputZp
// %4 = linalg.generic {...} ins(%2 : tensor<9408xi16>) outs(%3 : tensor<9408xi8>) {
//   ^bb0(%in: i16, %out: i8):
//     %5 = arith.extsi %in : i16 to i32 // no extsi for input i32
//     %6 = tosa.apply_scale %5, %c1106700928_i32, %c45_i8 {double_round = true} : (i32, i32, i8) ->
//     i32 %7 = arith.addi %6, %c125_i32 : i32 %8 = arith.maxsi %7, %c-128_i32 : i32 %9 =
//     arith.minsi %8, %c127_i32 : i32 %10 = arith.trunci %9 : i32 to i8 linalg.yield %10 : i8
//   } -> tensor<9408xi8>

// i16 -> i32
// %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
//                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
//                         iterator_types = ["parallel", "parallel", "parallel"]
//                      } ins(%10 : tensor<1x21x1024xi16>) outs(%11 : tensor<1x21x1024xi32>) {
//   ^bb0(%in: i16, %out: i32):
//     %63 = arith.extsi %in : i16 to i32
//     %64 = tosa.apply_scale %63, %c1073741824_i32, %c30_i8 {double_round = false}
//                          : (i32, i32, i8) -> i32
//     linalg.yield %64 : i32
//   } -> tensor<1x21x1024xi32>

// i32 -> i32
// %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
//                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
//                         iterator_types = ["parallel", "parallel", "parallel"]
//                       } ins(%23 : tensor<1x21x1024xi32>) outs(%11 : tensor<1x21x1024xi32>) {
//   ^bb0(%in: i32, %out: i32):
//     %63 = tosa.apply_scale %in, %c1758038019_i32, %c28_i8 {double_round = false}
//                          : (i32, i32, i8) -> i32
//     linalg.yield %63 : i32
//   } -> tensor<1x21x1024xi32>

struct RescaleOpConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult scalarProcessing(
        linalg::GenericOp srcOp, tosa::ApplyScaleOp applyScaleOp, PatternRewriter &rewriter
    ) const {

        auto outputType = cast<RankedTensorType>(srcOp.getResults()[0].getType());
        auto outputElementType = outputType.getElementType();

        ScaleInfo scaleInfo;

        auto output = create1DimTensorFromRescaleScalar(
            srcOp, applyScaleOp, scaleInfo, outputElementType, rewriter
        );

        if (output) {
            srcOp.getResults()[0].replaceAllUsesWith(output);
            rewriter.eraseOp(srcOp);
            return success();
        }

        return failure();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // TODO: this code could be reimplemented using foldForwardScaleClamp
        if (srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly one init tensor for RescaleOpConversion"
            );
        }

        const int dpsInputCount = srcOp.getNumDpsInputs();
        if (dpsInputCount > 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly one or zero(const) input tensor for RescaleOpConversion"
            );
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a linalg.yield terminator for RescaleOpConversion"
            );
        }

        auto yieldValues = yieldOp.getValues();
        if (yieldValues.size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly one yield value for RescaleOpConversion"
            );
        }

        int32_t outputMin = 0, outputMax = 0;
        int32_t outputZp = 0, input_zp = 0;

        tosa::ApplyScaleOp applyScaleOp;
        Value input;

        auto outputType = cast<RankedTensorType>(srcOp.getResults()[0].getType());
        auto outputElementType = outputType.getElementType();

        // scalar input case
        if (dpsInputCount == 0) {
            applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();
            if (!applyScaleOp) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected a defining operation for yield operand to be tosa.apply_scale"
                );
            }
            input = applyScaleOp.getValue();
            if (!input) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected a defining operation for apply_scale operand to be a value"
                );
            }

            return scalarProcessing(srcOp, applyScaleOp, rewriter);
        }

        input = srcOp.getInputs()[0];
        assert(input && "input is null");

        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto inputElementType = inputType.getElementType();

        if (inputElementType.isF32() || inputElementType.isBF16() || outputElementType.isF32() ||
            outputElementType.isBF16()) {
            return rewriter.notifyMatchFailure(
                srcOp, "Unsupported element fp type for RescaleOpConversion"
            );
        }

        if (inputElementType.isInteger() && outputElementType.isInteger(32)) {

            applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();

            outputMin = std::numeric_limits<int32_t>::min();
            outputMax = std::numeric_limits<int32_t>::max();
        }
        else {
            auto truncOp = yieldValues[0].getDefiningOp<arith::TruncIOp>();

            // Handle the case where CastI32MulPattern has converted rescale to i16 output.
            // In this case rescale ops feeding multiply ops are optimized to output i16 directly
            // (instead of i32), so there's no truncOp here.
            if (!truncOp && outputElementType.isInteger(16)) {
                applyScaleOp = yieldValues[0].getDefiningOp<tosa::ApplyScaleOp>();
                outputMin = std::numeric_limits<int32_t>::min();
                outputMax = std::numeric_limits<int32_t>::max();
            }
            else {
                if (!truncOp) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Expected a defining operation for yield operand to be arith.trunci"
                    );
                }
                auto minOp = truncOp.getIn().getDefiningOp<arith::MinSIOp>();
                if (!minOp) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Expected a defining operation for yield operand to be arith.minsi"
                    );
                }
                // The max constant is used in the min operation
                auto maybeMaxConst = getConstIntValue(minOp.getRhs());
                if (!maybeMaxConst) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "matching error minOp.getRhs() is not a constant!"
                    );
                }
                outputMax = *maybeMaxConst;

                auto maxOp = minOp.getLhs().getDefiningOp<arith::MaxSIOp>();
                if (!maxOp) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Expected a defining operation for yield operand to be arith.maxsi"
                    );
                }
                // The min constant is used in the max operation
                auto maybeMinConst = getConstIntValue(maxOp.getRhs());
                if (!maybeMinConst) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "matching error maxOp.getRhs() is not a constant!"
                    );
                }
                outputMin = *maybeMinConst;

                if (auto addOp = maxOp.getLhs().getDefiningOp<arith::AddIOp>()) {
                    auto maybeAddConst = getConstIntValue(addOp.getRhs());
                    if (!maybeAddConst) {
                        return rewriter.notifyMatchFailure(
                            srcOp, "matching error addOp.getRhs() is not a constant!"
                        );
                    }
                    outputZp = *maybeAddConst;
                    applyScaleOp = addOp.getLhs().getDefiningOp<tosa::ApplyScaleOp>();
                }
                else {
                    applyScaleOp = maxOp.getLhs().getDefiningOp<tosa::ApplyScaleOp>();
                }
            }
        }

        if (!applyScaleOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a defining operation for yield operand to be tosa.apply_scale"
            );
        }

        auto ms = getMultiplierAndShift(srcOp, applyScaleOp, 1);
        if (!ms) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to get multiplier and shift from apply_scale operation"
            );
        }

        // try to extract input_zp
        auto subOp = dyn_cast_or_null<arith::SubIOp>(applyScaleOp.getValue().getDefiningOp());
        if (subOp) {
            auto maybeInputZp = getConstIntValue(subOp.getRhs());
            if (maybeInputZp) {
                input_zp = *maybeInputZp;
            }
        }

        // FMA in hardware implements quantized rescale as:
        //   output = (((input_value - input_zp) * scale) >> shift) + outputZP
        // We model this as:
        //   ALU operation: input_value * weight, with weight = 1
        //   ACT operation: ((ALU_output + bias) * scale) >> shift + outputZP
        // where:
        //   bias  = -input_zp
        //   scale = multiplier
        //   shift = shiftFactor
        int32_t shiftFactor = ms.shift[0];
        int32_t bias_data = -input_zp;
        int8_t weight_data = 1;
        std::vector<int8_t> weights = {weight_data};
        const std::vector<int32_t> bias = {bias_data};
        const std::vector<int32_t> scale = {ms.multiplier[0]};

        LLVM_DEBUG({
            llvm::dbgs() << "rescale params : input_zp: " << input_zp << ", "
                         << "outputZp: " << outputZp << ", outputMin: " << outputMin << ", "
                         << "outputMax: " << outputMax << ", shiftFactor: " << shiftFactor << ", "
                         << "weight_data: " << weight_data << ", bias_data: " << bias_data << "\n";
        });

        auto fmaOp = torq_hl::FMAOp::create(
            rewriter, srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
            outputZp, outputMin, outputMax, shiftFactor,
            createI8Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{1}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );
        rewriter.replaceOp(srcOp, fmaOp.getOutput());

        return success();
    }
};

struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
    using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::PadOp padTensorOp, PatternRewriter &rewriter) const override {

        return static_cast<LogicalResult>(
            linalg::rewriteInDestinationPassingStyle(rewriter, padTensorOp)
        );
    }
};

} // namespace

struct GenericToBroadcastOpConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {

        // FIXME (iree-upgrade): see if we can use standard isaBroadcastOpInterface() here
        std::optional<SmallVector<int64_t>> equivalentToBroadcast =
            torqIsaBroadcastOpInterface(genericOp);
        if (!equivalentToBroadcast) {
            return failure();
        }
        auto input = genericOp.getDpsInputOperand(0)->get();
        auto dstTy = genericOp.getDpsInitOperand(0)->get().getType();
        auto dims = *equivalentToBroadcast;

        // HW lowering currently prefers a single broadcast dimension per op; decompose same-rank
        // multi-dim (incl. non-contiguous) broadcasts into chained single-dim broadcasts.
        if (dims.size() > 1) {
            auto inType = dyn_cast<RankedTensorType>(input.getType());
            auto outType = dyn_cast<RankedTensorType>(dstTy);
            if (inType && outType && inType.getRank() == outType.getRank()) {
                Value current = input;
                SmallVector<int64_t> currentShape(
                    inType.getShape().begin(), inType.getShape().end()
                );
                for (int64_t dim : dims) {
                    if (dim < 0 || dim >= outType.getRank()) {
                        return rewriter.notifyMatchFailure(
                            genericOp, "Invalid broadcast dimension for decomposition"
                        );
                    }
                    currentShape[dim] = outType.getShape()[dim];
                    auto midType = RankedTensorType::get(currentShape, outType.getElementType());
                    auto init = createInitTensor(genericOp, rewriter, midType);
                    current = rewriter
                                  .create<torq_hl::BroadcastOp>(
                                      genericOp.getLoc(), midType, init, SmallVector<int64_t>{dim},
                                      current
                                  )
                                  .getResult(0);
                }
                rewriter.replaceOp(genericOp, current);
                return success();
            }
        }

        auto op = torq_hl::BroadcastOp::create(
            rewriter, genericOp.getLoc(), dstTy,
            createInitTensor(genericOp, rewriter, mlir::cast<RankedTensorType>(dstTy)), dims, input
        );
        rewriter.replaceOp(genericOp, op.getResults());
        return success();
    }
};

// Input will always be in NCHW format
struct ResizeNearestNeighborOpConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {

        auto indexingMaps = genericOp.getIndexingMapsArray();
        if (indexingMaps.size() != 2) {
            return rewriter.notifyMatchFailure(
                genericOp, "Expected exactly 2 indexing maps for ResizeNearestNeighborOpConversion"
            );
        }
        auto inputMap = indexingMaps[0];
        if (inputMap.getNumResults() != 4) {
            return rewriter.notifyMatchFailure(
                genericOp,
                "Expected affine map with 4 results for ResizeNearestNeighborOpConversion"
            );
        }
        // NCHW
        auto dim0 = dyn_cast<AffineDimExpr>(inputMap.getResult(0));
        auto dim1 = dyn_cast<AffineDimExpr>(inputMap.getResult(1));
        auto dim2 = dyn_cast<AffineBinaryOpExpr>(inputMap.getResult(2));
        auto dim3 = dyn_cast<AffineBinaryOpExpr>(inputMap.getResult(3));
        if (!dim0 || !dim1 || !dim2 || !dim3) {
            return rewriter.notifyMatchFailure(
                genericOp, "Expected affine map with dim0, dim1 floordiv 2, dim2 floordiv 2, dim3 "
                           "for ResizeNearestNeighborOpConversion"
            );
        }
        // NCHW
        auto binaryExpr1 = dim2;
        auto binaryExpr2 = dim3;
        if (binaryExpr1.getKind() != AffineExprKind::FloorDiv ||
            binaryExpr2.getKind() != AffineExprKind::FloorDiv) {
            return rewriter.notifyMatchFailure(
                genericOp, "Expected affine map with dim1 and dim2 as floordiv for "
                           "ResizeNearestNeighborOpConversion"
            );
        }

        auto resizeType = genericOp.getResult(0).getType();
        Value resizeInput = genericOp.getDpsInputs()[0];
        Value resizeInit =
            createInitTensor(genericOp, rewriter, mlir::cast<RankedTensorType>(resizeType));
        auto resizeOp = rewriter.create<syna::torq_hl::ResizeNearestNeighborOp>(
            genericOp.getLoc(), resizeType, resizeInit, 2, resizeInput
        );

        rewriter.replaceOp(genericOp, resizeOp.getOutput());
        return success();
    }
};

void populateLinalgToTorqHLPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    if (markFuseGroups) {
        patterns.insert<TransposeOpConversionRewrite>(context, markFuseGroups);
        // NB: FillOpConversionRewrite must be the last markFuseGroups pattern (see comment inline
        // in the fuse case).
        patterns.insert<FillOpConversionRewrite>(context, markFuseGroups);
        return;
    }

    // IMPORTANT: Since sigmoid op contains exp in its body Exp pattern must be called after Sigmoid
    // pattern in order to make sure that exp pattern doesn't match sigmoid op, see:
    // https://mlir.llvm.org/docs/PatternRewriter/#walk-pattern-rewrite-driver
    // This works because we are using Greedy Pattern Rewrite Driver which applies patterns in the
    // order they are added to the pattern list.
    populateSigmoidPatterns(context, patterns);
    populateExpPatterns(context, patterns);
    populateSoftmaxPatterns(context, patterns);

    patterns.insert<TransposeOpConversion>(context);
    patterns.insert<FillOpConversion>(context);

    patterns.insert<TensorPadOpConversion>(context);

    // Make sure to add the new class to isTorqMatmulOp function if here
    // is changed
    patterns.insert<MatmulOpConversion<linalg::BatchMatmulOp>>(context);
    patterns.insert<MatmulOpConversion<linalg::DotOp>>(context);
    patterns.insert<MatmulOpConversion<linalg::MatvecOp>>(context);

    patterns.insert<ReduceOpConversion>(context);
    populateLinalgToTorqHLClampPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLMulPatterns(context, patterns, markFuseGroups);
    patterns.insert<CastOpPattern>(context);
    patterns.insert<BroadcastOpConversion>(context);

    patterns.insert<AbsOpPattern>(context);
    patterns.insert<NegateOpPattern>(context);
    patterns.insert<ClzOpPattern>(context);
    patterns.insert<CeilOpPattern>(context);
    patterns.insert<FloorOpPattern>(context);

    // FIXME (upgrade):
    // patterns.insert<RescaleOpConversion>(context);
    populateLinalgToTorqHLExtractPatterns(context, patterns, markFuseGroups);
    patterns.insert<AddOpPattern>(context);

    patterns.insert<ReinterpretCastOpPattern>(context);

    patterns.insert<GenericToBroadcastOpConversion>(context);
    patterns.insert<ResizeNearestNeighborOpConversion>(context);
}

} // namespace mlir::syna::torq
