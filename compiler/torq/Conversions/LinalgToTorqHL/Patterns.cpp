// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"

#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

    // NPU only supports BF16 and INT8 matmuls; F32 must fall back to host.
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        for (Value input : linalgOp.getDpsInputs()) {
            if (auto rankedType = dyn_cast<RankedTensorType>(input.getType())) {
                if (rankedType.getElementType().isF32()) {
                    failReason = "matmul with F32 inputs is not supported on NPU";
                    return false;
                }
            }
        }
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
    if (!isa_and_nonnull<arith::AddIOp>(reduceBodyOp) &&
        !isa_and_nonnull<arith::AddFOp>(reduceBodyOp)) {
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

        if (succeeded(foldForwardDepthToSpace(srcOp, rewriter, std::nullopt))) {
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
            if (auto attr = srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)) {
                maybeFuseGroupAttr = attr;
            }
        }

        if (srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 output");
        }

        if (succeeded(foldForwardDepthToSpace(srcOp, rewriter, maybeFuseGroupAttr))) {
            // foldForwardDepthToSpace called markOpFuseGroup
            return success();
        }

        if (_markFuseGroups) {
            markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr);
            return success();
        }

        // This pattern should only be called for doing the marking; for the
        // conversion TransposeOpConversion is called
        assert(false);
    }
};

// Helper: if `expr` is `dim[dimPos] floordiv C`, returns C. Otherwise nullopt.
static std::optional<int64_t> getDivConstant(AffineExpr expr, unsigned dimPos) {
    auto binOp = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binOp || binOp.getKind() != AffineExprKind::FloorDiv)
        return std::nullopt;
    auto dim = dyn_cast<AffineDimExpr>(binOp.getLHS());
    auto cst = dyn_cast<AffineConstantExpr>(binOp.getRHS());
    if (!dim || dim.getPosition() != dimPos || !cst)
        return std::nullopt;
    return cst.getValue();
}

// Matches the NCHW linalg.generic DepthToSpace pattern produced by
// FoldNchwDepthToSpacePattern (or FoldNhwcDepthToSpacePattern after transpose wrapping):
//
//   ins(%input : tensor<NxCxHxWxT>) outs(%out : tensor<NxC_outxH'xW'xT>)
//   indexing_maps = [
//     (d0,d1,d2,d3) -> (d0, ((d2 mod bH)*bW + d3 mod bW)*C_out + d1,
//                               d2 floordiv bH, d3 floordiv bW),
//     (d0,d1,d2,d3) -> (d0, d1, d2, d3)
//   ]
//   body: linalg.yield %in
//
// Converts to torq_hl::DepthToSpaceOp.
struct DepthToSpaceOpConversion : public OpConversionPattern<linalg::GenericOp> {

    DepthToSpaceOpConversion(MLIRContext *context) : OpConversionPattern(context) {
        setHasBoundedRewriteRecursion();
    }

    LogicalResult matchAndRewrite(
        linalg::GenericOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // 1. Shape/rank checks
        if (srcOp.getNumDpsInputs() != 1 || srcOp.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(srcOp, "Expected 1 input and 1 output");
        if (srcOp.getNumLoops() != 4 || srcOp.getNumParallelLoops() != 4)
            return rewriter.notifyMatchFailure(srcOp, "Expected 4 parallel loops");

        auto inputType = dyn_cast<RankedTensorType>(srcOp.getInputs()[0].getType());
        auto outputType = dyn_cast<RankedTensorType>(srcOp.getResult(0).getType());
        if (!inputType || !outputType || inputType.getRank() != 4 || outputType.getRank() != 4)
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D input and output tensors");

        // 2. Body must be a passthrough: linalg.yield %block_arg_0
        auto &body = srcOp.getRegion().front();
        auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1 ||
            yieldOp.getOperand(0) != body.getArgument(0))
            return rewriter.notifyMatchFailure(srcOp, "Body must be yield-only passthrough");

        // 3. Output indexing map must be identity
        auto maps = srcOp.getIndexingMapsArray();
        if (!maps[1].isIdentity())
            return rewriter.notifyMatchFailure(srcOp, "Output map must be identity");

        // 4. Parse block sizes from `d2 floordiv bH` and `d3 floordiv bW` in the input map
        //    Input map: (d0,d1,d2,d3) -> (d0, C_expr, d2 floordiv bH, d3 floordiv bW)
        AffineMap inputMap = maps[0];
        auto maybeBH = getDivConstant(inputMap.getResult(2), /*dimPos=*/2);
        auto maybeBW = getDivConstant(inputMap.getResult(3), /*dimPos=*/3);
        if (!maybeBH || !maybeBW)
            return rewriter.notifyMatchFailure(
                srcOp, "Input map results[2,3] are not (d2 floordiv bH, d3 floordiv bW)"
            );
        int64_t bH = *maybeBH;
        int64_t bW = *maybeBW;

        // 5. Verify dimension 0 maps to d0
        auto dim0 = dyn_cast<AffineDimExpr>(inputMap.getResult(0));
        if (!dim0 || dim0.getPosition() != 0)
            return rewriter.notifyMatchFailure(srcOp, "Input map result[0] must be d0");

        // 6. Verify DepthToSpace shape relationships
        auto inputShape = inputType.getShape();   // [N, C, H, W]
        auto outputShape = outputType.getShape(); // [N, C_out, H', W']
        int64_t C = inputShape[1];
        int64_t H = inputShape[2];
        int64_t W = inputShape[3];
        int64_t H_out = outputShape[2];
        int64_t W_out = outputShape[3];
        int64_t C_out = outputShape[1];

        if (H_out != H * bH || W_out != W * bW)
            return rewriter.notifyMatchFailure(
                srcOp, "Output spatial dims do not match block size"
            );

        if (C != bH * bW * C_out)
            return rewriter.notifyMatchFailure(srcOp, "DepthToSpace channel dimension mismatch");

        // square blocks.
        if (bH != bW)
            return rewriter.notifyMatchFailure(srcOp, "Non-square block size not supported");

        int64_t blockSize = bH;
        if (blockSize != 2)
            return rewriter.notifyMatchFailure(srcOp, "Depth2space only supports blockSize of 2");

        // 7. Determine DCR vs CRD by rebuilding the expected input maps and comparing.
        //
        //   DCR: C = ((d2 mod bH) * bW + (d3 mod bW)) * C_out + d1
        //   CRD: C = d1 * (bH * bW) + (d2 mod bH) * bW + (d3 mod bW)
        //
        //   Reconstruct both and pick the one that matches the actual map.
        MLIRContext *ctx = rewriter.getContext();
        auto d0 = getAffineDimExpr(0, ctx);
        auto d1 = getAffineDimExpr(1, ctx);
        auto d2 = getAffineDimExpr(2, ctx);
        auto d3 = getAffineDimExpr(3, ctx);

        // DCR: ((d2 mod bH)*bW + (d3 mod bW))*C_out + d1
        AffineExpr dcrC = (d2 % bH * bW + d3 % bW) * C_out + d1;
        AffineMap expectedDCR =
            AffineMap::get(4, 0, {d0, dcrC, d2.floorDiv(bH), d3.floorDiv(bW)}, ctx);

        // CRD: d1*(bH*bW) + (d2 mod bH)*bW + (d3 mod bW)
        AffineExpr crdC = d1 * (bH * bW) + d2 % bH * bW + d3 % bW;
        AffineMap expectedCRD =
            AffineMap::get(4, 0, {d0, crdC, d2.floorDiv(bH), d3.floorDiv(bW)}, ctx);

        torq_hl::DepthToSpaceModeEnum d2sMode;
        if (inputMap == expectedDCR) {
            d2sMode = torq_hl::DepthToSpaceModeEnum::DCR;
        }
        else if (inputMap == expectedCRD) {
            d2sMode = torq_hl::DepthToSpaceModeEnum::CRD;
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Input map does not match DCR or CRD DepthToSpace pattern"
            );
        }

        // 8. Only integer element types supported by hardware kernel
        auto elementType = inputType.getElementType();
        if (!elementType.isInteger())
            return rewriter.notifyMatchFailure(srcOp, "Only integer element types supported");

        int dtype_size = elementType.getIntOrFloatBitWidth() / 8;
        const auto wram_width = 32 / dtype_size;
        const auto num_inputs = 2;

        // 9. Emit torq_hl::DepthToSpaceOp
        auto d2sOp = torq_hl::DepthToSpaceOp::create(
            rewriter, srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
            blockSize, d2sMode,
            createI8Const(
                rewriter, srcOp, genD2SWeights(wram_width),
                llvm::ArrayRef<int64_t>{wram_width * num_inputs}
            ),
            srcOp.getInputs()[0]
        );

        rewriter.replaceOp(srcOp, d2sOp.getOutput());
        return success();
    }
};

// [#1508] A non-zero 32-bit fill cannot use the NSS pad-fill primitive: the
// pad_value register is 16 bits and is replicated across every 16-bit unit, so a
// 32-bit element ends up holding (lo16 << 16) | lo16 instead of the requested
// value. Realize such fills by broadcasting a single-element constant through the
// 32-bit data path (iram -> alu -> act) instead. This costs only a 4-byte
// constant rather than materializing a full-tensor constant.
static LogicalResult
fillViaBroadcast(PatternRewriter &rewriter, linalg::FillOp srcOp, TypedAttr valueAttr) {
    auto outTy = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
    SmallVector<int64_t> oneShape(outTy.getRank(), 1);
    auto inTy = RankedTensorType::get(oneShape, outTy.getElementType());
    auto denseAttr = DenseElementsAttr::get(inTy, ArrayRef<Attribute>{valueAttr});
    Value input = mlir::arith::ConstantOp::create(rewriter, srcOp.getLoc(), denseAttr);
    rewriter.replaceOpWithNewOp<torq_hl::BroadcastOp>(
        srcOp, outTy, createInitTensor(srcOp, rewriter, outTy), rewriter.getDenseI64ArrayAttr({}),
        input
    );
    return success();
}

struct FillOpConversion : public OpConversionPattern<linalg::FillOp> {

    FillOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        linalg::FillOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        if (srcOp.getInputs().size() != 1 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

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
                // Only 32-bit elements map to an NSS dtype the broadcast data path
                // can carry. Wider integers (e.g. i64) have no NSS representation,
                // so leave them to the host/CSS fallback instead of crashing in the
                // broadcast lowering.
                if (fillElementSize != 4)
                    return rewriter.notifyMatchFailure(srcOp, "Unsupported integer fill width");
                return fillViaBroadcast(rewriter, srcOp, valueAttr);
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
                    return fillViaBroadcast(rewriter, srcOp, valueAttr);
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

/// Map the arith op in a reduction body to the corresponding torq_hl reduce
/// kernel name. Shared by the linalg.generic and linalg.reduce conversions.
static FailureOr<std::string> getReduceOpName(Operation *reduceBodyOp) {
    if (isa<arith::AddIOp>(reduceBodyOp) || isa<arith::AddFOp>(reduceBodyOp)) {
        return std::string("reduce_sum");
    }
    if (isa<arith::MaxSIOp>(reduceBodyOp) || isa<arith::MaxUIOp>(reduceBodyOp) ||
        isa<arith::MaximumFOp>(reduceBodyOp)) {
        return std::string("reduce_max");
    }
    if (isa<arith::MinSIOp>(reduceBodyOp) || isa<arith::MinUIOp>(reduceBodyOp) ||
        isa<arith::MinimumFOp>(reduceBodyOp)) {
        return std::string("reduce_min");
    }
    if (isa<arith::OrIOp>(reduceBodyOp)) {
        return std::string("reduce_or");
    }
    if (isa<arith::AndIOp>(reduceBodyOp)) {
        return std::string("reduce_and");
    }
    if (isa<arith::XOrIOp>(reduceBodyOp)) {
        return std::string("reduce_xor");
    }
    if (isa<arith::MulFOp>(reduceBodyOp)) {
        return std::string("reduce_mul");
    }
    return failure();
}

/// Emit the torq_hl::ReduceOp with the standard fixed weights/shift. Callers
/// supply the scale_bias operand and output clamp (the linalg.reduce path may
/// override these for a folded per-channel conv bias).
template <typename SrcOpT>
static torq_hl::ReduceOp emitTorqReduce(
    PatternRewriter &rewriter, SrcOpT srcOp, RankedTensorType resultType, StringRef opName,
    int64_t axis, Value scaleBias, int32_t outputMin, int32_t outputMax, Value input
) {
    constexpr int shift_factor = 12;
    std::vector<int16_t> weights = {1, 1};
    Value weightsConst = createI16Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{2});
    return torq_hl::ReduceOp::create(
        rewriter, srcOp.getLoc(), resultType, createInitTensor(srcOp, rewriter, resultType),
        opName.str(), axis,
        /*output_zp*/ 0, outputMin, outputMax, shift_factor, weightsConst, scaleBias, input
    );
}

// A reduction generic may have one output (the common case) or two — a combined
// min+max reduce (DynamicQuantizeLinear) yields both bounds from a single pass
// over the input. Each output carries its own combine op, init and result type.
struct GenericReductionInfo {
    Value input;
    SmallVector<Value> inits;
    SmallVector<RankedTensorType> resultTypes;
    SmallVector<std::string> opNames;
    int64_t axis;
};

static FailureOr<GenericReductionInfo>
getGenericReductionInfo(linalg::GenericOp srcOp, PatternRewriter &rewriter) {
    if (srcOp.getNumReductionLoops() != 1) {
        return failure();
    }

    if (srcOp.getInputs().size() != 1) {
        return failure();
    }

    unsigned numInits = srcOp.getNumDpsInits();
    if (numInits < 1 || numInits > 2) {
        return failure();
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != numInits) {
        return failure();
    }

    SmallVector<unsigned> reductionDims;
    srcOp.getReductionDims(reductionDims);

    if (reductionDims.size() != 1) {
        return failure();
    }

    unsigned reductionLoopIdx = reductionDims[0];

    AffineMap inMap = srcOp.getIndexingMapsArray()[0];

    auto reductionAxis = inMap.getResultPosition(rewriter.getAffineDimExpr(reductionLoopIdx));

    if (!reductionAxis) {
        return failure();
    }

    GenericReductionInfo info;
    info.input = srcOp.getInputs()[0];
    info.axis = static_cast<int64_t>(*reductionAxis);
    for (unsigned i = 0; i < numInits; ++i) {
        auto reduceBodyOp = yieldOp.getOperand(i).getDefiningOp();
        if (!reduceBodyOp) {
            return failure();
        }
        auto opName = getReduceOpName(reduceBodyOp);
        if (failed(opName)) {
            return failure();
        }
        info.opNames.push_back(*opName);
        info.inits.push_back(srcOp.getDpsInitOperand(i)->get());
        info.resultTypes.push_back(cast<RankedTensorType>(srcOp.getResultTypes()[i]));
    }

    return info;
}

/// Combine a tiled reduction's per-tile partial with the running accumulator
/// (an scf.for iter_arg) using the reduction's own semantics: max/min fold with
/// an elementwise MAX/MIN, everything else (sum, ...) with an additive combine.
static FailureOr<Value> mergeReductionPartial(
    linalg::GenericOp srcOp, Value init, Value partial, StringRef opName, PatternRewriter &rewriter
) {
    if (opName == "reduce_max") {
        return makeElementWiseBinary(
            srcOp, rewriter, init, partial, torq_hl::ElementwiseOpEnum::MAXIMUM
        );
    }
    if (opName == "reduce_min") {
        return makeElementWiseBinary(
            srcOp, rewriter, init, partial, torq_hl::ElementwiseOpEnum::MINIMUM
        );
    }
    return addInitToResult(init, partial, rewriter);
}

static bool isScfAccumulator(Value value) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
        return false;
    return isa<scf::ForOp>(blockArg.getOwner()->getParentOp());
}

/// Lowers a single-reduction-loop `linalg.generic` to `torq_hl::ReduceOp`
struct GenericReductionConversion : public OpRewritePattern<linalg::GenericOp> {

    GenericReductionConversion(MLIRContext *context) : OpRewritePattern(context) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        auto info = getGenericReductionInfo(srcOp, rewriter);

        if (failed(info)) {
            return rewriter.notifyMatchFailure(srcOp, "not a supported reduction generic");
        }

        LLVM_DEBUG({
            llvm::dbgs() << "\nGenericReductionConversion: Reduce source op:\n";
            srcOp.dump();
        });

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        Value scaleBias = createI32Const(rewriter, srcOp, interleave(bias, scale));

        // Emit one torq_hl.reduce per output. For a combined min+max reduce both
        // reduces read the same `input`; block-local load dedup collapses that to
        // a single DMA per tile, so min and max share one streaming pass.
        SmallVector<Value> results;
        results.reserve(info->inits.size());
        for (auto [i, init] : llvm::enumerate(info->inits)) {
            auto reduceOp = emitTorqReduce(
                rewriter, srcOp, info->resultTypes[i], info->opNames[i], info->axis, scaleBias,
                /*outputMin*/ 0, /*outputMax*/ 0, info->input
            );
            Value result = reduceOp.getResult(0);

            // When tiled, the reduce produces a per-tile partial that must fold
            // into the scf.for accumulator with the reduction's own combine op.
            if (isScfAccumulator(init)) {
                FailureOr<Value> merged =
                    mergeReductionPartial(srcOp, init, result, info->opNames[i], rewriter);
                if (failed(merged)) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "failed to accumulate tiled reduction into scf iter_arg"
                    );
                }
                result = *merged;
            }
            results.push_back(result);
        }

        rewriter.replaceOp(srcOp, results);
        return success();
    }
};

/// Lowers a single-input, single-dimension `linalg.reduce` to `torq_hl::ReduceOp`
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

        auto reduceBodyOp = yieldOp.getOperand(0).getDefiningOp();
        auto opName = getReduceOpName(reduceBodyOp);
        if (failed(opName)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported reduction operation");
        }

        // reduceOp out_min and out_max is defined in kernel, here just initialize

        // TODO: materialize bias/scale weights for rescale precision
        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        // For bf16 reduce_sum with f32 output, preserve f32 type for better precision
        // If output is bf16, keep it as bf16 (user's choice for memory/performance)
        RankedTensorType resultType = srcOutputType;

        // A per-channel conv bias may have been folded into this reduce (see
        // FoldConvBiasIntoReducePattern). If so, materialize it as an fp32
        // scale_bias operand so the activation stage applies it per output channel
        // in fp32 (matching `round_bf16(sum_f32 + bias_f32)`). The fp32 scale_bias
        // element type also signals the HW kernel to take the biased path.
        int32_t outputMin = 0;
        int32_t outputMax = 0;
        Value scaleBias = createI32Const(rewriter, srcOp, interleave(bias, scale));
        if (auto biasAttr = srcOp->getAttrOfType<DenseFPElementsAttr>(kReducePerChannelBiasAttr)) {
            std::vector<float> biasVals;
            biasVals.reserve(biasAttr.getNumElements());
            for (auto v : biasAttr.getValues<APFloat>()) {
                biasVals.push_back(v.convertToFloat());
            }
            scaleBias = createConst(biasVals, rewriter, srcOp.getLoc());
            // fp32 rescaleClamp interprets clipMin/clipMax as float bit patterns.
            float minF = std::numeric_limits<float>::lowest();
            float maxF = std::numeric_limits<float>::max();
            outputMin = *reinterpret_cast<int32_t *>(&minF);
            outputMax = *reinterpret_cast<int32_t *>(&maxF);
        }

        auto reduceOp = emitTorqReduce(
            rewriter, srcOp, resultType, *opName, axis, scaleBias, outputMin, outputMax, input
        );
        rewriter.replaceOp(srcOp, reduceOp.getResult(0));

        return success();
    }
};

class AddOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    static std::optional<llvm::APFloat> fromConstScalar(arith::ConstantOp constOp) {
        if (!constOp) {
            return std::nullopt;
        }
        auto attr = constOp.getValue();
        if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
            return floatAttr.getValue();
        }
        auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr);
        if (!denseAttr) {
            return std::nullopt;
        }
        // Only treat as scalar if all dimensions are 1
        // e.g. tensor<1x1xbf16> is scalar, tensor<1x1000xbf16> is NOT
        if (!llvm::all_of(denseAttr.getType().getShape(), [](int64_t dim) { return dim == 1; })) {
            LLVM_DEBUG({
                llvm::dbgs() << "[fromConstScalar] Skipping non-scalar tensor: "
                             << denseAttr.getType() << "\n";
            });
            return std::nullopt;
        }
        return *denseAttr.getValues<llvm::APFloat>().begin();
    }

    static std::optional<llvm::APFloat> convertFloatToType(llvm::APFloat value, Type type) {
        auto floatType = dyn_cast<FloatType>(type);
        if (!floatType) {
            return std::nullopt;
        }
        bool losesInfo = false;
        value.convert(
            floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo
        );
        return value;
    }

    // Trace back to the defining arith.ConstantOp outside the linalg.generic.
    static arith::ConstantOp traceToConstOp(linalg::GenericOp srcOp, Value val) {
        if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
            return constOp;
        }
        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
            unsigned argIdx = blockArg.getArgNumber();
            auto inputs = srcOp.getDpsInputs();
            if (argIdx < inputs.size()) {
                Value input = inputs[argIdx];
                return input.getDefiningOp<arith::ConstantOp>();
            }
        }
        return nullptr;
    };

    // Trace back to a scalar float constant outside the linalg.generic.
    static std::optional<llvm::APFloat> traceToScalarFloat(linalg::GenericOp srcOp, Value val) {
        if (auto truncOp = val.getDefiningOp<arith::TruncFOp>()) {
            auto value = traceToScalarFloat(srcOp, truncOp.getIn());
            if (!value) {
                return std::nullopt;
            }
            return convertFloatToType(*value, truncOp.getOut().getType());
        }
        if (auto extOp = val.getDefiningOp<arith::ExtFOp>()) {
            auto value = traceToScalarFloat(srcOp, extOp.getIn());
            if (!value) {
                return std::nullopt;
            }
            return convertFloatToType(*value, extOp.getOut().getType());
        }
        // If direct defining op is a constant (scalar case), return it
        if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
            return fromConstScalar(constOp);
        }
        // If it's a BlockArgument, trace back to the linalg.generic input
        if (auto blockArg = dyn_cast<BlockArgument>(val)) {
            if (blockArg.getOwner() != srcOp.getBody()) {
                return std::nullopt;
            }
            unsigned argIdx = blockArg.getArgNumber();
            // Block args: first N are inputs, last M are outputs (inits)
            auto inputs = srcOp.getDpsInputs();
            if (argIdx < inputs.size()) {
                Value input = inputs[argIdx];
                return traceToScalarFloat(srcOp, input);
            }
        }
        return std::nullopt;
    };

    static Operation *getElementwiseAddSubOp(linalg::GenericOp op) {
        Operation *binaryOp = getElementwiseBinaryOp(op, true);
        if (binaryOp) {
            return binaryOp;
        }

        Value output = op.getResultTensors()[0];
        auto rank = cast<RankedTensorType>(output.getType()).getRank();
        if (rank > 0 && op.getNumLoops() < 1) {
            return nullptr;
        }
        if (op.getNumParallelLoops() != op.getNumLoops()) {
            return nullptr;
        }
        if (op.getNumDpsInputs() != 2 && op.getNumDpsInputs() != 1) {
            return nullptr;
        }
        if (op.getNumDpsInits() != 1) {
            return nullptr;
        }
        for (int i = 0; i < op.getNumDpsInputs(); i++) {
            if (!op.payloadUsesValueFromOperand(op.getDpsInputOperand(i))) {
                return nullptr;
            }
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1) {
            return nullptr;
        }

        binaryOp = yieldOp.getOperand(0).getDefiningOp();
        if (!binaryOp || !isa<arith::AddFOp, arith::SubFOp>(binaryOp) ||
            binaryOp->getNumOperands() != 2) {
            return nullptr;
        }

        auto isBlockArgOrScalarFloat = [&](Value value) {
            return isa<BlockArgument>(value) || traceToScalarFloat(op, value).has_value();
        };
        if (!isBlockArgOrScalarFloat(binaryOp->getOperand(0)) ||
            !isBlockArgOrScalarFloat(binaryOp->getOperand(1))) {
            return nullptr;
        }

        return binaryOp;
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

        auto data = traceToScalarFloat(srcOp, rhs);
        if (!data) {
            data = traceToScalarFloat(srcOp, lhs);
            if (data) {
                needReverse = true;
            }
        }
        if (data) {
            if (needReverse && numLinalgInputs == 2) {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[1];
            }
            else {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[0];
            }

            // When one operand is a scalar constant, we create a 1-element tensor for input1 to
            // preserve add op semantics. This scalar is incorporated into the bias term instead of
            // being used directly as a tensor input.
            auto elemType = mlir::cast<RankedTensorType>(input0.getType()).getElementType();
            auto scalar = convertFloatToType(*data, elemType);
            if (!scalar) {
                return rewriter.notifyMatchFailure(srcOp, "scalar constant must have float type");
            }
            newBias = scalar->convertToFloat();
            rhs_is_scalar = true;
            RankedTensorType constType = RankedTensorType::get({1}, elemType);
            DenseElementsAttr value = DenseElementsAttr::get(constType, *scalar);
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
        // When the constant stays scalar and is folded into bias, keep the old scalar encoding:
        // (-1 * %input) + cst. When the constant has already been broadcast into a non-scalar
        // tensor, keep the default [1, -1] weight ordering. Flipping the weights there computes
        // %input - cst after later broadcast/canonicalization.
        // (needReverse is set when the constant was the lhs of a subtraction)
        // Note: scale not supported for bf16 operations so we have to take this approach.
        if (needReverse && opName == "sub") {
            if (rhs_is_scalar) {
                weights[0] = llvm::APFloat(bf16, "-1.0");
                weights[1] = llvm::APFloat(bf16, "1.0");
            }
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

        Operation *binaryOp = getElementwiseAddSubOp(srcOp);
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

        if (outType.getElementType().isInteger(16) || outType.getElementType().isInteger(32)) {
            // one input must be constant, the other is the input tensor

            bool inputNeedReverse = false;
            auto lhs = binaryOp->getOperand(0);
            auto rhs = binaryOp->getOperand(1);

            auto constOp = traceToConstOp(srcOp, rhs);
            if (!constOp) {
                constOp = traceToConstOp(srcOp, lhs);
                if (constOp) {
                    input0 = input1;
                    inputNeedReverse = true;
                }
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

// Matches the SpaceToDepth-style segmentation chain:
//   %expanded = tensor.expand_shape %input [[0], [1], [2, 3], [4, 5]]
//   %transposed = linalg.transpose %expanded permutation = [0, 1, 3, 5, 2, 4]
//   %collapsed = tensor.collapse_shape %transposed [[0], [1, 2, 3], [4], [5]]
// and lowers it (when w_segments == 1) to:
//   %seg = torq_hl.segmentation(..., %input)
//   %reshape = tensor.reshape %seg -> collapsed result type
//
// Root is CollapseShapeOp so dialect-conversion replaceOp targets the matched op.
class SegmentationTransposeOpConversion : public OpRewritePattern<tensor::CollapseShapeOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    static bool hasSegmentationExpandReassociation(tensor::ExpandShapeOp expandOp) {
        auto reassociation = expandOp.getReassociationIndices();
        return reassociation.size() == 4 && reassociation[0].size() == 1 &&
               reassociation[0][0] == 0 && reassociation[1].size() == 1 &&
               reassociation[1][0] == 1 && reassociation[2].size() == 2 &&
               reassociation[2][0] == 2 && reassociation[2][1] == 3 &&
               reassociation[3].size() == 2 && reassociation[3][0] == 4 && reassociation[3][1] == 5;
    }

    static bool hasSegmentationCollapseReassociation(tensor::CollapseShapeOp collapseOp) {
        auto reassociation = collapseOp.getReassociationIndices();
        return reassociation.size() == 4 && reassociation[0].size() == 1 &&
               reassociation[0][0] == 0 && reassociation[1].size() == 3 &&
               reassociation[1][0] == 1 && reassociation[1][1] == 2 && reassociation[1][2] == 3 &&
               reassociation[2].size() == 1 && reassociation[2][0] == 4 &&
               reassociation[3].size() == 1 && reassociation[3][0] == 5;
    }

    static bool hasSegmentationPermutation(linalg::TransposeOp transposeOp) {
        return transposeOp.getPermutation() == ArrayRef<int64_t>{0, 1, 3, 5, 2, 4};
    }

    static Value staticTensorReshape(
        Value tensor, mlir::ArrayRef<int64_t> shape, PatternRewriter &rewriter, const Location &loc
    ) {
        auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
        auto newType = RankedTensorType::get(shape, tensorType.getElementType());
        auto shapeType = RankedTensorType::get({(int)shape.size()}, rewriter.getIndexType());
        auto shapeAttr = DenseIntElementsAttr::get(shapeType, shape);
        Value shapeConst = arith::ConstantOp::create(rewriter, loc, shapeAttr);
        return tensor::ReshapeOp::create(rewriter, loc, newType, tensor, shapeConst).getResult();
    }

    LogicalResult
    matchAndRewrite(tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {
        if (!hasSegmentationCollapseReassociation(collapseOp)) {
            return rewriter.notifyMatchFailure(collapseOp, "Not a segmentation collapse_shape");
        }

        auto transposeOp = collapseOp.getSrc().getDefiningOp<linalg::TransposeOp>();
        if (!transposeOp || !hasSegmentationPermutation(transposeOp)) {
            return rewriter.notifyMatchFailure(collapseOp, "Expected segmentation transpose");
        }

        Value expandedInput = transposeOp.getInput();
        auto expandOp = expandedInput.getDefiningOp<tensor::ExpandShapeOp>();
        if (!expandOp || !hasSegmentationExpandReassociation(expandOp)) {
            return rewriter.notifyMatchFailure(
                collapseOp, "Expected segmentation-style tensor.expand_shape"
            );
        }

        auto inputType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
        auto expandedType = dyn_cast<RankedTensorType>(expandOp.getResultType());
        auto outputType = dyn_cast<RankedTensorType>(collapseOp.getResultType());
        if (!inputType || !expandedType || !outputType || !inputType.hasStaticShape() ||
            !expandedType.hasStaticShape() || !outputType.hasStaticShape() ||
            inputType.getRank() != 4 || expandedType.getRank() != 6) {
            return rewriter.notifyMatchFailure(
                collapseOp, "Expected static segmentation expand/transpose layout"
            );
        }

        int64_t hSegments = expandedType.getShape()[3];
        int64_t wSegments = expandedType.getShape()[5];
        // TODO: we can support the wSegments > 1, we need to add support in the kernel code
        if (hSegments <= 0 || wSegments > 1) {
            return rewriter.notifyMatchFailure(collapseOp, "Invalid segmentation factors");
        }

        auto dummyWeights = createI8Const(
            rewriter, collapseOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );
        auto dummyScaleBias =
            createIConst(rewriter, collapseOp, std::vector<APInt>{APInt(32, 0), APInt(32, 1)});

        auto segmentationOp = torq_hl::SegmentationOp::create(
            rewriter, expandOp.getLoc(), inputType, createInitTensor(expandOp, rewriter, inputType),
            rewriter.getI32IntegerAttr(hSegments), rewriter.getI32IntegerAttr(wSegments),
            dummyWeights.getResult(), dummyScaleBias.getResult(), expandOp.getSrc()
        );
        auto output = staticTensorReshape(
            segmentationOp.getOutput(), outputType.getShape(), rewriter, collapseOp.getLoc()
        );
        rewriter.replaceOp(collapseOp, output);
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

// A per-channel rescale carries its multiplier as a channel-indexed operand, so the number of
// distinct scale values is that operand's element count. A per-tensor rescale folds a scalar
// multiplier into the generic body, leaving the count at 1.
static int getRescaleScaleCount(linalg::GenericOp genericOp, tosa::ApplyScaleOp applyScaleOp) {
    if (auto multArg = dyn_cast<BlockArgument>(applyScaleOp.getMultiplier())) {
        if (auto operand = genericOp.getMatchingOpOperand(multArg)) {
            if (auto multType = dyn_cast<RankedTensorType>(operand->get().getType())) {
                return multType.getNumElements();
            }
        }
    }
    return 1;
}

static SmallVector<int64_t> invertPermutation(llvm::ArrayRef<int64_t> perm) {
    SmallVector<int64_t> inv(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv[perm[i]] = static_cast<int64_t>(i);
    }
    return inv;
}

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
            rewriter.replaceOp(srcOp, output);
            return success();
        }

        return failure();
    }

    // Lower a matched integer rescale to a single torq_hl.fma. The hardware ACT applies one shift
    // with per-channel multipliers, so normalize the TOSA shifts to a common shift (the minimum)
    // and rescale each multiplier to it, mirroring the fused conv/matmul path; for a per-tensor
    // rescale (one scale value) this is just the multiplier and shift as-is. scale_bias is the 1-D
    // [2] pair for per-tensor, or a [channels, 2] table for per-channel (which FMAPattern loads per
    // channel block).
    LogicalResult emitRescale(
        linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input,
        RankedTensorType outputType, const MultiplierShiftInfo &ms, int scaleValuesCount,
        int32_t inputZp, int32_t outputZp, int32_t outputMin, int32_t outputMax,
        tosa::ApplyScaleOp applyScaleOp
    ) const {
        int32_t shiftFactor = *llvm::min_element(ms.shift);
        SmallVector<int32_t> bias(scaleValuesCount, -inputZp);
        int8_t weightData = 1;
        std::vector<int32_t> scale;

        // A negative shift is a scale-up (left shift by -shiftFactor). The ACT right-shift
        // (act_rsh) is an unsigned 6-bit hardware field and cannot go negative, so fold the left
        // shift into the ALU weight and bias, which are both applied before the ACT shift: with
        // weight = 2^k and bias = -inputZp * 2^k the FMA computes
        //   ((input - inputZp) * 2^k * multiplier) >> 0
        // i.e. the same value as ((input - inputZp) * multiplier) >> (-k). Only a per-tensor
        // rescale can be folded this way (a per-channel scale-up would need a per-channel weight,
        // which the FMA lacks); the weight is int8, so k <= 6 (2^7 overflows int8).
        if (shiftFactor < 0) {
            if (scaleValuesCount != 1) {
                return rewriter.notifyMatchFailure(
                    srcOp, "negative per-channel rescale shift cannot fold into int8 weight"
                );
            }
            int leftShift = -shiftFactor;
            if (leftShift > 6) {
                return rewriter.notifyMatchFailure(
                    srcOp, "rescale scale-up shift too large to fold into int8 weight"
                );
            }
            weightData = static_cast<int8_t>(1 << leftShift);
            bias[0] = -inputZp * (1 << leftShift);
            shiftFactor = 0;
            scale = {ms.multiplier[0]};
        }
        else {
            scale = compute_scale(ms.multiplier, ms.shift, shiftFactor);
        }

        SmallVector<int32_t> scaleBias(2 * scaleValuesCount);
        interleave_into(
            llvm::ArrayRef<int32_t>(bias), llvm::ArrayRef<int32_t>(scale), scaleBias.begin()
        );

        // Per-channel scale_bias is a [channels, 2] table and needs the channel on the innermost
        // dimension; per-tensor is the single 1-D [2] pair broadcast to every element.
        // If the channel dimension is not already innermost, transpose input/output so the FMA
        // sees a channel-innermost layout and transpose the result back.
        int64_t channelDim = outputType.getRank() - 1;
        bool needsTranspose = false;
        SmallVector<int64_t> perm;
        if (scaleValuesCount > 1) {
            if (auto multArg = dyn_cast<BlockArgument>(applyScaleOp.getMultiplier())) {
                auto indexingMaps = srcOp.getIndexingMapsArray();
                if (multArg.getArgNumber() < static_cast<int>(indexingMaps.size())) {
                    auto multMap = indexingMaps[multArg.getArgNumber()];
                    if (multMap.getNumResults() == 1) {
                        if (auto dimExpr = dyn_cast<AffineDimExpr>(multMap.getResult(0))) {
                            channelDim = dimExpr.getPosition();
                        }
                    }
                }
            }
            if (outputType.getDimSize(channelDim) != scaleValuesCount) {
                return rewriter.notifyMatchFailure(
                    srcOp, "per-channel rescale scale count does not match any dimension"
                );
            }
            needsTranspose = channelDim != outputType.getRank() - 1;
            if (needsTranspose) {
                for (int64_t i = 0; i < outputType.getRank(); ++i) {
                    if (i != channelDim)
                        perm.push_back(i);
                }
                perm.push_back(channelDim);
            }
        }

        Value scaleBiasConst;
        if (scaleValuesCount > 1) {
            scaleBiasConst = createI32Const(
                rewriter, srcOp, scaleBias, llvm::ArrayRef<int64_t>{scaleValuesCount, 2}
            );
        }
        else {
            scaleBiasConst = createI32Const(rewriter, srcOp, scaleBias);
        }

        Value weightConst = createI8Const(
            rewriter, srcOp, llvm::ArrayRef<int8_t>{weightData}, llvm::ArrayRef<int64_t>{1}
        );

        if (needsTranspose) {
            Value transposedInput = transposeValue(input, perm, srcOp.getLoc(), rewriter);
            RankedTensorType transposedOutputType = transposeType(outputType, perm);
            Value transposedInit = createInitTensor(srcOp, rewriter, transposedOutputType);
            auto fmaOp = torq_hl::FMAOp::create(
                rewriter, srcOp.getLoc(), transposedOutputType, transposedInit, outputZp, outputMin,
                outputMax, shiftFactor, weightConst, scaleBiasConst, transposedInput
            );
            Value output = transposeValue(
                fmaOp.getOutput(), invertPermutation(perm), srcOp.getLoc(), rewriter
            );
            rewriter.replaceOp(srcOp, output);
            return success();
        }

        auto fmaOp = torq_hl::FMAOp::create(
            rewriter, srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
            outputZp, outputMin, outputMax, shiftFactor, weightConst, scaleBiasConst, input
        );
        rewriter.replaceOp(srcOp, fmaOp.getOutput());
        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        // TODO: this code could be reimplemented using foldForwardScaleClamp
        if (srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly one init tensor for RescaleOpConversion"
            );
        }

        // A per-tensor rescale has a single data input; the multiplier/shift are scalar
        // constants folded into the body. A per-channel rescale additionally carries its
        // multiplier and/or shift as channel-indexed operands (up to two extra inputs), which
        // getMultiplierAndShift resolves via the apply_scale block arguments below.
        const int dpsInputCount = srcOp.getNumDpsInputs();
        if (dpsInputCount > 3) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected at most a data input plus per-channel multiplier/shift for "
                       "RescaleOpConversion"
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

        // Constant/scalar input: fold the rescale at compile time
        if (dpsInputCount == 0 ||
            (dpsInputCount == 1 && srcOp.getInputs()[0].getDefiningOp<arith::ConstantOp>())) {
            tosa::ApplyScaleOp constApplyScaleOp;
            srcOp.getBody()->walk([&](tosa::ApplyScaleOp op) { constApplyScaleOp = op; });
            if (constApplyScaleOp) {
                auto result = scalarProcessing(srcOp, constApplyScaleOp, rewriter);
                if (succeeded(result))
                    return result;
            }
            if (dpsInputCount == 0) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Failed to fold scalar rescale with no inputs"
                );
            }
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

            if (!truncOp) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected a defining operation for yield operand to be arith.trunci"
                );
            }
            auto minOp = truncOp.getIn().getDefiningOp<arith::MinSIOp>();
            if (minOp) {
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
            else {
                applyScaleOp = truncOp.getIn().getDefiningOp<tosa::ApplyScaleOp>();
                outputMin = std::numeric_limits<int32_t>::min();
                outputMax = std::numeric_limits<int32_t>::max();
            }
        }

        if (!applyScaleOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected a defining operation for yield operand to be tosa.apply_scale"
            );
        }

        int scaleValuesCount = getRescaleScaleCount(srcOp, applyScaleOp);

        auto ms = getMultiplierAndShift(srcOp, applyScaleOp, scaleValuesCount);
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

        LLVM_DEBUG({
            llvm::dbgs() << "rescale params : input_zp: " << input_zp << ", outputZp: " << outputZp
                         << ", outputMin: " << outputMin << ", outputMax: " << outputMax
                         << ", channels: " << scaleValuesCount << "\n";
        });

        return emitRescale(
            srcOp, rewriter, input, outputType, ms, scaleValuesCount, input_zp, outputZp, outputMin,
            outputMax, applyScaleOp
        );
    }
};

struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
    using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

    TensorPadOpConversion(MLIRContext *context) : OpRewritePattern(context) {}

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

        std::optional<SmallVector<int64_t>> equivalentToBroadcast =
            linalg::isaBroadcastOpInterface(genericOp);
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
                    current = torq_hl::BroadcastOp::create(
                                  rewriter, genericOp.getLoc(), midType, init,
                                  SmallVector<int64_t>{dim}, current
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

// [#1760] Decomposition of a permute+broadcast copy `linalg.generic` into a
// `torq_hl.transpose` followed by a `torq_hl.broadcast`.
struct TransposeBroadcastInfo {
    SmallVector<int64_t> perm;      // input dims reordered into ascending output order
    SmallVector<int64_t> bcastDims; // output dims to insert (empty => pure permutation)
};

// [#1760] Recognize a copy-only `linalg.generic` whose input indexing map both
// *permutes* its kept axes and *drops* others (a broadcast). Such ops arise from
// NHWC->NCHW layout conversion of `tosa.add` broadcast operands (e.g. input map
// (d0,d2,d1), identity output) and match neither `GenericToBroadcastOpConversion`
// (isaBroadcastOpInterface requires monotonically-increasing kept dims, i.e. no
// permutation) nor `TransposeOpConversion` (only the named `linalg.transpose`). On
// a match, return the permutation and broadcast dims that realize it as
// transpose+broadcast; a pure broadcast (monotonic projection) is left to
// GenericToBroadcastOpConversion.
static FailureOr<TransposeBroadcastInfo> matchTransposeBroadcastGeneric(linalg::GenericOp genericOp
) {
    // Structural: a copy body (yield of the input element), all-parallel,
    // single input/output, identity output map.
    if (!genericOp.isAllParallelLoops() || !genericOp.isSingleInputOutput() ||
        !genericOp.isSingleYieldOp()) {
        return failure();
    }
    auto maps = genericOp.getIndexingMapsArray();
    if (maps.size() != 2 || !maps[1].isIdentity()) {
        return failure();
    }
    Block *body = genericOp.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->back());
    if (!yieldOp || yieldOp.getNumOperands() != 1 ||
        yieldOp.getOperand(0) != body->getArgument(0)) {
        return failure();
    }

    // The input map must be a projected permutation: each result a distinct dim
    // id, with no symbols or compound exprs (isProjectedPermutation covers this).
    AffineMap srcMap = maps[0];
    int64_t outRank = maps[1].getNumResults();
    if (!srcMap.isProjectedPermutation()) {
        return failure();
    }
    SmallVector<int64_t> pos; // pos[j] = output dim fed by input dim j
    for (AffineExpr e : srcMap.getResults()) {
        pos.push_back(cast<AffineDimExpr>(e).getPosition());
    }
    // A monotonically-increasing projection is a pure broadcast, already handled
    // by GenericToBroadcastOpConversion (its dims are distinct, so is_sorted means
    // strictly increasing). Only act on a permutation.
    if (llvm::is_sorted(pos)) {
        return failure();
    }
    if (!isa<RankedTensorType>(genericOp.getDpsInputOperand(0)->get().getType()) ||
        !isa<RankedTensorType>(genericOp.getDpsInitOperand(0)->get().getType())) {
        return failure();
    }

    // kept = sorted output dims fed by the input; the rest are broadcast.
    SmallVector<int64_t> kept(pos.begin(), pos.end());
    llvm::sort(kept);
    TransposeBroadcastInfo info;
    for (int64_t d = 0; d < outRank; ++d) {
        if (!llvm::is_contained(kept, d)) {
            info.bcastDims.push_back(d);
        }
    }
    // Transpose permutation: order input dims so their kept output targets
    // ascend. Rank each input dim by its target's position among the sorted kept
    // dims (yielding a permutation of [0, inputRank)), then invert it so that
    // perm[k] is the input dim feeding the k-th kept output dim.
    Permutation rankByTarget(pos.size());
    for (size_t j = 0; j < pos.size(); ++j) {
        rankByTarget[j] = llvm::lower_bound(kept, pos[j]) - kept.begin();
    }
    info.perm = rankByTarget.reverse();
    return info;
}

// [#1760] Realize a permute+broadcast copy `linalg.generic` (see
// matchTransposeBroadcastGeneric) by splitting it into a `torq_hl.transpose`
// (kept dims into ascending output order) followed by a `torq_hl.broadcast`
// (insert the dropped dims).
struct GenericToTransposeBroadcastOpConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        auto info = matchTransposeBroadcastGeneric(genericOp);
        if (failed(info)) {
            return rewriter.notifyMatchFailure(genericOp, "not a permute+broadcast copy generic");
        }

        Location loc = genericOp.getLoc();
        Value src = genericOp.getDpsInputOperand(0)->get();
        auto inType = cast<RankedTensorType>(src.getType());
        auto outType = cast<RankedTensorType>(genericOp.getDpsInitOperand(0)->get().getType());

        // Transpose the kept dims into ascending output order.
        SmallVector<int64_t> tShape(info->perm.size());
        for (size_t k = 0; k < info->perm.size(); ++k) {
            tShape[k] = inType.getShape()[info->perm[k]];
        }
        auto tType = RankedTensorType::get(tShape, inType.getElementType());
        Value transposed =
            torq_hl::TransposeOp::create(
                rewriter, loc, tType, createInitTensor(genericOp, rewriter, tType), info->perm, src
            )
                .getOutput();

        if (info->bcastDims.empty()) {
            // Pure permutation (no dropped dim): the transpose is the result.
            rewriter.replaceOp(genericOp, transposed);
            return success();
        }
        auto bcast = torq_hl::BroadcastOp::create(
            rewriter, loc, outType, createInitTensor(genericOp, rewriter, outType), info->bcastDims,
            transposed
        );
        rewriter.replaceOp(genericOp, bcast.getResults());
        return success();
    }
};

// Input will always be in NCHW format
// Walk the backward slice of `v` within a linalg body, collecting which
// linalg.index loop dims it depends on and whether a math.floor is on the path.
// Used to recognise a nearest-neighbour gather: the N/C indices are copied
// straight from their loop dim while the H/W indices are floor(loopdim / scale).
static void analyzeGatherIndex(
    Value v, llvm::SmallDenseSet<int64_t> &dims, bool &hasFloor,
    llvm::SmallPtrSetImpl<Operation *> &visited
) {
    Operation *def = v.getDefiningOp();
    if (!def) {
        return; // block argument or externally-defined constant
    }
    if (auto idxOp = dyn_cast<linalg::IndexOp>(def)) {
        dims.insert(idxOp.getDim());
        return;
    }
    if (isa<math::FloorOp>(def)) {
        hasFloor = true;
    }
    if (!visited.insert(def).second) {
        return;
    }
    for (Value operand : def->getOperands()) {
        analyzeGatherIndex(operand, dims, hasFloor, visited);
    }
}

// Raises the nearest-neighbour resize form emitted by the ONNX importer to a
// broadcast + collapse_shape (a tileable op that runs on NSS) instead of leaving
// it as a data-dependent gather that falls to a CSS host kernel (which then overflows
// the NSS instruction/data TCM). The importer emits an all-parallel
// linalg.generic whose body computes h_src = floor(h_out / scale) and
// w_src = floor(w_out / scale) and does a single tensor.extract from the input.
// By the time this pass runs the op is already inside a dispatch and the unit
// batch dim has been squeezed off the gather source, so the rank-4 NCHW output
// gathers from a rank-3 [C, H, W] source; we expand that back to rank 4 for the
// NSS op.
struct ResizeNearestNeighborGatherConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {

        if (genericOp.getNumResults() != 1) {
            return rewriter.notifyMatchFailure(genericOp, "expected a single result");
        }
        auto outTy = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());
        // Only rank-4 NCHW upsampling is supported by the NSS resize.
        if (!outTy || outTy.getRank() != 4) {
            return rewriter.notifyMatchFailure(genericOp, "expected a rank-4 result");
        }
        if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
            return rewriter.notifyMatchFailure(genericOp, "expected all-parallel loops");
        }

        Block &body = genericOp.getRegion().front();

        // The body must be a pure gather: exactly one tensor.extract and the
        // yield must return its result directly (an interpolating resize such as
        // bilinear has several extracts feeding a weighted sum).
        tensor::ExtractOp extractOp;
        int extractCount = 0;
        for (Operation &op : body.without_terminator()) {
            if (auto e = dyn_cast<tensor::ExtractOp>(op)) {
                extractOp = e;
                extractCount++;
            }
        }
        if (extractCount != 1 || !extractOp) {
            return rewriter.notifyMatchFailure(genericOp, "expected exactly one tensor.extract");
        }
        auto yieldOp = cast<linalg::YieldOp>(body.getTerminator());
        if (yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != extractOp.getResult()) {
            return rewriter.notifyMatchFailure(genericOp, "yield must return the extracted value");
        }

        // The gather source is the resize input: rank 3 (batch squeezed) or 4,
        // static, and defined outside the generic body.
        Value src = extractOp.getTensor();
        auto srcTy = dyn_cast<RankedTensorType>(src.getType());
        int64_t srcRank = srcTy ? srcTy.getRank() : 0;
        if (!srcTy || !srcTy.hasStaticShape() || srcRank < 3 || srcRank > 4) {
            return rewriter.notifyMatchFailure(
                genericOp, "expected a rank 3/4 static gather source"
            );
        }
        if (Operation *srcDef = src.getDefiningOp()) {
            if (genericOp->isProperAncestor(srcDef)) {
                return rewriter.notifyMatchFailure(genericOp, "gather source defined inside body");
            }
        }

        auto indices = extractOp.getIndices();
        if (static_cast<int64_t>(indices.size()) != srcRank) {
            return rewriter.notifyMatchFailure(genericOp, "extract indices must match source rank");
        }

        // Resolve the static output shape. Before dispatch formation the generic
        // result is dynamic (tensor<1x8x?x?>) and cast to a static shape by a
        // downstream tensor.cast; use that cast to recover the concrete extents.
        RankedTensorType staticOutTy;
        if (outTy.hasStaticShape()) {
            staticOutTy = outTy;
        }
        else {
            for (Operation *user : genericOp.getResult(0).getUsers()) {
                if (auto castOp = dyn_cast<tensor::CastOp>(user)) {
                    auto ct = dyn_cast<RankedTensorType>(castOp.getType());
                    if (ct && ct.hasStaticShape() && ct.getRank() == 4) {
                        staticOutTy = ct;
                        break;
                    }
                }
            }
        }
        if (!staticOutTy) {
            return rewriter.notifyMatchFailure(
                genericOp, "could not determine static output shape"
            );
        }

        auto depsOf = [](Value v, bool &hasFloor) {
            llvm::SmallDenseSet<int64_t> dims;
            llvm::SmallPtrSet<Operation *, 16> visited;
            analyzeGatherIndex(v, dims, hasFloor, visited);
            return dims;
        };
        auto subsetOf = [](const llvm::SmallDenseSet<int64_t> &s, int64_t k) {
            for (int64_t x : s) {
                if (x != k) {
                    return false;
                }
            }
            return true;
        };

        // The output rank is 4 (NCHW); the (4 - srcRank) leading output dims are
        // the squeezed unit dims and must have extent 1. Source dim i aligns with
        // output dim i + leadingUnit. The trailing two source dims (H, W) must be
        // floor-downscaled from their loop dim; the rest (batch/channel) copied.
        int64_t leadingUnit = 4 - srcRank;
        auto oShape = staticOutTy.getShape();
        auto inShape = srcTy.getShape();
        for (int64_t i = 0; i < leadingUnit; i++) {
            if (oShape[i] != 1) {
                return rewriter.notifyMatchFailure(genericOp, "squeezed leading dim is not unit");
            }
        }
        for (int64_t i = 0; i < srcRank; i++) {
            // Walk each extract index backwards: `deps` is the set of loop dims it is
            // computed from, `hasFloor` whether a math.floor lies on that path. A
            // spatial (H/W) index must be a floor-downscale of exactly its own output
            // dim; a batch/channel index must be an identity read of its own dim
            // (same extent, no floor -- a floor-downscale with unchanged extent would
            // silently read the wrong channel if treated as identity).
            int64_t outDim = i + leadingUnit;
            bool hasFloor = false;
            auto deps = depsOf(indices[i], hasFloor);
            bool isSpatial = (i >= srcRank - 2);
            if (isSpatial) {
                if (!subsetOf(deps, outDim) || !deps.count(outDim) || !hasFloor) {
                    return rewriter.notifyMatchFailure(
                        genericOp, "H/W index is not a floor downscale"
                    );
                }
            }
            else {
                if (hasFloor || !subsetOf(deps, outDim)) {
                    return rewriter.notifyMatchFailure(
                        genericOp, "batch/channel index not identity"
                    );
                }
                if (inShape[i] != oShape[outDim]) {
                    return rewriter.notifyMatchFailure(genericOp, "batch/channel extent changed");
                }
            }
        }

        int64_t srcH = inShape[srcRank - 2], srcW = inShape[srcRank - 1];
        int64_t outH = oShape[2], outW = oShape[3];
        if (srcH == 0 || srcW == 0 || outH % srcH != 0 || outW % srcW != 0) {
            return rewriter.notifyMatchFailure(genericOp, "output is not an integer upscale");
        }
        int64_t scaleH = outH / srcH, scaleW = outW / srcW;
        if (scaleH != scaleW || scaleH < 2) {
            return rewriter.notifyMatchFailure(
                genericOp, "expected a uniform integer upscale >= 2"
            );
        }

        // The NSS resize works on rank-4 tensors; re-expand a squeezed source.
        Value src4 = src;
        if (srcRank < 4) {
            SmallVector<int64_t> expandedShape(leadingUnit, 1);
            expandedShape.append(inShape.begin(), inShape.end());
            auto expandedTy = RankedTensorType::get(expandedShape, srcTy.getElementType());
            // group the extra leading unit dims with the first real source dim
            SmallVector<ReassociationIndices> reassoc;
            ReassociationIndices first;
            for (int64_t i = 0; i <= leadingUnit; i++) {
                first.push_back(i);
            }
            reassoc.push_back(first);
            for (int64_t i = leadingUnit + 1; i < 4; i++) {
                reassoc.push_back({i});
            }
            src4 = tensor::ExpandShapeOp::create(
                       rewriter, genericOp.getLoc(), expandedTy, src, reassoc
            )
                       .getResult();
        }

        // Express the integer nearest-neighbour upsample as a broadcast + reshape
        // rather than a monolithic torq_hl.ResizeNearestNeighbor. A broadcast is a
        // tileable linalg op (raised to torq_hl.broadcast, which runs on NSS and
        // works for any element type), so tile-and-fuse can split large resizes to
        // fit LRAM -- the resize op has no TilingInterface and would overflow LRAM
        // whole. For src [N,C,H,W] and scale s the output element [n,c,oh,ow] with
        // oh = h*s+hs, ow = w*s+ws equals src[n,c,h,w] (independent of hs,ws), so
        // broadcasting src into [N,C,H,s,W,s] and collapsing to [N,C,H*s,W*s] is
        // exactly floor-asymmetric nearest upsampling.
        Location loc = genericOp.getLoc();
        auto elemTy = srcTy.getElementType();
        int64_t nDim = oShape[0], cDim = oShape[1];
        SmallVector<int64_t> bcastShape = {nDim, cDim, srcH, scaleH, srcW, scaleW};
        auto bcastTy = RankedTensorType::get(bcastShape, elemTy);
        Value bcastInit = tensor::EmptyOp::create(rewriter, loc, bcastShape, elemTy).getResult();

        // input map (n,c,h,hs,w,ws) -> (n,c,h,w) drops the two scale dims; output
        // map is the 6-D identity. This is the projected-permutation form the
        // GenericToBroadcastOpConversion recognises.
        MLIRContext *ctx = rewriter.getContext();
        AffineMap inMap = AffineMap::get(
            6, 0,
            {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1),
             rewriter.getAffineDimExpr(2), rewriter.getAffineDimExpr(4)},
            ctx
        );
        AffineMap outMap = rewriter.getMultiDimIdentityMap(6);
        SmallVector<utils::IteratorType> iterators(6, utils::IteratorType::parallel);
        auto bcastOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{bcastTy}, ValueRange{src4}, ValueRange{bcastInit},
            ArrayRef<AffineMap>{inMap, outMap}, iterators,
            [](OpBuilder &b, Location l, ValueRange args) {
                linalg::YieldOp::create(b, l, args[0]);
            }
        );

        // collapse [N,C,H,s,W,s] -> [N,C,H*s,W*s]
        SmallVector<ReassociationIndices> collapseReassoc = {{0}, {1}, {2, 3}, {4, 5}};
        Value result = tensor::CollapseShapeOp::create(
                           rewriter, loc, staticOutTy, bcastOp.getResult(0), collapseReassoc
        )
                           .getResult();

        // Re-wrap to the generic's (possibly dynamic) result type so existing
        // users -- e.g. the downstream tensor.cast -- stay well-typed.
        if (staticOutTy != outTy) {
            result = tensor::CastOp::create(rewriter, loc, outTy, result).getResult();
        }
        rewriter.replaceOp(genericOp, result);
        return success();
    }
};

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
        auto resizeOp = syna::torq_hl::ResizeNearestNeighborOp::create(
            rewriter, genericOp.getLoc(), resizeType, resizeInit, 2, resizeInput
        );

        rewriter.replaceOp(genericOp, resizeOp.getOutput());
        return success();
    }
};

struct Im2ColOpConversion : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG({
            llvm::dbgs() << "\n[Im2ColOpConversion] Attempting to match:\n";
            genericOp.dump();
        });

        if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: expected 1 input and 1 init, got "
                             << genericOp.getNumDpsInputs() << " input(s) and "
                             << genericOp.getNumDpsInits() << " init(s)\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected exactly one input and one init tensor for Im2ColOpConversion"
            );
        }
        if (genericOp.getIndexingMapsArray().size() != 2) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: expected 2 indexing maps, got "
                             << genericOp.getIndexingMapsArray().size() << "\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected exactly 2 indexing maps for Im2ColOpConversion"
            );
        }
        auto body = genericOp.getBody();
        if (body->getOperations().size() > 1) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: body has "
                             << body->getOperations().size() << " ops, expected 1\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected exactly one operation in the body for Im2ColOpConversion"
            );
        }
        if (!isa<linalg::YieldOp>(body->getOperations().front())) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[Im2ColOpConversion] Rejected: body's only op is not linalg.yield\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp,
                "Expected linalg.yield as the only operation in the body for Im2ColOpConversion"
            );
        }

        auto inputMap = genericOp.getIndexingMapsArray()[0];
        auto outputMap = genericOp.getIndexingMapsArray()[1];

        LLVM_DEBUG({
            llvm::dbgs() << "[Im2ColOpConversion] Input map:  " << inputMap << "\n";
            llvm::dbgs() << "[Im2ColOpConversion] Output map: " << outputMap << "\n";
        });

        if (!outputMap.isIdentity()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: output map is not identity\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected output indexing map to be non-identity for Im2ColOpConversion"
            );
        }
        if (inputMap.getNumResults() != 2) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: input map has "
                             << inputMap.getNumResults() << " results, expected 2\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected input indexing map to have 2 results for Im2ColOpConversion"
            );
        }

        // Checking whether the input map has a format (d0, d1) -> (d0 floordiv k, d0 mod kernelSize
        // + d1 * stride)
        auto binaryExpr1 = dyn_cast<AffineBinaryOpExpr>(inputMap.getResult(0));
        auto binaryExpr2 = dyn_cast<AffineBinaryOpExpr>(inputMap.getResult(1));
        if (!binaryExpr1 || !binaryExpr2) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: input map results are not binary "
                             << "expressions, result(0) kind="
                             << (binaryExpr1 ? "valid" : "invalid")
                             << ", result(1) kind=" << (binaryExpr2 ? "valid" : "invalid") << "\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected input indexing map results to be binary expressions for "
                           "Im2ColOpConversion"
            );
        }
        if (binaryExpr1.getKind() != AffineExprKind::FloorDiv ||
            binaryExpr2.getKind() != AffineExprKind::Add) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: input map result(0) kind="
                             << static_cast<int>(binaryExpr1.getKind())
                             << " (expected FloorDiv), result(1) kind="
                             << static_cast<int>(binaryExpr2.getKind()) << " (expected Add)\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected input indexing map with format (d0, d1) -> (d0 floordiv k, d0 "
                           "mod kernelSize + d1 * stride) for Im2ColOpConversion"
            );
        }

        // Extracting the kernel size from the first binary expression
        auto kernelSizeExpr = binaryExpr1.getRHS();
        auto kernelSizeConst = cast<AffineConstantExpr>(kernelSizeExpr);
        int64_t kernelSize = kernelSizeConst.getValue();
        LLVM_DEBUG(
            llvm::dbgs() << "[Im2ColOpConversion] Extracted kernelSize=" << kernelSize << "\n"
        );

        auto expr2 = binaryExpr2.getLHS();
        auto modExpr = cast<AffineBinaryOpExpr>(expr2);
        if (modExpr.getKind() != AffineExprKind::Mod) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: LHS of Add is not Mod (kind="
                             << static_cast<int>(modExpr.getKind()) << ")\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected the left-hand side of the second binary expression to be a "
                           "mod operation for Im2ColOpConversion"
            );
        }
        auto modConst = cast<AffineConstantExpr>(modExpr.getRHS());
        int modValue = modConst.getValue();
        LLVM_DEBUG(llvm::dbgs() << "[Im2ColOpConversion] Extracted modValue=" << modValue << "\n");
        if (kernelSize != modValue) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: kernelSize(" << kernelSize
                             << ") != modValue(" << modValue << ")\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected kernel size to match the mod value for Im2ColOpConversion"
            );
        }

        auto strideExpr = dyn_cast<AffineBinaryOpExpr>(binaryExpr2.getRHS());
        if (strideExpr && strideExpr.getKind() != AffineExprKind::Mul) {
            LLVM_DEBUG(
                llvm::dbgs() << "[Im2ColOpConversion] Rejected: RHS of Add is not Mul (kind="
                             << static_cast<int>(strideExpr.getKind()) << ")\n"
            );
            return rewriter.notifyMatchFailure(
                genericOp, "Expected the right-hand side of the second binary expression to be a "
                           "multiplication for Im2ColOpConversion"
            );
        }

        int strideValue = 1;
        if (strideExpr) {
            auto strideConst = cast<AffineConstantExpr>(strideExpr.getRHS());
            strideValue = strideConst.getValue();
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[Im2ColOpConversion] Extracted stride=" << strideValue
                         << ", dilation=1, kernelWidth=" << kernelSize << "\n"
        );

        auto torqIm2ColOp = syna::torq_hl::Im2ColOp::create(
            rewriter, genericOp.getLoc(), genericOp.getResultTypes(),
            genericOp.getDpsInitOperand(0)->get(), genericOp.getDpsInputOperand(0)->get(),
            strideValue, 1, kernelSize
        );
        LLVM_DEBUG({
            llvm::dbgs() << "[Im2ColOpConversion] Successfully created torq_hl.im2col:\n";
            torqIm2ColOp.dump();
        });
        rewriter.replaceOp(genericOp, torqIm2ColOp.getResults());
        return success();
    }
};

void populateResizeNearestNeighborRaisingPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.insert<ResizeNearestNeighborGatherConversion>(context);
}

void populateLinalgToTorqHLPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // Place patterns that run only in the marking pass inside this if.
    if (markFuseGroups) {
        // This pattern does the marking for TransposeOpConversion
        patterns.insert<TransposeOpConversionRewrite>(context, markFuseGroups);
    }

    // Patterns that have a marking mode:
    populateLinalgToTorqHLMulPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLMatmulPatterns(context, patterns, markFuseGroups);
    populateLinalgToTorqHLQuantizePatterns(context, patterns, markFuseGroups);

    // IMPORTANT: patterns below this line should not involve more than a single operation! If they
    // do, tile and fuse will break them. To prevent that, the pattern should be refactored to take
    // markFuseGroups, and have a marking mode.
    if (markFuseGroups)
        return;

    // IMPORTANT: Since sigmoid op contains exp in its body Exp pattern must be called after Sigmoid
    // pattern in order to make sure that exp pattern doesn't match sigmoid op, see:
    // https://mlir.llvm.org/docs/PatternRewriter/#walk-pattern-rewrite-driver
    // This works because we are using Greedy Pattern Rewrite Driver which applies patterns in the
    // order they are added to the pattern list.
    populateSigmoidPatterns(context, patterns);
    populateExpPatterns(context, patterns);
    populateTanhPatterns(context, patterns);
    populateSoftmaxPatterns(context, patterns);
    populateGeluPatterns(context, patterns);
    populateTrigPatterns(context, patterns);

    patterns.insert<TransposeOpConversion>(context);
    patterns.insert<DepthToSpaceOpConversion>(context);
    patterns.insert<FillOpConversion>(context);

    patterns.insert<TensorPadOpConversion>(context);

    patterns.insert<GenericReductionConversion>(context);
    patterns.insert<ReduceOpConversion>(context);
    patterns.insert<CastOpPattern>(context);
    patterns.insert<BroadcastOpConversion>(context);

    patterns.insert<AbsOpPattern>(context);
    patterns.insert<NegateOpPattern>(context);
    patterns.insert<ClzOpPattern>(context);
    patterns.insert<CeilOpPattern>(context);
    patterns.insert<FloorOpPattern>(context);

    patterns.insert<RescaleOpConversion>(context);

    // FIXME: these patterns ignore the markFuseGroups argument! They use
    // computeArithConst, which is difficult to refactor to use markFuseGroups.
    // Maybe it should be separated from the pattern, and run before the marking.
    populateLinalgToTorqHLExtractPatterns(context, patterns, markFuseGroups);

    patterns.insert<AddOpPattern>(context);

    patterns.insert<ReinterpretCastOpPattern>(context);

    patterns.insert<GenericToBroadcastOpConversion>(context);
    patterns.insert<GenericToTransposeBroadcastOpConversion>(context);
    patterns.insert<SegmentationTransposeOpConversion>(context);
    patterns.insert<ResizeNearestNeighborOpConversion>(context);
    patterns.insert<Im2ColOpConversion>(context);
    populateLinalgToTorqHLExpandWeightsPatterns(context, patterns);
}

} // namespace mlir::syna::torq
