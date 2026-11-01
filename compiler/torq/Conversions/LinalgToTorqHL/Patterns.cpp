// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
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
        // Unsigned i1/i8 -> float lowers to an ActOp i2f with a 1-weight so the
        // ALU multiplies instead of sign-extending the source element (see
        // CastOpPattern and ActPattern). Wider unsigned casts are not supported by
        // slice HW and require host/CSS fallback.
        if (isa<arith::UIToFPOp>(castOp) &&
            (inputElementType.isInteger(1) || inputElementType.isInteger(8))) {
            castOp = dyn_cast_or_null<arith::UIToFPOp>(castOp);
            if (isUnsigned) {
                *isUnsigned = true;
            }
        }
        else {
            castOp = dyn_cast_or_null<arith::SIToFPOp>(castOp);
        }
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

// Longest reduction axis that ReduceOpConversion makes by folding contiguous
// reduced dims into one. On the NSS simulator an integer single-axis reduce
// drops odd elements past index 32768, so lengths above 32769 are wrong there
// (hardware is correct). Longer runs are reduced one axis at a time instead.
constexpr int64_t kMaxFoldedReduceExtent = 32768;

// Whether the sorted reduced dims `dims` are contiguous and short enough to be
// folded into one axis (see kMaxFoldedReduceExtent).
static bool canFoldReduceDims(ShapedType inputType, ArrayRef<int64_t> dims) {
    if (dims.back() - dims.front() + 1 != (int64_t)dims.size()) {
        return false;
    }
    int64_t foldedExtent = 1;
    for (int64_t d : dims) {
        int64_t size = inputType.getDimSize(d);
        // Checked before multiplying so the product cannot overflow.
        if (size < 0 || size > kMaxFoldedReduceExtent / foldedExtent) {
            return false;
        }
        foldedExtent *= size;
    }
    return true;
}

// Returns why the reduction dims of `op` cannot be lowered to torq_hl.reduce,
// or std::nullopt if they can. Multi-dimension reductions are supported unless
// they reduce a dynamic dim.
static std::optional<StringRef> getUnsupportedReduceDimsReason(linalg::ReduceOp op) {
    ArrayRef<int64_t> dims = op.getDimensions();
    if (dims.empty()) {
        return StringRef("expected at least one reduction dimension");
    }
    if (dims.size() == 1) {
        return std::nullopt;
    }
    auto inputType = cast<ShapedType>(op.getInputs()[0].getType());
    for (int64_t d : dims) {
        if (inputType.isDynamicDim(d)) {
            return StringRef("multi-dimension reduction over a dynamic dimension");
        }
    }
    return std::nullopt;
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
            if (!markOpFuseGroup(srcOp, rewriter, maybeFuseGroupAttr))
                return rewriter.notifyMatchFailure(srcOp, "could not mark fuse group");
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
            srcOp, srcResultType,
            createInitTensor(
                rewriter, srcOp.getLoc(), srcResultType, srcOp.getDpsInitOperand(0)->get()
            ),
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
    if (isa<arith::MaxSIOp>(reduceBodyOp) || isa<arith::MaximumFOp>(reduceBodyOp)) {
        return std::string("reduce_max");
    }
    if (isa<arith::MinSIOp>(reduceBodyOp) || isa<arith::MinimumFOp>(reduceBodyOp)) {
        return std::string("reduce_min");
    }
    // Unsigned max/min compare the input as unsigned (see ReducePattern).
    if (isa<arith::MaxUIOp>(reduceBodyOp)) {
        return std::string("reduce_umax");
    }
    if (isa<arith::MinUIOp>(reduceBodyOp)) {
        return std::string("reduce_umin");
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
    SmallVector<Operation *> bodyOps;
    int64_t axis;
};

static FailureOr<GenericReductionInfo> getGenericReductionInfo(linalg::GenericOp srcOp) {
    if (!srcOp.hasPureTensorSemantics()) {
        return failure();
    }
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

    auto indexingMaps = srcOp.getIndexingMapsArray();
    AffineMap inMap = indexingMaps[0];

    auto reductionAxis =
        inMap.getResultPosition(getAffineDimExpr(reductionLoopIdx, srcOp.getContext()));

    if (!reductionAxis) {
        return failure();
    }

    // The kernel reduces the input directly and preserves the order of the
    // remaining axes; it cannot implement broadcasts or output permutations.
    if (!inMap.isPermutation()) {
        return failure();
    }
    for (AffineMap outMap : llvm::drop_begin(indexingMaps)) {
        if (outMap != inMap.dropResult(*reductionAxis)) {
            return failure();
        }
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
        // A matching terminator alone is insufficient: sum(x*x), for example,
        // must not become sum(x). Only the bf16 -> f32 accumulator extension
        // can be discarded by the hardware reduction.
        Value input = srcOp.getBody()->getArgument(0);
        Value accumulator = srcOp.getBody()->getArgument(i + 1);
        Value lhs = reduceBodyOp->getOperand(0);
        Value rhs = reduceBodyOp->getOperand(1);
        if (lhs == accumulator)
            std::swap(lhs, rhs);
        if (auto ext = lhs.getDefiningOp<arith::ExtFOp>()) {
            if (ext.getIn().getType().isBF16() && ext.getType().isF32())
                lhs = ext.getIn();
        }
        if (lhs != input || rhs != accumulator) {
            return failure();
        }
        info.opNames.push_back(*opName);
        info.bodyOps.push_back(reduceBodyOp);
        info.inits.push_back(srcOp.getDpsInitOperand(i)->get());
        info.resultTypes.push_back(cast<RankedTensorType>(srcOp.getResultTypes()[i]));
    }

    return info;
}

/// Combine a reduce result with an init the kernel did not start from (the
/// scf.for accumulator of a tiled reduction, or a non-identity init) using the
/// reduction's own semantics: max/min fold with an elementwise MAX/MIN,
/// everything else (sum, ...) with an additive combine.
template <typename SrcOpT>
static FailureOr<Value> mergeReductionPartial(
    SrcOpT srcOp, Value init, Value partial, StringRef opName, PatternRewriter &rewriter
) {
    bool isMax = opName.ends_with("max");
    if (isMax || opName.ends_with("min")) {
        auto type = cast<RankedTensorType>(init.getType());
        auto opType =
            isMax ? torq_hl::ElementwiseOpEnum::MAXIMUM : torq_hl::ElementwiseOpEnum::MINIMUM;
        bool isUnsigned = opName.starts_with("reduce_u");
        return torq_hl::ElementWiseBinaryOp::create(
                   rewriter, srcOp.getLoc(), type, createInitTensor(srcOp, rewriter, type), opType,
                   init, partial, isUnsigned
        )
            .getOutput();
    }
    return addInitToResult(init, partial, rewriter);
}

static bool isScfAccumulator(Value value) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
        return false;
    return isa<scf::ForOp>(blockArg.getOwner()->getParentOp());
}

// The value a reduction init comes from, looking through slices, casts, reshapes and
// scf.forall shared outputs, which is how tile-and-fuse hands a tile its part of
// the init. Also looks through the loops of ReductionTilingPass: they write each
// slot of their buffer once, so a slot a tile reads still holds the loop init.
static Value getInitSource(Value value) {
    while (true) {
        if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
            value = slice.getSource();
            continue;
        }
        if (auto cast = value.getDefiningOp<tensor::CastOp>()) {
            value = cast.getSource();
            continue;
        }
        // A reshape keeps a splat (or tensor.empty) and its element type.
        if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
            value = collapse.getSrc();
            continue;
        }
        if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
            value = expand.getSrc();
            continue;
        }
        if (auto result = dyn_cast<OpResult>(value)) {
            auto forOp = dyn_cast<scf::ForOp>(result.getOwner());
            if (forOp && forOp->hasAttr(TORQ_REDUCTION_TILING_LOOP)) {
                value = forOp.getInitArgs()[result.getResultNumber()];
                continue;
            }
        }
        if (auto arg = dyn_cast<BlockArgument>(value)) {
            Operation *parent = arg.getOwner()->getParentOp();
            if (auto forall = dyn_cast<scf::ForallOp>(parent)) {
                if (OpOperand *tied = forall.getTiedOpOperand(arg)) {
                    value = tied->get();
                    continue;
                }
            }
            auto forOp = dyn_cast<scf::ForOp>(parent);
            if (forOp && forOp->hasAttr(TORQ_REDUCTION_TILING_LOOP)) {
                if (OpOperand *init = forOp.getTiedLoopInit(arg)) {
                    value = init->get();
                    continue;
                }
            }
        }
        return value;
    }
}

// The constant `value` is filled with, or null if it is not a known splat. A
// fill already lowered to torq_hl keeps only a bit pattern (torq_hl.fill) or a
// one-element constant (torq_hl.broadcast), so decode those with the tensor's
// element type.
static TypedAttr getInitSplatValue(Value value) {
    Type elementType = cast<ShapedType>(value.getType()).getElementType();
    if (auto fill = value.getDefiningOp<torq_hl::FillOp>()) {
        unsigned width = elementType.getIntOrFloatBitWidth();
        APInt bits = APInt(32, static_cast<uint32_t>(fill.getValue())).trunc(width);
        if (auto intType = dyn_cast<IntegerType>(elementType))
            return IntegerAttr::get(intType, bits);
        if (auto floatType = dyn_cast<FloatType>(elementType))
            return FloatAttr::get(floatType, APFloat(floatType.getFloatSemantics(), bits));
        return {};
    }
    // A float conversion (e.g. the bf16 to f32 init that ReductionTilingPass
    // gives a keepdims reduction it rewrites) keeps a splat, converted. It is a linalg.generic with
    // an arith.extf or arith.truncf body, or a torq_hl.act "f2f" once lowered.
    Value convertedFrom;
    if (auto act = value.getDefiningOp<torq_hl::ActOp>()) {
        if (act.getName() == "f2f")
            convertedFrom = act.getInput();
    }
    else if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
        Block &body = generic.getRegion().front();
        if (generic.getNumDpsInputs() == 1 && body.getOperations().size() == 2 &&
            isa<arith::ExtFOp, arith::TruncFOp>(body.front()) &&
            body.front().getOperand(0) == body.getArgument(0))
            convertedFrom = generic.getDpsInputOperand(0)->get();
    }
    if (convertedFrom) {
        auto floatType = dyn_cast<FloatType>(elementType);
        auto source =
            dyn_cast_if_present<FloatAttr>(getInitSplatValue(getInitSource(convertedFrom)));
        if (!floatType || !source)
            return {};
        return FloatAttr::get(floatType, *convertFloatToType(source.getValue(), floatType));
    }
    if (auto broadcast = value.getDefiningOp<torq_hl::BroadcastOp>())
        value = broadcast.getInput();
    auto attr = dyn_cast_if_present<TypedAttr>(getConstantSplatAttr(value));
    if (!attr || attr.getType() != elementType)
        return {};
    return attr;
}

// True if starting the accumulation of `bodyOp` from `value` changes nothing,
// i.e. `value` is a start value the reduce kernel may use instead (it accepts
// both zeros for addf, and the infinity or the largest finite value for
// maximumf / minimumf).
static bool isReduceIdentity(TypedAttr value, Operation *bodyOp) {
    if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
        const APInt &v = intAttr.getValue();
        if (isa<arith::AddIOp, arith::OrIOp, arith::XOrIOp, arith::MaxUIOp>(bodyOp))
            return v.isZero();
        if (isa<arith::MaxSIOp>(bodyOp))
            return v.isMinSignedValue();
        if (isa<arith::MinSIOp>(bodyOp))
            return v.isMaxSignedValue();
        if (isa<arith::MinUIOp, arith::AndIOp>(bodyOp))
            return v.isAllOnes();
        return false;
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
        const APFloat &v = floatAttr.getValue();
        if (isa<arith::AddFOp>(bodyOp))
            return v.isZero();
        if (isa<arith::MaximumFOp>(bodyOp))
            return v.isNegative() && (v.isInfinity() || v.isLargest());
        if (isa<arith::MinimumFOp>(bodyOp))
            return !v.isNegative() && (v.isInfinity() || v.isLargest());
        if (isa<arith::MulFOp>(bodyOp))
            return v.isExactlyValue(1.0);
        return false;
    }
    return false;
}

// Whether a non-identity init of this reduction can be folded into the reduce
// result. Max and min take any init tensor. A sum needs a constant init: an
// integer add saturates while arith.addi wraps, so only an i32 sum is folded
// (narrow sums are often near the type bounds), and a bf16 sum adds the init in
// the kernel's f32 accumulator (see emitReduceWithInit).
static bool canCombineInit(Operation *bodyOp, Type elementType, bool isConstant) {
    if (isa<arith::AddIOp>(bodyOp))
        return isConstant && elementType.isInteger(32);
    if (isa<arith::AddFOp>(bodyOp))
        return isConstant && elementType.isBF16();
    if (isa<arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp>(bodyOp))
        return elementType.isInteger(8) || elementType.isInteger(16) || elementType.isInteger(32);
    if (isa<arith::MaximumFOp, arith::MinimumFOp>(bodyOp))
        return elementType.isBF16();
    return false;
}

// The part of a reduce init the kernel would drop, since it starts each output
// from the identity of the combine op. Both fields are null when the init can
// be ignored.
struct ReduceInit {
    // A known non-identity splat value.
    TypedAttr constant;
    // Any other init (including the scf.for accumulator of a tiled reduction),
    // merged into the result as is.
    Value tensor;
};

// Classifies the init of a reduce. Fails if the init matters but cannot be
// folded into the result, so the op is not lowered instead of giving a wrong
// result.
static FailureOr<ReduceInit> getReduceInit(Value init, Operation *bodyOp, Type elementType) {
    if (isScfAccumulator(init))
        return ReduceInit{{}, init};
    Value source = getInitSource(init);
    if (source.getDefiningOp<tensor::EmptyOp>())
        return ReduceInit{};
    if (TypedAttr value = getInitSplatValue(source)) {
        if (isReduceIdentity(value, bodyOp))
            return ReduceInit{};
        if (!canCombineInit(bodyOp, elementType, /*isConstant=*/true))
            return failure();
        return ReduceInit{value, {}};
    }
    if (!canCombineInit(bodyOp, elementType, /*isConstant=*/false))
        return failure();
    return ReduceInit{{}, init};
}

// The scale_bias operand of a reduce that applies no bias: bias 0, scale 1.
template <typename SrcOpT>
static Value createNoBiasScaleBias(PatternRewriter &rewriter, SrcOpT srcOp) {
    const std::vector<int32_t> bias = {0};
    const std::vector<int32_t> scale = {1};
    return createI32Const(rewriter, srcOp, interleave(bias, scale));
}

// Emits the torq_hl.reduce of `input` and folds `init` into its result. A
// constant init of a float sum is added to the f32 accumulator as a scalar fp32
// bias before rounding, like the sum itself; any other init is merged after the
// reduce. A new constant is used for a constant init instead of the original
// one, which addInitToResult cannot see through when it comes from an
// scf.forall. `noBiasScaleBias` (see createNoBiasScaleBias) is used otherwise,
// so outputs of one op can share it.
template <typename SrcOpT>
static FailureOr<Value> emitReduceWithInit(
    PatternRewriter &rewriter, SrcOpT srcOp, RankedTensorType resultType, StringRef opName,
    int64_t axis, Value input, const ReduceInit &init, Value noBiasScaleBias
) {
    Value scaleBias = noBiasScaleBias;
    int32_t outputMin = 0;
    int32_t outputMax = 0;
    auto floatInit = dyn_cast_if_present<FloatAttr>(init.constant);
    bool biasInKernel = floatInit && opName == "reduce_sum";
    if (biasInKernel) {
        scaleBias = createConst(
            std::vector<float>{floatInit.getValue().convertToFloat()}, rewriter, srcOp.getLoc()
        );
        // fp32 rescaleClamp interprets clipMin/clipMax as float bit patterns.
        outputMin = llvm::bit_cast<int32_t>(std::numeric_limits<float>::lowest());
        outputMax = llvm::bit_cast<int32_t>(std::numeric_limits<float>::max());
    }

    auto reduceOp = emitTorqReduce(
        rewriter, srcOp, resultType, opName, axis, scaleBias, outputMin, outputMax, input
    );
    Value result = reduceOp.getResult(0);

    Value mergeWith = init.tensor;
    if (init.constant && !biasInKernel) {
        mergeWith = arith::ConstantOp::create(
            rewriter, srcOp.getLoc(), resultType, SplatElementsAttr::get(resultType, init.constant)
        );
    }
    if (!mergeWith)
        return result;
    return mergeReductionPartial(srcOp, mergeWith, result, opName, rewriter);
}

/// Rewrites a single-input `linalg.generic` with two or more reduction loops
/// (e.g. ONNX ReduceSum over several axes) into a multi-dimension
/// `linalg.reduce`, which ReduceOpConversion lowers. The input map must be a
/// permutation. The output either drops the reduced input dims or keeps them as
/// size-1 dims (keepdims); the latter goes through collapse_shape / expand_shape.
/// A generic with one reduction loop is lowered directly by
/// GenericReductionConversion, which also handles a combined min+max.
struct MultiReductionGenericToReduce : public OpRewritePattern<linalg::GenericOp> {

    MultiReductionGenericToReduce(MLIRContext *context) : OpRewritePattern(context) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp.getNumReductionLoops() < 2 || srcOp.getNumDpsInputs() != 1 ||
            srcOp.getNumDpsInits() != 1 || !srcOp.hasPureTensorSemantics()) {
            return rewriter.notifyMatchFailure(srcOp, "not a single-input multi-reduction");
        }
        AffineMap inMap = srcOp.getMatchingIndexingMap(srcOp.getDpsInputOperand(0));
        if (!inMap.isPermutation()) {
            return rewriter.notifyMatchFailure(srcOp, "input map is not a permutation");
        }

        Value init = srcOp.getDpsInits()[0];
        auto initType = cast<RankedTensorType>(init.getType());
        unsigned rank = inMap.getNumResults();
        SmallVector<utils::IteratorType> iterators = srcOp.getIteratorTypesArray();
        AffineMap outMap = srcOp.getMatchingIndexingMap(srcOp.getDpsInitOperand(0));
        bool keepDims = outMap.getNumResults() == rank;

        // Work in input dims: input dim i is loop inMap.getDimPosition(i). The
        // output must list the kept input dims in input order (or keep the
        // reduced ones as size-1 dims at their input position). With keepdims,
        // each reduced dim of the init is attached to the nearest kept dim
        // before it (or the first kept dim when leading) to collapse it away.
        SmallVector<int64_t> reducedDims;
        SmallVector<AffineExpr> expectedOut;
        SmallVector<ReassociationIndices> reassoc;
        ReassociationIndices leading;
        for (unsigned i = 0; i < rank; ++i) {
            unsigned loop = inMap.getDimPosition(i);
            if (iterators[loop] == utils::IteratorType::reduction) {
                reducedDims.push_back(i);
                if (keepDims) {
                    if (initType.getDimSize(i) != 1) {
                        return rewriter.notifyMatchFailure(srcOp, "unsupported output map");
                    }
                    expectedOut.push_back(rewriter.getAffineConstantExpr(0));
                    if (reassoc.empty()) {
                        leading.push_back(i);
                    }
                    else {
                        reassoc.back().push_back(i);
                    }
                }
                continue;
            }
            expectedOut.push_back(rewriter.getAffineDimExpr(loop));
            if (keepDims) {
                reassoc.push_back(leading);
                reassoc.back().push_back(i);
                leading.clear();
            }
        }
        if (outMap.getResults() != ArrayRef(expectedOut)) {
            return rewriter.notifyMatchFailure(srcOp, "unsupported output map");
        }

        Value reduceInit = init;
        if (keepDims) {
            reduceInit = tensor::CollapseShapeOp::create(rewriter, srcOp.getLoc(), init, reassoc);
        }

        Block *body = srcOp.getBody();
        auto reduceOp = linalg::ReduceOp::create(
            rewriter, srcOp.getLoc(), ValueRange{srcOp.getDpsInputs()[0]}, ValueRange{reduceInit},
            reducedDims,
            [&](OpBuilder &b, Location, ValueRange args) {
                IRMapping mapping;
                mapping.map(body->getArguments(), args);
                for (Operation &op : *body) {
                    b.clone(op, mapping);
                }
            }
        );
        reduceOp->setDiscardableAttrs(srcOp->getDiscardableAttrDictionary());

        Value result = reduceOp.getResult(0);
        if (keepDims) {
            result =
                tensor::ExpandShapeOp::create(rewriter, srcOp.getLoc(), initType, result, reassoc);
        }
        rewriter.replaceOp(srcOp, result);
        return success();
    }
};

/// Lowers a single-reduction-loop `linalg.generic` to `torq_hl::ReduceOp`
struct GenericReductionConversion : public OpRewritePattern<linalg::GenericOp> {

    GenericReductionConversion(MLIRContext *context) : OpRewritePattern(context) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        auto info = getGenericReductionInfo(srcOp);

        if (failed(info)) {
            return rewriter.notifyMatchFailure(srcOp, "not a supported reduction generic");
        }

        LLVM_DEBUG({
            llvm::dbgs() << "\nGenericReductionConversion: Reduce source op:\n";
            srcOp.dump();
        });

        // The kernel starts each output from the combine identity, so the init
        // is folded in by emitReduceWithInit. Check every output before emitting
        // anything.
        SmallVector<ReduceInit> inits;
        for (auto [i, init] : llvm::enumerate(info->inits)) {
            FailureOr<ReduceInit> reduceInit =
                getReduceInit(init, info->bodyOps[i], info->resultTypes[i].getElementType());
            if (failed(reduceInit)) {
                return rewriter.notifyMatchFailure(srcOp, "init cannot be merged into reduce");
            }
            inits.push_back(*reduceInit);
        }

        Value scaleBias = createNoBiasScaleBias(rewriter, srcOp);

        // Emit one torq_hl.reduce per output. For a combined min+max reduce both
        // reduces read the same `input`; block-local load dedup collapses that to
        // a single DMA per tile, so min and max share one streaming pass.
        SmallVector<Value> results;
        results.reserve(info->inits.size());
        for (auto [i, init] : llvm::enumerate(inits)) {
            FailureOr<Value> result = emitReduceWithInit(
                rewriter, srcOp, info->resultTypes[i], info->opNames[i], info->axis, info->input,
                init, scaleBias
            );
            if (failed(result)) {
                return rewriter.notifyMatchFailure(srcOp, "failed to merge init into reduce");
            }
            results.push_back(*result);
        }

        rewriter.replaceOp(srcOp, results);
        return success();
    }
};

/// Lowers a single-input `linalg.reduce` to `torq_hl::ReduceOp`. Multiple reduced
/// dims are collapsed into one axis or reduced by a chain of single-axis ops.
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

        if (auto reason = getUnsupportedReduceDimsReason(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, *reason);
        }
        SmallVector<int64_t> dimensions(srcOp.getDimensions());
        llvm::sort(dimensions);

        // A multi-dimension reduce becomes one torq_hl.reduce over a collapsed
        // axis when it can be folded, otherwise a chain of single-axis reduces.
        // Only decide here; IR is built below.
        bool multiDim = dimensions.size() > 1;
        bool foldDims = multiDim && canFoldReduceDims(inputType, dimensions);

        auto reduceBodyOp = yieldOp.getOperand(0).getDefiningOp();
        auto opName = getReduceOpName(reduceBodyOp);
        if (failed(opName)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported reduction operation");
        }

        // The kernel starts from the combine identity; the init is folded in by
        // emitReduceWithInit.
        FailureOr<ReduceInit> init =
            getReduceInit(srcOp.getInits()[0], reduceBodyOp, srcOutputType.getElementType());
        if (failed(init)) {
            return rewriter.notifyMatchFailure(srcOp, "init cannot be merged into reduce");
        }

        Value scaleBias = createNoBiasScaleBias(rewriter, srcOp);
        int64_t axis = dimensions.front();
        if (multiDim && !foldDims) {
            // Reduce the higher axes first so lower axis indices stay valid. The
            // last step, over `axis`, is emitted below with the init.
            // A bf16 sum keeps its partial sums in f32 between steps, as one
            // torq_hl.reduce does in its accumulator; only the last step rounds.
            Type partialElementType = srcOutputType.getElementType();
            if (inputElementType.isBF16() && *opName == "reduce_sum") {
                partialElementType = rewriter.getF32Type();
            }
            for (int64_t d : llvm::reverse(ArrayRef(dimensions).drop_front())) {
                SmallVector<int64_t> shape(cast<RankedTensorType>(input.getType()).getShape());
                shape.erase(shape.begin() + d);
                auto stepType = RankedTensorType::get(shape, partialElementType);
                auto step = emitTorqReduce(
                    rewriter, srcOp, stepType, *opName, d, scaleBias, /*outputMin*/ 0,
                    /*outputMax*/ 0, input
                );
                input = step.getResult(0);
            }
        }
        else if (foldDims) {
            // Fold the contiguous reduced dims [front, back] into the single axis `front`.
            SmallVector<ReassociationIndices> reassoc;
            for (int64_t d = 0; d < inputType.getRank(); ++d) {
                if (d > dimensions.front() && d <= dimensions.back()) {
                    reassoc.back().push_back(d);
                }
                else {
                    reassoc.push_back({d});
                }
            }
            input = tensor::CollapseShapeOp::create(rewriter, srcOp.getLoc(), input, reassoc);
        }

        FailureOr<Value> result = emitReduceWithInit(
            rewriter, srcOp, srcOutputType, *opName, axis, input, *init, scaleBias
        );
        if (failed(result)) {
            return rewriter.notifyMatchFailure(srcOp, "failed to merge init into reduce");
        }
        rewriter.replaceOp(srcOp, *result);

        return success();
    }
};

class AddOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

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
            return isa<BlockArgument>(value) ||
                   traceToConstFloat(op, value, SplatPolicy::SingleElement).has_value();
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

        auto data = traceToConstFloat(srcOp, rhs, SplatPolicy::SingleElement);
        if (!data) {
            data = traceToConstFloat(srcOp, lhs, SplatPolicy::SingleElement);
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

        // Fold the forward ACT tail: a trailing clamp (and any second scalar
        // add) is absorbed into the torq_hl.add's clip range / ACT bias, and
        // the consumed generics are dropped. The torq_hl.add lowering already
        // applies act.rescaleClamp with these bounds.
        Value output = srcOp.getResult(0);
        ScaleClampInfo scInfo = foldForwardScaleClamp(output, 1, 12, 12, true);
        // Default: the open clip range (full float range).
        float min_f = std::numeric_limits<float>::lowest();
        int32_t output_min = *reinterpret_cast<int32_t *>(&min_f);
        float max_f = std::numeric_limits<float>::max();
        int32_t output_max = *reinterpret_cast<int32_t *>(&max_f);

        // The HW weight bus is 16-bit BF, so add/sub weights are always bf16. They are
        // exactly +/-1.0, which bf16 represents without loss, so the ALU multiply-accumulate
        // is exact regardless of the data precision.
        std::vector<llvm::APFloat> weights(2, llvm::APFloat(llvm::APFloat::BFloat(), "1.0"));
        // for sub, set weight of 2nd input to -1.0
        // so it becomes => %input0 + (-1.0) * %input1
        if (opName == "sub") {
            weights[1] = llvm::APFloat(llvm::APFloat::BFloat(), "-1.0");
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
        // A second scalar add folded from the tail adds to the ACT bias.
        if (scInfo.floatBias)
            biasScale[0] += *scInfo.floatBias;
        if (scInfo.hasFloatTail()) {
            output_min = scInfo.min;
            output_max = scInfo.max;
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
                weights[0] = llvm::APFloat(llvm::APFloat::BFloat(), "-1.0");
                weights[1] = llvm::APFloat(llvm::APFloat::BFloat(), "1.0");
            }
            opName = "add";
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        // The fp32 HW add uses the ALU accumulate mode, whose implicit weights are 1.0, so it
        // can only compute a + b: a tensor-vs-tensor sub (weights [1, -1]) -- including its
        // reversed encoding, a constant-minus-input renamed "add" -- would be silently
        // computed as a + b. Check the weight constant here, while it is still one, and
        // reject any non-unit-weight f32 tensor-tensor op before the torq_hl.add is created,
        // so it falls back to the host. An f32 op with a scalar rhs is rejected too: the
        // scalar paths load the input weight through the MUL mode's 16-bit weight bus, which
        // cannot carry an fp32 weight.
        if (srcResultType.getElementType().isF32() && rhs_is_scalar) {
            return rewriter.notifyMatchFailure(
                srcOp, "f32 add/sub with a scalar operand is not supported on the NPU"
            );
        }
        if (srcResultType.getElementType().isF32() &&
            !(weights[0].convertToFloat() == 1.0f && weights[1].convertToFloat() == 1.0f)) {
            return rewriter.notifyMatchFailure(
                srcOp, "f32 tensor-vs-tensor sub is not supported by the ALU accumulate mode"
            );
        }

        auto torqWeights = createConst(weights, rewriter, srcOp.getLoc());

        auto addOp = torq_hl::AddOp::create(
            rewriter, srcOp.getLoc(),
            scInfo.hasFloatTail() ? cast<RankedTensorType>(output.getType()) : srcResultType,
            createInitTensor(
                srcOp, rewriter,
                scInfo.hasFloatTail() ? cast<RankedTensorType>(output.getType()) : srcResultType
            ),
            opName,
            0, // input_zp
            0, // output_zp
            output_min, output_max,
            0, // shift_factor
            torqWeights, createConst(biasScale, rewriter, srcOp.getLoc()), input0, input1, false,
            rhs_is_scalar
        );
        if (scInfo.hasFloatTail()) {
            rewriter.replaceOp(output.getDefiningOp(), addOp.getResult(0));
            // Mark every consumed tail op: the driver must not reprocess it as
            // a root (a second rewrite of the same op would double-commit).
            for (auto *op : llvm::reverse(scInfo.floatTailOps))
                if (op != output.getDefiningOp())
                    rewriter.eraseOp(op);
            rewriter.eraseOp(srcOp);
        }
        else {
            rewriter.replaceOp(srcOp, addOp.getResult(0));
        }
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
        if (outType.getElementType().isF64()) {
            return rewriter.notifyMatchFailure(srcOp, "addOp doesn't support f64 right now\n");
        }
        if (outType.getElementType().isInteger(64)) {
            return rewriter.notifyMatchFailure(srcOp, "addOp doesn't support i64");
        }

        const bool isBF16 = outType.getElementType().isBF16();
        const bool isF32 = outType.getElementType().isF32();

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

        if (isBF16 || isF32) {
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

            auto constVal = traceToConstInt(srcOp, rhs, SplatPolicy::Any);
            if (!constVal) {
                constVal = traceToConstInt(srcOp, lhs, SplatPolicy::Any);
                if (constVal) {
                    input0 = input1;
                    inputNeedReverse = true;
                }
            }
            if (constVal) {
                auto data = static_cast<int32_t>(*constVal);
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

// A per-channel rescale carries its multiplier and/or shift as channel-indexed operands, so the
// number of distinct scale values is the element count of those operands (either one may be a
// scalar constant folded into the generic body). A per-tensor rescale has no such operand,
// leaving the count at 1.
static int getRescaleScaleCount(linalg::GenericOp genericOp, tosa::ApplyScaleOp applyScaleOp) {
    int64_t count = 1;
    for (Value table : {applyScaleOp.getMultiplier(), applyScaleOp.getShift()}) {
        if (auto arg = dyn_cast<BlockArgument>(table)) {
            if (auto operand = genericOp.getMatchingOpOperand(arg)) {
                if (auto type = dyn_cast<RankedTensorType>(operand->get().getType()))
                    count = std::max(count, type.getNumElements());
            }
        }
    }
    return static_cast<int>(count);
}

// Two extract_slice offsets select the same window iff they are the same SSA value or equal
// constants; anything else (e.g. %iv vs %iv + c) would pair one table's window with the
// other's, silently.
static bool sameSliceOffset(OpFoldResult a, OpFoldResult b) {
    if (Value av = dyn_cast<Value>(a))
        if (Value bv = dyn_cast<Value>(b))
            return av == bv;
    if (Attribute aa = dyn_cast<Attribute>(a))
        if (Attribute ba = dyn_cast<Attribute>(b))
            return aa == ba;
    return false;
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
        int32_t inputZp, int32_t outputZp, int32_t outputMin, int32_t outputMax, bool inputUnsigned,
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

        // Per-channel scale_bias is a [channels, 2] table; per-tensor is the single 1-D [2] pair
        // broadcast to every element.
        Value scaleBiasConst;
        if (scaleValuesCount > 1) {
            scaleBiasConst = createI32Const(
                rewriter, srcOp, scaleBias, llvm::ArrayRef<int64_t>{scaleValuesCount, 2}
            );
        }
        else {
            scaleBiasConst = createI32Const(rewriter, srcOp, scaleBias);
        }

        return emitRescaleFma(
            srcOp, rewriter, input, outputType, scaleBiasConst, scaleValuesCount, outputZp,
            outputMin, outputMax, shiftFactor, weightData, inputUnsigned, applyScaleOp
        );
    }

    // Shared FMA emission for the folded (emitRescale) and sliced (emitSlicedPerChannelRescale)
    // rescale paths: place the channel dimension — the one that indexes `scaleBias` — innermost,
    // transposing the data to and from it when it is not, and build the torq_hl.fma with the
    // given weight and shift. tableSize is the number of channels the table indexes (1 for a
    // per-tensor rescale, where the [2] pair broadcasts to every element and no channel detection
    // happens). The table is already in channel order (the output's channel-dim order), so only
    // the data is transposed.
    LogicalResult emitRescaleFma(
        linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input,
        RankedTensorType outputType, Value scaleBias, int64_t tableSize, int32_t outputZp,
        int32_t outputMin, int32_t outputMax, int32_t shiftFactor, int8_t weightData,
        bool inputUnsigned, tosa::ApplyScaleOp applyScaleOp
    ) const {
        int64_t channelDim = outputType.getRank() - 1;
        bool needsTranspose = false;
        SmallVector<int64_t> perm;
        if (tableSize > 1) {
            // The per-channel table is indexed by the channel dimension (innermost unless its
            // indexing map says otherwise); prefer the multiplier's map, then the shift's (the
            // multiplier may be a scalar constant in the body).
            for (Value table : {applyScaleOp.getMultiplier(), applyScaleOp.getShift()}) {
                auto arg = dyn_cast<BlockArgument>(table);
                if (!arg)
                    continue;
                auto indexingMaps = srcOp.getIndexingMapsArray();
                if (arg.getArgNumber() >= static_cast<int>(indexingMaps.size()))
                    continue;
                auto map = indexingMaps[arg.getArgNumber()];
                if (map.getNumResults() == 1) {
                    if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(0))) {
                        channelDim = dimExpr.getPosition();
                        break;
                    }
                }
            }
            if (outputType.getDimSize(channelDim) != tableSize) {
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

        Value weightConst =
            createI8Const(
                rewriter, srcOp, llvm::ArrayRef<int8_t>{weightData}, llvm::ArrayRef<int64_t>{1}
            )
                .getResult();

        auto emitFma = [&](Value fmaInput, RankedTensorType fmaType) {
            return torq_hl::FMAOp::create(
                rewriter, srcOp.getLoc(), fmaType, createInitTensor(srcOp, rewriter, fmaType),
                outputZp, outputMin, outputMax, shiftFactor, weightConst, scaleBias, fmaInput,
                inputUnsigned
            );
        };

        if (needsTranspose) {
            Value transposedInput = transposeValue(input, perm, srcOp.getLoc(), rewriter);
            RankedTensorType transposedOutputType = transposeType(outputType, perm);
            auto fmaOp = emitFma(transposedInput, transposedOutputType);
            rewriter.replaceOp(
                srcOp,
                transposeValue(fmaOp.getOutput(), invertPermutation(perm), srcOp.getLoc(), rewriter)
            );
            return success();
        }

        auto fmaOp = emitFma(input, outputType);
        rewriter.replaceOp(srcOp, fmaOp.getOutput());
        return success();
    }

    // A per-channel table sliced by an enclosing loop (LinalgSlicing) is not constant-foldable
    // as a slice, but its source — the unsliced table — is a compile-time constant. Resolve the
    // source, normalize it whole (common shift = the source's minimum shift, scales rescaled to
    // it: the same math as emitRescale, which is exact for any sub-window), build one [C, 2]
    // {bias, scale} constant, and emit the shared FMA (emitRescaleFma) over the extract_slice of
    // the normalized table that mirrors the data slice (same offsets, sizes [C_slice, 2]). From
    // then on the slice is an ordinary operand value — the lowering no longer depends on the
    // sliced table being constant-foldable (the old hard "multiplier must be a constant" error).
    //
    // Handled form: each per-channel multiplier/shift is a rank-1 tensor (or rank-2 [N, 1])
    // generic operand taken through a tensor.extract_slice that slices dim 0 only, with unit
    // strides and a static slice size, whose source is a compile-time constant; a multiplier or
    // shift that is a scalar constant in the generic body is broadcast to the other table's
    // window. The two tables' dim-0 slice offsets must be the same value (or equal constants),
    // and the source's minimum shift must be non-negative (a per-channel scale-up cannot be
    // folded into the FMA's int8 weight). Anything else bails with the reason printed under
    // --debug, and the rescale falls back to the CSS host as before.
    LogicalResult emitSlicedPerChannelRescale(
        linalg::GenericOp srcOp, PatternRewriter &rewriter, Value input,
        RankedTensorType outputType, int32_t inputZp, int32_t outputZp, int32_t outputMin,
        int32_t outputMax, bool inputUnsigned, tosa::ApplyScaleOp applyScaleOp
    ) const {
        auto fail = [&](Twine reason) -> LogicalResult {
            LLVM_DEBUG({ llvm::dbgs() << "sliced per-channel rescale: " << reason << "\n"; });
            return rewriter.notifyMatchFailure(srcOp, reason);
        };

        struct SlicedTable {
            std::optional<int64_t> scalar; // a constant in the generic body (not an operand)
            Value full;                    // the full source table
            OpFoldResult offset;           // the dim-0 slice offset (an IV or a constant)
            int64_t sliceSize = -1;
            int64_t fullSize = -1;
        };

        auto resolveTable = [&](Value tableOrScalar) -> FailureOr<SlicedTable> {
            SlicedTable t;
            if (auto arg = dyn_cast<BlockArgument>(tableOrScalar)) {
                Value operand = srcOp.getMatchingOpOperand(arg)->get();
                auto slice = operand.getDefiningOp<tensor::ExtractSliceOp>();
                if (!slice) {
                    // An unsliced table that is still not foldable (e.g. a runtime value):
                    // the same hard limitation as before; fullSize is filled from the fold
                    // attempt below.
                    t.full = operand;
                    return t;
                }
                auto srcType = cast<RankedTensorType>(slice.getSourceType());
                if (srcType.getRank() != 1 &&
                    !(srcType.getRank() == 2 && srcType.getDimSize(1) == 1))
                    return failure();
                auto mixedSizes = slice.getMixedSizes();
                auto mixedStrides = slice.getMixedStrides();
                auto toInt = [&](OpFoldResult fr) -> std::optional<int64_t> {
                    if (auto attr = dyn_cast<Attribute>(fr)) {
                        if (auto intAttr = dyn_cast<IntegerAttr>(attr))
                            return intAttr.getInt();
                        return std::nullopt;
                    }
                    IntegerAttr intAttr;
                    if (matchPattern(cast<Value>(fr), m_Constant(&intAttr)))
                        return intAttr.getInt();
                    return std::nullopt;
                };
                for (int64_t i = 0; i < srcType.getRank(); ++i) {
                    std::optional<int64_t> stride = toInt(mixedStrides[i]);
                    std::optional<int64_t> size = toInt(mixedSizes[i]);
                    if (!stride || !size)
                        return failure();
                    if (*stride != 1)
                        return failure();
                    if (i > 0 && *size != srcType.getDimSize(i))
                        return failure(); // only the channel (dim 0) may be partially sliced
                }
                t.full = slice.getSource();
                t.fullSize = srcType.getNumElements();
                t.offset = slice.getMixedOffsets()[0];
                std::optional<int64_t> sliceSize = toInt(mixedSizes[0]);
                if (!sliceSize)
                    return failure();
                t.sliceSize = *sliceSize;
                return t;
            }
            auto scalar = getConstIntValue(tableOrScalar);
            if (!scalar)
                return failure();
            t.scalar = *scalar;
            return t;
        };

        FailureOr<SlicedTable> multT = resolveTable(applyScaleOp.getMultiplier());
        FailureOr<SlicedTable> shiftT = resolveTable(applyScaleOp.getShift());
        if (failed(multT) || failed(shiftT)) {
            return fail("sliced per-channel rescale table has an unsupported form");
        }

        int64_t fullSize = -1, sliceSize = -1;
        OpFoldResult offset;
        SlicedTable *tables[2] = {&(*multT), &(*shiftT)};
        for (SlicedTable *tp : tables) {
            if (!tp->offset)
                continue;
            if (fullSize < 0) {
                fullSize = tp->fullSize;
                sliceSize = tp->sliceSize;
                offset = tp->offset;
            }
            else if (tp->fullSize != fullSize || tp->sliceSize != sliceSize ||
                     !sameSliceOffset(tp->offset, offset)) {
                return fail("sliced per-channel rescale tables disagree");
            }
        }
        if (fullSize < 0) {
            return fail("sliced per-channel rescale has no constant source table");
        }

        auto fullValues = [&](const SlicedTable &t) -> FailureOr<SmallVector<APInt>> {
            if (t.scalar) {
                return SmallVector<APInt>(fullSize, APInt(32, *t.scalar, /*isSigned=*/true));
            }
            auto constV = computeArithConst(t.full, true, {});
            if (failed(constV))
                return failure();
            auto attr = dyn_cast_or_null<DenseElementsAttr>(
                cast<arith::ConstantOp>(constV->getDefiningOp()).getValue()
            );
            if (!attr || attr.getNumElements() != fullSize)
                return failure();
            auto values = attr.getValues<APInt>();
            return SmallVector<APInt>(values.begin(), values.end());
        };

        FailureOr<SmallVector<APInt>> multFull = fullValues(*multT);
        FailureOr<SmallVector<APInt>> shiftFull = fullValues(*shiftT);
        if (failed(multFull) || failed(shiftFull)) {
            return fail("sliced per-channel rescale table source is not a constant");
        }

        SmallVector<int32_t> multFullV(fullSize);
        SmallVector<int8_t> shiftFullV(fullSize);
        for (int64_t i = 0; i < fullSize; ++i) {
            multFullV[i] = static_cast<int32_t>((*multFull)[i].getSExtValue());
            shiftFullV[i] = static_cast<int8_t>((*shiftFull)[i].getSExtValue());
        }

        int32_t globalShift = *std::min_element(shiftFullV.begin(), shiftFullV.end());
        if (globalShift < 0) {
            return fail("negative per-channel rescale shift cannot fold into int8 weight");
        }

        auto scaleFull = compute_scale(multFullV, shiftFullV, globalShift);
        SmallVector<int32_t> biasFull(fullSize, -inputZp);
        SmallVector<int32_t> table(2 * fullSize);
        interleave_into(
            llvm::ArrayRef<int32_t>(biasFull), llvm::ArrayRef<int32_t>(scaleFull), table.begin()
        );

        // The normalized table is loop-invariant: create it next to the (unsliced) source
        // constant it is derived from, not at the rescale inside the enclosing slicing loop.
        // That keeps it out of the loop body (no per-body clone can duplicate the full table)
        // and alongside the other dispatch constants for CSE. (The window extract_slice below
        // depends on the loop's IV and must stay at the rescale's position.)
        SlicedTable *slicedT = (*multT).offset ? &(*multT) : &(*shiftT);
        Value fullTableConst;
        {
            OpBuilder::InsertionGuard tableInsertGuard(rewriter);
            if (Operation *sourceDef = slicedT->full.getDefiningOp()) {
                rewriter.setInsertionPoint(sourceDef);
            }
            fullTableConst =
                createI32Const(rewriter, srcOp, table, llvm::ArrayRef<int64_t>{fullSize, 2})
                    .getResult();
        }

        // The FMA reads the same dim-0 window of the normalized table that the rescale reads of
        // the raw tables.
        auto sliceType = RankedTensorType::get({sliceSize, 2}, rewriter.getI32Type());
        Value scaleBias =
            tensor::ExtractSliceOp::create(
                rewriter, srcOp.getLoc(), sliceType, fullTableConst,
                ArrayRef<OpFoldResult>{offset, rewriter.getIndexAttr(0)},
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr(sliceSize), rewriter.getIndexAttr(2)},
                ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)}
            )
                .getResult();

        return emitRescaleFma(
            srcOp, rewriter, input, outputType, scaleBias, sliceSize, outputZp, outputMin,
            outputMax, globalShift, /*weightData=*/1, inputUnsigned, applyScaleOp
        );
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

        // try to extract input_zp
        auto subOp = dyn_cast_or_null<arith::SubIOp>(applyScaleOp.getValue().getDefiningOp());
        if (subOp) {
            auto maybeInputZp = getConstIntValue(subOp.getRhs());
            if (maybeInputZp) {
                input_zp = *maybeInputZp;
            }
        }

        // An unsigned input reaches the body zero-extended; the FMA must read it unsigned too,
        // or values with the top bit set are taken as negative.
        Value extendedInput = subOp ? subOp.getLhs() : applyScaleOp.getValue();
        const bool inputUnsigned = extendedInput.getDefiningOp<arith::ExtUIOp>() != nullptr;

        // A per-channel table is normally folded from its (constant) source. The attempt is
        // quiet for the per-channel case: a table sliced by an enclosing loop (LinalgSlicing)
        // is an IV-dependent extract_slice the fold cannot see through, and
        // emitSlicedPerChannelRescale below normalizes the slice's (constant) source instead —
        // the quiet attempt keeps that expected fallback free of the fold's error diagnostics
        // (the failure reason is still printed under --debug). Per-tensor rescales have no
        // fallback and keep the original errors.
        auto ms = getMultiplierAndShift(
            srcOp, applyScaleOp, scaleValuesCount, /*quiet=*/scaleValuesCount > 1
        );
        if (!ms) {
            if (scaleValuesCount > 1) {
                return emitSlicedPerChannelRescale(
                    srcOp, rewriter, input, outputType, input_zp, outputZp, outputMin, outputMax,
                    inputUnsigned, applyScaleOp
                );
            }
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to get multiplier and shift from apply_scale operation"
            );
        }

        LLVM_DEBUG({
            llvm::dbgs() << "rescale params : input_zp: " << input_zp << ", outputZp: " << outputZp
                         << ", outputMin: " << outputMin << ", outputMax: " << outputMax
                         << ", channels: " << scaleValuesCount << "\n";
        });

        return emitRescale(
            srcOp, rewriter, input, outputType, ms, scaleValuesCount, input_zp, outputZp, outputMin,
            outputMax, inputUnsigned, applyScaleOp
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

// Element types the hardware reduce accepts: bf16, f32 or 8/16/32-bit integer
// input, and an output of the same type. Only a sum can accumulate bf16 into f32.
static bool isSupportedReduceElementTypes(
    StringRef opName, Type inputElementType, Type outputElementType, std::string &failReason
) {
    auto inputIntType = dyn_cast<IntegerType>(inputElementType);
    bool inputOk =
        inputElementType.isBF16() || inputElementType.isF32() ||
        (inputIntType && (inputIntType.getWidth() == 8 || inputIntType.getWidth() == 16 ||
                          inputIntType.getWidth() == 32));
    bool outputOk = opName == "reduce_sum" && inputElementType.isBF16()
                        ? (outputElementType.isBF16() || outputElementType.isF32())
                        : (inputElementType == outputElementType);
    if (!inputOk || !outputOk) {
        failReason = "Element types not supported by the hardware reduce";
        return false;
    }
    return true;
}

static bool isSumMaxOrMin(StringRef opName) {
    return opName == "reduce_sum" || opName == "reduce_max" || opName == "reduce_min";
}

bool isTorqReduceOp(Operation *op, std::string &failReason) {
    auto reduceOp = dyn_cast<linalg::ReduceOp>(op);
    if (!reduceOp) {
        failReason = "Not a linalg.reduce op";
        return false;
    }
    if (reduceOp.getInputs().size() != 1) {
        failReason = "Reduce only supports exactly one input";
        return false;
    }
    auto inputType = dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(reduceOp->getResultTypes().front());
    if (!inputType || !outputType) {
        failReason = "Input or output is not a ranked tensor";
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(reduceOp.getBody()->getTerminator());
    Operation *reduceBodyOp = yieldOp ? yieldOp.getOperand(0).getDefiningOp() : nullptr;
    FailureOr<std::string> opName =
        reduceBodyOp ? getReduceOpName(reduceBodyOp) : FailureOr<std::string>(failure());
    if (failed(opName) || !isSumMaxOrMin(*opName)) {
        failReason = "Not a sum, max or min reduction";
        return false;
    }

    Type outputElementType = outputType.getElementType();
    if (!isSupportedReduceElementTypes(
            *opName, inputType.getElementType(), outputElementType, failReason
        )) {
        return false;
    }

    // Multi-dimension reductions are lowered by ReduceOpConversion (collapse or chain).
    if (auto reason = getUnsupportedReduceDimsReason(reduceOp)) {
        failReason = ("Reduce: " + *reason).str();
        return false;
    }

    // A max or min init (e.g. -inf) must fold into the kernel, or ReduceOpConversion
    // fails and the op is left unlowered. Sums are kept on the NPU as before.
    if (*opName != "reduce_sum" &&
        failed(getReduceInit(reduceOp.getInits()[0], reduceBodyOp, outputElementType))) {
        failReason = "init cannot be merged into the hardware reduce";
        return false;
    }

    return true;
}

// Like isTorqReduceOp, for a linalg.generic reduce that GenericReductionConversion
// lowers: single input, single reduction dim, and an init the kernel accepts.
bool isTorqGenericReduceOp(Operation *op, std::string &failReason) {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp) {
        failReason = "Not a linalg.generic op";
        return false;
    }

    FailureOr<GenericReductionInfo> info = getGenericReductionInfo(genericOp);
    if (failed(info)) {
        failReason = "Not a supported reduction generic";
        return false;
    }
    if (info->inits.size() != 1) {
        failReason = "Generic reduce expects exactly one output";
        return false;
    }
    StringRef opName = info->opNames.front();
    if (!isSumMaxOrMin(opName)) {
        failReason = "Not a sum, max or min reduction";
        return false;
    }

    auto inputType = dyn_cast<RankedTensorType>(info->input.getType());
    if (!inputType) {
        failReason = "Input is not a ranked tensor";
        return false;
    }
    Type outputElementType = info->resultTypes.front().getElementType();
    if (!isSupportedReduceElementTypes(
            opName, inputType.getElementType(), outputElementType, failReason
        )) {
        return false;
    }

    if (failed(getReduceInit(info->inits.front(), info->bodyOps.front(), outputElementType))) {
        failReason = "init cannot be merged into the hardware reduce";
        return false;
    }

    return true;
}

bool isIdentityReduceInit(Value init, Operation *combineOp) {
    FailureOr<ReduceInit> reduceInit = getReduceInit(init, combineOp, getElementTypeOrSelf(init));
    return succeeded(reduceInit) && !reduceInit->constant && !reduceInit->tensor;
}

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
        if (!dim0 || !dim1 || !dim2 || !dim3 || dim0.getPosition() != 0 ||
            dim1.getPosition() != 1) {
            return rewriter.notifyMatchFailure(
                genericOp, "expected NCHW nearest-neighbor input indexing"
            );
        }

        auto getScale = [](AffineBinaryOpExpr expr, unsigned position) -> std::optional<int64_t> {
            if (expr.getKind() != AffineExprKind::FloorDiv) {
                return std::nullopt;
            }
            auto dim = dyn_cast<AffineDimExpr>(expr.getLHS());
            auto scale = dyn_cast<AffineConstantExpr>(expr.getRHS());
            if (!dim || dim.getPosition() != position || !scale || scale.getValue() < 1) {
                return std::nullopt;
            }
            return scale.getValue();
        };
        std::optional<int64_t> scaleH = getScale(dim2, 2);
        std::optional<int64_t> scaleW = getScale(dim3, 3);
        if (!scaleH || !scaleW) {
            return rewriter.notifyMatchFailure(
                genericOp, "expected h floordiv scale_h and w floordiv scale_w indexing"
            );
        }

        auto resizeType = genericOp.getResult(0).getType();
        Value resizeInput = genericOp.getDpsInputs()[0];
        Value resizeInit =
            createInitTensor(genericOp, rewriter, mlir::cast<RankedTensorType>(resizeType));
        auto resizeOp = syna::torq_hl::ResizeNearestNeighborOp::create(
            rewriter, genericOp.getLoc(), resizeType, resizeInit, *scaleH, *scaleW, resizeInput
        );

        rewriter.replaceOp(genericOp, resizeOp.getOutput());
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

    // The raised MatMulInteger's i32 zero-point correction. Also registered in the pre-pass,
    // which claims it before tiling when the chain is small enough not to be sliced; a sliced
    // chain re-presents it here, per slice.
    populateLinalgToTorqHLMatMulIntegerCorrectPatterns(context, patterns, markFuseGroups);

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
    populateReciprocalPatterns(context, patterns);

    patterns.insert<TransposeOpConversion>(context);
    patterns.insert<DepthToSpaceOpConversion>(context);
    patterns.insert<FillOpConversion>(context);

    patterns.insert<TensorPadOpConversion>(context);

    patterns.insert<GenericReductionConversion>(context);
    patterns.insert<MultiReductionGenericToReduce>(context);
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
    populateLinalgToTorqHLExpandWeightsPatterns(context, patterns);
}

} // namespace mlir::syna::torq
