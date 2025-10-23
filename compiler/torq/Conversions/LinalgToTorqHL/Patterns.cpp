// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
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
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-patterns"

namespace mlir::syna::torq {

static llvm::cl::opt<bool> clACTBasedAdd(
    "torq-act-based-add",
    llvm::cl::desc(
        "use ACT based torq_hl::ElementWiseBinaryOp::ADD instead of ALU based torq_hl::AddOp"
    ),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clACTBasedSub(
    "torq-act-based-sub",
    llvm::cl::desc(
        "use ACT based torq_hl::ElementWiseBinaryOp::SUB instead of ALU based torq_hl::AddOp"
    ),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clMulCasti32Toi16(
    "torq-mul-cast-i32-to-i16",
    llvm::cl::desc("Automatically cast input from i32 to i16 for MUL operation"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clTableAsGather(
    "torq-convert-table-to-gather", llvm::cl::desc("use GatherOp instead of TosaOp for TOSA Table"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clConv1dAsMatmul(
    "torq-convert-conv1d-to-matmul", llvm::cl::desc("Convert conv1d to imToCol + matmul"),
    llvm::cl::init(false)
);

static StringRef castOpName(Value input, Value output) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto outputType = dyn_cast<RankedTensorType>(output.getType());
    auto inputElementType = inputType.getElementType();
    auto outputElementType = outputType.getElementType();

    if ((inputElementType.isF32() || inputElementType.isBF16()) && outputElementType.isInteger()) {
        return "f2i";
    }
    else if ((inputElementType.isF32() || inputElementType.isBF16()) &&
             (outputElementType.isF32() || outputElementType.isBF16())) {
        return "f2f";
    }
    else if (inputElementType.isInteger() && outputElementType.isInteger()) {
        return "i2i";
    }
    else if (inputElementType.isInteger() &&
             (outputElementType.isF32() || outputElementType.isBF16())) {
        return "i2f";
    }

    return "";
}

bool isTorqCastOp(Operation *op, std::string &opName, std::string &failReason) {

    auto srcOp = dyn_cast<linalg::GenericOp>(op);

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

    auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
    if (!yieldOp) {
        failReason = "Expected a linalg.yield terminator";
        return false;
    }

    auto castOp = yieldOp.getValues()[0].getDefiningOp();

    if (inputElementType.isF32() && outputElementType.isBF16()) {
        castOp = dyn_cast_or_null<arith::TruncFOp>(castOp);
    }
    else if (inputElementType.isBF16() && outputElementType.isF32()) {
        castOp = dyn_cast_or_null<arith::ExtFOp>(castOp);
    }
    else if (inputElementType.isInteger() && outputElementType.isInteger()) {
        // We should check the cast operation type here, as similar logic applies for ExtSIOp in
        // the following condition.
        if (inputElementType.isInteger(1) &&
            (outputElementType.isInteger(8) || outputElementType.isInteger(16)) &&
            isa<arith::ExtUIOp>(castOp)) {
            castOp = dyn_cast_or_null<arith::ExtUIOp>(castOp);
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

    opName = castOpName(input, srcOp.getResult(0));

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

namespace {

static Value create1DimTensorFromRescaleScalar(
    linalg::GenericOp srcOp, tosa::ApplyScaleOp applyScaleOp, const Type &elementType,
    PatternRewriter &rewriter
) {

    Value input = applyScaleOp.getValue();
    if (!input) {
        return nullptr;
    }

    auto ms = getMultiplierAndShift(srcOp, applyScaleOp, 1);
    if (!ms) {
        return nullptr;
    }

    double scaleFactor = static_cast<double>(ms.multiplier[0]) / (1l << ms.shift[0]);

    if (auto constOp = dyn_cast<arith::ConstantOp>(input.getDefiningOp())) {
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
                llvm::errs() << "constant is not scalar \n";
                return nullptr;
            }
        }
        else {
            llvm::errs() << "unsupported constant type \n";
            return nullptr;
        }

        data = static_cast<int32_t>(std::round(data * scaleFactor));

        RankedTensorType constType = RankedTensorType::get({}, elementType);
        DenseElementsAttr value;

        if (elementType.isInteger(16)) {
            value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(data));
        }
        else if (elementType.isInteger(32)) {
            value = DenseIntElementsAttr::get(constType, data);
        }
        else if (elementType.isInteger(8)) {
            value = DenseIntElementsAttr::get(constType, static_cast<int8_t>(data));
        }
        auto output = rewriter.create<arith::ConstantOp>(constOp.getLoc(), constType, value);
        return output.getResult();
    }
    return nullptr;
}

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

        auto trOp = rewriter.create<torq_hl::TransposeOp>(
            srcOp.getLoc(), srcOp.getResult()[0].getType(), srcOp.getInit(),
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

        auto outputElementSize = srcResultType.getElementType().getIntOrFloatBitWidth() / 8;
        int32_t outMin = -128, outMax = 127;
        getDTypeRange(outputElementSize, &outMin, &outMax);

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

struct ReduceOpConversion : public OpConversionPattern<linalg::ReduceOp> {

    ReduceOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        linalg::ReduceOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

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

        if (!inputElementType.isBF16() && !inputElementType.isInteger()) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support BF16 or integer element types for ReduceOp"
            );
        }

        if (inputElementType.getIntOrFloatBitWidth() != 8 &&
            inputElementType.getIntOrFloatBitWidth() != 16 &&
            inputElementType.getIntOrFloatBitWidth() != 32) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support 8, 16, or 32 bit element types for ReduceOp"
            );
        }

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());

        if (!yieldOp) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected linalg::YieldOp as the terminator of the ReduceOp body"
            );
        }

        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];

        auto inputShape = mlir::cast<RankedTensorType>(srcOp.getInputs()[0].getType()).getShape();

        auto srcOutputType = dyn_cast<RankedTensorType>(srcOp->getResultTypes().front());

        ArrayRef<int64_t> dimensions = srcOp.getDimensions();
        if (dimensions.size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "Only support one dimension reduction in ReduceOp"
            );
        }
        int64_t axis = dimensions[0];

        if (axis == inputShape.size() - 1 && inputShape.size() > 1) {

            SmallVector<int64_t> permutation(inputShape.size(), 0);

            permutation[0] = inputShape.size() - 1;

            for (int i = 1; i < permutation.size(); i++) {
                permutation[i] = i - 1;
            }

            llvm::SmallVector<int64_t> initShape{};
            for (int i = 0; i < inputShape.size(); i++) {
                initShape.push_back(inputShape[permutation[i]]);
            }

            auto initType = RankedTensorType::get(initShape, srcOutputType.getElementType());

            auto initValue =
                rewriter
                    .create<tensor::EmptyOp>(loc, initType.getShape(), initType.getElementType())
                    .getResult();

            auto transposeOp =
                rewriter.create<linalg::TransposeOp>(loc, input, initValue, permutation);

            input = transposeOp.getResult()[0];
            axis = 0;
        }

        if (axis != 0) {
            return rewriter.notifyMatchFailure(srcOp, "Only support the first dim reduction");
        }

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

        rewriter.replaceOpWithNewOp<syna::torq_hl::ReduceOp>(
            srcOp, srcOutputType, createInitTensor(srcOp, rewriter, srcOutputType), opName, axis, 0,
            0, 0, shift_factor,
            createI16Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{2}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );

        return success();
    }
};

static Value create1DimTensorFromScalar(
    arith::ConstantOp constOp, const Type &elementType, PatternRewriter &rewriter
) {
    if (!constOp)
        return {};

    if (elementType.isInteger()) {

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
                llvm::errs() << "constant is not scalar \n";
                return {};
            }
        }
        else {
            llvm::errs() << "unsupported constant type \n";
            return {};
        }

        RankedTensorType constType = RankedTensorType::get({1}, elementType);
        DenseElementsAttr value;

        if (elementType.isInteger(16)) {
            value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(data));
        }
        else if (elementType.isInteger(32)) {
            value = DenseIntElementsAttr::get(constType, data);
        }
        else if (elementType.isInteger(8))
            value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(data));
        else {
            return {};
        }

        return rewriter.create<arith::ConstantOp>(constOp.getLoc(), constType, value).getResult();
    }
    else if (elementType.isBF16() || elementType.isF32()) { // TODO: add F32 later

        auto attr = constOp.getValue();
        RankedTensorType constType = RankedTensorType::get({1}, elementType);
        DenseElementsAttr value;

        if (auto fpAttr = dyn_cast<FloatAttr>(attr)) {
            value = DenseElementsAttr::get(constType, fpAttr.getValue());
        }
        else if (auto denseAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
            if (denseAttr.isSplat()) {
                value = DenseElementsAttr::get(constType, denseAttr.getSplatValue<APFloat>());
            }
            else if (denseAttr.getType() == constType) {
                value = denseAttr;
            }
            else {
                llvm::errs() << "constant is not scalar \n";
                return {};
            }
        }
        else {
            llvm::errs() << "unsupported constant type \n";
            return {};
        }

        return rewriter.create<arith::ConstantOp>(constOp.getLoc(), constType, value).getResult();
    }
    else {
        return {};
    }
    return {};
}

class MulOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp.getInputs().empty() || srcOp.getInputs().size() > 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "mul expects generic op with 2 (or 1) inputs\n"
            );
        }

        Value input1 = srcOp.getInputs()[0];
        // Binary ops can actually have only one input if both operands are the same
        Value input2 = srcOp.getInputs()[srcOp.getInputs().size() > 1 ? 1 : 0];

        auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
        auto input1ElementType = input1Type.getElementType();
        auto input2ElementType = dyn_cast<RankedTensorType>(input2.getType()).getElementType();

        if (input1ElementType.isF32() || input1ElementType.isF64()) {
            return rewriter.notifyMatchFailure(srcOp, "mul expects i8, i16, bf16 inputs\n");
        }

        Operation *mulOp = getElementwiseBinaryOp(srcOp, /*allow constants*/ true);
        if (!mulOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        auto lhs = mulOp->getOperand(0);
        auto rhs = mulOp->getOperand(1);

        auto constOp = lhs.getDefiningOp<arith::ConstantOp>();
        if (constOp) {
            input2 = input1;
            input1 = nullptr;
        }
        else {
            constOp = rhs.getDefiningOp<arith::ConstantOp>();
            if (constOp) {
                input2 = nullptr;
            }
        }

        if (constOp) {

            Value v = create1DimTensorFromScalar(constOp, input1ElementType, rewriter);
            if (!v) {
                return rewriter.notifyMatchFailure(srcOp, "");
            }
            if (input1 == nullptr) {
                input1 = v;
            }
            else {
                input2 = v;
            }
        }

        if (!isa<arith::MulIOp>(mulOp) && !isa<arith::MulFOp>(mulOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Expected arith.muli or arith.mulf");
        }

        // TODO: check if the operations surrounding this Mul allows to use an i16 operation
        bool castToi16 = input1ElementType.isInteger(32) && input2ElementType.isInteger(32);
        if (clMulCasti32Toi16 && castToi16) {
            ArrayRef<int64_t> in1Shape = input1Type.getShape();
            auto inType = RankedTensorType::get(in1Shape, IntegerType::get(srcOp.getContext(), 16));

            input1 = rewriter
                         .create<torq_hl::ActOp>(
                             srcOp.getLoc(), inType, createInitTensor(srcOp, rewriter, inType),
                             "i2i", 0, 0, 0, 0, APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
                             APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input1
                         )
                         .getResult(0);

            input2 = rewriter
                         .create<torq_hl::ActOp>(
                             srcOp.getLoc(), inType, createInitTensor(srcOp, rewriter, inType),
                             "i2i", 0, 0, 0, 0, APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
                             APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input2
                         )
                         .getResult(0);
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        auto outputElementSize = srcResultType.getElementType().getIntOrFloatBitWidth() / 8;
        int32_t outMin = -128, outMax = 127;
        getDTypeRange(outputElementSize, &outMin, &outMax);

        rewriter.replaceOpWithNewOp<torq_hl::MulOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            outMin, outMax, createI32Const(rewriter, srcOp, interleave(bias, scale)), 0, input1,
            input2

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
        bool &needReverse, PatternRewriter &rewriter
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

        auto constOp = lhs.getDefiningOp<arith::ConstantOp>();
        needReverse = (numLinalgInputs == 2 && constOp != nullptr);
        if (!constOp) {
            constOp = rhs.getDefiningOp<arith::ConstantOp>();
        }

        FloatAttr data = fromConstScalar(constOp);
        if (data) {
            if (needReverse) {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[1];
            }
            else {
                // input0 is the tensor input, input1 is the const scalar
                input0 = srcOp.getInputs()[0];
            }

            newBias = data.getValue().convertToFloat();
            input1 = nullptr;
            return success();
        }

        input0 = srcOp.getInputs()[0];
        if (numLinalgInputs == 1) {
            input1 = srcOp.getInputs()[0];
        }
        else {
            input1 = srcOp.getInputs()[1];
        }
        // We take the const tensor as is, no need to invert the inputs
        needReverse = false;
        return success();
    }

    LogicalResult createBf16Add(
        std::string opName, int sign, linalg::GenericOp srcOp, Operation *binaryOp,
        PatternRewriter &rewriter
    ) const {

        bool rhs_is_const = false;

        const llvm::fltSemantics &bf16 = APFloat::BFloat();
        std::vector<llvm::APFloat> weights(2, llvm::APFloat(bf16, "1.0"));
        if (opName == "sub") {
            weights[1] = llvm::APFloat(bf16, "-1.0");
        }
        std::vector<float> biasScale{0.0, 1.0};
        auto torqWeights = createConst(weights, rewriter, srcOp.getLoc());

        if (srcOp.getNumDpsInits() != 1 && srcOp.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "bf16 addOp doesn't support single input with two different operands\n"
            );
        }

        Value input0;
        Value input1;
        float newBias;
        bool needReverse;
        auto res = get2Inputs(srcOp, binaryOp, input0, input1, newBias, needReverse, rewriter);
        if (failed(res)) {
            return res;
        }
        if (!input1) {
            rhs_is_const = true;
            biasScale[0] = newBias;
            input1 = input0;
        }
        if (needReverse && opName == "sub") {
            biasScale[1] = -1;
            opName = "add";
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        // special handling for float min/max
        float min_f = std::numeric_limits<float>::lowest();
        int32_t output_min_f = *reinterpret_cast<int32_t *>(&min_f);
        float max_f = std::numeric_limits<float>::max();
        int32_t output_max_f = *reinterpret_cast<int32_t *>(&max_f);

        rewriter.replaceOpWithNewOp<torq_hl::AddOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            opName,
            0, // input_zp
            0, // output_zp
            output_min_f, output_max_f,
            0, // shift_factor
            torqWeights, createConst(biasScale, rewriter, srcOp.getLoc()), input0, input1, false,
            rhs_is_const
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
        bool rhs_is_const = false;

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
                input1 = rewriter.create<arith::ConstantOp>(srcOp.getLoc(), constType, value)
                             .getResult();

                bias = {data * sign};
                rhs_is_const = true;

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

        auto outputElementSize = srcResultType.getElementType().getIntOrFloatBitWidth() / 8;
        int32_t outMin = -128, outMax = 127;
        getDTypeRange(outputElementSize, &outMin, &outMax);

        rewriter.replaceOpWithNewOp<torq_hl::AddOp>(
            srcOp, srcOp.getResult(0).getType(), createInitTensor(srcOp, rewriter, srcResultType),
            opName,
            0, // input_zp
            0, // output_zp
            outMin, outMax,
            0, // shift_factor
            outType.getElementType().isInteger(32) ? weightsI8 : weightsI16,
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input0, input1, false,
            rhs_is_const
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

        auto bcastOp = input1.getDefiningOp<linalg::BroadcastOp>();
        if (bcastOp) {
            input1 = bcastOp.getInput();
        }

        auto collapseOp = input1.getDefiningOp<tensor::CollapseShapeOp>();
        if (collapseOp) {
            input1 = collapseOp.getSrc();
        }

        while (foldBackwardRescale(input1, scaleInput1)) {
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input0, input1}, rewriter,
                eltOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        if (collapseOp) {
            auto collapseShape = collapseOp.getResultType().getShape();
            auto collapseInputElementType =
                dyn_cast<RankedTensorType>(input1.getType()).getElementType();
            auto newOutputType = RankedTensorType::get(collapseShape, collapseInputElementType);

            auto newCollapseOp = rewriter.create<tensor::CollapseShapeOp>(
                collapseOp.getLoc(), newOutputType, input1, collapseOp.getReassociationIndices()
            );
            collapseOp->getResult(0).replaceAllUsesWith(newCollapseOp.getResult());
            rewriter.eraseOp(collapseOp);
        }

        if (bcastOp) {
            auto bcastInputElementType =
                dyn_cast<RankedTensorType>(input1.getType()).getElementType();

            auto bOutputShape = bcastOp.getInit().getType().getShape();
            auto bOutputType = RankedTensorType::get(bOutputShape, bcastInputElementType);

            auto op = rewriter.create<linalg::BroadcastOp>(
                loc, bcastOp.getInput(), createInitTensor(bcastOp, rewriter, bOutputType),
                bcastOp.getDimensionsAttr()
            );
            rewriter.replaceOp(bcastOp, op.getResults()[0]);

            input1 = op.getResults()[0];
        }

        // Generate torq_hl op with input in the expected format
        input0 = transposeValue(input0, dataPerm, loc, rewriter);
        input1 = transposeValue(input1, dataPerm, loc, rewriter);

        // Compute scale and bias vectors
        const double outputScale = scInfo.scaleDouble[0];
        double multiplier0 = outputScale * scaleInput0.scale;
        double multiplier1 = outputScale * scaleInput1.scale;
        int scaleFactor = 1 << scInfo.scaleShift;
        auto weight0 = doubleToInt<int16_t>(multiplier0 * scaleFactor);
        auto bias0 = -doubleToInt<int32_t>(multiplier0 * scaleFactor * scaleInput0.zp);
        int16_t weight1 = doubleToInt<int16_t>(multiplier1 * scaleFactor) * sign;
        int32_t bias1 = -doubleToInt<int32_t>(multiplier1 * scaleFactor * scaleInput1.zp) * sign;

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

Value applyPaddingIfNeeded(
    Value input, RankedTensorType inputType, PatternRewriter &rewriter, Location loc,
    const PaddingInfo &padInfo
) {
    int64_t padLeft = padInfo.lrtbPad[0];
    int64_t padRight = padInfo.lrtbPad[1];

    bool needsPad = (padLeft > 1 || padRight > 1);
    if (!needsPad)
        return input;

    auto inputShape = inputType.getShape();
    SmallVector<int64_t> paddedShape(inputShape.begin(), inputShape.end());
    // paddedShape.back() += padLeft + padRight;

    auto elemType = inputType.getElementType();

    if (!elemType.isInteger(32)) {
        llvm::errs() << "Unsupported element type for FillOp (only i32 supported)\n";
        return input;
    }

    auto paddedType = RankedTensorType::get(paddedShape, elemType);
    auto initTensor = rewriter.create<tensor::EmptyOp>(loc, paddedShape, elemType);

    auto fillOp = rewriter.create<torq_hl::FillOp>(
        loc,
        paddedType,             // Output type
        initTensor.getResult(), // Init tensor value
        rewriter.getI32IntegerAttr(padInfo.padValue)
    );
    // Create InsertSliceOp: insert input tensor into padded tensor at correct offset
    SmallVector<OpFoldResult> offsets(inputShape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(inputShape.size(), rewriter.getIndexAttr(1));

    for (int64_t dim : inputShape)
        sizes.push_back(rewriter.getIndexAttr(dim));

    offsets.back() = rewriter.getIndexAttr(padLeft); // Insert input starting at padLeft

    Value inserted = rewriter.create<tensor::InsertSliceOp>(
        loc, input, fillOp.getOutput(), offsets, sizes, strides
    );

    return inserted;
}

void dumpModuleToFile(Operation *op, StringRef filename) {
    // Traverse upward to find the parent module
    mlir::Operation *parent = op;
    while (parent && !llvm::isa<mlir::ModuleOp>(parent))
        parent = parent->getParentOp();

    if (!parent) {
        llvm::errs() << "Failed to find parent ModuleOp for dumping IR.\n";
        return;
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(filename, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Failed to open file " << filename << ": " << ec.message() << "\n";
        return;
    }

    // Dump the whole module IR
    parent->print(file, mlir::OpPrintingFlags().useLocalScope());
}

template <class LinalgConvOp, class TorqConvOp>
struct Conv2dConvert : public OpRewritePattern<LinalgConvOp> {
  private:
    using MatchFn = bool(
        ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape, ArrayRef<int64_t> padShape
    );

    const int _channelDim;          // Channel dimension index in data shape
    const Permutation _dataPerm;    // Dim permutation for data transpose
    const Permutation _weightsPerm; // Weights permutation for weight transpose
    const int _shift8b;             // Scale shift for 8-bit integer operations
    const int _shift16b;            // Scale shift for 16-bit integer operations
    MatchFn *_matchFn;              // Function to match the convolution operation
    const bool _markFuseGroups;     // When true, mark the TI operations, don't convert.

  public:
    using OpRewritePattern<LinalgConvOp>::OpRewritePattern;
    Conv2dConvert(
        MLIRContext *context, int channelDim, const Permutation &dataPerm,
        const Permutation &weightsPerm, int shift8b, int shift16b, MatchFn *matchFn,
        bool markFuseGroups
    )
        : OpRewritePattern<LinalgConvOp>(context), _channelDim(channelDim), _dataPerm(dataPerm),
          _weightsPerm(weightsPerm), _shift8b(shift8b), _shift16b(shift16b), _matchFn(matchFn),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgConvOp convOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        constexpr int groups = 1; // We don't use it
        const auto loc = convOp.getLoc();

        // Get the input, weights, and output of the original operation
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];

        auto inputType = llvm::cast<RankedTensorType>(input.getType());
        auto shape = inputType.getShape();
        auto weightType = llvm::cast<RankedTensorType>(weights.getType());
        auto weightShape = weightType.getShape();

        bool isConv1D = (inputType.getRank() == 4 && shape[1] == 1);
        if (isConv1D) {
            return rewriteAsConv1D(convOp, rewriter);
        }

        // Fold padding if present
        PaddingInfo padInfo = foldBackwardPadding(input, rewriter);

        // Check if we can support this layer
        if (_matchFn && !_matchFn(shape, weightShape, padInfo.lrtbPad)) {
            return rewriter.notifyMatchFailure(
                convOp, "Conv does not match expected kernel dimension or padding"
            );
        }

        // Fold any per-channel bias
        const auto outType = cast<RankedTensorType>(output.getType());
        const int outChannelCount = outType.getShape()[_channelDim];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outChannelCount, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold operations that take care of zero-point in weight quantization if present
        int weightZp = foldForwardWeightZp(output);

        // Fold any additional per-channel bias
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        // Fold scale and clamp. This is mandatory for integer operations.
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, false);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                convOp, "Expected scale and clamp info for integer operations"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        // Convert weights to the required format
        DenseIntOrFPElementsAttr weightAttr;
        auto transposedWeights = transposeValue(weights, _weightsPerm, loc, rewriter);
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overlflow
            constexpr int scaleFactor = 2;
            transposedWeights =
                rescaleValue(transposedWeights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }
        weightAttr = computeConstant(transposedWeights);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                convOp, "Failed to create constant for transposed weights"
            );
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with input/output in the expected format
        input = transposeValue(input, _dataPerm, loc, rewriter);
        auto torqOutType = transposeType(output.getType(), _dataPerm);
        bool nhwcInput = _channelDim == 3 && _dataPerm.empty();
        auto torqConvOp = rewriter.create<TorqConvOp>(
            loc, torqOutType, createInitTensor(convOp, rewriter, torqOutType), padInfo.padValue, 0,
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift, groups, padInfo.lrtbPad,
            attrValues(convOp.getStrides()), attrValues(convOp.getDilations()),
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, input, nhwcInput
        );
        auto torqOut = transposeValue(torqConvOp.getOutput(), _dataPerm.reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    template <typename LinalgConv>
    LogicalResult rewriteAsConv1D(LinalgConv convOp, PatternRewriter &rewriter) const {
        if (!isa<linalg::Conv2DNhwcHwcfOp>(convOp)) {
            return rewriter.notifyMatchFailure(
                convOp, "Only linalg::Conv2DNhwcHwcfOp can be rewritten as Conv1D"
            );
        }
        if (_markFuseGroups && isMarkedFuseGroup(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Already marked");
        }

        auto loc = convOp.getLoc();
        constexpr int weightZp = 0;
        constexpr int groups = 1;
        Value input = convOp.image();
        Value weights = convOp.filter();
        Value output = convOp.getResultTensors()[0];
        ::mlir::DenseIntElementsAttr stridesAttr = convOp.getStrides();
        auto strideValue = stridesAttr.getValues<int64_t>()[1];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto outputType = cast<RankedTensorType>(output.getType());
        auto outElemType = outputType.getElementType();
        bool isInt = outElemType.isInteger();
        int outChannels = outputType.getShape()[_channelDim];

        VectorIntOrFloat bias(outChannels, isInt);
        while (foldForwardPerChannelAdd(output, _channelDim, bias)) {
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannels, _shift8b, _shift16b);
        if (!scInfo && isInt)
            return failure();

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(transposeValue(weights, _weightsPerm, loc, rewriter));
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        // TODO: Torq weights should be reorderes in multiple channels cases;
        if (!torqWeights)
            return failure();

        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Note: op is Conv2DNhwcHwcfOp
        int64_t batch = inputType.getShape()[0];
        int64_t channels = inputType.getShape()[3];
        int64_t out_len = outputType.getShape()[2];
        int64_t outputChannels = outputType.getShape()[3];

        auto weightType = cast<RankedTensorType>(weights.getType());
        int64_t filter_len = weightType.getShape()[1];

        int64_t op_rows = filter_len;
        int64_t op_cols = out_len;

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
                attrValues(convOp.getStrides()), weightType.getShape(), permAttr, input
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
            attrValues(convOp.getDilations()), torq_hl::VectorizationModeEnum::None, torqWeights,
            biasScale, input
        );
        Value torqOut = torqConv1Op.getOutput();

        if (useConv1dWithReduce) {
            // Add linalg.reduce to remove the extra dimension
            // Create reducedType from torqOutType by removing the last dimension
            auto reducedShape = torqOutType.getShape().drop_back();

            // Create a tensor filled with zeros of type torqOutType.getElementType()
            Value zeroValue = createZeroConstant(rewriter, loc, torqOutType.getElementType());
            auto cEmpty =
                rewriter.create<tensor::EmptyOp>(loc, reducedShape, torqOutType.getElementType());
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
        // // Overwrite torqOut with init tensor for debugging
        // torqOut = createInitTensor(convOp, rewriter, cast<RankedTensorType>(torqOut.getType()));
        // // Fill input with 1s for debugging
        // torqOut = rewriter.create<torq_hl::FillOp>(
        //     loc, cast<RankedTensorType>(torqOut.getType()), torqOut,
        //     rewriter.getI32IntegerAttr(/*0x3f800000*//*0x00003f80*/0)
        // ).getOutput();

        torqOut = transposeValue(torqOut, _dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }
};

struct Conv1DNcwFcwToLinalgMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgMatmul\n";
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

        // Step 1: Unfold the input tensor using im2col approach
        // Each position in the output corresponds to a patch of the input
        auto elemType = inputType.getElementType();
        auto outputElemType = outputType.getElementType();
        // Create a tensor to hold the unfolded input
        // Shape: [Ow, C*Kw] - each row contains a full patch for one output position
        SmallVector<int64_t> unfoldedShape = {Ow, C * Kw};
        auto unfoldedType = RankedTensorType::get(unfoldedShape, elemType);
        auto unfoldedInit = rewriter.create<tensor::EmptyOp>(loc, unfoldedShape, elemType);

        // Create the im2col transformation using a linalg.generic
        SmallVector<AffineExpr> unfoldIndexExprs;
        auto dim0 = rewriter.getAffineDimExpr(0); // Output position (Ow dimension)
        auto dim1 = rewriter.getAffineDimExpr(1); // Input channel and kernel position

        // dim1 / Kw gives us the channel index
        auto channelIdx = dim1.floorDiv(rewriter.getAffineConstantExpr(Kw));
        // dim1 % Kw gives us the kernel position
        auto kernelIdx = dim1 % rewriter.getAffineConstantExpr(Kw);
        // Calculate input position: outputPos * stride + kernelIdx * dilation
        auto inputPosExpr = dim0 * rewriter.getAffineConstantExpr(stride) +
                            kernelIdx * rewriter.getAffineConstantExpr(dilation);

        unfoldIndexExprs.push_back(rewriter.getAffineConstantExpr(0)); // N dimension (batch)
        unfoldIndexExprs.push_back(channelIdx);                        // C dimension (channels)
        unfoldIndexExprs.push_back(inputPosExpr);                      // W dimension (width)

        auto unfoldIndexMap = AffineMap::get(2, 0, unfoldIndexExprs, rewriter.getContext());
        auto outputIndexMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());

        // Create the generic op for unfolding with explicit iterator types
        SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

        auto im2col = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{unfoldedType}, ValueRange{input}, ValueRange{unfoldedInit},
            ArrayRef<AffineMap>{unfoldIndexMap, outputIndexMap}, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            }
        );

        // Set torq.im2col attribute so that we can easily recognize this op during tiling
        im2col->setAttr("torq.im2col", rewriter.getBoolAttr(true));
        auto unfoldedInput = im2col.getResult(0);

        // Step 2: Reshape the filter tensor from [F, C, Kw] to [F, C*Kw]
        SmallVector<int64_t> reshapedFilterShape = {F, C * Kw};
        auto reshapedFilterType =
            RankedTensorType::get(reshapedFilterShape, filterType.getElementType());
        auto reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
            loc, reshapedFilterType, filter, ArrayRef<ReassociationIndices>{{0}, {1, 2}}
        );

        // Step 3: Create the matmul operation
        // We'll do: [F, C*Kw] @ [Ow, C*Kw]^T -> [F, Ow]
        // First, we need to transpose the unfolded input
        SmallVector<int64_t> transposedUnfoldedShape = {C * Kw, Ow};
        // auto transposedUnfoldedType = RankedTensorType::get(transposedUnfoldedShape, elemType);
        auto transposedUnfoldedInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedUnfoldedShape, elemType);

        auto transposedUnfolded = rewriter.create<linalg::TransposeOp>(
            loc, unfoldedInput, transposedUnfoldedInit, ArrayRef<int64_t>{1, 0}
        );

        // Create the matmul output tensor [F, Ow]
        SmallVector<int64_t> matmulResultShape = {F, Ow};
        auto matmulResultType = RankedTensorType::get(matmulResultShape, outputElemType);
        auto matmulInit = rewriter.create<tensor::EmptyOp>(loc, matmulResultShape, outputElemType);

        // Perform the actual matmul
        // Perform the actual matmul
        SmallVector<Value> inputs;
        inputs.push_back(reshapedFilter.getResult());
        inputs.push_back(transposedUnfolded.getResults()[0]);

        SmallVector<Value> outputs;
        outputs.push_back(matmulInit.getResult());

        auto matmulOp =
            rewriter.create<linalg::MatmulOp>(loc, TypeRange{matmulResultType}, inputs, outputs);

        // Step 4: Reshape the result back to [N, F, Ow]
        if (N == 1) {
            // Simply reshape to add the batch dimension
            auto finalResult = rewriter.create<tensor::ExpandShapeOp>(
                loc, matmulResultType, matmulOp.getResults()[0],
                ArrayRef<ReassociationIndices>{{0, 1}, {2}}
            );

            rewriter.replaceOp(convOp, finalResult);
        }
        else {
            return rewriter.notifyMatchFailure(
                convOp, "Batched Conv1D not supported in this pattern"
            );
        }

        return success();
    }
};

struct Conv1DNcwFcwToLinalgConv2DPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp convOp, PatternRewriter &rewriter) const override {
        llvm::errs() << "Applying Conv1DNcwFcwToLinalgConv2D\n";
        auto loc = convOp.getLoc();

        // Get operands
        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        // Get types and shapes
        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        // Add height dimension (1) to input: [N,C,W] -> [N,C,1,W]
        // Need to use proper reassociation indices
        SmallVector<ReassociationIndices> inputReassoc = {{0}, {1}, {2, 3}};
        auto expandedInputType = RankedTensorType::get(
            {inputType.getShape()[0], inputType.getShape()[1], 1, inputType.getShape()[2]},
            inputType.getElementType()
        );

        auto expandedInput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedInputType, input, inputReassoc);

        // Transpose to NHWC format: [N,C,1,W] -> [N,1,W,C]
        SmallVector<int64_t> inputPerm = {0, 2, 3, 1};
        Value nhwcInput = transposeValue(expandedInput, inputPerm, loc, rewriter);

        // Add height dimension to filter: [F,C,W] -> [F,C,1,W]
        SmallVector<ReassociationIndices> filterReassoc = {{0}, {1}, {2, 3}};
        auto expandedFilterType = RankedTensorType::get(
            {filterType.getShape()[0], filterType.getShape()[1], 1, filterType.getShape()[2]},
            filterType.getElementType()
        );

        auto expandedFilter =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedFilterType, filter, filterReassoc);

        // Transpose to HWCF format: [F,C,1,W] -> [1,W,C,F]
        SmallVector<int64_t> filterPerm = {2, 3, 1, 0};
        Value hwcfFilter = transposeValue(expandedFilter, filterPerm, loc, rewriter);

        // Add height dimension to output: [N,F,W] -> [N,F,1,W]
        SmallVector<ReassociationIndices> outputReassoc = {{0}, {1}, {2, 3}};
        auto expandedOutputType = RankedTensorType::get(
            {outputType.getShape()[0], outputType.getShape()[1], 1, outputType.getShape()[2]},
            outputType.getElementType()
        );

        auto expandedOutput =
            rewriter.create<tensor::ExpandShapeOp>(loc, expandedOutputType, output, outputReassoc);

        // Transpose to NHWC format: [N,F,1,W] -> [N,1,W,F]
        SmallVector<int64_t> outputPerm = {0, 2, 3, 1};
        Value nhwcOutput = transposeValue(expandedOutput, outputPerm, loc, rewriter);

        // Get attributes
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();

        // Convert 1D strides/dilations to 2D (add height dimension)
        SmallVector<int64_t> strides2d = {1};
        strides2d.push_back(stridesAttr.getValues<int64_t>()[0]);
        SmallVector<int64_t> dilations2d = {1};
        dilations2d.push_back(dilationsAttr.getValues<int64_t>()[0]);

        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        // Create Conv2D
        auto conv2d = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
            loc, nhwcOutput.getType(), ValueRange{nhwcInput, hwcfFilter}, ValueRange{nhwcOutput},
            stridesAttr2d, dilationsAttr2d
        );

        // Transpose result back: [N,1,W,F] -> [N,F,1,W]
        Value transposedResult = transposeValue(conv2d.getResult(0), {0, 3, 1, 2}, loc, rewriter);

        // Collapse height dimension: [N,F,1,W] -> [N,F,W]
        auto collapsedResult = rewriter.create<tensor::CollapseShapeOp>(
            loc, outputType, transposedResult, outputReassoc
        );

        rewriter.replaceOp(convOp, collapsedResult.getResult());
        return success();
    }
};

struct FCMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    FCMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        const auto loc = srcOp.getLoc();
        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 input and 1 output");
        }

        Value inputA = srcOp.getInputs()[0];
        Value inputB = srcOp.getInputs()[1]; // weights
        Value output = srcOp.getResultTensors()[0];

        auto inputAType = llvm::cast<RankedTensorType>(inputA.getType());
        auto inputBType = llvm::cast<RankedTensorType>(inputB.getType());
        auto outputType = llvm::cast<RankedTensorType>(output.getType());
        if (inputAType.getRank() != 2 || inputBType.getRank() != 2 || outputType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects 2D inputs and outputs"
            );
        }
        auto inputAShape = inputAType.getShape();
        if (inputAShape[0] != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "FCMatmulOpConversion expects inputA shape[0] == 1"
            );
        }

        auto outputChannelCount = outputType.getShape()[1];
        bool isInt = outputType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp)) {
        }

        // check if output user is expand_shape
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            auto op = *output.getUsers().begin();
            if (_markFuseGroups) {
                output = op->getResult(0);
            }
            else {
                op->getResult(0).replaceAllUsesWith(output);
                rewriter.eraseOp(op);
            }
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected scale and clamp info for integer operations"
            );
        }

        // check if output is a tensor::CollapseShapeOp
        if (output.hasOneUse() && (isa<tensor::CollapseShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = output.getUsers().begin()->getResult(0);
        }

        // Prepare weights
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(inputB.getDefiningOp())) {
            inputB = transposeOp.getInput();
            // NOTE: inputB changed, re-get its type if need to process related
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {inputA, inputB}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weightAttr = computeConstant(inputB);
        auto torqWeights = createConst(weightAttr, rewriter, loc);
        if (!torqWeights) {
            return rewriter.notifyMatchFailure(
                srcOp, "Failed to create constant for transposed weights"
            );
        }
        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // get new output type as above various changes for output
        outputType = llvm::cast<RankedTensorType>(output.getType());

        auto fcOp = rewriter.create<torq_hl::FullyConnectedOp>(
            loc, outputType, createInitTensor(srcOp, rewriter, outputType), input_zp,
            0, // weight zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            torq_hl::VectorizationModeEnum::None, torqWeights, biasScale, inputA
        );
        rewriter.replaceOp(output.getDefiningOp(), fcOp.getOutput());

        return success();
    }
};

struct Conv2DMatmulOpConversion : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    Conv2DMatmulOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    Value convertWeights(
        mlir::linalg::MatmulOp srcOp, mlir::DenseIntOrFPElementsAttr weightAttr,
        PatternRewriter &rewriter
    ) const {
        // Reorder weights to OIHW
        auto weightElemType = weightAttr.getElementType();
        auto weightShape = dyn_cast<ShapedType>(weightAttr.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        assert(weightShape.size() == 2);

        // Assume shape was originally [OC, IC] from matmul-style
        int on = weightShape[0]; // OC
        int in = weightShape[1]; // IC
        int hn = 1;
        int wn = 1;
        std::vector<int64_t> weight_shape{on, in, hn, wn};

        if (weightElemType.isBF16()) {
            auto bfVals = weightAttr.getValues<APFloat>();
            const std::vector<APFloat> bfVec(bfVals.begin(), bfVals.end());
            std::vector<APFloat> reordered = get_weights_OIHW<APFloat>(bfVec, on, hn, wn, in);
            return createFConst(rewriter, srcOp, reordered, weight_shape);
        }
        else if (weightElemType.isInteger(8)) {
            auto rawVals = weightAttr.getValues<int8_t>();
            std::vector<int8_t> reordered(rawVals.begin(), rawVals.end());
            reordered = get_weights_OIHW<int8_t>(reordered, on, hn, wn, in);
            return createI8Const(rewriter, srcOp, reordered, weight_shape);
        }
        else {
            assert(false && "Unsupported weight type");
        }
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        Location loc = srcOp.getLoc();

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 1 input and 1 output");
        }

        Value lhs = srcOp.getInputs()[0];
        Value rhs = srcOp.getInputs()[1];
        Value output = srcOp.getResultTensors()[0];

        // Ensure inputs and output are 2D
        auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
        auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
        auto outType = llvm::cast<RankedTensorType>(output.getType());

        if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
            outType.getRank() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Conv2DMatmulOpConversion expects 2D inputs and outputs"
            );
        }

        // Check if the Conv2D input (lhs) is produced by a CollapseShapeOp —
        // this typically means the input tensor is being flattened before the convolution.
        while (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(lhs.getDefiningOp())) {
            lhs = extractSlice.getSource();
        }
        if (!lhs.getDefiningOp<tensor::CollapseShapeOp>() &&
            !isCollapseOrExpandShapeGeneric(lhs.getDefiningOp())) {
            return rewriter.notifyMatchFailure(srcOp, "LHS is not collapsed from 4D");
        }
        Value input = lhs.getDefiningOp()->getOperand(0);
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || inputType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected input to be 4D pre-collapse");
        }

        // Match transpose on weight
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(rhs.getDefiningOp())) {
            rhs = transposeOp.getInput();
        }

        // Check weights are supported
        auto weightElemType = dyn_cast<RankedTensorType>(rhs.getType()).getElementType();

        if (!weightElemType.isBF16() && !weightElemType.isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "Unsupported weight type");
        }

        auto weightShape = dyn_cast<ShapedType>(rhs.getType()).getShape();

        // Validate expected shape: [OC, IC] for matmul-style
        if (weightShape.size() != 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 2D weight tensor");
        }

        // fold bias
        auto outputChannelCount = outType.getShape()[1];
        bool isInt = outType.getElementType().isInteger();
        VectorIntOrFloat bias(outputChannelCount, isInt);
        int32_t input_zp = 0;
        int32_t weightZp = 0;

        while (foldForwardPerChannelAdd(output, 1, bias, &input_zp, input, &weightZp)) {
        }

        // check if output user is expand_shape
        RankedTensorType finalType = nullptr;
        if (output.hasOneUse() && (isa<tensor::ExpandShapeOp>(*output.getUsers().begin()) ||
                                   isCollapseOrExpandShapeGeneric(*output.getUsers().begin()))) {
            output = (*output.getUsers().begin())->getResult(0);
            finalType = cast<RankedTensorType>(output.getType());
        }
        else {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected expand_shape user to determine 4D output"
            );
        }

        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outputChannelCount, 20, 12);
        if (!scInfo) {
            if (isInt) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Expected scale info for integer operations"
                );
            }
            scInfo = getDefaultScaleClampInfo(finalType, srcOp);
        }
        else {
            finalType = cast<RankedTensorType>(output.getType());
        }
        if (!finalType || finalType.getRank() != 4) {
            return rewriter.notifyMatchFailure(srcOp, "Expected 4D output from expand");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, rhs}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto weights = rhs;
        if (weightZp != 0) {
            // Divide weights and wZp by two so we can safely add them together without overflow
            constexpr int scaleFactor = 2;
            weights = rescaleValue(weights, scaleFactor, -weightZp / 2, loc, rewriter);
            // Same for the bias
            for (auto &val : bias.ints) {
                val /= scaleFactor;
            }
            // Reduce the scaling shift to compensate
            // (We could multiply the scaling factor by 2 instead, but that could cause overflow)
            scInfo.scaleShift -= 1;
        }

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scInfo.scaleNpu), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Compute weights
        auto weightAttr = computeConstant(weights);
        if (!weightAttr) {
            return rewriter.notifyMatchFailure(srcOp, "Failed to fold weights");
        }

        finalType = convertTypeNHWCtoNCHW(finalType);
        Value initTensor = createInitTensor(srcOp, rewriter, finalType);
        auto vectorizationMode = torq_hl::VectorizationModeEnum::None;
        input = transposeValue(input, Permutation::nhwc2nchw(), loc, rewriter);

        auto pad = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
        auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
        auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

        auto torqWeights = convertWeights(srcOp, weightAttr, rewriter);

        auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
            loc, finalType, initTensor,
            input_zp, // input_zp
            0,        // weight_zp
            scInfo.zp, scInfo.min, scInfo.max, scInfo.scaleShift,
            1,        // groups
            pad,      // pad
            stride,   // stride
            dilation, // dilation
            vectorizationMode, torqWeights, biasScale, input
        );

        auto torqOut =
            transposeValue(conv2dOp.getOutput(), Permutation::nhwc2nchw().reverse(), loc, rewriter);
        rewriter.replaceOp(output.getDefiningOp(), torqOut);

        LLVM_DEBUG({ llvm::dbgs() << "Conv2DMatmulOpConversion success\n"; });
        return success();
    }
};

class ClampOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        std::string failReason;
        int32_t minIntValue = 0, maxIntValue = 0;
        float minFloatValue = 0.0f, maxFloatValue = 0.0f;

        if (!isTorqClampOp(
                srcOp, minIntValue, maxIntValue, minFloatValue, maxFloatValue, failReason
            )) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        Value input = srcOp.getInputs()[0];
        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, srcResultType, createInitTensor(srcOp, rewriter, srcResultType), "clamp", 0, 0,
            minIntValue, maxIntValue,
            APFloat(llvm::APFloat::IEEEsingle(), std::to_string(minFloatValue)),
            APFloat(llvm::APFloat::IEEEsingle(), std::to_string(maxFloatValue)), input
        );

        return success();
    }
};

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
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getInputs()[0]
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
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input
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
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input
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
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getInputs()[0]
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
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), input
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

        if (!isTorqCastOp(srcOp, opName, failReason)) {
            return rewriter.notifyMatchFailure(srcOp, failReason);
        }

        auto resultType = cast<RankedTensorType>(srcOp.getResult(0).getType());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opName, 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getInputs()[0]
        );

        return success();
    }
};

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

            // TODO: add UNE, OLE, OLT

            auto predicate = cmpFOp.getPredicate();

            if (predicate == arith::CmpFPredicate::OGE) {
                opType = torq_hl::ElementwiseOpEnum::GREATER_EQUAL;
            }
            else if (predicate == arith::CmpFPredicate::OGT) {
                opType = torq_hl::ElementwiseOpEnum::GREATER;
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

            // TODO: add ne, sle, slt

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
            else if (predicate == arith::CmpIPredicate::ult) {
                // Unsigned less than => Reverse inputs and use GREATER
                opType = torq_hl::ElementwiseOpEnum::GREATER;
                swapInputs = true;
                isUnsigned = true;
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

        auto opName = castOpName(srcOp.getIn(), srcOp.getOut());

        rewriter.replaceOpWithNewOp<torq_hl::ActOp>(
            srcOp, resultType, createInitTensor(srcOp, rewriter, resultType), opName, 0, 0, 0, 0,
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"),
            APFloat(llvm::APFloat::IEEEsingle(), "0.0"), srcOp.getIn()
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

        auto op = rewriter.create<torq_hl::BroadcastOp>(
            srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType),
            srcOp.getDimensionsAttr(), srcOp.getInput()
        );
        rewriter.replaceOp(srcOp, op.getOutput());

        return success();
    }
};

struct PoolingNhwcMaxOpConversion : public OpRewritePattern<linalg::PoolingNhwcMaxOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    PoolingNhwcMaxOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNhwcMaxOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }
        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResults()[0];

        auto attrStrides = attrValues(srcOp.getStrides());
        if (attrStrides.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two strides for PoolingNhwcMaxOp"
            );
        }
        if (attrStrides[0] > 2 || attrStrides[1] > 2) {
            return rewriter.notifyMatchFailure(srcOp, "Expected strides <= 2 for PoolingNhwcMaxOp");
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};
        const std::vector<int8_t> weight = {1};

        PaddingInfo padInfo = foldBackwardPadding(input, rewriter);

        auto kernels = mlir::cast<RankedTensorType>(srcOp.getInputs()[1].getType()).getShape();
        if (kernels.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingNhwcMaxOp"
            );
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input}, rewriter,
                srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        auto dataPerm =
            srcResultType.getRank() == 4 ? Permutation::nhwc2nchw() : Permutation::none();

        input = transposeValue(input, dataPerm, loc, rewriter);
        srcResultType = transposeType(srcResultType, dataPerm);

        auto maxpoolOp = rewriter.create<torq_hl::MaxPool2dOp>(
            loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType), padInfo.padValue,
            attrStrides, padInfo.lrtbPad, kernels,
            createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1}),
            createI32Const(rewriter, srcOp, interleave(bias, scale)), input
        );
        auto result = transposeValue(maxpoolOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), result);

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

        auto output =
            create1DimTensorFromRescaleScalar(srcOp, applyScaleOp, outputElementType, rewriter);

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

        // shiftFactor is used for torq hw scale computation, it request multiple of 4.
        // for 48bit value rescale, we need to check multiplier bit < 32bit in case overflow 80bit.
        int32_t shiftFactor = 4;
        double rnd_err = 0.5;

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
        else {

            input = srcOp.getInputs()[0];
            if (!input) {
                llvm::errs() << "RescaleOpConversion: input is null\n";
            }

            auto inputType = dyn_cast<RankedTensorType>(input.getType());
            auto inputElementType = inputType.getElementType();

            if (inputElementType.isF32() || inputElementType.isBF16() ||
                outputElementType.isF32() || outputElementType.isBF16()) {
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

                if (outputElementType.isInteger(8) &&
                    (inputElementType.isInteger(16) || inputElementType.isInteger(32))) {
                    // FIXME: not clear in which cases we actually need this shiftFactor
                    // TODO: compute this shiftFactor instead of hardcoding it
                    // 12 to make sure rescaled value doens't overflow 64bit
                    shiftFactor = 12;
                }

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

        double scaleFactor = static_cast<double>(ms.multiplier[0]) / (1l << ms.shift[0]);

        int16_t weight_data = static_cast<int32_t>(scaleFactor * (1 << shiftFactor) + rnd_err);
        int32_t bias_data = -static_cast<int32_t>(scaleFactor * (1 << shiftFactor) * input_zp);

        std::vector<int16_t> weights = {weight_data};
        const std::vector<APInt> bias = {APInt(32, bias_data)};
        const std::vector<APInt> scale = {APInt(32, 1)};

        LLVM_DEBUG({
            llvm::dbgs() << "rescale params : input_zp: " << input_zp << ", "
                         << "outputZp: " << outputZp << ", outputMin: " << outputMin << ", "
                         << "outputMax: " << outputMax << ", shiftFactor: " << shiftFactor << ", "
                         << "weight_data: " << weight_data << ", bias_data: " << bias_data << "\n";
        });

        auto fmaOp = rewriter.create<torq_hl::FMAOp>(
            srcOp.getLoc(), outputType, createInitTensor(srcOp, rewriter, outputType), outputZp,
            outputMin, outputMax, shiftFactor,
            createI16Const(rewriter, srcOp, weights, llvm::ArrayRef<int64_t>{1}),
            createIConst(rewriter, srcOp, interleave(bias, scale)), input
        );
        rewriter.replaceOp(srcOp, fmaOp.getOutput());

        return success();
    }
};

// TODO: only support int8 for now in order to use generic tiling for table op
class ExtractOpPattern : public OpRewritePattern<linalg::GenericOp> {
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

        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a linalg.yield terminator");
        }

        auto tensorExtractOp = yieldOp.getValues()[0].getDefiningOp<tensor::ExtractOp>();
        if (!tensorExtractOp) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a tensor.extract op");
        }

        auto &block = srcOp.getRegion().front();
        auto &firstOp = block.front();

        auto indices = tensorExtractOp.getIndices();
        bool isTableOp = false;
        if (!indices.empty()) {
            Value idx = indices[0];

            auto addOp = dyn_cast<arith::AddIOp>(idx.getDefiningOp());
            if (addOp) {
                isTableOp = true;
                if (!inputElementType.isInteger(8)) {
                    return rewriter.notifyMatchFailure(srcOp, "Only i8 TableOp supported");
                }

                auto indexCastOp = dyn_cast<arith::IndexCastOp>(firstOp);
                if (!indexCastOp) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "Expected firstOp is a arith.indexCast op"
                    );
                }
                // one operands of addOp is indexCastOp result
                Value indexCastResult = indexCastOp.getResult();
                if (indexCastResult != addOp.getLhs() && indexCastResult != addOp.getRhs()) {
                    return rewriter.notifyMatchFailure(
                        srcOp, "we expect one addOp operand from indexCastOp"
                    );
                }
            }
        }

        Value cst = tensorExtractOp.getTensor();
        if (!cst) {
            return rewriter.notifyMatchFailure(srcOp, "Expected a arith.addi op");
        }

        SmallVector<int32_t> convertedValues;
        auto cstData = computeConstant(cst);
        if (!clTableAsGather && isTableOp) {
            auto values = cstData.getValues<int8_t>();
            for (size_t i = 0; i < 256; i++) {
                int32_t shiftedValue = static_cast<int32_t>(values[i]) << 8;
                convertedValues.push_back(shiftedValue);
            }

            DenseI32ArrayAttr intArrayAttr =
                DenseI32ArrayAttr::get(rewriter.getContext(), convertedValues);

            const std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
            const std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};

            auto resultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
            rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
                srcOp, resultType, createInitTensor(srcOp, rewriter, resultType),
                createIConst(rewriter, srcOp, interleave(bias, scale)), input, intArrayAttr
            );
        }
        else {
            std::vector<APInt> tableValues;

            if (isTableOp) { // If a table converted Gather then the const values are input(table)
                             // and extract tensor is indices
                auto values = cstData.getValues<APInt>();
                tableValues.insert(tableValues.end(), values.begin(), values.end());
                int tableSize = tableValues.size();
                std::vector<APInt> modifiedTable;
                modifiedTable.reserve(tableSize);
                modifiedTable.insert(
                    modifiedTable.end(), tableValues.begin() + 128, tableValues.end()
                );
                modifiedTable.insert(
                    modifiedTable.end(), tableValues.begin(), tableValues.begin() + 128
                );
                tableValues = modifiedTable;

                auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
                rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                    srcOp, outType, createInitTensor(srcOp, rewriter, outType),
                    createIConst(rewriter, srcOp, tableValues), input
                );
            }
            else if (cstData) { // If not a table converted Gather and the const values are
                                // input(table)
                                // and extract tensor is indices
                auto values = cstData.getValues<APInt>();
                tableValues.insert(tableValues.end(), values.begin(), values.end());
                auto outType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());
                rewriter.replaceOpWithNewOp<syna::torq_hl::GatherOp>(
                    srcOp, outType, createInitTensor(srcOp, rewriter, outType),
                    createIConst(rewriter, srcOp, tableValues), input
                );
            }
            else { // If not a table converted Gather and the const values are indices and extract
                   // tensor is input
                auto dataPerm = Permutation::nhc2nch();
                auto outType = transposeType(
                    mlir::cast<RankedTensorType>(srcOp.getResult(0).getType()), dataPerm.reverse()
                );
                auto transposedValue = transposeValue(cst, dataPerm, srcOp.getLoc(), rewriter);
                auto out = rewriter.create<syna::torq_hl::GatherOp>(
                    srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                    transposedValue, input
                );
                auto resultTranspose =
                    transposeValue(out.getResult(0), dataPerm.reverse(), srcOp.getLoc(), rewriter);
                rewriter.replaceOp(srcOp, resultTranspose);
            }
        }

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

// Converts a `tensor.collapse_shape` operation into a `linalg.generic`
// operation that explicitly materializes the collapsed tensor via a copy.
// Note that linalg implementation of TilingInterface requires that the indexing_map of the output
// satisfies AffineMap::isProjectedPermutation. For this reason we start from the output map being
// the identity map, and construct the input map using floorDiv and mod.
// Example:
// source:
//   tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<10x20x30x40xi8> into tensor<6000x40xi8>
// target:
//   %2 = tensor.empty() : tensor<6000x40xi8>
//   linalg.generic {
//     indexing_maps = [
//       (d0, d1) -> ((d0 / 20*30) mod 10, (d0 / 30) mod 20, d0 mod 30, d1),
//       (d0, d1) -> (d0, d1)
//     ],
//     iterator_types = ["parallel", "parallel"]
//   }
//   ins(%1, tensor<10x20x30x40xi8>)
//   outs(%2 : tensor<6000x40xi8>) {
//   ^bb0(%in: i8, %out: i8) :
//     linalg.yield %in : i8
//   } -> tensor<6000x40xi8>
struct CollapseShapeOpToLinalgRewrite : public OpRewritePattern<tensor::CollapseShapeOp> {
    using OpRewritePattern<tensor::CollapseShapeOp>::OpRewritePattern;
    CollapseShapeOpToLinalgRewrite(MLIRContext *context)
        : OpRewritePattern<tensor::CollapseShapeOp>(context) {}
    LogicalResult
    matchAndRewrite(tensor::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {

        // We only convert the operation if it's part of a pattern that will be converted to torqHL.
        if (!isMarkedFuseGroup(collapseOp))
            return failure();

        Value inputTensor = collapseOp.getSrc();
        TensorType inputTensorType = cast<TensorType>(inputTensor.getType());
        int64_t numInputDims = inputTensorType.getRank();

        // TODO: should this be RankedTensorType?
        TensorType outputTensorType = cast<TensorType>(collapseOp.getResult().getType());
        int64_t numOutputDims = outputTensorType.getRank();

        Value emptyOutput = rewriter.create<tensor::EmptyOp>(
            collapseOp.getLoc(), collapseOp.getResult().getType(), mlir::ValueRange()
        );

        // identity map (d0, d1, ...) -> (d0, d1, ...).
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(numOutputDims);

        SmallVector<AffineExpr, 4> inputMapExprs(numInputDims);

        // Iterate through the reassociation groups defined by `tensor.collapse_shape`.
        // Each group specifies a set of original dimensions that collapse into a single
        // new output dimension.
        for (auto [groupIndex, groupAttr] : llvm::enumerate(collapseOp.getReassociation())) {
            ArrayAttr reassocGroup = cast<ArrayAttr>(groupAttr);

            int64_t sizeProd = 1;

            // Iterate through the original dimensions *within the group* from right to left
            // (i.e., from the innermost dimension to the outermost dimension that contributed
            // to the current collapsed dimension).
            for (int i = reassocGroup.size() - 1; i >= 0; --i) {
                int64_t originalDimIdx = cast<IntegerAttr>(reassocGroup[i]).getInt();
                int64_t originalDimSize = inputTensorType.getDimSize(originalDimIdx);

                inputMapExprs[originalDimIdx] =
                    rewriter.getAffineDimExpr(groupIndex).floorDiv(sizeProd);
                if (i > 0) {
                    // The mod in the last index (left most) is redundant. This
                    // also eliminates the mod from singletons.
                    inputMapExprs[originalDimIdx] = inputMapExprs[originalDimIdx] % originalDimSize;
                }
                sizeProd = sizeProd * originalDimSize;
            }
        }

        // TODO: are we allowed to bypass the rewriter like this?
        AffineMap inputMap = AffineMap::get(numOutputDims, 0, inputMapExprs, rewriter.getContext());

        SmallVector<utils::IteratorType, 4> iteratorTypes(
            numOutputDims, utils::IteratorType::parallel
        );

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            collapseOp,
            /*resultTypes=*/TypeRange(outputTensorType),
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{emptyOutput},
            /*indexingMaps=*/llvm::ArrayRef({inputMap, outputMap}),
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                // The body of the linalg.generic: simply yield the input value.
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            },
            // It's important that we clone the torq-fuse-group attribute
            collapseOp->getAttrs()
        );

        return success();
    }
};

// Converts a `tensor.expand_shape` operation into a `linalg.generic`
// operation that explicitly materializes the expand tensor via a copy.
// Example:
// source:
//   tensor.expand_shape %1 [[0, 1, 2], [3]] output_shape [10, 20, 30, 40] : tensor<6000x40xi8> into
//   tensor<10x20x30x40xi8>
// target:
//   %2 = tensor.empty() : tensor<10x20x30x40xi8>
//   linalg.generic {
//     indexing_maps = [
//       (d0, d1, d2, d3) -> (d0*20*30 + d1*30 + d2, d3)
//       (d0, d1, d2, d3) -> (d0, d1, d2, d3),
//     ],
//     iterator_types = ["parallel", "parallel"]
//   }
//   ins(%1 : tensor<6000x40xi8>)
//   outs(%2, tensor<10x20x30x40xi8>) {
//   ^bb0(%in: i8, %out: i8) :
//     linalg.yield %in : i8
//   } -> tensor<10x20x30x40xi8>
struct ExpandShapeOpToLinalgRewrite : public OpRewritePattern<tensor::ExpandShapeOp> {
    using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;
    ExpandShapeOpToLinalgRewrite(MLIRContext *context)
        : OpRewritePattern<tensor::ExpandShapeOp>(context) {}
    LogicalResult
    matchAndRewrite(tensor::ExpandShapeOp expandOp, PatternRewriter &rewriter) const override {

        // We only convert the operation if it's part of a pattern that will be converted to torqHL.
        if (!isMarkedFuseGroup(expandOp))
            return failure();

        Value inputTensor = expandOp.getSrc();
        TensorType inputTensorType = cast<TensorType>(inputTensor.getType());
        int64_t numInputDims = inputTensorType.getRank();

        // TODO: should this be RankedTensorType?
        TensorType outputTensorType = cast<TensorType>(expandOp.getResult().getType());
        int64_t numOutputDims = outputTensorType.getRank();

        Value emptyOutput = rewriter.create<tensor::EmptyOp>(
            expandOp.getLoc(), expandOp.getResult().getType(), mlir::ValueRange()
        );

        // identity map (d0, d1, ...) -> (d0, d1, ...).
        AffineMap outputMap = rewriter.getMultiDimIdentityMap(numOutputDims);

        SmallVector<AffineExpr, 4> inputMapExprs(numInputDims);

        // Iterate through the reassociation groups defined by `tensor.collapse_shape`.
        // Each group specifies a set of original dimensions that collapse into a single
        // new output dimension.
        for (auto [groupIndex, groupAttr] : llvm::enumerate(expandOp.getReassociation())) {
            ArrayAttr reassocGroup = cast<ArrayAttr>(groupAttr);

            AffineExpr expr = rewriter.getAffineConstantExpr(0);
            AffineExpr exprDimSize = rewriter.getAffineConstantExpr(1);

            // Iterate through the original dimensions *within the group* from right to left
            // (i.e., from the innermost dimension to the outermost dimension that contributed
            // to the current expanded dimension).
            for (size_t i = reassocGroup.size(); i > 0; --i) {
                int64_t originalDimIdx = cast<IntegerAttr>(reassocGroup[i - 1]).getInt();
                int64_t originalDimSize = outputTensorType.getDimSize(originalDimIdx);
                AffineExpr originalDimExpr = rewriter.getAffineDimExpr(originalDimIdx);

                expr = expr + (originalDimExpr * exprDimSize);
                exprDimSize = exprDimSize * originalDimSize;
            }
            inputMapExprs[groupIndex] = expr;
        }

        // TODO: are we allowed to bypass the rewriter like this?
        AffineMap inputMap = AffineMap::get(numOutputDims, 0, inputMapExprs, rewriter.getContext());

        SmallVector<utils::IteratorType, 4> iteratorTypes(
            numOutputDims, utils::IteratorType::parallel
        );

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            expandOp,
            /*resultTypes=*/TypeRange(outputTensorType),
            /*inputs=*/ValueRange{inputTensor},
            /*outputs=*/ValueRange{emptyOutput},
            /*indexingMaps=*/llvm::ArrayRef({inputMap, outputMap}),
            /*iteratorTypes=*/iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
                // The body of the linalg.generic: simply yield the input value.
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, blockArgs[0]);
            },
            // It's important that we clone the torq-fuse-group attribute
            expandOp->getAttrs()
        );

        return success();
    }
};

} // namespace

// Checker methods for convolutions with input: NHWC, weights: HWC(F)
struct Check {
    static constexpr int ih = 1, kh = 0;
    static constexpr int iw = ih + 1, kw = kh + 1;
    static constexpr int maxKerHW = 9;
    using Shape = ArrayRef<int64_t>;

    // Check that the kernel shape is small enough
    static bool isKerSmall(Shape iShape, Shape wShape, Shape padShape) {
        return iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] <= maxKerHW &&
               wShape[kw] <= maxKerHW;
    }

    // Check that the kernel shape is equal to the input shape (without padding)
    static bool isKerEqInput(Shape iShape, Shape wShape, Shape padShape) {
        bool noPadding = llvm::all_of(padShape, [](auto p) { return p == 0; });
        return noPadding && iShape.size() == 4 && wShape.size() >= 3 && wShape[kh] > 1 &&
               wShape[kw] > 1 && iShape[ih] == wShape[kh] && iShape[iw] == wShape[kw];
    }
};

struct TensorBitcastPattern : public OpRewritePattern<tensor::BitcastOp> {
    TensorBitcastPattern(MLIRContext *context)
        : OpRewritePattern<tensor::BitcastOp>(context, /*benefit=*/0) {
        setDebugName("TensorBitcastPattern");
    }
    LogicalResult
    matchAndRewrite(tensor::BitcastOp bitcastOp, PatternRewriter &rewriter) const override {

        auto inputType = dyn_cast<RankedTensorType>(bitcastOp.getSource().getType());
        auto resultType = dyn_cast<RankedTensorType>(bitcastOp.getResult().getType());

        if (!inputType || !resultType) {
            return failure();
        }

        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            bitcastOp.getLoc(), resultType.getShape(), resultType.getElementType()
        );

        size_t rank = inputType.getRank();

        SmallVector<AffineMap> maps{
            2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };
        SmallVector<utils::IteratorType> iteratorTypes{rank, utils::IteratorType::parallel};

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            bitcastOp, resultType, ValueRange{bitcastOp.getSource()}, ValueRange{emptyOp}, maps,
            iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
                auto castOp = nestedBuilder.create<arith::BitcastOp>(
                    loc, resultType.getElementType(), args[0]
                );
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{castOp});
            }
        );

        return success();
    }
};

template <typename OpTy> struct ArithOnTensorToLinalgPattern : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy origOp, PatternRewriter &rewriter) const override {

        Operation *op = origOp.getOperation();

        if (op->getNumResults() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected one result");
        }

        auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());

        if (!resultType) {
            return rewriter.notifyMatchFailure(op, "Expected ranked tensor type");
        }

        SmallVector<RankedTensorType> operandTypes;

        for (auto operand : op->getOperands()) {
            auto rankedType = dyn_cast<RankedTensorType>(operand.getType());

            if (!rankedType) {
                return rewriter.notifyMatchFailure(op, "Expected ranked tensor type");
            }

            if (rankedType.getShape() != resultType.getShape()) {
                return rewriter.notifyMatchFailure(op, "Expected same shape for all operands");
            }

            operandTypes.push_back(rankedType);
        }

        size_t rank = resultType.getRank();

        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            op->getLoc(), resultType.getShape(), resultType.getElementType()
        );

        SmallVector<AffineMap> maps{
            op->getNumOperands() + 1, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())
        };

        SmallVector<utils::IteratorType> iteratorTypes{rank, utils::IteratorType::parallel};

        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            op, resultType, op->getOperands(), ValueRange{emptyOp}, maps, iteratorTypes,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange args) {
                SmallVector<Value> argsVec;
                for (size_t i = 0; i < op->getNumOperands(); i++) {
                    argsVec.push_back(args[i]);
                }
                auto value =
                    nestedBuilder.create<OpTy>(loc, args.back().getType(), argsVec, op->getAttrs());
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{value});
            }
        );

        return success();
    }
};

// A pattern to lower linalg.quantized_batch_matmul to the following sequence:
// 1) linalg.add i8, i8 -> i16 (to handle zero point)
// 2) linalg.matmul i16, i16 -> i32
struct QuantizedBatchMatmulPattern : public OpRewritePattern<linalg::QuantizedBatchMatmulOp> {
    QuantizedBatchMatmulPattern(MLIRContext *context)
        : OpRewritePattern<linalg::QuantizedBatchMatmulOp>(context, /*benefit=*/0) {
        setDebugName("QuantizedBatchMatmulPattern");
    }

    // Add a dynamic scalar (0-D tensor) to a tensor<i16> using linalg.generic.
    Value addScalar0DWithGeneric(
        PatternRewriter &rewriter, Location loc, Value tensorI16, Value negZpI16
    ) const {
        auto tTy = cast<RankedTensorType>(tensorI16.getType());
        Type i16 = tTy.getElementType();

        // Wrap the scalar as tensor<i16> (rank-0).
        RankedTensorType scalarTy = RankedTensorType::get({}, i16);
        auto empty = rewriter.create<tensor::EmptyOp>(loc, scalarTy, ValueRange{});
        Value scalar0D = rewriter.create<tensor::InsertOp>(loc, negZpI16, empty, ValueRange{});

        // FIXME: we don't support tensor-scalar arith ops when scalar is non constant
        // so we create a tensor for the scalar with the same shape as the input tensor.
        // This is not efficient but should be ok for now as the scalar is expected to be
        // a zero point.
        Value cEmpty = rewriter.create<tensor::EmptyOp>(loc, tTy.getShape(), i16);

        SmallVector<int64_t> dim(tTy.getRank());
        for (int i = 0; i < tTy.getRank(); i++) {
            dim[i] = i;
        }
        Value cInit =
            rewriter.create<linalg::BroadcastOp>(loc, scalar0D, cEmpty, dim).getResults()[0];

        // Identity for the tensor, scalar map ()->() for the 0-D input, identity for the output.
        AffineMap id = rewriter.getMultiDimIdentityMap(tTy.getRank());
        SmallVector<utils::IteratorType> iters(tTy.getRank(), utils::IteratorType::parallel);

        Value out = rewriter.create<tensor::EmptyOp>(loc, tTy.getShape(), tTy.getElementType());
        auto g = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{tTy}, ValueRange{tensorI16, cInit}, ValueRange{out},
            ArrayRef<AffineMap>{id, id, id}, iters,
            [&](OpBuilder &b, Location l, ValueRange args) {
                // TODO: double-check if zeropoint must be substracted or added using llvm-cpu with
                // a given input matrix
                Value sum = b.create<arith::SubIOp>(l, args[0], args[1]);
                b.create<linalg::YieldOp>(l, sum);
            }
        );
        return g.getResult(0);
    }

    LogicalResult
    matchAndRewrite(linalg::QuantizedBatchMatmulOp op, PatternRewriter &rewriter) const override {

        Location loc = op.getLoc();

        // Expect: A(i8), B(i8), zpL(any int), zpR(any int); output init i32 tensor.
        Value aI8 = op.getDpsInputOperand(0)->get();
        Value bI8 = op.getDpsInputOperand(1)->get();
        Value zpLRaw = op->getOperand(2);
        Value zpRRaw = op->getOperand(3);
        Value outInit = op.getDpsInitOperand(0)->get();

        auto aTy = dyn_cast<RankedTensorType>(aI8.getType());
        auto bTy = dyn_cast<RankedTensorType>(bI8.getType());
        auto outTy = dyn_cast<RankedTensorType>(outInit.getType());
        if (!aTy || !bTy || !outTy)
            return rewriter.notifyMatchFailure(op, "expected ranked tensors");
        if (!aTy.getElementType().isInteger(8) || !bTy.getElementType().isInteger(8) ||
            !outTy.getElementType().isInteger(32))
            return rewriter.notifyMatchFailure(op, "expected A/B i8 and out i32");

        Type i16 = rewriter.getIntegerType(16);
        Value aI16 =
            rewriter.create<arith::ExtSIOp>(loc, RankedTensorType::get(aTy.getShape(), i16), aI8);
        Value bI16 =
            rewriter.create<arith::ExtSIOp>(loc, RankedTensorType::get(bTy.getShape(), i16), bI8);

        auto zpL16 = rewriter.create<arith::TruncIOp>(loc, i16, zpLRaw); // i8/i16 -> i16
        auto zpR16 = rewriter.create<arith::TruncIOp>(loc, i16, zpRRaw); // i8/i16 -> i16

        Value aAdj = addScalar0DWithGeneric(rewriter, loc, aI16, zpL16);
        Value bAdj = addScalar0DWithGeneric(rewriter, loc, bI16, zpR16);

        // Zero i32 accumulator and (batch_)matmul i16×i16 → i32.
        Value cEmpty =
            rewriter.create<tensor::EmptyOp>(loc, outTy.getShape(), outTy.getElementType());
        Value z32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
        Value cInit =
            rewriter.create<linalg::FillOp>(loc, ValueRange{z32}, ValueRange{cEmpty}).result();

        Value result;
        if (outTy.getRank() == 3) {
            result = rewriter
                         .create<linalg::BatchMatmulOp>(
                             loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
                         )
                         .getResult(0);
        }
        else {
            result = rewriter
                         .create<linalg::MatmulOp>(
                             loc, TypeRange{outTy}, ValueRange{aAdj, bAdj}, ValueRange{cInit}
                         )
                         .getResult(0);
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct BfloatReciprocalPattern : public OpRewritePattern<linalg::ReciprocalOp> {
    BfloatReciprocalPattern(MLIRContext *context)
        : OpRewritePattern<linalg::ReciprocalOp>(context, /*benefit=*/0) {
        setDebugName("BfloatReciprocalPattern");
    }
    LogicalResult
    matchAndRewrite(linalg::ReciprocalOp op, PatternRewriter &rewriter) const override {

        // checks
        if (op.getInputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one result");
        }
        if (op.getOutputs().size() != 1) {
            return rewriter.notifyMatchFailure(op, "Expected exactly one input");
        }
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }

        // Matched!  Define some useful values.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();
        auto bfTensorType = cast<RankedTensorType>(op.getType(0));
        auto shape = bfTensorType.getShape();
        auto rank = (size_t)bfTensorType.getRank();
        auto i1 = rewriter.getIntegerType(1);
        auto i8 = rewriter.getIntegerType(8);
        auto i16 = rewriter.getIntegerType(16);
        auto tType = [&](Type t) { return RankedTensorType::get(shape, t); };

        auto broadcast = [&](Value v, Type t, auto func) {
            return rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{tType(t)}, ValueRange{v},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, shape, t)},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                    }
                )
                .getResult(0);
        };
        auto i16Const = [&](int constant) {
            return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(i16, constant));
        };
        auto andi = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::AndIOp>(loc, constVal, args[0]);
            });
        };
        auto mul = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::MulIOp>(loc, constVal, args[0]);
            });
        };
        auto sub = [&](int constant, Value val) {
            auto constVal = i16Const(constant);
            return broadcast(val, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::SubIOp>(loc, constVal, args[0]);
            });
        };
        auto cmp = [&](int constant, Value val, arith::CmpIPredicate pred) {
            auto constVal = i16Const(constant);
            auto boolVal = broadcast(val, i1, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::CmpIOp>(l, pred, args[0], constVal);
            });
            return broadcast(boolVal, i16, [&](OpBuilder &b, Location l, ValueRange args) {
                return b.create<arith::ExtUIOp>(l, i16, args[0]);
            });
        };

        // bitcast to i16
        auto rawX = rewriter.create<tensor::BitcastOp>(loc, tType(i16), op.getInputs()[0]);

        // record sign bit
        auto xSign = andi(0b1000000000000000, rawX);

        // remove sign bit for following logic
        auto x = andi(0b0111111111111111, rawX);

        // create `or`-able mask for nan values (is this necessary?
        // My ML models never have nans to propagate...)
        auto inf = i16Const(0b0111111110000000);
        auto boolNanMask = broadcast(x, i1, [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ugt, args[0], inf);
        });
        auto nanMask = rewriter.create<arith::ExtSIOp>(loc, tType(i16), boolNanMask);

        // subnormal numbers are exactly the ones that map to
        // infinity.  Check which ones should map there.
        auto isSubnormal = cmp(0b0000000010000000, x, arith::CmpIPredicate::ult);

        // "big" means numbers that map to zero (we flush subnormals to zero).
        auto isNotBig = cmp(0b0111111010000000, x, arith::CmpIPredicate::ule);

        // extract exponent bits
        auto xExpo = andi(0b0111111110000000, x);

        // compute exponent of reciprocal from extracted reciprocal
        auto computedExpo = sub(0b0111111010000000, xExpo);

        // extract mantissa bits
        auto xMant = andi(0b0000000001111111, x);

        // we need to adjust our computed exponent for the special
        // case where mantissa == 0.
        auto specialMantissaValue = cmp(0b0000000000000000, xMant, arith::CmpIPredicate::eq);

        // where mantissa == 0, we need to add one to the exponent.
        auto specialExpoOffset = mul(0b0000000010000000, specialMantissaValue);
        auto realComputedExpo =
            rewriter.create<arith::AddIOp>(loc, specialExpoOffset, computedExpo);

        // Our magic LUT values!  See
        // scripts/torch/bfloat16_softmax.py to reproduce.
        std::vector<int8_t> lutData{
            // pad zeros for easier lowering
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
            // actual values
            0x00, 0x7E, 0x7C, 0x7A, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6F, 0x6D, 0x6C, 0x6A, 0x68,
            0x67, 0x65, 0x64, 0x62, 0x60, 0x5F, 0x5D, 0x5C, 0x5A, 0x59, 0x58, 0x56, 0x55, 0x53,
            0x52, 0x51, 0x4F, 0x4E, 0x4D, 0x4C, 0x4A, 0x49, 0x48, 0x47, 0x45, 0x44, 0x43, 0x42,
            0x41, 0x40, 0x3F, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37, 0x36, 0x35, 0x34, 0x33,
            0x32, 0x31, 0x30, 0x2F, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B, 0x2A, 0x29, 0x28, 0x27, 0x26,
            0x25, 0x25, 0x24, 0x23, 0x22, 0x21, 0x21, 0x20, 0x1F, 0x1E, 0x1E, 0x1D, 0x1C, 0x1B,
            0x1B, 0x1A, 0x19, 0x18, 0x18, 0x17, 0x16, 0x16, 0x15, 0x14, 0x14, 0x13, 0x12, 0x12,
            0x11, 0x10, 0x10, 0x0F, 0x0E, 0x0E, 0x0D, 0x0D, 0x0C, 0x0B, 0x0B, 0x0A, 0x0A, 0x09,
            0x09, 0x08, 0x07, 0x07, 0x06, 0x06, 0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02,
            0x01, 0x01
        };

        // create LUT
        auto lut = rewriter.create<arith::ConstantOp>(
            loc, DenseIntElementsAttr::get(
                     RankedTensorType::get(256, rewriter.getI8Type()), ArrayRef<int8_t>(lutData)
                 )
        );

        // use mantissa as index to LUT
        auto xMant8 = rewriter.create<arith::TruncIOp>(loc, tType(i8), xMant);
        auto outputTens = rewriter.create<tensor::EmptyOp>(loc, tType(i8), ValueRange{});
        auto indexOffset = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(128));
        auto lutVal =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{tType(i8)}, ValueRange{xMant8}, ValueRange{outputTens},
                    SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location loc, ValueRange args) {
                        auto baseIndex =
                            b.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), args[0]);
                        auto index = b.create<arith::AddIOp>(loc, baseIndex, indexOffset);
                        auto extracted = b.create<tensor::ExtractOp>(loc, lut, ValueRange{index});
                        b.create<linalg::YieldOp>(loc, ValueRange{extracted});
                    }
                )
                .getResult(0);

        auto lutVal16 = rewriter.create<arith::ExtSIOp>(loc, tType(i16), lutVal);

        // combine computed exponent and mantissa
        auto computed = rewriter.create<arith::OrIOp>(loc, lutVal16, realComputedExpo);

        // There are a few possible inputs (big inputs) that must also
        // be sqashed to zero.
        auto computed2 = rewriter.create<arith::MulIOp>(loc, computed, isNotBig);

        // conversely, we need a way to ensure all subnormals map to
        // infinity.
        auto maybeInf = mul(0b0111111110000000, isSubnormal);

        // We want to combine our computed maybeInf and our current
        // computed using bfloat addition.  Bitcast it real quick,
        // add, and bitcast back.
        auto computedBfloat = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, computed2);
        auto maybeInfBfloat = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, maybeInf);
        auto realComputedBfloat =
            rewriter.create<arith::AddFOp>(loc, computedBfloat, maybeInfBfloat);
        auto realComputed = rewriter.create<tensor::BitcastOp>(loc, tType(i16), realComputedBfloat);

        // add back our sign bit we saved earlier
        auto combined = rewriter.create<arith::OrIOp>(loc, xSign, realComputed);
        // final nan propagation
        auto combined2 = rewriter.create<arith::OrIOp>(loc, combined, nanMask);

        // bitcast final value back to bfloat16
        auto bfBitcast = rewriter.create<tensor::BitcastOp>(loc, bfTensorType, combined2);

        // done.
        rewriter.replaceOp(op, bfBitcast);
        return success();
    }
};

// This doesn't get its own rewriter because it only works with
// negative inputs, which can't in general be inferred at compile
// time.  Use this function in other algos when you know what you are
// doing.
Value bfloatNegExp(Value x, PatternRewriter &rewriter, Location loc) {

    // useful meta variables
    auto ctx = rewriter.getContext();
    auto bf16TensorType = cast<RankedTensorType>(x.getType());
    auto bf16 = bf16TensorType.getElementType();
    auto shape = bf16TensorType.getShape();
    auto rank = bf16TensorType.getRank();
    auto i16 = IntegerType::get(ctx, 16);
    auto i16TensorType = RankedTensorType::get(shape, i16);
    auto i8 = IntegerType::get(ctx, 8);
    auto i1 = IntegerType::get(ctx, 1);
    auto indx = rewriter.getIndexType();

    // helper funcs to create constants
    auto tConst = [&](Type type, int constant) {
        return rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(type, constant));
    };
    auto i16Tens = [&](std::vector<uint16_t> data) {
        return rewriter
            .create<arith::ConstantOp>(
                loc, DenseIntElementsAttr::get(
                         RankedTensorType::get(data.size(), i16), ArrayRef<uint16_t>(data)
                     )
            )
            .getResult();
    };

    // helper functions to create linalg.generics
    auto emptyLike = [&](Value val) {
        return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
    };
    auto broadcast = [&](Value v, Type t, auto func) {
        return rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{RankedTensorType::get(shape, t)}, ValueRange{v},
                ValueRange{rewriter.create<tensor::EmptyOp>(loc, shape, t)},
                SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                }
            )
            .getResult(0);
    };

    // create actual constants
    SmallVector<Value> sgnfBitmask;
    SmallVector<int> sgnfBitmaskData = {0b10000000, 0b1000000, 0b100000, 0b10000,
                                        0b1000,     0b100,     0b10,     0b1};
    SmallVector<Value> expoOffset;
    SmallVector<int> expoOffsetData = {7, 6, 5, 4, 3, 2, 1, 0};
    for (int i = 0; i < 8; i++) {
        sgnfBitmask.push_back(tConst(i8, sgnfBitmaskData[i]));
        expoOffset.push_back(tConst(i16, expoOffsetData[i]));
    }
    auto lutBits = i16Tens(
        {0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80,
         0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F7F, 0x3F7E, 0x3F7C, 0x3F78, 0x3F70, 0x3F62,
         0x3F47, 0x3F1B, 0x3EBC, 0x3E0B, 0x3C96, 0x39B0, 0x33F2, 0x2864, 0x114B, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
         0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}
    );

    // allocate output tensors
    auto intX = rewriter.create<tensor::BitcastOp>(loc, i16TensorType, x);

    auto dirtyExpo = broadcast(intX, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::ShRUIOp>(l, args[0], tConst(i16, 7));
    });
    // get rid of sign bit
    auto expo = broadcast(dirtyExpo, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::AndIOp>(l, args[0], tConst(i16, 0b0000000011111111));
    });
    auto mant = broadcast(intX, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::TruncIOp>(l, i8, args[0]);
    });
    auto sgnf = broadcast(mant, i8, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::OrIOp>(l, args[0], tConst(i8, 0b10000000));
    });
    auto nanBoolMask = broadcast(intX, i1, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return b.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::ugt, args[0], tConst(i16, 0b1111111110000000)
        );
    });
    auto nanMask = broadcast(nanBoolMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        // Sign extending comming in clutch.
        return b.create<arith::ExtSIOp>(l, i16, args[0]);
    });
    auto outAlloc = broadcast(nanMask, i16, [&](OpBuilder &b, Location l, ValueRange args) {
        return b.create<arith::OrIOp>(l, args[0], tConst(i16, 0b0011111110000000));
    });
    auto bfAlloc = rewriter.create<tensor::BitcastOp>(loc, bf16TensorType, outAlloc);

    // calculate e^x from mantissa, exponent.  To do so, create 8
    // value per input value and reduce mul over those values (M ->
    // Mx8 -> M).  Ideally, we would do this in a loop to avoid
    // creating any Mx8 tensors, but our lowering isn't yet smart
    // enough to deal with that, so now we make some gigantic tensors.
    SmallVector<AffineExpr> dims;
    SmallVector<AffineExpr> expandedDims;
    SmallVector<int64_t> bigShape;
    AffineExpr d;
    for (int i = 0; i < rank; i++) {
        d = getAffineDimExpr(i, ctx);
        dims.push_back(d);
        expandedDims.push_back(d);
        bigShape.push_back(shape[i]);
    }
    d = getAffineDimExpr(rank, ctx);
    SmallVector<AffineExpr> trailDim = {d};
    expandedDims.push_back(d);
    bigShape.push_back(8);
    auto expandBroadcast = [&](Value mTensor, SmallVector<Value> eightTensor, Type t, auto func) {
        Value fullOutTens = rewriter.create<tensor::EmptyOp>(loc, bigShape, t);
        auto partialOutTens = rewriter.create<tensor::EmptyOp>(loc, shape, t);
        auto tType = RankedTensorType::get(shape, t);
        SmallVector<OpFoldResult> offsets, sizes, strides(rank + 1, rewriter.getIndexAttr(1));
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));
        sizes.push_back(rewriter.getIndexAttr(1));
        for (int i = 0; i < 8; i++) {
            auto partialResult =
                rewriter
                    .create<linalg::GenericOp>(
                        loc, TypeRange{tType}, ValueRange{mTensor}, ValueRange{partialOutTens},
                        SmallVector<AffineMap>(2, AffineMap::getMultiDimIdentityMap(rank, ctx)),
                        SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                        [&](OpBuilder &b, Location l, ValueRange args) {
                            b.create<linalg::YieldOp>(
                                l, ValueRange{func(b, l, args, eightTensor[i])}
                            );
                        }
                    )
                    .getResult(0);
            offsets = SmallVector<OpFoldResult>(rank, rewriter.getIndexAttr(0));
            offsets.push_back(rewriter.getIndexAttr(i));
            fullOutTens = rewriter.create<tensor::InsertSliceOp>(
                loc, partialResult, fullOutTens, offsets, sizes, strides
            );
        }
        return fullOutTens;
    };
    auto unmaskedLutIndex = expandBroadcast(
        expo, expoOffset, i16,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return b.create<arith::AddIOp>(l, args[0], c);
        }
    );
    auto indexRawMask = expandBroadcast(
        sgnf, sgnfBitmask, i8,
        [&](OpBuilder &b, Location l, ValueRange args, Value c) {
            return b.create<arith::AndIOp>(l, args[0], c);
        }
    );
    auto bigI16TensorType = RankedTensorType::get(bigShape, i16);
    auto bigI8TensorType = RankedTensorType::get(bigShape, i8);
    auto staticBroadcast = [&](Value m8Tensor, auto func, auto type) {
        return rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{type}, ValueRange{m8Tensor},
                ValueRange{rewriter.create<tensor::EmptyOp>(loc, type, ValueRange{})},
                SmallVector<AffineMap>(2, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(l, ValueRange{func(b, l, args)});
                }
            )
            .getResult(0);
    };
    auto zero = tConst(i8, 0);
    auto one = tConst(i8, 1);
    auto index8Mask = staticBroadcast(
        indexRawMask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto tooLarge = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, one, args[0]);
            auto clampedAbove = b.create<arith::SelectOp>(l, tooLarge, one, args[0]);
            auto tooSmall =
                b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ugt, zero, clampedAbove);
            return b.create<arith::SelectOp>(l, tooSmall, zero, clampedAbove);
        },
        bigI8TensorType
    );
    auto indexMask = staticBroadcast(
        index8Mask,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<arith::ExtSIOp>(l, i16, args[0]);
        },
        bigI16TensorType
    );
    auto lutIndex =
        rewriter
            .create<linalg::GenericOp>(
                loc, TypeRange{bigI16TensorType}, ValueRange{indexMask, unmaskedLutIndex},
                ValueRange{emptyLike(indexMask)},
                SmallVector<AffineMap>(3, AffineMap::get(rank + 1, 0, expandedDims, ctx)),
                SmallVector<utils::IteratorType>(rank + 1, utils::IteratorType::parallel),
                [&](OpBuilder &b, Location l, ValueRange args) {
                    b.create<linalg::YieldOp>(
                        l, ValueRange{b.create<arith::MulIOp>(l, args[0], args[1])}
                    );
                }
            )
            .getResult(0);

    auto lutVal = staticBroadcast(
        lutIndex,
        [&](OpBuilder &b, Location l, ValueRange args) {
            return b.create<tensor::ExtractOp>(
                l, lutBits, ValueRange{b.create<arith::IndexCastOp>(l, indx, args[0])}
            );
        },
        bigI16TensorType
    );
    auto bfVal =
        rewriter.create<tensor::BitcastOp>(loc, RankedTensorType::get(bigShape, bf16), lutVal);
    // final reduce mul
    return rewriter
        .create<linalg::ReduceOp>(
            loc, ValueRange{bfVal}, ValueRange{bfAlloc}, rank,
            [&](OpBuilder &b, Location l, ValueRange args) {
                b.create<linalg::YieldOp>(
                    l, ValueRange{b.create<arith::MulFOp>(l, args[0], args[1])}
                );
            }
        )
        .getResult(0);
};

struct BfloatSoftmaxPattern : OpRewritePattern<linalg::SoftmaxOp> {
    BfloatSoftmaxPattern(MLIRContext *context)
        : OpRewritePattern<linalg::SoftmaxOp>(context, /*benefit=*/0) {
        setDebugName("BfloatSoftmaxPattern");
    }
    LogicalResult matchAndRewrite(linalg::SoftmaxOp op, PatternRewriter &rewriter) const override {

        // checks
        if (!cast<RankedTensorType>(op.getType(0)).getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(op, "Expected bf16 output");
        }

        // Matched!  Define some useful shorthands.
        auto loc = op.getLoc();
        auto ctx = rewriter.getContext();

        // get some covenient type constants.
        auto bf16TensType = cast<RankedTensorType>(op.getType(0));
        auto bf16 = bf16TensType.getElementType();
        auto shape = bf16TensType.getShape();
        auto rank = bf16TensType.getRank();
        auto i16 = IntegerType::get(ctx, 16);
        auto softmaxDim = op.getDimension();

        // Allocate a tensor like input.
        auto emptyLike = [&](Value val) {
            return rewriter.create<tensor::EmptyOp>(loc, val.getType(), ValueRange{});
        };

        // Generate affine maps and iterator lists for the various
        // operations to be performed.
        SmallVector<AffineExpr> dims;
        SmallVector<AffineExpr> nonSoftmaxDims;
        SmallVector<int64_t> reduceMaxShape;
        AffineExpr d;
        for (int i = 0; i < rank; i++) {
            d = getAffineDimExpr(i, ctx);
            dims.push_back(d);
            if (i == softmaxDim) {
            }
            else {
                nonSoftmaxDims.push_back(d);
                reduceMaxShape.push_back(shape[i]);
            }
        }
        SmallVector<AffineExpr> trailDim = {getAffineDimExpr(rank, ctx)};

        // It was difficult getting -inf in bf16.  This simplifies away during iree-opt.
        auto inf16 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(i16, 0b0111111110000000)
        );
        auto inf = rewriter.create<arith::BitcastOp>(loc, bf16, inf16);
        auto bfZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bf16));
        auto negInf = rewriter.create<arith::SubFOp>(loc, bfZero, inf);

        // subtract off bias
        auto x = op.getInput();
        auto reduceMaxType = RankedTensorType::get(reduceMaxShape, bf16);
        auto maxAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc, ValueRange{negInf},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceMaxType, ValueRange{})}
                )
                .getResult(0);
        auto max = rewriter
                       .create<linalg::ReduceOp>(
                           loc, x, maxAlloc, softmaxDim,
                           [&](OpBuilder &b, Location l, ValueRange args) {
                               b.create<linalg::YieldOp>(
                                   l, ValueRange{b.create<arith::MaximumFOp>(l, args[0], args[1])}
                               );
                           }
                       )
                       .getResult(0);
        auto biasedX =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{bf16TensType}, ValueRange{max, x}, ValueRange{emptyLike(x)},
                    (SmallVector<AffineMap>){
                        AffineMap::get(rank, 0, nonSoftmaxDims, ctx),
                        AffineMap::get(rank, 0, dims, ctx), AffineMap::get(rank, 0, dims, ctx)
                    },
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        auto biased = b.create<arith::SubFOp>(l, args[1], args[0]);
                        b.create<linalg::YieldOp>(l, ValueRange{biased});
                    }
                )
                .getResult(0);

        // perform e^x under the assumption that x <= 0 since we subtracted max
        auto ex = bfloatNegExp(biasedX, rewriter, loc);

        // obtain denominator for final softmax operation.
        auto reduceSumType = reduceMaxType;
        auto denomAlloc =
            rewriter
                .create<linalg::FillOp>(
                    loc,
                    ValueRange{rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(bf16))},
                    ValueRange{rewriter.create<tensor::EmptyOp>(loc, reduceSumType, ValueRange{})}
                )
                .getResult(0);
        auto denom = rewriter
                         .create<linalg::ReduceOp>(
                             loc, ex, denomAlloc, softmaxDim,
                             [&](OpBuilder &b, Location l, ValueRange args) {
                                 b.create<linalg::YieldOp>(
                                     l, ValueRange{b.create<arith::AddFOp>(l, args[0], args[1])}
                                 );
                             }
                         )
                         .getResult(0);
        auto recip =
            rewriter.create<linalg::ReciprocalOp>(loc, ValueRange{denom}, ValueRange{denom})
                .getResult(0);

        // final val.  Woot!
        auto softmax =
            rewriter
                .create<linalg::GenericOp>(
                    loc, TypeRange{bf16TensType}, ValueRange{recip, ex}, ValueRange{emptyLike(ex)},
                    (SmallVector<AffineMap>){
                        AffineMap::get(rank, 0, nonSoftmaxDims, ctx),
                        AffineMap::get(rank, 0, dims, ctx), AffineMap::get(rank, 0, dims, ctx)
                    },
                    SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
                    [&](OpBuilder &b, Location l, ValueRange args) {
                        auto result = b.create<arith::MulFOp>(l, args[0], args[1]);
                        b.create<linalg::YieldOp>(l, ValueRange{result});
                    }
                )
                .getResult(0);

        rewriter.replaceOp(op, softmax);
        return success();
    }
};

void populateLinalgToTorqHLPrePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<BfloatSoftmaxPattern>(context);
    patterns.insert<BfloatReciprocalPattern>(context);
    patterns.insert<TensorBitcastPattern>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::AndIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::CmpIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::MulIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::OrIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::AddIOp>>(context);
    patterns.insert<ArithOnTensorToLinalgPattern<arith::ExtSIOp>>(context);
    patterns.insert<Conv2dConvert<linalg::Conv2DNhwcHwcfOp, syna::torq_hl::Conv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwcf2fchw(), 28, 12, Check::isKerSmall,
        markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::nhwc2nchw(), Permutation::hwc2chw(), 20, 12, Check::isKerSmall,
        markFuseGroups
    );
    patterns.insert<Conv2dConvert<linalg::DepthwiseConv2DNhwcHwcOp, torq_hl::DepthwiseConv2DOp>>(
        context, 3, Permutation::none(), Permutation::none(), 20, 12, Check::isKerEqInput,
        markFuseGroups
    );

    patterns.insert<PoolingNhwcMaxOpConversion>(context, markFuseGroups);

    patterns.insert<FCMatmulOpConversion>(context, markFuseGroups);
    patterns.insert<Conv2DMatmulOpConversion>(context, markFuseGroups);
    if (clConv1dAsMatmul) {
        patterns.insert<Conv1DNcwFcwToLinalgMatmulPattern>(context);
    }
    else {
        patterns.insert<Conv1DNcwFcwToLinalgConv2DPattern>(context);
    }

    if (!markFuseGroups) {
        patterns.insert<QuantizedBatchMatmulPattern>(context);
    }
}

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

    patterns.insert<Conv2DOpBigStride>(context, markFuseGroups);
}

void populateArithToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<ArithCastOpPattern<arith::ExtUIOp>>(context);
    patterns.insert<ArithCastOpPattern<arith::TruncIOp>>(context);

    patterns.insert<ElementwiseBinaryArithOpPattern>(context);
    patterns.insert<ElementwiseUnaryArithOpPattern>(context);
    patterns.insert<ElementWiseShiftOpPattern>(context);

    patterns.insert<RoundingRightShiftPattern>(context);
}

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
    patterns.insert<ArithOnTensorToLinalgPattern<arith::ExtSIOp>>(context);

    patterns.insert<TransposeOpConversion>(context);
    patterns.insert<FillOpConversion>(context);
    patterns.insert<TensorPadOpConversion>(context);

    // Make sure to add the new class to isTorqMatmulOp function if here
    // is changed
    patterns.insert<MatmulOpConversion<linalg::BatchMatmulOp>>(context);
    patterns.insert<MatmulOpConversion<linalg::MatmulOp>>(context);
    patterns.insert<MatmulOpConversion<linalg::DotOp>>(context);
    patterns.insert<MatmulOpConversion<linalg::MatvecOp>>(context);

    patterns.insert<ReduceOpConversion>(context);
    patterns.insert<MulOpPattern>(context);
    patterns.insert<ClampOpPattern>(context);
    patterns.insert<CastOpPattern>(context);
    patterns.insert<BroadcastOpConversion>(context);

    patterns.insert<AbsOpPattern>(context);
    patterns.insert<NegateOpPattern>(context);
    patterns.insert<ClzOpPattern>(context);
    patterns.insert<CeilOpPattern>(context);
    patterns.insert<FloorOpPattern>(context);

    patterns.insert<RescaleOpConversion>(context);

    // for add/sub op, currently we have two ways/steps to process:
    // it will be converted firstly in
    // populateLinalgToTorqHLPrePatternsLowPrio::EltwiseBinaryConvert, if converted successfully,
    // its tiling will be done by TilePass, if converted failed, it will go through the general
    // linalg path, which is linalg tiling then will be converted by
    // populateLinalgToTorqHLPatterns::AddOpPattern to torq_hl::AddOp
    // now rescaled add/sub uses populateLinalgToTorqHLPrePatternsLowPrio
    // non-rescaled add/sub uses populateLinalgToTorqHLPatterns::AddOpPattern
    // this way we can make sure some add/sub op can make use of linalg tiling
    // in the end, we will switch all addOp with/without rescale to linalg tiling
    patterns.insert<AddOpPattern>(context);

    patterns.insert<ExtractOpPattern>(context);

    patterns.insert<ReinterpretCastOpPattern>(context);
}

void populateTensorToLinalgPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<CollapseShapeOpToLinalgRewrite>(context);
    patterns.insert<ExpandShapeOpToLinalgRewrite>(context);
}

} // namespace mlir::syna::torq
