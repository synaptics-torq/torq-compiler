// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/QuantPatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <cstring>
#include <optional>

#define DEBUG_TYPE "linalg-torq-quantize-pattern"

namespace mlir::syna::torq {

namespace {

static bool isHostExecutor(linalg::GenericOp op) {
    return op && getTargetExecutor(op) == torq_hl::Executor::Host;
}

static APFloat doubleToAPFloat(double value, const llvm::fltSemantics &semantics) {
    APFloat result(static_cast<float>(value));
    if (&semantics != &llvm::APFloat::IEEEsingle()) {
        bool ignored;
        result.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &ignored);
    }
    return result;
}

static Value
createFloatScalarConst(PatternRewriter &rewriter, Operation &op, Type elemType, double value) {
    auto tensorType = RankedTensorType::get({}, elemType);
    APFloat apValue = elemType.isBF16() ? doubleToAPFloat(value, llvm::APFloat::BFloat())
                                        : doubleToAPFloat(value, llvm::APFloat::IEEEsingle());
    auto attr = DenseFPElementsAttr::get(tensorType, apValue);
    return arith::ConstantOp::create(rewriter, op.getLoc(), attr);
}

static int32_t floatBits(float value) {
    int32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static Value createActOp(
    PatternRewriter &rewriter, Operation &op, StringRef name, Value input,
    RankedTensorType resultType
) {
    Value init = createInitTensor(op, rewriter, resultType);
    APFloat zeroFp(llvm::APFloat::IEEEsingle(), "0.0");
    return torq_hl::ActOp::create(
               rewriter, op.getLoc(), resultType, init, name, 0, 0, 0, 0, zeroFp, zeroFp, input,
               mlir::Value()
    )
        .getOutput();
}

// Fold a quantize generic with a constant float input to an integer constant.
static Value foldQuantGenericToConstant(
    linalg::GenericOp op, double scale, double zp, double min, double max, PatternRewriter &rewriter
) {
    Value input = op.getInputs()[0];
    auto cstOp = input.getDefiningOp<arith::ConstantOp>();
    if (!cstOp)
        return nullptr;
    auto inTy = dyn_cast<RankedTensorType>(input.getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inTy || !outTy || !inTy.getElementType().isF32())
        return nullptr;
    auto dense = dyn_cast<ElementsAttr>(cstOp.getValue());
    if (!dense)
        return nullptr;

    auto outElemTy = dyn_cast<IntegerType>(outTy.getElementType());
    if (!outElemTy)
        return nullptr;
    unsigned width = outElemTy.getWidth();
    int64_t typeMin = static_cast<int64_t>(APInt::getSignedMinValue(width).getSExtValue());
    int64_t typeMax = static_cast<int64_t>(APInt::getSignedMaxValue(width).getSExtValue());

    int64_t iMin = std::max<int64_t>(std::llround(min), typeMin);
    int64_t iMax = std::min<int64_t>(std::llround(max), typeMax);
    int64_t iZp = std::llround(zp);

    SmallVector<APInt> outVals;
    outVals.reserve(dense.size());
    for (const APFloat &f : dense.getValues<APFloat>()) {
        double dv = f.convertToDouble();
        int64_t iv = std::llround(dv / scale) + iZp;
        if (iv < iMin)
            iv = iMin;
        if (iv > iMax)
            iv = iMax;
        outVals.emplace_back(width, iv, /*isSigned=*/true);
    }
    auto attr = DenseIntElementsAttr::get(outTy, outVals);
    return arith::ConstantOp::create(rewriter, op.getLoc(), outTy, attr).getResult();
}

struct QuantizeOpConversion : public OpRewritePattern<linalg::GenericOp> {
    const bool _markFuseGroups;

    QuantizeOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (isCompileTimeConst(op))
            return rewriter.notifyMatchFailure(op, "compile-time constant");

        double scale, zp, min, max;
        if (!matchQuantGeneric(op, scale, zp, min, max))
            return rewriter.notifyMatchFailure(op, "not a PT2E quant generic");

        // Discovery mode: mark standalone quantize generics as tile/fuse groups.
        if (_markFuseGroups) {
            if (isMarkedFuseGroup(op.getOperation()))
                return rewriter.notifyMatchFailure(op, "already marked");
            auto fuseGroupAttr = op->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(op, "missing fuse group id");
            markOpFuseGroup(op, rewriter, fuseGroupAttr);
            return success();
        }

        Value input = op.getInputs()[0];
        auto inTy = cast<RankedTensorType>(input.getType());
        auto outTy = cast<RankedTensorType>(op.getResult(0).getType());

        // Fold constant rescales (e.g., Q bias scale adjustments) so they do not
        // survive into tile-and-fuse as generics operating on constants.
        if (auto folded = foldQuantGenericToConstant(op, scale, zp, min, max, rewriter)) {
            rewriter.replaceOp(op, folded);
            return success();
        }

        if (!outTy.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(op, "non-i8 output without constant input");

        Type elemType = inTy.getElementType();
        auto shape = inTy.getShape();
        Location loc = op.getLoc();

        // MulOp only supports bf16 floating-point, so do the scaling in bf16.
        auto bf16Type = RankedTensorType::get(shape, rewriter.getBF16Type());
        auto i32Type = RankedTensorType::get(shape, rewriter.getI32Type());

        Value scaledInput = input;
        if (elemType.isF32()) {
            scaledInput = createActOp(rewriter, *op, "f2f", input, bf16Type);
        }

        // MulOp: bf16_input * (1/scale) + zp
        Value mulInit = createInitTensor(op, rewriter, bf16Type);
        Value recipScaleConst =
            createFloatScalarConst(rewriter, *op, rewriter.getBF16Type(), 1.0 / scale);
        Value zpBias = createFloatScalarConst(rewriter, *op, rewriter.getF32Type(), zp);
        Value mulOut =
            torq_hl::MulOp::create(
                rewriter, loc, bf16Type, mulInit, rewriter.getI32IntegerAttr(0),
                rewriter.getI32IntegerAttr(0xff800000), rewriter.getI32IntegerAttr(0x7f800000),
                zpBias, rewriter.getI8IntegerAttr(0), scaledInput, recipScaleConst
            )
                .getOutput();

        // ActOp f2i: bf16 -> int32
        Value f2iOut = createActOp(rewriter, *op, "f2i", mulOut, i32Type);

        // FMAOp: int32 -> i8 clamp to [min, max]
        int32_t min_i = static_cast<int32_t>(std::llround(min));
        int32_t max_i = static_cast<int32_t>(std::llround(max));
        Value fmaInit = createInitTensor(op, rewriter, outTy);
        Value weights = createI8Const(rewriter, op, {1}, llvm::ArrayRef<int64_t>{1});
        Value biasScale = createI32Const(rewriter, op, interleave({0}, {1}));
        Value fmaOut = torq_hl::FMAOp::create(
                           rewriter, loc, outTy, fmaInit, rewriter.getI32IntegerAttr(0),
                           rewriter.getI32IntegerAttr(min_i), rewriter.getI32IntegerAttr(max_i),
                           rewriter.getI32IntegerAttr(0), weights, biasScale, f2iOut
        )
                           .getOutput();

        rewriter.replaceOp(op, fmaOut);
        return success();
    }
};

struct DequantizeOpConversion : public OpRewritePattern<linalg::GenericOp> {
    const bool _markFuseGroups;

    DequantizeOpConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        if (isCompileTimeConst(op))
            return rewriter.notifyMatchFailure(op, "compile-time constant");

        double scale;
        int32_t zp = 0;
        if (!matchDequantGeneric(op, scale, zp))
            return rewriter.notifyMatchFailure(op, "not a PT2E dequant generic");

        // Discovery mode: mark standalone dequantize generics as tile/fuse groups.
        if (_markFuseGroups) {
            if (isMarkedFuseGroup(op.getOperation()))
                return rewriter.notifyMatchFailure(op, "already marked");
            auto fuseGroupAttr = op->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(op, "missing fuse group id");
            markOpFuseGroup(op, rewriter, fuseGroupAttr);
            return success();
        }

        Value input = op.getInputs()[0];
        if (input.getDefiningOp<arith::ConstantOp>())
            return rewriter.notifyMatchFailure(op, "constant input");

        auto inTy = cast<RankedTensorType>(input.getType());
        auto outTy = cast<RankedTensorType>(op.getResult(0).getType());

        if (!inTy.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(op, "non-i8 input");

        auto shape = inTy.getShape();
        Location loc = op.getLoc();

        // MulOp only supports bf16 floating-point, so scale in bf16.
        bool outputIsF32 = outTy.getElementType().isF32();
        auto bf16Type = RankedTensorType::get(shape, rewriter.getBF16Type());

        // Convert the i8 input directly to bf16 for the multiply.
        Value bf16Input = createActOp(rewriter, *op, "i2f", input, bf16Type);

        // MulOp: bf16_input * scale + bias, bias = -zp*scale  =>  affine dequant (q-zp)*scale.
        Value mulInit = createInitTensor(op, rewriter, bf16Type);
        Value zpBias = createFloatScalarConst(
            rewriter, *op, rewriter.getF32Type(), -static_cast<double>(zp) * scale
        );
        Value scaleConst = createFloatScalarConst(rewriter, *op, rewriter.getBF16Type(), scale);
        Value mulOut =
            torq_hl::MulOp::create(
                rewriter, loc, bf16Type, mulInit, rewriter.getI32IntegerAttr(0),
                rewriter.getI32IntegerAttr(0xff800000), rewriter.getI32IntegerAttr(0x7f800000),
                zpBias, rewriter.getI8IntegerAttr(0), bf16Input, scaleConst
            )
                .getOutput();

        // Cast back to the original output type when it is f32.
        Value output = mulOut;
        if (outputIsF32) {
            output = createActOp(rewriter, *op, "f2f", mulOut, outTy);
        }

        rewriter.replaceOp(op, output);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQuantizePatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QuantizeOpConversion, DequantizeOpConversion>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
