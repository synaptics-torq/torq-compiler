// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "QuantPatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <limits>

#define DEBUG_TYPE "linalg-torq-quant-pattern-utils"

namespace mlir::syna::torq {

std::optional<double> getQFloatScalar(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    if (auto fattr = dyn_cast<FloatAttr>(cst->getAttr("value")))
        return fattr.getValueAsDouble();
    if (auto dense = dyn_cast<DenseFPElementsAttr>(cst->getAttr("value"))) {
        if (dense.isSplat())
            return dense.getSplatValue<double>();
        if (dense.size() == 1) {
            APFloat val = *dense.begin();
            return val.convertToDouble();
        }
    }
    return std::nullopt;
}

std::optional<double> getQGenericFloatConstant(Value v, linalg::GenericOp op) {
    if (!op)
        return std::nullopt;
    auto value = traceToConstFloat(op, v, SplatPolicy::Any);
    if (!value)
        return std::nullopt;
    return value->convertToDouble();
}

static std::optional<int64_t> getQGenericIntConstant(Value v, linalg::GenericOp op) {
    if (!op)
        return std::nullopt;
    return traceToConstInt(op, v, SplatPolicy::Any);
}

static bool
isValidDequantShape(linalg::GenericOp op, RankedTensorType &inTy, RankedTensorType &outTy) {
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;
    inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inTy || !outTy)
        return false;
    if (!inTy.getElementType().isIntOrIndex())
        return false;
    if (!outTy.getElementType().isF32() && !outTy.getElementType().isBF16())
        return false;
    return true;
}

// Signed integer dequant:
// linalg.generic body (zp = 0):
//   [arith.extsi(%in)] -> arith.sitofp -> arith.mulf(%scale)
// linalg.generic body (zp != 0):
//   arith.extsi(%in) -> arith.subi(%zp) -> arith.sitofp -> arith.mulf(%scale)
std::optional<DequantInfo> matchDequantSigned(linalg::GenericOp op) {
    RankedTensorType inTy, outTy;
    if (!isValidDequantShape(op, inTy, outTy))
        return std::nullopt;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return std::nullopt;
    auto mulf = yieldOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulf)
        return std::nullopt;

    Value sitofpVal = nullptr;
    Value scaleVal = nullptr;
    for (Value operand : {mulf.getLhs(), mulf.getRhs()}) {
        if (operand.getDefiningOp<arith::SIToFPOp>())
            sitofpVal = operand;
        else
            scaleVal = operand;
    }
    if (!sitofpVal || !scaleVal)
        return std::nullopt;

    auto maybeScale = getQGenericFloatConstant(scaleVal, op);
    if (!maybeScale)
        return std::nullopt;

    auto sitofp = sitofpVal.getDefiningOp<arith::SIToFPOp>();
    Value sitofpIn = sitofp.getIn();

    int32_t extractedZp = 0;
    Value inner = sitofpIn;
    if (auto subi = sitofpIn.getDefiningOp<arith::SubIOp>()) {
        auto lhs = subi.getLhs();
        auto rhs = subi.getRhs();
        if (auto extsi = lhs.getDefiningOp<arith::ExtSIOp>()) {
            inner = extsi.getIn();
            auto maybeZp = getQGenericIntConstant(rhs, op);
            if (!maybeZp)
                return std::nullopt;
            extractedZp = static_cast<int32_t>(*maybeZp);
        }
        else if (auto extsi = rhs.getDefiningOp<arith::ExtSIOp>()) {
            inner = extsi.getIn();
            auto maybeZp = getQGenericIntConstant(lhs, op);
            if (!maybeZp)
                return std::nullopt;
            extractedZp = static_cast<int32_t>(*maybeZp);
        }
        else {
            return std::nullopt;
        }
    }
    else if (auto extsi = sitofpIn.getDefiningOp<arith::ExtSIOp>()) {
        inner = extsi.getIn();
    }
    else {
        inner = sitofpIn;
    }

    auto inputArg = dyn_cast<BlockArgument>(inner);
    if (!inputArg || inputArg.getOwner() != op.getBody())
        return std::nullopt;

    return DequantInfo{/*scale=*/*maybeScale, /*zp=*/extractedZp};
}

// Scale-one signed integer dequant:
// linalg.generic body (zp = 0):
//   [arith.extsi(%in)] -> arith.sitofp
// linalg.generic body (zp != 0):
//   arith.extsi(%in) -> arith.subi(%zp) -> arith.sitofp
std::optional<DequantInfo> matchDequantScaleOne(linalg::GenericOp op) {
    RankedTensorType inTy, outTy;
    if (!isValidDequantShape(op, inTy, outTy))
        return std::nullopt;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return std::nullopt;
    auto sitofp = yieldOp.getOperand(0).getDefiningOp<arith::SIToFPOp>();
    if (!sitofp)
        return std::nullopt;

    Value sitofpIn = sitofp.getIn();
    int32_t extractedZp = 0;
    Value inner = sitofpIn;
    if (auto subi = sitofpIn.getDefiningOp<arith::SubIOp>()) {
        auto lhs = subi.getLhs();
        auto rhs = subi.getRhs();
        if (auto extsi = lhs.getDefiningOp<arith::ExtSIOp>()) {
            inner = extsi.getIn();
            auto maybeZp = getQGenericIntConstant(rhs, op);
            if (!maybeZp)
                return std::nullopt;
            extractedZp = static_cast<int32_t>(*maybeZp);
        }
        else if (auto extsi = rhs.getDefiningOp<arith::ExtSIOp>()) {
            inner = extsi.getIn();
            auto maybeZp = getQGenericIntConstant(lhs, op);
            if (!maybeZp)
                return std::nullopt;
            extractedZp = static_cast<int32_t>(*maybeZp);
        }
        else {
            return std::nullopt;
        }
    }
    else if (auto extsi = sitofpIn.getDefiningOp<arith::ExtSIOp>()) {
        inner = extsi.getIn();
    }

    auto inputArg = dyn_cast<BlockArgument>(inner);
    if (!inputArg || inputArg.getOwner() != op.getBody())
        return std::nullopt;

    return DequantInfo{/*scale=*/1.0, /*zp=*/extractedZp};
}

// Unsigned integer dequant (ONNX INT4 QDQ):
//
// ONNX INT4 QDQ quantized models produce unsigned dequant with a float
// zero-point (already scaled):
//   [arith.extui(%in)] -> arith.uitofp -> arith.mulf(%scale)                    (zp = 0)
//   [arith.extui(%in)] -> arith.uitofp -> arith.subf(%fp_zp) -> arith.mulf(%scale)  (zp != 0)
// The extui is optional: i32 inputs are already wide enough.
//
// The float zero-point is converted back to the integer convention used by
// TORQ HL: int_zp = round(fpZp / scale).
std::optional<DequantInfo> matchDequantUnsigned(linalg::GenericOp op) {
    RankedTensorType inTy, outTy;
    if (!isValidDequantShape(op, inTy, outTy))
        return std::nullopt;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return std::nullopt;
    auto mulf = yieldOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulf)
        return std::nullopt;

    // One operand is the uitofp chain, the other is the scale constant.
    Value dataChain, scaleVal;
    if (mulf.getLhs().getDefiningOp<arith::UIToFPOp>()) {
        dataChain = mulf.getLhs();
        scaleVal = mulf.getRhs();
    }
    else if (mulf.getRhs().getDefiningOp<arith::UIToFPOp>()) {
        dataChain = mulf.getRhs();
        scaleVal = mulf.getLhs();
    }
    else {
        return std::nullopt;
    }

    auto maybeScale = getQGenericFloatConstant(scaleVal, op);
    if (!maybeScale)
        return std::nullopt;

    auto uitofp = dataChain.getDefiningOp<arith::UIToFPOp>();
    Value uifpIn = uitofp.getIn();

    double fpZp = 0.0;
    bool hasFpZp = false;

    if (auto subf = uifpIn.getDefiningOp<arith::SubFOp>()) {
        Value subfLhs = subf.getLhs(), subfRhs = subf.getRhs();
        if (auto maybeZp = getQGenericFloatConstant(subfLhs, op)) {
            fpZp = *maybeZp;
            uifpIn = subfRhs;
            hasFpZp = true;
        }
        else if (auto maybeZp = getQGenericFloatConstant(subfRhs, op)) {
            fpZp = *maybeZp;
            uifpIn = subfLhs;
            hasFpZp = true;
        }
        else {
            return std::nullopt;
        }
    }

    Value inner = uifpIn;
    if (auto extui = uifpIn.getDefiningOp<arith::ExtUIOp>())
        inner = extui.getIn();

    auto inputArg = dyn_cast<BlockArgument>(inner);
    if (!inputArg || inputArg.getOwner() != op.getBody())
        return std::nullopt;

    int32_t intZp = hasFpZp ? static_cast<int32_t>(std::llround(fpZp / *maybeScale)) : 0;
    return DequantInfo{/*scale=*/*maybeScale, /*zp=*/intZp};
}

// All registered dequant variants, tried in order.  The first match wins.
using DequantVariantFn = std::optional<DequantInfo> (*)(linalg::GenericOp);
static const DequantVariantFn kDequantVariants[] = {
    matchDequantSigned,   // standard ONNX signed int8 QDQ
    matchDequantUnsigned, // ONNX unsigned INT4 QDQ (block_size path)
    matchDequantScaleOne, // scale folded away by canonicalization
};

bool matchDequantGeneric(linalg::GenericOp op, double &scale, int32_t &zp) {
    for (auto f : kDequantVariants) {
        if (auto info = f(op)) {
            scale = info->scale;
            zp = info->zp;
            return true;
        }
    }
    return false;
}

static bool
isValidQuantShape(linalg::GenericOp op, RankedTensorType &inTy, RankedTensorType &outTy) {
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;
    inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inTy || !outTy)
        return false;
    if (!outTy.getElementType().isIntOrIndex())
        return false;
    if (!inTy.getElementType().isF32() && !inTy.getElementType().isBF16())
        return false;
    return true;
}

// Walk backward from `v` through the quant clamp pair, extracting the constant
// bounds and the value that feeds the rest of the quant body.  Both orderings
// are accepted:
//   v = maxf(min_value, minf(max_value, data))
//   v = minf(max_value, maxf(min_value, data))
bool extractQuantClampBounds(Value v, linalg::GenericOp op, double &min, double &max, Value &data) {
    arith::MaximumFOp maxf = nullptr;
    arith::MinimumFOp minf = nullptr;
    Value cur = v;
    while (true) {
        if (auto mx = cur.getDefiningOp<arith::MaximumFOp>()) {
            if (maxf)
                return false; // duplicate maxf
            maxf = mx;
            // The data side is the operand that is NOT a float constant.
            if (getQGenericFloatConstant(mx.getLhs(), op))
                cur = mx.getRhs();
            else
                cur = mx.getLhs();
        }
        else if (auto mn = cur.getDefiningOp<arith::MinimumFOp>()) {
            if (minf)
                return false; // duplicate minf
            minf = mn;
            if (getQGenericFloatConstant(mn.getLhs(), op))
                cur = mn.getRhs();
            else
                cur = mn.getLhs();
        }
        else {
            break;
        }
    }
    if (!maxf || !minf)
        return false;

    // min is the constant operand of maxf.
    auto maybeMin0 = getQGenericFloatConstant(maxf.getLhs(), op);
    auto maybeMin1 = getQGenericFloatConstant(maxf.getRhs(), op);
    if (maybeMin0 && !maybeMin1)
        min = *maybeMin0;
    else if (maybeMin1 && !maybeMin0)
        min = *maybeMin1;
    else
        return false;

    // max is the constant operand of minf.
    auto maybeMax0 = getQGenericFloatConstant(minf.getLhs(), op);
    auto maybeMax1 = getQGenericFloatConstant(minf.getRhs(), op);
    if (maybeMax0 && !maybeMax1)
        max = *maybeMax0;
    else if (maybeMax1 && !maybeMax0)
        max = *maybeMax1;
    else
        return false;

    data = cur;
    return true;
}

// Signed integer quant:
// linalg.generic body:
//   divf/mulf -> math.roundeven -> arith.addf(zp) -> arith.maximumf(min)
//   -> arith.minimumf(max) -> arith.fptosi
std::optional<QuantInfo> matchQuantSigned(linalg::GenericOp op) {
    RankedTensorType inTy, outTy;
    if (!isValidQuantShape(op, inTy, outTy))
        return std::nullopt;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return std::nullopt;
    auto fptosi = yieldOp.getOperand(0).getDefiningOp<arith::FPToSIOp>();
    if (!fptosi)
        return std::nullopt;

    double min, max;
    Value chain;
    if (!extractQuantClampBounds(fptosi.getIn(), op, min, max, chain))
        return std::nullopt;

    auto addf = chain.getDefiningOp<arith::AddFOp>();
    if (!addf)
        return std::nullopt;
    auto maybeZp0 = getQGenericFloatConstant(addf.getLhs(), op);
    auto maybeZp1 = getQGenericFloatConstant(addf.getRhs(), op);
    Value roundVal;
    double zp;
    if (maybeZp0 && !maybeZp1) {
        zp = *maybeZp0;
        roundVal = addf.getRhs();
    }
    else if (maybeZp1 && !maybeZp0) {
        zp = *maybeZp1;
        roundVal = addf.getLhs();
    }
    else {
        return std::nullopt;
    }

    auto roundOp = roundVal.getDefiningOp<math::RoundEvenOp>();
    if (!roundOp)
        return std::nullopt;
    Value scaleVal = roundOp->getOperand(0);

    double scale;
    if (auto divf = scaleVal.getDefiningOp<arith::DivFOp>()) {
        auto s0 = getQGenericFloatConstant(divf.getLhs(), op);
        auto s1 = getQGenericFloatConstant(divf.getRhs(), op);
        if (s0 && !s1)
            scale = *s0;
        else if (s1 && !s0)
            scale = *s1;
        else
            return std::nullopt;
    }
    else if (auto mulf = scaleVal.getDefiningOp<arith::MulFOp>()) {
        auto s0 = getQGenericFloatConstant(mulf.getLhs(), op);
        auto s1 = getQGenericFloatConstant(mulf.getRhs(), op);
        if (s0 && !s1)
            scale = 1.0 / *s0;
        else if (s1 && !s0)
            scale = 1.0 / *s1;
        else
            return std::nullopt;
    }
    else {
        return std::nullopt;
    }

    return QuantInfo{scale, zp, min, max};
}

// Scale-one signed integer quant:
// linalg.generic body:
//   math.roundeven -> arith.addf(zp) -> arith.maximumf(min)
//   -> arith.minimumf(max) -> arith.fptosi
std::optional<QuantInfo> matchQuantScaleOne(linalg::GenericOp op) {
    RankedTensorType inTy, outTy;
    if (!isValidQuantShape(op, inTy, outTy))
        return std::nullopt;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return std::nullopt;
    auto fptosi = yieldOp.getOperand(0).getDefiningOp<arith::FPToSIOp>();
    if (!fptosi)
        return std::nullopt;

    double min, max;
    Value chain;
    if (!extractQuantClampBounds(fptosi.getIn(), op, min, max, chain))
        return std::nullopt;

    auto addf = chain.getDefiningOp<arith::AddFOp>();
    if (!addf)
        return std::nullopt;
    auto maybeZp0 = getQGenericFloatConstant(addf.getLhs(), op);
    auto maybeZp1 = getQGenericFloatConstant(addf.getRhs(), op);
    Value roundVal;
    double zp;
    if (maybeZp0 && !maybeZp1) {
        zp = *maybeZp0;
        roundVal = addf.getRhs();
    }
    else if (maybeZp1 && !maybeZp0) {
        zp = *maybeZp1;
        roundVal = addf.getLhs();
    }
    else {
        return std::nullopt;
    }

    if (!roundVal.getDefiningOp<math::RoundEvenOp>())
        return std::nullopt;

    return QuantInfo{/*scale=*/1.0, zp, min, max};
}

// All registered quant variants, tried in order.  The first match wins.
using QuantVariantFn = std::optional<QuantInfo> (*)(linalg::GenericOp);
static const QuantVariantFn kQuantVariants[] = {
    matchQuantSigned,   // standard ONNX signed int8 QDQ
    matchQuantScaleOne, // scale folded away by canonicalization
                        // Future: matchQuantUnsigned, etc.
};

bool matchQuantGeneric(linalg::GenericOp op, double &scale, double &zp, double &min, double &max) {
    for (auto f : kQuantVariants) {
        if (auto info = f(op)) {
            scale = info->scale;
            zp = info->zp;
            min = info->min;
            max = info->max;
            return true;
        }
    }
    return false;
}

bool computeMultiplierAndShift(double scale, int32_t &multiplier, int32_t &shift) {
    if (scale == 0.0) {
        multiplier = 0;
        // Shift is irrelevant when the multiplier is zero, but keep it a
        // hardware-legal multiple of 4.
        shift = 28;
        return true;
    }
    if (scale < 0.0)
        return false;

    // Largest multiple-of-4 shift supported by the current hardware encoding.
    constexpr int32_t kMaxShift = 60;
    for (int32_t s = kMaxShift; s >= 0; s -= 4) {
        int64_t mult = std::llround(scale * static_cast<double>(1LL << s));
        if (mult >= std::numeric_limits<int32_t>::min() &&
            mult <= std::numeric_limits<int32_t>::max()) {
            multiplier = static_cast<int32_t>(mult);
            shift = s;
            LLVM_DEBUG(llvm::dbgs() << "[computeMultiplierAndShift] scale=" << scale
                                    << " multiplier=" << multiplier << " shift=" << shift << "\n";);
            return true;
        }
    }
    LLVM_DEBUG(llvm::dbgs() << "[computeMultiplierAndShift] failed for scale=" << scale << "\n";);
    return false;
}

std::optional<int32_t> getScalarI32Const(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    auto ia = dyn_cast<IntegerAttr>(cst.getValue());
    if (!ia)
        return std::nullopt;
    return static_cast<int32_t>(ia.getInt());
}

Value computeInputZpCorrection(Value adjustedWeights, int32_t inputZp, PatternRewriter &rewriter) {
    auto wTy = cast<RankedTensorType>(adjustedWeights.getType());
    auto loc = adjustedWeights.getLoc();
    auto i32Ty = rewriter.getI32Type();
    SmallVector<int64_t> biasShape{wTy.getShape()[0]};
    auto reduceInitTy = RankedTensorType::get(biasShape, i32Ty);
    auto reduceInit =
        arith::ConstantOp::create(rewriter, loc, reduceInitTy, rewriter.getZeroAttr(reduceInitTy))
            .getResult();

    SmallVector<int64_t> reduceDims;
    for (int64_t i = 1; i < wTy.getRank(); ++i)
        reduceDims.push_back(i);

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, adjustedWeights, reduceInit, reduceDims,
        [](OpBuilder &b, Location loc, ValueRange args) {
            auto toI32 = [&](Value v) -> Value {
                if (v.getType() == b.getI32Type())
                    return v;
                return arith::ExtSIOp::create(b, loc, b.getI32Type(), v);
            };
            auto extL = toI32(args[0]);
            auto extR = toI32(args[1]);
            auto sum = arith::AddIOp::create(b, loc, extL, extR);
            linalg::YieldOp::create(b, loc, ArrayRef<Value>{sum});
        }
    );
    // The correction chain is folded later by CompileTimeConstComputePass, so it
    // must be routable to the Host: an unmarked linalg op is illegal in the
    // pre-conversion target and makes applyPartialConversion roll back the whole
    // pattern rewrite.
    setTargetExecutorAttr(reduceOp, torq_hl::Executor::Host);
    auto sumW = reduceOp.getResult(0);

    auto zpConst = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(-inputZp));
    auto mulInit = tensor::EmptyOp::create(rewriter, loc, biasShape, i32Ty).getResult();
    auto mulOp = linalg::GenericOp::create(
        rewriter, loc, mulInit.getType(), ValueRange{sumW, zpConst}, ValueRange{mulInit},
        SmallVector<AffineMap>{
            rewriter.getMultiDimIdentityMap(1), AffineMap::get(1, 0, {}, rewriter.getContext()),
            rewriter.getMultiDimIdentityMap(1)
        },
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [](OpBuilder &b, Location loc, ValueRange args) {
            auto mul = arith::MulIOp::create(b, loc, args[0], args[1]);
            linalg::YieldOp::create(b, loc, ArrayRef<Value>{mul});
        }
    );
    setTargetExecutorAttr(mulOp, torq_hl::Executor::Host);
    return mulOp.getResult(0);
}

Value addPerChannelBias(Value lhs, Value rhs, PatternRewriter &rewriter) {
    auto ty = cast<RankedTensorType>(lhs.getType());
    auto loc = lhs.getLoc();
    auto init =
        tensor::EmptyOp::create(rewriter, loc, ty.getShape(), ty.getElementType()).getResult();
    auto addOp = linalg::GenericOp::create(
        rewriter, loc, ty, ValueRange{lhs, rhs}, ValueRange{init},
        SmallVector<AffineMap>{
            rewriter.getMultiDimIdentityMap(1), rewriter.getMultiDimIdentityMap(1),
            rewriter.getMultiDimIdentityMap(1)
        },
        SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
        [](OpBuilder &b, Location loc, ValueRange args) {
            auto sum = arith::AddIOp::create(b, loc, args[0], args[1]);
            linalg::YieldOp::create(b, loc, ArrayRef<Value>{sum});
        }
    );
    setTargetExecutorAttr(addOp, torq_hl::Executor::Host);
    return addOp.getResult(0);
}

Value buildDynamicInterleavedBiasScale(
    Value bias, int32_t multiplier, Location loc, PatternRewriter &rewriter
) {
    auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasTy || biasTy.getRank() != 1 || biasTy.getElementType() != rewriter.getI32Type())
        return nullptr;
    int64_t C = biasTy.getShape()[0];
    if (ShapedType::isDynamic(C))
        return nullptr;

    SmallVector<int32_t> multVals(C, multiplier);
    auto multTy = RankedTensorType::get({C}, rewriter.getI32Type());
    auto multCst =
        arith::ConstantOp::create(rewriter, loc, multTy, rewriter.getI32TensorAttr(multVals))
            .getResult();

    auto sbTy = RankedTensorType::get({C * 2}, rewriter.getI32Type());
    auto init =
        tensor::EmptyOp::create(rewriter, loc, sbTy.getShape(), sbTy.getElementType()).getResult();
    // Place biases at even offsets (stride 2) and scales at odd offsets (stride 2)
    // so the final layout is [bias_0, mult, bias_1, mult, ...].
    auto withBias = tensor::InsertSliceOp::create(
        rewriter, loc, bias, init, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(C)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(2)}
    );
    auto scaleBias = tensor::InsertSliceOp::create(
        rewriter, loc, multCst, withBias, SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(C)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(2)}
    );
    return scaleBias;
}

// Return a non-null operation to attach diagnostics to. Prefer the defining op;
// for block arguments fall back to the first user, which is always present when
// these helpers are called from a pattern.
static Operation *getDiagOp(Value candidate) {
    if (Operation *op = candidate.getDefiningOp())
        return op;
    for (Operation *user : candidate.getUsers())
        return user;
    return nullptr;
}

LogicalResult
matchInputQuantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain) {
    Operation *anchor = getDiagOp(candidate);
    chain.inputDequantOp = dyn_cast_or_null<linalg::GenericOp>(candidate.getDefiningOp());
    if (!chain.inputDequantOp || !chain.inputDequantOp->getResult(0).hasOneUse())
        return rewriter.notifyMatchFailure(anchor, "failed to match input Q dequant");
    if (!matchDequantGeneric(chain.inputDequantOp, chain.inputInfo.scale, chain.inputInfo.zp))
        return rewriter.notifyMatchFailure(
            chain.inputDequantOp, "not a recognized input Q dequant"
        );
    return success();
}

LogicalResult
matchOutputDequantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain) {
    Operation *anchor = getDiagOp(candidate);
    if (!candidate.hasOneUse())
        return rewriter.notifyMatchFailure(anchor, "output candidate has multiple uses");

    chain.outputDequantOp = dyn_cast<linalg::GenericOp>(*candidate.getUsers().begin());
    if (!chain.outputDequantOp || !chain.outputDequantOp->getResult(0).hasOneUse())
        return rewriter.notifyMatchFailure(
            chain.outputDequantOp ? chain.outputDequantOp.getOperation() : anchor,
            "failed to match output Q dequant"
        );
    if (!matchDequantGeneric(
            chain.outputDequantOp, chain.outputDequantInfo.scale, chain.outputDequantInfo.zp
        ))
        return rewriter.notifyMatchFailure(
            chain.outputDequantOp, "not a recognized output Q dequant"
        );

    return success();
}

LogicalResult
matchOutputQuantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain) {
    Operation *anchor = getDiagOp(candidate);
    if (!candidate.hasOneUse())
        return rewriter.notifyMatchFailure(anchor, "quant input has multiple uses");

    chain.quantOp = dyn_cast<linalg::GenericOp>(*candidate.getUsers().begin());
    if (!chain.quantOp || !matchQuantGeneric(
                              chain.quantOp, chain.outputInfo.scale, chain.outputInfo.zp,
                              chain.outputInfo.min, chain.outputInfo.max
                          ))
        return rewriter.notifyMatchFailure(
            chain.quantOp ? chain.quantOp.getOperation() : anchor, "failed to match Q quant"
        );

    return success();
}

// Return the scalar tensor operand feeding block argument `v` of `op`, or null
// if `v` is not a runtime scalar block argument.
static Value getRuntimeScalarOperand(Value v, linalg::GenericOp op) {
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg || blockArg.getOwner() != op.getBody())
        return nullptr;
    unsigned idx = blockArg.getArgNumber();
    if (idx >= op.getNumDpsInputs())
        return nullptr;
    Value operand = op.getDpsInputOperand(idx)->get();
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType || tensorType.getNumElements() != 1)
        return nullptr;
    if (getQFloatScalar(operand))
        return nullptr; // compile-time constant -> not the runtime path
    return operand;
}

// Unsigned runtime quant (DynamicQuantizeLinear):
//   mulf(x, %inv_scale) -> addf(%zp) -> (math.roundeven) -> maximumf(min)
//   -> minimumf(max) -> arith.fptoui
// where %inv_scale and %zp are scalar tensor operands (not constants). %zp may
// be float or an integer converted with sitofp. Unlike the constant flavors,
// scale/zp cannot be folded to doubles, so they are returned as SSA operands.
bool matchQuantRuntime(linalg::GenericOp op, Value &invScale, Value &zp, double &min, double &max) {
    if (!op || op.getNumDpsInputs() < 2 || op.getNumDpsInits() != 1)
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;
    auto inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!inTy || !outTy)
        return false;
    if (!outTy.getElementType().isInteger(8))
        return false;
    if (!inTy.getElementType().isBF16() && !inTy.getElementType().isF32())
        return false;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;
    auto fptoui = yieldOp.getOperand(0).getDefiningOp<arith::FPToUIOp>();
    if (!fptoui)
        return false;

    Value chain;
    if (!extractQuantClampBounds(fptoui.getIn(), op, min, max, chain))
        return false;

    // Rounding is optional: the HW f2i epilogue rounds regardless, so the
    // linalg body may or may not carry an explicit roundeven.
    if (auto roundOp = chain.getDefiningOp<math::RoundEvenOp>())
        chain = roundOp.getOperand();

    // addf(mul, %zp): one operand is the scalar zero-point.
    auto addf = chain.getDefiningOp<arith::AddFOp>();
    if (!addf)
        return false;
    auto getRuntimeZp = [&](Value value) {
        if (auto castOp = value.getDefiningOp<arith::SIToFPOp>())
            value = castOp.getIn();
        return getRuntimeScalarOperand(value, op);
    };
    Value mulVal;
    if ((zp = getRuntimeZp(addf.getLhs())))
        mulVal = addf.getRhs();
    else if ((zp = getRuntimeZp(addf.getRhs())))
        mulVal = addf.getLhs();
    else
        return false;

    // mulf(x, %inv_scale): one operand is the scalar reciprocal scale.
    auto mulf = mulVal.getDefiningOp<arith::MulFOp>();
    if (!mulf)
        return false;
    if ((invScale = getRuntimeScalarOperand(mulf.getLhs(), op)))
        return true;
    if ((invScale = getRuntimeScalarOperand(mulf.getRhs(), op)))
        return true;
    return false;
}

} // namespace mlir::syna::torq
