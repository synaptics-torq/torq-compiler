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
    if (auto trunc = v.getDefiningOp<arith::TruncFOp>())
        return getQGenericFloatConstant(trunc.getIn(), op);
    if (auto ext = v.getDefiningOp<arith::ExtFOp>())
        return getQGenericFloatConstant(ext.getIn(), op);
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (!op || blockArg.getOwner() != op.getBody())
            return std::nullopt;
        unsigned idx = blockArg.getArgNumber();
        if (idx >= op.getNumDpsInputs())
            return std::nullopt;
        return getQFloatScalar(op.getDpsInputOperand(idx)->get());
    }
    return getQFloatScalar(v);
}

static std::optional<int64_t> getConstIntValue(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    if (auto iattr = dyn_cast<IntegerAttr>(cst->getAttr("value")))
        return iattr.getInt();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst->getAttr("value"))) {
        if (dense.isSplat())
            return dense.getSplatValue<int64_t>();
        if (dense.size() == 1)
            return *dense.getValues<int64_t>().begin();
    }
    return std::nullopt;
}

static std::optional<int64_t> getQGenericIntConstant(Value v, linalg::GenericOp op) {
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
        if (!op || blockArg.getOwner() != op.getBody())
            return std::nullopt;
        unsigned idx = blockArg.getArgNumber();
        if (idx >= op.getNumDpsInputs())
            return std::nullopt;
        return getConstIntValue(op.getDpsInputOperand(idx)->get());
    }
    return getConstIntValue(v);
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

// All registered dequant flavors, tried in order.  The first match wins.
using DequantFlavorFn = std::optional<DequantInfo> (*)(linalg::GenericOp);
static const DequantFlavorFn kDequantFlavors[] = {
    matchDequantSigned,   // standard ONNX signed int8 QDQ
    matchDequantUnsigned, // ONNX unsigned INT4 QDQ (block_size path)
};

bool matchDequantGeneric(linalg::GenericOp op, double &scale, int32_t &zp) {
    for (auto f : kDequantFlavors) {
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
static bool
extractQuantClampBounds(Value v, linalg::GenericOp op, double &min, double &max, Value &data) {
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

// All registered quant flavors, tried in order.  The first match wins.
using QuantFlavorFn = std::optional<QuantInfo> (*)(linalg::GenericOp);
static const QuantFlavorFn kQuantFlavors[] = {
    matchQuantSigned, // standard ONNX signed int8 QDQ
                      // Future: matchQuantUnsigned, etc.
};

bool matchQuantGeneric(linalg::GenericOp op, double &scale, double &zp, double &min, double &max) {
    for (auto f : kQuantFlavors) {
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

} // namespace mlir::syna::torq
