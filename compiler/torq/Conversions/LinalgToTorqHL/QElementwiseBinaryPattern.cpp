// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/QuantPatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <limits>
#include <optional>

#define DEBUG_TYPE "linalg-torq-q-elementwise-binary-pattern"

namespace mlir::syna::torq {

namespace {

// Match an elementwise add whose body is either:
//   arith.addf(%in0, %in1)
// or (for broadcast variants)
//   arith.addf(%in0, arith.mulf(%in1, 1.0))
bool matchElementwiseAddBody(linalg::GenericOp op, int &inputIdx0, int &inputIdx1) {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return false;
    if (op.getNumParallelLoops() != op.getNumLoops())
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;

    auto addf = yieldOp.getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!addf)
        return false;

    auto resolve = [&](Value v) -> std::optional<int> {
        if (auto mulf = v.getDefiningOp<arith::MulFOp>()) {
            Value other = nullptr;
            Value blockArg = nullptr;
            for (Value operand : {mulf.getLhs(), mulf.getRhs()}) {
                if (auto ba = dyn_cast<BlockArgument>(operand)) {
                    blockArg = ba;
                }
                else {
                    other = operand;
                }
            }
            if (!blockArg || !other)
                return std::nullopt;
            auto maybeC = getQGenericFloatConstant(other, op);
            if (!maybeC || std::fabs(*maybeC - 1.0) > 1e-6)
                return std::nullopt;
            v = blockArg;
        }
        auto ba = dyn_cast<BlockArgument>(v);
        if (!ba || ba.getOwner() != op.getBody())
            return std::nullopt;
        if (ba.getArgNumber() >= op.getNumDpsInputs())
            return std::nullopt;
        return static_cast<int>(ba.getArgNumber());
    };

    auto idx0 = resolve(addf.getLhs());
    auto idx1 = resolve(addf.getRhs());
    if (!idx0 || !idx1)
        return false;
    inputIdx0 = *idx0;
    inputIdx1 = *idx1;
    return true;
}

bool isScalarTensor(Value v) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty)
        return false;
    int64_t numElements = 1;
    for (int64_t dim : ty.getShape()) {
        if (dim == ShapedType::kDynamic)
            return false;
        numElements *= dim;
    }
    return numElements == 1;
}

// Validate that the matched dq -> elementwise -> q chain is one we can
// lower to a torq_hl.add today.
// TODO: extend to arith.subf (needs sign handling on weights) and to
// scalar/broadcasted inputs (needs rhs_is_scalar and folding the scalar
// value into the bias).
bool isSupportedQAddPattern(linalg::GenericOp addOp, Value input0, Value input1) {
    auto addf = addOp.getBody()->getTerminator()->getOperand(0).getDefiningOp<arith::AddFOp>();
    if (!addf)
        return false;
    if (isScalarTensor(input0) || isScalarTensor(input1))
        return false;
    return true;
}

bool computeSharedMultiplierAndShift(
    double scale0, double scale1, double outScale, int16_t &m0, int16_t &m1, int32_t &shift
) {
    if (outScale == 0.0)
        return false;
    double r0 = scale0 / outScale;
    double r1 = scale1 / outScale;

    if (r0 == 0.0 && r1 == 0.0) {
        m0 = m1 = 0;
        shift = 0;
        return true;
    }

    constexpr int32_t kMaxShift = 60;
    for (int32_t s = kMaxShift; s >= 0; s -= 4) {
        double factor = static_cast<double>(1LL << s);
        int64_t v0 = std::llround(r0 * factor);
        int64_t v1 = std::llround(r1 * factor);
        if (v0 >= std::numeric_limits<int16_t>::min() &&
            v0 <= std::numeric_limits<int16_t>::max() &&
            v1 >= std::numeric_limits<int16_t>::min() &&
            v1 <= std::numeric_limits<int16_t>::max()) {
            m0 = static_cast<int16_t>(v0);
            m1 = static_cast<int16_t>(v1);
            shift = s;
            return true;
        }
    }
    return false;
}

struct QElementwiseBinaryConvert : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    QElementwiseBinaryConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp addOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(addOp))
            return rewriter.notifyMatchFailure(addOp, "already marked");

        int inputIdx0 = -1, inputIdx1 = -1;
        if (!matchElementwiseAddBody(addOp, inputIdx0, inputIdx1))
            return rewriter.notifyMatchFailure(addOp, "not an elementwise add");

        Value addOutput = addOp.getResult(0);
        if (!addOutput.hasOneUse())
            return rewriter.notifyMatchFailure(addOp, "add output has multiple uses");

        auto quantOp = dyn_cast<linalg::GenericOp>(*addOutput.getUsers().begin());
        QuantInfo qInfo;
        if (!quantOp || !matchQuantGeneric(quantOp, qInfo.scale, qInfo.zp, qInfo.min, qInfo.max))
            return rewriter.notifyMatchFailure(addOp, "failed to match Q quant");

        Value dequantInput0 = addOp.getInputs()[inputIdx0];
        Value dequantInput1 = addOp.getInputs()[inputIdx1];
        auto dequant0 = dyn_cast<linalg::GenericOp>(dequantInput0.getDefiningOp());
        auto dequant1 = dyn_cast<linalg::GenericOp>(dequantInput1.getDefiningOp());
        DequantInfo dInfo0, dInfo1;
        if (!dequant0 || !matchDequantGeneric(dequant0, dInfo0.scale, dInfo0.zp) ||
            !dequant0->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(addOp, "failed to match first Q dequant");
        if (!dequant1 || !matchDequantGeneric(dequant1, dInfo1.scale, dInfo1.zp) ||
            !dequant1->getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(addOp, "failed to match second Q dequant");

        if (!isSupportedQAddPattern(addOp, dequant0.getInputs()[0], dequant1.getInputs()[0])) {
            return rewriter.notifyMatchFailure(
                addOp, "unsupported quantized elementwise pattern (sub or scalar input)"
            );
        }

        // Discovery mode: mark the dq -> add -> q chain as a single fusion group.
        if (_markFuseGroups) {
            auto fuseGroupAttr = addOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(addOp, "missing fuse group id");
            markFuseGroupBackward(
                quantOp.getResult(0), {dequant0.getInputs()[0], dequant1.getInputs()[0]}, rewriter,
                fuseGroupAttr
            );
            return success();
        }

        if (qInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(addOp, "quant scale is zero");

        int16_t m0, m1;
        int32_t shift;
        if (!computeSharedMultiplierAndShift(
                dInfo0.scale, dInfo1.scale, qInfo.scale, m0, m1, shift
            ))
            return rewriter.notifyMatchFailure(addOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QElementwiseBinaryConvert] m0=" << m0 << " m1=" << m1
                                << " shift=" << shift << "\n";);

        int64_t bias64 = -static_cast<int64_t>(dInfo0.zp) * static_cast<int32_t>(m0) -
                         static_cast<int64_t>(dInfo1.zp) * static_cast<int32_t>(m1);
        int32_t bias = static_cast<int32_t>(std::clamp<int64_t>(
            bias64, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()
        ));

        int32_t outputZp = static_cast<int32_t>(std::llround(qInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(qInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(qInfo.max));

        Location loc = addOp.getLoc();
        auto outTy = dyn_cast<RankedTensorType>(quantOp.getResult(0).getType());
        if (!outTy)
            return rewriter.notifyMatchFailure(addOp, "expected ranked output type");

        Value weights = createI16Const(rewriter, addOp, {m0, m1}, llvm::ArrayRef<int64_t>{2});
        std::vector<int32_t> biasScale =
            interleave(std::vector<int32_t>{bias}, std::vector<int32_t>{1});
        Value scaleBias = createI32Const(rewriter, addOp, biasScale);

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(quantOp);
        Value addResult = torq_hl::AddOp::create(
                              rewriter, loc, outTy, quantOp.getDpsInitOperand(0)->get(),
                              rewriter.getStringAttr("add"),
                              /*input_zp=*/0, outputZp, outputMin, outputMax, shift, weights,
                              scaleBias, dequant0.getInputs()[0], dequant1.getInputs()[0],
                              /*segment_output=*/false, /*rhs_is_scalar=*/false
        )
                              .getOutput();

        rewriter.replaceOp(quantOp, addResult);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQEWBinaryPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QElementwiseBinaryConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
