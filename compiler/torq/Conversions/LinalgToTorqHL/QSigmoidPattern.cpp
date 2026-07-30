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
#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <limits>

#define DEBUG_TYPE "linalg-torq-q-sigmoid-pattern"

namespace mlir::syna::torq {

namespace {

// Match a linalg.generic whose body is the sigmoid function:
//   div(1.0, add(exp(neg(x)), 1.0))
// The input may be f32 or bf16 (whatever the dequant produced).
bool matchSigmoidGeneric(linalg::GenericOp op) {
    auto isFloat = [](Type t) { return t.isF32() || t.isBF16(); };
    TorqStructuredOpMatcher<> matcher;
    return matcher.numInputs(1)
        .numOutputs(1)
        .inputElementType(0, isFloat)
        .yieldChain<arith::DivFOp, arith::AddFOp, math::ExpOp, arith::NegFOp>()
        .match(op);
}

// Build a 256-entry i8 lookup table that fuses dequant -> sigmoid -> quant.
// table[i] maps input i8 value (i - 128) to the quantized sigmoid output.
SmallVector<int32_t> buildSigmoidTable(const DequantInfo &dInfo, const QuantInfo &qInfo) {
    SmallVector<int32_t> table;
    table.reserve(256);

    for (int qIn = -128; qIn <= 127; ++qIn) {
        double x = (static_cast<double>(qIn) - static_cast<double>(dInfo.zp)) * dInfo.scale;
        double y = 1.0 / (1.0 + std::exp(-x));
        double qOutDouble = y / qInfo.scale + qInfo.zp;

        int64_t qOut = std::llround(qOutDouble);
        qOut = std::clamp<int64_t>(qOut, std::llround(qInfo.min), std::llround(qInfo.max));
        qOut = std::clamp<int64_t>(qOut, -128, 127);

        table.push_back(static_cast<int32_t>(qOut << 8));
    }

    return table;
}

struct QSigmoidConvert : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    QSigmoidConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp sigmoidOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(sigmoidOp))
            return rewriter.notifyMatchFailure(sigmoidOp, "already marked");

        if (!matchSigmoidGeneric(sigmoidOp))
            return rewriter.notifyMatchFailure(sigmoidOp, "not a sigmoid generic");

        QuantizedOpChain chain;
        if (failed(matchOutputQuantization(sigmoidOp.getResult(0), rewriter, chain)))
            return rewriter.notifyMatchFailure(sigmoidOp, "failed to match quant generic");

        if (failed(matchInputQuantization(sigmoidOp.getInputs()[0], rewriter, chain)))
            return rewriter.notifyMatchFailure(sigmoidOp, "failed to match dequant generic");

        Value input = chain.inputDequantOp.getInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto outputType = dyn_cast<RankedTensorType>(chain.quantOp.getResult(0).getType());
        if (!inputType || !outputType || !inputType.getElementType().isInteger(8) ||
            !outputType.getElementType().isInteger(8)) {
            return rewriter.notifyMatchFailure(sigmoidOp, "expected i8 input/output");
        }

        if (_markFuseGroups) {
            auto fuseGroupAttr = sigmoidOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(sigmoidOp, "missing fuse group id");
            markFuseGroupBackward(chain.quantOp.getResult(0), {input}, rewriter, fuseGroupAttr);
            return success();
        }

        if (chain.outputInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(sigmoidOp, "quant scale is zero");

        LLVM_DEBUG(llvm::dbgs() << "[QSigmoidConvert] input_scale=" << chain.inputInfo.scale
                                << " input_zp=" << chain.inputInfo.zp
                                << " output_scale=" << chain.outputInfo.scale
                                << " output_zp=" << chain.outputInfo.zp << "\n";);

        SmallVector<int32_t> tableValues = buildSigmoidTable(chain.inputInfo, chain.outputInfo);
        Value packedTable =
            createI32Const(rewriter, sigmoidOp, tableValues, llvm::ArrayRef<int64_t>{256});

        // TableOp's fixed scale_bias maps i8 [-128,127] to unsigned index [0,255].
        std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
        std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};
        Value scaleBias = createIConst(rewriter, sigmoidOp, interleave(bias, scale));

        Location loc = sigmoidOp.getLoc();
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(chain.quantOp);

        Value tableResult =
            torq_hl::TableOp::create(
                rewriter, loc, outputType, createInitTensor(chain.quantOp, rewriter, outputType),
                scaleBias, input, packedTable, nullptr
            )
                .getResult(0);

        rewriter.replaceOp(chain.quantOp, tableResult);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQSigmoidPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QSigmoidConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
