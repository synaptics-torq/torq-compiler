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

#define DEBUG_TYPE "linalg-torq-q-muldiv-pattern"

namespace mlir::syna::torq {

namespace {

enum class MulDivKind { Mul, Div };

// Match a mul/div body that is either:
//   arith.mulf(%in0, %in1)   -> MulDivKind::Mul
//   arith.divf(%in0, %in1)   -> MulDivKind::Div
// Both operands must be block arguments; scalar variants are not supported yet.
bool matchMulDivBody(linalg::GenericOp op, int &inputIdx0, int &inputIdx1, MulDivKind &kind) {
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return false;
    if (op.getNumParallelLoops() != op.getNumLoops())
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;

    Value yieldVal = yieldOp.getOperand(0);
    if (auto mulf = yieldVal.getDefiningOp<arith::MulFOp>()) {
        kind = MulDivKind::Mul;
    }
    else if (auto divf = yieldVal.getDefiningOp<arith::DivFOp>()) {
        kind = MulDivKind::Div;
    }
    else {
        return false;
    }

    auto binaryOp = yieldVal.getDefiningOp();
    assert(binaryOp && "yield value must define a binary op");

    auto resolve = [&](Value v) -> std::optional<int> {
        auto ba = dyn_cast<BlockArgument>(v);
        if (!ba || ba.getOwner() != op.getBody())
            return std::nullopt;
        if (ba.getArgNumber() >= op.getNumDpsInputs())
            return std::nullopt;
        return static_cast<int>(ba.getArgNumber());
    };

    auto idx0 = resolve(binaryOp->getOperand(0));
    auto idx1 = resolve(binaryOp->getOperand(1));
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

struct DequantSource {
    linalg::GenericOp dequant;
    Value quantizedInput;
    DequantInfo info;
};

// Walk backward from a value that feeds a binary mul/div to find the underlying
// quantized input and its dequant metadata. The path may contain tensor shape
// ops (collapse_shape/expand_shape/extract_slice) and identity broadcast
// generics that ONNX QDQ lowers between a small-rank dequant and the
// full-rank mul/div.
std::optional<DequantSource> findQuantizedDequantSource(Value v) {
    while (true) {
        if (auto dequant = dyn_cast_or_null<linalg::GenericOp>(v.getDefiningOp())) {
            DequantInfo info;
            if (matchDequantGeneric(dequant, info.scale, info.zp)) {
                return DequantSource{dequant, dequant.getInputs()[0], info};
            }
            // Not a dequant generic: it may be an identity broadcast generic, so
            // fall through to the broadcast check below.
        }

        if (auto collapse = dyn_cast_or_null<tensor::CollapseShapeOp>(v.getDefiningOp())) {
            if (!collapse->getResult(0).hasOneUse())
                return std::nullopt;
            v = collapse.getSrc();
            continue;
        }

        if (auto expand = dyn_cast_or_null<tensor::ExpandShapeOp>(v.getDefiningOp())) {
            if (!expand->getResult(0).hasOneUse())
                return std::nullopt;
            v = expand.getSrc();
            continue;
        }

        if (auto extract = dyn_cast_or_null<tensor::ExtractSliceOp>(v.getDefiningOp())) {
            if (!extract->getResult(0).hasOneUse())
                return std::nullopt;
            v = extract.getSource();
            continue;
        }

        auto broadcast = dyn_cast_or_null<linalg::GenericOp>(v.getDefiningOp());
        if (!broadcast)
            return std::nullopt;

        // Identity broadcast generic: one input, one output, and the body yields
        // the sole input block argument unchanged.
        if (broadcast.getNumDpsInputs() != 1 || broadcast.getNumDpsInits() != 1 ||
            !broadcast.getRegion().hasOneBlock())
            return std::nullopt;
        auto yieldOp = dyn_cast<linalg::YieldOp>(broadcast.getBody()->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1)
            return std::nullopt;
        auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
        if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != broadcast.getBody())
            return std::nullopt;
        if (!broadcast->getResult(0).hasOneUse())
            return std::nullopt;

        v = broadcast.getInputs()[0];
    }
}

// Subtract the zero-point from a quantized i8 input and widen to i16:
//   out = input - zp
// Implemented as FMAOp with weight=1, bias=-zp, scale=1, shift=0.
// When zp == 0 this is simply an i8 -> i16 cast. The output is i16 so the
// zero-point-corrected value and the downstream mul/div stay in range.
Value subtractZeroPoint(
    PatternRewriter &rewriter, linalg::GenericOp anchorOp, Value input, int32_t zp
) {
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTy)
        return {};
    auto i16Type = RankedTensorType::get(inputTy.getShape(), rewriter.getI16Type());

    std::vector<int8_t> weights = {1};
    std::vector<int32_t> bias = {-zp};
    std::vector<int32_t> scale = {1};

    return torq_hl::FMAOp::create(
               rewriter, anchorOp.getLoc(), i16Type, createInitTensor(anchorOp, rewriter, i16Type),
               /*output_zp=*/0,
               /*output_min=*/std::numeric_limits<int16_t>::min(),
               /*output_max=*/std::numeric_limits<int16_t>::max(),
               /*shift_factor=*/0,
               createI8Const(rewriter, anchorOp, weights, llvm::ArrayRef<int64_t>{1}),
               createI32Const(rewriter, anchorOp, interleave(bias, scale)), input
    )
        .getResult(0);
}

struct QMulDivConvert : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    QMulDivConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp mulOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(mulOp))
            return rewriter.notifyMatchFailure(mulOp, "already marked");

        int inputIdx0 = -1, inputIdx1 = -1;
        MulDivKind kind;
        if (!matchMulDivBody(mulOp, inputIdx0, inputIdx1, kind))
            return rewriter.notifyMatchFailure(mulOp, "not a mul/div");

        if (kind != MulDivKind::Mul)
            return rewriter.notifyMatchFailure(
                mulOp, "quantized div is not supported yet (needs reciprocal LUT)"
            );

        Value mulOutput = mulOp.getResult(0);
        if (!mulOutput.hasOneUse())
            return rewriter.notifyMatchFailure(mulOp, "mul output has multiple uses");

        auto quantOp = dyn_cast<linalg::GenericOp>(*mulOutput.getUsers().begin());
        QuantInfo qInfo;
        if (!quantOp || !matchQuantGeneric(quantOp, qInfo.scale, qInfo.zp, qInfo.min, qInfo.max))
            return rewriter.notifyMatchFailure(mulOp, "failed to match Q quant");

        Value mulDivInput0 = mulOp.getInputs()[inputIdx0];
        Value mulDivInput1 = mulOp.getInputs()[inputIdx1];

        auto source0 = findQuantizedDequantSource(mulDivInput0);
        auto source1 = findQuantizedDequantSource(mulDivInput1);
        if (!source0)
            return rewriter.notifyMatchFailure(mulOp, "failed to match first Q dequant");
        if (!source1)
            return rewriter.notifyMatchFailure(mulOp, "failed to match second Q dequant");

        if (isScalarTensor(source0->quantizedInput) || isScalarTensor(source1->quantizedInput))
            return rewriter.notifyMatchFailure(mulOp, "scalar input not supported");

        const DequantInfo &dInfo0 = source0->info;
        const DequantInfo &dInfo1 = source1->info;

        // Discovery mode: mark the dq -> mul -> q chain as a single fusion group.
        if (_markFuseGroups) {
            auto fuseGroupAttr = mulOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!fuseGroupAttr)
                return rewriter.notifyMatchFailure(mulOp, "missing fuse group id");
            markFuseGroupBackward(
                quantOp.getResult(0), {source0->quantizedInput, source1->quantizedInput}, rewriter,
                fuseGroupAttr
            );
            return success();
        }

        if (qInfo.scale == 0.0)
            return rewriter.notifyMatchFailure(mulOp, "quant scale is zero");

        double combinedScale = dInfo0.scale * dInfo1.scale / qInfo.scale;
        int32_t multiplier;
        int32_t shift;
        if (!computeMultiplierAndShift(combinedScale, multiplier, shift))
            return rewriter.notifyMatchFailure(mulOp, "failed to compute multiplier/shift");

        LLVM_DEBUG(llvm::dbgs() << "[QMulDivConvert] combinedScale=" << combinedScale
                                << " multiplier=" << multiplier << " shift=" << shift << "\n";);

        int32_t outputZp = static_cast<int32_t>(std::llround(qInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(qInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(qInfo.max));

        Location loc = mulOp.getLoc();
        auto outTy = dyn_cast<RankedTensorType>(quantOp.getResult(0).getType());
        if (!outTy)
            return rewriter.notifyMatchFailure(mulOp, "expected ranked output type");

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(quantOp);

        // Subtract each input's zero-point and widen to i16 (zp=0 becomes a cast).
        Value lhs = subtractZeroPoint(rewriter, mulOp, source0->quantizedInput, dInfo0.zp);
        Value rhs = subtractZeroPoint(rewriter, mulOp, source1->quantizedInput, dInfo1.zp);
        if (!lhs || !rhs)
            return rewriter.notifyMatchFailure(mulOp, "failed to subtract zero-point");

        std::vector<int32_t> bias = {0};
        std::vector<int32_t> scale = {multiplier};
        Value scaleBias = createI32Const(rewriter, mulOp, interleave(bias, scale));

        Value mulResult =
            torq_hl::MulOp::create(
                rewriter, loc, outTy, createInitTensor(quantOp, rewriter, outTy), outputZp,
                outputMin, outputMax, scaleBias, static_cast<int8_t>(shift), lhs, rhs
            )
                .getResult(0);

        rewriter.replaceOp(quantOp, mulResult);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQMulDivPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QMulDivConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
