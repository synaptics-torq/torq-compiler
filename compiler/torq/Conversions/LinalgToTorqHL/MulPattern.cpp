// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-mul-pattern"

namespace mlir::syna::torq {

static Value create1DimTensorFromScalar(
    arith::ConstantOp constOp, const Type &elementType, PatternRewriter &rewriter
) {
    if (!constOp)
        return {};

    if (elementType.isInteger()) {

        int32_t data = 0;
        if (!getIntegerConstantValue(constOp, &data)) {
            llvm::errs() << "cannot get constant value\n";
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

        return arith::ConstantOp::create(rewriter, constOp.getLoc(), constType, value).getResult();
    }
    else if (elementType.isBF16() || elementType.isF32()) {

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

        return arith::ConstantOp::create(rewriter, constOp.getLoc(), constType, value).getResult();
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

        int32_t bias = 0;
        int32_t scale = 1;
        Value output = srcOp.getResult(0);
        const int outChannelCount = 1; // No channels here, only one single scale
        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannelCount, 12, 12, false);
        auto outType = cast<RankedTensorType>(output.getType());
        auto [outMin, outMax] = getDTypeRange(outType.getElementType());
        int32_t outZp = 0;

        // shiftFactor is used for torq hw scale computation, it request multiple of 4.
        // for 48bit value rescale, we need to check multiplier bit < 32bit in case overflow 80bit.
        int32_t shiftFactor = 0;
        if (scInfo) {
            // Default
            shiftFactor = scInfo.scaleShift;
            int32_t hwScale = static_cast<int32_t>(scInfo.scaleDouble[0] * (1ull << shiftFactor));
            bias = scInfo.bias;
            scale = hwScale;
            outMin = scInfo.min;
            outMax = scInfo.max;
            outZp = scInfo.zp;
        }

        LLVM_DEBUG(
            llvm::dbgs() << "bias: " << bias << " scale: " << scale
                         << " shiftFactor: " << shiftFactor << " outMin: " << outMin
                         << " outMax: " << outMax << " outZp: " << outZp << "\n"
        );

        Value torqOut =
            torq_hl::MulOp::create(
                rewriter, srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                outZp, outMin, outMax, createI32Const(rewriter, srcOp, interleave({bias}, {scale})),
                shiftFactor, input1, input2
            )
                .getResult(0);

        if (scInfo) {
            rewriter.replaceOp(output.getDefiningOp(), torqOut);
            rewriter.eraseOp(srcOp);
        }
        else {
            rewriter.replaceOp(srcOp, torqOut);
        }

        return success();
    }
};

/// ============================================================================
/// MulRescaleOpPattern
/// ============================================================================
///
/// Matches a fused mul+rescale linalg.generic with the body:
///   ^bb0(%in0: i16, %in1: i16, %out: i8):
///     %0 = arith.muli %in0, %in1 : i16
///     %1 = arith.extsi %0 : i16 to i32
///     %2 = tosa.apply_scale %1, %mult, %shift {double_round = true}
///     %3 = arith.addi %2, %c_output_zp : i32
///     %4 = arith.maxsi %3, %c_min : i32
///     %5 = arith.minsi %4, %c_max : i32
///     %6 = arith.trunci %5 : i32 to i8
///     linalg.yield %6 : i8
///
/// No input zero-point subtraction. Extracts output quantization parameters
/// and lowers to torq_hl::MulOp.
///
class MulRescaleOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {

        LLVM_DEBUG(llvm::dbgs() << "\n[MulRescale] Checking op: " << srcOp << "\n");

        // Must have 2 inputs and 1 output
        if (srcOp.getNumDpsInputs() != 2 || srcOp.getNumDpsInits() != 1) {
            return rewriter.notifyMatchFailure(
                srcOp, "[MulRescale] Expected 2 inputs and 1 output"
            );
        }

        // All loops must be parallel
        if (srcOp.getNumParallelLoops() != srcOp.getNumLoops()) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] Not all loops are parallel");
        }

        // --- Walk the body from yield backwards ---
        auto yieldOp = dyn_cast<linalg::YieldOp>(srcOp.getBody()->getTerminator());
        if (!yieldOp || yieldOp.getNumOperands() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] Bad yield");
        }

        Value val = yieldOp.getOperand(0);

        // trunci i32 -> i8
        auto truncOp = val.getDefiningOp<arith::TruncIOp>();
        if (!truncOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No trunci");
        }
        val = truncOp.getIn();

        // minsi (clamp max)
        auto minsiOp = val.getDefiningOp<arith::MinSIOp>();
        if (!minsiOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No minsi");
        }
        auto maybeOutMax = getConstIntValue(minsiOp.getRhs());
        if (!maybeOutMax) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] minsi rhs not constant");
        }
        int32_t outMax = static_cast<int32_t>(*maybeOutMax);
        val = minsiOp.getLhs();

        // maxsi (clamp min)
        auto maxsiOp = val.getDefiningOp<arith::MaxSIOp>();
        if (!maxsiOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No maxsi");
        }
        auto maybeOutMin = getConstIntValue(maxsiOp.getRhs());
        if (!maybeOutMin) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] maxsi rhs not constant");
        }
        int32_t outMin = static_cast<int32_t>(*maybeOutMin);
        val = maxsiOp.getLhs();

        // addi (output zero-point)
        int32_t outZp = 0;
        if (auto addiOp = val.getDefiningOp<arith::AddIOp>()) {
            auto maybeOutZp = getConstIntValue(addiOp.getRhs());
            if (!maybeOutZp) {
                return rewriter.notifyMatchFailure(srcOp, "[MulRescale] addi rhs not constant");
            }
            outZp = static_cast<int32_t>(*maybeOutZp);
            val = addiOp.getLhs();
        }

        // tosa.apply_scale
        auto applyScaleOp = val.getDefiningOp<tosa::ApplyScaleOp>();
        if (!applyScaleOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No apply_scale");
        }

        auto ms = getMultiplierAndShift(srcOp, applyScaleOp, 1);
        if (!ms) {
            return rewriter.notifyMatchFailure(
                srcOp, "[MulRescale] Failed to get multiplier/shift"
            );
        }

        val = applyScaleOp.getValue();

        // extsi i16 -> i32
        auto extsiOp = val.getDefiningOp<arith::ExtSIOp>();
        if (!extsiOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No extsi after muli");
        }
        val = extsiOp.getIn();

        // arith.muli
        auto muliOp = val.getDefiningOp<arith::MulIOp>();
        if (!muliOp) {
            return rewriter.notifyMatchFailure(srcOp, "[MulRescale] No muli");
        }

        // Both muli operands must be block arguments (no input zp subtraction)
        if (!isa<BlockArgument>(muliOp.getLhs()) || !isa<BlockArgument>(muliOp.getRhs())) {
            return rewriter.notifyMatchFailure(
                srcOp, "[MulRescale] muli operands are not block args"
            );
        }

        LLVM_DEBUG(
            llvm::dbgs() << "[MulRescale] Matched! outZp=" << outZp << ", outMin=" << outMin
                         << ", outMax=" << outMax << "\n"
        );

        // --- Compute HW parameters ---
        Value input1 = srcOp.getInputs()[0];
        Value input2 = srcOp.getInputs()[1];

        auto outType = cast<RankedTensorType>(srcOp.getResult(0).getType());

        int32_t shiftFactor = ms.shift[0];
        int32_t bias = 0;
        int32_t scale = ms.multiplier[0];

        LLVM_DEBUG(
            llvm::dbgs() << "[MulRescale] shiftFactor=" << shiftFactor << ", scale=" << scale
                         << ", bias=" << bias << "\n"
        );

        // --- Create torq_hl::MulOp ---
        Value torqOut =
            torq_hl::MulOp::create(
                rewriter, srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                outZp, outMin, outMax, createI32Const(rewriter, srcOp, interleave({bias}, {scale})),
                shiftFactor, input1, input2
            )
                .getResult(0);

        rewriter.replaceOp(srcOp, torqOut);

        LLVM_DEBUG(llvm::dbgs() << "[MulRescale] Replaced with torq_hl::MulOp\n");

        return success();
    }
};

void populateLinalgToTorqHLMulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // FIXME (upgrade):
    // patterns.insert<MulOpPattern>(context);
    // patterns.insert<MulRescaleOpPattern>(context);
}

} // namespace mlir::syna::torq
