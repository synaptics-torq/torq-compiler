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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-mul-pattern"

namespace mlir::syna::torq {

class MulOpPattern : public OpRewritePattern<linalg::GenericOp> {
  private:
    const bool _markFuseGroups;

  public:
    MulOpPattern(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().empty() || srcOp.getInputs().size() > 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "mul expects generic op with 2 (or 1) inputs\n"
            );
        }

        Value input1 = srcOp.getInputs().front();
        // Binary ops can actually have only one input if both operands are the same
        Value input2 = srcOp.getInputs().back();

        auto input1Type = dyn_cast<RankedTensorType>(input1.getType());
        auto input1ElementType = input1Type.getElementType();

        if (input1ElementType.isF32() || input1ElementType.isF64()) {
            return rewriter.notifyMatchFailure(srcOp, "mul expects i8, i16, bf16 inputs\n");
        }

        auto input2Type = dyn_cast<RankedTensorType>(input2.getType());
        auto input2ElementType = input2Type.getElementType();

        if (input1ElementType.isInteger(32) && input2ElementType.isInteger(32)) {
            return rewriter.notifyMatchFailure(srcOp, "mul with i32 inputs is not supported");
        }

        Operation *mulOp = getElementwiseBinaryOp(srcOp, /*allow constants*/ true);
        if (!mulOp) {
            return rewriter.notifyMatchFailure(srcOp, "Not an elementwise binary op");
        }

        if (!isa<arith::MulIOp>(mulOp) && !isa<arith::MulFOp>(mulOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Expected arith.muli or arith.mulf");
        }

        Value output = srcOp.getResult(0);
        const int outChannelCount = 1; // No channels here, only one single scale
        ScaleClampInfo scInfo = foldForwardScaleClamp(output, outChannelCount, 12, 12, false);

        if (_markFuseGroups) {
            if (!markFuseGroupBackward(
                    output, {input1, input2}, rewriter,
                    srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        auto lhs = mulOp->getOperand(0);
        auto rhs = mulOp->getOperand(1);

        // A constant operand may be inlined in the body, an outer constant
        // referenced from it, a single-element constant tensor input, or a
        // width cast (arith.truncf/extf) of any of those -- the importer
        // decomposes f32 scalars over bf16 data that way. The traced value
        // already carries the cast the body applies, so the 1-element tensor
        // built here is exactly what the old direct-constant path built.
        auto constValue = [&](Value v) -> Value {
            if (auto f = traceToConstFloat(srcOp, v, SplatPolicy::SingleElement))
                // The traced value carries the body's element semantics, which
                // is the input element type for a type-correct mulf/muli.
                return createConst(std::vector<APFloat>{*f}, rewriter, srcOp.getLoc());
            if (auto i = traceToConstInt(srcOp, v, SplatPolicy::SingleElement)) {
                // DenseIntElementsAttr requires values of the element's bit
                // width; traceToConstInt sign-extends to i64.
                auto intTy = dyn_cast<IntegerType>(input1ElementType);
                if (!intTy)
                    return {};
                RankedTensorType constType = RankedTensorType::get({1}, input1ElementType);
                Attribute value;
                if (intTy.getWidth() == 8)
                    value = DenseIntElementsAttr::get(constType, static_cast<int8_t>(*i));
                else if (intTy.getWidth() == 16)
                    value = DenseIntElementsAttr::get(constType, static_cast<int16_t>(*i));
                else if (intTy.getWidth() == 32)
                    value = DenseIntElementsAttr::get(constType, static_cast<int32_t>(*i));
                else
                    value = DenseIntElementsAttr::get(constType, *i);
                return arith::ConstantOp::create(rewriter, srcOp.getLoc(), cast<TypedAttr>(value))
                    .getResult();
            }
            return {};
        };

        Value lhsCst = constValue(lhs);
        if (lhsCst) {
            input2 = input1;
            input1 = lhsCst;
        }
        else if (Value rhsCst = constValue(rhs)) {
            input2 = rhsCst;
        }

        // Float tail fold: for bf16 the ACT computes data * weight + floatBias
        // clamped to [lo, hi], so a trailing scalar add and/or clamp(s) are
        // absorbed into the torq_hl.mul (f32 bias record in BRAM + f32 clip
        // bit patterns) and the consumed generics are dropped. input1/input2
        // carry the extracted constant weight, if any.
        if (scInfo.hasFloatTail()) {
            auto tailOutType = cast<RankedTensorType>(output.getType());
            if (tailOutType.getElementType().isBF16()) {
                std::vector<float> biasF32{scInfo.floatBias.value_or(0.0f)};
                Value biasCst = createConst(biasF32, rewriter, srcOp.getLoc());
                Value torqOut = torq_hl::MulOp::create(
                                    rewriter, srcOp.getLoc(), tailOutType,
                                    createInitTensor(srcOp, rewriter, tailOutType),
                                    /*output_zp=*/0, scInfo.min, scInfo.max, biasCst, /*shift=*/0,
                                    input1, input2
                )
                                    .getResult(0);
                rewriter.replaceOp(output.getDefiningOp(), torqOut);
                // Mark every consumed tail op: the driver must not reprocess it
                // as a root (a second rewrite of the same op would double-commit).
                for (auto *op : llvm::reverse(scInfo.floatTailOps))
                    if (op != output.getDefiningOp())
                        rewriter.eraseOp(op);
                rewriter.eraseOp(srcOp);
                return success();
            }
        }

        int32_t bias = 0;
        int32_t scale = 1;
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

        std::vector<int32_t> a{bias};
        std::vector<int32_t> b{scale};
        Value torqOut =
            torq_hl::MulOp::create(
                rewriter, srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                outZp, outMin, outMax, createI32Const(rewriter, srcOp, interleave(a, b)),
                shiftFactor, input1, input2
            )
                .getResult(0);

        if (scInfo) {
            rewriter.replaceOp(output.getDefiningOp(), torqOut);
        }
        else {
            rewriter.replaceOp(srcOp, torqOut);
        }

        return success();
    }
};

/// ============================================================================
/// MulPerChannelBiasPattern
/// ============================================================================
///
/// Folds a trailing per-output-channel float add into the multiply:
///
///   %scaled = linalg.generic mulf(%a, %b)                    -> [.., N]
///   %out    = linalg.generic addf(%scaled, bcast(%bias[N]))  -> [.., N]
///
/// becomes one `torq_hl.mul` carrying %bias in `scale_bias`.
/// BRAM holds the bias as fp32, so a bf16->f32 conversion pass is still needed.
///
/// Restricted to a mul and add that already share a fuse group, i.e. that were
/// going to be lowered into one program anyway.
class MulPerChannelBiasPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    MulPerChannelBiasPattern(MLIRContext *context) : OpRewritePattern(context, /*benefit=*/2) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        Operation *mulOp = getElementwiseBinaryOp(srcOp, /*allowConstants=*/false);
        if (!mulOp || !isa<arith::MulFOp>(mulOp))
            return rewriter.notifyMatchFailure(srcOp, "not an elementwise mulf");

        Value scaled = srcOp.getResult(0);
        auto outType = dyn_cast<RankedTensorType>(scaled.getType());
        if (!outType || outType.getRank() < 2 || !scaled.hasOneUse())
            return rewriter.notifyMatchFailure(srcOp, "not a rank>=2 mulf with a single consumer");

        // Same operand restriction as MulOpPattern
        for (Value input : srcOp.getInputs()) {
            Type elementType = cast<RankedTensorType>(input.getType()).getElementType();
            if (elementType.isF32() || elementType.isF64())
                return rewriter.notifyMatchFailure(srcOp, "mul expects i8, i16, bf16 inputs");
        }

        auto addOp = dyn_cast<linalg::GenericOp>(*scaled.getUsers().begin());
        if (!addOp || addOp.getNumDpsInputs() != 2 || addOp.getNumDpsInits() != 1 ||
            addOp.getResult(0).getType() != outType)
            return rewriter.notifyMatchFailure(srcOp, "consumer is not a same-shape binary op");
        Operation *addFOp = getElementwiseBinaryOp(addOp, /*allowConstants=*/false);
        if (!addFOp || !isa<arith::AddFOp>(addFOp))
            return rewriter.notifyMatchFailure(srcOp, "consumer is not an addf");
        if (!checkShareFuseGroup(srcOp, addOp))
            return rewriter.notifyMatchFailure(srcOp, "mulf and addf are in different fuse groups");

        Value biasOperand =
            addOp.getInputs()[0] == scaled ? addOp.getInputs()[1] : addOp.getInputs()[0];
        Value bias = matchPerChannelFloatBias(biasOperand, outType.getShape());
        if (!bias)
            return rewriter.notifyMatchFailure(srcOp, "addf rhs is not a per-channel vector");

        Value output = addOp.getResult(0);
        ScaleClampInfo scInfo = foldForwardScaleClamp(output, 1, 12, 12);

        auto biasCst = bias.getDefiningOp<arith::ConstantOp>();
        auto dense = dyn_cast_or_null<DenseFPElementsAttr>(biasCst ? biasCst.getValue() : nullptr);

        // A scalar add folded from the tail can only be absorbed into the
        // per-channel bias when that bias is a compile-time constant with an
        // inlined (Dense) value: a dense_resource blob (how ONNX initializers
        // usually arrive) cannot be rebuilt, and absorbing would silently drop
        // the +k. Keep the add (and anything past it) alive otherwise:
        // AddOpPattern picks the chain up from there, clamp and all.
        if (scInfo.floatBias && !dense) {
            scInfo.floatTailOps.clear();
            scInfo.floatBias = std::nullopt;
            float fullLo = std::numeric_limits<float>::lowest();
            float fullHi = std::numeric_limits<float>::max();
            scInfo.min = llvm::bit_cast<int32_t>(fullLo);
            scInfo.max = llvm::bit_cast<int32_t>(fullHi);
            output = addOp.getResult(0);
        }
        if (scInfo.floatBias) {
            // Absorb the tail's scalar add into the per-channel bias constant,
            // rebuilt in f32 (the BRAM bias type) to avoid an extra rounding.
            auto biasTy = cast<RankedTensorType>(bias.getType());
            auto f32Ty = RankedTensorType::get(biasTy.getShape(), rewriter.getF32Type());
            SmallVector<APFloat> elems;
            for (APFloat f : dense.getValues<APFloat>()) {
                APFloat f32(f.convertToFloat());
                f32.add(APFloat(*scInfo.floatBias), APFloat::rmNearestTiesToEven);
                elems.push_back(f32);
            }
            bias = arith::ConstantOp::create(
                       rewriter, srcOp.getLoc(), DenseFPElementsAttr::get(f32Ty, elems)
            )
                       .getResult();
        }
        SmallVector<Operation *> foldedOps;
        for (Value current = addOp.getResult(0); current != output;) {
            Operation *next = getSingleUser(current);
            if (!next)
                return rewriter.notifyMatchFailure(
                    srcOp, "float epilogue is not a single-use chain"
                );
            foldedOps.push_back(next);
            current = next->getResult(0);
        }

        // Build at the add, not at the multiply: the bias is computed from the
        // multiply's own scale operand, so it is defined after the multiply and
        // would not dominate a use placed there.
        rewriter.setInsertionPoint(addOp);

        // BRam::load only takes fp32 (or an int bias:scale pair), so convert.
        Location loc = srcOp.getLoc();
        // Drop whatever leading unit dimensions the rank-align left on the bias, so the
        // conversion and the BRAM operand below both see the [N] vector they expect.
        auto biasType = cast<RankedTensorType>(bias.getType());
        if (biasType.getRank() > 1) {
            ReassociationIndices allDims;
            for (int64_t i = 0; i < biasType.getRank(); ++i)
                allDims.push_back(i);
            bias =
                tensor::CollapseShapeOp::create(
                    rewriter, loc,
                    RankedTensorType::get({outType.getShape().back()}, biasType.getElementType()),
                    bias, ArrayRef<ReassociationIndices>{allDims}
                )
                    .getResult();
        }
        if (!cast<RankedTensorType>(bias.getType()).getElementType().isF32()) {
            auto f32Type =
                RankedTensorType::get({outType.getShape().back()}, rewriter.getF32Type());
            bias = torq_hl::ActOp::create(
                       rewriter, loc, f32Type, createInitTensor(srcOp, rewriter, f32Type), "f2f",
                       /*input_zp=*/0, /*output_zp=*/0, /*min_int=*/0, /*max_int=*/0, APFloat(0.0f),
                       APFloat(0.0f), bias, /*weights=*/Value()
            )
                       .getResult(0);
        }

        auto [outMin, outMax] = getDTypeRange(outType.getElementType());
        if (!foldedOps.empty() || scInfo.hasFloatTail()) {
            outMin = scInfo.min;
            outMax = scInfo.max;
        }
        Value torqOut = torq_hl::MulOp::create(
                            rewriter, loc, outType, createInitTensor(srcOp, rewriter, outType),
                            /*output_zp=*/0, outMin, outMax, bias, /*shift=*/0,
                            srcOp.getInputs().front(), srcOp.getInputs().back()
        )
                            .getResult(0);
        Operation *outputOp = output.getDefiningOp();
        rewriter.replaceOp(outputOp, torqOut);
        // Erase the consumed chain ops newest-first: the rewriter commits
        // deferred (so at pattern time use_empty() cannot see consumers this
        // pattern just consumed), and each op's single user is the next chain
        // op, which unlinks before it is destroyed. The tail op itself was
        // just replaced. When the tail advanced past the per-channel add, the
        // add is consumed too: it is the multiply's only remaining user and
        // would otherwise dangle on the erased multiply.
        for (Operation *foldedOp : llvm::reverse(foldedOps))
            if (foldedOp != outputOp)
                rewriter.eraseOp(foldedOp);
        if (outputOp != addOp)
            rewriter.eraseOp(addOp);
        rewriter.eraseOp(srcOp);
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
        std::vector<int32_t> a{bias};
        std::vector<int32_t> b{scale};
        Value torqOut =
            torq_hl::MulOp::create(
                rewriter, srcOp.getLoc(), outType, createInitTensor(srcOp, rewriter, outType),
                outZp, outMin, outMax, createI32Const(rewriter, srcOp, interleave(a, b)),
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
    patterns.insert<MulOpPattern>(context, markFuseGroups);
    if (markFuseGroups)
        return;

    // Fuse-group marking is driven by whoever owns the chain, so this only folds
    // during the material conversion.
    patterns.insert<MulPerChannelBiasPattern>(context);

    // FIXME: this pattern should be refactor to take markFuseGroups and have a
    // marking mode.
    patterns.insert<MulRescaleOpPattern>(context);
}

} // namespace mlir::syna::torq
