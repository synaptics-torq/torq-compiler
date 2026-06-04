// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-fuse-relu-clamp-with-truncf"

namespace mlir::syna::torq {

namespace {

/// Match a linalg.generic that truncates f32 -> bf16 element-wise:
///   ins(%in : tensor<...xf32>) outs(%out : tensor<...xbf16>)
///   ^bb0(%arg0: f32, %arg1: bf16):
///     %t = arith.truncf %arg0 : f32 to bf16
///     linalg.yield %t : bf16
/// Returns the generic op if matched, nullptr otherwise.
linalg::GenericOp matchTruncfF32ToBF16Generic(Value v) {
    auto op = v.getDefiningOp<linalg::GenericOp>();
    if (!op)
        return nullptr;
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 || op.getNumResults() != 1)
        return nullptr;
    if (op.getNumParallelLoops() != op.getNumLoops())
        return nullptr;
    if (!checkIdentityLikeMaps(op))
        return nullptr;

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!inputType || !outputType)
        return nullptr;
    if (!inputType.getElementType().isF32() || !outputType.getElementType().isBF16())
        return nullptr;

    auto *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return nullptr;

    auto truncfOp = yieldOp.getOperand(0).getDefiningOp<arith::TruncFOp>();
    if (!truncfOp)
        return nullptr;

    auto blockArg = dyn_cast<BlockArgument>(truncfOp.getIn());
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != body)
        return nullptr;

    return op;
}

/// Match a linalg.generic doing a simple relu (max(x, low)) in bf16:
///   ins(%in : tensor<...xbf16>) outs(%out : tensor<...xbf16>)
///   ^bb0(%arg0: bf16, %arg1: bf16):
///     %c = arith.cmpf ugt, %arg0, %low : bf16   // %low == arith.constant 0.0
///     %s = arith.select %c, %arg0, %low : bf16
///     linalg.yield %s : bf16
/// Also accepts the equivalent reversed form:
///     %c = arith.cmpf ult, %low, %arg0 : bf16
///     %s = arith.select %c, %arg0, %low : bf16
/// On success, writes lowConst.
static bool matchBF16SimpleReluGeneric(linalg::GenericOp op, arith::ConstantOp &lowConst) {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 || op.getNumResults() != 1)
        return false;
    if (op.getNumParallelLoops() != op.getNumLoops())
        return false;
    if (!checkIdentityLikeMaps(op))
        return false;

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!inputType || !outputType)
        return false;
    if (!inputType.getElementType().isBF16() || !outputType.getElementType().isBF16())
        return false;

    auto *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;

    // Match select(cmpf(pred, lhs, rhs), trueVal, falseVal)
    auto sel = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!sel)
        return false;

    auto cmp = sel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!cmp)
        return false;

    auto pred = cmp.getPredicate();
    auto cmpLhs = cmp.getLhs();
    auto cmpRhs = cmp.getRhs();
    auto selTrue = sel.getTrueValue();
    auto selFalse = sel.getFalseValue();

    // We expect max(x, c) where x is block arg 0.
    // Accept two canonical forms:
    //   select(cmpf(ugt, x, c), x, c)
    //   select(cmpf(ult, c, x), x, c)
    Value dataValue, constValue;
    if (pred == arith::CmpFPredicate::UGT) {
        // cmpf ugt, x, c  →  true when x > c
        // select true, x, c  →  x when x > c, else c  →  max(x, c)
        if (selTrue != cmpLhs || selFalse != cmpRhs)
            return false;
        dataValue = cmpLhs;
        constValue = cmpRhs;
    }
    else if (pred == arith::CmpFPredicate::ULT) {
        // cmpf ult, c, x  →  true when c < x
        // select true, x, c  →  x when c < x, else c  →  max(x, c)
        if (selTrue != cmpRhs || selFalse != cmpLhs)
            return false;
        dataValue = cmpRhs;
        constValue = cmpLhs;
    }
    else {
        return false;
    }

    auto blockArg = dyn_cast<BlockArgument>(dataValue);
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != body)
        return false;

    auto lowConstOp = constValue.getDefiningOp<arith::ConstantOp>();
    if (!lowConstOp)
        return false;
    if (!isa<FloatAttr>(lowConstOp.getValue()))
        return false;

    lowConst = lowConstOp;
    return true;
}

/// Match a linalg.generic doing a relu6 clamp in bf16:
///   ins(%in : tensor<...xbf16>) outs(%out : tensor<...xbf16>)
///   ^bb0(%arg0: bf16, %arg1: bf16):
///     %c0  = arith.cmpf ult, %arg0, %low  : bf16   // %low  == arith.constant 0.0
///     %s0  = arith.select %c0, %low, %arg0 : bf16
///     %c1  = arith.cmpf ugt, %s0,   %high : bf16   // %high == arith.constant 6.0
///     %s1  = arith.select %c1, %high, %s0  : bf16
///     linalg.yield %s1 : bf16
/// On success, writes lowVal and highVal.
LogicalResult matchBF16ReluClampGeneric(linalg::GenericOp op, double &lowVal, double &highVal) {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 || op.getNumResults() != 1)
        return failure();
    if (op.getNumParallelLoops() != op.getNumLoops())
        return failure();
    if (!checkIdentityLikeMaps(op))
        return failure();

    auto inputType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!inputType || !outputType)
        return failure();
    if (!inputType.getElementType().isBF16() || !outputType.getElementType().isBF16())
        return failure();

    auto *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return failure();

    // Match outer select: select(cmpf(ugt, prev, high), high, prev)
    auto outerSel = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!outerSel)
        return failure();

    auto outerCmp = outerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!outerCmp || outerCmp.getPredicate() != arith::CmpFPredicate::UGT)
        return failure();

    // The true-value and the cmpf rhs must be the same high constant.
    if (outerSel.getTrueValue() != outerCmp.getRhs())
        return failure();
    auto highConstOp = outerCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!highConstOp)
        return failure();

    // The false-value of the outer select is the inner select result;
    // the cmpf lhs must be this same value.
    Value innerSelVal = outerSel.getFalseValue();
    if (outerCmp.getLhs() != innerSelVal)
        return failure();

    auto innerSel = innerSelVal.getDefiningOp<arith::SelectOp>();
    if (!innerSel)
        return failure();

    // Match inner select: select(cmpf(ult, in, low), low, in)
    auto innerCmp = innerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!innerCmp || innerCmp.getPredicate() != arith::CmpFPredicate::ULT)
        return failure();

    if (innerSel.getTrueValue() != innerCmp.getRhs())
        return failure();
    auto lowConstOp = innerCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lowConstOp)
        return failure();
    if (!isa<FloatAttr>(lowConstOp.getValue()))
        return failure();
    if (!isa<FloatAttr>(highConstOp.getValue()))
        return failure();

    // The false-value and cmpf lhs must both be block argument 0.
    if (innerSel.getFalseValue() != innerCmp.getLhs())
        return failure();
    auto blockArg = dyn_cast<BlockArgument>(innerSel.getFalseValue());
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != body)
        return failure();

    lowVal = cast<FloatAttr>(lowConstOp.getValue()).getValueAsDouble();
    highVal = cast<FloatAttr>(highConstOp.getValue()).getValueAsDouble();
    return success();
}

/// Match a linalg.generic doing a relu6 clamp in bf16 where the bounds are
/// passed as 0-d tensor inputs (3-input form):
///   ins(%in : tensor<...xbf16>, %low : tensor<bf16>, %high : tensor<bf16>)
///   outs(%out : tensor<...xbf16>)
///   ^bb0(%arg0: bf16, %arg1: bf16, %arg2: bf16, %arg3: bf16):
///     %c0 = arith.cmpf ult, %arg0, %arg1 : bf16
///     %s0 = arith.select %c0, %arg1, %arg0 : bf16
///     %c1 = arith.cmpf ugt, %s0,   %arg2 : bf16
///     %s1 = arith.select %c1, %arg2, %s0  : bf16
///     linalg.yield %s1 : bf16
/// On success, writes lowVal and highVal (the constant double values from the
/// outer dense<...> tensor inputs).
LogicalResult
matchBF16ReluClampGeneric3Inputs(linalg::GenericOp op, double &lowVal, double &highVal) {
    if (op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1 || op.getNumResults() != 1)
        return failure();
    if (op.getNumParallelLoops() != op.getNumLoops())
        return failure();

    auto dataType = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto outType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!dataType || !outType)
        return failure();
    if (!dataType.getElementType().isBF16() || !outType.getElementType().isBF16())
        return failure();

    auto maps = op.getIndexingMapsArray();
    unsigned outIdx = op.getNumDpsInputs();
    if (!maps[0].isIdentity() || !maps[outIdx].isIdentity())
        return failure();

    if (!extractBF16ConstantValue(op.getInputs()[1], lowVal))
        return failure();
    if (!extractBF16ConstantValue(op.getInputs()[2], highVal))
        return failure();

    // Verify the body structure.
    auto *body = op.getBody();
    auto yieldOp = dyn_cast<linalg::YieldOp>(body->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return failure();

    // Outer select: select(cmpf(ugt, prev, high_arg), high_arg, prev)
    auto outerSel = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!outerSel)
        return failure();
    auto outerCmp = outerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!outerCmp || outerCmp.getPredicate() != arith::CmpFPredicate::UGT)
        return failure();
    if (outerSel.getTrueValue() != outerCmp.getRhs())
        return failure();
    auto highArg = dyn_cast<BlockArgument>(outerCmp.getRhs());
    if (!highArg || highArg.getArgNumber() != 2 || highArg.getOwner() != body)
        return failure();

    Value innerSelVal = outerSel.getFalseValue();
    if (outerCmp.getLhs() != innerSelVal)
        return failure();

    // Inner select: select(cmpf(ult, in, low_arg), low_arg, in)
    auto innerSel = innerSelVal.getDefiningOp<arith::SelectOp>();
    if (!innerSel)
        return failure();
    auto innerCmp = innerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!innerCmp || innerCmp.getPredicate() != arith::CmpFPredicate::ULT)
        return failure();
    if (innerSel.getTrueValue() != innerCmp.getRhs())
        return failure();
    auto lowArg = dyn_cast<BlockArgument>(innerCmp.getRhs());
    if (!lowArg || lowArg.getArgNumber() != 1 || lowArg.getOwner() != body)
        return failure();
    if (innerSel.getFalseValue() != innerCmp.getLhs())
        return failure();
    auto dataArg = dyn_cast<BlockArgument>(innerSel.getFalseValue());
    if (!dataArg || dataArg.getArgNumber() != 0 || dataArg.getOwner() != body)
        return failure();

    return success();
}

/// Pattern that fuses a bf16 relu6 clamp with the preceding truncf generic into
/// a single linalg.generic that clamps in f32 then truncates to bf16.
///
/// Matches (when truncf result has no other users):
///   %truncf     = linalg.generic(truncf f32->bf16, ins(%f32_in))
///   %clamp_bf16 = linalg.generic(relu6 bf16->bf16, ins(%truncf))
///
/// Rewrites to a single fused op:
///   %fused = linalg.generic(relu6 clamp f32, then truncf to bf16,
///                            ins(%f32_in), outs(%bf16_init))
///   ^bb0(%in: f32, %out: bf16):
///     // clamp in f32
///     %c0 = arith.cmpf ult, %in,  %low_f32  : f32
///     %s0 = arith.select %c0, %low_f32,  %in  : f32
///     %c1 = arith.cmpf ugt, %s0,  %high_f32 : f32
///     %s1 = arith.select %c1, %high_f32, %s0  : f32
///     // truncate to bf16
///     %t  = arith.truncf %s1 : f32 to bf16
///     linalg.yield %t : bf16
///
/// Both the clamp and truncf generics are replaced by this single op.
struct FuseReluClampWithTruncfPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp clampOp, PatternRewriter &rewriter) const override {
        // 1. Confirm this op is a bf16 relu6 clamp (1-input or 3-input form)
        //    or a simple bf16 relu.
        arith::ConstantOp lowConst;
        double lowVal = 0.0, highVal = 0.0;
        bool isRelu6 = false;
        if (succeeded(matchBF16ReluClampGeneric(clampOp, lowVal, highVal))) {
            // 1-input relu6 form: bounds are inline arith.constant ops.
            isRelu6 = true;
        }
        else if (succeeded(matchBF16ReluClampGeneric3Inputs(clampOp, lowVal, highVal))) {
            // 3-input relu6 form: bounds come from tensor operands.
            isRelu6 = true;
        }
        else if (!matchBF16SimpleReluGeneric(clampOp, lowConst)) {
            return rewriter.notifyMatchFailure(
                clampOp, "not a bf16 relu6 clamp or simple relu generic"
            );
        }
        else {
            // Simple relu: only lowVal is needed.
            lowVal = cast<FloatAttr>(lowConst.getValue()).getValueAsDouble();
        }

        // 2. Confirm the data input (inputs()[0]) comes directly from a
        //    truncf f32->bf16 generic — no intervening ops.
        linalg::GenericOp truncfOp = matchTruncfF32ToBF16Generic(clampOp.getInputs()[0]);
        if (!truncfOp)
            return rewriter.notifyMatchFailure(
                clampOp, "clamp input is not directly from a truncf f32->bf16 generic"
            );

        // 3. The truncf result must not be used by anything other than this clamp op.
        if (!truncfOp.getResult(0).hasOneUse())
            return rewriter.notifyMatchFailure(
                clampOp, "truncf result has multiple uses; cannot fuse"
            );

        Location loc = clampOp.getLoc();
        Type f32Type = rewriter.getF32Type();
        Type bf16Type = rewriter.getBF16Type();

        // 4. Build f32 clamp bound constants (inserted just before the fused op).
        Value f32LowConst =
            arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(f32Type, lowVal));
        Value f32HighConst;
        if (isRelu6) {
            f32HighConst =
                arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(f32Type, highVal));
        }

        // 5. Build fused indexing maps:
        //    - input maps from the truncf op (f32 operands)
        //    - output maps from the clamp op (bf16 result)
        SmallVector<AffineMap> fusedMaps;
        auto truncfMaps = truncfOp.getIndexingMapsArray();
        for (unsigned i = 0; i < truncfOp.getNumDpsInputs(); ++i)
            fusedMaps.push_back(truncfMaps[i]);
        auto clampMaps = clampOp.getIndexingMapsArray();
        unsigned clampOutOffset = clampOp.getNumDpsInputs();
        for (unsigned i = 0; i < clampOp.getNumDpsInits(); ++i)
            fusedMaps.push_back(clampMaps[clampOutOffset + i]);

        // 6. Create the fused linalg.generic:
        //    ins  = truncf's f32 inputs
        //    outs = clamp's bf16 output inits
        //    result type = clamp's bf16 result type
        //    body = clamp in f32, then truncf to bf16
        //
        // Note: f32LowConst / f32HighConst are outer-scope SSA values captured
        // by the body region (linalg.generic is not isolated from above).
        auto fusedOp = linalg::GenericOp::create(
            rewriter, loc, clampOp.getResultTypes(), truncfOp.getInputs(), clampOp.getOutputs(),
            fusedMaps, clampOp.getIteratorTypesArray(),
            [isRelu6, f32LowConst, f32HighConst,
             bf16Type](OpBuilder &b, Location innerLoc, ValueRange args) {
                Value in = args[0]; // f32 element
                Value clamped;
                if (isRelu6) {
                    Value cmp0 = arith::CmpFOp::create(
                        b, innerLoc, arith::CmpFPredicate::ULT, in, f32LowConst
                    );
                    Value sel0 = arith::SelectOp::create(b, innerLoc, cmp0, f32LowConst, in);
                    Value cmp1 = arith::CmpFOp::create(
                        b, innerLoc, arith::CmpFPredicate::UGT, sel0, f32HighConst
                    );
                    clamped = arith::SelectOp::create(b, innerLoc, cmp1, f32HighConst, sel0);
                }
                else {
                    // Simple ReLU: max(x, low)
                    // The original bf16 pattern was:
                    //   cmpf ugt, x, low  →  true when x > low
                    //   select true, x, low  →  x when x > low, else low
                    // In f32 this is the same structure.
                    Value cmp0 = arith::CmpFOp::create(
                        b, innerLoc, arith::CmpFPredicate::UGT, in, f32LowConst
                    );
                    clamped = arith::SelectOp::create(b, innerLoc, cmp0, in, f32LowConst);
                }
                Value truncated = arith::TruncFOp::create(b, innerLoc, bf16Type, clamped);
                linalg::YieldOp::create(b, innerLoc, truncated);
            }
        );

        // 7. Replace the clamp result and erase both old ops.
        rewriter.replaceOp(clampOp, fusedOp.getResults());
        rewriter.eraseOp(truncfOp);

        return success();
    }
};

} // namespace

void populateFuseReluClampWithTruncfPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<FuseReluClampWithTruncfPattern>(ctx);
    linalg::populateFoldReshapeOpsByExpansionPatterns(patterns, [](OpOperand *fusedOperand) {
        if (!fusedOperand)
            return false;

        auto genericOp = dyn_cast<linalg::GenericOp>(fusedOperand->getOwner());
        if (!genericOp)
            return false;

        if (!fusedOperand->get().getDefiningOp<tensor::CollapseShapeOp>())
            return false;

        double lowVal = 0.0, highVal = 0.0;
        return succeeded(matchBF16ReluClampGeneric3Inputs(genericOp, lowVal, highVal));
    });
    // collpase
    // linalg.generic(%collapse)
}

} // namespace mlir::syna::torq
