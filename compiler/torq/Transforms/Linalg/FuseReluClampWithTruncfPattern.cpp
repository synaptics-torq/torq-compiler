// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"
#include "torq/Utils/ShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
static linalg::GenericOp matchTruncfF32ToBF16Generic(Value v) {
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

/// Match a linalg.generic doing a relu6 clamp in bf16:
///   ins(%in : tensor<...xbf16>) outs(%out : tensor<...xbf16>)
///   ^bb0(%arg0: bf16, %arg1: bf16):
///     %c0  = arith.cmpf ult, %arg0, %low  : bf16   // %low  == arith.constant 0.0
///     %s0  = arith.select %c0, %low, %arg0 : bf16
///     %c1  = arith.cmpf ugt, %s0,   %high : bf16   // %high == arith.constant 6.0
///     %s1  = arith.select %c1, %high, %s0  : bf16
///     linalg.yield %s1 : bf16
/// On success, writes lowConst and highConst.
static bool matchBF16ReluClampGeneric(
    linalg::GenericOp op, arith::ConstantOp &lowConst, arith::ConstantOp &highConst
) {
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

    // Match outer select: select(cmpf(ugt, prev, high), high, prev)
    auto outerSel = yieldOp.getOperand(0).getDefiningOp<arith::SelectOp>();
    if (!outerSel)
        return false;

    auto outerCmp = outerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!outerCmp || outerCmp.getPredicate() != arith::CmpFPredicate::UGT)
        return false;

    // The true-value and the cmpf rhs must be the same high constant.
    if (outerSel.getTrueValue() != outerCmp.getRhs())
        return false;
    auto highConstOp = outerCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!highConstOp)
        return false;

    // The false-value of the outer select is the inner select result;
    // the cmpf lhs must be this same value.
    Value innerSelVal = outerSel.getFalseValue();
    if (outerCmp.getLhs() != innerSelVal)
        return false;

    auto innerSel = innerSelVal.getDefiningOp<arith::SelectOp>();
    if (!innerSel)
        return false;

    // Match inner select: select(cmpf(ult, in, low), low, in)
    auto innerCmp = innerSel.getCondition().getDefiningOp<arith::CmpFOp>();
    if (!innerCmp || innerCmp.getPredicate() != arith::CmpFPredicate::ULT)
        return false;

    if (innerSel.getTrueValue() != innerCmp.getRhs())
        return false;
    auto lowConstOp = innerCmp.getRhs().getDefiningOp<arith::ConstantOp>();
    if (!lowConstOp)
        return false;

    // The false-value and cmpf lhs must both be block argument 0.
    if (innerSel.getFalseValue() != innerCmp.getLhs())
        return false;
    auto blockArg = dyn_cast<BlockArgument>(innerSel.getFalseValue());
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != body)
        return false;

    lowConst = lowConstOp;
    highConst = highConstOp;
    return true;
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
        // 1. Confirm this op is a bf16 relu6 clamp.
        arith::ConstantOp lowConst, highConst;
        if (!matchBF16ReluClampGeneric(clampOp, lowConst, highConst))
            return rewriter.notifyMatchFailure(clampOp, "not a bf16 relu6 clamp generic");

        // 2. Confirm the input comes from a truncf f32->bf16 generic.
        linalg::GenericOp truncfOp = matchTruncfF32ToBF16Generic(clampOp.getInputs()[0]);
        if (!truncfOp)
            return rewriter.notifyMatchFailure(
                clampOp, "clamp input is not from a truncf f32->bf16 generic"
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
        double lowVal = cast<FloatAttr>(lowConst.getValue()).getValueAsDouble();
        double highVal = cast<FloatAttr>(highConst.getValue()).getValueAsDouble();
        Value f32LowConst =
            arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(f32Type, lowVal));
        Value f32HighConst =
            arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(f32Type, highVal));

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
            [&](OpBuilder &b, Location innerLoc, ValueRange args) {
                Value in = args[0]; // f32 element
                Value cmp0 =
                    arith::CmpFOp::create(b, innerLoc, arith::CmpFPredicate::ULT, in, f32LowConst);
                Value sel0 = arith::SelectOp::create(b, innerLoc, cmp0, f32LowConst, in);
                Value cmp1 = arith::CmpFOp::create(
                    b, innerLoc, arith::CmpFPredicate::UGT, sel0, f32HighConst
                );
                Value sel1 = arith::SelectOp::create(b, innerLoc, cmp1, f32HighConst, sel0);
                Value truncated = arith::TruncFOp::create(b, innerLoc, bf16Type, sel1);
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
}

} // namespace mlir::syna::torq
