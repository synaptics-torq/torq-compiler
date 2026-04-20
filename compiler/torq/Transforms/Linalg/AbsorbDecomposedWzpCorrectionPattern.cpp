// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// After a new IREE version, tosa.conv2d with weight_zp != 0 emits a
// decomposed WZP-correction chain containing linalg.pooling_nhwc_sum and
// tensor.collapse_shape inside the fuse-group.  createClonedBlock() later
// hoists these ops before the conv anchor, producing a dominance violation.
//
// This pattern fixes the issue by folding weight_zp into the weight tensor at
// compile time (w' = w - wzp), which makes the runtime Σ(input)*wzp chain
// dead.  The addi(acc, izp*wzp*K) term (Term C) is intentionally kept: the
// weight-sum reduction still accumulates raw weights, so the constant offset
// must remain to produce the correct bias.

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-absorb-decomposed-wzp-correction"

namespace mlir::syna::torq {

namespace {

static std::optional<int64_t> getIntConst(Value v) {
    auto cst = v ? v.getDefiningOp<arith::ConstantOp>() : nullptr;
    if (!cst)
        return std::nullopt;
    if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return ia.getInt();
    return std::nullopt;
}

// Matches the pooling_nhwc_sum → collapse_shape → subi(acc, Σinput*wzp) chain
// produced by new IREE for quantized conv with weight_zp != 0, and replaces it
// by pre-subtracting wzp from the weights at compile time.
struct AbsorbWzpCorrectionPattern : public RewritePattern {
    AbsorbWzpCorrectionPattern(MLIRContext *ctx)
        : RewritePattern(linalg::PoolingNhwcSumOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        auto poolingOp = cast<linalg::PoolingNhwcSumOp>(op);

        // Match: pooling → collapse_shape → termB (subi(acc, Σinput*wzp))
        if (!poolingOp->hasOneUse())
            return rewriter.notifyMatchFailure(op, "pooling has multiple uses");

        auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(*poolingOp->getUsers().begin());
        if (!collapseOp || !collapseOp->hasOneUse())
            return rewriter.notifyMatchFailure(op, "expected single-use collapse_shape");

        auto termB = dyn_cast<linalg::GenericOp>(*collapseOp->getUsers().begin());
        if (!termB || termB.getNumDpsInputs() != 2)
            return rewriter.notifyMatchFailure(op, "unexpected termB shape");

        // Extract wzp from: yield(subi(acc, muli(Σinput, wzp_const)))
        auto yieldOp = cast<linalg::YieldOp>(termB.getBody()->getTerminator());
        auto subi = yieldOp.getValues()[0].getDefiningOp<arith::SubIOp>();
        auto muli = subi ? subi.getRhs().getDefiningOp<arith::MulIOp>() : nullptr;
        if (!muli)
            return rewriter.notifyMatchFailure(op, "termB body not subi(acc, muli(...))");

        auto wzp = getIntConst(muli.getRhs());
        if (!wzp)
            return rewriter.notifyMatchFailure(op, "wzp not a constant");

        LLVM_DEBUG(llvm::dbgs() << "[AbsorbWzpCorrection] wzp=" << *wzp << "\n");

        // Walk back through intervening generics to find the conv/matmul.
        Operation *convAnchor = nullptr;
        Value val = termB.getDpsInputOperand(0)->get();
        for (int i = 0; i < 20 && val; ++i) {
            Operation *def = val.getDefiningOp();
            if (!def)
                break;
            if (isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp,
                    linalg::DepthwiseConv2DNhwcHwcOp, linalg::DepthwiseConv2DNchwChwOp,
                    linalg::MatmulOp>(def)) {
                convAnchor = def;
                break;
            }
            auto g = dyn_cast<linalg::GenericOp>(def);
            if (!g || g.getNumDpsInputs() < 1)
                break;
            val = g.getDpsInputOperand(0)->get();
        }
        if (!convAnchor)
            return rewriter.notifyMatchFailure(op, "conv/matmul anchor not found");

        if (convAnchor->getNumOperands() < 2)
            return rewriter.notifyMatchFailure(op, "anchor has too few operands");

        Value origWeights = convAnchor->getOperand(1);
        auto wTy = dyn_cast<RankedTensorType>(origWeights.getType());
        if (!wTy)
            return rewriter.notifyMatchFailure(op, "weights not a ranked tensor");

        // Build w' = sext_i16(w) - wzp ahead of the conv anchor.
        auto loc = convAnchor->getLoc();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(convAnchor);

        auto i16Ty = rewriter.getIntegerType(16);
        auto idMap = rewriter.getMultiDimIdentityMap(wTy.getRank());
        SmallVector<utils::IteratorType> iters(wTy.getRank(), utils::IteratorType::parallel);
        auto wzpConst =
            arith::ConstantOp::create(
                rewriter, loc, rewriter.getIntegerAttr(i16Ty, static_cast<int16_t>(*wzp))
            )
                .getResult();
        auto empty = tensor::EmptyOp::create(rewriter, loc, wTy.getShape(), i16Ty).getResult();
        auto adjustedOp = linalg::GenericOp::create(
            rewriter, loc, RankedTensorType::get(wTy.getShape(), i16Ty), ValueRange{origWeights},
            ValueRange{empty}, SmallVector<AffineMap>{idMap, idMap}, iters,
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value ext = arith::ExtSIOp::create(b, l, i16Ty, args[0]);
                linalg::YieldOp::create(
                    b, l, arith::SubIOp::create(b, l, ext, wzpConst).getResult()
                );
            }
        );
        setCompileTimeConstAttr(adjustedOp);

        // Swap the weight operand and fix the block arg type.
        rewriter.modifyOpInPlace(convAnchor, [&]() {
            convAnchor->setOperand(1, adjustedOp.getResult(0));
            if (!convAnchor->getRegions().empty()) {
                Block &body = convAnchor->getRegion(0).front();
                if (body.getNumArguments() > 1)
                    body.getArgument(1).setType(i16Ty);
            }
        });

        // Drop termB (the runtime Σinput*wzp correction is no longer needed).
        // Keep Term C (addi(acc, izp*wzp*K)) alive: the weight-sum reduction
        // still accumulates raw weights, so the constant bias offset must
        // remain to compensate for the wzp*K difference.
        rewriter.replaceOp(termB, termB.getDpsInputOperand(0)->get());

        if (collapseOp->use_empty())
            rewriter.eraseOp(collapseOp);
        if (poolingOp->use_empty())
            rewriter.eraseOp(poolingOp);

        return success();
    }
};

} // namespace

void populateAbsorbDecomposedWzpCorrectionPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<AbsorbWzpCorrectionPattern>(ctx);
}

} // namespace mlir::syna::torq
