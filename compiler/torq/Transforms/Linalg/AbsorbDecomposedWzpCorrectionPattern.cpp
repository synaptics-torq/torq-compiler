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
// dead.
//
// With a nonzero input_zp the chain also has subi(acc, izp*Σw) and
// addi(acc, izp*wzp*K) (Term C).  Since Σw' = Σw - wzp*K, the weight sum is
// moved onto w' and Term C is dropped.  The weight sum then reads the conv's
// own filter, which later layout passes rely on.  If that chain is not in the
// exact form IREE emits, the sum keeps the raw weights and Term C stays.

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
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

// The input-zero-point part of the decomposed correction:
//   izpCorrection = subi(acc, muli(Σw, izp))   (feeds termB)
//   termC         = addi(termB, izp*wzp*K)
// where Σw is a reduction of `weights` to one value per output channel.
struct InputZpCorrection {
    linalg::GenericOp weightSum;
    linalg::GenericOp termC;
};

// Returns the yielded value of a single-result generic.
static Value getYieldedValue(linalg::GenericOp generic) {
    auto yieldOp = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
    return yieldOp.getNumOperands() == 1 ? yieldOp.getOperand(0) : Value();
}

static std::optional<InputZpCorrection>
matchInputZpCorrection(linalg::GenericOp termB, Value weights, int64_t wzp) {
    auto izpCorrection = termB.getDpsInputOperand(0)->get().getDefiningOp<linalg::GenericOp>();
    if (!izpCorrection || izpCorrection.getNumDpsInputs() != 2 ||
        !izpCorrection->getResult(0).hasOneUse())
        return std::nullopt;

    // yield(subi(acc, muli(Σw, izp)))
    Block *body = izpCorrection.getBody();
    auto subi = dyn_cast_or_null<arith::SubIOp>(getYieldedValue(izpCorrection).getDefiningOp());
    if (!subi || subi.getLhs() != body->getArgument(0))
        return std::nullopt;
    auto muli = subi.getRhs().getDefiningOp<arith::MulIOp>();
    if (!muli || muli.getLhs() != body->getArgument(1))
        return std::nullopt;
    auto izp = getIntConst(muli.getRhs());
    if (!izp)
        return std::nullopt;

    // Σw: a single-use reduction of the conv weights, accumulating sext(w).
    auto weightSum = izpCorrection.getDpsInputOperand(1)->get().getDefiningOp<linalg::GenericOp>();
    if (!weightSum || !weightSum->getResult(0).hasOneUse() || weightSum.getNumDpsInputs() != 1 ||
        weightSum.getDpsInputOperand(0)->get() != weights || !weightSum.getNumReductionLoops())
        return std::nullopt;
    Value weightArg = weightSum.getBody()->getArgument(0);
    if (!weightArg.hasOneUse() || !isa<arith::ExtSIOp>(*weightArg.getUsers().begin()))
        return std::nullopt;
    auto weightsTy = cast<RankedTensorType>(weights.getType());
    auto sumTy = dyn_cast<RankedTensorType>(weightSum->getResult(0).getType());
    if (!sumTy || !weightsTy.hasStaticShape() || !sumTy.hasStaticShape() ||
        weightsTy.getNumElements() % sumTy.getNumElements())
        return std::nullopt;
    int64_t kernelSize = weightsTy.getNumElements() / sumTy.getNumElements();

    // termC: yield(addi(acc, izp*wzp*K))
    auto termC = getSingleUser<linalg::GenericOp>(termB->getResult(0));
    if (!termC || termC.getNumDpsInputs() != 1 ||
        termC.getDpsInputOperand(0)->get() != termB->getResult(0))
        return std::nullopt;
    auto addi = dyn_cast_or_null<arith::AddIOp>(getYieldedValue(termC).getDefiningOp());
    if (!addi || addi.getLhs() != termC.getBody()->getArgument(0))
        return std::nullopt;
    auto crossTerm = getIntConst(addi.getRhs());
    if (!crossTerm || *crossTerm != *izp * wzp * kernelSize)
        return std::nullopt;

    return InputZpCorrection{weightSum, termC};
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
        auto collapseOp = getSingleUser<tensor::CollapseShapeOp>(poolingOp->getResult(0));
        if (!collapseOp)
            return rewriter.notifyMatchFailure(
                op, "expected single-use collapse_shape after pooling"
            );

        auto termB = getSingleUser<linalg::GenericOp>(collapseOp.getResult());
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

        auto izpCorrection = matchInputZpCorrection(termB, origWeights, *wzp);

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
        auto adjustedConst =
            createCompileTimeConstOp(adjustedOp, rewriter).value_or(adjustedOp.getResult(0));

        // Swap the weight operand and fix the block arg type.
        rewriter.modifyOpInPlace(convAnchor, [&]() {
            convAnchor->setOperand(1, adjustedConst);
            if (!convAnchor->getRegions().empty()) {
                Block &body = convAnchor->getRegion(0).front();
                if (body.getNumArguments() > 1)
                    body.getArgument(1).setType(i16Ty);
            }
        });

        // Sum the adjusted weights, which already include -wzp*K per channel,
        // so Term C is no longer needed.
        if (izpCorrection) {
            linalg::GenericOp weightSum = izpCorrection->weightSum;
            rewriter.modifyOpInPlace(weightSum, [&]() {
                weightSum.getDpsInputOperand(0)->set(adjustedConst);
                weightSum.getBody()->getArgument(0).setType(i16Ty);
            });
            rewriter.replaceOp(izpCorrection->termC, termB->getResult(0));
        }

        // Drop termB (the runtime Σinput*wzp correction is no longer needed).
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
