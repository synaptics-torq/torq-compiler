// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RaiseMatMulInteger
// ------------------
// torch-mlir lowers onnx.MatMulInteger sub-optmially: it
// zero-extends the (unsigned) activation and sign-extends the (signed)
// weight to i32, subtracts their zero-points in the i32 domain, and runs a
// plain i32 x i32 -> i32 linalg.matmul. The dequant (sitofp * runtime scale)
// stays as separate ops. This widens both operands to i32 for the whole
// matmul, which is unnecessary DRAM/LRAM traffic for the weight (it could
// stay int8) and precludes the native u8 x i8 -> i32 MAC.
//
// This raise runs before HW lowering and rewrites the chain to keep the activation u8 and
// the weight i8, correcting every zero point in i32 before the one rounding to bf16:
//
//   raw           = matmul(A_u8, W_i8) -> i32                     // native u8 x i8 MAC
//   corrected     = raw + a_zp*neg_colsum[n] - delta*rowsum_A[m]  // exact, i32
//   out           = sitofp(corrected) * scale                     // one rounding
//   neg_colsum[n] = -sum_k (W - w_zp)[k,n]                        // compile-time [N] i32
//
// `scale` is the front end's runtime a_scale*b_scale, reused as-is. `a_zp` is its integer
// zero point, exactly the one the u8 MAC embedded, so the column term cancels exactly.
//
// The order matters: `raw` is DC-inflated by `a_zp*colsum_W[n]`, several times the signal it
// carries, so rounding it first lands ~0.4% of that DC on the corrected result. No bf16 value
// may appear on either side of a subtract that removes a zero-point term.
//
// The weight zero point is folded into the weight constant itself: the bytes stored are
// `W - center`, and the residual `delta = w_zp - center` rides on a rowsum over A. `center`
// is `w_zp` clamped to [max(W) - 127, min(W) + 128]: the values for which `W - center` fits
// int8. That interval is never empty for an 8-bit weight, so there is no fail-closed path,
// and clamping into it rather than to the type's midpoint keeps `delta` as small as the
// weight allows.
//
// A q/k/v projection is often exported as one fused MatMulInteger over [K, 3N] whose
// i32 result is cut back into three branches with `tensor.extract_slice`, each with its
// own dequant. That is raised one branch at a time, against a compile-time slice of the
// weight -- so the fused matmul becomes three. On this hardware that is free at decode
// (identical weight bytes, and decode is memory-bound) and better at prefill: a fuse
// group has one producer, so a shared matmul would strand the other two branches
// reading the wide i32 intermediate back from DRAM.
//
// Limitation: the weight must be a compile-time constant, both for the column-sum
// fold and for that correction. A runtime weight would need `colsum_W` computed per
// inference, which is a second streaming pass over the whole weight -- affordable at
// prefill, a 2x regression on the decode path that dominates. Chains with a runtime
// weight keep the front end's naive i32 form.

#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include <algorithm>
#include <limits>

namespace mlir::syna::torq {

namespace {

// Look through the rank-only reshapes a single-element tensor picks up on its way
// between ops; whether one is there depends on the surrounding shapes.
static Value peelScalarReshape(Value v) {
    while (true) {
        if (auto expand = v.getDefiningOp<tensor::ExpandShapeOp>())
            v = expand.getSrc();
        else if (auto collapse = v.getDefiningOp<tensor::CollapseShapeOp>())
            v = collapse.getSrc();
        else
            return v;
    }
}

// Look through the explicit broadcast a standardization pass may have
// materialized around a runtime scalar, back to the 1-element tensor
// underneath.
static Value peelToScalar(Value v) {
    while (true) {
        auto ty = dyn_cast<RankedTensorType>(v.getType());
        if (!ty)
            return nullptr;
        if (ty.getNumElements() == 1) {
            // The reshapes a scalar picks up depend on the shapes around it: an
            // activation of rank 3 gets the zero point as <1x1xT>, one of rank 2 as
            // <1xT>. Peel them off, and take the smallest form the rewrite can consume.
            v = peelScalarReshape(v);
            auto peeledTy = cast<RankedTensorType>(v.getType());
            return peeledTy.getRank() <= 1 ? v : nullptr;
        }
        auto g = v.getDefiningOp<linalg::GenericOp>();
        if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 ||
            g.getNumParallelLoops() != g.getNumLoops() || !g.getRegion().hasOneBlock())
            return nullptr;
        auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
            return nullptr;
        // A pure broadcast: the body yields its input unchanged.
        auto arg = dyn_cast<BlockArgument>(yield.getOperand(0));
        if (!arg || arg.getOwner() != g.getBody() || arg.getArgNumber() != 0)
            return nullptr;
        v = g.getInputs()[0];
    }
}

// Peel the explicit broadcast a standardization pass may have materialized around the
// dequant scale, back to the smallest tensor underneath: a single element for a
// per-tensor scale, or [N] for a per-output-channel one.
static Value peelToScaleOperand(Value v, int64_t n) {
    while (true) {
        auto ty = dyn_cast<RankedTensorType>(v.getType());
        if (!ty)
            return nullptr;
        if (ty.getNumElements() == 1 && ty.getRank() <= 1)
            return v;
        if (ty.getRank() == 1 && ty.getShape()[0] == n)
            return v;
        // A rank-3 MatMulInteger result commonly lifts the [N] scale to [1,1,N]
        // with an expand_shape before the final dequant multiply.
        if (auto expand = v.getDefiningOp<tensor::ExpandShapeOp>()) {
            auto srcTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
            if (srcTy && srcTy.getRank() == 1 && srcTy.getShape()[0] == n) {
                v = expand.getSrc();
                continue;
            }
        }
        auto g = v.getDefiningOp<linalg::GenericOp>();
        if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 ||
            g.getNumParallelLoops() != g.getNumLoops() || !g.getRegion().hasOneBlock())
            return nullptr;
        auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
            return nullptr;
        // A pure broadcast: the body yields its input unchanged.
        auto arg = dyn_cast<BlockArgument>(yield.getOperand(0));
        if (!arg || arg.getOwner() != g.getBody() || arg.getArgNumber() != 0)
            return nullptr;
        v = g.getInputs()[0];
    }
}

// Peel a `tensor.extract_slice` that takes one branch's columns out of a fused
// [M, Ntotal] result: all of M, a contiguous unit-stride column range, nothing else.
// A decode-shaped slice rank-reduces the unit M away, which is what M=1 looks like
// and is fine. Returns the sliced source and fills in the column range; null when
// `v` is not such a slice.
static Value peelColumnSlice(Value v, int64_t &offset, int64_t &width) {
    auto es = v.getDefiningOp<tensor::ExtractSliceOp>();
    if (!es)
        return nullptr;
    auto srcTy = dyn_cast<RankedTensorType>(es.getSource().getType());
    if (!srcTy || srcTy.getRank() != 2)
        return nullptr;
    ArrayRef<int64_t> offsets = es.getStaticOffsets();
    ArrayRef<int64_t> sizes = es.getStaticSizes();
    ArrayRef<int64_t> strides = es.getStaticStrides();
    if (offsets.size() != 2 || sizes.size() != 2 || strides.size() != 2)
        return nullptr;
    if (llvm::any_of(offsets, ShapedType::isDynamic) ||
        llvm::any_of(sizes, ShapedType::isDynamic) || llvm::any_of(strides, ShapedType::isDynamic))
        return nullptr;
    if (strides[0] != 1 || strides[1] != 1)
        return nullptr;
    // All of M, so the branch is a pure column range and every row keeps its meaning.
    if (offsets[0] != 0 || sizes[0] != srcTy.getShape()[0])
        return nullptr;
    offset = offsets[1];
    width = sizes[1];
    return es.getSource();
}

// Peel the unit batch dimension torch-mlir wraps a rank-2 weight in when it lowers a
// 3-D x 2-D `aten.matmul` to `linalg.batch_matmul`: a `tensor.expand_shape` of [K,N]
// into [1,K,N]. Null when `v` is not that.
static Value peelUnitBatchExpand(Value v) {
    auto expand = v.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand)
        return nullptr;
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    auto srcTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
    if (!ty || !srcTy || ty.getRank() != 3 || srcTy.getRank() != 2 || ty.getShape()[0] != 1)
        return nullptr;
    return expand.getSrc();
}

// True when every user of `v` is a column slice. A fused q/k/v matmul has one per
// branch, and each branch is rewritten on its own visit -- so this is what says the
// naive i32 chain dies once they all have been, instead of surviving next to the
// raised matmuls.
static bool allUsersAreColumnSlices(Value v) {
    int64_t offset, width;
    return !v.use_empty() && llvm::all_of(v.getUsers(), [&](Operation *user) {
        auto es = dyn_cast<tensor::ExtractSliceOp>(user);
        return es && peelColumnSlice(es.getResult(), offset, width);
    });
}

// True when `v` is a single-input `linalg.generic` whose body is a bare `sitofp`.
static bool isBareSIToFPGeneric(Value v) {
    auto g = v.getDefiningOp<linalg::GenericOp>();
    if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || !g.getRegion().hasOneBlock())
        return false;
    auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
    return yield && yield.getNumOperands() == 1 &&
           yield.getOperand(0).getDefiningOp<arith::SIToFPOp>();
}

// Peel a single-input `linalg.generic` whose body is just `ext{u,s}i`,
// returning the value it widens. Null if `v` is not that shape.
static Value peelExtGeneric(Value v, bool isUnsigned) {
    auto g = v.getDefiningOp<linalg::GenericOp>();
    if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 ||
        g.getNumParallelLoops() != g.getNumLoops() || !g.getRegion().hasOneBlock())
        return nullptr;
    auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
        return nullptr;
    Operation *ext = yield.getOperand(0).getDefiningOp();
    bool extKindOk =
        isUnsigned ? isa_and_nonnull<arith::ExtUIOp>(ext) : isa_and_nonnull<arith::ExtSIOp>(ext);
    if (!extKindOk)
        return nullptr;
    auto arg = dyn_cast<BlockArgument>(ext->getOperand(0));
    if (!arg || arg.getOwner() != g.getBody() || arg.getArgNumber() != 0)
        return nullptr;
    return g.getInputs()[0];
}

// Peel `subi(ext(narrow), zp_i32)`. The front end converts the operand and
// its zero point to si32 separately, each with the extension its ONNX
// signedness calls for, so the subtract itself is a plain i32 subi of two
// block arguments.
//
// On success `narrow` is the i8 operand and `zpWide` the i32 zero point --
// what the rewrite consumes, so it inherits the front end's extension rather
// than re-deriving the sign. `zpNarrow` is what to test for a compile-time
// constant: usually `zpWide`'s pre-widening source, but a *constant* zero
// point has already had its widening folded away (the symmetric-weight
// `w_zp = 0` always has), in which case `zpWide` is itself the constant.
static LogicalResult matchZpSubtractedOperand(
    Value v, bool isUnsigned, Value &narrow, Value &zpNarrow, Value &zpWide,
    bool allowSplatZeroPoint
) {
    auto g = v.getDefiningOp<linalg::GenericOp>();
    if (!g || g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1 ||
        g.getNumParallelLoops() != g.getNumLoops() || !g.getRegion().hasOneBlock())
        return failure();
    auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
        return failure();
    auto subi = yield.getOperand(0).getDefiningOp<arith::SubIOp>();
    if (!subi)
        return failure();
    auto lhsArg = dyn_cast<BlockArgument>(subi.getLhs());
    auto rhsArg = dyn_cast<BlockArgument>(subi.getRhs());
    if (!lhsArg || !rhsArg || lhsArg.getOwner() != g.getBody() ||
        rhsArg.getOwner() != g.getBody() || lhsArg.getArgNumber() != 0 ||
        rhsArg.getArgNumber() != 1)
        return failure();

    Value zpInput = g.getInputs()[1];
    zpWide = peelToScalar(zpInput);
    if (!zpWide && allowSplatZeroPoint)
        zpWide = zpInput;
    if (!zpWide)
        return failure();
    narrow = peelExtGeneric(g.getInputs()[0], isUnsigned);
    if (!zpWide || !narrow)
        return failure();
    zpNarrow = peelExtGeneric(zpWide, isUnsigned);
    if (!zpNarrow)
        zpNarrow = zpWide;
    return success();
}

// Extract a scalar float constant.
static std::optional<APFloat> getScalarFloatConst(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    if (auto fa = dyn_cast<FloatAttr>(cst.getValue()))
        return fa.getValue();
    if (auto dense = dyn_cast<DenseFPElementsAttr>(cst.getValue())) {
        if (dense.isSplat())
            return dense.getSplatValue<APFloat>();
        if (dense.getNumElements() == 1)
            return *dense.getValues<APFloat>().begin();
    }
    return std::nullopt;
}

// Extract a scalar or splat integer constant. MLIR integers are signless, so the caller says
// how to read a narrow one; a value the front end already widened to si32 reads the
// same either way.
static std::optional<int64_t> getIntConst(Value v, bool isUnsigned = false) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    auto read = [&](const APInt &i) {
        return isUnsigned ? static_cast<int64_t>(i.getZExtValue()) : i.getSExtValue();
    };
    if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
        return read(ia.getValue());
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
        if (dense.isSplat())
            return read(dense.getSplatValue<APInt>());
        if (dense.getNumElements() == 1)
            return read(*dense.getValues<APInt>().begin());
    }
    return std::nullopt;
}

class RaiseMatMulInteger : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        // ---- match the runtime-scale dequant mul (the anchor) ----
        if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(op, "not a 2-input elementwise op");
        if (op.getNumParallelLoops() != op.getNumLoops() || !op.getRegion().hasOneBlock())
            return rewriter.notifyMatchFailure(op, "not a parallel elementwise op");

        // Rank 1 is the same dequant on a fused q/k/v branch at M=1: the column slice
        // that cut the branch out rank-reduced the unit M away. Rank 3 is a transformer's
        // activation, `[batch, seq, K]`, which the front end lowers to a
        // `linalg.batch_matmul` with the weight expanded into the batch.
        auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!outType || !outType.getElementType().isBF16() || outType.getRank() < 1 ||
            outType.getRank() > 3)
            return rewriter.notifyMatchFailure(op, "expected a rank-1/2/3 bf16 matmul dequant");
        const bool rankReduced = outType.getRank() == 1;
        // Only a unit batch: a wider one is a stack of independent matmuls sharing one
        // weight, which this rewrite's [M,N] epilogue does not express. It is also not
        // reachable: the front end's own batch broadcast is float-only, so an integer
        // batch_matmul over batch > 1 fails to build before this pattern runs.
        const bool batched = outType.getRank() == 3;
        if (batched && outType.getShape()[0] != 1)
            return rewriter.notifyMatchFailure(op, "expected a unit batch");

        auto yield = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
            return rewriter.notifyMatchFailure(op, "bad yield");
        auto mulf = yield.getOperand(0).getDefiningOp<arith::MulFOp>();
        if (!mulf)
            return rewriter.notifyMatchFailure(op, "not a mulf dequant");

        auto arg0 = dyn_cast<BlockArgument>(mulf.getLhs());
        auto arg1 = dyn_cast<BlockArgument>(mulf.getRhs());
        if (!arg0 || !arg1 || arg0.getOwner() != op.getBody() || arg1.getOwner() != op.getBody())
            return rewriter.notifyMatchFailure(op, "mulf operands are not the generic's inputs");
        Value in0 = op.getInputs()[arg0.getArgNumber()];
        Value in1 = op.getInputs()[arg1.getArgNumber()];

        // One operand is the sitofp(matmul) chain (same shape as the result), the other
        // is the dequant scale. Which is which has to be decided by looking for the
        // cast, not by shape: a per-output-channel scale can be broadcast to the full
        // shape too, and then both operands look alike.
        //
        // A fused q/k/v MatMulInteger is sliced back into per-branch dequants, and the
        // slice sits on either side of the cast depending on the export. Look through
        // one here; the other placement is peeled below the cast.
        Value castArg, scaleArg;
        int64_t colOffset = 0, colWidth = 0;
        bool slicedAfterCast = false;
        for (auto [cand, other] : {std::pair{in0, in1}, std::pair{in1, in0}}) {
            auto ty = dyn_cast<RankedTensorType>(cand.getType());
            if (!ty || ty.getShape() != outType.getShape())
                continue;
            Value peeled = cand;
            int64_t offset = 0, width = 0;
            if (Value src = peelColumnSlice(cand, offset, width)) {
                peeled = src;
                slicedAfterCast = true;
                colOffset = offset;
                colWidth = width;
            }
            if (isBareSIToFPGeneric(peeled)) {
                castArg = peeled;
                scaleArg = other;
                break;
            }
            // Not this operand after all -- drop the column range with it, or the other
            // operand would raise against a branch that is not its own.
            slicedAfterCast = false;
            colOffset = 0;
            colWidth = 0;
        }
        if (!castArg)
            return rewriter.notifyMatchFailure(op, "not a (matmul-cast) * (dequant scale) mul");

        // The scale is either the runtime combined per-tensor scalar or, for a
        // per-output-channel weight scale, an [N] vector -- in both cases possibly
        // already broadcast to the full shape by a standardization pass.
        scaleArg = peelToScaleOperand(scaleArg, outType.getShape().back());
        if (!scaleArg)
            return rewriter.notifyMatchFailure(op, "dequant scale is neither per-tensor nor [N]");

        // ---- peel the bare sitofp cast down to the naive i32 x i32 matmul ----
        // When the branch slice sits above the cast, the cast is legitimately shared --
        // by the sibling branches' slices, and only by those.
        if (slicedAfterCast ? !allUsersAreColumnSlices(castArg) : !castArg.hasOneUse())
            return rewriter.notifyMatchFailure(op, "sitofp cast is shared by other consumers");
        auto castG = castArg.getDefiningOp<linalg::GenericOp>();
        if (!castG || castG.getNumDpsInputs() != 1 || castG.getNumDpsInits() != 1 ||
            !castG.getRegion().hasOneBlock())
            return rewriter.notifyMatchFailure(op, "not a bare cast generic");
        auto castYield = dyn_cast<linalg::YieldOp>(castG.getBody()->getTerminator());
        if (!castYield || castYield.getNumOperands() != 1 ||
            !castYield.getOperand(0).getDefiningOp<arith::SIToFPOp>())
            return rewriter.notifyMatchFailure(op, "cast is not a bare sitofp");

        Value matmulResult = castG.getInputs()[0];
        // The other slice placement: the i32 result is cut into branches before the cast.
        bool slicedBeforeCast = false;
        if (!slicedAfterCast) {
            int64_t offset = 0, width = 0;
            if (Value src = peelColumnSlice(matmulResult, offset, width)) {
                matmulResult = src;
                slicedBeforeCast = true;
                colOffset = offset;
                colWidth = width;
            }
        }
        if (slicedBeforeCast ? !allUsersAreColumnSlices(matmulResult) : !matmulResult.hasOneUse())
            return rewriter.notifyMatchFailure(op, "matmul is shared by other consumers");
        Operation *matmulDef = matmulResult.getDefiningOp();
        if (batched ? !isa_and_nonnull<linalg::BatchMatmulOp>(matmulDef)
                    : !isa_and_nonnull<linalg::MatmulOp>(matmulDef))
            return rewriter.notifyMatchFailure(op, "cast input is not an i32 matmul");
        auto matmulOutTy = dyn_cast<RankedTensorType>(matmulDef->getResult(0).getType());
        if (!matmulOutTy || !matmulOutTy.getElementType().isInteger(32))
            return rewriter.notifyMatchFailure(op, "cast input is not an i32 matmul");
        const bool sliced = slicedAfterCast || slicedBeforeCast;
        if (!sliced)
            colWidth = matmulOutTy.getShape().back();
        // The dequant has to cover exactly this branch, so that replacing it with an
        // epilogue built over [M, colWidth] is shape-for-shape what was there.
        const bool shapeMatches =
            rankReduced ? (matmulOutTy.getShape()[0] == 1 && outType.getShape()[0] == colWidth)
                        : (outType.getShape().drop_back() == matmulOutTy.getShape().drop_back() &&
                           outType.getShape().back() == colWidth);
        if (!shapeMatches)
            return rewriter.notifyMatchFailure(op, "dequant shape is not the matmul branch");

        // ---- peel the zero-point-subtracted operands ----
        SmallVector<Value> matmulInputs = cast<linalg::LinalgOp>(matmulDef).getDpsInputs();
        Value weightOperand = matmulInputs[1];
        if (batched) {
            // The weight is the same [K,N] constant for every batch, wrapped in the unit
            // batch the batch matmul's operand rank needs.
            weightOperand = peelUnitBatchExpand(weightOperand);
            if (!weightOperand)
                return rewriter.notifyMatchFailure(op, "batched weight is not a unit-batch expand");
        }

        Value A, aZpNarrow, aZp, W, wZpNarrow, wZpWide;
        if (failed(matchZpSubtractedOperand(
                matmulInputs[0], /*isUnsigned=*/true, A, aZpNarrow, aZp, false
            )))
            return rewriter.notifyMatchFailure(op, "lhs is not a zp-subtracted u8 activation");
        // ONNX lets B be int8 or uint8, and the front end widens it with the matching
        // extension, so which one it is is readable off the chain.
        bool wUnsigned = false;
        if (failed(matchZpSubtractedOperand(
                weightOperand, /*isUnsigned=*/false, W, wZpNarrow, wZpWide, true
            ))) {
            wUnsigned = true;
            if (failed(matchZpSubtractedOperand(
                    weightOperand, /*isUnsigned=*/true, W, wZpNarrow, wZpWide, true
                )))
                return rewriter.notifyMatchFailure(op, "rhs is not a zp-subtracted i8 weight");
        }

        auto aTy = dyn_cast<RankedTensorType>(A.getType());
        auto wTy = dyn_cast<RankedTensorType>(W.getType());
        if (!aTy || !wTy || !aTy.getElementType().isInteger(8) ||
            !wTy.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(op, "activation/weight are not i8");
        if (aTy.getRank() != (batched ? 3 : 2) || wTy.getRank() != 2)
            return rewriter.notifyMatchFailure(
                op, "expected a rank-2 weight and rank-2/3 activation"
            );

        auto aZpTy = dyn_cast<RankedTensorType>(aZp.getType());
        auto scaleTy = dyn_cast<RankedTensorType>(scaleArg.getType());
        if (!aZpTy || aZpTy.getNumElements() != 1 || !scaleTy)
            return rewriter.notifyMatchFailure(op, "a_zp is not a scalar");
        const bool perChannelScale = scaleTy.getNumElements() > 1;

        // The weight zero point is folded into the weight itself below, so it only has
        // to be a compile-time value. Prefer the front end's already-widened si32 form,
        // which carries the right extension; a narrow constant is read per its ONNX
        // signedness.
        auto wZpVal = getIntConst(wZpWide);
        if (!wZpVal)
            wZpVal = getIntConst(wZpNarrow, wUnsigned);
        if (!wZpVal)
            return rewriter.notifyMatchFailure(op, "weight zero-point is not a compile-time splat");

        // weight must be a compile-time constant (for the column-sum fold).
        auto wConst = W.getDefiningOp<arith::ConstantOp>();
        auto wAttr = wConst ? dyn_cast<ElementsAttr>(wConst.getValue()) : nullptr;
        if (!wAttr)
            return rewriter.notifyMatchFailure(op, "weight is not a compile-time constant");

        // Scan only this branch's columns: a fused q/k/v weight is one [K, Ntotal]
        // constant and each branch raises against its own [K, N] slice of it, taken
        // here at compile time. Row-major, so column `n0 + j` of row k is element
        // k*Ntotal+n0+j.
        const int64_t nTotal = wTy.getShape()[1];
        SmallVector<int64_t> colsum(colWidth, 0);
        int64_t wLo = std::numeric_limits<int64_t>::max();
        int64_t wHi = std::numeric_limits<int64_t>::min();
        int64_t wIdx = 0;
        for (const APInt &v : wAttr.getValues<APInt>()) {
            int64_t col = wIdx++ % nTotal;
            if (col < colOffset || col >= colOffset + colWidth)
                continue;
            int64_t raw = wUnsigned ? static_cast<int64_t>(v.getZExtValue()) : v.getSExtValue();
            wLo = std::min(wLo, raw);
            wHi = std::max(wHi, raw);
            colsum[col - colOffset] += raw - *wZpVal;
        }
        // `W - center` fits int8 exactly for `center` in [wHi - 127, wLo + 128], an interval
        // that is never empty for an 8-bit weight (its range spans at most 255). Take the
        // point of it closest to `w_zp`: that is `w_zp` itself whenever the exact fold fits,
        // leaving `delta == 0` and nothing to emit, and otherwise the smallest residual
        // available -- which is what the added rounding is proportional to. See the header
        // comment.
        const int64_t center = std::clamp(*wZpVal, wHi - 127, wLo + 128);
        const int64_t delta = *wZpVal - center; // 0 when the exact fold fits, and always int8
        assert(delta >= -128 && delta <= 127);

        // ---- rewrite ----
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        auto bf16 = rewriter.getBF16Type();
        int64_t M = matmulOutTy.getShape()[batched ? 1 : 0];
        // N is this branch's width, which is the whole matmul when it was not sliced.
        int64_t N = colWidth;
        auto rawOutTy = RankedTensorType::get({M, N}, matmulOutTy.getElementType());

        // The matmul is the rank-2 form whatever the activation's rank was, so a batched
        // activation gives up its unit batch here. [1,M,K] -> [M,K] is contiguous, and
        // the matmul reads its operand through a descriptor, so this costs nothing.
        Value aOperand = A;
        if (batched) {
            aOperand = tensor::CollapseShapeOp::create(
                           rewriter, loc,
                           RankedTensorType::get({M, aTy.getShape()[2]}, aTy.getElementType()), A,
                           ArrayRef<ReassociationIndices>{
                               ReassociationIndices{0, 1}, ReassociationIndices{2}
                           }
            ).getResult();
        }

        // 1. raw = matmul(A_u8, W_i8) -> i32. Weights stay i8; A keeps its
        // native u8 bit pattern (no int zero-point correction pass over A).
        // A splat constant (not tensor.empty+linalg.fill) so the zero-init
        // survives fuse-group extraction as a plain compile-time constant.
        Value zeroInit =
            arith::ConstantOp::create(rewriter, loc, rawOutTy, rewriter.getZeroAttr(rawOutTy))
                .getResult();
        Value wOperand = W;
        if (center != 0 || sliced) {
            SmallVector<APInt> vals;
            vals.reserve(wTy.getShape()[0] * colWidth);
            wIdx = 0;
            for (const APInt &v : wAttr.getValues<APInt>()) {
                int64_t col = wIdx++ % nTotal;
                if (col < colOffset || col >= colOffset + colWidth)
                    continue;
                int64_t raw = wUnsigned ? static_cast<int64_t>(v.getZExtValue()) : v.getSExtValue();
                int64_t adjusted = raw - center;
                assert(adjusted >= -128 && adjusted <= 127);
                vals.emplace_back(8, adjusted, /*isSigned=*/true);
            }
            auto wOpTy = RankedTensorType::get({wTy.getShape()[0], N}, wTy.getElementType());
            wOperand = arith::ConstantOp::create(
                           rewriter, loc, wOpTy, DenseIntElementsAttr::get(wOpTy, vals)
            )
                           .getResult();
        }
        auto rawOp = linalg::MatmulOp::create(
            rewriter, loc, TypeRange{rawOutTy}, ValueRange{aOperand, wOperand}, ValueRange{zeroInit}
        );
        // The dequant below is a runtime-scale float epilogue, which is
        // neither a QDQ requantize nor something the FC kernel can fuse. Mark
        // the matmul so the LinalgToTorqHL conversion emits it bare instead of
        // stranding it on the host/CSS fallback.
        rawOp->setAttr(TORQ_RAISED_MATMUL_INTEGER, rewriter.getUnitAttr());
        // A reached the naive chain through `extui`, so its i8 elements are a
        // u8 bit pattern. i8 is signless in MLIR and the MAC defaults to
        // signed, so this has to be carried explicitly all the way to the
        // kernel or every activation byte >= 128 multiplies as negative.
        rawOp->setAttr(TORQ_MATMUL_LHS_UNSIGNED, rewriter.getUnitAttr());
        Value raw = rawOp.getResult(0);

        // Scalar params below are kept as rank-1 <1xT> (not rank-0): the torq_hl
        // elementwise-binary/cast kernel lowering requires at least rank-1
        // addressing. Depending on the shapes, `aZp`/`scaleArg` arrive either
        // rank-0 or already <1xT>, so only the rank-0 ones need reshaping.
        SmallVector<OpFoldResult> scalarOutputShape{rewriter.getIndexAttr(1)};
        auto reshapeToRank1 = [&](Value v, Type elemTy) -> Value {
            if (cast<RankedTensorType>(v.getType()).getRank() == 1)
                return v;
            return tensor::ExpandShapeOp::create(
                       rewriter, loc, RankedTensorType::get({1}, elemTy), v,
                       ArrayRef<ReassociationIndices>{}, scalarOutputShape
            )
                .getResult();
        };
        Value scale1 = perChannelScale ? scaleArg : reshapeToRank1(scaleArg, bf16);

        // The dequant is built over the *dequant's own* shape, which carries the unit
        // batch when the activation was rank 3. Writing it directly is what keeps the
        // batch dimension off the result reshape: a `tensor.expand_shape` there is not
        // free, it materializes a full copy of the result.
        const unsigned epiRank = batched ? 3 : 2;
        SmallVector<int64_t> epiShape(outType.getShape().begin(), outType.getShape().end());
        if (rankReduced)
            epiShape = {M, N};
        auto bf16TyEpi = RankedTensorType::get(epiShape, bf16);
        auto idMapEpi = AffineMap::getMultiDimIdentityMap(epiRank, ctx);
        // (d...) -> (0): broadcast a <1xT> scalar into the epilogue's iteration space.
        auto scalarBcastMapEpi = AffineMap::get(epiRank, 0, {getAffineConstantExpr(0, ctx)}, ctx);
        // (d...) -> (dLast): a per-output-channel [N] operand.
        auto nBcastMapEpi = AffineMap::get(epiRank, 0, {getAffineDimExpr(epiRank - 1, ctx)}, ctx);
        // The matmul result is rank 2 whatever the epilogue's rank is; a batched epilogue
        // reads it with the batch dimension dropped, and that is where the rank comes
        // back -- no reshape op.
        auto rawMapEpi =
            batched
                ? AffineMap::get(3, 0, {getAffineDimExpr(1, ctx), getAffineDimExpr(2, ctx)}, ctx)
                : idMapEpi;
        SmallVector<utils::IteratorType> parallelEpi(epiRank, utils::IteratorType::parallel);

        // 2. corrected = raw + a_zp*neg_colsum_W[n] - delta*rowsum_A[m], all in i32.
        auto i32Ty = rewriter.getIntegerType(32);
        auto i32TyEpi = RankedTensorType::get(epiShape, i32Ty);
        // Negated at compile time so the kernel can express the correction as a MAC term
        // against `a_zp` rather than a subtract.
        SmallVector<APInt> negColsumVals;
        negColsumVals.reserve(N);
        for (int64_t s : colsum)
            negColsumVals.emplace_back(32, -s, /*isSigned=*/true);
        auto colsumI32Ty = RankedTensorType::get({N}, i32Ty);
        Value negColsumConst =
            arith::ConstantOp::create(
                rewriter, loc, colsumI32Ty, DenseIntElementsAttr::get(colsumI32Ty, negColsumVals)
            )
                .getResult();
        // The runtime uint8 zero point, reshaped to rank 1. It is widened per-element inside
        // the [M,N] correction below: a standalone [N] integer op has no slice lowering, so
        // folding `a_zp * colsum` into the one [M,N] pass keeps it off host.
        auto aZpNarrowTy = cast<RankedTensorType>(aZpNarrow.getType());
        Value aZp1 = reshapeToRank1(aZpNarrow, aZpNarrowTy.getElementType());

        // delta * rowsum_A[m]: the residual weight zero point the compile-time fold could not
        // absorb.
        //
        // `[M, 2]`, not `[M]`: the zero second column makes the collapsed result the `{value, 0}`
        // interleaving the ACT's `{bias, scale}` record needs.
        Value drs2;
        if (delta != 0) {
            const int64_t K = aTy.getShape()[batched ? 2 : 1];
            auto deltaColTy = RankedTensorType::get({K, 2}, rewriter.getIntegerType(8));
            SmallVector<APInt> deltaColVals;
            deltaColVals.reserve(2 * K);
            for (int64_t k = 0; k < K; ++k) {
                deltaColVals.emplace_back(8, delta, /*isSigned=*/true);
                deltaColVals.emplace_back(8, 0, /*isSigned=*/true);
            }
            Value deltaCol =
                arith::ConstantOp::create(
                    rewriter, loc, deltaColTy, DenseIntElementsAttr::get(deltaColTy, deltaColVals)
                )
                    .getResult();
            auto drsTy = RankedTensorType::get({M, 2}, matmulOutTy.getElementType());
            Value drsInit =
                arith::ConstantOp::create(rewriter, loc, drsTy, rewriter.getZeroAttr(drsTy))
                    .getResult();
            auto drsOp = linalg::MatmulOp::create(
                rewriter, loc, TypeRange{drsTy}, ValueRange{aOperand, deltaCol}, ValueRange{drsInit}
            );
            drsOp->setAttr(TORQ_RAISED_MATMUL_INTEGER, rewriter.getUnitAttr());
            drsOp->setAttr(TORQ_MATMUL_LHS_UNSIGNED, rewriter.getUnitAttr());
            // Column 0 as a rank-reduced [M] view: a `(d..) -> (dM, 0)` map on the [M, 2]
            // value is normalized into a full-shape broadcast, the [M, N] copy this avoids.
            drs2 = tensor::ExtractSliceOp::create(
                       rewriter, loc, RankedTensorType::get({M}, drsTy.getElementType()),
                       drsOp.getResult(0),
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(M), rewriter.getIndexAttr(1)},
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)}
            )
                       .getResult();
        }

        // The scalar a_zp and the [N] neg_colsum ride in as slice inputs (exactly like the float
        // scale/bias) and neither is ever broadcast to [M,N].
        Value corrected =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{i32TyEpi}, ValueRange{raw, aZp1, negColsumConst},
                ValueRange{tensor::EmptyOp::create(rewriter, loc, epiShape, i32Ty)},
                ArrayRef<AffineMap>{rawMapEpi, scalarBcastMapEpi, nBcastMapEpi, idMapEpi},
                parallelEpi,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    Value az = args[1];
                    unsigned w = az.getType().getIntOrFloatBitWidth();
                    if (w < 32)
                        az = arith::ExtUIOp::create(b, l, i32Ty, az).getResult();
                    else if (w > 32)
                        az = arith::TruncIOp::create(b, l, i32Ty, az).getResult();
                    Value corr = arith::MulIOp::create(b, l, az, args[2]).getResult();
                    linalg::YieldOp::create(
                        b, l, arith::AddIOp::create(b, l, args[0], corr).getResult()
                    );
                }
            ).getResult(0);

        // 2b. Its own pass: there is one ACT bias record per pass and the column term uses one.
        if (drs2) {
            // (d...) -> (dM): the per-row term broadcast over the columns.
            auto drsMapEpi =
                AffineMap::get(epiRank, 0, {getAffineDimExpr(batched ? 1 : 0, ctx)}, ctx);
            corrected = linalg::GenericOp::create(
                            rewriter, loc, TypeRange{i32TyEpi}, ValueRange{corrected, drs2},
                            ValueRange{tensor::EmptyOp::create(rewriter, loc, epiShape, i32Ty)},
                            ArrayRef<AffineMap>{idMapEpi, drsMapEpi, idMapEpi}, parallelEpi,
                            [&](OpBuilder &b, Location l, ValueRange args) {
                                linalg::YieldOp::create(
                                    b, l, arith::SubIOp::create(b, l, args[0], args[1]).getResult()
                                );
                            }
            ).getResult(0);
        }

        // 3. rawBf16 = sitofp(corrected): the only bf16 rounding, on the small corrected value.
        Value rawBf16Init = tensor::EmptyOp::create(rewriter, loc, epiShape, bf16);
        Value rawBf16 =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{bf16TyEpi}, ValueRange{corrected}, ValueRange{rawBf16Init},
                ArrayRef<AffineMap>{idMapEpi, idMapEpi}, parallelEpi,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    linalg::YieldOp::create(
                        b, l, arith::SIToFPOp::create(b, l, bf16, args[0]).getResult()
                    );
                }
            ).getResult(0);

        // 4. out = rawBf16 * scale (the runtime combined a_scale*b_scale).
        Value scaledInit = tensor::EmptyOp::create(rewriter, loc, epiShape, bf16);
        Value scaled =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{bf16TyEpi}, ValueRange{rawBf16, scale1},
                ValueRange{scaledInit},
                ArrayRef<AffineMap>{
                    idMapEpi, perChannelScale ? nBcastMapEpi : scalarBcastMapEpi, idMapEpi
                },
                parallelEpi,
                [&](OpBuilder &b, Location l, ValueRange args) {
                    linalg::YieldOp::create(
                        b, l, arith::MulFOp::create(b, l, args[0], args[1]).getResult()
                    );
                }
            ).getResult(0);

        // A rank-reducing branch slice dropped a unit M the epilogue above built over, so
        // put it back.
        Value result = scaled;
        if (rankReduced) {
            result = tensor::CollapseShapeOp::create(
                         rewriter, loc, outType, scaled,
                         ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}}
            ).getResult();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

} // namespace

void populateRaiseMatMulIntegerOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseMatMulInteger>(context);
}

} // namespace mlir::syna::torq
