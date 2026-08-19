// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RaiseDynamicQuantize
// --------------------
// torch-mlir decomposes onnx.DynamicQuantizeLinear (per-tensor, unsigned uint8)
// into ~18 linalg ops in one dispatch: a min and a max reduction over the input
// `x`, a scalar scale/zero-point derivation, and a final elementwise quantize
//   y = fptoui(clamp(roundeven(x / scale) + zp, 0, 255))
// with a *runtime* scale/zp.
//
// This raise runs before reduction tiling  and rewrites the whole thing into an efficient,
// tileable form: cast x to bf16, take min and max in a single combined 2-output reduce
// (one streaming pass over x), derive inv_scale via one linalg.reciprocal, and
// emit the reused runtime-scale quantize (mul + round + clamp -> uint8). We
// recompute scale/zp ourselves — the unrounded bf16 zero-point
//   zp = -min(0, min(x)) * inv_scale   (provably in [0,255], no clip needed)
// avoids a systematic dequant bias, so the old scalar chain is left to die.
//
// Raised form:
//   x_bf16      = truncf(x)                       // only when x is f32
//   min_x,max_x = reduce_min/max(x_bf16)          // one combined 2-output reduce
//   min_adj     = min(min_x, 0)
//   max_adj     = max(max_x, 0)
//   range       = max_adj - min_adj
//   range_safe  = max(range, eps)                 // eps guards the all-equal input
//   inv_scale   = reciprocal(range_safe) * 255    // = 255 / range_safe
//   zp          = -min_adj * inv_scale            // unrounded, in [0, 255]
//   y           = fptoui(clamp(roundeven(x_bf16 * inv_scale + zp), 0, 255))

#include "Patterns.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

namespace {

// Match a scalar (captured) float constant, tolerating a small tolerance.
static bool isConstFloat(Value v, double expected) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return false;
    auto fattr = dyn_cast<FloatAttr>(cst.getValue());
    if (!fattr)
        return false;
    return std::abs(fattr.getValueAsDouble() - expected) < 1e-6;
}

// For a commutative binary op with one constant operand == `expected`, return
// the other (data) operand; null otherwise.
static Value dataOperandOfClamp(Operation *binOp, double expected) {
    if (isConstFloat(binOp->getOperand(0), expected))
        return binOp->getOperand(1);
    if (isConstFloat(binOp->getOperand(1), expected))
        return binOp->getOperand(0);
    return nullptr;
}

// Is `user` a single-input reduction linalg.generic over `x` whose body is the
// given float combine op? Returns true for the DQL min (MinimumFOp) / max
// (MaximumFOp) reductions we expect to feed the scale.
template <typename CombineOp> static bool isReductionOverX(Operation *user, Value x) {
    auto g = dyn_cast<linalg::GenericOp>(user);
    if (!g || g.getNumReductionLoops() == 0 || g.getInputs().size() != 1)
        return false;
    if (g.getInputs()[0] != x)
        return false;
    auto yield = dyn_cast<linalg::YieldOp>(g.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
        return false;
    return isa_and_nonnull<CombineOp>(yield.getOperand(0).getDefiningOp());
}

class RaiseDynamicQuantize : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        // ---- match the quantize generic (the anchor) ----
        if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
            return rewriter.notifyMatchFailure(op, "not a single-input elementwise op");
        if (op.getNumParallelLoops() != op.getNumLoops() || !op.getRegion().hasOneBlock())
            return rewriter.notifyMatchFailure(op, "not a parallel elementwise op");

        Value x = op.getInputs()[0];
        auto xType = dyn_cast<RankedTensorType>(x.getType());
        auto outType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!xType || !outType || !outType.getElementType().isInteger(8))
            return rewriter.notifyMatchFailure(op, "output is not i8");
        if (!xType.getElementType().isF32() && !xType.getElementType().isBF16())
            return rewriter.notifyMatchFailure(op, "input is not f32/bf16");

        auto yield = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
        if (!yield || yield.getNumOperands() != 1)
            return rewriter.notifyMatchFailure(op, "bad yield");
        auto fptoui = yield.getOperand(0).getDefiningOp<arith::FPToUIOp>();
        if (!fptoui)
            return rewriter.notifyMatchFailure(op, "not an unsigned quantize");

        // fptoui(minimumf(maximumf(addf(roundeven(divf(x, scale)), zp), 0), 255))
        auto minf = fptoui.getIn().getDefiningOp<arith::MinimumFOp>();
        if (!minf)
            return rewriter.notifyMatchFailure(op, "no upper clamp");
        Value maxData = dataOperandOfClamp(minf, 255.0);
        auto maxf = maxData ? maxData.getDefiningOp<arith::MaximumFOp>() : nullptr;
        if (!maxf)
            return rewriter.notifyMatchFailure(op, "no lower clamp");
        Value addData = dataOperandOfClamp(maxf, 0.0);
        auto addf = addData ? addData.getDefiningOp<arith::AddFOp>() : nullptr;
        if (!addf)
            return rewriter.notifyMatchFailure(op, "no zero-point add");

        // The zero-point add's data operand is round(x / scale); the explicit
        // roundeven is optional (the fptoui epilogue rounds regardless), and
        // either add operand may hold it.
        arith::DivFOp divf;
        for (Value operand : {addf.getLhs(), addf.getRhs()}) {
            Value v = operand;
            if (auto rnd = v.getDefiningOp<math::RoundEvenOp>())
                v = rnd.getOperand();
            if ((divf = v.getDefiningOp<arith::DivFOp>()))
                break;
        }
        if (!divf)
            return rewriter.notifyMatchFailure(op, "not a divide-by-scale quantize");
        auto numArg = dyn_cast<BlockArgument>(divf.getLhs());
        if (!numArg || numArg.getArgNumber() != 0)
            return rewriter.notifyMatchFailure(op, "numerator is not the elementwise input");

        // ---- confirm this is a per-tensor dynamic quant: x is reduced by both
        // a min and a max reduction (the scale derivation). ----
        bool hasMin = false, hasMax = false;
        for (Operation *user : x.getUsers()) {
            hasMin |= isReductionOverX<arith::MinimumFOp>(user, x);
            hasMax |= isReductionOverX<arith::MaximumFOp>(user, x);
        }
        if (!hasMin || !hasMax)
            return rewriter.notifyMatchFailure(op, "x is not min+max reduced (not DQL)");

        // If x is the bf16 result of a --torq-convert-dtypes elementwise truncf
        // from an f32 source, anchor on that f32 source: the reduce and the
        // quantize then convert per-tile below instead of both reading a shared,
        // materialized bf16 copy of x (which spills to DRAM — x is read twice).
        // The bf16 cast is left to die if the DQL was its only consumer.
        if (xType.getElementType().isBF16()) {
            if (auto castG = x.getDefiningOp<linalg::GenericOp>()) {
                auto castYield = dyn_cast<linalg::YieldOp>(castG.getBody()->getTerminator());
                auto tf = castG.getNumDpsInputs() == 1 &&
                                  castG.getNumParallelLoops() == castG.getNumLoops() && castYield &&
                                  castYield.getNumOperands() == 1
                              ? castYield.getOperand(0).getDefiningOp<arith::TruncFOp>()
                              : nullptr;
                auto srcArg = tf ? dyn_cast<BlockArgument>(tf.getIn()) : nullptr;
                if (srcArg && srcArg.getArgNumber() == 0) {
                    Value src = castG.getInputs()[0];
                    if (auto srcTy = dyn_cast<RankedTensorType>(src.getType());
                        srcTy && srcTy.getElementType().isF32()) {
                        x = src;
                        xType = srcTy;
                    }
                }
            }
        }

        // ---- rewrite ----
        // Scalar params are kept as rank-1 <1xbf16> (not rank-0): the torq_hl
        // elementwise-binary kernel promotes a scalar constant operand to <1x..>
        // and requires both operands to share that shape.
        Location loc = op.getLoc();
        MLIRContext *ctx = rewriter.getContext();
        auto bf16 = rewriter.getBF16Type();
        int64_t rank = xType.getRank();
        auto xShape = xType.getShape();
        auto idMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
        auto scalarBcastMap = AffineMap::get(rank, 0, {getAffineConstantExpr(0, ctx)}, ctx);
        SmallVector<utils::IteratorType> parallel(rank, utils::IteratorType::parallel);
        auto scalarTy = RankedTensorType::get({1}, bf16);

        // 1. helper to cast x to bf16 (only when x is f32). Used solely by the
        // quantize (step 5); the reduce (step 3) truncs in its own body. This
        // avoids one shared full-tensor bf16 copy of x: such a value has two
        // consumers in different (post-tiling) loops, so it bufferizes to a
        // DRAM-resident buffer written once and read back twice. Reducing x
        // directly plus a single-use quantize cast keeps the bf16 tiles
        // resident in LRAM instead.
        auto castToBf16 = [&]() -> Value {
            if (!xType.getElementType().isF32())
                return x;
            Value init = tensor::EmptyOp::create(rewriter, loc, xShape, bf16);
            return linalg::GenericOp::create(
                       rewriter, loc, TypeRange{RankedTensorType::get(xShape, bf16)}, ValueRange{x},
                       ValueRange{init}, ArrayRef<AffineMap>{idMap, idMap}, parallel,
                       [&](OpBuilder &b, Location l, ValueRange args) {
                           linalg::YieldOp::create(
                               b, l, arith::TruncFOp::create(b, l, bf16, args[0]).getResult()
                           );
                       }
            ).getResult(0);
        };
        // 2. reshape x to <1 x N> so the reduce has a single reduction dim
        // (tileable). Keep the original element type: the reduce truncs to bf16
        // in its own body (below), so it reads x directly and no full bf16 copy
        // of x is materialized for the reduction path.
        Type xElemTy = xType.getElementType();
        int64_t numElts = xType.getNumElements();
        Value xFlat;
        {
            ReassociationIndices group;
            for (int64_t i = 0; i < rank; ++i)
                group.push_back(i);
            Value flat1d = rank == 1
                               ? x
                               : tensor::CollapseShapeOp::create(
                                     rewriter, loc, x, SmallVector<ReassociationIndices>{group}
                                 )
                                     .getResult();
            xFlat = tensor::ExpandShapeOp::create(
                        rewriter, loc, RankedTensorType::get({1, numElts}, xElemTy), flat1d,
                        SmallVector<ReassociationIndices>{{0, 1}}
            ).getResult();
        }

        // 3. combined min+max in one 2-output reduce (one streaming pass over x).
        Value minInit = makeScalarFill(rewriter, loc, scalarTy, /*neg=*/false); // +inf
        Value maxInit = makeScalarFill(rewriter, loc, scalarTy, /*neg=*/true);  // -inf
        auto inMap = AffineMap::getMultiDimIdentityMap(2, ctx);
        auto outMap = AffineMap::get(2, 0, {getAffineDimExpr(0, ctx)}, ctx);
        auto minmax = linalg::GenericOp::create(
            rewriter, loc, TypeRange{scalarTy, scalarTy}, ValueRange{xFlat},
            ValueRange{minInit, maxInit}, ArrayRef<AffineMap>{inMap, outMap, outMap},
            ArrayRef<utils::IteratorType>{
                utils::IteratorType::parallel, utils::IteratorType::reduction
            },
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value xb = xElemTy.isF32()
                               ? arith::TruncFOp::create(b, l, bf16, args[0]).getResult()
                               : args[0];
                Value mn = arith::MinimumFOp::create(b, l, xb, args[1]);
                Value mx = arith::MaximumFOp::create(b, l, xb, args[2]);
                linalg::YieldOp::create(b, l, ValueRange{mn, mx});
            }
        );
        Value minX = minmax.getResult(0);
        Value maxX = minmax.getResult(1);

        // 4. scalar param math (bf16, off the x path).
        // Fix B: derive the whole scale/zp chain in a dedicated kernel
        // (torq_hl.derive_quant_params), which is legal through LinalgToTorqHL (passes
        // through untouched) and lowers to a sequence of slice_tasks inside one program
        // body (one code load). It folds in max_adj/neg_min_adj/range_safe, the full-range
        // reciprocal (bit-identical to the linalg BfloatReciprocalPattern), and the two
        // scale/zp muls. See the op's slot layout in DeriveQuantParamsPattern.cpp; the
        // results land in init[0] = inv_scale, init[1] = zp.
        auto i32Ty = rewriter.getI32Type();
        auto sbTy = RankedTensorType::get({2}, i32Ty);
        auto makeScaleBias = [&](int32_t bias, int32_t scale) -> Value {
            return arith::ConstantOp::create(
                       rewriter, loc,
                       DenseIntElementsAttr::get(sbTy, ArrayRef<int32_t>{bias, scale})
            )
                .getResult();
        };
        // ACT (bias, scale) for the reciprocal exponent-subtract (0x7E80 - x) and the
        // mantissa-LUT decode (base<<8), matching AddPattern / ExtractPattern; and the
        // 255.0 WRAM weight for inv_scale = recip * 255.
        Value subScaleBias = makeScaleBias(-0x7E80, -1);
        Value tableScaleBias = makeScaleBias(-128, 128);
        // WRAM constants for the two multiplies: inv_scale = recip * 255 and
        // scale = range_safe * (1/255).
        Value mulWeight = makeConstVector(rewriter, loc, bf16, {255.0, 1.0 / 255.0});
        auto blobTy = RankedTensorType::get({11}, bf16);
        Value blobInit = tensor::EmptyOp::create(rewriter, loc, blobTy.getShape(), bf16);
        Value blob =
            torq_hl::DeriveQuantParamsOp::create(
                rewriter, loc, blobTy, blobInit, minX, maxX, subScaleBias, tableScaleBias, mulWeight
            )
                .getResult(0);
        auto extractScalar = [&](int64_t slot) -> Value {
            return tensor::ExtractSliceOp::create(
                       rewriter, loc, scalarTy, blob,
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(slot)},
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)},
                       ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)}
            )
                .getResult();
        };
        Value invScale = extractScalar(0); // INV_SCALE slot
        Value zp = extractScalar(1);       // ZP slot

        // 5. runtime-scale quantize: y = fptoui(clamp(round(x*inv_scale + zp))).
        // A fresh single-use cast (see step 1) so it fuses into the quantize
        // loop instead of reloading a materialized bf16 copy of x.
        Value xbf16Quant = castToBf16();
        Value outInit =
            tensor::EmptyOp::create(rewriter, loc, outType.getShape(), outType.getElementType());
        Value y = linalg::GenericOp::create(
                      rewriter, loc, TypeRange{outType}, ValueRange{xbf16Quant, invScale, zp},
                      ValueRange{outInit},
                      ArrayRef<AffineMap>{idMap, scalarBcastMap, scalarBcastMap, idMap}, parallel,
                      [&](OpBuilder &b, Location l, ValueRange args) {
                          Value c0 = arith::ConstantOp::create(b, l, b.getFloatAttr(bf16, 0.0));
                          Value c255 = arith::ConstantOp::create(b, l, b.getFloatAttr(bf16, 255.0));
                          Value m = arith::MulFOp::create(b, l, args[0], args[1]);
                          Value biased = arith::AddFOp::create(b, l, m, args[2]);
                          Value r = math::RoundEvenOp::create(b, l, biased);
                          Value lo = arith::MaximumFOp::create(b, l, r, c0);
                          Value hi = arith::MinimumFOp::create(b, l, lo, c255);
                          Value u = arith::FPToUIOp::create(b, l, outType.getElementType(), hi);
                          linalg::YieldOp::create(b, l, u);
                      }
        ).getResult(0);

        rewriter.replaceOp(op, y);
        return success();
    }

  private:
    static Value
    makeScalarConst(PatternRewriter &rewriter, Location loc, RankedTensorType scalarTy, double v) {
        APFloat ap(v);
        bool losesInfo;
        ap.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
        return arith::ConstantOp::create(rewriter, loc, DenseFPElementsAttr::get(scalarTy, ap))
            .getResult();
    }

    // A dense bf16 constant vector, one element per value.
    static Value
    makeConstVector(PatternRewriter &rewriter, Location loc, Type bf16, ArrayRef<double> values) {
        SmallVector<APFloat> aps;
        for (double v : values) {
            APFloat ap(v);
            bool losesInfo;
            ap.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
            aps.push_back(ap);
        }
        auto ty = RankedTensorType::get({static_cast<int64_t>(values.size())}, bf16);
        return arith::ConstantOp::create(rewriter, loc, DenseFPElementsAttr::get(ty, aps))
            .getResult();
    }

    static Value
    makeScalarFill(PatternRewriter &rewriter, Location loc, RankedTensorType scalarTy, bool neg) {
        APFloat inf = APFloat::getInf(APFloat::BFloat(), neg);
        Value cst = arith::ConstantOp::create(
                        rewriter, loc, scalarTy.getElementType(),
                        rewriter.getFloatAttr(scalarTy.getElementType(), inf)
        )
                        .getResult();
        Value init =
            tensor::EmptyOp::create(rewriter, loc, scalarTy.getShape(), scalarTy.getElementType());
        return linalg::FillOp::create(rewriter, loc, ValueRange{cst}, ValueRange{init})
            .getResult(0);
    }
};

} // namespace

void populateRaiseDynamicQuantizeOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseDynamicQuantize>(context);
}

} // namespace mlir::syna::torq
