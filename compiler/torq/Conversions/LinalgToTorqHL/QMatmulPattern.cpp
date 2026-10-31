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
#include "torq/Utils/ExecutorAssignment.h"

#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#define DEBUG_TYPE "linalg-torq-q-matmul-pattern"

namespace mlir::syna::torq {

namespace {

// Quantized matmul (QDQ Gemm) lowering.
//
// torch-mlir fuses DQ -> Gemm -> Q into linalg.quantized_matmul; IREE's
// GlobalOptimization pass QuantizedMatmulToMatmul
// (third_party/iree/compiler/src/iree/compiler/GlobalOptimization/QuantizedMatmulToMatmul.cpp,
// scheduled unconditionally in GlobalOptimization/Passes.cpp) then decomposes
// it into a plain integer matmul plus explicit zero-point arithmetic
// (Section 2.3 of https://arxiv.org/abs/1712.05877).  So the TORQ backend
// never sees a `_q` matmul — it sees the decomposed chain:
//
//   %tw  = linalg.transpose ins(%w : tensor<OxKxi8>) permutation [1, 0]
//   %mm  = linalg.matmul ins(%in : tensor<MxKxi8>, %tw : tensor<KxOxi8>)
//                        outs(%init : tensor<MxOxi32>)           // raw sum(x*w)
//   %red = linalg.generic ins(%tw)   outs(%z : tensor<Oxi32>)   // sumW[o] = sum_k W[k,o]
//   %acc = linalg.generic ins(%mm, %red, %azp) outs(...)        // acc - azp*sumW
//   %dq  = linalg.generic ins(%acc)  outs(... : f32)            // sitofp * (a_scale*b_scale)
//   %bdq = linalg.generic ins(%bias_i32) outs(... : f32)        // bias dequant, same scale
//   %add = linalg.generic ins(%dq, %bdq) outs(...)              // addf   (opt)
//   %q   = linalg.generic ins(%add)  outs(... : i8)             // div/round/clamp/fptosi
//
// (The conv pass QuantizedConvToConv only decomposes NHWC quantized
// convs, which is why QConv2D still sees linalg.conv_2d_nchw_fchw_q while the
// matmul arrives decomposed.)
//
// This pattern matches the whole chain and rewrites it to a single
// torq_hl.fully_connected op.  The input_zp correction and the bias are folded
// into the interleaved {2*O} scale_bias tensor using the deferred (JIT)
// constant-resolution helpers in QuantPatternUtils.
//
// After tiling, the matmul weights and the reduce input may be different
// extract_slice SSA values of the same source tensor.  Peel slices to compare
// the underlying tensors.
Value peelSlice(Value v) {
    while (auto es = v.getDefiningOp<tensor::ExtractSliceOp>())
        v = es.getSource();
    return v;
}

// Peel expand_shape/collapse_shape shape casts (e.g. [O] -> [1, O]).
Value peelShapeCasts(Value v) {
    while (true) {
        if (auto expandOp = v.getDefiningOp<tensor::ExpandShapeOp>()) {
            v = expandOp.getSrc();
            continue;
        }
        if (auto collapseOp = v.getDefiningOp<tensor::CollapseShapeOp>()) {
            v = collapseOp.getSrc();
            continue;
        }
        return v;
    }
}

// Returns the single user of `v` as a linalg::GenericOp, or nullptr when `v`
// has multiple uses or the user is not a generic.
linalg::GenericOp getSingleGenericUser(Value v) {
    if (!v.hasOneUse())
        return nullptr;
    return dyn_cast<linalg::GenericOp>(*v.getUsers().begin());
}

// Returns the defining op of the single yielded value of a generic's body.
Operation *getYieldedValueDef(linalg::GenericOp op) {
    if (!op.getRegion().hasOneBlock())
        return nullptr;
    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return nullptr;
    return yieldOp.getOperand(0).getDefiningOp();
}

// Returns the block argument number of `v` if it is a block argument of `op`'s
// body.
std::optional<unsigned> getBlockArgNumber(Value v, linalg::GenericOp op) {
    auto ba = dyn_cast<BlockArgument>(v);
    if (!ba || ba.getOwner() != op.getBody())
        return std::nullopt;
    return ba.getArgNumber();
}

// Erase the dead weight-sum producer once its correction is folded into scale_bias.
// A partial-result merge also leaves its source loop dead.
void eraseIfDead(Value v, PatternRewriter &rewriter) {
    Operation *op = v.getDefiningOp();
    if (!op || !op->use_empty())
        return;
    Value partials;
    if (auto merge = dyn_cast<linalg::ReduceOp>(op))
        partials = merge.getInputs()[0];
    rewriter.eraseOp(op);
    if (partials)
        eraseIfDead(partials, rewriter);
}

// Matches the weight column-sum reduce produced by the quantized-matmul
// decomposition for the input zero-point correction:
//   linalg.generic ins(%weights : tensor<KxOxi8>) outs(%zero : tensor<Oxi32>) {
//     extsi, addi, yield }
// with indexing maps [(d0, d1) -> (d0, d1), (d0, d1) -> (d1)] and iterators
// ["parallel", "reduction"].
bool isWeightSumReduce(linalg::GenericOp op, Value weights) {
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
        return false;
    if (peelSlice(op.getInputs()[0]) != peelSlice(weights))
        return false;
    auto iterTypes = op.getIteratorTypesArray();
    if (iterTypes.size() != 2 || iterTypes[0] != utils::IteratorType::parallel ||
        iterTypes[1] != utils::IteratorType::reduction)
        return false;
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!outTy || outTy.getRank() != 1 || !outTy.getElementType().isInteger(32))
        return false;
    auto addi = dyn_cast_or_null<arith::AddIOp>(getYieldedValueDef(op));
    if (!addi)
        return false;
    auto ext = addi.getLhs().getDefiningOp<arith::ExtSIOp>();
    if (!ext)
        ext = addi.getRhs().getDefiningOp<arith::ExtSIOp>();
    if (!ext)
        return false;
    return getBlockArgNumber(ext.getIn(), op) == 0;
}

// The weight column-sum reduce may have been tiled for LRAM into an scf.for
// over the K dim (TileReductionForLram). Two forms occur after tiling:
//   1. direct:  %r = scf.for ... iter_args(acc [O]) { reduce(slice of weights);
//               insert into acc }
//   2. table:   an scf.for fills an [O, chunks] partials table (each chunk's
//               reduce over its weight slice), followed by a linalg.reduce
//               that merges the independent partials with addi; the sum may be
//               read through a per-chunk extract_slice.
// The slicing plan may also extract the sum out of the dispatch, in which case
// it arrives as a block argument, possibly read through a per-chunk
// extract_slice. The rewrite recomputes the correction from the weight
// constant, so for that form only the shape is load-bearing (the value is
// never read; matchZpCorrectionCombine checks it against O).
bool isWeightSumProducer(Value v, Value weights) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || ty.getRank() != 1 || !ty.getElementType().isInteger(32))
        return false;
    if (auto genOp = v.getDefiningOp<linalg::GenericOp>())
        return isWeightSumReduce(genOp, weights);

    // Form 2: peel to the linalg.reduce over the tiled partials table.
    Value peeled = peelSlice(v);
    if (auto reduceOp = peeled.getDefiningOp<linalg::ReduceOp>()) {
        if (reduceOp.getNumDpsInputs() != 1 || reduceOp.getDimensions() != ArrayRef<int64_t>{1})
            return false;
        // Compare the partials table's dim 0 with the reduced (peeled) sum, not
        // with the unpeeled `v`: the sum may be read through a per-chunk
        // extract_slice whose dim 0 is a prefix of the table's.
        auto peeledTy = dyn_cast<RankedTensorType>(peeled.getType());
        auto partialTy = dyn_cast<RankedTensorType>(reduceOp.getInputs()[0].getType());
        if (!peeledTy || !partialTy || partialTy.getRank() != 2 ||
            partialTy.getDimSize(0) != peeledTy.getDimSize(0) ||
            partialTy.getElementType() != peeledTy.getElementType())
            return false;
        // The merge reduce adds its two block arguments.
        auto yield = dyn_cast<linalg::YieldOp>(reduceOp.getBody()->getTerminator());
        auto add = yield && yield.getNumOperands() == 1
                       ? yield.getOperand(0).getDefiningOp<arith::AddIOp>()
                       : nullptr;
        if (!add)
            return false;
        auto isArg = [&](Value operand, unsigned index) {
            auto arg = dyn_cast<BlockArgument>(operand);
            return arg && arg.getOwner() == reduceOp.getBody() && arg.getArgNumber() == index;
        };
        if (!((isArg(add.getLhs(), 0) && isArg(add.getRhs(), 1)) ||
              (isArg(add.getLhs(), 1) && isArg(add.getRhs(), 0))))
            return false;
        auto tableFor = reduceOp.getInputs()[0].getDefiningOp<scf::ForOp>();
        if (!tableFor)
            return false;
        for (Operation &bodyOp : tableFor.getBody()->getOperations()) {
            auto genOp = dyn_cast<linalg::GenericOp>(&bodyOp);
            if (!genOp || genOp.getNumDpsInputs() != 1)
                continue;
            if (peelSlice(genOp.getInputs()[0]) != peelSlice(weights))
                continue;
            auto iterTypes = genOp.getIteratorTypesArray();
            if (iterTypes.size() == 2 && iterTypes[0] == utils::IteratorType::parallel &&
                iterTypes[1] == utils::IteratorType::reduction)
                return true;
        }
        return false;
    }
    // Extracted dispatch-arg form.
    if (isa<BlockArgument>(peeled))
        return true;

    // Form 1: the scf.for accumulates the sums itself.
    auto forOp = peeled.getDefiningOp<scf::ForOp>();
    if (!forOp)
        return false;
    for (Operation &bodyOp : forOp.getBody()->getOperations()) {
        auto genOp = dyn_cast<linalg::GenericOp>(&bodyOp);
        if (!genOp || genOp.getNumDpsInputs() != 1)
            continue;
        if (peelSlice(genOp.getInputs()[0]) != peelSlice(weights))
            continue;
        auto iterTypes = genOp.getIteratorTypesArray();
        if (iterTypes.size() == 2 && iterTypes[0] == utils::IteratorType::parallel &&
            iterTypes[1] == utils::IteratorType::reduction)
            return true;
    }
    return false;
}

// Matches the combine generic that applies the input zero-point correction to
// the raw i32 accumulator:
//   out = acc - sumW * azp
// with 3 inputs (acc tensor<MxOxi32>, sumW tensor<Oxi32>, azp i32 scalar
// constant) and all-parallel iterators.  On success `inputZp` receives azp.
bool matchZpCorrectionCombine(linalg::GenericOp op, Value acc, Value weights, int32_t &inputZp) {
    if (!op || op.getNumDpsInputs() != 3 || op.getNumDpsInits() != 1)
        return false;
    if (op.getInputs()[0] != acc)
        return false;
    if (!op.isAllParallelLoops())
        return false;
    auto outTy = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!outTy || !outTy.getElementType().isInteger(32))
        return false;

    // Input 1 must be the per-output-channel weight sum (direct or tiled).
    if (!isWeightSumProducer(op.getInputs()[1], weights))
        return false;
    // The sum is over the reduction dim, so its length is the number of output
    // channels. This also bounds the block-argument form, where the producer is
    // not visible inside the extracted dispatch.
    auto sumWTy = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    if (!sumWTy || ShapedType::isDynamic(sumWTy.getShape()[0]) ||
        ShapedType::isDynamic(outTy.getShape()[1]) || sumWTy.getShape()[0] != outTy.getShape()[1])
        return false;

    // Input 2 must be a scalar i32 constant: the input zero-point.
    auto maybeZp = getScalarI32Const(op.getInputs()[2]);
    if (!maybeZp)
        return false;

    // Body: yield( subi(accArg, muli(sumWArg, zpArg)) ).
    auto subi = dyn_cast_or_null<arith::SubIOp>(getYieldedValueDef(op));
    if (!subi)
        return false;
    if (getBlockArgNumber(subi.getLhs(), op) != 0)
        return false;
    auto muli = subi.getRhs().getDefiningOp<arith::MulIOp>();
    if (!muli)
        return false;
    auto lhsNo = getBlockArgNumber(muli.getLhs(), op);
    auto rhsNo = getBlockArgNumber(muli.getRhs(), op);
    if (!lhsNo || !rhsNo)
        return false;
    if (!((*lhsNo == 1 && *rhsNo == 2) || (*lhsNo == 2 && *rhsNo == 1)))
        return false;

    inputZp = *maybeZp;
    return true;
}

// Matches a generic that dequantizes an i32 bias constant to f32 with the
// given scale:  { sitofp, mulf scale } on a single arith.constant input.
// Returns the i32 constant value on success.
Value matchDequantizedBiasConst(Value v, double dequantScale) {
    auto genOp = v.getDefiningOp<linalg::GenericOp>();
    if (!genOp || genOp.getNumDpsInputs() != 1 || genOp.getNumDpsInits() != 1)
        return nullptr;
    if (!genOp.isAllParallelLoops())
        return nullptr;
    Value constOperand = genOp.getInputs()[0];
    // After tiling the bias constant may be read through an extract_slice.
    // Peel it to validate the root constant, but keep the tiled view so the
    // returned value carries the correct (tiled) shape.
    if (!peelSlice(constOperand).getDefiningOp<arith::ConstantOp>())
        return nullptr;
    auto constTy = dyn_cast<RankedTensorType>(constOperand.getType());
    if (!constTy || constTy.getRank() != 1 || !constTy.getElementType().isInteger(32))
        return nullptr;

    auto mulf = dyn_cast_or_null<arith::MulFOp>(getYieldedValueDef(genOp));
    if (!mulf)
        return nullptr;
    Value sitofpVal = nullptr;
    Value scaleVal = nullptr;
    for (Value operand : {mulf.getLhs(), mulf.getRhs()}) {
        if (operand.getDefiningOp<arith::SIToFPOp>())
            sitofpVal = operand;
        else
            scaleVal = operand;
    }
    if (!sitofpVal || !scaleVal)
        return nullptr;
    auto maybeScale = getQGenericFloatConstant(scaleVal, genOp);
    if (!maybeScale)
        return nullptr;
    // Both scales are a_scale * b_scale, but f64->f32 rounding can make the
    // two constants differ slightly. Use a small tolerance.
    if (std::fabs(*maybeScale - dequantScale) > 1e-9 * std::fmax(1.0, std::fabs(dequantScale)))
        return nullptr;
    return constOperand;
}

// Matches an elementwise f32 add where one operand is `chain` and the other is
// a dequantized i32 bias constant.  Returns the bias constant (or nullptr when
// the add does not match).
Value extractBiasFromAddGeneric(linalg::GenericOp op, Value chain, double dequantScale) {
    if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return nullptr;
    if (!op.isAllParallelLoops())
        return nullptr;
    auto addf = dyn_cast_or_null<arith::AddFOp>(getYieldedValueDef(op));
    if (!addf)
        return nullptr;
    auto lhsNo = getBlockArgNumber(addf.getLhs(), op);
    auto rhsNo = getBlockArgNumber(addf.getRhs(), op);
    if (!lhsNo || !rhsNo)
        return nullptr;

    Value other = nullptr;
    if (op.getInputs()[0] == chain && *lhsNo == 0)
        other = op.getInputs()[1];
    else if (op.getInputs()[1] == chain && *rhsNo == 1)
        other = op.getInputs()[0];
    else
        return nullptr;
    // The dequantized bias may go through a shape cast (e.g. [O] -> [1, O])
    // before the add; peel it to reach the dequant generic.
    return matchDequantizedBiasConst(peelShapeCasts(other), dequantScale);
}

// Integer-domain bias add between the accumulator chain and the fused
// rescale:  addi(chain, broadcast(const [O] i32)).  Returns the rank-1 i32
// bias constant (shape casts and yield-only broadcasts are peeled).
static Value extractIntBiasFromAddGeneric(linalg::GenericOp op, Value chain) {
    if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return nullptr;
    if (!op.isAllParallelLoops())
        return nullptr;
    auto addi = dyn_cast_or_null<arith::AddIOp>(getYieldedValueDef(op));
    if (!addi)
        return nullptr;
    auto lhsNo = getBlockArgNumber(addi.getLhs(), op);
    auto rhsNo = getBlockArgNumber(addi.getRhs(), op);
    if (!lhsNo || !rhsNo)
        return nullptr;

    Value other = nullptr;
    if (op.getInputs()[0] == chain && *lhsNo == 0)
        other = op.getInputs()[1];
    else if (op.getInputs()[1] == chain && *rhsNo == 1)
        other = op.getInputs()[0];
    else
        return nullptr;

    Value peeled = peelShapeCasts(other);
    while (true) {
        auto g = peeled.getDefiningOp<linalg::GenericOp>();
        if (!g || g.getNumDpsInputs() != 1 || g.getNumDpsInits() != 1 || !g.isAllParallelLoops() ||
            !g.getRegion().hasOneBlock())
            break;
        auto y = dyn_cast_or_null<linalg::YieldOp>(g.getBody()->getTerminator());
        if (!y || y.getNumOperands() != 1 || !isa<BlockArgument>(y.getOperand(0)))
            break;
        peeled = peelShapeCasts(g.getInputs()[0]);
    }
    // After tiling the bias constant may be read through an extract_slice; the
    // root must be a constant, but the tiled rank-1 [O] view is kept (it is the
    // shape the scale/bias builder expects).
    Value root = peeled;
    while (auto es = root.getDefiningOp<tensor::ExtractSliceOp>())
        root = es.getSource();
    auto cst = root.getDefiningOp<arith::ConstantOp>();
    if (cst && isa<DenseIntElementsAttr>(cst.getValue())) {
        auto ty = dyn_cast<RankedTensorType>(peeled.getType());
        if (ty && ty.getRank() == 1 && ty.getElementType().isInteger(32))
            return peeled;
    }
    return nullptr;
}

struct QMatmulToFCConvert : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

    // Build the {2*O} interleaved scale_bias tensor:
    //   scaleBias[2o]   = bias[o] - inputZp * sum_k(weights[o, k])
    //   scaleBias[2o+1] = multiplier
    // All computation is deferred to the JIT constant pipeline (Host-marked
    // ops + CompileTimeConst).
    Value buildQGemmScaleBias(
        Value bias, Value torqWeights, int32_t inputZp, int32_t multiplier, int64_t O, Location loc,
        PatternRewriter &rewriter
    ) const {
        // Zero bias when no bias add is present.
        if (!bias) {
            auto zeroTy = RankedTensorType::get({O}, rewriter.getI32Type());
            bias = arith::ConstantOp::create(rewriter, loc, zeroTy, rewriter.getI32IntegerAttr(0))
                       .getResult();
        }

        // A size-1 (per-tensor) bias broadcasts across the O channels.
        if (auto biasTy = dyn_cast<RankedTensorType>(bias.getType()))
            if (biasTy.getShape().size() == 1 && biasTy.getShape()[0] == 1 && O > 1) {
                Value root = bias;
                while (auto es = root.getDefiningOp<tensor::ExtractSliceOp>())
                    root = es.getSource();
                if (auto cst = root.getDefiningOp<arith::ConstantOp>())
                    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
                        APInt val = *dense.getValues<APInt>().begin();
                        SmallVector<int32_t> vals(O, static_cast<int32_t>(val.getSExtValue()));
                        bias = arith::ConstantOp::create(
                                   rewriter, loc, RankedTensorType::get({O}, rewriter.getI32Type()),
                                   rewriter.getI32TensorAttr(vals)
                        )
                                   .getResult();
                    }
            }

        // Fold the input zero-point correction into the bias:
        //   correction[o] = -inputZp * sum_k(weights[o, k])
        if (inputZp != 0) {
            Value correction = computeInputZpCorrection(torqWeights, inputZp, rewriter);
            bias = addPerChannelBias(bias, correction, rewriter);
        }

        Value scaleBias = buildDynamicInterleavedBiasScale(bias, multiplier, loc, rewriter);
        if (!scaleBias)
            return nullptr;
        return createCompileTimeConstOp(scaleBias.getDefiningOp(), rewriter).value_or(scaleBias);
    }

    // Emit torq_hl.fully_connected and replace the matched chain.
    LogicalResult rewriteQGemmChain(
        linalg::MatmulOp matmulOp, linalg::GenericOp combineOp, Value weightSum,
        QuantizedOpChain &qChain, linalg::GenericOp biasAddOp, Value input, Value torqWeights,
        Value scaleBias, int32_t inputZp, int32_t shift, PatternRewriter &rewriter
    ) const {
        Location loc = matmulOp.getLoc();
        Value fcInit = qChain.quantOp.getDpsInitOperand(0)->get();
        auto fcOutTy = cast<RankedTensorType>(fcInit.getType());

        int32_t outputZp = static_cast<int32_t>(std::llround(qChain.outputInfo.zp));
        int32_t outputMin = static_cast<int32_t>(std::llround(qChain.outputInfo.min));
        int32_t outputMax = static_cast<int32_t>(std::llround(qChain.outputInfo.max));

        rewriter.setInsertionPoint(qChain.quantOp);
        Value fcResult = torq_hl::FullyConnectedOp::create(
                             rewriter, loc, fcOutTy, fcInit, inputZp, /*weight_zp=*/0, outputZp,
                             outputMin, outputMax, shift, torq_hl::VectorizationModeEnum::None,
                             torqWeights, scaleBias, input
        )
                             .getResult(0);

        // Replace the matched chain; each op is replaced by its own init so
        // remaining (now dead) producers fold away.
        rewriter.replaceOp(matmulOp, matmulOp.getDpsInitOperand(0)->get());
        if (combineOp)
            rewriter.replaceOp(combineOp, combineOp.getDpsInitOperand(0)->get());
        if (qChain.outputDequantOp)
            rewriter.replaceOp(
                qChain.outputDequantOp, qChain.outputDequantOp.getDpsInitOperand(0)->get()
            );
        if (biasAddOp)
            rewriter.replaceOp(biasAddOp, biasAddOp.getDpsInitOperand(0)->get());
        rewriter.replaceOp(qChain.quantOp, fcResult);
        if (weightSum)
            eraseIfDead(weightSum, rewriter);

        LLVM_DEBUG(llvm::dbgs() << "[QMatmulToFCConvert] replaced Q chain with torq_hl.fc\n");
        return success();
    }

  public:
    using OpRewritePattern::OpRewritePattern;
    QMatmulToFCConvert(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::MatmulOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp matmulOp, PatternRewriter &rewriter) const override {
        TorqStructuredOpMatcher<linalg::MatmulOp> matcher;
        if (!matcher.addPredicate(notMarkedFuseGroupIf(_markFuseGroups)).match(matmulOp)) {
            return rewriter.notifyMatchFailure(matmulOp, "Q matmul match failed");
        }

        // Match the chain with the side-effect-free shared matcher (also used
        // by the claim check isQuantizedMatmulChain, so the claim can never
        // outgrow what this pattern converts).
        QMatmulFcChain fc;
        if (!matchQMatmulFcChain(matmulOp, fc)) {
            return rewriter.notifyMatchFailure(matmulOp, "Q matmul chain match failed");
        }
        Value input = fc.input;
        Value rhs = fc.rhs;
        int64_t O = cast<RankedTensorType>(matmulOp.getResult(0).getType()).getShape()[1];

        if (_markFuseGroups) {
            if (!markFuseGroupBackward(
                    fc.qChain.quantOp.getResult(0), {input, rhs}, rewriter,
                    matmulOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        if (fc.qChain.outputInfo.scale == 0.0) {
            return rewriter.notifyMatchFailure(matmulOp, "quant scale is zero");
        }
        double scaleFactor = fc.qChain.inputScale() / fc.qChain.outputInfo.scale;
        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(scaleFactor, multiplier, shift)) {
            return rewriter.notifyMatchFailure(matmulOp, "failed to compute multiplier/shift");
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QMatmulToFCConvert] multiplier=" << multiplier << " shift=" << shift
                         << " inputZp=" << fc.inputZp << "\n"
        );

        // Place the constant chain before quantOp: in tiled variants the bias
        // extract_slice is defined after dequantOp (just before the bias
        // dequant), so inserting earlier would violate dominance for the
        // bias+correction add.
        rewriter.setInsertionPoint(fc.qChain.quantOp);
        Location loc = matmulOp.getLoc();

        // Bring the weights to [O, K] for the zero-point correction below,
        // when the constant was pre-transposed.
        Value weightsOMajor = fc.weightsOMajor;
        if (fc.needsTranspose)
            weightsOMajor =
                transposeValue(weightsOMajor, SmallVector<int64_t, 4>{1, 0}, loc, rewriter);

        ScaleClampInfo dummyScInfo;
        // torq_hl.fully_connected wants its `weights` operand in [K, O]
        // (reduction-dim major), like the Conv1D/Conv2D-as-FC lowerings.
        // `rhs` is already [K, O] per linalg.matmul semantics, so use it as-is.
        Value torqWeights = preConversionWeights(
            rhs, Permutation::none(), /*weightZpV=*/std::nullopt, dummyScInfo, rewriter,
            /*isDepthwise=*/false
        );

        Value scaleBias =
            buildQGemmScaleBias(fc.bias, weightsOMajor, fc.inputZp, multiplier, O, loc, rewriter);
        if (!scaleBias) {
            return rewriter.notifyMatchFailure(matmulOp, "failed to build scale_bias");
        }

        return rewriteQGemmChain(
            matmulOp, fc.combineOp, fc.combineOp ? fc.combineOp.getInputs()[1] : nullptr, fc.qChain,
            fc.biasAddOp, input, torqWeights, scaleBias, fc.inputZp, shift, rewriter
        );
    }
};

} // namespace

// Matches the QDQ matmul chain into `chain` (see QMatmulFcChain for the
// accepted forms): a rank-2 i8 x i8 -> i32 linalg.matmul with a single-use
// result, feeding at most one strict input-zero-point correction combine,
// then either the float form (dequant generic, optional dequantized-bias
// add, quant generic) or the fused integer form (optional integer bias add,
// then a per-tensor apply_scale rescale).  Side-effect free: shared by
// QMatmulToFCConvert and the claim check isQuantizedMatmulChain, so the
// claim can never outgrow what the consumer converts.
bool matchQMatmulFcChain(linalg::MatmulOp matmulOp, QMatmulFcChain &m) {
    m.matmul = matmulOp;
    m.input = matmulOp.getInputs()[0];
    m.rhs = matmulOp.getInputs()[1];

    auto inTy = dyn_cast<RankedTensorType>(m.input.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(m.rhs.getType());
    auto outTy = dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
    if (!inTy || !rhsTy || !outTy || inTy.getRank() != 2 || rhsTy.getRank() != 2 ||
        outTy.getRank() != 2)
        return false;
    if (!inTy.getElementType().isInteger(8) || !rhsTy.getElementType().isInteger(8) ||
        !outTy.getElementType().isInteger(32))
        return false;
    int64_t O = outTy.getShape()[1];
    int64_t K = inTy.getShape()[1];

    // Recover an [O, K] view of the weights, needed below for the input
    // zero-point correction (reduces over K, keeps O). The decomposition
    // transposes constant [O, K] weights to [K, O] for the matmul; peel
    // the transpose if it survived, otherwise transpose rhs back below.
    m.weightsOMajor = m.rhs;
    if (auto transposeOp = m.rhs.getDefiningOp<linalg::TransposeOp>())
        m.weightsOMajor = transposeOp.getInput();
    auto weightsTy = cast<RankedTensorType>(m.weightsOMajor.getType());
    m.needsTranspose = weightsTy.getShape()[0] != O;
    if (m.needsTranspose && (weightsTy.getShape()[1] != O || weightsTy.getShape()[0] != K))
        return false;

    // Optional zp-correction combine after the matmul.
    Value chain = matmulOp.getResult(0);
    if (auto user = getSingleGenericUser(chain);
        user && matchZpCorrectionCombine(user, chain, m.rhs, m.inputZp)) {
        m.combineOp = user;
        chain = user.getResult(0);
    }
    else if (!chain.hasOneUse())
        return false;
    m.chain = chain;

    // Float form: dequant -> [dequantized-bias add] -> quant.
    if (matchOutputDequantizationPure(chain, m.qChain)) {
        if (m.qChain.outputDequantInfo.zp != 0)
            return false;
        Value dqResult = m.qChain.outputDequantOp.getResult(0);
        Value tail = dqResult;
        if (auto user = getSingleGenericUser(dqResult)) {
            if (Value biasConst =
                    extractBiasFromAddGeneric(user, dqResult, m.qChain.outputDequantInfo.scale)) {
                m.biasAddOp = user;
                m.bias = biasConst;
                tail = user.getResult(0);
            }
        }
        return matchOutputQuantizationPure(tail, m.qChain);
    }

    // Fused integer rescale form (tosa-style): the i32 accumulator is
    // rescaled directly into the output type by one generic (apply_scale +
    // zp + clamp + trunc), with an optional integer bias add between the
    // correction and the rescale.  There is no float dequant stage: the i32
    // domain is already the dequantized domain, so the chain is recorded with
    // an identity output dequant and the rescale generic as the quant.  The
    // correction term may arrive as a dispatch block argument when the
    // slicing plan extracted the weight column-sum reduce; the rewrite
    // recomputes it from the weight constant.
    m.fusedIntForm = true;
    Value cur = chain;
    if (auto user = getSingleGenericUser(cur)) {
        if (Value biasConst = extractIntBiasFromAddGeneric(user, cur)) {
            m.biasAddOp = user;
            m.bias = biasConst;
            cur = user.getResult(0);
        }
    }
    if (!cur.hasOneUse())
        return false;
    auto rescale = dyn_cast<linalg::GenericOp>(*cur.getUsers().begin());
    if (!rescale)
        return false;
    Value rescaled = cur;
    ScaleClampInfo scInfo = foldForwardScaleClamp(
        rescaled, /*scaleValuesCount=*/1, /*shift8b=*/0, /*shift16b=*/0,
        /*isElementWiseOp=*/false
    );
    if (!scInfo || rescaled.getDefiningOp() != rescale)
        return false;
    if (scInfo.scaleDouble.empty() || scInfo.scaleDouble[0] == 0.0)
        return false;
    m.qChain.outputDequantInfo = DequantInfo{/*scale=*/1.0, /*zp=*/0};
    m.qChain.outputDequantOp = nullptr;
    m.qChain.quantOp = rescale;
    // Express the tosa multiplier/shift in the ONNX-style output scale:
    // y = x / s_out + zp with s_out = 2^shift / multiplier, so the
    // inputScale()/outputInfo.scale pair recovers it.
    m.qChain.outputInfo = QuantInfo{
        /*scale=*/1.0 / scInfo.scaleDouble[0], /*zp=*/static_cast<double>(scInfo.zp),
        /*min=*/static_cast<double>(scInfo.min), /*max=*/static_cast<double>(scInfo.max)
    };
    return true;
}

bool isExtractedWeightSum(Value v) {
    // The inline form is a linalg.generic reduction over the weight constant,
    // which the fusion walk traces; the extracted forms the walk cannot trace
    // are the dispatch block argument and the LRAM-tiled reductions (the
    // linalg.reduce over the scf.for partials table, or the scf.for
    // self-accumulate form - see isWeightSumProducer).
    Value peeled = peelSlice(v);
    if (peeled.getDefiningOp<linalg::GenericOp>())
        return false;
    if (isa<BlockArgument>(peeled))
        return true;
    if (auto reduceOp = peeled.getDefiningOp<linalg::ReduceOp>())
        return reduceOp.getNumDpsInputs() == 1;
    return peeled.getDefiningOp<scf::ForOp>() != nullptr;
}

void populateLinalgToTorqHLQMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QMatmulToFCConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
