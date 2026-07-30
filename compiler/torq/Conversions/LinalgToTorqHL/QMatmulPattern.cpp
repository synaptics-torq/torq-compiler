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

// Erase the op producing `v` when it has no remaining users.  Used for the
// original weight-sum producer (direct reduce or tiled scf.for): it is dead
// once the zp correction is folded into scale_bias, and scf ops are not legal
// in the pre-conversion target.
void eraseIfDead(Value v, PatternRewriter &rewriter) {
    Operation *op = v.getDefiningOp();
    if (op && op->use_empty())
        rewriter.eraseOp(op);
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
// over the K dim (TileReductionForLram):
//   %r = scf.for ... iter_args(%acc = %zero) {
//     %slice = tensor.extract_slice %weights[k, 0] [t, O] ...
//     %sum = linalg.generic ... ins(%slice) ... {extsi, addi}
//     %ins = tensor.insert_slice %sum into %acc ...
//     scf.yield %ins }
// Accept both the direct and the tiled producer form.
bool isWeightSumProducer(Value v, Value weights) {
    auto ty = dyn_cast<RankedTensorType>(v.getType());
    if (!ty || ty.getRank() != 1 || !ty.getElementType().isInteger(32))
        return false;
    if (auto genOp = v.getDefiningOp<linalg::GenericOp>())
        return isWeightSumReduce(genOp, weights);
    auto forOp = v.getDefiningOp<scf::ForOp>();
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

struct QMatmulToFCConvert : public OpRewritePattern<linalg::MatmulOp> {
  private:
    const bool _markFuseGroups;

    // Match the quantized matmul output rescale chain:
    //   matmul (i32) -> [zp-correction combine] -> DQ (f32) -> [bias add] -> Q (i8)
    // The caller handles the optional zp-correction combine before calling this
    // helper.  This helper uses the split output-DQ/output-Q matchers so it can
    // inspect and consume the optional bias add between the DQ and Q.
    LogicalResult matchQGemmQuantChain(
        linalg::MatmulOp matmulOp, Value chain, PatternRewriter &rewriter, QuantizedOpChain &qChain,
        linalg::GenericOp &biasAddOp, Value &bias
    ) const {
        if (failed(matchOutputDequantization(chain, rewriter, qChain)))
            return rewriter.notifyMatchFailure(matmulOp, "failed to match Q dequant");

        if (qChain.outputDequantInfo.zp != 0)
            return rewriter.notifyMatchFailure(matmulOp, "output DQ zero-point must be zero");

        Value dqResult = qChain.outputDequantOp.getResult(0);
        Value tail = dqResult;
        if (auto user = getSingleGenericUser(dqResult)) {
            if (Value biasConst =
                    extractBiasFromAddGeneric(user, dqResult, qChain.outputDequantInfo.scale)) {
                biasAddOp = user;
                bias = biasConst;
                tail = user.getResult(0);
            }
        }

        if (failed(matchOutputQuantization(tail, rewriter, qChain)))
            return rewriter.notifyMatchFailure(matmulOp, "failed to match Q quant");
        return success();
    }

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

        Value input = matmulOp.getInputs()[0];
        Value rhs = matmulOp.getInputs()[1];

        auto inTy = dyn_cast<RankedTensorType>(input.getType());
        auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
        auto outTy = dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
        if (!inTy || !rhsTy || !outTy || inTy.getRank() != 2 || rhsTy.getRank() != 2 ||
            outTy.getRank() != 2) {
            return rewriter.notifyMatchFailure(matmulOp, "expects rank-2 tensors");
        }
        if (!inTy.getElementType().isInteger(8) || !rhsTy.getElementType().isInteger(8) ||
            !outTy.getElementType().isInteger(32)) {
            return rewriter.notifyMatchFailure(matmulOp, "expects i8 x i8 -> i32 matmul");
        }
        int64_t O = outTy.getShape()[1];
        int64_t K = inTy.getShape()[1];

        // Canonicalize weights to the [O, K] orientation expected by the FC op.
        // The decomposition transposes constant [O, K] weights to [K, O] for
        // the matmul.  When the transpose survives as an op, peel it to recover
        // the original layout; when it was folded into the constant, transpose
        // back during the rewrite.
        Value weights = rhs;
        if (auto transposeOp = rhs.getDefiningOp<linalg::TransposeOp>())
            weights = transposeOp.getInput();
        auto weightsTy = cast<RankedTensorType>(weights.getType());
        bool transposeWeights = weightsTy.getShape()[0] != O;
        if (transposeWeights && (weightsTy.getShape()[1] != O || weightsTy.getShape()[0] != K)) {
            return rewriter.notifyMatchFailure(matmulOp, "unexpected weight layout");
        }

        // Optional zp-correction combine after the matmul.
        int32_t inputZp = 0;
        linalg::GenericOp combineOp = nullptr;
        Value weightSum = nullptr;
        Value chain = matmulOp.getResult(0);
        if (auto user = getSingleGenericUser(chain);
            user && matchZpCorrectionCombine(user, chain, rhs, inputZp)) {
            combineOp = user;
            weightSum = user.getInputs()[1];
            chain = user.getResult(0);
        }
        else if (!chain.hasOneUse()) {
            return rewriter.notifyMatchFailure(matmulOp, "matmul result has multiple uses");
        }

        QuantizedOpChain qChain;
        linalg::GenericOp biasAddOp = nullptr;
        Value bias = nullptr;
        if (failed(matchQGemmQuantChain(matmulOp, chain, rewriter, qChain, biasAddOp, bias))) {
            return failure();
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                qChain.quantOp.getResult(0), {input, rhs}, rewriter,
                matmulOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        if (qChain.outputInfo.scale == 0.0) {
            return rewriter.notifyMatchFailure(matmulOp, "quant scale is zero");
        }
        double scaleFactor = qChain.inputScale() / qChain.outputInfo.scale;
        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(scaleFactor, multiplier, shift)) {
            return rewriter.notifyMatchFailure(matmulOp, "failed to compute multiplier/shift");
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QMatmulToFCConvert] multiplier=" << multiplier << " shift=" << shift
                         << " inputZp=" << inputZp << "\n"
        );

        // Place the constant chain before quantOp: in tiled variants the bias
        // extract_slice is defined after dequantOp (just before the bias
        // dequant), so inserting earlier would violate dominance for the
        // bias+correction add.
        rewriter.setInsertionPoint(qChain.quantOp);
        Location loc = matmulOp.getLoc();

        // Bring the weights to [O, K] when the constant was pre-transposed.
        if (transposeWeights)
            weights = transposeValue(weights, SmallVector<int64_t, 4>{1, 0}, loc, rewriter);

        ScaleClampInfo dummyScInfo;
        Value torqWeights = preConversionWeights(
            weights, Permutation::none(), /*weightZpV=*/std::nullopt, dummyScInfo, rewriter,
            /*isDepthwise=*/false
        );

        Value scaleBias =
            buildQGemmScaleBias(bias, torqWeights, inputZp, multiplier, O, loc, rewriter);
        if (!scaleBias) {
            return rewriter.notifyMatchFailure(matmulOp, "failed to build scale_bias");
        }

        return rewriteQGemmChain(
            matmulOp, combineOp, weightSum, qChain, biasAddOp, input, torqWeights, scaleBias,
            inputZp, shift, rewriter
        );
    }
};

} // namespace

void populateLinalgToTorqHLQMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QMatmulToFCConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
