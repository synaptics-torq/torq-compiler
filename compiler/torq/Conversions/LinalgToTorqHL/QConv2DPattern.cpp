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

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#define DEBUG_TYPE "linalg-torq-q-conv2d-pattern"

namespace mlir::syna::torq {

namespace {

static bool isElementwiseAddI32(linalg::GenericOp op) {
    if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;
    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;
    auto addi = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();
    if (!addi)
        return false;
    auto lhs = dyn_cast<BlockArgument>(addi.getLhs());
    auto rhs = dyn_cast<BlockArgument>(addi.getRhs());
    if (!lhs || !rhs)
        return false;
    return lhs.getOwner() == op.getBody() && rhs.getOwner() == op.getBody();
}

Value extractBiasOperand(linalg::GenericOp addOp, Value convOutput) {
    Value biasOperand =
        addOp->getOperand(0) == convOutput ? addOp->getOperand(1) : addOp->getOperand(0);
    if (auto broadcastOp = biasOperand.getDefiningOp<linalg::BroadcastOp>())
        biasOperand = broadcastOp.getInput();
    return biasOperand;
}

struct QConv2dConvert : public OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
  private:
    const bool _markFuseGroups;

    // Match the dequant -> quant chain that follows the conv (or the optional
    // bias add).  On success `dequantOp` and `quantOp` are populated together
    // with their scales/zero-points.
    LogicalResult matchQConv2DQuantChain(
        linalg::Conv2DNchwFchwQOp convOp, Value chain, PatternRewriter &rewriter,
        linalg::GenericOp &dequantOp, double &dequantScale, linalg::GenericOp &quantOp,
        double &quantScale, double &quantZp, double &quantMin, double &quantMax
    ) const {
        if (!chain.hasOneUse()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[QConv2dConvert] chain does not have single use before dequant, uses: "
                << std::distance(chain.getUsers().begin(), chain.getUsers().end()) << "\n"
            );
            return rewriter.notifyMatchFailure(
                convOp, "chain does not have single use before dequant"
            );
        }
        dequantOp = dyn_cast<linalg::GenericOp>(*chain.getUsers().begin());
        int32_t ignoredDequantZp = 0;
        if (!dequantOp || !matchDequantGeneric(dequantOp, dequantScale, ignoredDequantZp)) {
            if (dequantOp)
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] failed to match Q dequant, user: "
                                 << *chain.getUsers().begin() << "\n"
                );
            return rewriter.notifyMatchFailure(convOp, "failed to match Q dequant");
        }
        Value dequantResult = dequantOp->getResult(0);
        if (!dequantResult.hasOneUse()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] dequant result has "
                             << std::distance(
                                    dequantResult.getUsers().begin(), dequantResult.getUsers().end()
                                )
                             << " uses\n"
            );
            return rewriter.notifyMatchFailure(convOp, "dequant result has multiple uses");
        }

        quantOp = dyn_cast<linalg::GenericOp>(*dequantResult.getUsers().begin());
        if (!quantOp || !matchQuantGeneric(quantOp, quantScale, quantZp, quantMin, quantMax)) {
            if (quantOp)
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] failed to match Q quant, user: "
                                 << *dequantResult.getUsers().begin() << "\n"
                );
            return rewriter.notifyMatchFailure(convOp, "failed to match Q quant");
        }
        return success();
    }

    // Build the {C*2} interleaved scale_bias tensor for the conv.  Uses the
    // same deferred constant resolution pipeline as regular Conv2D:
    // createCompileTimeConstOp → CompileTimeConstOutlinePass →
    // CompileTimeConstComputePass.
    Value buildQConv2DScaleBias(
        linalg::Conv2DNchwFchwQOp convOp, RankedTensorType convOutTy, Value bias, Value torqWeights,
        int32_t inputZp, int32_t multiplier, PatternRewriter &rewriter
    ) const {
        Location loc = convOp.getLoc();

        // Create a zero bias when no bias add is present.
        if (!bias) {
            int64_t C = convOutTy.getShape()[1];
            auto zeroTy = RankedTensorType::get({C}, rewriter.getI32Type());
            bias = arith::ConstantOp::create(rewriter, loc, zeroTy, rewriter.getI32IntegerAttr(0))
                       .getResult();
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] using zero bias\n");
        }

        // If the bias comes from a quant generic, mark it Host so it
        // survives pre-conversion and is resolved later by
        // CompileTimeConstComputePass.  We do NOT wrap it here
        // to avoid nested CompileInputToConstOp wrappers that cause
        // "not already in an operation block" assertions.
        if (auto biasGeneric = dyn_cast<linalg::GenericOp>(bias.getDefiningOp()))
            setTargetExecutorAttr(biasGeneric, torq_hl::Executor::Host);
        auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
        if (!biasTy || biasTy.getRank() != 1 || biasTy.getElementType() != rewriter.getI32Type()) {
            return nullptr;
        }

        if (inputZp != 0) {
            // Compute per-channel -inputZp * sum(weights) correction.
            // The correction ops are marked Host by computeInputZpCorrection
            // and survive pre-conversion.  Only the final scale_bias is
            // wrapped with CompileInputToConstOp (see below).
            Value correction = computeInputZpCorrection(torqWeights, inputZp, rewriter);

            // Add correction to bias.  addPerChannelBias marks the add Host.
            bias = addPerChannelBias(bias, correction, rewriter);
        }

        // Interleave bias values with multiplier.
        Value scaleBias = buildDynamicInterleavedBiasScale(bias, multiplier, loc, rewriter);
        auto sbFolded = createCompileTimeConstOp(scaleBias.getDefiningOp(), rewriter);
        if (succeeded(sbFolded))
            return *sbFolded;
        return scaleBias;
    }

    LogicalResult rewriteQConv2DChain(
        linalg::Conv2DNchwFchwQOp convOp, linalg::GenericOp addOp, linalg::GenericOp dequantOp,
        linalg::GenericOp quantOp, Value input, Value torqWeights, Value scaleBias,
        const PaddingInfo &padInfo, int32_t inputZp, int32_t multiplier, int32_t shift,
        double quantZp, double quantMin, double quantMax, PatternRewriter &rewriter
    ) const {
        Location loc = convOp.getLoc();

        if (auto padOp = convOp->getOperand(0).getDefiningOp<tensor::PadOp>()) {
            if (llvm::all_of(padInfo.lrtbPad, [](int64_t p) { return p == 0; })) {
                return rewriter.notifyMatchFailure(convOp, "failed to fold padding");
            }
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] padding info lrtb=[" << padInfo.lrtbPad[0] << ","
                         << padInfo.lrtbPad[1] << "," << padInfo.lrtbPad[2] << ","
                         << padInfo.lrtbPad[3] << "]\n"
        );

        auto dilations = convOp.getDilations().getValues<int64_t>();
        std::vector<int64_t> finalDilationVec(dilations.begin(), dilations.end());
        Value dilatedWeights = torqWeights;
        auto mWts = getDilatedWts(torqWeights, finalDilationVec, /*isDW1DStride1=*/false, rewriter);
        if (!failed(mWts))
            dilatedWeights = *mWts;

        std::vector<int64_t> strideVec = attrValuesAsVec(convOp.getStrides());

        Value convInit = quantOp.getDpsInitOperand(0)->get();
        auto outTy = cast<RankedTensorType>(convInit.getType());

        int32_t outputZp = static_cast<int32_t>(std::llround(quantZp));
        int32_t outputMin = static_cast<int32_t>(std::llround(quantMin));
        int32_t outputMax = static_cast<int32_t>(std::llround(quantMax));

        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] creating torq_hl.conv2d, outTy=" << outTy << "\n"
        );
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(quantOp);

            Value convResult = torq_hl::Conv2DOp::create(
                                   rewriter, loc, outTy, convInit, inputZp, /*weight_zp=*/0,
                                   outputZp, outputMin, outputMax, shift,
                                   /*groups=*/1, padInfo.lrtbPad, strideVec, finalDilationVec,
                                   torq_hl::VectorizationModeEnum::None, dilatedWeights, scaleBias,
                                   input, /*nhwc_input=*/false
            )
                                   .getResult(0);

            rewriter.replaceOp(convOp, convOp.getDpsInitOperand(0)->get());
            if (addOp)
                rewriter.replaceOp(addOp, addOp.getDpsInitOperand(0)->get());
            rewriter.replaceOp(dequantOp, dequantOp.getDpsInitOperand(0)->get());
            rewriter.replaceOp(quantOp, convResult);
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] replaced Q chain with torq_hl.conv2d\n");
        }

        return success();
    }

  public:
    using OpRewritePattern::OpRewritePattern;
    QConv2dConvert(MLIRContext *context, bool markFuseGroups, PatternBenefit benefit = 1)
        : OpRewritePattern<linalg::Conv2DNchwFchwQOp>(context, benefit),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::Conv2DNchwFchwQOp convOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] matching conv: " << convOp << "\n");
        TorqStructuredOpMatcher<linalg::Conv2DNchwFchwQOp> matcher;
        if (!matcher.addPredicate(notMarkedFuseGroupIf(_markFuseGroups)).match(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Q conv match failed");
        }

        Value convOutput = convOp->getResult(0);
        auto convOutTy = dyn_cast<RankedTensorType>(convOutput.getType());
        if (!convOutTy || !convOutTy.getElementType().isInteger(32)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] expected i32 conv output, got: "
                             << convOutput.getType() << "\n"
            );
            return rewriter.notifyMatchFailure(convOp, "expected i32 conv output");
        }

        auto maybeInputZp = getScalarI32Const(convOp->getOperand(2));
        auto maybeWeightZp = getScalarI32Const(convOp->getOperand(3));
        if (!maybeInputZp || !maybeWeightZp) {
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] input/weight zp not constant scalar\n");
            return rewriter.notifyMatchFailure(convOp, "input/weight zp not constant scalar");
        }
        int32_t inputZp = *maybeInputZp;
        int32_t weightZp = *maybeWeightZp;

        Value bias = nullptr;
        linalg::GenericOp addOp = nullptr;
        Value chain = convOutput;
        if (convOutput.hasOneUse()) {
            auto user = dyn_cast<linalg::GenericOp>(*convOutput.getUsers().begin());
            if (user && isElementwiseAddI32(user)) {
                addOp = user;
                bias = extractBiasOperand(user, convOutput);
                chain = user->getResult(0);
                LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] found bias add\n");
            }
            else {
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] conv output user is not add: "
                                 << *convOutput.getUsers().begin() << "\n"
                );
            }
        }
        else {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] conv output has "
                             << std::distance(
                                    convOutput.getUsers().begin(), convOutput.getUsers().end()
                                )
                             << " uses\n"
            );
        }

        linalg::GenericOp dequantOp;
        double dequantScale;
        linalg::GenericOp quantOp;
        double quantScale, quantZp, quantMin, quantMax;
        if (failed(matchQConv2DQuantChain(
                convOp, chain, rewriter, dequantOp, dequantScale, quantOp, quantScale, quantZp,
                quantMin, quantMax
            ))) {
            return failure();
        }

        Value input = convOp->getOperand(0);
        Value weightsValue = convOp->getOperand(1);
        Value output = quantOp->getResult(0);

        // Fold backward padding so the pad op (if any) is included in the fusion
        // group and the conv can use hardware padding.
        PaddingInfo padInfo = foldBackwardPadding(input, rewriter, /*nchw=*/true);

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, weightsValue}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        if (quantScale == 0.0) {
            return rewriter.notifyMatchFailure(convOp, "quant scale is zero");
        }
        double scaleFactor = dequantScale / quantScale;
        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(scaleFactor, multiplier, shift)) {
            return rewriter.notifyMatchFailure(convOp, "failed to compute multiplier/shift");
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] multiplier=" << multiplier << " shift=" << shift
                         << "\n"
        );

        std::optional<Value> optionalWeightZpV;
        if (weightZp != 0)
            optionalWeightZpV = convOp->getOperand(3);
        ScaleClampInfo dummyScInfo;
        Value torqWeights = preConversionWeights(
            weightsValue, Permutation::none(), optionalWeightZpV, dummyScInfo, rewriter,
            /*isDepthwise=*/false
        );

        // Place correction chain + interleave before dequantOp, which is
        // guaranteed to be after both the Q conv and the bias add.  In tiled
        // variants the bias extract_slice appears after the Q conv; placing
        // here avoids a dominance violation for the correction chain's use of
        // the bias.
        rewriter.setInsertionPoint(dequantOp);

        Value scaleBias = buildQConv2DScaleBias(
            convOp, convOutTy, bias, torqWeights, inputZp, multiplier, rewriter
        );
        if (!scaleBias) {
            return rewriter.notifyMatchFailure(convOp, "failed to build scale_bias");
        }

        return rewriteQConv2DChain(
            convOp, addOp, dequantOp, quantOp, input, torqWeights, scaleBias, padInfo, inputZp,
            multiplier, shift, quantZp, quantMin, quantMax, rewriter
        );
    }
};

} // namespace

void populateLinalgToTorqHLQConv2DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QConv2dConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
