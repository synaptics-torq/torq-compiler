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
#include <tuple>

#define DEBUG_TYPE "linalg-torq-q-conv2d-pattern"

namespace mlir::syna::torq {

namespace {

bool isElementwiseAddI32(linalg::GenericOp op) {
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

// Set the insertion point to after the latest defining op among the given
// values, or after fallbackOp when no value has a defining op in the same
// block.  This ensures every consumed value dominates the ops created next.
void setInsertionPointAfterLatest(
    PatternRewriter &rewriter, Operation *fallbackOp, ArrayRef<Value> values
) {
    Operation *latest = fallbackOp;
    for (Value v : values) {
        if (!v)
            continue;
        Operation *def = v.getDefiningOp();
        if (!def)
            continue; // block argument or constant already dominates
        if (def->getBlock() != latest->getBlock())
            continue; // cannot compare across blocks; keep current latest
        if (latest->isBeforeInBlock(def))
            latest = def;
    }
    rewriter.setInsertionPointAfter(latest);
}

// Build the {C*2} interleaved scale_bias tensor for a Q conv/depthwise conv.
// Uses the same deferred constant resolution pipeline as regular Conv2D:
// createCompileTimeConstOp -> CompileTimeConstOutlinePass ->
// CompileTimeConstComputePass.
Value buildQConvScaleBias(
    Location loc, RankedTensorType convOutTy, Value bias, Value torqWeights, int32_t inputZp,
    int32_t multiplier, PatternRewriter &rewriter
) {
    // Create a zero bias when no bias add is present.
    if (!bias) {
        int64_t C = convOutTy.getShape()[1];
        auto zeroTy = RankedTensorType::get({C}, rewriter.getI32Type());
        bias = arith::ConstantOp::create(rewriter, loc, zeroTy, rewriter.getI32IntegerAttr(0))
                   .getResult();
        LLVM_DEBUG(llvm::dbgs() << "[QConvScaleBias] using zero bias\n");
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

// Shared helper: match the output rescale chain for a quantized conv/depthwise
// conv: (i32 result) -> DQ -> Q.  The caller is responsible for walking past
// any allowed intermediate ops (bias add, collapse_shape, etc.) to obtain the
// value that feeds the DQ.
LogicalResult
matchConvRescaleChain(Value chain, PatternRewriter &rewriter, QuantizedOpChain &qChain) {
    if (failed(matchOutputDequantization(chain, rewriter, qChain)))
        return failure();
    if (failed(matchOutputQuantization(qChain.outputDequantOp->getResult(0), rewriter, qChain)))
        return failure();
    return success();
}

// Validate that the quantized conv has an i32 accumulator and extract the
// scalar input/weight zero points.
LogicalResult getConvQuantizationParams(
    Operation *op, PatternRewriter &rewriter, int32_t &inputZp, int32_t &weightZp
) {
    auto convOutTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!convOutTy || !convOutTy.getElementType().isInteger(32)) {
        return rewriter.notifyMatchFailure(op, "expected i32 conv output");
    }
    auto maybeInputZp = getScalarI32Const(op->getOperand(2));
    auto maybeWeightZp = getScalarI32Const(op->getOperand(3));
    if (!maybeInputZp || !maybeWeightZp) {
        return rewriter.notifyMatchFailure(op, "input/weight zp not constant scalar");
    }
    inputZp = *maybeInputZp;
    weightZp = *maybeWeightZp;
    return success();
}

// Compute the final multiplier/shift from the quant chain's effective input
// scale and output scale.
LogicalResult computeConvMultiplierAndShift(
    Operation *op, const QuantizedOpChain &qChain, PatternRewriter &rewriter, int32_t &multiplier,
    int32_t &shift
) {
    if (qChain.outputInfo.scale == 0.0) {
        return rewriter.notifyMatchFailure(op, "quant scale is zero");
    }
    double scaleFactor = qChain.inputScale() / qChain.outputInfo.scale;
    if (!computeMultiplierAndShift(scaleFactor, multiplier, shift)) {
        return rewriter.notifyMatchFailure(op, "failed to compute multiplier/shift");
    }
    return success();
}

// Round the quant chain's output zp/min/max to i32 for the torq_hl op.
std::tuple<int32_t, int32_t, int32_t> getRoundedOutputQuantParams(const QuantizedOpChain &qChain) {
    return {
        static_cast<int32_t>(std::llround(qChain.outputInfo.zp)),
        static_cast<int32_t>(std::llround(qChain.outputInfo.min)),
        static_cast<int32_t>(std::llround(qChain.outputInfo.max))
    };
}

// Apply pre-conversion weight processing (permutation, weight-zp correction,
// sign-adjustment) for a quantized conv/depthwise conv.
Value prepareQConvWeights(
    Operation *op, Value weights, int32_t weightZp, bool isDepthwise, PatternRewriter &rewriter
) {
    std::optional<Value> optionalWeightZpV;
    if (weightZp != 0)
        optionalWeightZpV = op->getOperand(3);
    ScaleClampInfo dummyScInfo;
    return preConversionWeights(
        weights, Permutation::none(), optionalWeightZpV, dummyScInfo, rewriter, isDepthwise
    );
}

// Build the scale_bias tensor and place it after the latest of its operands
// so every consumed value dominates it.
Value buildQConvScaleBiasWithPlacement(
    PatternRewriter &rewriter, Location loc, RankedTensorType convOutTy, Value bias,
    Value torqWeights, int32_t inputZp, int32_t multiplier, Operation *insertAfter
) {
    setInsertionPointAfterLatest(rewriter, insertAfter, {bias, torqWeights});
    return buildQConvScaleBias(loc, convOutTy, bias, torqWeights, inputZp, multiplier, rewriter);
}

std::vector<int64_t> getQConvStrides(Operation *op) {
    if (auto nchw = dyn_cast<linalg::Conv2DNchwFchwQOp>(op))
        return attrValuesAsVec(nchw.getStrides());
    return attrValuesAsVec(cast<linalg::Conv2DNgchwGfchwQOp>(op).getStrides());
}

std::vector<int64_t> applyQConvDilations(Operation *op, Value &weights, PatternRewriter &rewriter) {
    std::vector<int64_t> dilations;
    if (auto nchw = dyn_cast<linalg::Conv2DNchwFchwQOp>(op)) {
        auto vals = nchw.getDilations().getValues<int64_t>();
        dilations.assign(vals.begin(), vals.end());
    }
    else {
        auto vals = cast<linalg::Conv2DNgchwGfchwQOp>(op).getDilations().getValues<int64_t>();
        dilations.assign(vals.begin(), vals.end());
    }
    auto mWts = getDilatedWts(weights, dilations, /*isDW1DStride1=*/false, rewriter);
    if (!failed(mWts))
        weights = *mWts;
    return dilations;
}

struct QConv2DConvert : public OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
  private:
    const bool _markFuseGroups;

    LogicalResult rewriteQConv2DChain(
        linalg::Conv2DNchwFchwQOp convOp, linalg::GenericOp addOp, QuantizedOpChain &qChain,
        Value input, Value torqWeights, Value scaleBias, const PaddingInfo &padInfo,
        int32_t inputZp, int32_t shift, PatternRewriter &rewriter
    ) const {
        Location loc = convOp.getLoc();

        if (auto padOp = convOp->getOperand(0).getDefiningOp<tensor::PadOp>()) {
            if (llvm::all_of(padInfo.lrtbPad, [](int64_t p) { return p == 0; })) {
                return rewriter.notifyMatchFailure(convOp, "failed to fold padding");
            }
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2DConvert] padding info lrtb=[" << padInfo.lrtbPad[0] << ","
                         << padInfo.lrtbPad[1] << "," << padInfo.lrtbPad[2] << ","
                         << padInfo.lrtbPad[3] << "]\n"
        );

        Value dilatedWeights = torqWeights;
        std::vector<int64_t> finalDilationVec =
            applyQConvDilations(convOp, dilatedWeights, rewriter);
        std::vector<int64_t> strideVec = getQConvStrides(convOp);

        Value convInit = qChain.quantOp.getDpsInitOperand(0)->get();
        auto outTy = cast<RankedTensorType>(convInit.getType());

        auto [outputZp, outputMin, outputMax] = getRoundedOutputQuantParams(qChain);

        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2DConvert] creating torq_hl.conv2d, outTy=" << outTy << "\n"
        );
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(qChain.quantOp);

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
            rewriter.replaceOp(
                qChain.outputDequantOp, qChain.outputDequantOp.getDpsInitOperand(0)->get()
            );
            rewriter.replaceOp(qChain.quantOp, convResult);
            LLVM_DEBUG(llvm::dbgs() << "[QConv2DConvert] replaced Q chain with torq_hl.conv2d\n");
        }

        return success();
    }

  public:
    using OpRewritePattern::OpRewritePattern;
    QConv2DConvert(MLIRContext *context, bool markFuseGroups, PatternBenefit benefit = 1)
        : OpRewritePattern<linalg::Conv2DNchwFchwQOp>(context, benefit),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::Conv2DNchwFchwQOp convOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[QConv2DConvert] matching conv: " << convOp << "\n");
        TorqStructuredOpMatcher<linalg::Conv2DNchwFchwQOp> matcher;
        if (!matcher.addPredicate(notMarkedFuseGroupIf(_markFuseGroups)).match(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Q conv match failed");
        }

        int32_t inputZp, weightZp;
        if (failed(getConvQuantizationParams(convOp, rewriter, inputZp, weightZp)))
            return failure();

        Value bias = nullptr;
        linalg::GenericOp addOp = nullptr;
        Value chain = convOp->getResult(0);
        if (chain.hasOneUse()) {
            auto user = dyn_cast<linalg::GenericOp>(*chain.getUsers().begin());
            if (user && isElementwiseAddI32(user)) {
                addOp = user;
                bias = extractBiasOperand(user, chain);
                chain = user->getResult(0);
                LLVM_DEBUG(llvm::dbgs() << "[QConv2DConvert] found bias add\n");
            }
            else {
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2DConvert] conv output user is not add: "
                                 << *chain.getUsers().begin() << "\n"
                );
            }
        }
        else {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2DConvert] conv output has "
                             << std::distance(chain.getUsers().begin(), chain.getUsers().end())
                             << " uses\n"
            );
        }

        QuantizedOpChain qChain;
        if (failed(matchConvRescaleChain(chain, rewriter, qChain))) {
            return rewriter.notifyMatchFailure(convOp, "failed to match conv Q chain");
        }

        Value input = convOp->getOperand(0);
        Value weightsValue = convOp->getOperand(1);
        Value output = qChain.quantOp->getResult(0);

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

        int32_t multiplier, shift;
        if (failed(computeConvMultiplierAndShift(convOp, qChain, rewriter, multiplier, shift)))
            return failure();

        Value torqWeights =
            prepareQConvWeights(convOp, weightsValue, weightZp, /*isDepthwise=*/false, rewriter);

        auto convOutTy = dyn_cast<RankedTensorType>(convOp->getResult(0).getType());
        Value scaleBias = buildQConvScaleBiasWithPlacement(
            rewriter, convOp.getLoc(), convOutTy, bias, torqWeights, inputZp, multiplier,
            qChain.outputDequantOp
        );
        if (!scaleBias) {
            return rewriter.notifyMatchFailure(convOp, "failed to build scale_bias");
        }

        return rewriteQConv2DChain(
            convOp, addOp, qChain, input, torqWeights, scaleBias, padInfo, inputZp, shift, rewriter
        );
    }
};

// Quantized depthwise convolution: linalg::Conv2DNgchwGfchwQOp
//
// ONNX/Torch lowers depthwise QDQ convolutions as a grouped convolution with
// group == input channels.  The linalg op carries 5-D shapes:
//   input  [N, G, C/G, H, W]   with C/G == 1
//   filter [G, F/G, KH, KW]    with F/G == 1
//   output [N, G, F/G, H, W]   with F/G == 1
//
// We collapse the unit group/channel dims and lower to
// torq_hl::DepthwiseConv2DOp.

bool isDepthwiseNgchwGfchwQ(linalg::Conv2DNgchwGfchwQOp op) {
    auto inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto wTy = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getDpsInitOperand(0)->get().getType());
    if (!inTy || !wTy || !outTy)
        return false;
    if (inTy.getRank() != 5 || wTy.getRank() != 5 || outTy.getRank() != 5)
        return false;
    return inTy.getShape()[2] == 1 && wTy.getShape()[1] == 1 && outTy.getShape()[2] == 1 &&
           inTy.getShape()[1] == wTy.getShape()[0] && inTy.getShape()[1] == outTy.getShape()[1];
}

// The bias is folded into the conv init as:
//   expand_shape(broadcast(quant_generic(dequant_generic(bias_constant))))
Value extractDepthwiseBias(linalg::Conv2DNgchwGfchwQOp op) {
    Value init = op.getDpsInitOperand(0)->get();
    if (auto expand = init.getDefiningOp<tensor::ExpandShapeOp>())
        init = expand.getSrc();
    if (auto broadcast = init.getDefiningOp<linalg::BroadcastOp>())
        init = broadcast.getInput();
    auto biasTy = dyn_cast<RankedTensorType>(init.getType());
    if (!biasTy || biasTy.getRank() != 1 || !biasTy.getElementType().isInteger(32))
        return nullptr;
    return init;
}

bool adjustSliceForFoldedPad(
    int64_t &off, int64_t &size, int64_t paddedSize, int32_t &frontPad, int32_t &backPad,
    llvm::StringRef dimName
) {
    // Back edge (bottom/right padding).
    if (off + size <= paddedSize - backPad) {
        backPad = 0;
    }
    else if (off + size == paddedSize) {
        size -= backPad;
    }
    else {
        LLVM_DEBUG(
            llvm::dbgs() << "[prepareDepthwiseInput] unexpected " << dimName << " back padding\n"
        );
        return false;
    }

    // Front edge (top/left padding).
    if (off >= frontPad) {
        off -= frontPad;
        frontPad = 0;
    }
    else if (off == 0) {
        size -= frontPad;
    }
    else {
        LLVM_DEBUG(
            llvm::dbgs() << "[prepareDepthwiseInput] unexpected " << dimName << " front padding\n"
        );
        return false;
    }

    return true;
}

// Recover a 4-D NCHW input from the 5-D expanded form and fold any padding.
// Handles both the un-tiled case (input is the result of the expand_shape)
// and the tiled case (input is an extract_slice of the expanded tensor).
Value prepareDepthwiseInput(
    linalg::Conv2DNgchwGfchwQOp op, Value outputValue, PatternRewriter &rewriter,
    PaddingInfo &padInfo
) {
    Value input = op.getInputs()[0];
    auto inputTy = dyn_cast<RankedTensorType>(input.getType());
    if (!inputTy || inputTy.getRank() != 5)
        return nullptr;

    // Non-tiled path: the conv input is produced directly by an expand_shape.
    if (auto expand = input.getDefiningOp<tensor::ExpandShapeOp>()) {
        Value input4D = expand.getSrc();
        padInfo = foldBackwardPadding(input4D, rewriter, /*nchw=*/true, outputValue);
        if (input4D.getDefiningOp<tensor::PadOp>())
            return nullptr;
        return input4D;
    }

    // Tiled path: the conv input is an extract_slice of a 5-D expanded tensor.
    // Fold the padding on the full 4-D padded tensor first, then translate the
    // 5-D slice into a 4-D slice on the unpadded source.  The channel offset is
    // preserved as a mixed value so that channel-tiled convs are handled
    // correctly.
    if (auto extractSlice = input.getDefiningOp<tensor::ExtractSliceOp>()) {
        Value expanded = extractSlice.getSource();
        auto expand = expanded.getDefiningOp<tensor::ExpandShapeOp>();
        if (!expand)
            return nullptr;
        Value padded4D = expand.getSrc();
        auto expandedSrcTy = dyn_cast<RankedTensorType>(expanded.getType());
        if (!expandedSrcTy || expandedSrcTy.getRank() != 5)
            return nullptr;

        // Fold padding on the full padded 4-D tensor, not on the tile.
        padInfo = foldBackwardPadding(padded4D, rewriter, /*nchw=*/true, outputValue);
        if (padded4D.getDefiningOp<tensor::PadOp>())
            return nullptr;

        auto mixedOffsets = extractSlice.getMixedOffsets();
        auto mixedSizes = extractSlice.getMixedSizes();
        auto mixedStrides = extractSlice.getMixedStrides();
        if (mixedOffsets.size() != 5 || mixedSizes.size() != 5 || mixedStrides.size() != 5)
            return nullptr;

        auto staticOffsets5D = extractSlice.getStaticOffsets();
        auto staticSizes5D = extractSlice.getStaticSizes();
        // Spatial offsets/sizes must be static; the channel offset/size may be
        // dynamic when the conv is tiled over the channel dimension.
        if (ShapedType::isDynamic(staticOffsets5D[3]) ||
            ShapedType::isDynamic(staticOffsets5D[4]) || ShapedType::isDynamic(staticSizes5D[3]) ||
            ShapedType::isDynamic(staticSizes5D[4]))
            return nullptr;

        int64_t hOff = staticOffsets5D[3];
        int64_t wOff = staticOffsets5D[4];
        int64_t hSize = staticSizes5D[3];
        int64_t wSize = staticSizes5D[4];
        int64_t paddedH = expandedSrcTy.getShape()[3];
        int64_t paddedW = expandedSrcTy.getShape()[4];

        int32_t left = padInfo.lrtbPad[0];
        int32_t right = padInfo.lrtbPad[1];
        int32_t top = padInfo.lrtbPad[2];
        int32_t bottom = padInfo.lrtbPad[3];

        // Translate the slice from padded-tensor coordinates into
        // unpadded-source coordinates and rewrite the whole-tensor padding
        // into the per-tile padding this slice actually needs.
        if (!adjustSliceForFoldedPad(hOff, hSize, paddedH, top, bottom, "H") ||
            !adjustSliceForFoldedPad(wOff, wSize, paddedW, left, right, "W"))
            return nullptr;

        if (hSize <= 0 || wSize <= 0) {
            LLVM_DEBUG(llvm::dbgs() << "[prepareDepthwiseInput] non-positive spatial slice size\n");
            return nullptr;
        }

        padInfo.lrtbPad[0] = left;
        padInfo.lrtbPad[1] = right;
        padInfo.lrtbPad[2] = top;
        padInfo.lrtbPad[3] = bottom;

        // 5-D layout: [N, G, 1, H, W] -> 4-D layout: [N, C, H, W] with C == G.
        SmallVector<OpFoldResult> offsets4D{
            mixedOffsets[0], mixedOffsets[1], rewriter.getIndexAttr(hOff),
            rewriter.getIndexAttr(wOff)
        };
        SmallVector<OpFoldResult> sizes4D{
            mixedSizes[0], mixedSizes[1], rewriter.getIndexAttr(hSize), rewriter.getIndexAttr(wSize)
        };
        SmallVector<OpFoldResult> strides4D{
            mixedStrides[0], mixedStrides[1], mixedStrides[3], mixedStrides[4]
        };

        SmallVector<int64_t> sliceShape4D{
            ShapedType::isDynamic(staticSizes5D[0]) ? ShapedType::kDynamic : staticSizes5D[0],
            ShapedType::isDynamic(staticSizes5D[1]) ? ShapedType::kDynamic : staticSizes5D[1],
            hSize, wSize
        };
        auto slice4DTy = RankedTensorType::get(sliceShape4D, inputTy.getElementType());
        Value input4D =
            tensor::ExtractSliceOp::create(
                rewriter, input.getLoc(), slice4DTy, padded4D, offsets4D, sizes4D, strides4D
            )
                .getResult();

        if (input4D.getDefiningOp<tensor::PadOp>())
            return nullptr;
        return input4D;
    }

    // Fallback: collapse a 5-D input that is not produced by expand_shape.
    SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}, {3}, {4}};
    SmallVector<int64_t> shape4D{
        inputTy.getShape()[0], inputTy.getShape()[1] * inputTy.getShape()[2], inputTy.getShape()[3],
        inputTy.getShape()[4]
    };
    Value input4D = tensor::CollapseShapeOp::create(
                        rewriter, input.getLoc(),
                        RankedTensorType::get(shape4D, inputTy.getElementType()), input, reassoc
    )
                        .getResult();
    padInfo = foldBackwardPadding(input4D, rewriter, /*nchw=*/true, outputValue);
    if (input4D.getDefiningOp<tensor::PadOp>())
        return nullptr;
    return input4D;
}

// Collapse the 5-D filter [G,1,1,KH,KW] to the 3-D depthwise form [G,KH,KW].
Value collapseDepthwiseWeights(Value weights, PatternRewriter &rewriter) {
    auto wTy = dyn_cast<RankedTensorType>(weights.getType());
    if (!wTy || wTy.getRank() != 5)
        return nullptr;
    auto shape = wTy.getShape();
    if (shape[1] != 1 || shape[2] != 1)
        return nullptr;
    auto loc = weights.getLoc();
    auto collapsedTy = RankedTensorType::get({shape[0], shape[3], shape[4]}, wTy.getElementType());
    SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2, 3}, {4}};
    return tensor::CollapseShapeOp::create(rewriter, loc, collapsedTy, weights, reassoc)
        .getResult();
}

struct QDepthwiseConv2DConvert : public OpRewritePattern<linalg::Conv2DNgchwGfchwQOp> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern::OpRewritePattern;
    QDepthwiseConv2DConvert(MLIRContext *context, bool markFuseGroups, PatternBenefit benefit = 1)
        : OpRewritePattern<linalg::Conv2DNgchwGfchwQOp>(context, benefit),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::Conv2DNgchwGfchwQOp convOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[QDepthwiseConv2DConvert] matching conv: " << convOp << "\n");

        TorqStructuredOpMatcher<linalg::Conv2DNgchwGfchwQOp> matcher;
        if (!matcher.addPredicate(notMarkedFuseGroupIf(_markFuseGroups)).match(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Q depthwise conv match failed");
        }

        if (!isDepthwiseNgchwGfchwQ(convOp))
            return rewriter.notifyMatchFailure(convOp, "not a depthwise grouped conv");

        int32_t inputZp, weightZp;
        if (failed(getConvQuantizationParams(convOp, rewriter, inputZp, weightZp)))
            return failure();

        Value bias = extractDepthwiseBias(convOp);

        QuantizedOpChain qChain;
        Value chain = convOp->getResult(0);
        if (chain.hasOneUse()) {
            if (auto collapse = dyn_cast<tensor::CollapseShapeOp>(*chain.getUsers().begin()))
                chain = collapse.getResult();
        }
        if (failed(matchConvRescaleChain(chain, rewriter, qChain))) {
            return rewriter.notifyMatchFailure(convOp, "failed to match depthwise Q chain");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                qChain.quantOp->getResult(0), {convOp.getInputs()[0], convOp.getInputs()[1]},
                rewriter, convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        int32_t multiplier, shift;
        if (failed(computeConvMultiplierAndShift(convOp, qChain, rewriter, multiplier, shift)))
            return failure();

        Value outputInit = qChain.quantOp.getDpsInitOperand(0)->get();

        PaddingInfo padInfo;
        Value input = prepareDepthwiseInput(convOp, outputInit, rewriter, padInfo);
        if (!input)
            return rewriter.notifyMatchFailure(convOp, "failed to prepare depthwise input");

        Value weights = collapseDepthwiseWeights(convOp.getInputs()[1], rewriter);
        if (!weights)
            return rewriter.notifyMatchFailure(convOp, "failed to collapse depthwise weights");

        Value torqWeights =
            prepareQConvWeights(convOp, weights, weightZp, /*isDepthwise=*/true, rewriter);

        auto convOutTy = dyn_cast<RankedTensorType>(convOp->getResult(0).getType());
        Value scaleBias = buildQConvScaleBiasWithPlacement(
            rewriter, convOp.getLoc(), convOutTy, bias, torqWeights, inputZp, multiplier,
            qChain.outputDequantOp
        );
        if (!scaleBias)
            return rewriter.notifyMatchFailure(convOp, "failed to build scale_bias");

        auto outTy = cast<RankedTensorType>(outputInit.getType());
        auto [outputZp, outputMin, outputMax] = getRoundedOutputQuantParams(qChain);

        Value dilatedWeights = torqWeights;
        std::vector<int64_t> finalDilationVec =
            applyQConvDilations(convOp, dilatedWeights, rewriter);
        std::vector<int64_t> strideVec = getQConvStrides(convOp);

        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(qChain.quantOp);
            int32_t groups = static_cast<int32_t>(convOutTy.getShape()[1]);
            Value convResult =
                torq_hl::DepthwiseConv2DOp::create(
                    rewriter, convOp.getLoc(), outTy, outputInit, inputZp,
                    /*weight_zp=*/0, outputZp, outputMin, outputMax, shift, groups, padInfo.lrtbPad,
                    strideVec, finalDilationVec, torq_hl::VectorizationModeEnum::None,
                    dilatedWeights, scaleBias, input, /*nhwc_input=*/false,
                    /*segment_output=*/false, /*is_dw1d_stride1=*/false
                )
                    .getResult(0);
            rewriter.replaceOp(convOp, convOp.getDpsInitOperand(0)->get());
            rewriter.replaceOp(qChain.quantOp, convResult);
        }
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLQConv2DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QConv2DConvert>(context, markFuseGroups);
    patterns.insert<QDepthwiseConv2DConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
