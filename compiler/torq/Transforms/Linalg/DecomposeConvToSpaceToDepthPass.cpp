// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// DecomposeConvToSpaceToDepth Pass
//===----------------------------------------------------------------------===//
//
// Decomposes linalg.conv_2d_nchw_fchw ops with large kernels (kH > 7 or
// kW > 7 or stride > 2) into an equivalent SpaceToDepth reshape followed by a convolution
// with a smaller kernel that fits within the HW limit (max 7x7, stride <= 2).
//
// Block sizes bH and bW are chosen independently per axis as the largest
// divisor of (kDim, stride, inputDim) such that kDim/b <= 7 and stride/b <= 2.
//
// The H axis is left native (bH = 1) only for a fully valid (2,1) conv (stride
// (2,1), kernel <= 7, no border pad on either spatial axis) — the one
// stride-2 mix verified to lower correctly on the HW in this tree without a
// data movement. Every other axis keeps the baseline block search (folding
// when a valid block exists). Leaving any other stride-2 axis native would be
// a pure loss: the S2D transpose lowers to a real runtime slice task, and the
// non-(2,1) stride-2/pad combinations are either mis-computed or rejected by
// the HW lowering, so those axes must be folded.
// The pattern fires only when at least one axis requires decomposition.
//
// PRECONDITIONS (per axis, independently for H and W):
//   - kDim % b == 0  AND  stride % b == 0  AND  inputDim % b == 0
//   - kDim / b <= 7   AND  stride / b <= 2
//   - dilation == 1  (dilated convolutions not supported)
//
// TRANSFORMATION:
//
//   BEFORE:
//     input  : [N, C,   H,   W  ]   (NCHW)
//     filter : [F, C,   kH,  kW ]   (FCHW)
//     conv stride [sH, sW]
//     output : [N, F,   oH,  oW ]
//
//   AFTER:
//     s2d_input  = SpaceToDepth(input,  bH, bW) : [N, C*bH*bW, H/bH,  W/bW ]
//     s2d_filter = SpaceToDepth(filter, bH, bW) : [F, C*bH*bW, kH/bH, kW/bW]
//     conv(s2d_input, s2d_filter, stride=[sH/bH, sW/bW]) : [N, F, oH, oW]
//
// SpaceToDepth is implemented as expand -> transpose -> collapse:
//   [N, C, H, W]
//     -> expand  [N, C, H/bH, bH, W/bW, bW]
//     -> transpose perm [0,1,3,5,2,4]
//        -> [N, C, bH, bW, H/bH, W/bW]
//     -> collapse [N, C*bH*bW, H/bH, W/bW]
//
// The same transform applied to both input and filter ensures the channel
// ordering is consistent and the convolution result is numerically identical.
//
// EXAMPLE:
//   input   [1, 1, 40, 80],  filter [64, 1, 4, 10],  stride [1, 2]
//   bH = 1  (kH=4 already <= 7, no height decomposition)
//   bW = 2  (kW=10 > 7; largest b dividing kW=10, sW=2, W=80 with kW/b<=7)
//   s2d_input  [1, 2, 40, 40],  s2d_filter [64, 2, 4, 5],  stride [1, 1]
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/LayoutTransformUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-conv-to-space-to-depth"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Chooses the largest block size `b` that can be folded into channels.
/// A valid `b` must divide the kernel size, stride, and input size, and after
/// decomposition the reduced kernel and stride must fit the HW limits
/// [kernel size <=7 and stride <= 2]:
///   kDim % b == 0, stride % b == 0, inputDim % b == 0
///   kDim / b <= maxKernel, stride / b <= 2
/// Returns 1 when no useful decomposition is possible.
/// Example: if kDim = 10, stride = 4, inputDim = 80, maxKernel = 7,
/// then b = 2 is valid because 10 % 2 == 0, 4 % 2 == 0, 80 % 2 == 0,
/// kDim / b = 5 <= 7, and stride / b = 2 <= 2.
static int64_t selectBlock(
    int64_t kDim, int64_t stride, int64_t inputDim, int64_t maxKernel, int64_t maxStride = 2
) {
    if (kDim <= maxKernel && stride <= maxStride)
        return 1; // already within HW limit, no decomposition needed

    int64_t best = 1;
    // Try divisors of kDim in descending order
    for (int64_t b = kDim; b >= 2; --b) {
        if (kDim % b != 0)
            continue;
        if (stride % b != 0)
            continue;
        if (inputDim % b != 0)
            continue;
        if (kDim / b > maxKernel)
            continue; // reduction not sufficient
        if (stride / b > maxStride)
            continue; // reduction not sufficient (only s1 & s2 are supported by hardware)
        best = b;
        break;
    }
    return best;
}

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

/// The static border pad visible on one spatial axis of the conv input.
///
/// A stride-2 axis is only kept native (not S2D-folded) for a fully valid
/// (2,1) conv; see the gate in DecomposeConvWithSpaceToDepthPattern.
/// Anything that looks like a border on the kept axes forces the baseline
/// folded decomposition.
struct AxisPadInfo {
    enum class Kind {
        None,     // no border pad on this axis (valid conv)
        Explicit, // static pad resolved: (before, after)
        Unknown,  // a border-adding op is present but not statically resolvable
    } kind = Kind::None;
    int64_t before = 0, after = 0;
};

/// Resolves the static border pad on conv-input axis `dim` (2 = H, 3 = W)
/// for the pad forms that reach this pass:
///   - a rank-4 tensor.pad feeding the conv (2D conv),
///   - a tensor.pad under a unit-dim expand_shape (conv1d as conv2d),
///   - a materialized pad: insert_slice into a zero-filled tensor.
/// An extract_slice hiding a border, or dynamic pads, yield Unknown.
static AxisPadInfo inputAxisPad(Value input, int64_t dim) {
    if (auto padOp = input.getDefiningOp<tensor::PadOp>()) {
        auto srcType = cast<RankedTensorType>(padOp.getSource().getType());
        if (srcType.getRank() != 4)
            return {AxisPadInfo::Kind::Unknown, 0, 0};
        auto low = getConstantIntValue(padOp.getMixedLowPad()[dim]);
        auto high = getConstantIntValue(padOp.getMixedHighPad()[dim]);
        if (!low || !high)
            return {AxisPadInfo::Kind::Unknown, 0, 0};
        return {AxisPadInfo::Kind::Explicit, *low, *high};
    }
    if (auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>()) {
        if (auto padOp = expandOp.getSrc().getDefiningOp<tensor::PadOp>()) {
            auto resultShape = cast<RankedTensorType>(expandOp.getResult().getType()).getShape();
            for (auto [srcDim, group] : llvm::enumerate(expandOp.getReassociationIndices())) {
                if (!llvm::is_contained(group, dim))
                    continue;
                // The border of a source dim lands on the single non-unit
                // result dim of its group; unit result dims carry no pad.
                int64_t nonUnitDim = -1;
                for (int64_t d : group) {
                    if (resultShape[d] == 1)
                        continue;
                    if (nonUnitDim != -1)
                        return {AxisPadInfo::Kind::Unknown, 0, 0};
                    nonUnitDim = d;
                }
                auto low = getConstantIntValue(padOp.getMixedLowPad()[srcDim]);
                auto high = getConstantIntValue(padOp.getMixedHighPad()[srcDim]);
                if (!low || !high)
                    return {AxisPadInfo::Kind::Unknown, 0, 0};
                if (nonUnitDim == -1 && (*low != 0 || *high != 0))
                    // A non-zero pad on a source dim that expands to an all-unit
                    // group is malformed for our purposes; be conservative.
                    return {AxisPadInfo::Kind::Unknown, 0, 0};
                return {
                    AxisPadInfo::Kind::Explicit, nonUnitDim == dim ? *low : 0,
                    nonUnitDim == dim ? *high : 0
                };
            }
            return {AxisPadInfo::Kind::Unknown, 0, 0};
        }
    }
    if (auto insertOp = input.getDefiningOp<tensor::InsertSliceOp>()) {
        if (insertOp.getDest().getDefiningOp<linalg::FillOp>()) {
            auto offsets = insertOp.getStaticOffsets();
            auto sizes = insertOp.getStaticSizes();
            auto strides = insertOp.getStaticStrides();
            auto destShape = cast<RankedTensorType>(insertOp.getDest().getType()).getShape();
            if (offsets.size() != 4 || sizes.size() != 4 ||
                llvm::any_of(offsets, ShapedType::isDynamic) ||
                llvm::any_of(sizes, ShapedType::isDynamic) ||
                llvm::any_of(strides, [](int64_t s) { return s != 1; }))
                return {AxisPadInfo::Kind::Unknown, 0, 0};
            return {
                AxisPadInfo::Kind::Explicit, offsets[dim],
                destShape[dim] - sizes[dim] - offsets[dim]
            };
        }
    }
    if (input.getDefiningOp<tensor::ExtractSliceOp>())
        return {AxisPadInfo::Kind::Unknown, 0, 0};
    return {AxisPadInfo::Kind::None, 0, 0};
}

/// Whether the axis carries a border pad (an unknown border counts as one).
static bool hasBorderPad(const AxisPadInfo &pad) {
    return pad.kind == AxisPadInfo::Kind::Unknown ||
           (pad.kind == AxisPadInfo::Kind::Explicit && (pad.before != 0 || pad.after != 0));
}

/// Rewrites linalg.conv_2d_nchw_fchw with a large kernel by inserting
/// SpaceToDepth on the input and reusing the same transform on the filter.
///
/// Both input  [N, C,  H,  W ]  and filter [F, C, kH, kW] are 4-D NCHW-like
/// tensors, so torq::getSpaceToDepth() applies identically to both:
///   input  -> [N, C*bH*bW, H/bH,  W/bW ]
///   filter -> [F, C*bH*bW, kH/bH, kW/bW]
/// The channel ordering is consistent, so the convolution result is unchanged.
/// No constant-only restriction — the filter can be any Value.
struct DecomposeConvWithSpaceToDepthPattern : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv2DNchwFchwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        Value input = convOp.getInputs()[0];   // [N, C, H, W]    NCHW
        Value filter = convOp.getInputs()[1];  // [F, C, kH, kW]  FCHW
        Value output = convOp.getOutputs()[0]; // [N, F, oH, oW]  NFHW

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        ArrayRef<int64_t> inShape = inputType.getShape();   // N C H W
        ArrayRef<int64_t> fltShape = filterType.getShape(); // F C kH kW

        if (inShape.size() != 4 || fltShape.size() != 4)
            return rewriter.notifyMatchFailure(convOp, "unexpected tensor rank");

        const int64_t H = inShape[2];
        const int64_t W = inShape[3];
        const int64_t kH = fltShape[2];
        const int64_t kW = fltShape[3];

        // Strides and dilations
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();
        int64_t sH = stridesAttr.getValues<int64_t>()[0];
        int64_t sW = stridesAttr.getValues<int64_t>()[1];
        int64_t dH = dilationsAttr.getValues<int64_t>()[0];
        int64_t dW = dilationsAttr.getValues<int64_t>()[1];

        // Only handle unit dilation — dilated convs require separate algorithm.
        // TODO: expandthefilter by inserting zeros to support dilation > 1.
        if (dH != 1 || dW != 1)
            return rewriter.notifyMatchFailure(convOp, "dilation != 1 not supported");

        // only s1 and s2 are supported by hardware
        int64_t maxStride = 2;

        if (kH <= HwInfo::nss_max_kernel_height && kW <= HwInfo::nss_max_kernel_width &&
            sH <= maxStride && sW <= maxStride && sH == sW)
            return rewriter.notifyMatchFailure(
                convOp, "no S2D decomposition needed (kernel and stride already within HW limit)"
            );
        // Determine per-axis block sizes independently BEFORE mutating IR.
        // Max decompose stride is set to 1
        maxStride = 1;
        // The H axis may stay at its native stride 2 only when the result is a
        // fully valid (2,1) conv: the stride-2 quadrant lowering is verified
        // for that shape, and mis-computes or aborts on every other stride-2
        // mix — any border pad on a kept axis (the quadrant phase ignores it),
        // a kept W axis under sH == 1 (the stride-1 path requires the W pad to
        // equal the W border), or a kept H axis over a padded stride-1 W. A
        // folded axis absorbs its border pad into data, so only the kept
        // axes' pads matter. (2,2) in-limits convs never reach here (early
        // exit above). Everything else keeps the baseline per-axis block
        // search, so this change only removes the H-axis over-fold.
        bool hMustFold = kH > HwInfo::nss_max_kernel_height || sH > 2;
        bool wMustFold = kW > HwInfo::nss_max_kernel_width || sW > 2;
        AxisPadInfo hPad = inputAxisPad(input, 2);
        AxisPadInfo wPad = inputAxisPad(input, 3);
        bool hKeepNative =
            !hMustFold && sH == 2 && !hasBorderPad(hPad) && (wMustFold || !hasBorderPad(wPad));
        bool hFold = !hKeepNative;
        bool wFold = true; // baseline: the block search always ran per axis
        const int64_t pHs = (H + sH - 1) / sH * sH;
        const int64_t pWs = (W + sW - 1) / sW * sW;

        // Predict padded kernel sizes (same as weights_pad_with_zero) without emitting ops.
        const int64_t paddedKH = align_ceil(kH, sH);
        const int64_t paddedKW = align_ceil(kW, sW);
        int64_t bW =
            wFold ? selectBlock(paddedKW, sW, pWs, HwInfo::nss_max_kernel_width, maxStride) : 1;
        int64_t bH =
            hFold ? selectBlock(paddedKH, sH, pHs, HwInfo::nss_max_kernel_height, maxStride) : 1;

        if (bH == 1 && bW == 1)
            return rewriter.notifyMatchFailure(
                convOp, "no S2D decomposition possible (kernel already within HW limit "
                        "or divisibility conditions not met)"
            );

        // Actual pad targets: multiples of the block, not of the stride, so a
        // non-folded axis (b == 1) stays completely unpadded. Padding a native
        // axis to a multiple of the stride would leave a zero-padded kernel
        // row/column behind, which the HW lowering cannot support.
        const int64_t pH = (H + bH - 1) / bH * bH;
        const int64_t pW = (W + bW - 1) / bW * bW;
        bool widthPad = pW != W;
        bool heightPad = pH != H;

        LLVM_DEBUG(
            llvm::dbgs() << "[S2D] Decomposing conv: kH=" << kH << " kW=" << kW << " sH=" << sH
                         << " sW=" << sW << " -> bH=" << bH << " bW=" << bW << "\n"
        );

        // Pad weights only after we know decomposition will proceed.
        // Pad by the block (b == s on a folded axis with maxStride = 1, a no-op
        // on a non-folded axis) so the pad targets match the input padding.
        filter = weights_pad_with_zero(rewriter, loc, filter, 2, bH);
        filter = weights_pad_with_zero(rewriter, loc, filter, 3, bW);

        // ------------------------------------------------------------------ //
        // 1. SpaceToDepth on input:  [N, C, H, W] -> [N, C*bH*bW, H/bH, W/bW]
        // 2. SpaceToDepth on filter: [F, C, kH, kW] treated as [n=F, c=C, h=kH, w=kW]
        //                         -> [F, C*bH*bW, kH/bH, kW/bW]
        //
        // Both use getSpaceToDepth which interprets dim layout as [N, C, H, W].
        // The channel ordering (c * bH*bW + bh * bW + bw) is identical for both,
        // so the convolution dot product is unchanged.
        // ------------------------------------------------------------------ //
        Value s2dFilter = getSpaceToDepth(filter, bH, bW, rewriter);
        s2dFilter =
            createCompileTimeConstOp(s2dFilter.getDefiningOp(), rewriter).value_or(s2dFilter);

        // If input needs height or width padding (odd/unaligned dims), create a
        // zero-filled tensor of the aligned shape and insert the original input
        if (heightPad || widthPad) {
            auto curInType = cast<RankedTensorType>(input.getType());
            ArrayRef<int64_t> curShape = curInType.getShape();
            SmallVector<int64_t> paddedInputShape = {curShape[0], curShape[1], pH, pW};
            Value paddedInput =
                createZeroFilledTensor(rewriter, loc, paddedInputShape, curInType.getElementType());
            SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
            SmallVector<OpFoldResult> sizes =
                getAsIndexOpFoldResult(rewriter.getContext(), curShape);
            SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
            input = tensor::InsertSliceOp::create(
                        rewriter, loc, input, paddedInput, offsets, sizes, strides
            )
                        .getResult();
        }

        Value s2dInput = getSpaceToDepth(input, bH, bW, rewriter);

        // ------------------------------------------------------------------ //
        // 3. New conv with reduced kernel and absorbed stride.
        //    stride: [sH/bH, sW/bW]
        //
        // Build the fusion plan BEFORE emitting anything so that we know which
        // ops (bias, rescale, clamp, …) are fused onto the conv result.  The
        // plan walks forward from `output` and collects single-use elementwise
        // ops.  We then rebuild the entire chain with bigger tensors (when the
        // input was padded) and do exactly ONE extract_slice after the last
        // fused op — so no fusion patterns see an extract_slice between them.
        // ------------------------------------------------------------------ //
        int64_t newSH = sH / bH;
        int64_t newSW = sW / bW;
        auto newStridesAttr = rewriter.getI64TensorAttr({newSH, newSW});

        auto s2dInShape = cast<RankedTensorType>(s2dInput.getType()).getShape();
        auto s2dFltShape = cast<RankedTensorType>(s2dFilter.getType()).getShape();
        ArrayRef<int64_t> origOutShape = outputType.getShape(); // [N, F, oH, oW]

        // Compute actual output spatial dims after the padded S2D conv.
        int64_t newOH = (s2dInShape[2] - s2dFltShape[2]) / newSH + 1;
        int64_t newOW = (s2dInShape[3] - s2dFltShape[3]) / newSW + 1;
        bool outputPadded = (newOH != origOutShape[2] || newOW != origOutShape[3]);

        // Collect the fusion chain (bias / rescale / clamp …) rooted at the
        // original conv output.  We need this list BEFORE rewriting.
        Value fusionAnchor = output;
        FailureOr<FusionPlan> fusionPlanOr = buildFusionPlanAndRebindOutput(fusionAnchor);
        SmallVector<Operation *> fusedOps;
        if (succeeded(fusionPlanOr) && fusionPlanOr->isFusable())
            fusedOps = fusionPlanOr->neededOps;

        if (!outputPadded) {
            // ---- Fast path: no padding, shapes unchanged -----------------
            Value newConvResult = linalg::Conv2DNchwFchwOp::create(
                                      rewriter, loc, outputType, ValueRange{s2dInput, s2dFilter},
                                      ValueRange{output}, newStridesAttr, dilationsAttr
            )
                                      ->getResult(0);
            rewriter.replaceOp(convOp, newConvResult);
            return success();
        }

        // ---- Padded path: rebuild conv + fused ops with bigger shapes ----
        //
        // a) Conv into bigOutput.
        SmallVector<int64_t> bigOutShape = {origOutShape[0], origOutShape[1], newOH, newOW};
        Value bigOutput =
            createZeroFilledTensor(rewriter, loc, bigOutShape, outputType.getElementType());
        auto bigOutType = cast<RankedTensorType>(bigOutput.getType());
        Value currentResult = linalg::Conv2DNchwFchwOp::create(
                                  rewriter, loc, bigOutType, ValueRange{s2dInput, s2dFilter},
                                  ValueRange{bigOutput}, newStridesAttr, dilationsAttr
        )
                                  ->getResult(0);

        // b) Rebuild only the forward chain that consumes the conv result.
        //    Auxiliary fused ops such as weight-side or constant-side producers
        //    remain untouched even if they are part of the broader fusion plan.
        llvm::SmallPtrSet<Operation *, 8> fusedOpSet(fusedOps.begin(), fusedOps.end());
        Operation *terminalOp = convOp.getOperation();
        Value prevOrigResult = convOp->getResult(0);
        while (true) {
            Operation *nextOp = nullptr;
            for (Operation *user : prevOrigResult.getUsers()) {
                if (fusedOpSet.contains(user)) {
                    nextOp = user;
                    break;
                }
            }
            if (!nextOp)
                break;

            auto genericOp = dyn_cast<linalg::GenericOp>(nextOp);
            if (!genericOp)
                break;

            SmallVector<Value> newInputs;
            for (Value operand : genericOp.getInputs())
                newInputs.push_back(operand == prevOrigResult ? currentResult : operand);

            Value bigInit =
                createZeroFilledTensor(rewriter, loc, bigOutShape, outputType.getElementType());
            SmallVector<AffineMap> newMaps = genericOp.getIndexingMapsArray();
            currentResult =
                rebuildGenericWithNewLayout(rewriter, genericOp, newInputs, bigInit, newMaps);
            terminalOp = nextOp;
            prevOrigResult = nextOp->getResult(0);
        }

        // c) extract_slice from the final (possibly fused) big result back to
        //    the original output shape — this is the only extract_slice.
        SmallVector<OpFoldResult> sliceOffsets(4, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sliceSizes =
            getAsIndexOpFoldResult(rewriter.getContext(), origOutShape);
        SmallVector<OpFoldResult> sliceStrides(4, rewriter.getIndexAttr(1));
        Value finalResult =
            tensor::ExtractSliceOp::create(
                rewriter, loc, outputType, currentResult, sliceOffsets, sliceSizes, sliceStrides
            )
                .getResult();

        // Replace the terminal op in the conv-result chain so downstream users
        // see the sliced result shape.
        rewriter.replaceOp(terminalOp, finalResult);
        if (terminalOp != convOp.getOperation())
            rewriter.replaceOp(convOp, convOp->getResult(0)); // keep SSA valid
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DecomposeConvToSpaceToDepthPass
    : public impl::DecomposeConvToSpaceToDepthBase<DecomposeConvToSpaceToDepthPass> {

    void runOnOperation() override {
        auto func = getOperation();
        MLIRContext *ctx = func->getContext();

        RewritePatternSet patterns(ctx);
        patterns.add<DecomposeConvWithSpaceToDepthPattern>(ctx);

        if (failed(applyPatternsGreedily(func, std::move(patterns))))
            return signalPassFailure();
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDecomposeConvToSpaceToDepthPass() {
    return std::make_unique<DecomposeConvToSpaceToDepthPass>();
}

} // namespace mlir::syna::torq
