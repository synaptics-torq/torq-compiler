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
// The pattern fires only when at least one axis requires decomposition, and
// only when the fold is not beaten by the alternative: if the conv would
// take DecomposeConvsForHardwarePass' residue split instead, that form is
// cheaper when it moves fewer bytes ((N+2)*I + W + (4N-2)*O versus the
// fold's 3*I + W*(E/k) + O - the zero-padded kernel) - in practice tiny
// inputs with large F*C - and it has a bf16/f16 output (f32 merge adds
// fall to the host under the default f32 fallback; quantized chains risk
// the channel-sliced per-channel-rescale CSS cliff).  See
// residueSplitCheaperThanS2dFold.
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

#include "Patterns.h"

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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <limits>

#define DEBUG_TYPE "torq-decompose-conv-to-space-to-depth"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Strategy override for the S2D gate (A/B testing; "auto" runs the cost
/// comparison in residueSplitCheaperThanS2dFold).
static llvm::cl::opt<std::string> clS2DStrategy(
    "torq-s2d-strategy",
    llvm::cl::desc("Strategy for large-stride/large-kernel convs: auto "
                   "(default cost gate), fold (space-to-depth with the "
                   "zero-padded kernel), or split (the DecomposeConvs "
                   "residue split)"),
    llvm::cl::init("auto"), llvm::cl::Hidden
);

/// Cost comparison of the S2D fold against the residue split
/// (rewriteNoPadStride in DecomposeConvsForHardwarePass) that the conv would
/// take if this pattern bails.  Both forms stream the weight once, so the
/// differences are the input and output traffic:
///   fold:  the input is transposed into channels (2*inBytes) and read by
///          the conv (inBytes); the zero-padded kernel inflates the weight
///          stream to weightBytes * paddedWeightRatio; one padded output
///          write (outBytes).
///   split: N windowed gathers of the input (N*inBytes) plus the
///          materialized thinned inputs (inBytes) plus the conv reads
///          (inBytes); the thinned kernels partition the original taps
///          (weightBytes, no padding); and (4N-2)*outBytes of output
///          traffic (N sub-conv writes + the zero fill + 3*(N-1) merge-add
///          read/read/write).
/// The split is cheaper only when the padded-weight inflation outweighs its
/// extra input/output traffic - tiny inputs with large F*C.  It is only a
/// clean all-NSS alternative for bf16/f16 outputs: f32 merge adds fall to
/// the host (--torq-fallback-f32-to-host is the default), and quantized
/// chains risk the channel-sliced per-channel-rescale CSS cliff.  Measured
/// A/B: the fold wins or ties on every corpus shape (the cmodel
/// additionally over-weights the fold's input transpose, so the fold's
/// measured margin is conservative).
static bool residueSplitCheaperThanS2dFold(
    int64_t inBytes, int64_t weightBytes, int64_t outBytes, int64_t residueCount,
    double paddedWeightRatio, Type outElem, bool stride1SubConvs, int64_t outElems
) {
    if (clS2DStrategy == "fold")
        return false;
    if (clS2DStrategy == "split")
        return true;
    if (!outElem || (!outElem.isBF16() && !outElem.isF16()))
        return false;
    // Small all-stride-1 splits of a promoted conv1d (unit spatial axis) hit
    // the byte model's fixed-cost blind spot: the byte terms are small
    // (hundreds of bytes) and the compile-time estimate is dominated by the
    // per-program/per-DMA fixed costs, where the fold's C-inflated conv plus
    // input S2D transpose plus padded-output extract loses to the split's
    // small stride-1 sub-convs plus merge adds.  Measured on the
    // sweepconv-bf16 corpus on both sim chips: the 1D all-stride-1 small
    // cases (sweepconv-bf16-10/11/18) are estimate-cheaper split, the 2D
    // case (sweepconv-bf16-1) and the stride-2-sub-conv cases
    // (sweepconv-bf16-17/19) fold; in the 2D case the split's sub-convs
    // repeat the full 2D kernel per residue class, which the fixed-cost
    // saving does not cover.  cmodel is equal in this regime
    // (startup-floor dominated), so the split is not a cmodel regression.
    // Large outputs stay on the byte model, where the split's (4N-2) merge
    // traffic eventually outweighs the fixed costs.
    if (stride1SubConvs && outElems <= 4096)
        return true;
    int64_t foldCost = 3 * inBytes + (int64_t)llround(weightBytes * paddedWeightRatio) + outBytes;
    int64_t splitCost =
        (residueCount + 2) * inBytes + weightBytes + (4 * residueCount - 2) * outBytes;
    return splitCost < foldCost;
}

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

        // A conv within the HW kernel and stride limits has no residue split:
        // rewriteNoPadStride takes only a stride above 2 or a kernel above the
        // stride-2 cap. Thus the cost gate below does not apply to it.
        const bool withinHwLimits = kH <= HwInfo::nss_max_kernel_height &&
                                    kW <= HwInfo::nss_max_kernel_width && sH <= maxStride &&
                                    sW <= maxStride;
        if (withinHwLimits && sH == sW)
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

        // Cost gate: compare the fold against the residue split the conv
        // would take if we bail (rewriteNoPadStride in
        // DecomposeConvsForHardwarePass).  Per-axis thinning factor of the
        // split: stride/2 when the thinned kernel fits the stride-2 cap,
        // else the full stride (axes with stride 1 never split).  The fold's
        // input padding (up to a multiple of the block) and padded output
        // are charged at their actual sizes.  A conv within the HW limits
        // (stride (2, 1) or (1, 2)) has no split, so it always folds: without
        // the fold, these strides give wrong results or stop on an assert in
        // convAdjustPadding.
        if (!withinHwLimits) {
            int64_t residueCount = 1;
            // Whether every split axis keeps the full stride (sub-convs are
            // all stride-1) or thins to stride 2 (the thinned kernel fits the
            // stride-2 cap).
            bool stride1SubConvs = true;
            // Whether every sub-kernel of the split is 1. The split takes a
            // conv whose image has a border in memory only in this case.
            bool unitSubKernels = true;
            for (int d = 0; d < 2; ++d) {
                int64_t s = d == 0 ? sH : sW;
                int64_t k = d == 0 ? kH : kW;
                int64_t cap = d == 0 ? HwInfo::nss_max_kernel_height : HwInfo::nss_max_kernel_width;
                bool thinned = s % 2 == 0 && (k - 1) / (s / 2) + 1 <= cap;
                residueCount *= thinned ? s / 2 : s;
                if (s > 1 && thinned)
                    stride1SubConvs = false;
                if (k > (thinned ? s / 2 : s))
                    unitSubKernels = false;
            }
            double paddedWeightRatio = 1.0;
            if (bH > 1)
                paddedWeightRatio *= (double)align_ceil(kH, sH) / (double)kH;
            if (bW > 1)
                paddedWeightRatio *= (double)align_ceil(kW, sW) / (double)kW;
            // Padded conv output per axis: a folded axis convolves the
            // block-padded input with the stride-aligned (zero-padded)
            // kernel; a non-folded axis keeps its native geometry.
            int64_t newOH = (pH - (bH > 1 ? align_ceil(kH, sH) : kH)) / sH + 1;
            int64_t newOW = (pW - (bW > 1 ? align_ceil(kW, sW) : kW)) / sW + 1;
            int64_t inB = inShape[0] * inShape[1] * pH * pW;
            int64_t wtB = fltShape[0] * inShape[1] * kH * kW;
            int64_t outB = outputType.getShape()[0] * outputType.getShape()[1] * newOH * newOW;
            int64_t e = cast<RankedTensorType>(inputType).getElementTypeBitWidth() / 8;
            int64_t outElems = 1;
            for (int64_t d : outputType.getShape())
                outElems *= d;
            const bool splitTakesConv = unitSubKernels || !imageHasBackwardPadding(input);
            if (splitTakesConv &&
                residueSplitCheaperThanS2dFold(
                    inB * e, wtB * e, outB * e, residueCount, paddedWeightRatio,
                    outputType.getElementType(),
                    stride1SubConvs && ((kW == 1 && sW == 1) || (kH == 1 && sH == 1)), outElems
                ))
                return rewriter.notifyMatchFailure(
                    convOp, "residue split is cheaper than the S2D fold (cost gate)"
                );
        }

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

        // Promote out-of-range conv1ds (stride > 2) to their conv2d form
        // (unit W extent) so the S2D pattern below can fold the large stride
        // into channels — a single in-limits conv instead of the
        // residue-class split + full-tensor add chain that
        // DecomposeConvsForHardwarePass would emit (32 sub-convs + 31 adds
        // for a stride-64 conv, which is catastrophic for runtime). In-range
        // conv1ds (stride <= 2) are left alone and keep their promotion in
        // OptimizeLinalgForTorqPass.
        populateConv1dPromotionPatterns(
            ctx, patterns, /*minStride=*/3, /*maxStride=*/std::numeric_limits<int64_t>::max()
        );

        // Thin the kernel-1 axes of the stride (2, 2) convs that the stride-2 kernels do not run
        // correctly. This gives a conv2d with stride (2, 1) or (1, 2), which the S2D pattern
        // below folds as it does for any other conv2d with these strides. The thinning patterns
        // have a higher benefit, so they run before the S2D pattern on the same conv.
        populateThinUnitKernelStride2ConvPatterns(ctx, patterns);

        patterns.add<DecomposeConvWithSpaceToDepthPattern>(ctx);

        if (failed(applyPatternsGreedily(func, std::move(patterns))))
            return signalPassFailure();
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDecomposeConvToSpaceToDepthPass() {
    return std::make_unique<DecomposeConvToSpaceToDepthPass>();
}

} // namespace mlir::syna::torq
