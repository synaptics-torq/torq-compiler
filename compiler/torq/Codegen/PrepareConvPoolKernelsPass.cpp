// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prepares the conv2d, depthwise_conv2d and max_pool2d ops for their kernels, after tiling: gives
// each op a padding the kernels support, then the weight layout and input segmentation they read
// (ConvPoolKernelLayout.cpp).

#include "ConvPoolKernelLayout.h"
#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-prepare-conv-pool-kernels"

namespace mlir::syna::torq {

namespace {

float decodeEncodedFloatPadValue(int32_t encodedPadValue) {
    return llvm::bit_cast<float>(static_cast<uint32_t>(encodedPadValue));
}

// Fill attribute for the encoded pad value (raw int, or bit-cast float) in elemType.
// Returns null for unsupported element types.
TypedAttr makePadFillAttr(Builder &b, Type elemType, int32_t encodedPadValue) {
    if (auto intType = llvm::dyn_cast<IntegerType>(elemType))
        return b.getIntegerAttr(intType, encodedPadValue);
    if (auto floatType = llvm::dyn_cast<FloatType>(elemType))
        return b.getFloatAttr(floatType, decodeEncodedFloatPadValue(encodedPadValue));
    return nullptr;
}

// Returns true if the H padding does NOT match either valid hardware same-pad variant.
// Hardware always pads with kernel_top/kernel_bottom rows regardless of stride.
// For asymmetric kernels two orderings are valid: (top,bot)=(kernel_top,kernel_bottom) or swapped.
bool needsSamePadConversionH(llvm::ArrayRef<int64_t> pads, int64_t ksize_h) {
    int64_t kernel_top = (ksize_h - 1) / 2;
    int64_t kernel_bottom = ksize_h - kernel_top - 1;
    return !(
        (pads[LRTBDim::Top] == kernel_top && pads[LRTBDim::Bottom] == kernel_bottom) ||
        (pads[LRTBDim::Top] == kernel_bottom && pads[LRTBDim::Bottom] == kernel_top)
    );
}

// Returns true if the W padding does NOT match either valid hardware same-pad variant.
bool needsSamePadConversionW(llvm::ArrayRef<int64_t> pads, int64_t ksize_w) {
    int64_t kernel_left = (ksize_w - 1) / 2;
    int64_t kernel_right = ksize_w - kernel_left - 1;
    return !(
        (pads[LRTBDim::Left] == kernel_left && pads[LRTBDim::Right] == kernel_right) ||
        (pads[LRTBDim::Left] == kernel_right && pads[LRTBDim::Right] == kernel_left)
    );
}

// A conv/pool that is effectively 1D: one spatial axis is degenerate (extent 1, kernel 1,
// stride 1) while the other carries stride 2 with zero (valid) padding. The NPU executes such
// a valid stride-2 op correctly, but mis-executes the asymmetric SAME padding that the
// valid->same rewrites would introduce, corrupting ~99% of the outputs. Keep these valid.
// True 2D stride-2 convs (both axes strided) still require the SAME conversion: the NPU cannot
// execute a valid (grown-to-even) 2D stride-2 conv directly.
bool isValid1DStride2Conv(
    llvm::ArrayRef<int64_t> shape, int64_t kh, int64_t kw, int64_t strideH, int64_t strideW,
    llvm::ArrayRef<int64_t> pads
) {
    if (!llvm::all_of(pads, [](int64_t p) { return p == 0; }))
        return false;
    const bool degenerateW = shape[NCHW::W] == 1 && kw == 1 && strideW == 1;
    const bool degenerateH = shape[NCHW::H] == 1 && kh == 1 && strideH == 1;
    return (degenerateW && strideH == 2) || (degenerateH && strideW == 2);
}

// Extracts the H/W kernel size for a conv/pool op: MaxPool from its kernel attribute,
// Conv2D and DepthwiseConv2D from rank-4 weights [out_ch, in_ch, kh, kw]. Returns failure for
// other weight ranks.
template <class TorqConvPoolOp>
LogicalResult getConvPoolKernelSize(TorqConvPoolOp op, int64_t &ksizeH, int64_t &ksizeW) {
    if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
        ksizeH = op.getKernel()[0];
        ksizeW = op.getKernel()[1];
        return success();
    }
    else {
        auto weightType = llvm::dyn_cast<RankedTensorType>(op.getWeights().getType());
        if (!weightType)
            return failure();
        auto weightShape = weightType.getShape();
        if (weightType.getRank() == 4) {
            ksizeH = weightShape[2];
            ksizeW = weightShape[3];
            return success();
        }
        return failure();
    }
}

// stride_offset selects between the two hardware-valid SAME-pad alignments for a stride-2 op:
// 1 = bottom/right-heavy (default), 0 = top/left-heavy. Non stride-2 ops, 1x1 kernels, and
// inputs already using centre padding are top/left-heavy. See ConvertConvValidPadToSamePadPattern
// for the full explanation.
int computeStrideOffset(
    int64_t strideH, int64_t ksizeH, int64_t ksizeW, llvm::ArrayRef<int64_t> pads
) {
    if (strideH != 2)
        return 0;
    int mid = (ksizeH - 1) / 2;
    if (pads[LRTBDim::Left] == mid || pads[LRTBDim::Top] == mid)
        return 0;
    if (ksizeH == 1 && ksizeW == 1)
        return 0;
    return 1;
}

// Total SAME padding along one axis for any stride >= 1:
//   stride 1      -> ksize - 1
//   stride s (>1) -> (ceil(in/s) - 1) * s + ksize - in, clamped at 0
int64_t totalSamePad(int64_t stride, int64_t ksize, int64_t inExtent) {
    if (stride == 1)
        return ksize - 1;
    int64_t out = (inExtent + stride - 1) / stride;
    return std::max((out - 1) * stride + ksize - inExtent, int64_t(0));
}

// Creates a copy of `op` that reads `input` with the padding `pads` into a new output of shape
// `outShape`. `inputZp`, when given, replaces the input zero point (the padding value) of `op`.
template <class TorqConvPoolOp>
TorqConvPoolOp createWithPadding(
    PatternRewriter &rewriter, TorqConvPoolOp op, Value input, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> outShape, std::optional<int32_t> inputZp = std::nullopt
) {
    Location loc = op.getLoc();
    Type elemType = cast<RankedTensorType>(op.getInit().getType()).getElementType();
    auto outType = RankedTensorType::get(outShape, elemType);
    Value init = tensor::EmptyOp::create(rewriter, loc, outShape, elemType);
    auto padAttr = rewriter.getDenseI64ArrayAttr(pads);
    int32_t zp = inputZp.value_or(op.getInputZp());
    if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
        return TorqConvPoolOp::create(
            rewriter, loc, outType, init, zp, op.getOutputMin(), op.getOutputMax(), op.getStride(),
            padAttr, op.getKernel(), op.getWeights(), op.getScaleBias(), input
        );
    }
    else {
        return TorqConvPoolOp::create(
            rewriter, loc, outType, init, zp, op.getWeightZp(), op.getOutputZp(), op.getOutputMin(),
            op.getOutputMax(), op.getShiftFactor(), op.getGroups(), padAttr, op.getStride(),
            op.getDilation(), op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), input
        );
    }
}

// Where the original windows of an axis converted from VALID to SAME padding start on the SAME
// padded op.
//
// `prepend` is the number of extra input elements to prepend so that the expanded SAME padding has
// the window phase that `strideOffset` selects: for stride 2, (centre - leadPad) must have the same
// parity as strideOffset, so XOR-ing the two and masking to 1 bit gives 0 (aligned) or 1 (one extra
// element). E.g. 3x3 (centre 1), leadPad 1, strideOffset 1: ((1 - 1) ^ 1) & 1 = 1.
//
// `firstOutput` is the first output of the SAME padded op that is an output of the original op: the
// original window starts at input (centre + prepend - leadPad), divided by the stride. E.g. 5x5
// (centre 2), leadPad 0, prepend 1: (2 + 1 - 0) / 2 = 1, the first output is dropped.
struct ValidAxisShift {
    int64_t prepend;
    int64_t firstOutput;
};
ValidAxisShift getValidAxisShift(int64_t ksize, int64_t leadPad, int64_t stride, int strideOffset) {
    int64_t centre = (ksize - 1) / 2;
    int64_t prepend = ((centre - leadPad) ^ strideOffset) & (stride - 1);
    return {prepend, (centre + prepend - leadPad) / stride};
}

template <class TorqConvPoolOp>
class ConvertConvValidPadToSamePadPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {

        auto inputType = llvm::cast<RankedTensorType>(op.getInput().getType());
        if (!inputType)
            return failure();

        auto input_shape = inputType.getShape();
        if (input_shape.size() != 4 || !inputType.hasStaticShape())
            return failure();

        int64_t ksize_h, ksize_w;
        if (failed(getConvPoolKernelSize(op, ksize_h, ksize_w)))
            return failure();

        int32_t stride_h = op.getStride()[0];
        int32_t stride_w = op.getStride()[1];

        auto pads = op.getPad();
        // Check which dimensions need conversion to hardware same-padding
        bool has_valid_pad_h = needsSamePadConversionH(pads, ksize_h);
        bool has_valid_pad_w = needsSamePadConversionW(pads, ksize_w);

        if (!(has_valid_pad_h || has_valid_pad_w))
            return failure();

        // Check if the dimensions with VALID padding have kernel size > 1.
        // The two axes have different limits; see the cap definitions for what sets each.
        if (ksize_h < 1 || ksize_w < 1 || ksize_h > HwInfo::nss_max_kernel_height ||
            ksize_w > HwInfo::nss_max_kernel_width)
            return failure();

        auto output_type = llvm::dyn_cast<RankedTensorType>(op.getInit().getType());
        if (!output_type || !output_type.hasStaticShape())
            return failure();
        auto output_shape = output_type.getShape();

        // For stride>=2 only: if VALID padding already produces the correct output size,
        // converting to SAME is wasteful — it inflates the intermediate output by 1 row and
        // forces an extra extract_slice to trim back (e.g. MBv2: H=118, k=3, s=2, pad=[0,1,0,1]
        // → VALID H_out=59 == required 59, no SAME needed).
        // This guard must NOT apply to stride=1: for stride=1 VALID output size always equals
        // SAME output size mathematically, so the guard would wrongly skip cases where hardware
        // genuinely requires SAME-format padding (e.g. kh=4, stride=1 needs (1,2) not (0,0)).
        if (stride_h >= 2 && has_valid_pad_h) {
            int64_t valid_out_h =
                (input_shape[NCHW::H] + pads[LRTBDim::Top] + pads[LRTBDim::Bottom] - ksize_h) /
                    stride_h +
                1;
            if (valid_out_h == output_shape[NCHW::H])
                has_valid_pad_h = false;
        }
        // On W the VALID output must also fill the input quadrant: the stride-2 kernels step
        // through a quadrant (input W / stride wide) with the output row width. Fewer output
        // rows than the quadrant are fine, fewer output columns are not (e.g. 8x40, k=3, s=2,
        // pad 0: VALID W_out=19 but the quadrant is 20 wide).
        if (stride_w >= 2 && has_valid_pad_w) {
            int64_t valid_out_w =
                (input_shape[NCHW::W] + pads[LRTBDim::Left] + pads[LRTBDim::Right] - ksize_w) /
                    stride_w +
                1;
            if (valid_out_w == output_shape[NCHW::W] &&
                output_shape[NCHW::W] * stride_w == input_shape[NCHW::W])
                has_valid_pad_w = false;
        }
        if (!(has_valid_pad_h || has_valid_pad_w))
            return failure();

        // Total SAME padding, computed only for the axes that need conversion.
        int64_t total_pad_h =
            has_valid_pad_h ? totalSamePad(stride_h, ksize_h, input_shape[NCHW::H]) : 0;
        int64_t total_pad_w =
            has_valid_pad_w ? totalSamePad(stride_w, ksize_w, input_shape[NCHW::W]) : 0;

        if (total_pad_h == 0 && total_pad_w == 0)
            return failure();

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());

        // stride_offset encodes which of the two hardware-valid SAME-padding alignments applies
        // to stride-2 ops. For a stride-2 conv the hardware supports two asymmetric variants:
        //
        //   stride_offset = 1 (default) — bottom/right-heavy padding:
        //     3x3 → (top=0, bot=1),  5x5 → (top=1, bot=2),  7x7 → (top=2, bot=3)
        //   stride_offset = 0           — top/left-heavy padding:
        //     3x3 → (top=1, bot=0),  5x5 → (top=2, bot=1),  7x7 → (top=3, bot=2)
        //
        // If the existing pad already equals the kernel centre (mid), the input was already
        // using top/left-heavy padding, so stride_offset = 0. 1x1 kernels have no asymmetry
        // and always use stride_offset = 0. Either dimension (H or W) can be used to detect
        // this because both axes always share the same orientation.
        int stride_offset = computeStrideOffset(stride_h, ksize_h, ksize_w, pads);

        int64_t extractOffsetH = 0, extractOffsetW = 0;
        if (has_valid_pad_h) {
            extractOffsetH =
                getValidAxisShift(ksize_h, pads[LRTBDim::Top], stride_h, stride_offset).firstOutput;

            // Distribute total SAME padding symmetrically (half top, remainder bottom), then
            // swap if the W axis is already top/left-heavy to keep both axes consistent.
            newPads[LRTBDim::Top] = total_pad_h / 2;
            newPads[LRTBDim::Bottom] = total_pad_h - newPads[LRTBDim::Top];
            if (pads[LRTBDim::Left] > pads[LRTBDim::Right] && stride_offset) {
                std::swap(newPads[LRTBDim::Top], newPads[LRTBDim::Bottom]);
            }
        }

        if (has_valid_pad_w) {
            extractOffsetW =
                getValidAxisShift(ksize_w, pads[LRTBDim::Left], stride_w, stride_offset)
                    .firstOutput;
            // Distribute W SAME padding symmetrically, swap if H axis is top/left-heavy.
            newPads[LRTBDim::Left] = total_pad_w / 2;
            newPads[LRTBDim::Right] = total_pad_w - newPads[LRTBDim::Left];
            if (pads[LRTBDim::Top] > pads[LRTBDim::Bottom] && stride_offset) {
                std::swap(newPads[LRTBDim::Left], newPads[LRTBDim::Right]);
            }
        }

        SmallVector<int64_t, 4> same_pad_conv_output_shape = {
            output_shape[NCHW::N], output_shape[NCHW::C], output_shape[NCHW::H],
            output_shape[NCHW::W]
        };

        // Add padding only for dimensions that were converted from VALID to SAME
        if (has_valid_pad_h) {
            same_pad_conv_output_shape[NCHW::H] =
                (input_shape[NCHW::H] + newPads[LRTBDim::Top] + newPads[LRTBDim::Bottom] - ksize_h
                ) / stride_h +
                1;
        }
        if (has_valid_pad_w) {
            same_pad_conv_output_shape[NCHW::W] =
                (input_shape[NCHW::W] + newPads[LRTBDim::Left] + newPads[LRTBDim::Right] - ksize_w
                ) / stride_w +
                1;
        }

        // For stride-2 with odd H, skip wasteful SAME conversion when it does not change output.
        if (stride_h == 2 && (input_shape[NCHW::H] & 1) &&
            same_pad_conv_output_shape[NCHW::H] == output_shape[NCHW::H]) {
            return failure();
        }

        auto loc = op.getLoc();
        auto samepadOp =
            createWithPadding(rewriter, op, op.getInput(), newPads, same_pad_conv_output_shape);

        // Extract slice offsets: only offset dimensions that were converted from VALID to SAME
        int64_t offset_h = has_valid_pad_h ? extractOffsetH : 0;
        int64_t offset_w = has_valid_pad_w ? extractOffsetW : 0;

        auto offsets = createVector({0, 0, offset_h, offset_w}, rewriter);
        auto sizes = createVector(
            {output_shape[NCHW::N], output_shape[NCHW::C], output_shape[NCHW::H],
             output_shape[NCHW::W]},
            rewriter
        );
        auto slice_strides = createVector({1, 1, 1, 1}, rewriter);
        auto extractSliceOp = tensor::ExtractSliceOp::create(
            rewriter, loc, samepadOp.getOutput(), offsets, sizes, slice_strides
        );

        rewriter.replaceOp(op, extractSliceOp.getResult());
        return success();
    }
};

// Clean up the one-extra-row artifact created by ConvertOddDimensionStrideConvPattern.
// Phase 1 pads an odd-H input to even, then converts to SAME padding which inflates the
// output H by 1 (e.g. H=58 becomes H=59). It then adds an extract_slice to trim back.
// If that extra row can be eliminated by simply reducing padB by 1, do so directly:
//   conv(inH, pad=[L,R,T,B]) → H_out   →  extract_slice → targetH
//   ⟹  conv(inH, pad=[L,R,T,B-1]) → targetH   (no extract_slice needed)
template <class TorqConvPoolOp>
class EliminateTrailingConvRowPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {
        auto output = op.getOutput();
        if (!output.hasOneUse())
            return failure();

        auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(*output.getUsers().begin());
        if (!extractSlice)
            return failure();

        // Collect static offsets / sizes / strides of the extract_slice.
        SmallVector<int64_t, 4> sliceOffsets, sliceSizes;
        for (auto o : extractSlice.getMixedOffsets()) {
            auto v = getConstantIntValue(o);
            if (!v)
                return failure();
            sliceOffsets.push_back(*v);
        }
        for (auto s : extractSlice.getMixedSizes()) {
            auto v = getConstantIntValue(s);
            if (!v)
                return failure();
            sliceSizes.push_back(*v);
        }
        for (auto s : extractSlice.getMixedStrides()) {
            auto v = getConstantIntValue(s);
            if (!v || *v != 1)
                return failure();
        }
        if (sliceOffsets.size() != 4 || sliceSizes.size() != 4)
            return failure();

        // All offsets must be 0: we only handle bottom-row trimming, not top-row skipping.
        for (auto o : sliceOffsets)
            if (o != 0)
                return failure();

        auto convOutType = llvm::cast<RankedTensorType>(output.getType());
        auto convOutShape = convOutType.getShape();

        // N, C, W must be unchanged; H must decrease by exactly 1.
        if (sliceSizes[NCHW::N] != convOutShape[NCHW::N] ||
            sliceSizes[NCHW::C] != convOutShape[NCHW::C] ||
            sliceSizes[NCHW::W] != convOutShape[NCHW::W] ||
            convOutShape[NCHW::H] - sliceSizes[NCHW::H] != 1)
            return failure();

        int64_t targetH = sliceSizes[NCHW::H];

        auto pads = op.getPad();
        int64_t padT = pads[LRTBDim::Top];
        int64_t padB = pads[LRTBDim::Bottom];
        if (padB == 0)
            return failure();

        auto inputType = llvm::cast<RankedTensorType>(op.getInput().getType());
        auto inputShape = inputType.getShape();
        int32_t stride_h = op.getStride()[0];

        int64_t kh, kw;
        if (failed(getConvPoolKernelSize(op, kh, kw)))
            return failure();

        // Check: does reducing padB by 1 produce exactly targetH?
        int64_t newPadB = padB - 1;
        int64_t newOutH = (inputShape[NCHW::H] + padT + newPadB - kh) / stride_h + 1;
        if (newOutH != targetH)
            return failure();

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());
        newPads[LRTBDim::Bottom] = newPadB;
        SmallVector<int64_t, 4> newOutShape = {
            convOutShape[NCHW::N], convOutShape[NCHW::C], targetH, convOutShape[NCHW::W]
        };
        auto newOp = createWithPadding(rewriter, op, op.getInput(), newPads, newOutShape);

        rewriter.replaceOp(extractSlice, newOp.getOutput());
        rewriter.eraseOp(op);
        return success();
    }
};

// The stride-2 kernels compute the windows of output o from input [2o - B, 2o - B + k - 1] with
// B = (k - 1) / 2, i.e. with a leading padding of B: for an even kernel the leading padding (B + 1)
// of the other "same" variant, or no padding, gives wrong results
// (synaptics-torq/torq-compiler-dev#2717, e.g. a 4x4 stride-2 depthwise conv with padding 2 then 1
// or 0 then 1), and the other patterns of this pass would rewrite it to the (B, B + 1) variant
// without moving the data. Output o of the op covers [2o - p, 2o - p + k - 1] with p its leading
// padding: prepend e = (B - p) mod 2 padding elements to the input explicitly, compute the op
// with the padding (B, k - 1 - B) on an even input, and drop its first c = (B + e - p) / 2
// outputs, whose output o + c covers [2(o + c) - B - e, ...] = [2o - p, ...] of the original input.
template <class TorqConvPoolOp>
class AlignEvenKernelStride2PaddingPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {
        auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
        auto outputType = llvm::dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (!inputType || inputType.getRank() != 4 || !inputType.hasStaticShape() || !outputType ||
            !outputType.hasStaticShape()) {
            return failure();
        }
        auto strides = op.getStride();
        auto pads = op.getPad();
        if (strides.size() != 2 || pads.size() != 4) {
            return failure();
        }
        int64_t kh, kw;
        if (failed(getConvPoolKernelSize(op, kh, kw))) {
            return failure();
        }
        auto inShape = inputType.getShape();
        auto outShape = outputType.getShape();

        // Per axis (H, W): input size, padding, prepended and appended elements, dropped outputs
        struct Axis {
            int dim;
            int64_t k, stride, lead, trail;
            int64_t inSize, outSize;
            int64_t prepend = 0, append = 0, drop = 0;
            bool fix = false;
        };
        Axis axes[2] = {
            {NCHW::H, kh, strides[0], pads[LRTBDim::Top], pads[LRTBDim::Bottom], inShape[NCHW::H],
             outShape[NCHW::H]},
            {NCHW::W, kw, strides[1], pads[LRTBDim::Left], pads[LRTBDim::Right], inShape[NCHW::W],
             outShape[NCHW::W]},
        };
        bool anyFix = false;
        for (Axis &a : axes) {
            const int64_t border = (a.k - 1) / 2;
            if (a.stride != 2 || a.k % 2 != 0 || a.lead == border) {
                continue;
            }
            a.prepend = (((border - a.lead) % 2) + 2) % 2;
            a.drop = (border + a.prepend - a.lead) / 2;
            if (border + a.prepend - a.lead < 0) {
                return rewriter.notifyMatchFailure(op, "leading padding larger than k / 2");
            }
            // Even input, with enough elements for the outputs to keep after the dropped ones
            // (with the padding (B, k - 1 - B), an even input of n elements gives n / 2 outputs)
            int64_t size = std::max(a.inSize + a.prepend, 2 * (a.outSize + a.drop));
            size += size % 2;
            a.append = size - a.inSize - a.prepend;
            a.fix = true;
            anyFix = true;
        }
        if (!anyFix) {
            return failure();
        }

        Location loc = op.getLoc();
        Type elemType = inputType.getElementType();
        SmallVector<int64_t, 4> paddedShape(inShape.begin(), inShape.end());
        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());
        SmallVector<int64_t, 4> newOutShape(outShape.begin(), outShape.end());
        SmallVector<int64_t, 4> dropOffsets(4, 0);
        SmallVector<OpFoldResult> insertOffsets(4, rewriter.getIndexAttr(0));
        for (Axis &a : axes) {
            if (!a.fix) {
                continue;
            }
            const int64_t border = (a.k - 1) / 2;
            paddedShape[a.dim] = a.inSize + a.prepend + a.append;
            newOutShape[a.dim] = paddedShape[a.dim] / 2;
            dropOffsets[a.dim] = a.drop;
            insertOffsets[a.dim] = rewriter.getIndexAttr(a.prepend);
            if (a.dim == NCHW::H) {
                newPads[LRTBDim::Top] = border;
                newPads[LRTBDim::Bottom] = a.k - 1 - border;
            }
            else {
                newPads[LRTBDim::Left] = border;
                newPads[LRTBDim::Right] = a.k - 1 - border;
            }
        }

        // Input padded explicitly with the padding value of the op. A maxpool without padding has
        // no padding value (input_zp is then 0): pad it with the lowest value of its type
        int32_t padValue = op.getInputZp();
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            if (llvm::all_of(pads, [](int64_t p) { return p == 0; })) {
                if (auto intType = dyn_cast<IntegerType>(elemType)) {
                    padValue = static_cast<int32_t>(
                        APInt::getSignedMinValue(intType.getWidth()).getSExtValue()
                    );
                }
                else {
                    padValue = llvm::bit_cast<int32_t>(-std::numeric_limits<float>::infinity());
                }
            }
        }
        TypedAttr fillAttr = makePadFillAttr(rewriter, elemType, padValue);
        if (!fillAttr) {
            return rewriter.notifyMatchFailure(op, "unsupported padding value");
        }
        Value padded = tensor::EmptyOp::create(rewriter, loc, paddedShape, elemType);
        Value fillValue = arith::ConstantOp::create(rewriter, loc, fillAttr);
        padded = linalg::FillOp::create(rewriter, loc, ValueRange{fillValue}, ValueRange{padded})
                     .getResult(0);
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : inShape) {
            sizes.push_back(rewriter.getIndexAttr(dim));
        }
        Value newInput = tensor::InsertSliceOp::create(
            rewriter, loc, op.getInput(), padded, insertOffsets, sizes,
            SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1))
        );

        auto newOp = createWithPadding(rewriter, op, newInput, newPads, newOutShape, padValue);
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            op, newOp.getOutput(), createVector(dropOffsets, rewriter),
            createVector({outShape[0], outShape[1], outShape[2], outShape[3]}, rewriter),
            createVector({1, 1, 1, 1}, rewriter)
        );
        return success();
    }
};

// Fix stride-2 convolutions with odd dimensions by adding explicit padding.
// NPU segmentation for stride-2 requires even H/W dimensions.
template <class TorqConvPoolOp>
class ConvertOddDimensionStrideConvPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {
        auto inputType = llvm::cast<RankedTensorType>(op.getInput().getType());
        if (!inputType || inputType.getRank() != 4)
            return failure();

        auto shape = inputType.getShape();
        auto strides = op.getStride();
        auto pads = op.getPad();

        if (strides.size() != 2 || pads.size() != 4)
            return failure();

        const bool oddHeight = (shape[NCHW::H] & 1) && strides[0] == 2;
        const bool oddWidth = (shape[NCHW::W] & 1) && strides[1] == 2;

        if (!oddHeight && !oddWidth)
            return failure();

        int64_t kh, kw;
        if (failed(getConvPoolKernelSize(op, kh, kw)))
            return failure();

        SmallVector<int64_t, 4> paddedShape(shape.begin(), shape.end());
        if (oddHeight)
            paddedShape[NCHW::H]++;
        if (oddWidth)
            paddedShape[NCHW::W]++;

        auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());
        auto origOutShape = outputType.getShape();

        // An effectively-1D valid stride-2 conv (see isValid1DStride2Conv) must stay valid: grow
        // the odd axis to even by appending one zero rather than converting to asymmetric SAME,
        // which the NPU mis-executes. Only safe when growing preserves the output shape (odd
        // kernels); even kernels, and true 2D stride-2 convs, fall through to the SAME handling
        // below (the NPU cannot execute a valid grown-to-even 2D stride-2 conv directly).
        const bool keepValid =
            isValid1DStride2Conv(shape, kh, kw, strides[0], strides[1], pads) &&
            (paddedShape[NCHW::H] - kh) / strides[0] + 1 == origOutShape[NCHW::H] &&
            (paddedShape[NCHW::W] - kw) / strides[1] + 1 == origOutShape[NCHW::W];
        const bool validPadH = !keepValid && oddHeight && needsSamePadConversionH(pads, kh);
        const bool validPadW = !keepValid && oddWidth && needsSamePadConversionW(pads, kw);
        const bool convertToSame = validPadH || validPadW;

        auto loc = op.getLoc();
        auto elemType = inputType.getElementType();

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());

        // This calculation always returns dominant bottom & right padding: (1, 2, 1, 2) for 5x5,
        // (0, 1, 0, 0) for 3x3.
        int64_t totalPadH = totalSamePad(strides[0], kh, paddedShape[NCHW::H]);
        int64_t totalPadW = totalSamePad(strides[1], kw, paddedShape[NCHW::W]);
        newPads[LRTBDim::Left] = totalPadW / 2;
        newPads[LRTBDim::Right] = totalPadW - newPads[LRTBDim::Left];
        newPads[LRTBDim::Top] = totalPadH / 2;
        newPads[LRTBDim::Bottom] = totalPadH - newPads[LRTBDim::Top];

        // Find the stride_offset 0 or 1, from the pad info
        int stride_offset = computeStrideOffset(strides[0], kh, kw, pads);

        if (stride_offset == 0) {
            std::swap(newPads[LRTBDim::Top], newPads[LRTBDim::Bottom]);
            std::swap(newPads[LRTBDim::Left], newPads[LRTBDim::Right]);
        }

        // Keep a valid conv valid: the grown buffer uses zero padding, not SAME.
        if (keepValid) {
            newPads.assign(pads.begin(), pads.end());
        }
        else {
            // totalSamePad + stride_offset swap is only for axes we grew.
            // Growing odd H must not rewrite even-W pads: the swap turns
            // [0,1,0,0] (right=1) into [1,0,0,0] and enables a left halo.
            if (!oddHeight) {
                newPads[LRTBDim::Top] = pads[LRTBDim::Top];
                newPads[LRTBDim::Bottom] = pads[LRTBDim::Bottom];
            }
            if (!oddWidth) {
                newPads[LRTBDim::Left] = pads[LRTBDim::Left];
                newPads[LRTBDim::Right] = pads[LRTBDim::Right];
            }
        }

        // A VALID conv is emulated on the SAME-padded (grown) buffer: the input is placed after
        // `prepend` elements and the outputs from `firstOutput` on are the VALID outputs.
        ValidAxisShift shiftH{0, 0}, shiftW{0, 0};
        if (validPadH) {
            shiftH = getValidAxisShift(kh, pads[LRTBDim::Top], strides[0], stride_offset);
        }
        if (validPadW) {
            shiftW = getValidAxisShift(kw, pads[LRTBDim::Left], strides[1], stride_offset);
        }

        SmallVector<int64_t, 4> convOutShape;
        if (convertToSame) {
            convOutShape.push_back(origOutShape[NCHW::N]);
            convOutShape.push_back(origOutShape[NCHW::C]);
            convOutShape.push_back(
                (paddedShape[NCHW::H] + newPads[LRTBDim::Top] + newPads[LRTBDim::Bottom] - kh) /
                    strides[0] +
                1
            );
            convOutShape.push_back(
                (paddedShape[NCHW::W] + newPads[LRTBDim::Left] + newPads[LRTBDim::Right] - kw) /
                    strides[1] +
                1
            );
        }
        else {
            convOutShape.assign(origOutShape.begin(), origOutShape.end());
        }

        SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
        offsets[NCHW::H] = rewriter.getIndexAttr(shiftH.prepend);
        offsets[NCHW::W] = rewriter.getIndexAttr(shiftW.prepend);
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));
        auto padTensor = tensor::EmptyOp::create(rewriter, loc, paddedShape, elemType);

        Value convInput = padTensor.getResult();
        // Fill the grown buffer when either:
        // - keepValid: 1D stride-2 VALID growth used to skip fill; the extra row was
        //   tensor.empty and FPGA read it as NaN (C-model zeros it).
        // - the original pads/halo changed, so the kernel can see the new border.
        // Do not fill every odd-H/W 2D stride-2 rewrite: that added extra NSS/DMA on
        // MBv2 tiles that already had a defined halo.
        bool needFill = keepValid || pads[LRTBDim::Top] || pads[LRTBDim::Bottom] ||
                        (pads[LRTBDim::Left] != newPads[LRTBDim::Left]) ||
                        (pads[LRTBDim::Right] != newPads[LRTBDim::Right]);
        if (needFill) {
            TypedAttr fillAttr = makePadFillAttr(rewriter, elemType, op.getInputZp());
            if (!fillAttr)
                return failure();
            auto fillConst = arith::ConstantOp::create(rewriter, loc, fillAttr);
            auto fillOp = linalg::FillOp::create(
                rewriter, loc, ValueRange{fillConst}, ValueRange{padTensor.getResult()}
            );
            convInput = fillOp.getResult(0);
        }

        auto insertOp = tensor::InsertSliceOp::create(
            rewriter, loc, op.getInput(), convInput, offsets, sizes,
            SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1))
        );

        auto newOp = createWithPadding(rewriter, op, insertOp, newPads, convOutShape);

        if (convertToSame) {
            rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
                op, newOp.getOutput(),
                createVector({0, 0, shiftH.firstOutput, shiftW.firstOutput}, rewriter),
                createVector(
                    {origOutShape[NCHW::N], origOutShape[NCHW::C], origOutShape[NCHW::H],
                     origOutShape[NCHW::W]},
                    rewriter
                ),
                createVector({1, 1, 1, 1}, rewriter)
            );
        }
        else {
            rewriter.replaceOp(op, newOp.getResult(0));
        }

        return success();
    }
};

// The stride-2 maxpool descriptor encodes both axes' window phase in one stride_offset, so a
// pool whose top-pad phase differs from its left-pad phase (e.g. pad=[1,0,0,1] from H-tiling
// an odd-height pool) lowers to silently wrong output. Normalize: grow the mismatched axis
// with pad-value rows until its leading pad reaches the kernel center, then re-crop.
class ConvertMixedPadPhasePoolPattern : public OpRewritePattern<torq_hl::MaxPool2dOp> {
  public:
    using OpRewritePattern<torq_hl::MaxPool2dOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) const override {
        auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
        auto outputType = llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());
        if (!inputType || !outputType || inputType.getRank() != 4)
            return failure();

        auto strides = op.getStride();
        auto kernel = op.getKernel();
        auto pads = op.getPad();
        if (strides.size() != 2 || kernel.size() != 2 || pads.size() != 4)
            return failure();
        if (strides[0] != 2 || strides[1] != 2)
            return failure();

        const int64_t centerY = (kernel[0] - 1) / 2;
        const int64_t centerX = (kernel[1] - 1) / 2;
        const bool phaseX = pads[LRTBDim::Left] != centerX;
        const bool phaseY = pads[LRTBDim::Top] != centerY;
        if (phaseX == phaseY)
            return failure();

        int64_t axis, k, padBeginIdx, padEndIdx;
        if (phaseY) {
            axis = NCHW::H;
            k = kernel[0];
            padBeginIdx = LRTBDim::Top;
            padEndIdx = LRTBDim::Bottom;
        }
        else {
            axis = NCHW::W;
            k = kernel[1];
            padBeginIdx = LRTBDim::Left;
            padEndIdx = LRTBDim::Right;
        }
        const int64_t center = (k - 1) / 2;
        const int64_t growBegin = center - pads[padBeginIdx];
        if (growBegin <= 0)
            return failure(); // leading pad beyond the kernel center: not handled

        auto shape = inputType.getShape();
        SmallVector<int64_t, 4> paddedShape(shape.begin(), shape.end());
        paddedShape[axis] += growBegin;
        paddedShape[axis] += paddedShape[axis] % 2; // quadrant layout needs even extents

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());
        newPads[padBeginIdx] = center;

        auto origOutShape = outputType.getShape();
        SmallVector<int64_t, 4> newOutShape(origOutShape.begin(), origOutShape.end());
        newOutShape[axis] =
            (paddedShape[axis] + newPads[padBeginIdx] + newPads[padEndIdx] - k) / 2 + 1;
        // Growing the leading edge shifts every output window by growBegin positions.
        const int64_t extractOffset = growBegin;
        if (newOutShape[axis] < origOutShape[axis] + extractOffset)
            return failure();

        auto elemType = inputType.getElementType();
        Location loc = op.getLoc();

        TypedAttr fillAttr = makePadFillAttr(rewriter, elemType, op.getInputZp());
        if (!fillAttr)
            return failure();

        auto padTensor = tensor::EmptyOp::create(rewriter, loc, paddedShape, elemType);
        auto fillConst = arith::ConstantOp::create(rewriter, loc, fillAttr);
        auto fillOp = linalg::FillOp::create(
            rewriter, loc, ValueRange{fillConst}, ValueRange{padTensor.getResult()}
        );

        SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
        offsets[axis] = rewriter.getIndexAttr(growBegin);
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));
        auto insertOp = tensor::InsertSliceOp::create(
            rewriter, loc, op.getInput(), fillOp.getResult(0), offsets, sizes,
            SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1))
        );

        auto newOp = createWithPadding(rewriter, op, insertOp, newPads, newOutShape);

        SmallVector<int64_t, 4> extractOffsets(4, 0);
        extractOffsets[axis] = extractOffset;
        rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
            op, newOp.getOutput(),
            createVector(
                {extractOffsets[0], extractOffsets[1], extractOffsets[2], extractOffsets[3]},
                rewriter
            ),
            createVector(
                {origOutShape[NCHW::N], origOutShape[NCHW::C], origOutShape[NCHW::H],
                 origOutShape[NCHW::W]},
                rewriter
            ),
            createVector({1, 1, 1, 1}, rewriter)
        );
        return success();
    }
};

// Adds `Pattern<Conv2DOp>`, `Pattern<DepthwiseConv2DOp>` and `Pattern<MaxPool2dOp>`
template <template <class> class Pattern> void addConvPoolPatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<Pattern<torq_hl::Conv2DOp>>(ctx);
    patterns.add<Pattern<torq_hl::DepthwiseConv2DOp>>(ctx);
    patterns.add<Pattern<torq_hl::MaxPool2dOp>>(ctx);
}

class PrepareConvPoolKernelsPass
    : public impl::PrepareConvPoolKernelsBase<PrepareConvPoolKernelsPass> {
  public:
    using PrepareConvPoolKernelsBase<PrepareConvPoolKernelsPass>::PrepareConvPoolKernelsBase;
    void runOnOperation() override;
};

void PrepareConvPoolKernelsPass::runOnOperation() {
    MLIRContext *ctx = getOperation().getContext();

    auto applyPhase = [&](RewritePatternSet &&patterns, bool existingOpsOnly) {
        GreedyRewriteConfig cfg;
        if (existingOpsOnly) {
            cfg.setStrictness(GreedyRewriteStrictness::ExistingOps);
        }
        return applyPatternsGreedily(getOperation(), std::move(patterns), cfg);
    };

    // Phase 0: give the stride-2 ops with an even kernel the leading padding the kernels support
    RewritePatternSet alignPatterns(ctx);
    addConvPoolPatterns<AlignEvenKernelStride2PaddingPattern>(alignPatterns);
    if (failed(applyPhase(std::move(alignPatterns), /*existingOpsOnly=*/true))) {
        return signalPassFailure();
    }

    // Phase 1: Apply odd-dimension stride fixing first (highest priority)
    // This ensures explicit padding is added for odd dimensions before other patterns run
    RewritePatternSet oddDimPatterns(ctx);
    addConvPoolPatterns<ConvertOddDimensionStrideConvPattern>(oddDimPatterns);
    if (failed(applyPhase(std::move(oddDimPatterns), /*existingOpsOnly=*/true))) {
        return signalPassFailure();
    }

    // Phase 2: Apply other padding transformations after odd-dimension fixing is complete
    RewritePatternSet padPatterns(ctx);
    padPatterns.add<ConvertMixedPadPhasePoolPattern>(ctx);
    addConvPoolPatterns<ConvertConvValidPadToSamePadPattern>(padPatterns);
    addConvPoolPatterns<EliminateTrailingConvRowPattern>(padPatterns);
    if (failed(applyPhase(std::move(padPatterns), /*existingOpsOnly=*/true))) {
        return signalPassFailure();
    }

    // Phase 3: the weight layout and input segmentation of the kernels, on the final padding
    RewritePatternSet layoutPatterns(ctx);
    populateConvPoolKernelLayoutPatterns(layoutPatterns);
    if (failed(applyPhase(std::move(layoutPatterns), /*existingOpsOnly=*/false))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPrepareConvPoolKernelsPass() {
    return std::make_unique<PrepareConvPoolKernelsPass>();
}

} // namespace mlir::syna::torq
