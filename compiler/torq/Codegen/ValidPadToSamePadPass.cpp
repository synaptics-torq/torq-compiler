// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-valid-pad"

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
// Conv2D from rank-4 weights [out_ch, in_ch, kh, kw], DepthwiseConv2D from rank-3
// weights [ch, kh, kw]. Returns failure for unexpected weight ranks.
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
        if (weightType.getRank() == 3) {
            ksizeH = weightShape[1];
            ksizeW = weightShape[2];
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

// Walks up to maxDepth defining ops from `start` and reports whether an InterleavedInsertOp is
// reached. Follows the data producer at each step: for a destination-style op operand(0) is the
// init/destination, so its first DPS input is used instead of blindly taking operand(0).
bool chainHasInterleavedInsert(Value start, int maxDepth) {
    Value current = start;
    for (int i = 0; i < maxDepth && current; ++i) {
        Operation *def = current.getDefiningOp();
        if (!def)
            break;
        if (isa<torq_hl::InterleavedInsertOp>(def))
            return true;
        if (auto dps = dyn_cast<DestinationStyleOpInterface>(def)) {
            if (dps.getNumDpsInputs() == 0)
                break;
            current = dps.getDpsInputOperand(0)->get();
            continue;
        }
        if (def->getNumOperands() == 0)
            break;
        current = def->getOperand(0);
    }
    return false;
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
        if (input_shape.size() != 4)
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

        // Skip only when the conv input is directly interleaved_insert.
        // DirectPattern converts that case. When interleaved sits under an
        // insert_slice border (ConvTranspose upsample + output_padding), this
        // generic SAME rewrite on the padded tensor is required: it yields
        // EK-legal pads (e.g. [0,1,2,2]) plus extract_slice. Peeling the border
        // into attrs as [1,1,...] is not EK-legal for even kw and breaks NDL.
        if (ksize_w <= 7 && ksize_h <= 7) {
            if (isa_and_nonnull<torq_hl::InterleavedInsertOp>(op.getInput().getDefiningOp()))
                return failure();
        }

        auto output_type = llvm::dyn_cast<RankedTensorType>(op.getInit().getType());
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
        if (stride_w >= 2 && has_valid_pad_w) {
            int64_t valid_out_w =
                (input_shape[NCHW::W] + pads[LRTBDim::Left] + pads[LRTBDim::Right] - ksize_w) /
                    stride_w +
                1;
            if (valid_out_w == output_shape[NCHW::W])
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

        int offsetH = 0, offsetW = 0;
        int extractOffsetH = 0, extractOffsetW = 0;
        if (has_valid_pad_h) {
            int centre = (ksize_h - 1) / 2;

            // offsetH: number of extra input rows to prepend at the top so that the expanded
            // SAME padding aligns with the hardware's expected stride_offset parity.
            //
            // For stride=2, (centre - pad_top) must have the same parity as stride_offset.
            // XOR-ing the difference with stride_offset and masking to 1 bit gives the
            // misalignment: 0 = already aligned, 1 = one extra row needed at the top.
            //
            // Example — 3x3 (centre=1), pad_top=1, stride_offset=1:
            //   ((1-1) ^ 1) & 1 = 1  →  insert 1 row at top, shifting pad from (1,0) to (2,0).
            // Example — 5x5 (centre=2), pad_top=0, stride_offset=1:
            //   ((2-0) ^ 1) & 1 = 1  →  insert 1 row at top, shifting pad from (0,2) to (1,2).
            offsetH = ((centre - pads[LRTBDim::Top]) ^ stride_offset) & (stride_h - 1);

            // extractOffsetH: the SAME-padded conv output includes extra leading rows produced
            // by the expanded top padding. This is the index of the first output row that maps
            // back to the original valid output, i.e., the number of output rows to skip.
            //
            // The correct output origin lies at input position (centre + offsetH - pad_top);
            // dividing by stride gives the output row index.
            //
            // Example — 5x5 (centre=2), pad_top=0, offsetH=1:
            //   (2+1-0)/2 = 1  →  skip the first output row; valid output begins at row 1.
            // Example — 3x3 (centre=1), pad_top=1, offsetH=1 (new top pad becomes 2):
            //   (1+1-1)/2 = 0  →  no rows to skip; valid output begins at row 0.
            extractOffsetH = (centre + offsetH - pads[LRTBDim::Top]) / stride_h;

            // Distribute total SAME padding symmetrically (half top, remainder bottom), then
            // swap if the W axis is already top/left-heavy to keep both axes consistent.
            newPads[LRTBDim::Top] = total_pad_h / 2;
            newPads[LRTBDim::Bottom] = total_pad_h - newPads[LRTBDim::Top];
            if (pads[LRTBDim::Left] > pads[LRTBDim::Right] && stride_offset) {
                std::swap(newPads[LRTBDim::Top], newPads[LRTBDim::Bottom]);
            }
        }

        if (has_valid_pad_w) {
            int centre = (ksize_w - 1) / 2;
            // Same parity-alignment logic as offsetH, applied to the W axis.
            offsetW = ((centre - pads[LRTBDim::Left]) ^ stride_offset) & (stride_w - 1);
            // Same output-origin calculation as extractOffsetH, applied to the W axis.
            extractOffsetW = (centre + offsetW - pads[LRTBDim::Left]) / stride_w;
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

        auto new_output_type =
            RankedTensorType::get(same_pad_conv_output_shape, output_type.getElementType());

        // Create new SAME pad attribute
        auto newPadAttr = rewriter.getDenseI64ArrayAttr(newPads);
        auto loc = op.getLoc();

        auto ConvInit = tensor::EmptyOp::create(
            rewriter, op.getLoc(), new_output_type.getShape(), new_output_type.getElementType()
        );

        TorqConvPoolOp samepadOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            samepadOp = TorqConvPoolOp::create(
                rewriter, loc, new_output_type, ConvInit.getResult(), op.getInputZp(),
                op.getOutputMin(), op.getOutputMax(), op.getStride(), newPadAttr, op.getKernel(),
                op.getWeights(), op.getScaleBias(), op.getInput()
            );
        }
        else {
            samepadOp = TorqConvPoolOp::create(
                rewriter, loc, new_output_type, ConvInit.getResult(), op.getInputZp(),
                op.getWeightZp(), op.getOutputZp(), op.getOutputMin(), op.getOutputMax(),
                op.getShiftFactor(), op.getGroups(), newPadAttr, op.getStride(), op.getDilation(),
                op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), op.getInput()
            );
        }

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

// Convert VALID padding to hardware SAME padding when interleaved_insert is present.
// Handles both cases:
//   1. Direct: interleaved_insert -> conv (prevents explicit padding creation)
//   2. Cleanup: interleaved_insert -> fill+insert_slice -> conv (removes explicit padding)
// Only applies within the per-axis kernel caps.
template <class TorqConvPoolOp>
class ConvertConvValidToSamePadDirectPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {

        auto inputType = llvm::cast<RankedTensorType>(op.getInput().getType());
        if (!inputType)
            return failure();

        auto input_shape = inputType.getShape();
        if (input_shape.size() != 4)
            return failure();

        int64_t ksize_h, ksize_w;
        if (failed(getConvPoolKernelSize(op, ksize_h, ksize_w)))
            return failure();

        if (ksize_h > HwInfo::nss_max_kernel_height || ksize_w > HwInfo::nss_max_kernel_width)
            return failure();

        int32_t stride_h = op.getStride()[0];
        int32_t stride_w = op.getStride()[1];

        int32_t pad_left = op.getPad()[LRTBDim::Left];
        int32_t pad_right = op.getPad()[LRTBDim::Right];
        int32_t pad_top = op.getPad()[LRTBDim::Top];
        int32_t pad_bottom = op.getPad()[LRTBDim::Bottom];

        if (!(pad_top == 0 && pad_left == 0 && pad_right == 0 && pad_bottom == 0 && ksize_h > 1 &&
              ksize_w > 1))
            return failure();

        BorderPeel peel;
        if (failed(
                tryPeelConvTransposeBorder(op, rewriter, ksize_h, ksize_w, stride_h, stride_w, peel)
            ))
            return failure();

        // Without a peelable border the conv input chain must still originate from an
        // interleaved_insert for this pattern to apply.
        Value sourceData = peel.peeled ? peel.sourceData : op.getInput();
        if (!peel.peeled && !chainHasInterleavedInsert(sourceData, 5))
            return failure();

        int64_t new_pad_top, new_pad_bottom, new_pad_left, new_pad_right;
        if (peel.peeled) {
            new_pad_top = peel.padTop;
            new_pad_bottom = peel.padBottom;
            new_pad_left = peel.padLeft;
            new_pad_right = peel.padRight;
        }
        else {
            // DirectPattern only supports stride 1 and 2.
            if ((stride_h != 1 && stride_h != 2) || (stride_w != 1 && stride_w != 2))
                return failure();
            int64_t total_pad_h = totalSamePad(stride_h, ksize_h, input_shape[NCHW::H]);
            int64_t total_pad_w = totalSamePad(stride_w, ksize_w, input_shape[NCHW::W]);
            if (total_pad_h == 0 && total_pad_w == 0)
                return failure();

            new_pad_top = total_pad_h / 2;
            new_pad_bottom = total_pad_h - new_pad_top;
            new_pad_left = total_pad_w / 2;
            new_pad_right = total_pad_w - new_pad_left;
        }

        auto output_type = llvm::dyn_cast<RankedTensorType>(op.getInit().getType());
        auto output_shape = output_type.getShape();

        SmallVector<int64_t, 4> new_output_shape = {
            output_shape[NCHW::N], output_shape[NCHW::C], output_shape[NCHW::H],
            output_shape[NCHW::W]
        };

        auto new_output_type =
            RankedTensorType::get(new_output_shape, output_type.getElementType());

        SmallVector<int64_t, 4> newPads{new_pad_left, new_pad_right, new_pad_top, new_pad_bottom};
        auto newPadAttr = rewriter.getDenseI64ArrayAttr(newPads);

        rewriter.setInsertionPoint(op);
        auto loc = op.getLoc();

        auto ConvInit = tensor::EmptyOp::create(
            rewriter, loc, new_output_shape, new_output_type.getElementType()
        );

        TorqConvPoolOp samepadOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            samepadOp = TorqConvPoolOp::create(
                rewriter, loc, new_output_type, ConvInit.getResult(), op.getInputZp(),
                op.getOutputMin(), op.getOutputMax(), op.getStride(), newPadAttr, op.getKernel(),
                op.getWeights(), op.getScaleBias(), sourceData
            );
        }
        else {
            samepadOp = TorqConvPoolOp::create(
                rewriter, loc, new_output_type, ConvInit.getResult(), op.getInputZp(),
                op.getWeightZp(), op.getOutputZp(), op.getOutputMin(), op.getOutputMax(),
                op.getShiftFactor(), op.getGroups(), newPadAttr, op.getStride(), op.getDilation(),
                op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), sourceData
            );
        }

        rewriter.replaceOp(op, samepadOp.getOutput());
        return success();
    }

  private:
    struct BorderPeel {
        Value sourceData;
        int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
        bool peeled = false;
    };

    // ConvTranspose upsample produces interleaved_insert -> insert_slice into a zero border (the
    // fill may already be outlined to a block arg). When that border is EK-legal (symmetric(k) or
    // swapped in W, and within kernel extent in H) it is peeled into TorqHL pad attributes and
    // returned via `out` (out.peeled == true). Borders that are present but not EK-legal, or whose
    // pads do not reproduce the conv output shape, are rejected with failure() so the generic SAME
    // pattern can rewrite them on the padded tensor instead. When there is no peelable insert
    // border, success() is returned with out.peeled == false.
    LogicalResult tryPeelConvTransposeBorder(
        TorqConvPoolOp op, PatternRewriter &rewriter, int64_t ksize_h, int64_t ksize_w,
        int32_t stride_h, int32_t stride_w, BorderPeel &out
    ) const {
        auto insertSlice = op.getInput().template getDefiningOp<tensor::InsertSliceOp>();
        if (!insertSlice)
            return success();

        auto offsets = insertSlice.getStaticOffsets();
        auto destShape = cast<RankedTensorType>(insertSlice.getDest().getType()).getShape();
        auto srcType = dyn_cast<RankedTensorType>(insertSlice.getSource().getType());
        if (offsets.size() != 4 || !srcType || srcType.getRank() != 4 ||
            ShapedType::isDynamicShape(offsets))
            return success();

        auto srcShape = srcType.getShape();
        int64_t top = offsets[NCHW::H];
        int64_t left = offsets[NCHW::W];
        int64_t bottom = destShape[NCHW::H] - offsets[NCHW::H] - srcShape[NCHW::H];
        int64_t right = destShape[NCHW::W] - offsets[NCHW::W] - srcShape[NCHW::W];
        if (bottom < 0 || right < 0)
            return success();

        if (!chainHasInterleavedInsert(insertSlice.getSource(), 5))
            return success();

        auto outTy = cast<RankedTensorType>(op.getInit().getType());
        auto outShape = outTy.getShape();
        int64_t expectH = (srcShape[NCHW::H] + top + bottom - ksize_h) / stride_h + 1;
        int64_t expectW = (srcShape[NCHW::W] + left + right - ksize_w) / stride_w + 1;
        if (expectH != outShape[NCHW::H] || expectW != outShape[NCHW::W]) {
            return rewriter.notifyMatchFailure(
                op, "insert border pads do not reproduce conv output shape"
            );
        }

        int64_t kernLeft = (ksize_w - 1) / 2;
        int64_t kernRight = ksize_w - kernLeft - 1;
        int64_t kernTop = (ksize_h - 1) / 2;
        int64_t kernBottom = ksize_h - kernTop - 1;
        bool wOk =
            (left == kernLeft && right == kernRight) || (left == kernRight && right == kernLeft);
        bool hOk = top <= kernTop && bottom <= kernBottom;
        // Border not EK-legal — defer to generic SAME on the padded tensor.
        if (!wOk || !hOk)
            return failure();

        out.sourceData = insertSlice.getSource();
        out.peeled = true;
        out.padTop = top;
        out.padBottom = bottom;
        out.padLeft = left;
        out.padRight = right;
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

        int64_t kh = 1;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            kh = op.getKernel()[0];
        }
        else {
            auto wt = llvm::cast<RankedTensorType>(op.getWeights().getType());
            auto ws = wt.getShape();
            kh = (wt.getRank() == 4) ? ws[2] : ws[1];
        }

        // Check: does reducing padB by 1 produce exactly targetH?
        int64_t newPadB = padB - 1;
        int64_t newOutH = (inputShape[NCHW::H] + padT + newPadB - kh) / stride_h + 1;
        if (newOutH != targetH)
            return failure();

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());
        newPads[LRTBDim::Bottom] = newPadB;
        auto newPadAttr = rewriter.getDenseI64ArrayAttr(newPads);

        auto loc = op.getLoc();
        auto outputElType = convOutType.getElementType();
        SmallVector<int64_t, 4> newOutShape = {
            convOutShape[NCHW::N], convOutShape[NCHW::C], targetH, convOutShape[NCHW::W]
        };
        auto newOutType = RankedTensorType::get(newOutShape, outputElType);
        auto newInit = tensor::EmptyOp::create(rewriter, loc, newOutShape, outputElType);

        TorqConvPoolOp newOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutType, newInit.getResult(), op.getInputZp(), op.getOutputMin(),
                op.getOutputMax(), op.getStride(), newPadAttr, op.getKernel(), op.getWeights(),
                op.getScaleBias(), op.getInput()
            );
        }
        else {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutType, newInit.getResult(), op.getInputZp(), op.getWeightZp(),
                op.getOutputZp(), op.getOutputMin(), op.getOutputMax(), op.getShiftFactor(),
                op.getGroups(), newPadAttr, op.getStride(), op.getDilation(),
                op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), op.getInput()
            );
        }

        rewriter.replaceOp(extractSlice, newOp.getOutput());
        rewriter.eraseOp(op);
        return success();
    }
};

// Remove redundant explicit padding when conv already has matching SAME padding.
template <class TorqConvPoolOp>
class EliminateRedundantConvPaddingPattern : public OpRewritePattern<TorqConvPoolOp> {
  public:
    using OpRewritePattern<TorqConvPoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvPoolOp op, PatternRewriter &rewriter) const override {

        auto input = op.getInput();

        auto insertSliceOp = input.template getDefiningOp<tensor::InsertSliceOp>();
        if (!insertSliceOp)
            return failure();

        auto fillOp = insertSliceOp.getDest().template getDefiningOp<linalg::FillOp>();
        if (!fillOp)
            return failure();

        auto sourceData = insertSliceOp.getSource();
        auto sourceType = llvm::cast<RankedTensorType>(sourceData.getType());
        auto destType = llvm::cast<RankedTensorType>(insertSliceOp.getType());

        if (!sourceType || !destType || sourceType.getRank() != 4 || destType.getRank() != 4)
            return failure();

        auto sourceShape = sourceType.getShape();
        auto destShape = destType.getShape();

        auto weightType = llvm::cast<RankedTensorType>(op.getWeights().getType());
        if (!weightType || weightType.getRank() != 4)
            return failure();

        auto weightShape = weightType.getShape();
        const int64_t ksize_h = weightShape[2];
        const int64_t ksize_w = weightShape[3];

        if (ksize_h > HwInfo::nss_max_kernel_height || ksize_w > HwInfo::nss_max_kernel_width)
            return failure();

        auto offsets = insertSliceOp.getMixedOffsets();
        if (offsets.size() != 4)
            return failure();

        SmallVector<int64_t, 4> offsetValues;
        for (auto offset : offsets) {
            if (auto val = getConstantIntValue(offset))
                offsetValues.push_back(*val);
            else
                return failure();
        }

        // Prevent the pattern from triggering on unsupported stride-2 cases
        bool rank4 = sourceType && destType && sourceType.getRank() == 4 && destType.getRank() == 4;
        if (!rank4 || op.getStride()[0] != 1 || op.getStride()[1] != 1)
            return failure();

        // Verify padding only in height dimension: offsets = [0, 0, pad_top, 0]
        int64_t pad_top = offsetValues[NCHW::H];
        int64_t pad_bottom = destShape[NCHW::H] - sourceShape[NCHW::H] - pad_top;
        if (pad_bottom < 0)
            return failure();

        if (destShape[NCHW::W] != sourceShape[NCHW::W])
            return failure();

        auto padAttr = op.getPad();
        if (padAttr.size() != 4)
            return failure();

        int64_t conv_pad_top = padAttr[LRTBDim::Top];
        int64_t conv_pad_bottom = padAttr[LRTBDim::Bottom];

        if (conv_pad_top != pad_top || conv_pad_bottom != pad_bottom)
            return failure();

        if (destType != llvm::cast<RankedTensorType>(input.getType()))
            return failure();

        auto convOutputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());

        SmallVector<int64_t, 4> newOutputShape = {
            sourceShape[NCHW::N], sourceShape[NCHW::C], sourceShape[NCHW::H], sourceShape[NCHW::W]
        };

        auto newOutputType = RankedTensorType::get(newOutputShape, convOutputType.getElementType());
        auto loc = op.getLoc();

        auto newInit =
            tensor::EmptyOp::create(rewriter, loc, newOutputShape, convOutputType.getElementType());

        TorqConvPoolOp newOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInit.getResult(), op.getInputZp(),
                op.getOutputMin(), op.getOutputMax(), op.getStride(), op.getPad(), op.getKernel(),
                op.getWeights(), op.getScaleBias(), sourceData
            );
        }
        else {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInit.getResult(), op.getInputZp(),
                op.getWeightZp(), op.getOutputZp(), op.getOutputMin(), op.getOutputMax(),
                op.getShiftFactor(), op.getGroups(), op.getPadAttr(), op.getStride(),
                op.getDilation(), op.getVectorizationMode(), op.getWeights(), op.getScaleBias(),
                sourceData
            );
        }

        // Update extract_slice if present
        if (auto extractSliceOp = getSingleUser<tensor::ExtractSliceOp>(op.getOutput())) {
            auto extractOffsets = extractSliceOp.getMixedOffsets();
            auto extractSizes = extractSliceOp.getMixedSizes();
            auto extractStrides = extractSliceOp.getMixedStrides();

            if (extractOffsets.size() == 4) {
                SmallVector<OpFoldResult, 4> newExtractOffsets;
                for (size_t i = 0; i < extractOffsets.size(); ++i) {
                    if (i == NCHW::H) {
                        newExtractOffsets.push_back(rewriter.getIndexAttr(0));
                    }
                    else {
                        newExtractOffsets.push_back(extractOffsets[i]);
                    }
                }

                auto newExtractSlice = tensor::ExtractSliceOp::create(
                    rewriter, extractSliceOp.getLoc(), newOp.getOutput(), newExtractOffsets,
                    extractSizes, extractStrides
                );

                rewriter.replaceOp(extractSliceOp, newExtractSlice.getResult());
                rewriter.eraseOp(op);
                return success();
            }
        }

        rewriter.replaceOp(op, newOp.getOutput());
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

        int offsetH = 0, offsetW = 0;
        int extractOffsetH = 0, extractOffsetW = 0;
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

        // A VALID conv is emulated on the SAME-padded (grown) buffer. Relative to the
        // kernel centre, the original VALID window starts `centre - pad` samples in; the
        // sub-stride phase of that start (mod stride, XOR stride_offset for interleave
        // parity) is `offset`, and `extractOffset` is the index of the first SAME-conv
        // output element that coincides with the VALID output (used by a later extract).
        if (validPadH) {
            int centre = (kh - 1) / 2;
            offsetH = ((centre - pads[LRTBDim::Top]) ^ stride_offset) & (strides[0] - 1);
            extractOffsetH = (centre + offsetH - pads[LRTBDim::Top]) / strides[0];
        }
        if (validPadW) {
            int centre = (kw - 1) / 2;
            offsetW = ((centre - pads[LRTBDim::Left]) ^ stride_offset) & (strides[1] - 1);
            extractOffsetW = (centre + offsetW - pads[LRTBDim::Left]) / strides[1];
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
        offsets[NCHW::H] = rewriter.getIndexAttr(offsetH);
        offsets[NCHW::W] = rewriter.getIndexAttr(offsetW);
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

        auto outputElType = outputType.getElementType();
        auto newOutputType = RankedTensorType::get(convOutShape, outputElType);
        auto newInitTensor = tensor::EmptyOp::create(rewriter, loc, convOutShape, outputElType);
        auto newPadsAttr = rewriter.getDenseI64ArrayAttr(newPads);
        TorqConvPoolOp newOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInitTensor, op.getInputZp(), op.getOutputMin(),
                op.getOutputMax(), op.getStride(), newPadsAttr, op.getKernel(), op.getWeights(),
                op.getScaleBias(), insertOp
            );
        }
        else {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInitTensor, op.getInputZp(), op.getWeightZp(),
                op.getOutputZp(), op.getOutputMin(), op.getOutputMax(), op.getShiftFactor(),
                op.getGroups(), newPadsAttr, op.getStride(), op.getDilation(),
                op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), insertOp
            );
        }

        if (convertToSame) {
            rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
                op, newOp.getOutput(),
                createVector(
                    {0, 0, validPadH ? extractOffsetH : 0, validPadW ? extractOffsetW : 0}, rewriter
                ),
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

        auto newOutputType = RankedTensorType::get(newOutShape, outputType.getElementType());
        auto newInitTensor =
            tensor::EmptyOp::create(rewriter, loc, newOutShape, outputType.getElementType());
        auto newOp = torq_hl::MaxPool2dOp::create(
            rewriter, loc, newOutputType, newInitTensor, op.getInputZp(), op.getOutputMin(),
            op.getOutputMax(), op.getStride(), rewriter.getDenseI64ArrayAttr(newPads),
            op.getKernel(), op.getWeights(), op.getScaleBias(), insertOp
        );

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

class ValidToSamePadPass : public impl::ValidToSamePadPassBase<ValidToSamePadPass> {
  public:
    using ValidToSamePadPassBase<ValidToSamePadPass>::ValidToSamePadPassBase;
    void runOnOperation() override;
};

void ValidToSamePadPass::runOnOperation() {
    MLIRContext *ctx = getOperation().getContext();

    // Phase 1: Apply odd-dimension stride fixing first (highest priority)
    // This ensures explicit padding is added for odd dimensions before other patterns run
    {
        RewritePatternSet oddDimPatterns(ctx);
        oddDimPatterns.add<ConvertOddDimensionStrideConvPattern<torq_hl::Conv2DOp>>(ctx);
        oddDimPatterns.add<ConvertOddDimensionStrideConvPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        oddDimPatterns.add<ConvertOddDimensionStrideConvPattern<torq_hl::MaxPool2dOp>>(ctx);

        GreedyRewriteConfig cfg;
        cfg.setStrictness(GreedyRewriteStrictness::ExistingOps);
        if (failed(applyPatternsGreedily(getOperation(), std::move(oddDimPatterns), cfg))) {
            return signalPassFailure();
        }
    }

    // Phase 2: Apply other padding transformations after odd-dimension fixing is complete
    {
        RewritePatternSet patterns(ctx);
        // Register patterns for Conv2DOp
        patterns.add<ConvertConvValidToSamePadDirectPattern<torq_hl::Conv2DOp>>(ctx);
        patterns.add<ConvertConvValidPadToSamePadPattern<torq_hl::Conv2DOp>>(ctx);
        patterns.add<EliminateTrailingConvRowPattern<torq_hl::Conv2DOp>>(ctx);
        patterns.add<EliminateRedundantConvPaddingPattern<torq_hl::Conv2DOp>>(ctx);

        // Register patterns for DepthwiseConv2DOp
        patterns.add<ConvertConvValidToSamePadDirectPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<ConvertConvValidPadToSamePadPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<EliminateTrailingConvRowPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<EliminateRedundantConvPaddingPattern<torq_hl::DepthwiseConv2DOp>>(ctx);

        // Register patterns for MaxPool2dOp
        patterns.add<ConvertMixedPadPhasePoolPattern>(ctx);
        patterns.add<ConvertConvValidToSamePadDirectPattern<torq_hl::MaxPool2dOp>>(ctx);
        patterns.add<ConvertConvValidPadToSamePadPattern<torq_hl::MaxPool2dOp>>(ctx);
        patterns.add<EliminateTrailingConvRowPattern<torq_hl::MaxPool2dOp>>(ctx);
        patterns.add<EliminateRedundantConvPaddingPattern<torq_hl::MaxPool2dOp>>(ctx);

        GreedyRewriteConfig cfg;
        cfg.setStrictness(GreedyRewriteStrictness::ExistingOps);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), cfg))) {
            return signalPassFailure();
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createValidToSamePadPass() {
    return std::make_unique<ValidToSamePadPass>();
}

} // namespace mlir::syna::torq
