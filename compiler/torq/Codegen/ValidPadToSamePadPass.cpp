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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-valid-pad"

namespace mlir::syna::torq {

namespace {

float decodeEncodedFloatPadValue(int32_t encodedPadValue) {
    return llvm::bit_cast<float>(static_cast<uint32_t>(encodedPadValue));
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

        auto weight_type = llvm::cast<RankedTensorType>(op.getWeights().getType());
        if (!weight_type)
            return failure();

        auto weight_shape = weight_type.getShape();

        int64_t ksize_h, ksize_w;
        // For MaxPool, get kernel size from kernel attribute
        // For conv/depthwise, get kernel size from weight tensor shape
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            ksize_h = op.getKernel()[0];
            ksize_w = op.getKernel()[1];
        }
        else {
            // Support both Conv2D (rank-4: [out_ch, in_ch, kh, kw]) and
            // DepthwiseConv2D (rank-3: [ch, kh, kw])
            if (weight_type.getRank() == 4) {
                ksize_h = weight_shape[2];
                ksize_w = weight_shape[3];
            }
            else if (weight_type.getRank() == 3) {
                ksize_h = weight_shape[1];
                ksize_w = weight_shape[2];
            }
            else {
                return failure();
            }
        }

        int32_t stride_h = op.getStride()[0];
        int32_t stride_w = op.getStride()[1];

        auto pads = op.getPad();
        // Check which dimensions need conversion to hardware same-padding
        bool has_valid_pad_h = needsSamePadConversionH(pads, ksize_h);
        bool has_valid_pad_w = needsSamePadConversionW(pads, ksize_w);

        if (!(has_valid_pad_h || has_valid_pad_w))
            return failure();

        // Check if the dimensions with VALID padding have kernel size > 1
        // Reject large kernels (>7x7) as they cannot be handled by hardware or software efficiently
        if (ksize_h < 1 || ksize_w < 1 || ksize_h > 7 || ksize_w > 7)
            return failure();

        // Skip if this conv can use hardware padding (handled by DirectPattern)
        if (ksize_w <= 7 && ksize_h <= 7) {
            Value currentInput = op.getInput();
            for (int depth = 0; depth < 5 && currentInput; ++depth) {
                auto definingOp = currentInput.getDefiningOp();
                if (!definingOp)
                    break;

                if (isa<torq_hl::InterleavedInsertOp>(definingOp))
                    return failure();

                if (definingOp->getNumOperands() == 0)
                    break;

                currentInput = definingOp->getOperand(0);
            }
        }

        auto output_type = llvm::dyn_cast<RankedTensorType>(op.getInit().getType());
        auto output_shape = output_type.getShape();

        // Calculate SAME padding for dimensions with VALID padding
        int64_t total_pad_h = 0, total_pad_w = 0;

        if (has_valid_pad_h) {
            if (stride_h == 1) {
                total_pad_h = ksize_h - 1;
            }
            else {
                // For stride > 1: Calculate expected SAME output size first
                // SAME output = ceil(input / stride)
                int64_t same_output_h = (input_shape[NCHW::H] + stride_h - 1) / stride_h;
                // Then calculate required padding: total_pad = (output - 1) * stride + kernel -
                // input
                total_pad_h = std::max(
                    (same_output_h - 1) * stride_h + ksize_h - input_shape[NCHW::H], int64_t(0)
                );
            }
        }

        if (has_valid_pad_w) {
            if (stride_w == 1) {
                total_pad_w = ksize_w - 1;
            }
            else {
                // For stride > 1: Calculate expected SAME output size first
                int64_t same_output_w = (input_shape[NCHW::W] + stride_w - 1) / stride_w;
                total_pad_w = std::max(
                    (same_output_w - 1) * stride_w + ksize_w - input_shape[NCHW::W], int64_t(0)
                );
            }
        }

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
        int stride_offset = (stride_h == 2) ? 1 : 0;
        int mid = (ksize_h - 1) / 2;
        if (stride_h == 2 && (pads[LRTBDim::Left] == mid || pads[LRTBDim::Top] == mid)) {
            stride_offset = 0;
        }
        if (stride_h == 2 && (ksize_h == 1 && ksize_w == 1)) {
            stride_offset = 0;
        }

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
            if (pads[LRTBDim::Left] > pads[LRTBDim::Right]) {
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
            if (pads[LRTBDim::Top] > pads[LRTBDim::Bottom]) {
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
                op.getOutputMin(), op.getOutputMax(), op.getStride(), op.getPad(), op.getKernel(),
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
// Only applies when kernel size <= 7 (hardware constraint).
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

        auto weight_type = llvm::cast<RankedTensorType>(op.getWeights().getType());
        if (!weight_type)
            return failure();

        auto weight_shape = weight_type.getShape();
        int64_t ksize_h, ksize_w;

        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            // For MaxPool, get kernel size directly
            ksize_h = op.getKernel()[0];
            ksize_w = op.getKernel()[1];
        }
        else {
            if (weight_type.getRank() == 4) {
                ksize_h = weight_shape[2];
                ksize_w = weight_shape[3];
            }
            else if (weight_type.getRank() == 3) {
                ksize_h = weight_shape[1];
                ksize_w = weight_shape[2];
            }
            else {
                return failure();
            }
        }

        if (ksize_h > 7 || ksize_w > 7)
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

        auto input = op.getInput();
        Value sourceData = input;

        // If input is from insert_slice + fill, use the source instead
        if (auto insertSlice = input.template getDefiningOp<tensor::InsertSliceOp>()) {
            if (insertSlice.getDest().template getDefiningOp<linalg::FillOp>()) {
                sourceData = insertSlice.getSource();
            }
        }

        // Look for interleaved_insert in the input chain
        bool found = false;
        Value current = sourceData;
        for (int i = 0; i < 5 && current; ++i) {
            auto op = current.getDefiningOp();
            if (!op)
                break;

            if (isa<torq_hl::InterleavedInsertOp>(op)) {
                found = true;
                break;
            }

            if (op->getNumOperands() == 0)
                break;

            current = op->getOperand(0);
        }

        if (!found)
            return failure();

        int64_t total_pad_h = 0, total_pad_w = 0;
        if (stride_h == 1) {
            total_pad_h = ksize_h - 1;
        }
        else if (stride_h == 2) {
            int64_t out_h = (input_shape[NCHW::H] + 1) / 2;
            total_pad_h = std::max((out_h - 1) * 2 + ksize_h - input_shape[NCHW::H], int64_t(0));
        }
        else {
            return failure();
        }

        if (stride_w == 1) {
            total_pad_w = ksize_w - 1;
        }
        else if (stride_w == 2) {
            int64_t out_w = (input_shape[NCHW::W] + 1) / 2;
            total_pad_w = std::max((out_w - 1) * 2 + ksize_w - input_shape[NCHW::W], int64_t(0));
        }
        else {
            return failure();
        }

        if (total_pad_h == 0 && total_pad_w == 0)
            return failure();

        int64_t new_pad_top = total_pad_h / 2;
        int64_t new_pad_bottom = total_pad_h - new_pad_top;
        int64_t new_pad_left = total_pad_w / 2;
        int64_t new_pad_right = total_pad_w - new_pad_left;

        auto output_type = llvm::dyn_cast<RankedTensorType>(op.getInit().getType());
        auto output_shape = output_type.getShape();

        // Conv2d output shape matches sourceData (without explicit padding) since hardware applies
        // padding internally
        auto sourceType = llvm::cast<RankedTensorType>(sourceData.getType());
        auto source_shape = sourceType.getShape();
        SmallVector<int64_t, 4> new_output_shape = {
            source_shape[NCHW::N], source_shape[NCHW::C], source_shape[NCHW::H],
            source_shape[NCHW::W]
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
                op.getOutputMin(), op.getOutputMax(), op.getStride(), op.getPad(), op.getKernel(),
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

        // NPU handles padding internally, so output matches input dimensions
        // Only add extract_slice if final output dimensions differ from input
        if (output_shape[NCHW::N] == input_shape[NCHW::N] &&
            output_shape[NCHW::C] == input_shape[NCHW::C] &&
            output_shape[NCHW::H] == input_shape[NCHW::H] &&
            output_shape[NCHW::W] == input_shape[NCHW::W]) {
            // No dimension change needed - NPU output is already correct
            rewriter.replaceOp(op, samepadOp.getOutput());
        }
        else {
            // Extract to final output dimensions if they differ (e.g., width reduction)
            auto offsets = createVector({0, 0, 0, 0}, rewriter);
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
        }
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

        if (ksize_h > 7 || ksize_w > 7)
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
        if (op.getOutput().hasOneUse()) {
            auto user = *op.getOutput().getUsers().begin();
            if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
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

        int64_t kh = 1, kw = 1;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            // For MaxPool, get kernel size directly
            kh = op.getKernel()[0];
            kw = op.getKernel()[1];
        }
        else {
            auto weightType = llvm::cast<RankedTensorType>(op.getWeights().getType());
            if (!weightType)
                return failure();
            auto weightShape = weightType.getShape();
            if (weightType.getRank() == 4) {
                kh = weightShape[2];
                kw = weightShape[3];
            }
            else if (weightType.getRank() == 3) {
                kh = weightShape[1];
                kw = weightShape[2];
            }
            else {
                return failure();
            }
        }

        const bool validPadH = oddHeight && needsSamePadConversionH(pads, kh);
        const bool validPadW = oddWidth && needsSamePadConversionW(pads, kw);
        const bool convertToSame = validPadH || validPadW;

        SmallVector<int64_t, 4> paddedShape(shape.begin(), shape.end());
        if (oddHeight)
            paddedShape[NCHW::H]++;
        if (oddWidth)
            paddedShape[NCHW::W]++;

        auto loc = op.getLoc();
        auto elemType = inputType.getElementType();

        int offsetH = 0, offsetW = 0;
        int extractOffsetH = 0, extractOffsetW = 0;
        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());

        // This caluclation always return dominat bottom & right padding (1, 2, 1, 2) for 5x5, (0,
        // 1, 0, 0) for 3x3
        int64_t outh = (paddedShape[NCHW::H] + strides[0] - 1) / strides[0];
        int64_t outw = (paddedShape[NCHW::W] + strides[1] - 1) / strides[1];
        int64_t totalPadH =
            std::max((outh - 1) * strides[0] + kh - paddedShape[NCHW::H], int64_t(0));
        int64_t totalPadW =
            std::max((outw - 1) * strides[1] + kw - paddedShape[NCHW::W], int64_t(0));
        newPads[LRTBDim::Left] = totalPadW / 2;
        newPads[LRTBDim::Right] = totalPadW - newPads[LRTBDim::Left];
        newPads[LRTBDim::Top] = totalPadH / 2;
        newPads[LRTBDim::Bottom] = totalPadH - newPads[LRTBDim::Top];

        // Find the stride_offset 0 or 1, from the pad info
        int stride_offset = (strides[0] == 2) ? 1 : 0;
        int mid = (kh - 1) / 2;
        if (strides[0] == 2 && (pads[LRTBDim::Left] == mid || pads[LRTBDim::Top] == mid)) {
            stride_offset = 0;
        }
        if (strides[0] == 2 && (kh == 1 && kw == 1)) {
            stride_offset = 0;
        }

        if (stride_offset == 0) {
            std::swap(newPads[LRTBDim::Top], newPads[LRTBDim::Bottom]);
            std::swap(newPads[LRTBDim::Left], newPads[LRTBDim::Right]);
        }

        if (validPadH) {
            int centre = (kh - 1) / 2;
            offsetH = ((centre - pads[LRTBDim::Top]) ^ stride_offset) & 1;
            extractOffsetH = (centre + offsetH - pads[LRTBDim::Top]) / 2;
        }
        if (validPadW) {
            int centre = (kw - 1) / 2;
            offsetW = ((centre - pads[LRTBDim::Left]) ^ stride_offset) & 1;
            extractOffsetW = (centre + offsetW - pads[LRTBDim::Left]) / 2;
        }

        auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());
        auto origOutShape = outputType.getShape();

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
        if ((pads[LRTBDim::Top] || pads[LRTBDim::Bottom]) || !convertToSame) {
            const int32_t encodedPadValue = op.getInputZp();
            TypedAttr fillAttr;
            if (auto intType = llvm::dyn_cast<IntegerType>(elemType)) {
                fillAttr = rewriter.getIntegerAttr(intType, encodedPadValue);
            }
            else if (auto floatType = llvm::dyn_cast<FloatType>(elemType)) {
                fillAttr =
                    rewriter.getFloatAttr(floatType, decodeEncodedFloatPadValue(encodedPadValue));
            }
            else {
                return failure();
            }
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
        TorqConvPoolOp newOp;
        if constexpr (std::is_same_v<TorqConvPoolOp, torq_hl::MaxPool2dOp>) {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInitTensor, op.getInputZp(), op.getOutputMin(),
                op.getOutputMax(), op.getStride(), op.getPad(), op.getKernel(), op.getWeights(),
                op.getScaleBias(), insertOp
            );
        }
        else {
            newOp = TorqConvPoolOp::create(
                rewriter, loc, newOutputType, newInitTensor, op.getInputZp(), op.getWeightZp(),
                op.getOutputZp(), op.getOutputMin(), op.getOutputMax(), op.getShiftFactor(),
                op.getGroups(), rewriter.getDenseI64ArrayAttr(newPads), op.getStride(),
                op.getDilation(), op.getVectorizationMode(), op.getWeights(), op.getScaleBias(),
                insertOp
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
        patterns.add<EliminateRedundantConvPaddingPattern<torq_hl::Conv2DOp>>(ctx);

        // Register patterns for DepthwiseConv2DOp
        patterns.add<ConvertConvValidToSamePadDirectPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<ConvertConvValidPadToSamePadPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<EliminateRedundantConvPaddingPattern<torq_hl::DepthwiseConv2DOp>>(ctx);

        // Register patterns for MaxPool2dOp
        patterns.add<ConvertConvValidToSamePadDirectPattern<torq_hl::MaxPool2dOp>>(ctx);
        patterns.add<ConvertConvValidPadToSamePadPattern<torq_hl::MaxPool2dOp>>(ctx);
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
