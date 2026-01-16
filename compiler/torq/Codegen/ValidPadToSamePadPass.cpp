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
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-valid-pad"

namespace mlir::syna::torq {

namespace {

template <class TorqConvOp>
class ConvertConvValidPadToSamePadPattern : public OpRewritePattern<TorqConvOp> {
  public:
    using OpRewritePattern<TorqConvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvOp conv2dOp, PatternRewriter &rewriter) const override {

        auto inputType = llvm::cast<RankedTensorType>(conv2dOp.getInput().getType());
        if (!inputType)
            return failure();

        auto input_shape = inputType.getShape();
        if (input_shape.size() != 4)
            return failure();

        auto weight_type = llvm::cast<RankedTensorType>(conv2dOp.getWeights().getType());
        if (!weight_type)
            return failure();

        auto weight_shape = weight_type.getShape();

        // Support both Conv2D (rank-4: [out_ch, in_ch, kh, kw]) and
        // DepthwiseConv2D (rank-3: [ch, kh, kw])
        int64_t ksize_h, ksize_w;
        if (weight_shape.size() == 4) {
            ksize_h = weight_shape[2];
            ksize_w = weight_shape[3];
        }
        else if (weight_shape.size() == 3) {
            ksize_h = weight_shape[1];
            ksize_w = weight_shape[2];
        }
        else {
            return failure();
        }

        int32_t stride_h = conv2dOp.getStride()[0];
        int32_t stride_w = conv2dOp.getStride()[1];

        int32_t pad_left = conv2dOp.getPad()[0];
        int32_t pad_right = conv2dOp.getPad()[1];
        int32_t pad_top = conv2dOp.getPad()[2];
        int32_t pad_bottom = conv2dOp.getPad()[3];

        // Check which dimensions have VALID padding (pad == 0)
        bool has_valid_pad_h = (pad_top == 0 && pad_bottom == 0);
        bool has_valid_pad_w = (pad_left == 0 && pad_right == 0);

        // Only transform if at least one dimension has VALID padding and filter size > 1
        if (!has_valid_pad_h && !has_valid_pad_w)
            return failure();

        // Check if the dimensions with VALID padding have kernel size > 1
        if (has_valid_pad_h && ksize_h <= 1)
            has_valid_pad_h = false;
        if (has_valid_pad_w && ksize_w <= 1)
            has_valid_pad_w = false;

        if (!has_valid_pad_h && !has_valid_pad_w)
            return failure();

        // Reject large kernels (>7x7) as they cannot be handled by hardware or software efficiently
        if (ksize_h > 7 || ksize_w > 7)
            return failure();

        // Skip if this conv can use hardware padding (handled by DirectPattern)
        if (ksize_w <= 7 && ksize_h <= 7) {
            Value currentInput = conv2dOp.getInput();
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

        auto output_type = llvm::dyn_cast<RankedTensorType>(conv2dOp.getInit().getType());
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
                int64_t same_output_h = (input_shape[2] + stride_h - 1) / stride_h;
                // Then calculate required padding: total_pad = (output - 1) * stride + kernel -
                // input
                total_pad_h =
                    std::max((same_output_h - 1) * stride_h + ksize_h - input_shape[2], int64_t(0));
            }
        }

        if (has_valid_pad_w) {
            if (stride_w == 1) {
                total_pad_w = ksize_w - 1;
            }
            else {
                // For stride > 1: Calculate expected SAME output size first
                int64_t same_output_w = (input_shape[3] + stride_w - 1) / stride_w;
                total_pad_w =
                    std::max((same_output_w - 1) * stride_w + ksize_w - input_shape[3], int64_t(0));
            }
        }

        if (total_pad_h == 0 && total_pad_w == 0)
            return failure();

        // Calculate new padding values, preserving existing padding for dimensions that already
        // have it
        int64_t new_pad_top, new_pad_bottom, new_pad_left, new_pad_right;

        if (has_valid_pad_h) {
            new_pad_top = total_pad_h / 2;
            new_pad_bottom = total_pad_h - new_pad_top;
        }
        else {
            // Keep existing padding
            new_pad_top = pad_top;
            new_pad_bottom = pad_bottom;
        }

        if (has_valid_pad_w) {
            new_pad_left = total_pad_w / 2;
            new_pad_right = total_pad_w - new_pad_left;
        }
        else {
            // Keep existing padding
            new_pad_left = pad_left;
            new_pad_right = pad_right;
        }

        SmallVector<int64_t, 4> same_pad_conv_output_shape = {
            output_shape[NCHW::N], output_shape[NCHW::C], output_shape[NCHW::H],
            output_shape[NCHW::W]
        };

        // Add padding only for dimensions that were converted from VALID to SAME
        if (has_valid_pad_h) {
            same_pad_conv_output_shape[NCHW::H] += (new_pad_top + new_pad_bottom);
        }
        if (has_valid_pad_w) {
            same_pad_conv_output_shape[NCHW::W] += (new_pad_left + new_pad_right);
        }

        auto new_output_type =
            RankedTensorType::get(same_pad_conv_output_shape, output_type.getElementType());

        // Create new SAME pad attribute
        SmallVector<int64_t, 4> newPads{new_pad_left, new_pad_right, new_pad_top, new_pad_bottom};
        auto newPadAttr = rewriter.getDenseI64ArrayAttr(newPads);

        rewriter.setInsertionPoint(conv2dOp);
        auto loc = conv2dOp.getLoc();

        auto ConvInit = rewriter.create<tensor::EmptyOp>(
            conv2dOp.getLoc(), new_output_type.getShape(), new_output_type.getElementType()
        );

        auto samepadConv = rewriter.create<TorqConvOp>(
            loc, new_output_type, ConvInit.getResult(), conv2dOp.getInputZp(),
            conv2dOp.getWeightZp(), conv2dOp.getOutputZp(), conv2dOp.getOutputMin(),
            conv2dOp.getOutputMax(), conv2dOp.getShiftFactor(), conv2dOp.getGroups(), newPadAttr,
            conv2dOp.getStride(), conv2dOp.getDilation(), conv2dOp.getVectorizationMode(),
            conv2dOp.getWeights(), conv2dOp.getScaleBias(), conv2dOp.getInput()
        );

        // Extract slice offsets: only offset dimensions that were converted from VALID to SAME
        int64_t offset_h = has_valid_pad_h ? new_pad_top : 0;
        int64_t offset_w = has_valid_pad_w ? new_pad_left : 0;

        auto offsets = createVector({0, 0, offset_h, offset_w}, rewriter);
        auto sizes = createVector(
            {output_shape[NCHW::N], output_shape[NCHW::C], output_shape[NCHW::H],
             output_shape[NCHW::W]},
            rewriter
        );
        auto slice_strides = createVector({1, 1, 1, 1}, rewriter);
        auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            loc, samepadConv.getOutput(), offsets, sizes, slice_strides
        );

        rewriter.replaceOp(conv2dOp, extractSliceOp.getResult());
        return success();
    }
};

// Convert VALID padding to hardware SAME padding when interleaved_insert is present.
// Handles both cases:
//   1. Direct: interleaved_insert -> conv (prevents explicit padding creation)
//   2. Cleanup: interleaved_insert -> fill+insert_slice -> conv (removes explicit padding)
// Only applies when kernel size <= 7 (hardware constraint).
template <class TorqConvOp>
class ConvertConvValidToSamePadDirectPattern : public OpRewritePattern<TorqConvOp> {
  public:
    using OpRewritePattern<TorqConvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvOp conv2dOp, PatternRewriter &rewriter) const override {

        auto inputType = llvm::cast<RankedTensorType>(conv2dOp.getInput().getType());
        if (!inputType)
            return failure();

        auto input_shape = inputType.getShape();
        if (input_shape.size() != 4)
            return failure();

        auto weight_type = llvm::cast<RankedTensorType>(conv2dOp.getWeights().getType());
        if (!weight_type)
            return failure();

        auto weight_shape = weight_type.getShape();
        int64_t ksize_h, ksize_w;

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

        if (ksize_h > 7 || ksize_w > 7)
            return failure();

        int32_t stride_h = conv2dOp.getStride()[0];
        int32_t stride_w = conv2dOp.getStride()[1];

        int32_t pad_left = conv2dOp.getPad()[0];
        int32_t pad_right = conv2dOp.getPad()[1];
        int32_t pad_top = conv2dOp.getPad()[2];
        int32_t pad_bottom = conv2dOp.getPad()[3];

        if (!(pad_top == 0 && pad_left == 0 && pad_right == 0 && pad_bottom == 0 && ksize_h > 1 &&
              ksize_w > 1))
            return failure();

        auto input = conv2dOp.getInput();
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

        auto output_type = llvm::dyn_cast<RankedTensorType>(conv2dOp.getInit().getType());
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

        rewriter.setInsertionPoint(conv2dOp);
        auto loc = conv2dOp.getLoc();

        auto ConvInit = rewriter.create<tensor::EmptyOp>(
            loc, new_output_shape, new_output_type.getElementType()
        );

        auto samepadConv = rewriter.create<TorqConvOp>(
            loc, new_output_type, ConvInit.getResult(), conv2dOp.getInputZp(),
            conv2dOp.getWeightZp(), conv2dOp.getOutputZp(), conv2dOp.getOutputMin(),
            conv2dOp.getOutputMax(), conv2dOp.getShiftFactor(), conv2dOp.getGroups(), newPadAttr,
            conv2dOp.getStride(), conv2dOp.getDilation(), conv2dOp.getVectorizationMode(),
            conv2dOp.getWeights(), conv2dOp.getScaleBias(), sourceData
        );

        // NPU handles padding internally, so output matches input dimensions
        // Only add extract_slice if final output dimensions differ from input
        if (output_shape[NCHW::N] == input_shape[NCHW::N] &&
            output_shape[NCHW::C] == input_shape[NCHW::C] &&
            output_shape[NCHW::H] == input_shape[NCHW::H] &&
            output_shape[NCHW::W] == input_shape[NCHW::W]) {
            // No dimension change needed - NPU output is already correct
            rewriter.replaceOp(conv2dOp, samepadConv.getOutput());
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
            auto extractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
                loc, samepadConv.getOutput(), offsets, sizes, slice_strides
            );
            rewriter.replaceOp(conv2dOp, extractSliceOp.getResult());
        }
        return success();
    }
};

// Remove redundant explicit padding when conv already has matching SAME padding.
template <class TorqConvOp>
class EliminateRedundantConvPaddingPattern : public OpRewritePattern<TorqConvOp> {
  public:
    using OpRewritePattern<TorqConvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvOp conv2dOp, PatternRewriter &rewriter) const override {

        auto input = conv2dOp.getInput();

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

        auto weightType = llvm::cast<RankedTensorType>(conv2dOp.getWeights().getType());
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
            if (offset.template is<Attribute>()) {
                auto intAttr = llvm::dyn_cast<IntegerAttr>(offset.template get<Attribute>());
                if (!intAttr)
                    return failure();
                offsetValues.push_back(intAttr.getInt());
            }
            else {
                return failure();
            }
        }

        // Verify padding only in height dimension: offsets = [0, 0, pad_top, 0]
        if (offsetValues[NCHW::N] != 0 || offsetValues[NCHW::C] != 0 || offsetValues[NCHW::W] != 0)
            return failure();

        int64_t pad_top = offsetValues[NCHW::H];
        int64_t pad_bottom = destShape[NCHW::H] - sourceShape[NCHW::H] - pad_top;
        if (pad_bottom < 0)
            return failure();

        if (destShape[NCHW::W] != sourceShape[NCHW::W])
            return failure();

        auto padAttr = conv2dOp.getPad();
        if (padAttr.size() != 4)
            return failure();

        int64_t conv_pad_top = padAttr[2];
        int64_t conv_pad_bottom = padAttr[3];

        if (conv_pad_top != pad_top || conv_pad_bottom != pad_bottom)
            return failure();

        if (destType != llvm::cast<RankedTensorType>(input.getType()))
            return failure();

        auto convOutputType = llvm::cast<RankedTensorType>(conv2dOp.getResult(0).getType());

        SmallVector<int64_t, 4> newOutputShape = {
            sourceShape[NCHW::N], sourceShape[NCHW::C], sourceShape[NCHW::H], sourceShape[NCHW::W]
        };

        auto newOutputType = RankedTensorType::get(newOutputShape, convOutputType.getElementType());
        auto loc = conv2dOp.getLoc();

        auto newInit =
            rewriter.create<tensor::EmptyOp>(loc, newOutputShape, convOutputType.getElementType());

        auto newConv = rewriter.create<TorqConvOp>(
            loc, newOutputType, newInit.getResult(), conv2dOp.getInputZp(), conv2dOp.getWeightZp(),
            conv2dOp.getOutputZp(), conv2dOp.getOutputMin(), conv2dOp.getOutputMax(),
            conv2dOp.getShiftFactor(), conv2dOp.getGroups(), conv2dOp.getPadAttr(),
            conv2dOp.getStride(), conv2dOp.getDilation(), conv2dOp.getVectorizationMode(),
            conv2dOp.getWeights(), conv2dOp.getScaleBias(), sourceData
        );

        // Update extract_slice if present
        if (conv2dOp.getOutput().hasOneUse()) {
            auto user = *conv2dOp.getOutput().getUsers().begin();
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

                    auto newExtractSlice = rewriter.create<tensor::ExtractSliceOp>(
                        extractSliceOp.getLoc(), newConv.getOutput(), newExtractOffsets,
                        extractSizes, extractStrides
                    );

                    rewriter.replaceOp(extractSliceOp, newExtractSlice.getResult());
                    rewriter.eraseOp(conv2dOp);
                    return success();
                }
            }
        }

        rewriter.replaceOp(conv2dOp, newConv.getOutput());
        return success();
    }
};

// Fix stride-2 convolutions with odd dimensions by adding explicit padding.
// NPU segmentation for stride-2 requires even H/W dimensions.
template <class TorqConvOp>
class ConvertOddDimensionStrideConvPattern : public OpRewritePattern<TorqConvOp> {
  public:
    using OpRewritePattern<TorqConvOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(TorqConvOp op, PatternRewriter &rewriter) const override {
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

        const bool validPadH = oddHeight && pads[2] == 0 && pads[3] == 0;
        const bool validPadW = oddWidth && pads[0] == 0 && pads[1] == 0;
        const bool convertToSame = validPadH || validPadW;

        if (!convertToSame && ((oddHeight && pads[2] == 0 && pads[3] == 0) ||
                               (oddWidth && pads[0] == 0 && pads[1] == 0)))
            return failure();

        auto weightType = llvm::cast<RankedTensorType>(op.getWeights().getType());
        if (!weightType)
            return failure();

        auto weightShape = weightType.getShape();
        int64_t kh, kw;
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

        SmallVector<int64_t, 4> paddedShape(shape.begin(), shape.end());
        if (oddHeight)
            paddedShape[NCHW::H]++;
        if (oddWidth)
            paddedShape[NCHW::W]++;

        auto loc = op.getLoc();
        auto elemType = inputType.getElementType();

        int32_t inputZp = op.getInputZp();
        TypedAttr fillAttr = rewriter.getIntegerAttr(elemType, inputZp);
        auto fillConst = rewriter.create<arith::ConstantOp>(loc, fillAttr);
        auto padTensor = rewriter.create<tensor::EmptyOp>(loc, paddedShape, elemType);
        auto fillOp = rewriter.create<linalg::FillOp>(
            loc, ValueRange{fillConst}, ValueRange{padTensor.getResult()}
        );

        SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : shape)
            sizes.push_back(rewriter.getIndexAttr(dim));

        auto insertOp = rewriter.create<tensor::InsertSliceOp>(
            loc, op.getInput(), fillOp.getResult(0), offsets, sizes,
            SmallVector<OpFoldResult>(4, rewriter.getIndexAttr(1))
        );

        SmallVector<int64_t, 4> newPads(pads.begin(), pads.end());
        if (convertToSame) {
            if (validPadH) {
                int64_t outh = (paddedShape[NCHW::H] + strides[0] - 1) / strides[0];
                int64_t totalPadH =
                    std::max((outh - 1) * strides[0] + kh - paddedShape[NCHW::H], int64_t(0));
                newPads[2] = totalPadH / 2;
                newPads[3] = totalPadH - newPads[2];
            }
            if (validPadW) {
                int64_t outw = (paddedShape[NCHW::W] + strides[1] - 1) / strides[1];
                int64_t totalPadW =
                    std::max((outw - 1) * strides[1] + kw - paddedShape[NCHW::W], int64_t(0));
                newPads[0] = totalPadW / 2;
                newPads[1] = totalPadW - newPads[0];
            }
        }
        else {
            if (oddHeight && newPads[3] > 0)
                newPads[3]--;
            if (oddWidth && newPads[1] > 0)
                newPads[1]--;
        }

        auto outputType = llvm::cast<RankedTensorType>(op.getResult(0).getType());
        auto origOutShape = outputType.getShape();

        SmallVector<int64_t, 4> convOutShape;
        if (convertToSame) {
            convOutShape.push_back(origOutShape[NCHW::N]);
            convOutShape.push_back(origOutShape[NCHW::C]);
            convOutShape.push_back(
                (paddedShape[NCHW::H] + newPads[2] + newPads[3] - kh) / strides[0] + 1
            );
            convOutShape.push_back(
                (paddedShape[NCHW::W] + newPads[0] + newPads[1] - kw) / strides[1] + 1
            );
        }
        else {
            convOutShape.assign(origOutShape.begin(), origOutShape.end());
        }

        auto newConv = rewriter.create<TorqConvOp>(
            loc, RankedTensorType::get(convOutShape, outputType.getElementType()),
            rewriter.create<tensor::EmptyOp>(loc, convOutShape, outputType.getElementType()),
            op.getInputZp(), op.getWeightZp(), op.getOutputZp(), op.getOutputMin(),
            op.getOutputMax(), op.getShiftFactor(), op.getGroups(),
            rewriter.getDenseI64ArrayAttr(newPads), op.getStride(), op.getDilation(),
            op.getVectorizationMode(), op.getWeights(), op.getScaleBias(), insertOp
        );

        if (convertToSame) {
            rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
                op, newConv.getOutput(),
                createVector(
                    {0, 0, validPadH ? newPads[2] : 0, validPadW ? newPads[0] : 0}, rewriter
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
            rewriter.replaceOp(op, newConv.getResult(0));
        }

        return success();
    }
};

class ValidToSamePadPass : public ValidToSamePadPassBase<ValidToSamePadPass> {
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

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(oddDimPatterns)))) {
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

        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createValidToSamePadPass() {
    return std::make_unique<ValidToSamePadPass>();
}

} // namespace mlir::syna::torq
