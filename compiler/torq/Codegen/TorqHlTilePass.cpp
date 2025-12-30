// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-tile"

namespace mlir::syna::torq {

namespace {

int getTensorSize(ShapedType shapedType) {
    int size = shapedType.getElementType().getIntOrFloatBitWidth() / 8;

    // FIXME: we must compute the required paddding
    for (int i = 0; i < shapedType.getRank(); i++) {
        size *= shapedType.getShape()[i];
    }
    return size;
}

// FIXME: should consider dtype width
int getFrameSize(RankedTensorType tensorType) {
    auto shape = tensorType.getShape();
    if (shape.size() < 4) {
        llvm::errs() << "Only 4D tensors are supported, got: " << shape.size() << "\n";
        return -1;
    }

    int byteWidth = tensorType.getElementType().getIntOrFloatBitWidth() / 8;
    return align_ceil((shape[2] + 2) * shape[3] * byteWidth, 64);
}

int getChannelCount(RankedTensorType tensorType) { return tensorType.getShape()[1]; }

class Conv1DPattern : public OpRewritePattern<torq_hl::Conv1DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::Conv1DOp convOp, PatternRewriter &rewriter) const override {
        bool isStride2conv = convOp.getStride().size() == 2 && convOp.getStride()[0] == 2 &&
                             convOp.getStride()[1] == 2;

        auto segmentationOp = convOp.getInput().getDefiningOp<torq_hl::SegmentationOp>();

        auto inType = llvm::cast<RankedTensorType>(convOp.getInput().getType());
        int inputSize = getTensorSize(inType);
        auto outType = convOp.getOutput().getType();
        int outputSize = getTensorSize(outType);

        // FIXME: compute the real needs
        auto weightsType = convOp.getWeights().getType();
        int memoryRequired = outputSize + inputSize + getTensorSize(weightsType) +
                             getTensorSize(convOp.getScaleBias().getType());

        const int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // Tile only if memory required is more than what we have
        if (memoryRequired < memoryAvailable) {
            return failure();
        }

        const int maxInputSize = memoryAvailable / 2;
        const int maxWeightAndOutputSize = memoryAvailable / 2;

        int outFrameSize = getFrameSize(outType);

        // FIXME: this is not a problem when we tile XY
        if (outFrameSize > maxWeightAndOutputSize) {
            return convOp.emitError("Frame size is too large to fit in LRAM");
        }

        const auto weightsShape = weightsType.getShape();
        const int kernelHeight = weightsShape[2];

        int maxChannelsPerTile =
            maxWeightAndOutputSize /
            (outFrameSize + weightsShape[1] * weightsShape[2] * weightsShape[3] *
                                weightsType.getElementType().getIntOrFloatBitWidth() / 8);

        // we process blocks of 4, 8 or 16 channels based on the alignment support
        maxChannelsPerTile = [=] {
            auto t = align_floor(maxChannelsPerTile, 16);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 8);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 4);
            if (t != 0)
                return t;

            return 1u;
        }();
        int outStripesCount = (inputSize + maxInputSize - 1) / maxInputSize;

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();

        auto inShape = inType.getShape();
        auto inTileSizes = createVector({inShape[0], inShape[1], inShape[2], inShape[3]}, rewriter);

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(convOp.getLoc(), outShape, outType.getElementType());

        int outPaddingLines = (kernelHeight - 1) / 2;
        int inPaddingLines = outPaddingLines;
        if (isStride2conv) {
            // Make number of padding lines even to avoid misalignment with stride
            inPaddingLines = align_ceil(inPaddingLines, 2);
        }
        const auto tileStrides = createVector({1, 1, 1, 1}, rewriter);
        const auto outTileStrides = createVector({1, 1, 1, 1, 1}, rewriter);

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto inTileOffsets = createVector({0, 0, 0, 0}, rewriter);
            auto outTileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0, 0}, rewriter);
            auto outTileSizes = createVector(
                {outShape[0], channelCount, outShape[2], outShape[3], outShape[4]}, rewriter
            );

            auto tiledWeights = rewriter.create<tensor::ExtractSliceOp>(
                convOp.getLoc(), convOp.getWeights(),
                createVector({i * maxChannelsPerTile, 0, 0, 0}, rewriter),
                createVector(
                    {channelCount, weightsShape[1], weightsShape[2], weightsShape[3]}, rewriter
                ),
                createVector({1, 1, 1, 1}, rewriter)
            );

            int scaleBiasItems = outType.getElementType().isInteger() ? 2 : 1;
            auto tiledScaleBias = rewriter.create<tensor::ExtractSliceOp>(
                convOp.getLoc(), convOp.getScaleBias(),
                createVector({scaleBiasItems * i * maxChannelsPerTile}, rewriter),
                createVector({channelCount * scaleBiasItems}, rewriter), createVector({1}, rewriter)
            );

            int remainingOutStripes = outShape[2] % outStripesCount;

            int remainingInStripes = inShape[2] % outStripesCount;

            int fixedInStripeHeight = inShape[2] / outStripesCount;
            int fixedOutStripeHeight = outShape[2] / outStripesCount;

            if (remainingOutStripes > 0) {
                outStripesCount += 1;
            }

            for (int s = 0; s < outStripesCount; s++) {

                int currentOutStripeHeight = fixedOutStripeHeight;
                int currentInStripeHeight = fixedInStripeHeight;

                if (s == outStripesCount - 1 && remainingOutStripes > 0) {
                    currentOutStripeHeight = remainingOutStripes;
                    currentInStripeHeight = remainingInStripes;
                }

                const int inPadTop = s > 0 ? inPaddingLines : 0;
                const int inPadBottom = s < outStripesCount - 1 ? inPaddingLines : 0;

                const int outPadTop = s > 0 ? outPaddingLines : 0;
                const int outPadBottom = s < outStripesCount - 1 ? outPaddingLines : 0;

                currentOutStripeHeight += outPadTop + outPadBottom;
                currentInStripeHeight += inPadTop + inPadBottom;

                auto inStripeOffsets = inTileOffsets;
                inStripeOffsets[2] = rewriter.getIndexAttr(s * fixedInStripeHeight - inPadTop);
                auto outStripeOffsets = outTileOffsets;
                outStripeOffsets[2] = rewriter.getIndexAttr(s * fixedOutStripeHeight - outPadTop);
                auto inStripeSizes = inTileSizes;
                inStripeSizes[2] = rewriter.getIndexAttr(currentInStripeHeight);
                auto outStripeSizes = outTileSizes;
                outStripeSizes[2] = rewriter.getIndexAttr(currentOutStripeHeight);

                Value inputTile;
                if (segmentationOp) {
                    auto segmentationInputTile = rewriter.create<tensor::ExtractSliceOp>(
                        convOp.getLoc(), segmentationOp.getInput(), inStripeOffsets, inStripeSizes,
                        tileStrides
                    );
                    auto segmentedType = segmentationInputTile.getType();

                    auto segmentationInitTile = rewriter.create<tensor::EmptyOp>(
                        convOp.getLoc(), segmentedType.getShape(), segmentedType.getElementType()
                    );

                    auto inputTileOp = rewriter.create<torq_hl::SegmentationOp>(
                        segmentationOp.getLoc(), segmentationInputTile.getType(),
                        segmentationInitTile.getResult(), segmentationOp.getHSegments(),
                        segmentationOp.getWSegments(), segmentationOp.getWeights(),
                        segmentationOp.getScaleBias(), segmentationInputTile
                    );
                    inputTile = inputTileOp.getOutput();
                }
                else {
                    auto inputTileOp = rewriter.create<tensor::ExtractSliceOp>(
                        convOp.getLoc(), convOp.getInput(), inStripeOffsets, inStripeSizes,
                        tileStrides
                    );
                    inputTile = inputTileOp.getResult();
                }

                auto tileType = RankedTensorType::get(
                    {outShape[0], channelCount, currentOutStripeHeight, outShape[3], outShape[4]},
                    outType.getElementType()
                );

                // Create the init vector as empty tensor.
                // This generates a simpler dependency graph than taking a subview of the original
                // init tensor and allows the optimization steps to remove redundand copyOp
                auto initTile = rewriter.create<tensor::EmptyOp>(
                    convOp.getLoc(), tileType.getShape(), tileType.getElementType()
                );

                const int32_t groups = 1;
                auto outputTileWithPad = rewriter.create<torq_hl::Conv1DOp>(
                    convOp.getLoc(), tileType, initTile.getResult(), convOp.getInputZp(),
                    convOp.getWeightZp(), convOp.getOutputZp(), convOp.getOutputMin(),
                    convOp.getOutputMax(), convOp.getShiftFactor(), groups, convOp.getPad(),
                    convOp.getStride(), convOp.getDilation(), convOp.getVectorizationMode(),
                    tiledWeights.getResult(), tiledScaleBias.getResult(), inputTile
                );

                if (convOp.getSegmentOutput()) {
                    outputTileWithPad->setAttr(
                        convOp.getSegmentOutputAttrName(), BoolAttr::get(convOp.getContext(), true)
                    );
                }

                auto lramOutStripeOffsets = createVector({0, 0, outPadTop, 0, 0}, rewriter);
                auto lramOutStripeSizes = outStripeSizes;
                lramOutStripeSizes[2] =
                    rewriter.getIndexAttr(currentOutStripeHeight - outPadTop - outPadBottom);
                auto outputTile = rewriter.create<tensor::ExtractSliceOp>(
                    convOp.getLoc(), outputTileWithPad.getOutput(), lramOutStripeOffsets,
                    lramOutStripeSizes, outTileStrides
                );

                outStripeOffsets[2] = rewriter.getIndexAttr(s * fixedOutStripeHeight);
                outStripeSizes[2] =
                    rewriter.getIndexAttr(currentOutStripeHeight - outPadTop - outPadBottom);
                outputTensor = rewriter
                                   .create<tensor::InsertSliceOp>(
                                       convOp.getLoc(), outputTile.getResult(), outputTensor,
                                       outStripeOffsets, outStripeSizes, outTileStrides
                                   )
                                   .getResult();
            }
        }

        rewriter.replaceOp(convOp, outputTensor);
        if (segmentationOp) {
            rewriter.eraseOp(segmentationOp);
        }

        return success();
    }
};

class Conv2DPattern : public OpRewritePattern<torq_hl::Conv2DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::Conv2DOp convOp, PatternRewriter &rewriter) const override {
        bool isStride2conv = convOp.getStride().size() == 2 && convOp.getStride()[0] == 2 &&
                             convOp.getStride()[1] == 2;

        auto segmentationOp = convOp.getInput().getDefiningOp<torq_hl::SegmentationOp>();

        auto inType = llvm::cast<RankedTensorType>(convOp.getInput().getType());
        int inputSize = getTensorSize(inType);
        auto outType = convOp.getOutput().getType();
        int outputSize = getTensorSize(outType);

        // FIXME: compute the real needs
        auto weightsType = convOp.getWeights().getType();
        int memoryRequired = outputSize + inputSize + getTensorSize(weightsType) +
                             getTensorSize(convOp.getScaleBias().getType());

        // Ensure that the input can fit twice in memory in case we have to segment or transpose
        // or convert
        memoryRequired = std::max(memoryRequired, 2 * inputSize);

        // Ensure that the output can fit twice in memory in case we have a convert
        float outputSizeFactor = 1;
        if (memoryRequired < 2 * outputSize) {
            outputSizeFactor = 2.0 * outputSize / memoryRequired;
            memoryRequired = 2 * outputSize;
        }

        const int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // Tile only if memory required is more than what we have
        if (memoryRequired <= memoryAvailable) {
            return failure();
        }

        const int maxInputSize = memoryAvailable / 2;
        const int maxWeightAndOutputSize = memoryAvailable / 2;

        int outFrameSize = getFrameSize(outType);

        // FIXME: this is not a problem when we tile XY
        if (outFrameSize > maxWeightAndOutputSize) {
            return convOp.emitError("Frame size is too large to fit in LRAM");
        }

        const auto weightsShape = weightsType.getShape();
        const int kernelHeight = weightsShape[2];

        int maxChannelsPerTile =
            maxWeightAndOutputSize /
            (outFrameSize + weightsShape[1] * weightsShape[2] * weightsShape[3] *
                                weightsType.getElementType().getIntOrFloatBitWidth() / 8) /
            outputSizeFactor;

        // we process blocks of 4, 8 or 16 channels based on the alignment support
        maxChannelsPerTile = [=] {
            auto t = align_floor(maxChannelsPerTile, 16);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 8);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 4);
            if (t != 0)
                return t;

            return 1u;
        }();
        int outStripesCount = (inputSize + maxInputSize - 1) / maxInputSize;

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();

        auto inShape = inType.getShape();
        auto inTileSizes = createVector({inShape[0], inShape[1], inShape[2], inShape[3]}, rewriter);

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(convOp.getLoc(), outShape, outType.getElementType());

        int outPaddingLines = (kernelHeight - 1) / 2;
        int inPaddingLines = outPaddingLines;
        if (isStride2conv) {
            // Make number of padding lines even to avoid misalignment with stride
            inPaddingLines = align_ceil(inPaddingLines, 2);
        }
        const auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto inTileOffsets = createVector({0, 0, 0, 0}, rewriter);
            auto outTileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0}, rewriter);
            auto outTileSizes =
                createVector({outShape[0], channelCount, outShape[2], outShape[3]}, rewriter);

            auto tiledWeights = rewriter.create<tensor::ExtractSliceOp>(
                convOp.getLoc(), convOp.getWeights(),
                createVector({i * maxChannelsPerTile, 0, 0, 0}, rewriter),
                createVector(
                    {channelCount, weightsShape[1], weightsShape[2], weightsShape[3]}, rewriter
                ),
                createVector({1, 1, 1, 1}, rewriter)
            );

            int scaleBiasItems = outType.getElementType().isInteger() ? 2 : 1;
            auto tiledScaleBias = rewriter.create<tensor::ExtractSliceOp>(
                convOp.getLoc(), convOp.getScaleBias(),
                createVector({scaleBiasItems * i * maxChannelsPerTile}, rewriter),
                createVector({channelCount * scaleBiasItems}, rewriter), createVector({1}, rewriter)
            );

            int remainingOutStripes = outShape[2] % outStripesCount;

            int remainingInStripes = inShape[2] % outStripesCount;

            int fixedInStripeHeight = inShape[2] / outStripesCount;
            int fixedOutStripeHeight = outShape[2] / outStripesCount;

            if (remainingOutStripes > 0) {
                outStripesCount += 1;
            }

            for (int s = 0; s < outStripesCount; s++) {

                int currentOutStripeHeight = fixedOutStripeHeight;
                int currentInStripeHeight = fixedInStripeHeight;

                if (s == outStripesCount - 1 && remainingOutStripes > 0) {
                    currentOutStripeHeight = remainingOutStripes;
                    currentInStripeHeight = remainingInStripes;
                }

                const int inPadTop = s > 0 ? inPaddingLines : 0;
                const int inPadBottom = s < outStripesCount - 1 ? inPaddingLines : 0;

                const int outPadTop = s > 0 ? outPaddingLines : 0;
                const int outPadBottom = s < outStripesCount - 1 ? outPaddingLines : 0;

                currentOutStripeHeight += outPadTop + outPadBottom;
                currentInStripeHeight += inPadTop + inPadBottom;

                auto inStripeOffsets = inTileOffsets;
                inStripeOffsets[2] = rewriter.getIndexAttr(s * fixedInStripeHeight - inPadTop);
                auto outStripeOffsets = outTileOffsets;
                outStripeOffsets[2] = rewriter.getIndexAttr(s * fixedOutStripeHeight - outPadTop);
                auto inStripeSizes = inTileSizes;
                inStripeSizes[2] = rewriter.getIndexAttr(currentInStripeHeight);
                auto outStripeSizes = outTileSizes;
                outStripeSizes[2] = rewriter.getIndexAttr(currentOutStripeHeight);

                Value inputTile;
                if (segmentationOp) {
                    auto segmentationInputTile = rewriter.create<tensor::ExtractSliceOp>(
                        convOp.getLoc(), segmentationOp.getInput(), inStripeOffsets, inStripeSizes,
                        tileStrides
                    );
                    auto segmentedType = segmentationInputTile.getType();

                    auto segmentationInitTile = rewriter.create<tensor::EmptyOp>(
                        convOp.getLoc(), segmentedType.getShape(), segmentedType.getElementType()
                    );

                    auto inputTileOp = rewriter.create<torq_hl::SegmentationOp>(
                        segmentationOp.getLoc(), segmentationInputTile.getType(),
                        segmentationInitTile.getResult(), segmentationOp.getHSegments(),
                        segmentationOp.getWSegments(), segmentationOp.getWeights(),
                        segmentationOp.getScaleBias(), segmentationInputTile
                    );
                    inputTile = inputTileOp.getOutput();
                }
                else {
                    auto inputTileOp = rewriter.create<tensor::ExtractSliceOp>(
                        convOp.getLoc(), convOp.getInput(), inStripeOffsets, inStripeSizes,
                        tileStrides
                    );
                    inputTile = inputTileOp.getResult();
                }

                auto tileType = RankedTensorType::get(
                    {outShape[0], channelCount, currentOutStripeHeight, outShape[3]},
                    outType.getElementType()
                );

                // Create the init vector as empty tensor.
                // This generates a simpler dependency graph than taking a subview of the original
                // init tensor and allows the optimization steps to remove redundand copyOp
                auto initTile = rewriter.create<tensor::EmptyOp>(
                    convOp.getLoc(), tileType.getShape(), tileType.getElementType()
                );

                const int32_t groups = 1;
                auto outputTileWithPad = rewriter.create<torq_hl::Conv2DOp>(
                    convOp.getLoc(), tileType, initTile.getResult(), convOp.getInputZp(),
                    convOp.getWeightZp(), convOp.getOutputZp(), convOp.getOutputMin(),
                    convOp.getOutputMax(), convOp.getShiftFactor(), groups, convOp.getPad(),
                    convOp.getStride(), convOp.getDilation(), convOp.getVectorizationMode(),
                    tiledWeights.getResult(), tiledScaleBias.getResult(), inputTile
                );

                if (convOp.getSegmentOutput()) {
                    outputTileWithPad->setAttr(
                        convOp.getSegmentOutputAttrName(), BoolAttr::get(convOp.getContext(), true)
                    );
                }

                auto lramOutStripeOffsets = createVector({0, 0, outPadTop, 0}, rewriter);
                auto lramOutStripeSizes = outStripeSizes;
                lramOutStripeSizes[2] =
                    rewriter.getIndexAttr(currentOutStripeHeight - outPadTop - outPadBottom);
                auto outputTile = rewriter.create<tensor::ExtractSliceOp>(
                    convOp.getLoc(), outputTileWithPad.getOutput(), lramOutStripeOffsets,
                    lramOutStripeSizes, tileStrides
                );

                outStripeOffsets[2] = rewriter.getIndexAttr(s * fixedOutStripeHeight);
                outStripeSizes[2] =
                    rewriter.getIndexAttr(currentOutStripeHeight - outPadTop - outPadBottom);
                outputTensor = rewriter
                                   .create<tensor::InsertSliceOp>(
                                       convOp.getLoc(), outputTile.getResult(), outputTensor,
                                       outStripeOffsets, outStripeSizes, tileStrides
                                   )
                                   .getResult();
            }
        }

        rewriter.replaceOp(convOp, outputTensor);
        if (segmentationOp) {
            rewriter.eraseOp(segmentationOp);
        }

        return success();
    }
};

class DepthWise2DPattern : public OpRewritePattern<torq_hl::DepthwiseConv2DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::DepthwiseConv2DOp dwOp, PatternRewriter &rewriter) const override {

        auto segmentationOp = dwOp.getInput().getDefiningOp<torq_hl::SegmentationOp>();

        auto outType = dwOp.getOutput().getType();
        auto inType = llvm::cast<RankedTensorType>(dwOp.getInput().getType());

        int outputSize = getTensorSize(outType);
        int inputSize = getTensorSize(inType);

        // Ensure that the input can fit twice in memory in case we have to segment or transpose
        // or convert
        int memoryRequired = std::max(outputSize + inputSize, 2 * inputSize);

        int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // FIXME: For some reason memoryAvailable needs to be maximum 300KB for
        // dw_f5_s2_128x128x72.mlir test case to pass.
        if (memoryAvailable >= 1024 * 300) {
            memoryAvailable = 1024 * 300;
        }

        // Tile only if memory required is more than what we have
        if (memoryRequired <= memoryAvailable) {
            return failure();
        }

        int channelDataSize =
            std::max(getFrameSize(outType) + getFrameSize(inType), 2 * getFrameSize(inType));

        if (channelDataSize > memoryAvailable) {
            return dwOp.emitError("Frame size is too large to fit in LRAM");
        }

        int maxChannelsPerTile = memoryAvailable / channelDataSize;

        // we process blocks of 4, 8 or 16 channels based on the alignment support
        maxChannelsPerTile = [=] {
            auto t = align_floor(maxChannelsPerTile, 16);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 8);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 4);
            if (t != 0)
                return t;

            return 1u;
        }();

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();
        auto inShape = dwOp.getInput().getType().getShape();

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(dwOp.getLoc(), outShape, outType.getElementType());

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto tileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0}, rewriter);

            auto outTileSizes =
                createVector({outShape[0], channelCount, outShape[2], outShape[3]}, rewriter);
            auto inTileSizes =
                createVector({inShape[0], channelCount, inShape[2], inShape[3]}, rewriter);

            auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

            Value inputTile;

            if (segmentationOp) {

                auto segmentationInputTile = rewriter.create<tensor::ExtractSliceOp>(
                    dwOp.getLoc(), segmentationOp.getInput(), tileOffsets, inTileSizes, tileStrides
                );

                auto segmentedType = segmentationInputTile.getType();
                auto segmentationInitTile = rewriter.create<tensor::EmptyOp>(
                    dwOp.getLoc(), segmentedType.getShape(), segmentedType.getElementType()
                );

                inputTile = rewriter
                                .create<torq_hl::SegmentationOp>(
                                    segmentationOp.getLoc(), segmentationInputTile.getType(),
                                    segmentationInitTile.getResult(), segmentationOp.getHSegments(),
                                    segmentationOp.getWSegments(), segmentationOp.getWeights(),
                                    segmentationOp.getScaleBias(), segmentationInputTile
                                )
                                .getOutput();
            }
            else {
                inputTile = rewriter.create<tensor::ExtractSliceOp>(
                    dwOp.getLoc(), dwOp.getInput(), tileOffsets, inTileSizes, tileStrides
                );
            }

            auto tileType = RankedTensorType::get(
                {outShape[0], channelCount, outShape[2], outShape[3]}, outType.getElementType()
            );

            // Create the init vector as empty tensor.
            // This generates a simpler dependency graph than taking a subview of the original
            // init tensor and allows the optimization steps to remove redundand copyOp
            auto initTile = rewriter.create<tensor::EmptyOp>(
                dwOp.getLoc(), tileType.getShape(), tileType.getElementType()
            );

            auto weightsShape = dwOp.getWeights().getType().getShape();

            SmallVector<int64_t> weightOffsetValues(weightsShape.size(), 0);
            weightOffsetValues[0] = i * maxChannelsPerTile;
            // Linalg depthwise conv2d has 3D weights: OHW
            SmallVector<int64_t> weightSizeValues =
                weightsShape.size() == 4
                    ? SmallVector<int64_t>(
                          {channelCount, weightsShape[1], weightsShape[2], weightsShape[3]}
                      )
                    : SmallVector<int64_t>({channelCount, weightsShape[1], weightsShape[2]});
            SmallVector<int64_t> weightStrideValues(weightsShape.size(), 1);

            auto tiledWeights = rewriter.create<tensor::ExtractSliceOp>(
                dwOp.getLoc(), dwOp.getWeights(), createVector(weightOffsetValues, rewriter),
                createVector(weightSizeValues, rewriter), createVector(weightStrideValues, rewriter)
            );

            int scaleBiasItems = outType.getElementType().isInteger() ? 2 : 1;
            auto tiledScaleBias = rewriter.create<tensor::ExtractSliceOp>(
                dwOp.getLoc(), dwOp.getScaleBias(),
                createVector({scaleBiasItems * i * maxChannelsPerTile}, rewriter),
                createVector({channelCount * scaleBiasItems}, rewriter), createVector({1}, rewriter)
            );

            auto outputTile = rewriter.create<torq_hl::DepthwiseConv2DOp>(
                dwOp.getLoc(), tileType, initTile.getResult(), dwOp.getInputZp(),
                dwOp.getWeightZp(), dwOp.getOutputZp(), dwOp.getOutputMin(), dwOp.getOutputMax(),
                dwOp.getShiftFactor(), channelCount, dwOp.getPad(), dwOp.getStride(),
                dwOp.getDilation(), dwOp.getVectorizationMode(), tiledWeights.getResult(),
                tiledScaleBias.getResult(), inputTile
            );

            if (dwOp.getSegmentOutput()) {
                outputTile->setAttr(
                    dwOp.getSegmentOutputAttrName(), BoolAttr::get(dwOp.getContext(), true)
                );
            }

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   dwOp.getLoc(), outputTile.getOutput(), outputTensor, tileOffsets,
                                   outTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(dwOp, outputTensor);

        if (segmentationOp) {
            rewriter.eraseOp(segmentationOp);
        }

        return success();
    }
};

class FullyConnectedPattern : public OpRewritePattern<torq_hl::FullyConnectedOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::FullyConnectedOp fcOp, PatternRewriter &rewriter) const override {
        auto outputType = fcOp.getOutput().getType();
        auto outputShape = outputType.getShape();
        size_t outputSize = getShapeTypeDataSize(outputType);

        auto inputType = fcOp.getInput().getType();
        size_t inputSize = getShapeTypeDataSize(inputType);

        size_t weightsSize = getShapeTypeDataSize(fcOp.getWeights().getType());
        auto weightShape = fcOp.getWeights().getType().getShape();

        size_t scaleBiasSize = getShapeTypeDataSize(fcOp.getScaleBias().getType());

        int memoryRequired = inputSize + outputSize + weightsSize + scaleBiasSize;

        // Tile only if memory required is more than what we have
        const uint32_t mem_max_size = TorqHw::get().getAvailableMemoryForTiling();

        if (memoryRequired < mem_max_size) {
            return failure();
        }

        int maxChannelsPerTile = (mem_max_size - inputSize) /
                                 div_ceil(outputSize + weightsSize + scaleBiasSize, outputShape[1]);

        // tiles needs to be a multiple of 64, we round down to avoid exceeding the available memory
        int channelsPerTile = align_floor(maxChannelsPerTile, 64);
        if (maxChannelsPerTile < 64) {
            fcOp.emitOpError("Unable to tile FullyConnectedOp in the available memory");
        }

        int trailingChannels = outputShape[1] % channelsPerTile;

        int totalFullTiles = outputShape[1] / channelsPerTile;

        // since the weights are organized in blocks of 64 channels, we should never end up in this
        // situation
        assert(totalFullTiles * channelsPerTile + trailingChannels == outputShape[1]);

        int totalTiles = totalFullTiles + (trailingChannels > 0 ? 1 : 0);

        Value outputTensor = rewriter.create<tensor::EmptyOp>(
            fcOp.getLoc(), outputShape, outputType.getElementType()
        );

        for (int i = 0; i < totalTiles; i++) {

            int weigthOffset = i * (channelsPerTile);
            int tileChannels = channelsPerTile;

            if (trailingChannels > 0 && i == totalTiles - 1) {
                tileChannels = trailingChannels;
            }

            auto tileStrides = createVector({1, 1}, rewriter);

            auto outputTileOffsets = createVector({0, i * channelsPerTile}, rewriter);
            auto outputTileSizes = createVector({outputShape[0], tileChannels}, rewriter);
            auto outputTileType =
                RankedTensorType::get({outputShape[0], tileChannels}, outputType.getElementType());

            auto initTile = rewriter.create<tensor::EmptyOp>(
                fcOp.getLoc(), outputTileType.getShape(), outputTileType.getElementType()
            );

            auto weightTile = rewriter.create<tensor::ExtractSliceOp>(
                fcOp.getLoc(), fcOp.getWeights(), createVector({weigthOffset, 0}, rewriter),
                createVector({tileChannels, weightShape[1]}, rewriter),
                createVector({1, 1}, rewriter)
            );

            int scaleBiasItems = outputType.getElementType().isInteger() ? 2 : 1;
            auto scaleBiasTile = rewriter.create<tensor::ExtractSliceOp>(
                fcOp.getLoc(), fcOp.getScaleBias(),
                createVector({i * channelsPerTile * scaleBiasItems}, rewriter),
                createVector({tileChannels * scaleBiasItems}, rewriter), createVector({1}, rewriter)
            );

            auto outputTile = rewriter.create<torq_hl::FullyConnectedOp>(
                fcOp.getLoc(), outputTileType, initTile.getResult(), fcOp.getInputZp(),
                fcOp.getWeightZp(), fcOp.getOutputZp(), fcOp.getOutputMin(), fcOp.getOutputMax(),
                fcOp.getShiftFactor(), fcOp.getVectorizationMode(), weightTile.getResult(),
                scaleBiasTile.getResult(), fcOp.getInput()
            );

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   fcOp.getLoc(), outputTile.getOutput(), outputTensor,
                                   outputTileOffsets, outputTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(fcOp, outputTensor);

        return success();
    }
};

class AddPattern : public OpRewritePattern<torq_hl::AddOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::AddOp op, PatternRewriter &rewriter) const override {

        auto outType = op.getOutput().getType();
        auto inType = llvm::cast<RankedTensorType>(op.getInput1().getType());

        int outputSize = getTensorSize(outType);
        int inputSize = getTensorSize(inType);

        // FIXME: compute the real needs
        int memoryRequired = outputSize + inputSize * 2;

        const int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // Tile only if memory required is more than what we have
        if (memoryRequired < memoryAvailable) {
            return failure();
        }

        int channelDataSize = getFrameSize(outType) + getFrameSize(inType) * 2;
        if (channelDataSize < 0) {
            return failure();
        }

        if (channelDataSize > memoryAvailable) {
            return op.emitError("Frame size is too large to fit in LRAM");
        }

        int maxChannelsPerTile = memoryAvailable / channelDataSize;
        maxChannelsPerTile = align_floor(maxChannelsPerTile, 4);

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();
        auto inShape = op.getInput1().getType().getShape();

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), outShape, outType.getElementType());

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto tileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0}, rewriter);

            auto outTileSizes =
                createVector({outShape[0], channelCount, outShape[2], outShape[3]}, rewriter);
            auto inTileSizes =
                createVector({inShape[0], channelCount, inShape[2], inShape[3]}, rewriter);

            auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

            Value input1Tile, input2Tile;

            input1Tile = rewriter.create<tensor::ExtractSliceOp>(
                op.getLoc(), op.getInput1(), tileOffsets, inTileSizes, tileStrides
            );
            input2Tile = rewriter.create<tensor::ExtractSliceOp>(
                op.getLoc(), op.getInput2(), tileOffsets, inTileSizes, tileStrides
            );

            auto tileType = RankedTensorType::get(
                {outShape[0], channelCount, outShape[2], outShape[3]}, outType.getElementType()
            );

            auto initTile = rewriter.create<tensor::EmptyOp>(
                op.getLoc(), tileType.getShape(), tileType.getElementType()
            );

            auto outputTile = rewriter.create<torq_hl::AddOp>(
                op.getLoc(), tileType, initTile.getResult(), op.getName(), op.getInputZp(),
                op.getOutputZp(), op.getOutputMin(), op.getOutputMax(), op.getShiftFactor(),
                op.getWeights(), op.getScaleBias(), input1Tile, input2Tile
            );

            if (op.getSegmentOutput()) {
                outputTile->setAttr(
                    op.getSegmentOutputAttrName(), BoolAttr::get(op.getContext(), true)
                );
            }

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   op.getLoc(), outputTile.getOutput(), outputTensor, tileOffsets,
                                   outTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class DepthToSpacePattern : public OpRewritePattern<torq_hl::DepthToSpaceOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::DepthToSpaceOp op, PatternRewriter &rewriter) const override {
        auto outType = op.getOutput().getType();
        auto inType = llvm::cast<RankedTensorType>(op.getInput().getType());
        int outputSize = getTensorSize(outType);
        int inputSize = getTensorSize(inType);
        int memoryRequired = outputSize + inputSize;
        const int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // Tile only if memory required is more than what we have
        if (memoryRequired < memoryAvailable) {
            return failure();
        }

        auto inShape = op.getInput().getType().getShape();
        auto outShape = outType.getShape();
        int channels = getChannelCount(outType);
        int inputChannels = getChannelCount(inType);
        int spaceFactor = inputChannels / channels;
        assert(spaceFactor == op.getBlockSize() * op.getBlockSize());

        int channelDataSize = getFrameSize(outType) + getFrameSize(inType) * spaceFactor;
        if (channelDataSize > memoryAvailable) {
            return op.emitError("Frame size is too large to fit in LRAM");
        }

        int maxChPerTile = memoryAvailable / channelDataSize;
        int fullTilesCount = channels / maxChPerTile;
        int remainingChannels = channels % maxChPerTile;
        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        // Output tensor starts empty, each tile will write a slice of it
        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), outShape, outType.getElementType());

        for (int i = 0; i < totalTiles; i++) {
            int chCount = maxChPerTile;
            if (i == totalTiles - 1 && remainingChannels > 0) {
                chCount = remainingChannels;
            }

            auto tileOffsets = createVector({0, i * maxChPerTile, 0, 0}, rewriter);
            auto inTileOffsets = createVector({0, i * maxChPerTile * spaceFactor, 0, 0}, rewriter);

            auto outTileSizes =
                createVector({outShape[0], chCount, outShape[2], outShape[3]}, rewriter);
            auto inTileSizes =
                createVector({inShape[0], chCount * spaceFactor, inShape[2], inShape[3]}, rewriter);
            auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

            Value input1Tile = rewriter.create<tensor::ExtractSliceOp>(
                op.getLoc(), op.getInput(), inTileOffsets, inTileSizes, tileStrides
            );
            auto tileType = RankedTensorType::get(
                {outShape[0], chCount, outShape[2], outShape[3]}, outType.getElementType()
            );
            auto initTile = rewriter.create<tensor::EmptyOp>(
                op.getLoc(), tileType.getShape(), tileType.getElementType()
            );
            auto outputTile = rewriter.create<torq_hl::DepthToSpaceOp>(
                op.getLoc(), tileType, initTile.getResult(), op.getBlockSize(), op.getModeType(),
                op.getWeights(), input1Tile
            );

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   op.getLoc(), outputTile.getOutput(), outputTensor, tileOffsets,
                                   outTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class MaxPoolPattern : public OpRewritePattern<torq_hl::MaxPool2dOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) const override {

        auto segmentationOp = op.getInput().getDefiningOp<torq_hl::SegmentationOp>();

        auto outType = op.getOutput().getType();
        auto inType = llvm::cast<RankedTensorType>(op.getInput().getType());

        int outputSize = getTensorSize(outType);
        int inputSize = getTensorSize(inType);

        bool isStride2 =
            op.getStride().size() == 2 && op.getStride()[0] == 2 && op.getStride()[1] == 2;

        // FIXME: compute the real needs
        int memoryRequired = outputSize + inputSize;
        // if maxpool stride is 2, output might smaller than input
        // segmentation working on maxpool input, and segmentation total memory size is inputSize *
        // 2 since segmentation op inserted after maxpool tiling, we need to make sure tiled memory
        // size of maxpool and its segementation both meet memory requirement
        if (isStride2) {
            memoryRequired = inputSize + inputSize;
        }

        const int memoryAvailable = TorqHw::get().getAvailableMemoryForTiling();

        // Tile only if memory required is more than what we have
        if (memoryRequired < memoryAvailable) {
            return failure();
        }

        int channelDataSize = getFrameSize(outType) + getFrameSize(inType);
        if (isStride2) {
            channelDataSize = getFrameSize(inType) + getFrameSize(inType);
        }

        if (channelDataSize > memoryAvailable) {
            return op.emitError("Frame size is too large to fit in LRAM");
        }

        int maxChannelsPerTile = memoryAvailable / channelDataSize;

        // we process blocks of 4, 8 or 16 channels based on the alignment support
        maxChannelsPerTile = [=] {
            auto t = align_floor(maxChannelsPerTile, 16);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 8);
            if (t != 0)
                return t;
            t = align_floor(maxChannelsPerTile, 4);
            if (t != 0)
                return t;

            return 1u;
        }();

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();
        auto inShape = op.getInput().getType().getShape();

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), outShape, outType.getElementType());

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto tileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0}, rewriter);

            auto outTileSizes =
                createVector({outShape[0], channelCount, outShape[2], outShape[3]}, rewriter);
            auto inTileSizes =
                createVector({inShape[0], channelCount, inShape[2], inShape[3]}, rewriter);

            auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

            Value inputTile;

            if (segmentationOp) {

                auto segmentationInputTile = rewriter.create<tensor::ExtractSliceOp>(
                    op.getLoc(), segmentationOp.getInput(), tileOffsets, inTileSizes, tileStrides
                );

                auto segmentedType = segmentationInputTile.getType();
                auto segmentationInitTile = rewriter.create<tensor::EmptyOp>(
                    op.getLoc(), segmentedType.getShape(), segmentedType.getElementType()
                );

                inputTile = rewriter
                                .create<torq_hl::SegmentationOp>(
                                    segmentationOp.getLoc(), segmentationInputTile.getType(),
                                    segmentationInitTile.getResult(), segmentationOp.getHSegments(),
                                    segmentationOp.getWSegments(), segmentationOp.getWeights(),
                                    segmentationOp.getScaleBias(), segmentationInputTile
                                )
                                .getOutput();
            }
            else {
                inputTile = rewriter.create<tensor::ExtractSliceOp>(
                    op.getLoc(), op.getInput(), tileOffsets, inTileSizes, tileStrides
                );
            }

            auto tileType = RankedTensorType::get(
                {outShape[0], channelCount, outShape[2], outShape[3]}, outType.getElementType()
            );

            auto initTile = rewriter.create<tensor::EmptyOp>(
                op.getLoc(), tileType.getShape(), tileType.getElementType()
            );

            auto outputTile = rewriter.create<torq_hl::MaxPool2dOp>(
                op.getLoc(), tileType, initTile.getResult(), op.getInputZp(), op.getStride(),
                op.getPad(), op.getKernel(), op.getWeights(), op.getScaleBias(), inputTile
            );

            if (op.getSegmentOutput()) {
                outputTile->setAttr(
                    op.getSegmentOutputAttrName(), BoolAttr::get(op.getContext(), true)
                );
            }

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   op.getLoc(), outputTile.getOutput(), outputTensor, tileOffsets,
                                   outTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(op, outputTensor);

        if (segmentationOp) {
            rewriter.eraseOp(segmentationOp);
        }

        return success();
    }
};

class ScatterPattern : public OpRewritePattern<torq_hl::ScatterOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::ScatterOp scatterOp, PatternRewriter &rewriter) const override {
        auto inputShape = mlir::cast<RankedTensorType>(scatterOp.getInput().getType()).getShape();

        auto outType = scatterOp.getInit().getType();
        auto outShape = mlir::cast<RankedTensorType>(outType).getShape();

        if (outShape[1] == 1) {
            return failure();
        }

        auto outRankedType = mlir::cast<RankedTensorType>(outType);
        Value outputTensor = createInitTensor(
            scatterOp, rewriter, outRankedType
        ); // Empty init tensor for stacking up tiles

        for (int i = 0; i < inputShape[1]; ++i) {
            auto inputTileOffsets = createVector({0, i, 0}, rewriter);
            auto inputTileSizes = createVector({1, 1, inputShape[2]}, rewriter);
            auto tileStrides = createVector({1, 1, 1}, rewriter);
            auto inputTile = rewriter.create<tensor::ExtractSliceOp>(
                scatterOp.getLoc(), scatterOp.getInput(), inputTileOffsets, inputTileSizes,
                tileStrides
            );

            auto tileType =
                RankedTensorType::get({outShape[0], 1, outShape[2]}, outType.getElementType());

            auto outputTileOffsets = createVector({0, i, 0}, rewriter);
            auto outputTileSizes = createVector({outShape[0], 1, outShape[2]}, rewriter);
            auto outputTileStrides = createVector({1, 1, 1}, rewriter);

            auto outTile = rewriter.create<tensor::ExtractSliceOp>(
                scatterOp.getLoc(), scatterOp.getInit(), outputTileOffsets, outputTileSizes,
                outputTileStrides
            );

            auto outputTile = rewriter.create<torq_hl::ScatterOp>(
                scatterOp.getLoc(), tileType, outTile.getResult(), scatterOp.getIndices(),
                inputTile, scatterOp.getScaleBias()
            );

            outputTensor =
                rewriter
                    .create<tensor::InsertSliceOp>(
                        scatterOp.getLoc(),
                        outputTile.getOutput(), // outputTile.getOutput() contains updated values_in
                        outputTensor, outputTileOffsets, outputTileSizes, outputTileStrides
                    )
                    .getResult();
        }
        rewriter.replaceOp(scatterOp, outputTensor);
        return success();
    }
};

class ResizeNearestPattern : public OpRewritePattern<torq_hl::ResizeNearestNeighborOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::ResizeNearestNeighborOp op, PatternRewriter &rewriter) const override {

        auto outType = op.getOutput().getType();
        auto inType = llvm::cast<RankedTensorType>(op.getInput().getType());

        int outputSize = getTensorSize(outType);
        int inputSize = getTensorSize(inType);

        // FIXME: compute the real needs
        int memoryRequired = outputSize + inputSize;

        const int memoryAvailable =
            TorqHw::get().getAvailableMemoryForTiling(); // leave some margin

        // Tile only if memory required is more than what we have
        if (memoryRequired < memoryAvailable) {
            return failure();
        }

        int channelDataSize = getFrameSize(outType) + getFrameSize(inType);
        if (channelDataSize < 0) {
            return failure();
        }

        if (channelDataSize > memoryAvailable) {
            return op.emitError("Frame size is too large to fit in LRAM");
        }

        int maxChannelsPerTile = 64;

        int channels = getChannelCount(outType);

        int fullTilesCount = channels / maxChannelsPerTile;

        int remainingChannels = channels % maxChannelsPerTile;

        int totalTiles = fullTilesCount + (remainingChannels > 0 ? 1 : 0);

        auto outShape = outType.getShape();
        auto inShape = op.getInput().getType().getShape();

        Value outputTensor =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), outShape, outType.getElementType());

        for (int i = 0; i < totalTiles; i++) {
            int channelCount = maxChannelsPerTile;

            if (i == totalTiles - 1 && remainingChannels > 0) {
                channelCount = remainingChannels;
            }

            auto tileOffsets = createVector({0, i * maxChannelsPerTile, 0, 0}, rewriter);

            auto outTileSizes =
                createVector({outShape[0], channelCount, outShape[2], outShape[3]}, rewriter);
            auto inTileSizes =
                createVector({inShape[0], channelCount, inShape[2], inShape[3]}, rewriter);

            auto tileStrides = createVector({1, 1, 1, 1}, rewriter);

            Value inputTile = rewriter.create<tensor::ExtractSliceOp>(
                op.getLoc(), op.getInput(), tileOffsets, inTileSizes, tileStrides
            );

            auto tileType = RankedTensorType::get(
                {outShape[0], channelCount, outShape[2], outShape[3]}, outType.getElementType()
            );

            auto initTile = rewriter.create<tensor::EmptyOp>(
                op.getLoc(), tileType.getShape(), tileType.getElementType()
            );

            auto outputTile = rewriter.create<torq_hl::ResizeNearestNeighborOp>(
                op.getLoc(), tileType, initTile.getResult(), op.getScaleFactor(), inputTile
            );

            outputTensor = rewriter
                               .create<tensor::InsertSliceOp>(
                                   op.getLoc(), outputTile.getOutput(), outputTensor, tileOffsets,
                                   outTileSizes, tileStrides
                               )
                               .getResult();
        }

        rewriter.replaceOp(op, outputTensor);

        return success();
    }
};

class TorqHlTilePass : public TorqHlTileBase<TorqHlTilePass> {
  public:
    using TorqHlTileBase::TorqHlTileBase;

    void runOnOperation() override;
};

void TorqHlTilePass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<Conv1DPattern>(ctx);
    patterns.add<Conv2DPattern>(ctx);
    patterns.add<DepthWise2DPattern>(ctx);
    patterns.add<FullyConnectedPattern>(ctx);
    patterns.add<MaxPoolPattern>(ctx);
    patterns.add<AddPattern>(ctx);
    patterns.add<DepthToSpacePattern>(ctx);
    patterns.add<ScatterPattern>(ctx);
    patterns.add<ResizeNearestPattern>(ctx);

    tensor::ControlConstantExtractSliceFusionFn controlFn = [](tensor::ExtractSliceOp op) {
        return true;
    };

    tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHlTilePass() {
    return std::make_unique<TorqHlTilePass>();
}

} // namespace mlir::syna::torq
