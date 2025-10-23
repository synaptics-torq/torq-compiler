// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/EncodingRequirements.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-slicing"

namespace mlir::syna::torq {

namespace {

// Number of slices in which each operation is splitted
constexpr int kSliceCount = HwInfo::slice_count;

SmallVector<OpFoldResult>
createVector(const SmallVector<int64_t> &values, PatternRewriter &rewriter) {
    SmallVector<OpFoldResult> result;
    for (int i = 0; i < values.size(); i++) {
        result.push_back(rewriter.getIndexAttr(values[i]));
    }
    return result;
}

SmallVector<OpFoldResult> createVector(int64_t value, int size, PatternRewriter &rewriter) {
    SmallVector<OpFoldResult> result;
    for (int i = 0; i < size; i++) {
        result.push_back(rewriter.getIndexAttr(value));
    }
    return result;
}

// Create an empty tensor with the given type
Value createEmptyTensor(RankedTensorType tensorType, Location loc, PatternRewriter &rewriter) {
    return rewriter
        .create<tensor::EmptyOp>(
            loc, tensorType.getShape(), tensorType.getElementType(), tensorType.getEncoding()
        )
        ->getResult(0);
}

// Extract a slice from a tensor
static Value getSlice(
    Value value, const SmallVector<OpFoldResult> &offsets, const SmallVector<OpFoldResult> &sizes,
    const SmallVector<OpFoldResult> &strides, PatternRewriter &rewriter
) {
    auto sliceOp =
        rewriter.create<tensor::ExtractSliceOp>(value.getLoc(), value, offsets, sizes, strides);
    return sliceOp.getResult();
}

// Extract a slice from a tensor with constant sizes
static Value getSlice(
    Value value, const SmallVector<OpFoldResult> &offsets, const SmallVector<int64_t> &sizes,
    PatternRewriter &rewriter
) {
    auto strides = createVector(SmallVector<int64_t>(sizes.size(), 1), rewriter);
    return getSlice(value, offsets, createVector(sizes, rewriter), strides, rewriter);
}

// Create a forall loop with the given count
scf::ForallOp
createForallOp(int count, RankedTensorType returnType, Location loc, PatternRewriter &rewriter) {
    OpFoldResult zero = rewriter.getIndexAttr(0);
    OpFoldResult one = rewriter.getIndexAttr(1);
    OpFoldResult parallel_count = rewriter.getIndexAttr(count);
    auto outputTensor = createEmptyTensor(returnType, loc, rewriter);
    return rewriter.create<scf::ForallOp>(
        loc, ArrayRef<OpFoldResult>{zero}, ArrayRef<OpFoldResult>{parallel_count},
        ArrayRef<OpFoldResult>{one}, ValueRange({outputTensor}),
        ArrayAttr::get(rewriter.getContext(), {})
    );
}

FailureOr<torq_hl::ConvertOp> getConvertResultToXram(Operation *op) {

    if (!op->getResult(0).hasOneUse()) {
        return failure();
    }

    auto it = op->getResult(0).user_begin();

    auto outputXramConvertOp = dyn_cast<torq_hl::ConvertOp>(*it);

    if (!outputXramConvertOp) {
        return failure();
    }

    auto outputXramType = dyn_cast<RankedTensorType>(outputXramConvertOp.getResult(0).getType());

    if (!outputXramType) {
        return failure();
    }

    if (getEncodingMemorySpace(outputXramType) != torq_hl::MemorySpace::Xram) {
        return failure();
    }

    return outputXramConvertOp;
}

// Compute the result of an affine expression applied to the given argument.
// The code to evaluate the affine expression is generated at the current cursor position.
OpFoldResult affineEval(AffineExpr affineExpr, Value arg, PatternRewriter &rewriter) {
    Location loc = UnknownLoc::get(rewriter.getContext());
    mlir::AffineMap affineMap = mlir::AffineMap::get(1, 0, affineExpr, rewriter.getContext());
    auto affineApplyOp = rewriter.create<affine::AffineApplyOp>(loc, affineMap, ValueRange{arg});
    return affineApplyOp->getResult(0);
}

class FullyConnectedPattern : public OpRewritePattern<torq_hl::FullyConnectedOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = op.getContext();

        // Do nothing if the containing operation is a forall (already sliced) or slicing disabled
        if (dyn_cast<scf::ForallOp>(op->getParentOp()) || !kSliceCount) {
            return failure();
        }

        // try to find the XRAM buffer where the output is stored, if we can't find it do not slice
        auto maybeConvertToXram = getConvertResultToXram(op);

        if (failed(maybeConvertToXram)) {
            return rewriter.notifyMatchFailure(
                op, "cannot find convert to XRAM operation for the output"
            );
        }

        // Compute number of channels for each slice
        auto outType = op.getOutput().getType();
        auto outShape = outType.getShape();
        int channels = outShape[1];
        const auto weightsType = op.getWeights().getType();
        const auto weightsShape = weightsType.getShape();
        const auto weightsGrouping = weightsShape[weightsShape.size() - 1];
        // Make channel count for each slice a multiple of the weights grouping
        channels /= kSliceCount;
        if (channels % weightsGrouping) {
            // TODO: support non-multiple of weightsGrouping channel counts
            return failure();
        }
        channels = align_ceil(channels, weightsGrouping);

        // Split the operation on each slice using a parallel forall loop
        scf::ForallOp forallOp =
            createForallOp(kSliceCount, maybeConvertToXram->getOutput().getType(), loc, rewriter);

        // Get the loop induction variable (iteration index) of the forall operation
        auto iv = forallOp.getLoopInductionVars().value()[0];

        // An affine expression returning its first argument
        AffineExpr arg = getAffineDimExpr(0, ctx);

        // Position the write cursor at the begining of the loop body
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forallOp.getBody());

        // Compute the weights tile for each iteration
        SmallVector<int64_t> weightTileShape(weightsShape);
        weightTileShape[0] = channels / weightsGrouping;
        SmallVector<OpFoldResult> weightOffset = createVector(0, weightsShape.size(), rewriter);
        weightOffset[0] = affineEval(arg * (channels / weightsGrouping), iv, rewriter);
        auto weightsTile = getSlice(op.getWeights(), weightOffset, weightTileShape, rewriter);

        // Compute the bias tile for each iteration
        SmallVector<OpFoldResult> biasOffset = createVector({0}, rewriter);
        biasOffset[0] = affineEval(arg * (channels * 2), iv, rewriter);
        auto scaleBiasTile = getSlice(op.getScaleBias(), biasOffset, {channels * 2}, rewriter);

        // Perform the sliced operation using the sliced weights and scale/bias
        // /!\ FIXME: if the original channel count is not a multiple of the weights grouping,
        // we are accessing out-of-bounds memory in the in, out and scale/bias tensors.
        SmallVector<int64_t> outTileShape(outShape);
        outTileShape[1] = channels;

        // create an unencoded (therefore XRAM) output type, we will encode it later
        auto outTileType = RankedTensorType::get(outTileShape, outType.getElementType(), nullptr);
        SmallVector<OpFoldResult> outTileOffsets = createVector(0, outTileShape.size(), rewriter);
        outTileOffsets[1] = affineEval(arg * channels, iv, rewriter);
        auto outTileSizes = createVector(outTileShape, rewriter);
        const auto outTileStrides = createVector(1, outTileShape.size(), rewriter);

        auto initTile = createEmptyTensor(outTileType, loc, rewriter);
        auto outTileOp = rewriter.create<torq_hl::FullyConnectedOp>(
            loc, TypeRange{outTileType},
            ValueRange{initTile, weightsTile, scaleBiasTile, op.getInput()}, op->getAttrs()
        );

        // Merge the result in the destination tensor
        rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
        rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, outTileOp.getResult(0), forallOp.getRegionIterArgs()[0], outTileOffsets,
            outTileSizes, outTileStrides
        );

        // encode the new FullyConnected operation, this will not change the input (as it is already
        // encoded) but will insert a convert op after the FullyConnected operation to set the
        // output in the correct encoding

        auto encodingRequirements = outTileOp.getKernelEncoding();
        auto destXramTile = getSlice(
            forallOp.getRegionIterArgs()[0], outTileOffsets, {outTileSizes}, outTileStrides,
            rewriter
        );
        if (failed(torq_hl::encodeKernelInputOutputs(
                outTileOp, encodingRequirements, rewriter, destXramTile
            ))) {
            llvm::report_fatal_error("Failed to encode sliced FullyConnected operation");
        }

        // replace the convert to XRAM operation with the result of the forall operation
        // that that returns an XRAM tensor
        rewriter.replaceOp(*maybeConvertToXram, forallOp->getResult(0));

        // Erase the fully connected operation that has no more users (we checked the convert
        // op was the only one at the beginning of the function)
        rewriter.eraseOp(op);

        return success();
    }
};

class Conv2DPattern : public OpRewritePattern<torq_hl::Conv2DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::Conv2DOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = op.getContext();

        // Do nothing if the containing operation is a forall (already sliced) or slicing disabled
        if (dyn_cast<scf::ForallOp>(op->getParentOp()) || !kSliceCount) {
            return failure();
        }

        // try to find the XRAM buffer where the output is stored, if we can't find it do not slice
        auto maybeConvertToXram = getConvertResultToXram(op);

        if (failed(maybeConvertToXram)) {
            return rewriter.notifyMatchFailure(
                op, "cannot find convert to XRAM operation for the output"
            );
        }

        // Compute number of channels for each slice
        // Currently we only support slicing if the channel count is a multiple of
        // (kSliceCount * channel grouping)
        auto outType = op.getOutput().getType();
        auto outShape = outType.getShape();
        int channels = outShape[1];
        const auto weightsType = op.getWeights().getType();
        const auto weightsShape = weightsType.getShape();
        const auto weightsGrouping = weightsShape[weightsShape.size() - 1];
        if (channels % (kSliceCount * weightsGrouping)) {
            return failure();
        }
        channels /= kSliceCount;

        // Split the operation on each slice using a parallel forall loop
        scf::ForallOp forallOp =
            createForallOp(kSliceCount, maybeConvertToXram->getOutput().getType(), loc, rewriter);

        // Get the loop induction variable (iteration index) of the forall operation
        auto iv = forallOp.getLoopInductionVars().value()[0];

        // An affine expression returning its first argument
        AffineExpr arg = getAffineDimExpr(0, ctx);

        // Position the write cursor at the begining of the loop body
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forallOp.getBody());

        // Compute the weights tile for each iteration
        SmallVector<int64_t> weightTileShape(weightsShape);
        weightTileShape[0] = channels / weightsGrouping;
        SmallVector<OpFoldResult> weightOffset = createVector(0, weightsShape.size(), rewriter);
        weightOffset[0] = affineEval(arg * (channels / weightsGrouping), iv, rewriter);
        auto weightsTile = getSlice(op.getWeights(), weightOffset, weightTileShape, rewriter);

        // Compute the bias tile for each iteration
        SmallVector<OpFoldResult> biasOffset = createVector({0}, rewriter);

        // We need to compute the tile of the bias/scale tensor based on the original
        // size because in some cases we have 2 values per channel and in some cases 1 value
        auto biasScaleTileSize = op.getScaleBias().getType().getShape()[0] / kSliceCount;

        biasOffset[0] = affineEval(arg * biasScaleTileSize, iv, rewriter);
        auto scaleBiasTile = getSlice(op.getScaleBias(), biasOffset, {biasScaleTileSize}, rewriter);

        // Perform the sliced operation using the sliced weights and scale/bias
        SmallVector<int64_t> outTileShape(outShape);
        outTileShape[1] = channels;

        // Create an unencoded (therefore XRAM) output type, we will encode it later
        auto outTileType = RankedTensorType::get(outTileShape, outType.getElementType(), nullptr);
        SmallVector<OpFoldResult> outTileOffsets = createVector(0, outTileShape.size(), rewriter);
        outTileOffsets[1] = affineEval(arg * channels, iv, rewriter);
        auto outTileSizes = createVector(outTileShape, rewriter);
        const auto outTileStrides = createVector(1, outTileShape.size(), rewriter);

        auto initTile = createEmptyTensor(outTileType, loc, rewriter);
        auto outTileOp = rewriter.create<torq_hl::Conv2DOp>(
            loc, TypeRange{outTileType},
            ValueRange{initTile, weightsTile, scaleBiasTile, op.getInput()}, op->getAttrs()
        );

        // Merge the result in the destination tensorb
        rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
        rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, outTileOp.getResult(0), forallOp.getRegionIterArgs()[0], outTileOffsets,
            outTileSizes, outTileStrides
        );

        // Encode the new Conv2D operation, this will not change the input (as it is already
        // encoded) but will insert a convert op after the Conv2D operation to set the output
        // in the correct encoding
        // As init tensor for the decoded out use the final XRAM tile to avoid additional copies
        auto encodingRequirements = outTileOp.getKernelEncoding();
        auto destXramTile = getSlice(
            forallOp.getRegionIterArgs()[0], outTileOffsets, {outTileSizes}, outTileStrides,
            rewriter
        );
        if (failed(torq_hl::encodeKernelInputOutputs(
                outTileOp, encodingRequirements, rewriter, destXramTile
            ))) {
            llvm::report_fatal_error("Failed to encode sliced Conv2D operation");
        }

        // Replace the convert op that converted to xram the output of the convolution
        // with the output of the forall operation
        rewriter.replaceOp(*maybeConvertToXram, forallOp->getResult(0));

        // Erase the convolution operation that has no more users (we checked the convert
        // op was the only one at the beginning of the function)
        rewriter.eraseOp(op);

        return success();
    }
};

class DepthWise2DPattern : public OpRewritePattern<torq_hl::DepthwiseConv2DOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::DepthwiseConv2DOp op, PatternRewriter &rewriter) const override {
        auto loc = op.getLoc();
        auto ctx = op.getContext();

        // Do nothing if the containing operation is a forall (already sliced) or slicing disabled
        if (dyn_cast<scf::ForallOp>(op->getParentOp()) || !kSliceCount) {
            return failure();
        }

        // try to find the XRAM buffer where the output is stored, if we can't find it do not slice
        auto maybeConvertToXram = getConvertResultToXram(op);

        if (failed(maybeConvertToXram)) {
            return rewriter.notifyMatchFailure(
                op, "cannot find convert to XRAM operation for the output"
            );
        }

        // Compute number of channels for each slice
        // Currently we only support slicing if the channel count is a multiple of
        // (kSliceCount * channel grouping)
        auto outType = op.getOutput().getType();
        auto outShape = outType.getShape();
        int channels = outShape[1];
        const auto weightsType = op.getWeights().getType();
        const auto weightsShape = weightsType.getShape();
        const auto weightsGrouping = weightsShape[weightsShape.size() - 1];
        if (channels % (kSliceCount * weightsGrouping)) {
            return failure();
        }
        channels /= kSliceCount;

        // Be sure input data is in NOT in LRAM (to avoid a memref.copy of the subview at each iter)
        if (op.getStride()[0] == 2) {
            // If the stride is 2, the input is already in LRAM because of the segmentation layer
            // This generates a memref.copy of the subview at each iteration which we don't support
            // FIXME
            return rewriter.notifyMatchFailure(
                op, "Slicing of DepthWise2D with stride 2 not supported"
            );
        }

        // Split the operation on each slice using a parallel forall loop
        Value inputTile = op.getInput();
        scf::ForallOp forallOp =
            createForallOp(kSliceCount, maybeConvertToXram->getOutput().getType(), loc, rewriter);

        // Get the loop induction variable (iteration index) of the forall operation
        auto iv = forallOp.getLoopInductionVars().value()[0];

        // An affine expression returning its first argument
        AffineExpr arg = getAffineDimExpr(0, ctx);

        // Position the write cursor at the begining of the loop body
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(forallOp.getBody());

        // Compute the weights tile for each iteration
        SmallVector<int64_t> weightTileShape(weightsShape);
        weightTileShape[0] = channels / weightsGrouping;
        SmallVector<OpFoldResult> weightOffset = createVector(0, weightsShape.size(), rewriter);
        weightOffset[0] = affineEval(arg * (channels / weightsGrouping), iv, rewriter);
        auto weightsTile = getSlice(op.getWeights(), weightOffset, weightTileShape, rewriter);

        // Compute the bias tile for each iteration
        SmallVector<OpFoldResult> biasOffset = createVector({0}, rewriter);

        // We need to compute the tile of the bias/scale tensor based on the original
        // size because in some cases we have 2 values per channel and in some cases 1 value
        auto biasScaleTileSize = op.getScaleBias().getType().getShape()[0] / kSliceCount;
        biasOffset[0] = affineEval(arg * biasScaleTileSize, iv, rewriter);
        auto scaleBiasTile = getSlice(op.getScaleBias(), biasOffset, {biasScaleTileSize}, rewriter);

        // Compute the input tile for each iteration
        SmallVector<int64_t> inTileShape(op.getInput().getType().getShape());
        inTileShape[1] = channels;
        SmallVector<OpFoldResult> inTileOffsets = createVector({0, 0, 0, 0}, rewriter);
        inTileOffsets[1] = affineEval(arg * channels, iv, rewriter);
        inputTile = getSlice(inputTile, inTileOffsets, inTileShape, rewriter);

        // Perform the sliced operation using the sliced input, weights and scale/bias
        SmallVector<int64_t> outTileShape(outShape);
        outTileShape[1] = channels;

        // create an unencoded (therefore XRAM) output type, we will encode it later
        auto outTileType = RankedTensorType::get(outTileShape, outType.getElementType(), nullptr);
        SmallVector<OpFoldResult> outTileOffsets = createVector(0, outTileShape.size(), rewriter);
        outTileOffsets[1] = affineEval(arg * channels, iv, rewriter);
        auto outTileSizes = createVector(outTileShape, rewriter);
        const auto outTileStrides = createVector(1, outTileShape.size(), rewriter);

        auto initTile = createEmptyTensor(outTileType, loc, rewriter);
        auto outTileOp = rewriter.create<torq_hl::DepthwiseConv2DOp>(
            loc, TypeRange{outTileType},
            ValueRange{initTile, weightsTile, scaleBiasTile, inputTile}, op->getAttrs()
        );

        // Merge the result in the destination tensor
        rewriter.setInsertionPointToEnd(forallOp.getTerminator().getBody());
        rewriter.create<tensor::ParallelInsertSliceOp>(
            loc, outTileOp.getResult(0), forallOp.getRegionIterArgs()[0], outTileOffsets,
            outTileSizes, outTileStrides
        );

        // encode the new DepthwiseConv2D operation, this will not change the input (as it is
        // already encoded) but will insert a convert op after the DepthwiseConv2D operation to set
        // the output in the correct encoding

        auto encodingRequirements = outTileOp.getKernelEncoding();
        auto destXramTile = getSlice(
            forallOp.getRegionIterArgs()[0], outTileOffsets, {outTileSizes}, outTileStrides,
            rewriter
        );
        if (failed(torq_hl::encodeKernelInputOutputs(
                outTileOp, encodingRequirements, rewriter, destXramTile
            ))) {
            llvm::report_fatal_error("Failed to encode sliced DepthwiseConv2D operation");
        }

        // replace the result of the convert to XRAM operation with the result of the forall
        // operation that that returns an XRAM tensor
        rewriter.replaceOp(*maybeConvertToXram, forallOp->getResult(0));

        // Erase the depthwise convolution operation that has no more users (we checked the convert
        // op was the only one at the beginning of the function)
        rewriter.eraseOp(op);

        return success();
    }
};

class SlicingPass : public SlicingBase<SlicingPass> {
  public:
    using SlicingBase<SlicingPass>::SlicingBase;

    void runOnOperation() override;
};

void SlicingPass::runOnOperation() {
    auto funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<FullyConnectedPattern>(ctx);
    patterns.add<Conv2DPattern>(ctx);
    patterns.add<DepthWise2DPattern>(ctx);

    tensor::ControlConstantExtractSliceFusionFn controlFn = [](tensor::ExtractSliceOp op) {
        return true;
    };

    tensor::populateFoldConstantExtractSlicePatterns(patterns, controlFn);
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSlicingPass() {
    return std::make_unique<SlicingPass>();
}

} // namespace mlir::syna::torq
