// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Gives the conv2d, depthwise_conv2d and max_pool2d ops of each tile the weight layout and the
// input segmentation their kernels read, on the padding the conv/pool padding passes gave them
// (ConvPoolPaddingPasses.cpp).

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/EncodingRequirements.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/HwInfo.h"
#include "torq/Utils/LayoutTransformUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-conv-pool-kernel-layout"

namespace mlir::syna::torq {

namespace {
// Return the smallest multiple of d that is greater or equal to n
inline int64_t smallest_multiple(int64_t n, int64_t d) { return ((n + d - 1) / d) * d; }

// Append additional output channels if needed to make it a multiple of
// inner_on. This allows to accomodate the parallel computation of inner_on
// output channels.
template <typename T>
static void biasScale_inflate(Value &biases, int64_t inner_on, T biasValue, T scaleValue) {
    auto biasType = llvm::cast<RankedTensorType>(biases.getType());
    auto shape = biasType.getShape().vec();
    auto out_ch = shape[0];
    auto inflated_out_ch = 2 * std::max(smallest_multiple(out_ch, inner_on), inner_on);
    shape[0] = inflated_out_ch;

    auto padCh = (inflated_out_ch - out_ch) / 2;

    OpBuilder builder(biases.getContext());

    builder.setInsertionPointAfterValue(biases);

    auto padTensorType = RankedTensorType::get({padCh}, biasType.getElementType());

    DenseElementsAttr biasAttr =
        DenseElementsAttr::get(padTensorType, llvm::SmallVector<T>(padCh, biasValue));
    auto padBiasV = arith::ConstantOp::create(builder, biases.getLoc(), biasAttr);

    DenseElementsAttr scaleAttr =
        DenseElementsAttr::get(padTensorType, llvm::SmallVector<T>(padCh, scaleValue));
    auto padScaleV = arith::ConstantOp::create(builder, biases.getLoc(), scaleAttr);

    // Create inflated empty tensor after constants are placed
    auto inflatedBias =
        tensor::EmptyOp::create(builder, biases.getLoc(), shape, biasType.getElementType());

    // Insert original bias/scale data at [0..out_ch) with stride=1
    auto iOp = tensor::InsertSliceOp::create(
                   builder, biases.getLoc(), biases, inflatedBias,
                   /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
                   /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch)},
                   /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(1)}
    )
                   .getResult();

    // Insert pad bias values at [out_ch .. out_ch+padCh) with stride=1
    iOp = tensor::InsertSliceOp::create(
              builder, biases.getLoc(), padBiasV, iOp,
              /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch)},
              /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(padCh)},
              /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(1)}
    )
              .getResult();

    // Insert pad scale values at [out_ch+padCh .. inflated_out_ch) with stride=1
    iOp = tensor::InsertSliceOp::create(
              builder, biases.getLoc(), padScaleV, iOp,
              /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch + padCh)},
              /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(padCh)},
              /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(1)}
    )
              .getResult();

    IRRewriter irRewriter(builder.getContext());
    biases = createCompileTimeConstOp(iOp.getDefiningOp(), irRewriter).value_or(iOp);
}

// Convert weights from OI[KhKw] to OI[KhKw]O layout via reshape + transpose.
// Equivalent to linalg.pack with innerDimsPos=[0] and innerTiles=[inner_on], but expressed
// as expand_shape + linalg.transpose so NSS can execute the transpose in hardware.
// The number of output channels must be a multiple of inner_on.
// inner_on: number of channels to be moved in the inner dimension
static mlir::Value weights_OIHW_to_OIHWO(
    PatternRewriter &rewriter, mlir::Location loc, Value weights, int inner_on, Type ty
) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape()); // [O, I, H, W]
    auto elemTy = wtRankedTy.getElementType();

    const int64_t O = shape[0], I = shape[1], H = shape[2], W = shape[3];
    const int64_t paddedO = align_ceil(O, static_cast<int64_t>(inner_on));
    const int64_t O_outer = paddedO / inner_on;

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(weights);

    // Match linalg.pack semantics: when O is not a multiple of inner_on,
    // pad channels with zeros before tiling/transposing.
    Value packedInput = weights;
    if (paddedO != O) {
        SmallVector<int64_t> paddedShape = {paddedO, I, H, W};
        Value paddedEmpty = tensor::EmptyOp::create(rewriter, loc, paddedShape, elemTy);
        auto zeroAttr = rewriter.getZeroAttr(ty);
        Value zeroVal = arith::ConstantOp::create(rewriter, loc, ty, zeroAttr);
        paddedEmpty = linalg::FillOp::create(rewriter, loc, zeroVal, paddedEmpty).getResult(0);

        SmallVector<OpFoldResult> insOff(4, rewriter.getIndexAttr(0));
        SmallVector<OpFoldResult> insSz = getAsIndexOpFoldResult(rewriter.getContext(), shape);
        SmallVector<OpFoldResult> insStride(4, rewriter.getIndexAttr(1));
        packedInput = tensor::InsertSliceOp::create(
                          rewriter, loc, weights, paddedEmpty, insOff, insSz, insStride
        )
                          .getResult();
        packedInput =
            createCompileTimeConstOp(packedInput.getDefiningOp(), rewriter).value_or(packedInput);
    }

    // Step 1: ExpandShape [O, I, H, W] -> [O/N, N, I, H, W]
    // 32x5x3x3 --> 8x4x5x3x3  (for inner_on=4)
    SmallVector<ReassociationIndices> reassoc = {{0, 1}, {2}, {3}, {4}};
    SmallVector<int64_t> expandedShape = {O_outer, (int64_t)inner_on, I, H, W};
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, RankedTensorType::get(expandedShape, elemTy), packedInput, reassoc
    );
    expanded = createCompileTimeConstOp(expanded.getDefiningOp(), rewriter).value_or(expanded);

    // Step 2: Transpose [O/N, N, I, H, W] -> [O/N, I, H, W, N]  perm = [0, 2, 3, 4, 1]
    // 8x4x5x3x3 --> 8x5x3x3x4  (for inner_on=4)
    SmallVector<int64_t> transposedShape = {O_outer, I, H, W, (int64_t)inner_on};
    auto transposedType = RankedTensorType::get(transposedShape, elemTy);
    Value initTensor = tensor::EmptyOp::create(rewriter, loc, transposedShape, elemTy);
    SmallVector<int64_t> perm = {0, 2, 3, 4, 1};
    auto permAttr = rewriter.getDenseI64ArrayAttr(perm);
    auto transposeOp =
        torq_hl::TransposeOp::create(rewriter, loc, transposedType, initTensor, permAttr, expanded);

    return createCompileTimeConstOp(transposeOp, rewriter).value_or(transposeOp.getOutput());
}

// Get the vectorization mode from the Op
template <typename T> static torq_hl::VectorizationModeEnum getVectorizationMode(T convOp) {
    // Only depthwise has a layout-dependent mode; everything else runs 64x4.
    if constexpr (std::is_same_v<T, torq_hl::DepthwiseConv2DOp>) {
        if (convOp.getNhwcInput())
            return torq_hl::VectorizationModeEnum::_32x32;
    }
    return torq_hl::VectorizationModeEnum::_64x4;
}

static bool isStride2(llvm::ArrayRef<int64_t> strides) {
    return strides.size() == 2 && strides[0] == 2 && strides[1] == 2;
}

template <typename Stride2Op>
void insertSegmentationOp(Stride2Op op, PatternRewriter &rewriter, int h, int w) {
    ShapedType inputType = op.getInput().getType();
    assert(inputType.getRank() == 4 && "Expecting 4D input tensor");
    auto outputType = inputType;
    auto segmentationInput = op.getInput();

    rewriter.setInsertionPoint(op);
    if (!inputType.hasStaticShape())
        return;
    Value initTensor = tensor::EmptyOp::create(rewriter, op.getLoc(), outputType, ValueRange{});
    auto dummy_weights =
        createI8Const(rewriter, op, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
    auto dummy_scale_bias =
        createIConst(rewriter, op, std::vector<APInt>{APInt(32, 0), APInt(32, 1)});

    auto insertSliceOp = dyn_cast_or_null<tensor::InsertSliceOp>(op.getInput().getDefiningOp());
    if (insertSliceOp && h == 2 && w == 2 &&
        dyn_cast_or_null<tensor::EmptyOp>(insertSliceOp.getDest().getDefiningOp())) {
        // Op input comes from an insert slice into an empty tensor.
        // Fold the insert slice into the segmentation if possible by using its source as input
        // to the segmentation and adjusting the offsets accordingly.
        auto source = insertSliceOp.getSource();
        auto sourceType = llvm::dyn_cast<RankedTensorType>(source.getType());
        auto destType = llvm::dyn_cast<RankedTensorType>(insertSliceOp.getDest().getType());
        if (!sourceType || !destType)
            return;
        if (sourceType.getRank() == 4 && destType.getRank() == 4) {
            // Check if the insert slice is effectively copying the entire input tensor into a
            // larger tensor with padding. This is the common pattern we expect to see after the
            // valid-tosame padding pass.
            const auto &destShape = destType.getShape();
            const auto &sourceShape = sourceType.getShape();
            // Check if the insert slice is copying the entire source tensor into the top-left
            // corner of the destination tensor, and the source tensor has odd H or W dimensions
            auto offsets = SmallVector<int64_t>(insertSliceOp.getStaticOffsets());
            if (destShape[0] == sourceShape[0] && destShape[1] == sourceShape[1] &&
                destShape[2] >= sourceShape[2] && destShape[3] >= sourceShape[3] &&
                offsets == SmallVector<int64_t, 4>{0, 0, 0, 0}) {
                // Adjust the segmentation input to use the source of the insert slice
                segmentationInput = source;
            }
        }
    }

    auto segmentationOp = syna::torq_hl::SegmentationOp::create(
        rewriter, op.getLoc(), outputType, initTensor, h, w, dummy_weights.getResult(),
        dummy_scale_bias.getResult(), segmentationInput
    );

    rewriter.modifyOpInPlace(op, [&]() {
        op.setOperand(op.getInputMutable().getOperandNumber(), segmentationOp.getOutput());
    });
}

// Reshapes a tensor to the specified compile-time shape
static Value staticTensorReshape(
    Value tensor, mlir::ArrayRef<int64_t> shape, PatternRewriter &rewriter, const Location &loc
) {
    auto tensorType = dyn_cast<RankedTensorType>(tensor.getType());
    auto newType = RankedTensorType::get(shape, tensorType.getElementType());
    auto shapeType = RankedTensorType::get({(int)shape.size()}, rewriter.getIndexType());
    auto shapeAttr = DenseIntElementsAttr::get(shapeType, shape);
    Value shapeConst = arith::ConstantOp::create(rewriter, loc, shapeAttr);
    return tensor::ReshapeOp::create(rewriter, loc, newType, tensor, shapeConst).getResult();
}

// This function rearranges weights by placing even-indexed values first,
// followed by odd-indexed values, on the specified dimension.
static mlir::Value weights_swap_even_odd(
    PatternRewriter &rewriter, Location loc, Value weights, int swapDim, int stride
) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape());
    const int rank = shape.size();
    assert(swapDim < rank && "weight swap even odd: swapDim exceeds rank\n");
    SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(rewriter.getContext(), shape);
    const int64_t evenColSz = (shape[swapDim] + 1) / stride;
    const int64_t oddColSz = shape[swapDim] / stride;
    // Normally a kernel must be padded with 0s before calling this function.
    // However, in the specific case of stride==2 and swapping the last dimension,
    // we allow oddColSz != evenColSz since the HW can handle it.
    assert(stride == 2 && swapDim == 3 || evenColSz == oddColSz);

    SmallVector<int64_t> evenShape(shape);
    evenShape[swapDim] = evenColSz;
    SmallVector<int64_t> oddShape(shape);
    oddShape[swapDim] = oddColSz;
    auto evenTy = RankedTensorType::get(evenShape, wtRankedTy.getElementType());
    auto oddTy = RankedTensorType::get(oddShape, wtRankedTy.getElementType());
    // Constants
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value cStride = arith::ConstantIndexOp::create(rewriter, loc, stride);
    Value cevenColSz = arith::ConstantIndexOp::create(rewriter, loc, evenColSz);

    // Extract even channels: offsets [0,0,0,0], sizes [O,I,H,evenColSz], strides [1,1,1,2]
    SmallVector<OpFoldResult> evenOffsets(rank, rewriter.getIndexAttr(0));
    evenOffsets[swapDim] = c0;
    SmallVector<OpFoldResult> evenSizes(sizes);
    evenSizes[swapDim] = rewriter.getIndexAttr(evenColSz);
    SmallVector<OpFoldResult> evenStrides(rank, rewriter.getIndexAttr(1));
    evenStrides[swapDim] = cStride;
    Value evenExtract = tensor::ExtractSliceOp::create(
        rewriter, loc, evenTy, weights, evenOffsets, evenSizes, evenStrides
    );

    // Extract odd channels: offsets [0,0,0,1], sizes [O,I,H,oddColSz], strides [1,1,1,2]
    SmallVector<OpFoldResult> oddOffsets(rank, rewriter.getIndexAttr(0));
    oddOffsets[swapDim] = c1;
    SmallVector<OpFoldResult> oddSizes(sizes);
    oddSizes[swapDim] = rewriter.getIndexAttr(oddColSz);
    SmallVector<OpFoldResult> oddStrides(rank, rewriter.getIndexAttr(1));
    oddStrides[swapDim] = cStride;
    Value oddExtract = tensor::ExtractSliceOp::create(
        rewriter, loc, oddTy, weights, oddOffsets, oddSizes, oddStrides
    );

    // Stitch evens then odds into a new tensor
    Value empty = tensor::EmptyOp::create(rewriter, loc, shape, wtRankedTy.getElementType());

    // Insert evens at offset last-dim = 0
    SmallVector<OpFoldResult> ins0Off(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> ins0Sz(sizes);
    ins0Sz[swapDim] = rewriter.getIndexAttr(evenColSz);
    SmallVector<OpFoldResult> insStride(rank, rewriter.getIndexAttr(1));
    Value r1 = tensor::InsertSliceOp::create(
        rewriter, loc, evenExtract, empty, ins0Off, ins0Sz, insStride
    );

    // Insert odds at offset last-dim = evenColSz
    SmallVector<OpFoldResult> ins1Off(rank, rewriter.getIndexAttr(0));
    ins1Off[swapDim] = cevenColSz;
    SmallVector<OpFoldResult> ins1Sz(sizes);
    ins1Sz[swapDim] = rewriter.getIndexAttr(oddColSz);
    auto res =
        tensor::InsertSliceOp::create(rewriter, loc, oddExtract, r1, ins1Off, ins1Sz, insStride);

    return createCompileTimeConstOp(res, rewriter).value_or(res.getResult());
}

// Puts `top` zero rows above and `bottom` zero rows below the [O, I, KH, KW] `weights`.
static Value padZeroKernelRows(
    PatternRewriter &rewriter, Location loc, Value weights, int64_t top, int64_t bottom
) {
    if (top == 0 && bottom == 0) {
        return weights;
    }
    auto weightType = cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(weightType.getShape());
    SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(rewriter.getContext(), shape);
    shape[2] += top + bottom;
    Type elemType = weightType.getElementType();
    Value zero = arith::ConstantOp::create(rewriter, loc, elemType, rewriter.getZeroAttr(elemType));
    Value padded = tensor::EmptyOp::create(rewriter, loc, shape, elemType);
    padded = linalg::FillOp::create(rewriter, loc, zero, padded).getResult(0);
    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    offsets[2] = rewriter.getIndexAttr(top);
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    return tensor::InsertSliceOp::create(rewriter, loc, weights, padded, offsets, sizes, strides)
        .getResult();
}

// How a conv with stride [sh, 1] runs as a stride-1 conv on the sh row phases of its input.
struct RowPhaseSplit {
    // Zero kernel rows above the kernel. They make the top padding a multiple of sh.
    int64_t extraTop;
    // Rows of each kernel phase.
    int64_t kernelRows;
    // Top and bottom padding of the stride-1 conv.
    int64_t top, bottom;
};

// Computes the RowPhaseSplit of a conv with stride [sh, 1], kernel height kh and top padding top.
// The input has inH rows and the output has outH rows. Kernel row k reads input row
// sh * o - top + k. When top is not a multiple of sh, kernel phase p reads another input phase.
// Zero kernel rows above the kernel, with the same number of extra top padding rows, give the same
// result. Then kernel phase p reads input phase p. The stride-1 HW conv reads at most
// (kernelRows - 1) / 2 padding rows on top and the remaining border rows at the bottom (see
// convAdjustPadding). Zero kernel rows below the kernel make the top border larger. Each of them
// also adds one bottom padding row. The bottom border grows by one row or less. Thus these rows
// cannot make the bottom padding fit. Returns std::nullopt when the HW cannot run the split.
static std::optional<RowPhaseSplit>
getRowPhaseSplit(int64_t sh, int64_t kh, int64_t top, int64_t inH, int64_t outH) {
    if (inH % sh != 0) {
        return std::nullopt;
    }
    RowPhaseSplit split;
    split.extraTop = (sh - top % sh) % sh;
    split.top = (top + split.extraTop) / sh;
    split.kernelRows = std::max(llvm::divideCeilSigned(kh + split.extraTop, sh), 2 * split.top + 1);
    // The stride-1 conv gives outH = inH / sh + top + bottom - kernelRows + 1 rows
    split.bottom = outH - 1 + split.kernelRows - inH / sh - split.top;
    const int64_t maxBottom = split.kernelRows - 1 - (split.kernelRows - 1) / 2;
    if (split.bottom < 0 || split.bottom > maxBottom) {
        return std::nullopt;
    }
    return split;
}

// Pack depthwise 1D weights from [Ch, Kh, Kw] to [(Ch/32), Kh, Kw, 32]
// Used for depthwise 1D stride=1 special case
static mlir::Value weights_ChHW_to_ChHW32(
    PatternRewriter &rewriter, Location loc, Value weights, int inner_ch, Type elementType
) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape());

    const int ch = shape[0];
    const int kh = shape[2];
    const int kw = shape[3];

    assert(ch % inner_ch == 0 && "Channel dimension must be divisible by inner_ch");

    const int ch_outer = ch / inner_ch;

    // Create destination tensor shape: [(Ch/32), Kh, Kw, 32]
    SmallVector<int64_t> destShape = {ch_outer, kh, kw, inner_ch};

    // Use PackOp to transform [Ch, Kh, Kw] -> [(Ch/32), Kh, Kw, 32]
    // innerDimsPos=[0] means we're packing the first dimension (Ch)
    // innerTiles=[inner_ch] means we're splitting Ch into tiles of size inner_ch
    llvm::SmallVector<int64_t> innerDimsPos(1, 0);
    llvm::SmallVector<OpFoldResult> innerTiles(1, OpFoldResult(rewriter.getIndexAttr(inner_ch)));

    auto empty = linalg::PackOp::createDestinationTensor(
        rewriter, loc, weights, innerTiles, innerDimsPos, {}
    );

    auto res = linalg::PackOp::create(rewriter, loc, weights, empty, innerDimsPos, innerTiles);

    return createCompileTimeConstOp(res, rewriter).value_or(res.getResult());
}

template <typename ConvOpT> class ConvLikeKernelLayoutPattern : public OpRewritePattern<ConvOpT> {
  public:
    using OpRewritePattern<ConvOpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvOpT op, PatternRewriter &rewriter) const {

        if (op.getVectorizationMode() != torq_hl::VectorizationModeEnum::None) {
            // If the vectorization mode is already set, we don't need to select a kernel
            return failure();
        }

        const auto vectorizationMode = getVectorizationMode(op);

        // The enum has already the value for the number of parallel outputs (4, 8 or 16)
        const int parallel_outs = static_cast<int>(vectorizationMode);
        Value outputs = op.getOutput();
        Value weights = op.getWeights();

        Value biases = op.getScaleBias();

        auto weightType = cast<RankedTensorType>(weights.getType());
        auto weightElementType = weightType.getElementType();
        auto weightShape = weightType.getShape().vec();

        // Check if this is a depthwise 1D stride=1 special case
        bool isDw1dStride1 = false;
        if (auto dwOp = dyn_cast<torq_hl::DepthwiseConv2DOp>(op.getOperation())) {
            isDw1dStride1 = dwOp.getIsDw1dStride1();
        }

        if (vectorizationMode == torq_hl::VectorizationModeEnum::_32x32) {
            if (isDw1dStride1) {
                // Special case: depthwise 1D stride=1
                // Pack weights from [Ch, Kh, Kw] to [(Ch/inner), Kh, Kw, inner]
                // For BF16: inner=16 (WRAM holds 16*2=32 bytes)
                // For INT8: inner=32 (WRAM holds 32*1=32 bytes)
                int inner_tile = weightElementType.isBF16() ? 16 : 32;
                weights = weights_ChHW_to_ChHW32(
                    rewriter, op.getLoc(), weights, inner_tile, weightElementType
                );
            }
            else {
                weights = weights_pad_with_zero(rewriter, op.getLoc(), weights, 3, parallel_outs);
                weights =
                    createCompileTimeConstOp(weights.getDefiningOp(), rewriter).value_or(weights);
            }

            auto biasType = mlir::cast<RankedTensorType>(biases.getType());
            auto biasElemType = biasType.getElementType();
            if (biasElemType.isInteger()) {
                biasScale_inflate(biases, parallel_outs, APInt(32, 0), APInt(32, 0));
            }

            rewriter.modifyOpInPlace(op, [&]() {
                op.setVectorizationMode(vectorizationMode);
                op.setOperand(1, weights);
                op.setOperand(2, biases);
            });
            return success();
        }

        const int on = weightShape[0];
        const int in = weightShape[1];
        const int hn = weightShape[2];
        const int wn = weightShape[3];
        std::vector<int64_t> weight_shape{on, in, hn, wn};

        const auto &strides = op.getStride();
        std::optional<RowPhaseSplit> rowSplit;
        if (strides[0] > 1 && strides[1] == 1) {
            auto inputType = cast<RankedTensorType>(op.getInput().getType());
            auto initType = cast<RankedTensorType>(op.getInit().getType());
            if (inputType.hasStaticShape() && initType.hasStaticShape()) {
                rowSplit = getRowPhaseSplit(
                    strides[0], hn, op.getPad()[2], inputType.getShape()[2], initType.getShape()[2]
                );
            }
            if (!rowSplit) {
                return op.emitOpError("the stride-1 HW conv cannot run the row phases of this ")
                       << "stride [" << strides[0] << ", 1] conv";
            }
        }

        rewriter.modifyOpInPlace(op, [&]() { op.setVectorizationMode(vectorizationMode); });

        // Inflate and reorder weights
        if (isStride2(strides) && weight_shape[3] > 1) {
            // Kernel requires column 1 and column 2 in weights to be swapped
            weights = weights_swap_even_odd(rewriter, op.getLoc(), weights, 3, 2);
        }
        else if (rowSplit) {
            // Conv with stride [SH, 1], input [N, C, H, W] and kernel [KH, KW] is equivalent to
            // one with stride [1, 1], input [N, SH*C, H/SH, W], kernel [O,SH*I,div_ceil(KH,SH), KW]
            // where both the kernel and the input have been segmented (transposed) in SH groups
            // (in case of SH==2 this means even/odd rows). See getRowPhaseSplit for the zero
            // kernel rows and the padding.
            const int sh = strides[0];
            const int64_t bottomRows = sh * rowSplit->kernelRows - hn - rowSplit->extraTop;
            weights =
                padZeroKernelRows(rewriter, op.getLoc(), weights, rowSplit->extraTop, bottomRows);

            // Partition kernel rows in sh groups (even/odd rows for sh == 2)
            weights = weights_swap_even_odd(rewriter, op.getLoc(), weights, 2, sh);

            // Reshape weights by multiplying channels by sh and dividing rows by sh
            // Note: in the case of DW it is not normally allowed to change the number of input
            // channels, but our HW kernel supports accumulation over multiple input channels
            // to generate one output.
            auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
            // Collapse dimensions 1 and 2 of weights using a memref::CollapseShapeOp
            weightShape = wtRankedTy.getShape().vec();
            SmallVector<ReassociationIndices, 4> reassoc{{0}, {1, 2}, {3}};
            SmallVector<int64_t, 4> collapsedShape{
                weightShape[0], weightShape[1] * weightShape[2], weightShape[3]
            };
            weights = tensor::CollapseShapeOp::create(
                          rewriter, op.getLoc(),
                          RankedTensorType::get(collapsedShape, wtRankedTy.getElementType()),
                          weights, reassoc
            )
                          .getResult();

            // Expand dimensions 1 weights using a memref::ExpandShapeOp
            weightShape[1] = weightShape[1] * sh;
            weightShape[2] /= sh;
            weights = tensor::ExpandShapeOp::create(
                          rewriter, op.getLoc(),
                          RankedTensorType::get(weightShape, wtRankedTy.getElementType()), weights,
                          reassoc
            )
                          .getResult();
            weights = createCompileTimeConstOp(weights.getDefiningOp(), rewriter).value_or(weights);

            // Segment input rown in sh groups (even/odd rows for sh == 2) and reshape accordingly
            insertSegmentationOp(op, rewriter, sh, 0);
            Value input = op.getInput();
            auto inRankedTy = dyn_cast<RankedTensorType>(input.getType());
            auto inShape = inRankedTy.getShape().vec();
            inShape[1] = inShape[1] * sh;
            inShape[2] /= sh;
            input = staticTensorReshape(input, inShape, rewriter, op.getLoc());

            // Set strides to [1, 1]
            SmallVector<int64_t> newStrides{1, 1};
            rewriter.modifyOpInPlace(op, [&]() {
                op.setOperand(3, input);
                op.setStride(newStrides);
            });

            // Update top and bottom padding. Note: padding is represented as LRTB
            SmallVector<int64_t> padding(op.getPad());
            padding[2] = rowSplit->top;
            padding[3] = rowSplit->bottom;
            rewriter.modifyOpInPlace(op, [&]() { op.setPad(padding); });
        }
        if constexpr (std::is_same_v<ConvOpT, torq_hl::Conv2DOp>) {
            if (isWideKernel1DValidConv(op)) {
                // The kernel runs each group of 3 kernel columns (Alu::kerWidth()) as one HW
                // kernel, so a partial last group needs zero weights. The extra column reads one
                // input column past the end: allowed in LRAM, and junk * 0 is 0 for the integer
                // types that reach this path.
                constexpr int aluKernelWidth = 3;
                weights = weights_pad_with_zero(rewriter, op.getLoc(), weights, 3, aluKernelWidth);
                weights =
                    createCompileTimeConstOp(weights.getDefiningOp(), rewriter).value_or(weights);
            }
        }
        if (on >= parallel_outs) {
            weights = weights_OIHW_to_OIHWO(
                rewriter, op.getLoc(), weights, parallel_outs, weightElementType
            );
        }
        rewriter.modifyOpInPlace(op, [&]() { op.setOperand(1, weights); });

        if (isStride2(strides)) {
            insertSegmentationOp(op, rewriter, 2, 2);
        }

        return success();
    }
};

class MaxPool2dSegmentationPattern : public OpRewritePattern<torq_hl::MaxPool2dOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::MaxPool2dOp op, PatternRewriter &rewriter) const {

        // only stride-2 maxpool needs segmentation
        const auto &strides = op.getStride();
        if (!isStride2(strides)) {
            return failure();
        }

        // segmentation already inserted
        if (op->hasAttr("torq-segmented-input")) {
            // already segmented
            return failure();
        }

        insertSegmentationOp(op, rewriter, 2, 2);

        rewriter.modifyOpInPlace(op, [&]() {
            op->setAttr("torq-segmented-input", rewriter.getUnitAttr());
        });

        return success();
    }
};

class ConvPoolKernelLayoutPass : public impl::ConvPoolKernelLayoutBase<ConvPoolKernelLayoutPass> {
  public:
    void runOnOperation() override {
        MLIRContext *ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<ConvLikeKernelLayoutPattern<torq_hl::Conv2DOp>>(ctx);
        patterns.add<ConvLikeKernelLayoutPattern<torq_hl::DepthwiseConv2DOp>>(ctx);
        patterns.add<MaxPool2dSegmentationPattern>(ctx);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createConvPoolKernelLayoutPass() {
    return std::make_unique<ConvPoolKernelLayoutPass>();
}

} // namespace mlir::syna::torq
