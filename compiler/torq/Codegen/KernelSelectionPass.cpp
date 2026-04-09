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
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-kernel-selection"
#define ENABLE_HW_32x8_16x16 0

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
    auto inflatedBias =
        tensor::EmptyOp::create(builder, biases.getLoc(), shape, biasType.getElementType());

    // Inflate the biases tensor by padding biasValue and scaleValue at the end in interleaved
    // fashion
    builder.setInsertionPointAfterValue(biases);

    DenseElementsAttr biasAttr = DenseElementsAttr::get(
        RankedTensorType::get({padCh}, biasType.getElementType()),
        llvm::SmallVector<T>(padCh, biasValue)
    );
    auto padBiasV =
        arith::ConstantOp::create(builder, biases.getLoc(), biasType.getElementType(), biasAttr);

    DenseElementsAttr scaleAttr = DenseElementsAttr::get(
        RankedTensorType::get({padCh}, biasType.getElementType()),
        llvm::SmallVector<T>(padCh, scaleValue)
    );
    auto padScaleV =
        arith::ConstantOp::create(builder, biases.getLoc(), biasType.getElementType(), scaleAttr);
    auto iOp = tensor::InsertSliceOp::create(
                   builder, biases.getLoc(), biases, inflatedBias,
                   /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(0)},
                   /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch)},
                   /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(1)}
    )
                   .getResult();

    iOp = tensor::InsertSliceOp::create(
              builder, biases.getLoc(), padBiasV, iOp,
              /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch)},
              /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(padCh)},
              /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(2)}
    )
              .getResult();
    iOp = tensor::InsertSliceOp::create(
              builder, biases.getLoc(), padScaleV, iOp,
              /*offsets=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(out_ch + 1)},
              /*sizes=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(padCh)},
              /*strides=*/ArrayRef<OpFoldResult>{builder.getIndexAttr(2)}
    )
              .getResult();

    biases = iOp;

    setCompileTimeConstAttr(biases.getDefiningOp());
}

// Convert weights from OI[HW] to OI[HW]O layout
// The number of output channels must be a multiple of inner_on
// inner_on: number of channels to be moved in the inner dimension
static mlir::Value weights_OIHW_to_OIHWO(
    PatternRewriter &rewriter, mlir::Location loc, Value weights, int inner_on, Type ty
) {
    llvm::SmallVector<int64_t> innerDimsPos(1, 0);
    llvm::SmallVector<OpFoldResult> innerTiles(1, OpFoldResult(rewriter.getIndexAttr(inner_on)));

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(weights);
    auto empty = linalg::PackOp::createDestinationTensor(
        rewriter, loc, weights, innerTiles, innerDimsPos, {}
    );
    auto zeroAttr = rewriter.getZeroAttr(ty);
    auto zeroVal = arith::ConstantOp::create(rewriter, loc, ty, zeroAttr);

    // tensor.PackOp %weights { inner_tiles = [inner_on], inner_dims_pos = [0] }
    // 32x5x3x3 --> 8x5x3x3x4  (for inner_on=4)
    auto packedWeights =
        linalg::PackOp::create(rewriter, loc, weights, empty, innerDimsPos, innerTiles, zeroVal);

    setCompileTimeConstAttr(packedWeights);
    return packedWeights.getResult();
}

// FIXME: remove and use createI8Const from CoversionUtils.h
arith::ConstantOp createI8Const2(
    PatternRewriter &rewriter, Location loc, ArrayRef<char> values, llvm::ArrayRef<int64_t> shape
) {
    int64_t count = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
    assert(
        static_cast<int64_t>(values.size()) == count &&
        "Mismatch in number of elements vs tensor shape"
    );

    auto type = RankedTensorType::get(shape, rewriter.getI8Type());
    auto attr = DenseIntElementsAttr::get(type, values);
    return arith::ConstantOp::create(rewriter, loc, attr);
}

// Get the vectorization mode from the Op
template <typename T> static torq_hl::VectorizationModeEnum getVectorizationMode(T convOp) {

    if constexpr (std::is_same_v<T, torq_hl::DepthwiseConv2DOp>) {
        if (convOp.getNhwcInput())
            return torq_hl::VectorizationModeEnum::_32x32;
        else
            return torq_hl::VectorizationModeEnum::_64x4;
    }

    auto vectorizationMode = torq_hl::VectorizationModeEnum::_64x4;

#if ENABLE_HW_32x8_16x16
    auto outShape = llvm::cast<RankedTensorType>(convOp.getOutput().getType()).getShape().vec();
    if (outShape.size() <= 2) {
        return vectorizationMode;
    }
    // Find the best Mode for Hardware based on the macc count
    auto outFrameSize = outShape[2] * outShape[3]; // Height * Widith
    auto out_channel = outShape[1];
    // Compute MACC counts
    int64_t maccCount_64x4 = div_ceil(outFrameSize, 64) * div_ceil(out_channel, 4);
    int64_t maccCount_32x8 = div_ceil(outFrameSize, 32) * div_ceil(out_channel, 8);
    int64_t maccCount_16x16 = div_ceil(outFrameSize, 16) * div_ceil(out_channel, 16);
    // Default selection is 64x4
    int64_t bestValue = maccCount_64x4;
    // Check 32x8
    if (maccCount_32x8 < bestValue && out_channel >= 8) {
        bestValue = maccCount_32x8;
        vectorizationMode = torq_hl::VectorizationModeEnum::_32x8;
    }
    // Check 16x16
    if (maccCount_16x16 < bestValue && out_channel >= 16)
        vectorizationMode = torq_hl::VectorizationModeEnum::_16x16;
#endif
    return vectorizationMode;
}

static bool isStride2(llvm::ArrayRef<int64_t> strides) {
    return strides.size() == 2 && strides[0] == 2 && strides[1] == 2;
}

template <typename Stride2Op>
void insertSegmentationOp(Stride2Op op, PatternRewriter &rewriter, int h, int w) {
    ShapedType inputType = op.getInput().getType();
    assert(inputType.getRank() == 4 && "Expecting 4D input tensor");
    auto outputType = inputType;

    rewriter.setInsertionPoint(op);
    Value initTensor = tensor::EmptyOp::create(rewriter, op.getLoc(), outputType, ValueRange{});
    auto dummy_weights =
        createI8Const(rewriter, op, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
    auto dummy_scale_bias =
        createIConst(rewriter, op, std::vector<APInt>{APInt(32, 0), APInt(32, 1)});

    auto segmentationOp = syna::torq_hl::SegmentationOp::create(
        rewriter, op.getLoc(), outputType, initTensor, h, w, dummy_weights.getResult(),
        dummy_scale_bias.getResult(), op.getInput()
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

    setCompileTimeConstAttr(res);
    return res.getResult();
}

// Insert a dimension of size 1 at the specified position
static mlir::Value
weights_insert_dimension(PatternRewriter &rewriter, Location loc, Value weights, int insertDim) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape());
    const int rank = shape.size();
    assert(insertDim <= rank && "weight insert dimension: insertDim exceeds rank\n");
    SmallVector<ReassociationIndices> reassoc;
    for (int i = 0; i < rank + 1; ++i) {
        reassoc.push_back({i <= insertDim ? i : i - 1});
    }
    SmallVector<int64_t> newShape = shape;
    newShape.insert(newShape.begin() + insertDim, 1);
    auto newTy = RankedTensorType::get(newShape, wtRankedTy.getElementType());
    auto res = tensor::ExpandShapeOp::create(rewriter, loc, newTy, weights, reassoc);
    setCompileTimeConstAttr(res);
    return res.getResult();
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

    setCompileTimeConstAttr(res);
    return res.getResult();
}

// Pad the specified dimension to padDimAlignment with 0s at the end
static mlir::Value weights_pad_with_zero(
    PatternRewriter &rewriter, Location loc, Value weights, int padDim, int padDimAlignment
) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape());
    const int rank = shape.size();
    assert(padDim < rank && "weight pad with zero: padDim exceeds rank\n");
    int paddedDimSize = align_ceil(shape[padDim], padDimAlignment);
    if (paddedDimSize == shape[padDim]) {
        return weights;
    }

    SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(rewriter.getContext(), shape);
    auto paddedShape = shape;
    paddedShape[padDim] = paddedDimSize;
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    // Create empty padded tensor
    Value paddedEmpty =
        tensor::EmptyOp::create(rewriter, loc, paddedShape, wtRankedTy.getElementType());
    // Fill with zeros
    auto zeroAttr = rewriter.getZeroAttr(wtRankedTy.getElementType());
    Value zeroVal = arith::ConstantOp::create(rewriter, loc, wtRankedTy.getElementType(), zeroAttr);
    paddedEmpty = linalg::FillOp::create(rewriter, loc, zeroVal, paddedEmpty).getResult(0);
    // Insert original weights into padded tensor
    SmallVector<OpFoldResult> insOff(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> insSz = sizes;
    auto res =
        tensor::InsertSliceOp::create(rewriter, loc, weights, paddedEmpty, insOff, insSz, strides);
    setCompileTimeConstAttr(res);
    return res.getResult();
}

template <typename ConvOpT> class ConvLikeKernelSelection : public OpRewritePattern<ConvOpT> {
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

        rewriter.modifyOpInPlace(op, [&]() { op.setVectorizationMode(vectorizationMode); });

        // Inflate and reorder weights
        const auto &strides = op.getStride();
        if (isStride2(strides) && weight_shape[3] > 1) {
            // Kernel requires column 1 and column 2 in weights to be swapped
            weights = weights_swap_even_odd(rewriter, op.getLoc(), weights, 3, 2);
        }
        else if (strides[0] > 1 && strides[1] == 1) {
            // Conv with stride [SH, 1], input [N, C, H, W] and kernel [KH, KW] is equivalent to
            // one with stride [1, 1], input [N, SH*C, H/SH, W], kernel [O,SH*I,div_ceil(KH,SH), KW]
            // where both the kernel and the input have been segmented (transposed) in SH groups
            // (in case of SH==2 this means even/odd rows).

            // Pad weights on H dimension to be multiple of sh
            const int sh = strides[0];
            weights = weights_pad_with_zero(rewriter, op.getLoc(), weights, 2, sh);

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
            setCompileTimeConstAttr(weights.getDefiningOp());

            // Expand dimensions 1 weights using a memref::ExpandShapeOp
            weightShape[1] = weightShape[1] * sh;
            weightShape[2] /= sh;
            weights = tensor::ExpandShapeOp::create(
                          rewriter, op.getLoc(),
                          RankedTensorType::get(weightShape, wtRankedTy.getElementType()), weights,
                          reassoc
            )
                          .getResult();
            setCompileTimeConstAttr(weights.getDefiningOp());

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
            padding[2] /= sh;
            padding[3] /= sh;
            rewriter.modifyOpInPlace(op, [&]() { op.setPad(padding); });
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

class FullyConnectedKernelSelection : public OpRewritePattern<torq_hl::FullyConnectedOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::FullyConnectedOp op, PatternRewriter &rewriter) const {

        // kernel already selected
        if (op.getVectorizationMode() != torq_hl::VectorizationModeEnum::None) {
            return failure();
        }

        Value weights = op.getWeights();
        auto weightTy = mlir::cast<RankedTensorType>(weights.getType());
        auto vectorizationMode = getVectorizationMode(op);

        auto outputElementType = op.getInit().getType().getElementType();
        // FIXME: the test below should be improved to avoid hardcoding these values
        int parallel_outs = outputElementType.getIntOrFloatBitWidth() <= 8 ? 64 : 32;

        weights = weights_OIHW_to_OIHWO(
            rewriter, op.getLoc(), weights, parallel_outs, weightTy.getElementType()
        );

        // FIXME: use createI8Const from CoversionUtils.h
        rewriter.modifyOpInPlace(op, [&]() {
            op.setVectorizationMode(vectorizationMode);
            op.setOperand(1, weights);
        });
        return success();
    }
};

class MaxPool2dKernelSelectionOp : public OpRewritePattern<torq_hl::MaxPool2dOp> {
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

class KernelSelectionPass : public impl::KernelSelectionBase<KernelSelectionPass> {
  public:
    using KernelSelectionBase<KernelSelectionPass>::KernelSelectionBase;
    void runOnOperation() override;
};

void KernelSelectionPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<ConvLikeKernelSelection<torq_hl::Conv2DOp>>(ctx);
    patterns.add<ConvLikeKernelSelection<torq_hl::DepthwiseConv2DOp>>(ctx);
    patterns.add<FullyConnectedKernelSelection>(ctx);
    patterns.add<MaxPool2dKernelSelectionOp>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createKernelSelectionPass() {
    return std::make_unique<KernelSelectionPass>();
}

} // namespace mlir::syna::torq
