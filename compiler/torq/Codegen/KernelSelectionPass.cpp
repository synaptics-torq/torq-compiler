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

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-kernel-selection"

namespace mlir::syna::torq {

namespace {
// Return the smallest multiple of d that is greater or equal to n
inline int64_t smallest_multiple(int64_t n, int64_t d) { return ((n + d - 1) / d) * d; }

// Append additional output channels if needed to make it a multiple of
// inner_on. This allows to accomodate the parallel computation of inner_on
// output channels.
template <typename T>
static void biasScale_inflate(
    std::vector<T> &biases, std::vector<int64_t> &shape, int64_t inner_on, T biasValue, T scaleValue
) {
    auto out_ch = shape[0];
    auto inflated_out_ch = 2 * std::max(smallest_multiple(out_ch, inner_on), inner_on);
    shape[0] = inflated_out_ch;
    biases.reserve(inflated_out_ch);

    auto pad_ch = (inflated_out_ch - out_ch) / 2;
    for (int i = 0; i < pad_ch; ++i) {
        biases.push_back(biasValue);
        biases.push_back(scaleValue);
    }
}

// Convert weights from OI[HW] to OI[HW]O layout
// The number of output channels must be a multiple of inner_on
// inner_on: number of channels to be moved in the inner dimension
static mlir::Value weights_OIHW_to_OIHWO(
    PatternRewriter &rewriter, mlir::Location loc, Value weights, int inner_on, Type ty
) {
    llvm::SmallVector<int64_t> innerDimsPos(1, 0);
    llvm::SmallVector<OpFoldResult> innerTiles(1, OpFoldResult(rewriter.getIndexAttr(inner_on)));
    auto empty = tensor::PackOp::createDestinationTensor(
        rewriter, loc, weights, innerTiles, innerDimsPos, {}
    );
    auto zeroAttr = rewriter.getZeroAttr(ty);
    auto zeroVal = rewriter.create<arith::ConstantOp>(loc, ty, zeroAttr);

    // tensor.PackOp %weights { inner_tiles = [inner_on], inner_dims_pos = [0] }
    // 32x5x3x3 --> 8x5x3x3x4  (for inner_on=4)
    auto packedWeights =
        rewriter.create<tensor::PackOp>(loc, weights, empty, innerDimsPos, innerTiles, zeroVal);

    setCompileTimeConstAttr(packedWeights);
    setTargetExecutorAttr(packedWeights, torq_hl::Executor::NSS);
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
    return rewriter.create<arith::ConstantOp>(loc, attr);
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
    return vectorizationMode;
}

template <class T> static bool isStride2(T convLikeOp) {
    auto strides = convLikeOp.getStride();
    return strides.size() == 2 && strides[0] == 2 && strides[1] == 2;
}

template <typename Stride2Op> void insertSegmentationOp(Stride2Op op, PatternRewriter &rewriter) {
    ShapedType inputType = op.getInput().getType();
    assert(inputType.getRank() == 4 && "Expecting 4D input tensor");
    auto outputType = inputType;

    rewriter.setInsertionPoint(op);
    Value initTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), outputType, ValueRange{});
    auto dummy_weights =
        createI8Const(rewriter, op, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
    auto dummy_scale_bias =
        createIConst(rewriter, op, std::vector<APInt>{APInt(32, 0), APInt(32, 1)});

    auto segmentationOp = rewriter.create<syna::torq_hl::SegmentationOp>(
        op.getLoc(), outputType, initTensor, 0, 0, 0, 0, dummy_weights.getResult(),
        dummy_scale_bias.getResult(), op.getInput()
    );

    rewriter.modifyOpInPlace(op, [&]() {
        op.setOperand(op.getInputMutable().getOperandNumber(), segmentationOp.getOutput());
    });
}

// This function rearranges weights by placing even-indexed values first,
// followed by odd-indexed values, on the specified dimension.
static mlir::Value
weights_swap_even_odd(PatternRewriter &rewriter, Location loc, Value weights, int swapDim) {
    auto wtRankedTy = dyn_cast<RankedTensorType>(weights.getType());
    SmallVector<int64_t> shape(wtRankedTy.getShape());
    const int rank = shape.size();
    assert(swapDim < rank && "weight swap even odd: swapDim exceeds rank\n");
    SmallVector<OpFoldResult> sizes = getAsIndexOpFoldResult(rewriter.getContext(), shape);
    const int64_t evenColSz = (shape[swapDim] + 1) / 2;
    const int64_t oddColSz = shape[swapDim] / 2;

    SmallVector<int64_t> evenShape(shape);
    evenShape[swapDim] = evenColSz;
    SmallVector<int64_t> oddShape(shape);
    oddShape[swapDim] = oddColSz;
    auto evenTy = RankedTensorType::get(evenShape, wtRankedTy.getElementType());
    auto oddTy = RankedTensorType::get(oddShape, wtRankedTy.getElementType());
    // Constants
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value c2 = rewriter.create<arith::ConstantIndexOp>(loc, 2);
    Value cevenColSz = rewriter.create<arith::ConstantIndexOp>(loc, evenColSz);

    // Extract even channels: offsets [0,0,0,0], sizes [O,I,H,evenColSz], strides [1,1,1,2]
    SmallVector<OpFoldResult> evenOffsets(rank, rewriter.getIndexAttr(0));
    evenOffsets[swapDim] = c0;
    SmallVector<OpFoldResult> evenSizes(sizes);
    evenSizes[swapDim] = rewriter.getIndexAttr(evenColSz);
    SmallVector<OpFoldResult> evenStrides(rank, rewriter.getIndexAttr(1));
    evenStrides[swapDim] = c2;
    Value evenExtract = rewriter.create<tensor::ExtractSliceOp>(
        loc, evenTy, weights, evenOffsets, evenSizes, evenStrides
    );

    // Extract odd channels: offsets [0,0,0,1], sizes [O,I,H,oddColSz], strides [1,1,1,2]
    SmallVector<OpFoldResult> oddOffsets(rank, rewriter.getIndexAttr(0));
    oddOffsets[swapDim] = c1;
    SmallVector<OpFoldResult> oddSizes(sizes);
    oddSizes[swapDim] = rewriter.getIndexAttr(oddColSz);
    SmallVector<OpFoldResult> oddStrides(rank, rewriter.getIndexAttr(1));
    oddStrides[swapDim] = c2;
    Value oddExtract = rewriter.create<tensor::ExtractSliceOp>(
        loc, oddTy, weights, oddOffsets, oddSizes, oddStrides
    );

    // Stitch evens then odds into a new tensor
    Value empty = rewriter.create<tensor::EmptyOp>(loc, shape, wtRankedTy.getElementType());

    // Insert evens at offset last-dim = 0
    SmallVector<OpFoldResult> ins0Off(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> ins0Sz(sizes);
    ins0Sz[swapDim] = rewriter.getIndexAttr(evenColSz);
    SmallVector<OpFoldResult> insStride(rank, rewriter.getIndexAttr(1));
    Value r1 =
        rewriter.create<tensor::InsertSliceOp>(loc, evenExtract, empty, ins0Off, ins0Sz, insStride);

    // Insert odds at offset last-dim = evenColSz
    SmallVector<OpFoldResult> ins1Off(rank, rewriter.getIndexAttr(0));
    ins1Off[swapDim] = cevenColSz;
    SmallVector<OpFoldResult> ins1Sz(sizes);
    ins1Sz[swapDim] = rewriter.getIndexAttr(oddColSz);
    auto res =
        rewriter.create<tensor::InsertSliceOp>(loc, oddExtract, r1, ins1Off, ins1Sz, insStride);

    setCompileTimeConstAttr(res);
    setTargetExecutorAttr(res, torq_hl::Executor::NSS);
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
    auto res = rewriter.create<tensor::ExpandShapeOp>(loc, newTy, weights, reassoc);
    setCompileTimeConstAttr(res);
    setTargetExecutorAttr(res, torq_hl::Executor::NSS);
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
        rewriter.create<tensor::EmptyOp>(loc, paddedShape, wtRankedTy.getElementType());
    // Insert original weights into padded tensor
    SmallVector<OpFoldResult> insOff(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> insSz = sizes;
    auto res =
        rewriter.create<tensor::InsertSliceOp>(loc, weights, paddedEmpty, insOff, insSz, strides);
    setCompileTimeConstAttr(res);
    setTargetExecutorAttr(res, torq_hl::Executor::NSS);
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
        arith::ConstantOp constOp = weights.getDefiningOp<arith::ConstantOp>();

        Value biases = op.getScaleBias();
        arith::ConstantOp biasConstOp = biases.getDefiningOp<arith::ConstantOp>();

        if (!constOp || !biasConstOp) {
            op->emitError() << "weights or biases don't come from a arith::ConstantOp";

            // here we assert because we need to always select a kernel
            llvm::report_fatal_error("cannot select kernel");
        }

        auto weightType = cast<RankedTensorType>(weights.getType());
        auto weightElementType = weightType.getElementType();
        auto weightShape = weightType.getShape().vec();

        auto convBias = biasConstOp.getValue();
        auto biasValues = mlir::cast<DenseIntOrFPElementsAttr>(convBias);
        auto biasType = biasValues.getType();
        auto biasShape = biasType.getShape().vec();

        if (vectorizationMode == torq_hl::VectorizationModeEnum::_32x32) {
            if (weightShape.size() == 3) {
                // Linalg depthwise conv2d has 3D weights: HW0, make it IHW0
                weights = weights_insert_dimension(rewriter, op.getLoc(), weights, 0);
            }
            weights = weights_pad_with_zero(rewriter, op.getLoc(), weights, 3, parallel_outs);

            std::vector<llvm::APInt> biasData(
                biasValues.value_begin<llvm::APInt>(), biasValues.value_end<llvm::APInt>()
            );
            biasScale_inflate(biasData, biasShape, parallel_outs, APInt(32, 0), APInt(32, 0));
            auto bias_tensor = createIConst(rewriter, op, biasData, biasShape);

            rewriter.modifyOpInPlace(op, [&]() {
                op.setVectorizationMode(vectorizationMode);
                op.setOperand(1, weights);
                op.setOperand(2, bias_tensor);
            });

            return success();
        }

        if (weightShape.size() == 3) {
            // Linalg depthwise conv2d has 3D weights: OHW, make it OIHW
            // Ex: 576x3x3 --> 576x1x3x3
            weights = weights_insert_dimension(rewriter, op.getLoc(), weights, 1);
            rewriter.modifyOpInPlace(op, [&]() { op.setOperand(1, weights); });
            weightShape = {weightShape[0], 1, weightShape[1], weightShape[2]};
        }

        const int on = weightShape[0];
        const int in = weightShape[1];
        const int hn = weightShape[2];
        const int wn = weightShape[3];
        std::vector<int64_t> weight_shape{on, in, hn, wn};

        rewriter.modifyOpInPlace(op, [&]() { op.setVectorizationMode(vectorizationMode); });
        if (weightElementType.isInteger()) {
            auto bitWidth = weightElementType.getIntOrFloatBitWidth();
            switch (bitWidth) {
            case 8: {
                // Inflate and reorder weights
                if (isStride2(op) && weight_shape[3] > 1) {
                    // Kernel requires column 1 and column 2 in weights to be swapped
                    weights = weights_swap_even_odd(rewriter, op.getLoc(), weights, 3);
                }
                if (on >= parallel_outs) {
                    weights = weights_OIHW_to_OIHWO(
                        rewriter, op.getLoc(), weights, parallel_outs, weightElementType
                    );
                }
                rewriter.modifyOpInPlace(op, [&]() { op.setOperand(1, weights); });
                break;
            }
            default:
                op->emitError() << "Unsupported weight bitwidth: " << bitWidth;
                llvm::report_fatal_error("cannot select kernel");
            }
        }
        else if (weightElementType.isBF16()) {
            // Inflate and reorder weights
            if (isStride2(op) && weight_shape[3] > 1) {
                // Kernel requires column 1 and column 2 in weights to be swapped
                weights = weights_swap_even_odd(rewriter, op.getLoc(), weights, 3);
            }
            if (on >= parallel_outs) {
                weights = weights_OIHW_to_OIHWO(
                    rewriter, op.getLoc(), weights, parallel_outs, weightElementType
                );
            }
            rewriter.modifyOpInPlace(op, [&]() { op.setOperand(1, weights); });
        }
        else {
            op->emitError() << "Unsupported weight element type: " << weightElementType;
            llvm::report_fatal_error("cannot select kernel");
        }

        if (isStride2(op)) {
            insertSegmentationOp(op, rewriter);
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
        arith::ConstantOp constOp = weights.getDefiningOp<arith::ConstantOp>();
        if (!constOp) {
            op->emitError() << "weights don't come from a arith::ConstantOp";
            llvm::report_fatal_error("cannot select kernel", true);
        }

        auto convWeights = constOp.getValue();
        auto weightValues = mlir::cast<DenseIntOrFPElementsAttr>(convWeights);
        auto weightData = weightValues.getRawData().vec();
        auto weightShape = weightValues.getType().getShape().vec();

        auto vectorizationMode = getVectorizationMode(op);

        int parallel_outs = 64;

        weights = weights_OIHW_to_OIHWO(
            rewriter, op.getLoc(), weights, parallel_outs, weightValues.getType().getElementType()
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
        if (!isStride2(op)) {
            return failure();
        }

        // segmentation already inserted
        if (op->hasAttr("torq-segmented-input")) {
            // already segmented
            return failure();
        }

        insertSegmentationOp(op, rewriter);

        rewriter.modifyOpInPlace(op, [&]() {
            op->setAttr("torq-segmented-input", rewriter.getUnitAttr());
        });

        return success();
    }
};

class KernelSelectionPass : public KernelSelectionBase<KernelSelectionPass> {
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

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createKernelSelectionPass() {
    return std::make_unique<KernelSelectionPass>();
}

} // namespace mlir::syna::torq
