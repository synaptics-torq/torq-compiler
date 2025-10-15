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
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
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
// output channels. Weight tensor expected to be in OIHW or OI format
template <typename T>
static void
weights_inflate(std::vector<T> &weight, std::vector<int64_t> &shape, int inner_on, T value) {
    // Assume 1 single slice. If more slices are used, the operation and the
    // corresponding weights should be splitted before reaching this point
    auto out_ch = shape[0];
    auto inflated_out_ch = smallest_multiple(out_ch, inner_on);
    if (auto pad_ch = inflated_out_ch - out_ch) {
        shape[0] = inflated_out_ch;
        auto weights_per_ch = shape.size() == 4 ? shape[1] * shape[2] * shape[3] : shape[1];
        weight.resize(weight.size() + pad_ch * weights_per_ch, value);
    }
}

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

template <typename T>
static void weights_inflate_last_channel(
    std::vector<T> &weight, std::vector<int64_t> &shape, int inner_on, T value
) {

    auto channel_dim_index = shape.size() - 1;
    auto original_channels = shape[channel_dim_index];
    auto padded_channels = smallest_multiple(original_channels, inner_on);

    shape[channel_dim_index] = padded_channels;

    int64_t outer_size = 1;
    for (size_t i = 0; i < channel_dim_index; ++i)
        outer_size *= shape[i];

    std::vector<T> inflated_weight;
    inflated_weight.reserve(outer_size * padded_channels);

    for (int64_t i = 0; i < outer_size; ++i) {
        for (int64_t c = 0; c < original_channels; ++c)
            inflated_weight.push_back(weight[i * original_channels + c]);
        for (int64_t c = 0; c < (padded_channels - original_channels); ++c)
            inflated_weight.push_back(value);
    }
    weight = std::move(inflated_weight);
}

// Convert weights from OI[HW] to OI[HW]O layout
// The number of output channels must be a multiple of inner_on
// inner_on: number of channels to be moved in the inner dimension
template <typename T>
static void
weights_OIHW_to_OIHWO(std::vector<T> &weight, std::vector<int64_t> &shape, int inner_on) {
    int on, in, hn, wn;
    if (shape.size() == 4) {
        on = shape[0];
        in = shape[1];
        hn = shape[2];
        wn = shape[3];
    }
    else if (shape.size() == 2) {
        on = shape[0];
        in = shape[1];
        hn = 1;
        wn = 1;
    }
    else {
        llvm::report_fatal_error("weights_OIHW_to_OIHWO: shape size must be 2 or 4\n", true);
    }

    if (on / inner_on * inner_on != on) {
        llvm::report_fatal_error("on is not a multiple of inner_on\n", true);
    }

    // Adjust shape to reflect the new layout
    on /= inner_on;
    shape[0] = on;
    shape.push_back(inner_on);

    // Convert weights from OIHW to OIHWO
    std::vector<T> weight_oihwo;
    weight_oihwo.reserve(weight.size());
    for (int o = 0; o < on * inner_on; o += inner_on) {
        for (int i = 0; i < in; i++) {
            for (int h = 0; h < hn; h++) {
                for (int w = 0; w < wn; w++) {
                    for (int io = 0; io < inner_on; io++) {
                        int index = (o + io) * in * hn * wn + i * hn * wn + h * wn + w;
                        weight_oihwo.push_back(weight[index]);
                    }
                }
            }
        }
    }
    weight = std::move(weight_oihwo);
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
    }

    auto vectorizationMode = torq_hl::VectorizationModeEnum::_64x4;

    auto outShape = llvm::cast<RankedTensorType>(convOp.getOutput().getType()).getShape().vec();
    if (outShape.size() <= 2) {
        return vectorizationMode;
    }
    auto outFrameSize = outShape[2] * outShape[3];

    if (outFrameSize < outShape[1]) {
        if (outFrameSize < 64)
            vectorizationMode = torq_hl::VectorizationModeEnum::_32x8;
        if (outFrameSize < 32)
            vectorizationMode = torq_hl::VectorizationModeEnum::_16x16;
    }
    return vectorizationMode;
}

template <class T> static bool isStride2(T convLikeOp) {
    auto strides = convLikeOp.getStride();
    return strides.size() == 2 && strides[0] == 2 && strides[1] == 2;
}

template <typename Stride2Op> void insertSegmentationOp(Stride2Op op, PatternRewriter &rewriter) {

    ShapedType inputType = op.getInput().getType();

    assert(inputType.getRank() == 4 && "Expecting 4D input tensor");

    auto inputShape = inputType.getShape();

    // FIXME: do we need to do div_ceil or floor here?
    auto outputType = RankedTensorType::get(
        {inputShape[0], inputShape[1], 4, div_ceil(inputShape[3], 2), div_ceil(inputShape[2], 2)},
        inputType.getElementType()
    );

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

// This function reorders the weights to match the access pattern used in stride-2 convolution cases
// for filters with size greater than 1. It rearranges weights by placing even-indexed values first,
// followed by odd-indexed values, for each output channel slice.
template <typename T>
void weights_swap_even_odd(std::vector<T> &weight, const std::vector<int64_t> &shape) {
    assert(shape.size() == 4 && "weight swap even odd: shape size is not 4\n");

    auto wc = shape[3];

    for (size_t i = 0; i < weight.size(); i += wc) {
        std::vector<T> reordered;
        int left = 0;
        while (left < wc) {
            reordered.push_back(weight[i + left]);
            left += 2;
        }
        int right = 1;
        while (right < wc) {
            reordered.push_back(weight[i + right]);
            right += 2;
        }
        for (int k = 0; k < wc; ++k) {
            weight[i + k] = reordered[k];
        }
    }
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

        auto convWeights = constOp.getValue();
        auto weightValues = mlir::cast<DenseIntOrFPElementsAttr>(convWeights);
        auto weightType = weightValues.getType();
        auto weightElementType = weightType.getElementType();
        auto weightShape = weightType.getShape().vec();

        auto convBias = biasConstOp.getValue();
        auto biasValues = mlir::cast<DenseIntOrFPElementsAttr>(convBias);
        auto biasType = biasValues.getType();
        auto biasShape = biasType.getShape().vec();

        if (vectorizationMode == torq_hl::VectorizationModeEnum::_32x32) {
            if (weightShape.size() == 3) {
                // Linalg depthwise conv2d has 3D weights: HW0, make it IHW0
                weightShape.insert(weightShape.begin(), 1);
            }
            auto weightData = weightValues.getRawData().vec();
            weights_inflate_last_channel(
                weightData, weightShape, parallel_outs, static_cast<char>(0)
            );

            weights_inflate_last_channel(
                weightData, weightShape, parallel_outs, static_cast<char>(0)
            );
            auto weight_tensor = createI8Const2(rewriter, op.getLoc(), weightData, weightShape);

            std::vector<llvm::APInt> biasData(
                biasValues.value_begin<llvm::APInt>(), biasValues.value_end<llvm::APInt>()
            );
            biasScale_inflate(biasData, biasShape, parallel_outs, APInt(32, 0), APInt(32, 0));
            auto bias_tensor = createIConst(rewriter, op, biasData, biasShape);

            rewriter.modifyOpInPlace(op, [&]() {
                op.setVectorizationMode(vectorizationMode);
                op.setOperand(1, weight_tensor);
                op.setOperand(2, bias_tensor);
            });

            return success();
        }

        if (weightShape.size() == 3) {
            // Linalg depthwise conv2d has 3D weights: OHW, make it OIHW
            weightShape.insert(weightShape.begin() + 1, 1);
        }

        const int on = weightShape[0];
        const int in = weightShape[1];
        const int hn = weightShape[2];
        const int wn = weightShape[3];
        std::vector<int64_t> weight_shape{on, in, hn, wn};

        if (weightElementType.isInteger()) {
            auto bitWidth = weightElementType.getIntOrFloatBitWidth();
            std::vector<llvm::APInt> weightData(
                weightValues.value_begin<llvm::APInt>(), weightValues.value_end<llvm::APInt>()
            );
            switch (bitWidth) {
            case 8: {
                // Inflate and reorder weights
                if (isStride2(op) && weight_shape[3] > 1) {
                    // Kernel requires column 1 and column 2 in weights to be swapped
                    weights_swap_even_odd(weightData, weight_shape);
                }
                weights_inflate(weightData, weightShape, parallel_outs, APInt(8, 0));
                weights_OIHW_to_OIHWO(weightData, weightShape, parallel_outs);
                auto weight_tensor = createIConst(rewriter, op, weightData, weightShape);

                rewriter.modifyOpInPlace(op, [&]() {
                    op.setVectorizationMode(vectorizationMode);
                    op.setOperand(1, weight_tensor);
                });

                break;
            }
            default:
                op->emitError() << "Unsupported weight bitwidth: " << bitWidth;
                llvm::report_fatal_error("cannot select kernel");
            }
        }
        else if (weightElementType.isBF16()) {
            std::vector<llvm::APFloat> weightData(
                weightValues.value_begin<llvm::APFloat>(), weightValues.value_end<llvm::APFloat>()
            );
            // Inflate and reorder weights
            if (isStride2(op) && weight_shape[3] > 1) {
                // Kernel requires column 1 and column 2 in weights to be swapped
                weights_swap_even_odd(weightData, weight_shape);
            }
            weights_inflate(
                weightData, weightShape, parallel_outs, APFloat(llvm::APFloat::BFloat(), "0.0")
            );
            weights_OIHW_to_OIHWO(weightData, weightShape, parallel_outs);
            auto weight_tensor = createFConst(rewriter, op, weightData, weightShape);

            rewriter.modifyOpInPlace(op, [&]() {
                op.setVectorizationMode(vectorizationMode);
                op.setOperand(1, weight_tensor);
            });
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
        auto weightValues = mlir::cast<DenseIntElementsAttr>(convWeights);
        auto weightData = weightValues.getRawData().vec();
        auto weightShape = weightValues.getType().getShape().vec();

        auto vectorizationMode = getVectorizationMode(op);

        int parallel_outs = 64;

        weights_inflate(weightData, weightShape, parallel_outs, static_cast<char>(0));
        weights_OIHW_to_OIHWO(weightData, weightShape, parallel_outs);

        // FIXME: use createI8Const from CoversionUtils.h
        auto weight_tensor = createI8Const2(rewriter, op.getLoc(), weightData, weightShape);
        rewriter.modifyOpInPlace(op, [&]() {
            op.setVectorizationMode(vectorizationMode);
            op.setOperand(1, weight_tensor);
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
