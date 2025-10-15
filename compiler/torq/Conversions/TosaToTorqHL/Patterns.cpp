// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <type_traits>

// Set to false to disable weights inflation and reordering
#define TORQHL_WEIGHT_REORDER true

#define DEBUG_TYPE "torq-patterns"

namespace mlir::syna::torq {

namespace {

static bool typeIsInt8(Type type) {
    Type elementType = getElementTypeOrSelf(type);
    return elementType.isSignedInteger() && elementType.getIntOrFloatBitWidth() == 8;
}

static bool checkOpIsInt8(Operation *op) {
    if (!op->getNumResults()) {
        return false;
    }
    Type elementType = getElementTypeOrSelf(op->getResult(0).getType());
    if (typeIsInt8(elementType)) {
        return false;
    }

    return true;
}

static LogicalResult getScaleFactor(tosa::RescaleOp &rescaleOp, double &scaleFactor) {
    if (rescaleOp.getPerChannel()) {
        return failure();
    }
    scaleFactor =
        static_cast<double>(rescaleOp.getMultiplier()[0]) / (1l << rescaleOp.getShift()[0]);

    return success();
}

Value fuseInputRescaleIfPossible(
    Value input, int32_t *input_zp = nullptr, double *scale = nullptr
) {
    if (auto rescaleOp = input.getDefiningOp<tosa::RescaleOp>()) {
        // The input is the output of a rescale operator
        // Extract the zero point
        uint32_t in_zp = rescaleOp.getInputZp();
        if (input_zp && in_zp) {
            if (*input_zp) {
                return input;
            }
            *input_zp = in_zp;
        }

        // Extract the scale
        if (scale) {
            double scaleFactor;

            // we cannot compute the scale factor, so we cannot fuse
            if (failed(getScaleFactor(rescaleOp, scaleFactor))) {
                return input;
            }

            *scale *= scaleFactor;
        }

        // let's try to fuse the rescale with the current operator
        if (mlir::cast<RankedTensorType>(rescaleOp.getOperand().getType())
                    .getElementType()
                    .getIntOrFloatBitWidth() == 32 &&
            mlir::cast<RankedTensorType>(rescaleOp.getResult().getType())
                    .getElementType()
                    .getIntOrFloatBitWidth() == 32) {
            // the rescale is a 32 to 32 bit operation, we can fuse it with its
            // parent rescale that hopefully is 8 to 32 bit
            return fuseInputRescaleIfPossible(rescaleOp.getOperand(), input_zp, scale);
        }
        else if (mlir::cast<RankedTensorType>(rescaleOp.getResult().getType())
                     .getElementType()
                     .getIntOrFloatBitWidth() == 32) {
            // always fuse if rescale 8 -> 32
            return rescaleOp.getOperand();
        }
    }

    // Nothing to fuse
    return input;
}

LogicalResult getFromConst(Value value, ElementsAttr &attr) {
    tosa::ConstOp constOp = mlir::dyn_cast<tosa::ConstOp>(value.getDefiningOp());
    if (!constOp) {
        return failure();
    }
    attr = constOp.getValue();

    return success();
}

LogicalResult getFromArithConst(Value value, ElementsAttr &attr) {
    arith::ConstantOp constOp = mlir::dyn_cast<arith::ConstantOp>(value.getDefiningOp());
    if (!constOp) {
        return failure();
    }
    attr = mlir::cast<mlir::ElementsAttr>(constOp.getValue());

    return success();
}

// Weights converted to
// Example 32x4x4x32
// Expand 32x2x2x2x2x32
// Transpose 2x2x32x2x2x32 perm [2, 4, 0, 1, 5, 3]
// Collapse 128x2x2x32
// Reverse oxhxwxi -> oxrev_hxrev_wxi
LogicalResult reverseTConvWeights(
    tosa::RescaleOp srcOp, Value &weights, llvm::SmallVector<int64_t> outputShape,
    ConversionPatternRewriter &rewriter
) {
    mlir::Location loc = mlir::NameLoc::get(rewriter.getStringAttr("weight_restructure_op"));
    auto elementType = mlir::cast<RankedTensorType>(weights.getType()).getElementType();

    llvm::SmallVector<llvm::SmallVector<int64_t, 2>> expandReassoc{{0}, {1, 2}, {3, 4}, {5}};
    llvm::SmallVector<int64_t> modWeightShape{outputShape[0],     2,
                                              outputShape[1] / 2, 2,
                                              outputShape[2] / 2, outputShape[3]};
    auto expandType = mlir::RankedTensorType::get(modWeightShape, elementType);

    auto expand = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandType, weights, llvm::ArrayRef<llvm::SmallVector<int64_t, 2>>(expandReassoc)
    );

    llvm::SmallVector<int64_t> perm{2, 4, 0, 1, 3, 5};
    llvm::SmallVector<int64_t> modTransposeShape{
        outputShape[1] / 2, outputShape[2] / 2, outputShape[0], 2, 2, outputShape[3]
    };
    auto transposeType = mlir::RankedTensorType::get(modTransposeShape, elementType);
    auto transposeOutput = createInitTensor(loc, transposeType, rewriter);
    auto transpose =
        rewriter.create<linalg::TransposeOp>(loc, expand.getResult(), transposeOutput, perm);

    llvm::SmallVector<llvm::SmallVector<int64_t, 2>> collapseReassoc{{0, 1, 2}, {3}, {4}, {5}};
    llvm::SmallVector<int64_t> newWeightShape{
        outputShape[0] * (outputShape[1] / 2) * (outputShape[2] / 2), 2, 2, outputShape[3]
    };
    auto collapseType = mlir::RankedTensorType::get(newWeightShape, elementType);

    auto collapse = rewriter.create<tensor::CollapseShapeOp>(
        loc, collapseType, *transpose.getResults().begin(),
        llvm::ArrayRef<llvm::SmallVector<int64_t, 2>>(collapseReassoc)
    );

    // 32x4x4x32
    // 32x2x2x2x2x32
    // 2x2x32x2x2x32
    // 128x2x2x32
    // rev : 128x2x2x32
    // 0 1
    // 2 3
    // -> 3 2
    //	  1 0
    // linalg.generic(1-i, 1-j -> i, j)
    AffineExpr o, h, w, i;
    bindDims(rewriter.getContext(), o, h, w, i);

    auto inputMap = AffineMap::get(4, 0, {o, 1 - h, 1 - w, i}, rewriter.getContext());
    auto outputMap = AffineMap::get(4, 0, {o, h, w, i}, rewriter.getContext());

    llvm::SmallVector<AffineMap, 2> indexingMaps{inputMap, outputMap};

    SmallVector<mlir::utils::IteratorType> iteratorTypes(4, mlir::utils::IteratorType::parallel);

    auto newWeights = createInitTensor(loc, collapseType, rewriter);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{newWeights.getType()}, ValueRange{collapse.getResult()},
        ValueRange{newWeights}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
            b.create<linalg::YieldOp>(loc, args[0]);
        }
    );

    llvm::SmallVector<int64_t> oihwPerm{0, 3, 1, 2};
    llvm::SmallVector<int64_t> oihwShape{
        newWeightShape[0], newWeightShape[3], newWeightShape[1], newWeightShape[2]
    };
    auto oihwType = mlir::RankedTensorType::get(oihwShape, elementType);
    auto oihwOutput = createInitTensor(loc, oihwType, rewriter);
    auto oihwTransposeOp =
        rewriter.create<linalg::TransposeOp>(loc, genericOp.getResult(0), oihwOutput, oihwPerm);
    auto newWeightsOIHW = computeValue(*oihwTransposeOp.getResults().begin(), true, {});
    if (failed(newWeightsOIHW)) {
        return rewriter.notifyMatchFailure(oihwTransposeOp, "Weight computation failed");
    }

    weights = createConst(*newWeightsOIHW, rewriter, loc);

    return success();
}

std::vector<int32_t> duplicateBiasScale(std::vector<int32_t> &biasScale, int n) {
    std::vector<int32_t> modBiasScale;
    modBiasScale.reserve(biasScale.size() * n); // Optional: reserve space for efficiency

    for (int i = 0; i < n; ++i) {
        modBiasScale.insert(modBiasScale.end(), biasScale.begin(), biasScale.end());
    }
    return modBiasScale;
}

template <typename Iterator>
std::vector<int32_t> convertAPIntRangeToInt64Vector(Iterator begin, Iterator end) {
    std::vector<int32_t> result;
    result.reserve(std::distance(begin, end));
    for (auto it = begin; it != end; ++it) {
        result.push_back((*it).getSExtValue()); // or getZExtValue() for unsigned
    }
    return result;
}

// tosa.transpose_conv2d
// Converts:
// Conv + DepthToSpace
// Weights are rearranged accordingly
// 32x4x4x32 -> 128x2x2x32
static LogicalResult fuseWithRescaleOp(
    tosa::TransposeConv2DOp tconvOp, tosa::RescaleOp rescaleOp, ConversionPatternRewriter &rewriter
) {
    auto elementType =
        mlir::cast<RankedTensorType>(rescaleOp.getResult().getType()).getElementType();

    auto weights = tconvOp.getFilter();
    llvm::ArrayRef<int64_t> weightShapeArr =
        mlir::cast<RankedTensorType>(weights.getType()).getShape();
    llvm::SmallVector<int64_t> weightShape(weightShapeArr.begin(), weightShapeArr.end());

    if (failed(reverseTConvWeights(rescaleOp, weights, weightShape, rewriter)
        )) /*Rearrange the weights in SpaceToDepth style modification and reverse the weights*/ {
        return rewriter.notifyMatchFailure(rescaleOp, "TConv reverse failed");
    }

    auto tconvOutput = rescaleOp.getOutput();
    llvm::ArrayRef<int64_t> outputShapeArr =
        mlir::cast<RankedTensorType>(tconvOutput.getType()).getShape();
    llvm::SmallVector<int64_t> outputShape(outputShapeArr.begin(), outputShapeArr.end());

    auto transInput = tconvOp.getInput();

    int blockSize = tconvOp.getStride()[0]; // TODO current support only for blocksize 2
    assert(blockSize == 2 && "Current upscale of only 2 supported");

    auto vectorizationMode = torq_hl::VectorizationModeEnum::None;

    auto pad = rewriter.getDenseI64ArrayAttr({0, 1, 0, 1});
    auto stride = rewriter.getDenseI64ArrayAttr({1, 1});
    auto dilation = rewriter.getDenseI64ArrayAttr({1, 1});

    auto inputZp = tconvOp.getQuantizationInfo()->getInputZp();
    auto outputZp = rescaleOp.getOutputZp();

    auto tosaBias = tconvOp.getBias();

    int32_t outMin = -128;
    int32_t outMax = 127;
    if (elementType.isInteger(16)) {
        outMin = -32768;
        outMax = 32767;
    }

    ElementsAttr weightValues;
    if (failed(getFromArithConst(weights, weightValues))) {
        return rewriter.notifyMatchFailure(tconvOp, "TConv weights not found");
    }
    std::vector<int8_t> weightVec(
        weightValues.getValues<int8_t>().begin(), weightValues.getValues<int8_t>().end()
    );

    // Re-fetch new weight shape after transformation
    llvm::ArrayRef<int64_t> newWeightShapeArr =
        mlir::cast<RankedTensorType>(weights.getType()).getShape();
    llvm::SmallVector<int64_t> newWeightShape(newWeightShapeArr.begin(), newWeightShapeArr.end());

    // Compute bias
    auto per_channel_weights_sum = per_channel_sum(
        weightVec, newWeightShape[0], newWeightShape[1], newWeightShape[2], newWeightShape[3]
    );

    ElementsAttr tosaBiasAttr;
    if (failed(getFromConst(tosaBias, tosaBiasAttr))) {
        return rewriter.notifyMatchFailure(tconvOp, "TConv bias not found");
    }
    auto tosaBiasValues = tosaBiasAttr.getValues<llvm::APInt>();
    auto tosaBiasCastI64 =
        convertAPIntRangeToInt64Vector(tosaBiasValues.begin(), tosaBiasValues.end());
    auto duplicated_bias = duplicateBiasRoundRobin(tosaBiasCastI64, newWeightShape[0]);
    auto bias = computeCorrectedBiasRoundRobin(duplicated_bias, per_channel_weights_sum, inputZp);

    auto shifts = rescaleOp.getShift();
    int shiftFactor = *std::min_element(shifts.begin(), shifts.end());
    auto tosaMultiplier = rescaleOp.getMultiplier();
    auto scale = compute_scale(tosaMultiplier, shifts, shiftFactor);
    if (scale.size() == 1) {
        scale = std::vector<int32_t>(newWeightShape[0], scale[0]);
    }
    else if (scale.size() != newWeightShape[0]) {
        scale =
            duplicateBiasRoundRobin(scale, newWeightShape[0]); // reuse duplication logic for scale
    }

    auto biasScale = interleave(bias, scale);
    auto biasScaleValue = createConst(biasScale, rewriter, rescaleOp.getLoc());

    Value input =
        transposeValue(transInput, Permutation::nhwc2nchw(), rescaleOp.getLoc(), rewriter);
    auto inputShape = mlir::cast<RankedTensorType>(input.getType()).getShape();

    llvm::SmallVector<int64_t> padOutShape{
        inputShape[0], inputShape[1], inputShape[2] + 1, inputShape[3] + 1
    };

    RankedTensorType padInitType =
        RankedTensorType::get(llvm::ArrayRef<int64_t>(padOutShape), elementType);

    llvm::SmallVector<OpFoldResult> lowPad{
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)
    };
    llvm::SmallVector<OpFoldResult> highPad{
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(0)
    };
    Value zero = rewriter.create<arith::ConstantOp>(
        rescaleOp.getLoc(), rewriter.getIntegerAttr(elementType, inputZp)
    );

    // Conv2D
    auto padOp = rewriter.create<tensor::PadOp>(
        rescaleOp.getLoc(), padInitType, input, lowPad, highPad, zero
    );

    auto padOut =
        transposeValue(padOp, Permutation::nhwc2nchw().reverse(), rescaleOp.getLoc(), rewriter);
    auto convInput = transposeValue(padOut, Permutation::nhwc2nchw(), rescaleOp.getLoc(), rewriter);

    llvm::SmallVector<int64_t> convOutShape{
        padOutShape[0], padOutShape[1] * blockSize * blockSize, padOutShape[2], padOutShape[3]
    }; // output shape of conv will be square of the blocksize, so that next
       // depth2space will rearrange it
    RankedTensorType convInitType =
        RankedTensorType::get(llvm::ArrayRef<int64_t>(convOutShape), elementType);

    auto convInitTensor = createInitTensor(rescaleOp.getLoc(), convInitType, rewriter);

    auto conv2dOp = rewriter.create<syna::torq_hl::Conv2DOp>(
        rescaleOp.getLoc(), convInitType, convInitTensor,
        inputZp, // input_zp
        0,       // weight_zp
        outputZp, outMin, outMax, shiftFactor,
        1,        // groups
        pad,      // pad
        stride,   // stride
        dilation, // dilation
        vectorizationMode, weights, biasScaleValue, convInput
    );

    auto convOut = transposeValue(
        conv2dOp.getOutput(), Permutation::nhwc2nchw().reverse(), rescaleOp.getLoc(), rewriter
    );

    // DepthToSpace
    int dtype_size = elementType.getIntOrFloatBitWidth() / 8;
    const auto wram_width = 32 / dtype_size;
    const auto num_inputs = 2;

    auto d2s_input_type = Permutation::nhwc2nchw();
    auto d2s_output_type = d2s_input_type.reverse();
    auto d2s_input = transposeValue(convOut, d2s_input_type, rescaleOp.getLoc(), rewriter);

    // Create weights for the d2s interleaving operation
    auto d2s_weights = genD2SWeights(wram_width);
    auto d2s_enum_mode = torq_hl::DepthToSpaceModeEnum::DCR;

    llvm::SmallVector<int64_t> d2sOutputShape{
        convOutShape[0], convOutShape[1] / (blockSize * blockSize), convOutShape[2] * blockSize,
        convOutShape[3] * blockSize
    };
    RankedTensorType d2sInitType =
        RankedTensorType::get(llvm::ArrayRef<int64_t>(d2sOutputShape), elementType);

    auto d2sOp = rewriter.create<torq_hl::DepthToSpaceOp>(
        rescaleOp.getLoc(), d2sInitType,
        createInitTensor(rescaleOp.getLoc(), d2sInitType, rewriter), blockSize, d2s_enum_mode,
        createI8Const(
            rewriter, rescaleOp, d2s_weights, llvm::ArrayRef<int64_t>{wram_width * num_inputs}
        ),
        // transposeValue(conv2dOp.getOperand(0), d2s_input_type, rescaleOp.getLoc(), rewriter)
        d2s_input
    );
    auto d2sTransposedOut =
        transposeValue(d2sOp.getOutput(), d2s_output_type, rescaleOp.getLoc(), rewriter);

    // Extract slice
    llvm::SmallVector<OpFoldResult> extractOffset = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)
    };

    llvm::SmallVector<OpFoldResult> extractSize = {
        rewriter.getIndexAttr(d2sOutputShape[0]), rewriter.getIndexAttr(d2sOutputShape[1]),
        rewriter.getIndexAttr(d2sOutputShape[2] - 2), rewriter.getIndexAttr(d2sOutputShape[3] - 2)
    };

    llvm::SmallVector<OpFoldResult> extractStrides = {
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
        rewriter.getIndexAttr(1)
    };

    Value extractInput =
        transposeValue(d2sTransposedOut, Permutation::nhwc2nchw(), rescaleOp.getLoc(), rewriter);
    auto extractSlice = rewriter.create<tensor::ExtractSliceOp>(
        rescaleOp.getLoc(), extractInput, extractOffset, extractSize, extractStrides
    );
    auto extractOut = transposeValue(
        extractSlice, Permutation::nhwc2nchw().reverse(), rescaleOp.getLoc(), rewriter
    );

    rewriter.eraseOp(tconvOp);
    rewriter.replaceOp(rescaleOp, extractOut);
    return success();
}

static LogicalResult fuseWithRescaleOp(
    tosa::ReduceSumOp op, tosa::RescaleOp srcOp, ConversionPatternRewriter &rewriter
) {

    Operation *inputOp = op;

    // fuse multiple reduce sum operators in one single operator
    Value tosaOpInput;
    std::vector<int32_t> axis;
    while (tosa::ReduceSumOp tosaReduceSumOp =
               mlir::dyn_cast_if_present<tosa::ReduceSumOp>(inputOp)) {
        axis.insert(axis.begin(), tosaReduceSumOp.getAxis());
        tosaOpInput = tosaReduceSumOp.getOperand();
        inputOp = tosaOpInput.getDefiningOp();
    }

    // at the moment we implement multiple reduce sums with an AvgPool layer so
    // we must make sure the reduction is compatible
    if (axis.size() != 2 || axis[0] != 1 || axis[1] != 2) {
        return failure();
    }

    RankedTensorType input_type = mlir::cast<RankedTensorType>(tosaOpInput.getType());
    auto in_s = input_type.getShape();
    int32_t input_zp = 0;
    double input_scale = 1.0;
    tosaOpInput = fuseInputRescaleIfPossible(tosaOpInput, &input_zp, &input_scale);
    const int32_t output_zp = srcOp.getOutputZp();
    double output_scale;
    if (failed(getScaleFactor(srcOp, output_scale))) {
        return failure();
    }

    int32_t out_min = -128;
    int32_t out_max = 127;
    constexpr int shift_factor = 20;
    double multiplier = output_scale / input_scale;

    if (!srcOp.getOutput().hasOneUse()) {
        return failure();
    }

    // Input shape is NHWC
    if (in_s.size() != 4) {
        // Currently reduce sum is implemented as average_pool2d so it only
        // supports 4D tensors
        return failure();
    }
    const int channel_dimension = 3;
    if (std::find(in_s.begin(), in_s.end(), channel_dimension) != in_s.end()) {
        return failure();
    }
    int32_t num_channels = in_s[channel_dimension];

    // Compute how many elements are reduced into one
    int32_t reduced_elements =
        std::accumulate(axis.begin(), axis.end(), 1, [&in_s](int32_t acc, int32_t ax) {
            return acc * in_s[ax];
        });
    const std::vector<APInt> scale(num_channels, APInt(32, multiplier * (1 << shift_factor)));
    const std::vector<APInt> bias(num_channels, APInt(32, -reduced_elements * input_zp));

    Operation *targetOp = srcOp;
    ArrayRef<int64_t> outShape = srcOp.getType().getShape();

    if (srcOp->hasOneUse()) {
        if (auto reshapeOp = dyn_cast<tosa::ReshapeOp>(*(srcOp.getOutput().user_begin()))) {
            outShape = reshapeOp.getType().getShape();
            targetOp = reshapeOp.getOperation();
        }
    }

    auto out_type = RankedTensorType::get(outShape, srcOp.getResult().getType().getElementType());

    auto avgPool2DOp = rewriter.create<syna::torq_hl::AvgPool2DOp>(
        srcOp.getLoc(), out_type, createInitTensor(srcOp, rewriter, out_type),
        MakeI32Attr(srcOp, input_zp), MakeI32Attr(srcOp, output_zp), MakeI32Attr(srcOp, out_min),
        MakeI32Attr(srcOp, out_max), MakeI32Attr(srcOp, shift_factor),
        createI8Const(rewriter, srcOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}),
        createIConst(rewriter, srcOp, interleave(bias, scale)), tosaOpInput
    );

    rewriter.replaceOp(targetOp, avgPool2DOp.getOutput());

    // we need to ensure the srcOp is removed as it it the root of the pattern
    if (targetOp != srcOp) {
        rewriter.eraseOp(srcOp);
    }

    return success();
}

struct RescaleOpConversion : public OpConversionPattern<tosa::RescaleOp> {

    RescaleOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::RescaleOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        // input is linked to a block argument
        Operation *inputOp = adaptor.getInput().getDefiningOp();
        if (!inputOp) {

            auto isReturnOp = [](Operation *op) { return mlir::isa<func::ReturnOp>(op); };

            // if only rescaleOp in model, convert to FMAOp
            // it is used for single rescaleOp check, in practice, fmaOp will work with specific op
            // together
            if (llvm::all_of(srcOp.getResult().getUsers(), isReturnOp)) {
                return rewriter.notifyMatchFailure(srcOp, "RescaleOp is processed in linalg");
            }

            return failure();
        }

        // fuse operation with operation that generates it, if supported
        auto ret = TypeSwitch<Operation *, LogicalResult>(inputOp)
                       .Case<tosa::ReduceSumOp>([&](auto typedOp) {
                           return fuseWithRescaleOp(typedOp, srcOp, rewriter);
                       })
                       .Case<tosa::TransposeConv2DOp>([&](auto typedOp) {
                           return fuseWithRescaleOp(typedOp, srcOp, rewriter);
                       })
                       .Default([&](Operation *op) {
                           return rewriter.notifyMatchFailure(
                               srcOp, "Unsupported input operation for RescaleOp"
                           );
                       });
        return ret;
    }
};

// follow tosa spec for scaler caculation
typedef struct {
    int32_t multiplier;
    int8_t shift;
} scale_t;

int32_t count_leading_zeros(int32_t a) {
    int32_t acc = 32;
    if (a != 0) {
        uint32_t mask;
        mask = 1 << (32 - 1); // width of int32_t - 1
        acc = 0;
        while ((mask & a) == 0) {
            mask = mask >> 1;
            acc = acc + 1;
        }
    }
    return acc;
}

scale_t reciprocal_scale(uint32_t value) {
    assert(value > 0);
    scale_t scale;
    int32_t k = 32 - count_leading_zeros(value - 1); // (1 << k) / 2 < value <= (1 << k)
    int64_t numerator = ((1 << 30) + 1) << k;
    scale.multiplier = numerator / value; // (1 << 30) <= multiplier < (1 << 31)
    scale.shift = 30 + k;
    return scale;
}

struct TableOpConversion : public OpConversionPattern<tosa::TableOp> {
    TableOpConversion(MLIRContext *context) : OpConversionPattern(context) {}

    LogicalResult matchAndRewrite(
        tosa::TableOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        Value tableTensor = adaptor.getTable();

        // Perform table lookup using a gather operation if the table is int8.
        auto tableType = mlir::dyn_cast<RankedTensorType>(tableTensor.getType());
        if (tableType.getElementType().isInteger(8)) {
            return rewriter.notifyMatchFailure(srcOp, "table i8 is implemented in linalg");
        }

        // Get input operands.
        Value input = adaptor.getInput();

        // Retrieve the op that defines the table tensor.
        Operation *definingOp = tableTensor.getDefiningOp();
        if (!definingOp) {
            return failure();
        }

        DenseElementsAttr attr;
        if (auto constOp = mlir::dyn_cast<tosa::ConstOp>(definingOp)) {
            attr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValue());
        }
        else if (auto arithConst = mlir::dyn_cast<arith::ConstantOp>(definingOp)) {
            attr = mlir::dyn_cast<DenseElementsAttr>(arithConst.getValue());
        }
        else {
            return failure();
        }

        SmallVector<int32_t> convertedValues;
        auto elementType = attr.getType().getElementType();
        if (elementType.isInteger(16)) {
            auto values = attr.getValues<int16_t>();
            for (size_t i = 0; i < 512; i++) {
                int16_t tb = values[i];                               // Extract base value
                int16_t ts = (values[i + 1] - values[i]);             // Compute slope
                int32_t packed = ((int32_t)ts << 16) | (tb & 0xFFFF); // Pack slope and base
                convertedValues.push_back(packed);
            }
        }
        else if (elementType.isInteger(8)) {
            auto values = attr.getValues<int8_t>();
            for (size_t i = 0; i < 256; i++) {
                int32_t shiftedValue = static_cast<int32_t>(values[i]) << 8;
                convertedValues.push_back(shiftedValue);
            }
        }
        DenseI32ArrayAttr intArrayAttr =
            DenseI32ArrayAttr::get(rewriter.getContext(), convertedValues);
        Value initTensor = createInitTensor(srcOp, rewriter);

        const std::vector<APInt> bias = {APInt(32, -128, /*isSigned=*/true)};
        const std::vector<APInt> scale = {APInt(32, 128, /*isSigned=*/true)};

        rewriter.replaceOpWithNewOp<syna::torq_hl::TableOp>(
            srcOp, srcOp.getOutput().getType(), initTensor,
            createIConst(rewriter, srcOp, interleave(bias, scale)), input, intArrayAttr
        );

        return success();
    }
};

struct ArgMaxOpConversion : public OpConversionPattern<tosa::ArgMaxOp> {

    ArgMaxOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::ArgMaxOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        if (!checkOpIsInt8(srcOp)) {
            return failure();
        }

        Location loc = srcOp.getLoc();
        Value inputVal = adaptor.getInput();
        int64_t axisVal = srcOp.getAxis();

        // Only do transpose trick if argmax on axis=2 on an [N, C, H] layout
        // so we can perform argmax on axis=1 in [N, H, C].
        bool transposeInput = (axisVal == 2);
        if (transposeInput) {
            inputVal = convertNCHtoNHC(inputVal, loc, rewriter);
            axisVal = 1;
        }

        auto axisAttr = rewriter.getI32IntegerAttr(axisVal);
        const std::vector<APInt> bias = {APInt(32, 0)};
        const std::vector<APInt> scale = {APInt(32, 1)};

        rewriter.replaceOpWithNewOp<syna::torq_hl::ArgMaxOp>(
            srcOp, srcOp.getOutput().getType(), createInitTensor(srcOp, rewriter), axisAttr,
            createIConst(rewriter, srcOp, interleave(bias, scale)), inputVal
        );

        return success();
    }
};

struct AvgPool2DOpConversion : public OpConversionPattern<tosa::AvgPool2dOp> {

    AvgPool2DOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::AvgPool2dOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        const int64_t input_zp = srcOp.getQuantizationInfo()->getInputZp();
        const int64_t output_zp = srcOp.getQuantizationInfo()->getOutputZp();

        int32_t out_min = -128;
        int32_t out_max = 127;

        Type elementType = getElementTypeOrSelf(srcOp.getResult().getType());
        if (elementType.isSignedInteger() && elementType.getIntOrFloatBitWidth() == 8) {
            out_min = -128;
            out_max = 127;
        }
        else {
            return failure();
        }

        llvm::ArrayRef<int64_t> kernel = srcOp.getKernel();
        assert(kernel.size() == 2);

        int32_t count = kernel[0] * kernel[1];
        scale_t scaler = reciprocal_scale(count);

        auto input_shape = mlir::cast<RankedTensorType>(srcOp.getInput().getType()).getShape();
        // tosa spec request
        assert(input_shape.size() == 4);

        int32_t num_channels = input_shape[3];

        const std::vector<APInt> scale(
            num_channels, APInt(32, scaler.multiplier * (1 << scaler.shift))
        );
        const std::vector<APInt> bias(num_channels, APInt(32, count * input_zp));

        if (!srcOp->hasOneUse()) {
            return failure();
        }

        auto reshapeOp = dyn_cast<tosa::ReshapeOp>(*(srcOp.getOutput().user_begin()));

        if (!reshapeOp) {
            return failure();
        }

        auto outShape = reshapeOp.getType().getShape();

        auto out_type =
            RankedTensorType::get(outShape, srcOp.getResult().getType().getElementType());

        auto avgPool2DOp = rewriter.replaceOpWithNewOp<syna::torq_hl::AvgPool2DOp>(
            srcOp, out_type, createInitTensor(srcOp, rewriter, out_type),
            MakeI32Attr(srcOp, input_zp), MakeI32Attr(srcOp, output_zp),
            MakeI32Attr(srcOp, out_min), MakeI32Attr(srcOp, out_max),
            MakeI32Attr(srcOp, scaler.shift),
            createI8Const(
                rewriter, srcOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
            ),
            createIConst(rewriter, srcOp, interleave(bias, scale)),
            fuseInputRescaleIfPossible(adaptor.getInput())
        );

        rewriter.replaceOp(reshapeOp, avgPool2DOp.getOutput());

        // we need to ensure the srcOp is removed as it it the root of the pattern
        rewriter.eraseOp(srcOp);

        return success();
    }
};

struct IdentityOpConversion : public OpConversionPattern<tosa::IdentityOp> {

    IdentityOpConversion(MLIRContext *context) : OpConversionPattern(context) {}
    LogicalResult matchAndRewrite(
        tosa::IdentityOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        rewriter.replaceOpWithNewOp<torq_hl::IdentityOp>(
            srcOp, srcOp.getOutput().getType(), createInitTensor(srcOp, rewriter),
            adaptor.getInput1()
        );
        return success();
    }
};

struct ScatterOpConversion : public OpConversionPattern<tosa::ScatterOp> {
    ScatterOpConversion(MLIRContext *context) : OpConversionPattern(context) {}

    LogicalResult matchAndRewrite(
        tosa::ScatterOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        auto valuesIn = convertNHCtoNCH(adaptor.getValuesIn(), srcOp.getLoc(), rewriter);
        auto indices = srcOp.getIndices();

        // TODO Values are currently loaded from const,
        // but it may come from input.
        // Can check when we have proper model

        ElementsAttr valuesAttr;

        if (failed(getFromConst(indices, valuesAttr))) {
            return failure();
        }

        auto values = valuesAttr.getValues<int32_t>();

        std::vector<int16_t> valuesRef(values.begin(), values.end());
        auto indices_value = createI16Const(rewriter, srcOp, valuesRef);

        auto input = convertNHCtoNCH(adaptor.getInput(), srcOp.getLoc(), rewriter);

        auto outType = convertTypeNHCtoNCH(srcOp.getValuesIn().getType());

        auto output = rewriter.create<torq_hl::ScatterOp>(
            srcOp.getLoc(), outType, valuesIn, indices_value, input,
            createI32Const(rewriter, srcOp, std::vector<int32_t>{0, 1})
        );

        auto transposeOp = convertNCHtoNHC(output.getOutput(), srcOp.getLoc(), rewriter);
        rewriter.replaceOp(srcOp, transposeOp);
        return success();
    }
};

template <typename OpTy>
struct ElementWiseShiftOpConversion final : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpTy srcOp, typename OpTy::Adaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {

        torq_hl::ShiftModeEnum opName;
        bool round = false;
        if constexpr (std::is_same_v<OpTy, tosa::ArithmeticRightShiftOp>) {

            auto input1Type = dyn_cast<RankedTensorType>(adaptor.getInput1().getType());
            auto input2Type = dyn_cast<RankedTensorType>(adaptor.getInput2().getType());
            if (!input1Type || !input2Type || (input1Type != input2Type)) {
                return rewriter.notifyMatchFailure(
                    srcOp, "Input1 and Input2 must have the same type for ASR operation"
                );
            }

            opName = torq_hl::ShiftModeEnum::ASR;
            round = srcOp.getRound();
        }
        else {
            llvm::report_fatal_error("Unsupported elementwise shift operation", true);
            return failure();
        }

        rewriter.replaceOpWithNewOp<torq_hl::ElementWiseShiftOp>(
            srcOp, srcOp.getResult().getType(), createInitTensor(srcOp, rewriter), opName, round,
            adaptor.getInput1(), adaptor.getInput2()
        );

        return success();
    }
};

struct ResizeNearestNeighborOpConversion : public OpConversionPattern<tosa::ResizeOp> {
    ResizeNearestNeighborOpConversion(MLIRContext *context) : OpConversionPattern(context) {}

    LogicalResult matchAndRewrite(
        tosa::ResizeOp srcOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter
    ) const override {
        auto scale = adaptor.getScale();
        auto mode = adaptor.getMode();
        if (mode != "NEAREST_NEIGHBOR") {
            return srcOp.emitError("Current support only for NEAREST_NEIGHBOR with scale 2");
        }
        auto output = rewriter.create<syna::torq_hl::ResizeNearestNeighborOp>(
            srcOp.getLoc(), convertTypeNHWCtoNCHW(srcOp.getResult().getType()),
            createInitTensorNCHW(srcOp, rewriter), scale[0],
            convertNHWCtoNCHW(srcOp.getInput(), srcOp.getLoc(), rewriter)
        );
        auto transposeOp = convertNCHWtoNHWC(output.getOutput(), srcOp.getLoc(), rewriter);
        rewriter.replaceOp(srcOp, transposeOp);
        return success();
    }
};

} // namespace

void populateTOSAToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns) {

    patterns.insert<RescaleOpConversion>(context);
    patterns.insert<AvgPool2DOpConversion>(context);
    patterns.insert<TableOpConversion>(context);
    patterns.insert<IdentityOpConversion>(context);
    patterns.insert<ArgMaxOpConversion>(context);
    patterns.insert<ScatterOpConversion>(context);
    patterns.insert<ResizeNearestNeighborOpConversion>(context);
}

} // namespace mlir::syna::torq
