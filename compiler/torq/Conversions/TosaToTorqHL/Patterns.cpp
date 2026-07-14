// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/IR/Types.h"
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
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <numeric>
#include <type_traits>

// Set to false to disable weights inflation and reordering
#define TORQHL_WEIGHT_REORDER true

#define DEBUG_TYPE "torq-patterns"

namespace mlir::syna::torq {

namespace {

llvm::cl::opt<bool> clEnableTosaIdentity(
    "torq-enable-tosa-identity", llvm::cl::desc("Enable identity pattern for tosa ops"),
    llvm::cl::init(false)
);

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

LogicalResult getFromConst(Value value, ElementsAttr &attr) {
    if (auto tosaConstOp = value.getDefiningOp<tosa::ConstOp>()) {
        attr = tosaConstOp.getValues();
        return success();
    }
    // Fall back to arith.constant
    if (auto arithConstOp = value.getDefiningOp<arith::ConstantOp>()) {
        attr = mlir::cast<mlir::ElementsAttr>(arithConstOp.getValue());
        return success();
    }
    return failure();
}

template <typename T> FailureOr<SmallVector<T>> getCst(Value val) {
    ElementsAttr attr;
    if (failed(getFromConst(val, attr))) {
        return failure();
    }
    return llvm::to_vector(llvm::map_range(attr.getValues<IntegerAttr>(), [](IntegerAttr a) -> T {
        return static_cast<T>(a.getInt());
    }));
}

static LogicalResult getScaleFactor(tosa::RescaleOp rescaleOp, double &scaleFactor) {
    if (rescaleOp.getPerChannel()) {
        return failure();
    }

    auto maybeShift = getCst<int32_t>(rescaleOp.getShift());
    if (failed(maybeShift)) {
        llvm::errs() << "shift value must be constant in RescaleOp\n";
        return failure();
    }

    auto maybeMultiplier = getCst<int32_t>(rescaleOp.getMultiplier());
    if (failed(maybeMultiplier)) {
        llvm::errs() << "multiplier value must be constant in RescaleOp\n";
        return failure();
    }

    scaleFactor = static_cast<double>((*maybeMultiplier)[0]) / (1l << (*maybeShift)[0]);
    return success();
}

Value fuseInputRescaleIfPossible(
    Value input, int32_t *input_zp = nullptr, double *scale = nullptr
) {
    if (auto rescaleOp = input.getDefiningOp<tosa::RescaleOp>()) {
        // The input is the output of a rescale operator
        // Extract the zero point
        FailureOr<int64_t> in_zp = rescaleOp.getInputZeroPoint();
        if (input_zp && succeeded(in_zp)) {
            if (*input_zp) {
                return input;
            }
            *input_zp = static_cast<int32_t>(*in_zp);
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
        if (mlir::cast<RankedTensorType>(rescaleOp.getOperand(0).getType())
                    .getElementType()
                    .getIntOrFloatBitWidth() == 32 &&
            mlir::cast<RankedTensorType>(rescaleOp.getResult().getType())
                    .getElementType()
                    .getIntOrFloatBitWidth() == 32) {
            // the rescale is a 32 to 32 bit operation, we can fuse it with its
            // parent rescale that hopefully is 8 to 32 bit
            return fuseInputRescaleIfPossible(rescaleOp.getOperand(0), input_zp, scale);
        }
        else if (mlir::cast<RankedTensorType>(rescaleOp.getResult().getType())
                     .getElementType()
                     .getIntOrFloatBitWidth() == 32) {
            // always fuse if rescale 8 -> 32
            return rescaleOp.getOperand(0);
        }
    }

    // Nothing to fuse
    return input;
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

    FailureOr<int64_t> outputZp = srcOp.getOutputZeroPoint();
    if (!succeeded(outputZp)) {
        return rewriter.notifyMatchFailure(srcOp, "RescaleOp output zero point not found");
    }
    const int32_t output_zp = static_cast<int32_t>(*outputZp);

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

    auto avgPool2DOp = syna::torq_hl::AvgPool2DOp::create(
        rewriter, srcOp.getLoc(), out_type, createInitTensor(srcOp, rewriter, out_type),
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
            return rewriter.notifyMatchFailure(srcOp, "Cannot scatter non-constant indices");
        }

        auto values = valuesAttr.getValues<int32_t>();

        std::vector<int16_t> valuesRef(values.begin(), values.end());
        auto indices_value = createI16Const(rewriter, srcOp, valuesRef);

        auto input = convertNHCtoNCH(adaptor.getInput(), srcOp.getLoc(), rewriter);

        auto outType = convertTypeNHCtoNCH(srcOp.getValuesIn().getType());

        auto output = torq_hl::ScatterOp::create(
            rewriter, srcOp.getLoc(), outType, valuesIn, indices_value, input,
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

} // namespace

void populateTOSAToTorqHLPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    if (clEnableTosaIdentity) {
        patterns.insert<IdentityOpConversion>(context);
    }
    patterns.insert<ArgMaxOpConversion>(context);
    patterns.insert<ScatterOpConversion>(context);
}

} // namespace mlir::syna::torq
