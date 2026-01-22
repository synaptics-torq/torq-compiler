// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::syna::torq {

// Convert a vector of integers to a SmallVector of OpFoldResult
SmallVector<OpFoldResult> createVector(::llvm::ArrayRef<int64_t> values, PatternRewriter &rewriter);

template <typename T, typename R> Value createInitTensor(T &srcOp, R &rewriter) {
    return rewriter
        .template create<mlir::tensor::EmptyOp>(
            srcOp.getLoc(), srcOp.getResult().getType().getShape(),
            srcOp.getResult().getType().getElementType()
        )
        .getResult();
}

inline Value
createInitTensor(Location loc, RankedTensorType type, ConversionPatternRewriter &rewriter) {
    return rewriter.create<mlir::tensor::EmptyOp>(loc, type.getShape(), type.getElementType())
        .getResult();
}

template <typename T, typename R>
Value createInitTensor(T &srcOp, R &rewriter, RankedTensorType type) {
    return rewriter
        .template create<mlir::tensor::EmptyOp>(
            srcOp.getLoc(), type.getShape(), type.getElementType()
        )
        .getResult();
}

Value createZeroConstant(OpBuilder &b, Location loc, Type elemTy);

template <class T> IntegerAttr MakeI32Attr(T &op, int32_t value) {
    return IntegerAttr::get(IntegerType::get(op.getContext(), 32), value);
}

inline bool isBF16(const llvm::APFloat &x) { return &x.getSemantics() == &llvm::APFloat::BFloat(); }
inline bool isF16(const llvm::APFloat &x) {
    return &x.getSemantics() == &llvm::APFloat::IEEEhalf();
}
inline bool isF32(const llvm::APFloat &x) {
    return &x.getSemantics() == &llvm::APFloat::IEEEsingle();
}
inline bool isF64(const llvm::APFloat &x) {
    return &x.getSemantics() == &llvm::APFloat::IEEEdouble();
}

template <typename T> Type getMLIRFloatType(MLIRContext *context, const T &value) {
    if constexpr (std::is_same_v<T, float>) {
        return FloatType::getF32(context); // Map C++ float to MLIR f32
    }
    else if constexpr (std::is_same_v<T, double>) {
        return FloatType::getF64(context); // Map C++ double to MLIR f64
    }
    else if constexpr (std::is_same_v<T, APFloat>) { // Map APFloat to MLIR floattype
        if (isBF16(value))
            return FloatType::getBF16(context);
        else if (isF16(value))
            return FloatType::getF16(context);
        else if (isF32(value))
            return FloatType::getF32(context);
        else if (isF64(value))
            return FloatType::getF64(context);
        else
            assert(false && "Unsupported APFloat type for MLIR type conversion");
    }
    else
        static_assert(!std::is_same_v<T, T>, "Unsupported C++ Float type for MLIR type conversion");
}
// Get the MLIR type corresponding to a C++ type
template <typename T> Type getMLIRType(MLIRContext *context) {
    if constexpr (std::is_same_v<T, float>) {
        return FloatType::getF32(context); // Map C++ float to MLIR f32
    }
    else if constexpr (std::is_same_v<T, double>) {
        return FloatType::getF64(context); // Map C++ double to MLIR f64
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
        return IntegerType::get(context, 8); // Map C++ int8_t to MLIR i8
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        return IntegerType::get(context, 16); // Map C++ int16_t to MLIR i16
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        return IntegerType::get(context, 32); // Map C++ int32_t to MLIR i32
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        return IntegerType::get(context, 64); // Map C++ int64_t to MLIR i64
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        return IntegerType::get(context, 32, IntegerType::Unsigned); // Map uint32_t to MLIR ui32
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
        return IntegerType::get(context, 64, IntegerType::Unsigned); // Map uint64_t to MLIR ui64
    }
    else {
        static_assert(!std::is_same_v<T, T>, "Unsupported C++ type for MLIR type conversion");
    }
}

// Create a constant op from a std::vector
template <typename T>
Value createConst(const std::vector<T> &values, OpBuilder &builder, Location loc) {
    DenseIntOrFPElementsAttr denseAttr;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                  std::is_same_v<T, llvm::APFloat>) {
        auto type = RankedTensorType::get(
            values.size(), getMLIRFloatType<T>(builder.getContext(), values[0])
        );
        denseAttr = DenseFPElementsAttr::get(type, values);
    }
    else {
        auto type = RankedTensorType::get(values.size(), getMLIRType<T>(builder.getContext()));
        denseAttr = DenseIntElementsAttr::get(type, values);
    }
    return builder.create<arith::ConstantOp>(loc, denseAttr);
}

// Create a constant op from a DenseIntOrFPElementsAttr
inline Value createConst(DenseIntOrFPElementsAttr denseAttr, OpBuilder &builder, Location loc) {
    if (!denseAttr) {
        return {};
    }
    return builder.create<mlir::arith::ConstantOp>(loc, denseAttr);
}

template <typename R, class T>
arith::ConstantOp
createI8Const(R &rewriter, T &op, ArrayRef<int8_t> values, llvm::ArrayRef<int64_t> shape) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(
                         RankedTensorType::get(shape, IntegerType::get(op.getContext(), 8)), values
                     )
    );
}

template <typename R, class T>
arith::ConstantOp
createI16Const(R &rewriter, T &op, ArrayRef<int16_t> values, llvm::ArrayRef<int64_t> shape) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(
                         RankedTensorType::get(shape, IntegerType::get(op.getContext(), 16)), values
                     )
    );
}

template <typename R, class T>
arith::ConstantOp createI16Const(R &rewriter, T &op, ArrayRef<int16_t> values) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get(values.size(), IntegerType::get(op.getContext(), 16)), values
        )
    );
}

template <typename R, class T>
arith::ConstantOp createI32Const(R &rewriter, T &op, ArrayRef<int32_t> values) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get(
                static_cast<int64_t>(values.size()), IntegerType::get(op.getContext(), 32)
            ),
            values
        )
    );
}

template <typename R, class T>
arith::ConstantOp
createI32Const(R &rewriter, T &op, ArrayRef<int32_t> values, llvm::ArrayRef<int64_t> shape) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(),
        DenseIntElementsAttr::get(RankedTensorType::get(shape, rewriter.getI32Type()), values)
    );
}

template <typename R, class T>
arith::ConstantOp createIConst(R &rewriter, T &op, ArrayRef<APInt> values) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(
                         RankedTensorType::get(
                             static_cast<int64_t>(values.size()),
                             IntegerType::get(op.getContext(), values[0].getBitWidth())
                         ),
                         values
                     )
    );
}

template <typename R, class T>
arith::ConstantOp
createIConst(R &rewriter, T &op, ArrayRef<APInt> values, llvm::ArrayRef<int64_t> shape) {
    return rewriter.template create<arith::ConstantOp>(
        op.getLoc(), DenseIntElementsAttr::get(
                         RankedTensorType::get(
                             shape, IntegerType::get(op.getContext(), values[0].getBitWidth())
                         ),
                         values
                     )
    );
}

template <typename R, class T>
arith::ConstantOp createFConst(R &rewriter, T &op, ArrayRef<APFloat> values) {
    if (&values[0].getSemantics() == &llvm::APFloat::BFloat()) {
        return rewriter.template create<arith::ConstantOp>(
            op.getLoc(),
            DenseFPElementsAttr::get(
                RankedTensorType::get(
                    static_cast<int64_t>(values.size()), BFloat16Type::get(op.getContext())
                ),
                values
            )
        );
    }
    else {
        return rewriter.template create<arith::ConstantOp>(
            op.getLoc(),
            DenseFPElementsAttr::get(
                RankedTensorType::get(
                    static_cast<int64_t>(values.size()), Float32Type::get(op.getContext())
                ),
                values
            )
        );
    }
}

template <typename R, class T>
arith::ConstantOp
createFConst(R &rewriter, T &op, ArrayRef<APFloat> values, llvm::ArrayRef<int64_t> shape) {
    if (&values[0].getSemantics() == &llvm::APFloat::BFloat()) {
        return rewriter.template create<arith::ConstantOp>(
            op.getLoc(),
            DenseFPElementsAttr::get(
                RankedTensorType::get(shape, BFloat16Type::get(op.getContext())), values
            )
        );
    }
    else {
        return rewriter.template create<arith::ConstantOp>(
            op.getLoc(), DenseFPElementsAttr::get(
                             RankedTensorType::get(shape, Float32Type::get(op.getContext())), values
                         )
        );
    }
}

// return a pair with the minimum and maximum value for the given type
std::pair<int32_t, int32_t> getDTypeRange(Type type);

std::vector<int32_t>
interleave(const std::vector<int32_t> &a, const std::vector<int32_t> &b, bool broadcast_b = true);

std::vector<APInt>
interleave(const std::vector<APInt> &a, const std::vector<APInt> &b, bool broadcast_b = true);
std::vector<int64_t>
per_channel_sum(const std::vector<int8_t> &weights_values, int on, int in, int hn, int wn);

std::vector<int32_t> compute_scale(
    ::llvm::ArrayRef<int32_t> tosaMultiplier, ::llvm::ArrayRef<int8_t> tosaShift, int32_t shift
);

inline double computeScaleDouble(int32_t tosaMultiplier, int8_t tosaShift) {
    // Here we compute the scale factor as a floating point value according to
    // tosa spec https://www.mlplatform.org/tosa/tosa_spec.html#_rescale
    return static_cast<double>(tosaMultiplier) / (1ul << tosaShift);
}

std::vector<double>
computeScaleDouble(::llvm::ArrayRef<int32_t> tosaMultiplier, ::llvm::ArrayRef<int8_t> tosaShift);

// TODO: make this a non-template function
template <class T>
std::vector<int32_t> compute_bias(
    const T &biases_values, const std::vector<int64_t> &per_channel_weights_sum, int64_t input_zp
) {
    // If > 0 we will only generate partial tensors for weight to simpify debugging
    constexpr int max_weights_channel_count = 0;
    size_t num_bias_values = std::distance(biases_values.begin(), biases_values.end());
    size_t per_channel_weights_sum_size = per_channel_weights_sum.size();
    if (max_weights_channel_count && num_bias_values > max_weights_channel_count) {
        per_channel_weights_sum_size = num_bias_values = max_weights_channel_count;
    }

    if (num_bias_values != 1 && num_bias_values != per_channel_weights_sum_size) {
        llvm::errs() << "tosa bias and per_channel_weights_sum arrays have "
                        "different sizes:"
                     << num_bias_values << " and " << per_channel_weights_sum.size() << "\n";
        llvm::report_fatal_error("", true);
    }

    // Here we combine the bias with the per channel weights sum so that we
    // don't have to apply the zero point to the input as a separate operation
    std::vector<int32_t> bias_elements;
    auto bias_iter = biases_values.begin();
    for (int i = 0; i < per_channel_weights_sum_size; i++) {
        auto value = num_bias_values == 1 ? *bias_iter : *bias_iter++;
        int64_t bias = static_cast<int64_t>(value) - per_channel_weights_sum[i] * input_zp;
        bias_elements.push_back(bias);
    }
    return bias_elements;
}

// Duplicates original biases in round-robin pattern
template <typename T>
std::vector<T> duplicateBiasRoundRobin(const std::vector<T> &values, int expanded_size) {
    std::vector<T> result;
    int original_size = values.size();
    for (int i = 0; i < expanded_size; ++i) {
        result.push_back(values[i % original_size]);
    }
    return result;
}

// Bias correction: duplicated_bias[i] - per_channel_sum[i] * input_zp
template <typename T>
std::vector<int32_t> computeCorrectedBiasRoundRobin(
    const std::vector<T> &duplicated_bias, const std::vector<int64_t> &per_channel_weights_sum,
    int64_t input_zp
) {
    assert(
        duplicated_bias.size() == per_channel_weights_sum.size() &&
        "Bias and weight sum must have same number of channels"
    );

    std::vector<int32_t> corrected_bias;
    for (size_t i = 0; i < duplicated_bias.size(); ++i) {
        int64_t corrected =
            static_cast<int64_t>(duplicated_bias[i]) - per_channel_weights_sum[i] * input_zp;
        corrected_bias.push_back(static_cast<int32_t>(corrected));
    }
    return corrected_bias;
}

template <typename T>
std::vector<T>
get_weights_OIHW(const std::vector<T> &weights_values, int on, int hn, int wn, int in) {
    std::vector<T> elements;
    elements.reserve(on * in * hn * wn);

    for (int o = 0; o < on; o++) {
        for (int i = 0; i < in; i++) {
            for (int h = 0; h < hn; h++) {
                for (int w = 0; w < wn; w++) {
                    int index = o * hn * wn * in + h * wn * in + w * in + i;
                    elements.push_back(weights_values[index]);
                }
            }
        }
    }

    return elements;
}

// Convert a range of DenseElementsAttr iterators to an std::vector of the same item type
// eg std::vector<int64_t> vec = denseToVector(denseElementAttr.getValues<int64_t>());
template <typename T>
std::vector<T>
rangeValues(mlir::detail::ElementsAttrRange<mlir::DenseElementsAttr::ElementIterator<T>> range) {
    return std::vector<T>(range.begin(), range.end());
}

// Convert a DenseIntElementsAttr to an std::vector<int64_t>
inline std::vector<int64_t> attrValues(DenseIntElementsAttr denseElementAttr) {
    return rangeValues(denseElementAttr.getValues<int64_t>());
}

RankedTensorType convertTypeNCHWtoNHWC(Type origType);
RankedTensorType convertTypeNHWCtoNCHW(Type origType);
Value convertNHWCtoNCHW(Value input, Location loc, PatternRewriter &rewriter);
Value convertNCHWtoNHWC(Value input, Location loc, PatternRewriter &rewriter);

// Transpose the type of a tensor according to the specifed  permutation
RankedTensorType transposeType(Type origType, const SmallVector<int64_t> &permutation);

// Transpose the value of a tensor according to the specifed  permutation
Value transposeValue(
    Value input, const SmallVector<int64_t> &permutation, Location loc, PatternRewriter &rewriter
);

// Rescale the input value  by dividing with divFactor factor and adding bias
Value rescaleValue(Value input, int divFactor, int bias, Location loc, PatternRewriter &rewriter);

RankedTensorType convertTypeNHCtoNCH(Type origType);
Value convertNHCtoNCH(Value input, Location loc, PatternRewriter &rewriter);
Value convertNCHtoNHC(Value input, Location loc, PatternRewriter &rewriter);
Value createInitTensorNCHW(Operation *srcOp, PatternRewriter &rewriter);

void getTransposeDimSize(
    torq_hl::TransposeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t *transpose_dim,
    uint32_t *transpose_dim_size, uint32_t *rest_dim_size
);
struct Permutation : SmallVector<int64_t> {
    using SmallVector::SmallVector;

    // return the reverse of this permutation
    Permutation reverse() const;

    // Common permutations
    static const Permutation &nhwc2nchw();
    static const Permutation &hwcf2fchw();
    static const Permutation &hwc2chw();
    static const Permutation &nhc2nch();
    static const Permutation &none();
};

bool supportedByOptimizedTranspose(const ArrayRef<int64_t> &perm);

void identifyTransposeDim(
    torq_hl::TransposeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t &transposeDim,
    int32_t &leftDim, int32_t &rightDim
);
void identifyTransposeDim(
    torq_hl::TransposeReshapeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t &transposeDim,
    int32_t &leftDim, int32_t &rightDim
);

void groupContinuousDim(
    llvm::ArrayRef<int64_t> inputShape, int32_t &leftDim, int32_t &transposeDim, int32_t &rightDim,
    int32_t &leftDimSize, int32_t &transposeDimSize, int32_t &rightDimSize
);

struct ShapeStrideTuple {
    int64_t size;
    int64_t stride;
};

// Groups the given dimensions that are continuous
// Eg: Size(2, 3, 4, 5) Stride(192, 64, 5, 1) -> Size(6, 20) Stride (3, 5)
llvm::SmallVector<ShapeStrideTuple> groupAsDenseDims(
    llvm::SmallVector<int64_t> indices_shape, llvm::SmallVector<int64_t> indices_strides
);

std::vector<int8_t> genD2SWeights(size_t result_size);

// Generic struct capturing everything for a 2-input op needs
template <typename BinaryOpTy> struct BinaryOpParams {

    BinaryOpParams(BinaryOpTy op_, PatternRewriter &rewriter_)
        : op(op_), rewriter(rewriter_), ctx(rewriter_.getContext()), loc(op_.getLoc()) {}

    BinaryOpTy op;
    PatternRewriter &rewriter;
    MLIRContext *ctx;
    Location loc;

    // inputs + init/output
    Value input1, input2, init;
    MemRefType type1, type2, initType, outputType;

    // input element info
    mlir::Type inputElementType;
    uint32_t inputElementSize; // bytes per element

    // shape/offset
    llvm::ArrayRef<int64_t> input1Shape, input2Shape, outputShape;
    uint64_t inputOffset;      // raw byte offset from memref
    uint64_t inputOffsetBytes; // #elements total

    // strides
    SmallVector<int64_t> input1Strides, input2Strides, outputStrides;
};

template <typename OpTy> LogicalResult prepareParams(BinaryOpParams<OpTy> &P) {

    P.input1 = P.op.getInput1();
    P.input2 = P.op.getInput2();
    P.init = P.op.getInit();
    P.type1 = llvm::dyn_cast<MemRefType>(P.input1.getType());
    P.type2 = llvm::dyn_cast<MemRefType>(P.input2.getType());
    P.outputType = llvm::dyn_cast<MemRefType>(P.init.getType());

    assert(P.type1.getElementType() == P.type2.getElementType() && "Input types must match");
    P.inputElementType = P.type1.getElementType();
    P.inputElementSize = P.inputElementType.getIntOrFloatBitWidth() / 8;

    P.input1Shape = P.type1.getShape();
    P.input2Shape = P.type2.getShape();
    P.input1Strides = getEncodedStridesElements(P.type1);
    P.input2Strides = getEncodedStridesElements(P.type2);
    P.inputOffset = 0;
    P.inputOffsetBytes = P.inputOffset * P.inputElementSize;

    P.outputShape = P.outputType.getShape();
    P.outputStrides = getEncodedStridesElements(P.outputType);

    return success();
}

Value convertScalarToRankedTensor(Value &input, Location loc, PatternRewriter &rewriter);

bool hasEkLoweringConv(mlir::syna::torq_hl::Conv2DOp op);
bool hasEkLoweringConv(mlir::syna::torq_hl::DepthwiseConv2DOp op);
std::optional<SmallVector<int64_t>> isaBroadcastOpInterface(linalg::GenericOp genericOp);

} // namespace mlir::syna::torq
