// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-conversion-utils"

namespace mlir::syna::torq {

SmallVector<OpFoldResult>
createVector(::llvm::ArrayRef<int64_t> values, PatternRewriter &rewriter) {
    SmallVector<OpFoldResult> result;
    for (int i = 0; i < values.size(); i++) {
        result.push_back(rewriter.getIndexAttr(values[i]));
    }
    return result;
}

void getDTypeRange(uint32_t element_size, int32_t *v_min, int32_t *v_max) {
    if (!v_min || !v_max) {
        llvm::report_fatal_error("Missing input v_min/v_max", true);
    }

    if (element_size == 1) {
        *v_min = std::numeric_limits<int8_t>::min();
        *v_max = std::numeric_limits<int8_t>::max();
    }
    else if (element_size == 2) {
        *v_min = std::numeric_limits<int16_t>::min();
        *v_max = std::numeric_limits<int16_t>::max();
    }
    else if (element_size == 4) {
        *v_min = std::numeric_limits<int32_t>::min();
        *v_max = std::numeric_limits<int32_t>::max();
    }
    else {
        auto msg = std::string("Unsupported element size ") + std::to_string(element_size);
        llvm::report_fatal_error(StringRef(msg), true);
    }
}

Value createZeroConstant(OpBuilder &b, Location loc, Type elemTy) {
    TypedAttr attr;
    if (isa<IndexType>(elemTy)) {
        // builder helper for index attribute
        attr = b.getIndexAttr(0);
    }
    else if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
        attr = b.getIntegerAttr(intTy, llvm::APInt::getZero(intTy.getWidth()));
    }
    else if (auto fTy = dyn_cast<FloatType>(elemTy)) {
        attr = FloatAttr::get(fTy, llvm::APFloat(fTy.getFloatSemantics()));
    }
    else {
        llvm_unreachable("unsupported element type for zero constant");
    }
    return b.create<arith::ConstantOp>(loc, attr).getResult();
}

std::vector<int32_t>
interleave(const std::vector<int32_t> &a, const std::vector<int32_t> &b, bool broadcast_b) {
    if (a.size() != b.size()) {
        if (!broadcast_b || b.size() != 1) {
            llvm::errs() << "interleaving arrays of different sizes:" << a.size() << " and "
                         << b.size() << "\n";
            llvm::report_fatal_error("", true);
        }
    }
    std::vector<int32_t> result;
    for (size_t i = 0; i < a.size(); i++) {
        result.push_back(a[i]);
        result.push_back(b[i < b.size() ? i : 0]);
    }
    return result;
}

std::vector<APInt>
interleave(const std::vector<APInt> &a, const std::vector<APInt> &b, bool broadcast_b) {
    if (a.size() != b.size()) {
        if (!broadcast_b || b.size() != 1) {
            llvm::errs() << "interleaving arrays of different sizes:" << a.size() << " and "
                         << b.size() << "\n";
            llvm::report_fatal_error("", true);
        }
    }
    std::vector<APInt> result;
    for (size_t i = 0; i < a.size(); i++) {
        result.push_back(a[i]);
        result.push_back(b[i < b.size() ? i : 0]);
    }
    return result;
}

std::vector<int64_t>
per_channel_sum(const std::vector<int8_t> &weights_values, int on, int in, int hn, int wn) {
    std::vector<int64_t> per_channel_weights_sum;

    // Let's compute the sum of the items per channel
    for (int o = 0; o < on; o++) {
        int64_t sum = 0;
        for (int i = 0; i < in; i++) {
            for (int h = 0; h < hn; h++) {
                for (int w = 0; w < wn; w++) {
                    int index = o * in * hn * wn + i * hn * wn + h * wn + w;
                    sum += weights_values[index];
                }
            }
        }
        per_channel_weights_sum.push_back(sum);
    }

    return per_channel_weights_sum;
}

std::vector<double>
computeScaleDouble(::llvm::ArrayRef<int32_t> tosaMultiplier, ::llvm::ArrayRef<int8_t> tosaShift) {
    if (tosaMultiplier.size() != tosaShift.size()) {
        llvm::errs() << "multiplier and shift arrays have different sizes:" << tosaMultiplier.size()
                     << " and " << tosaShift.size() << "\n";
        llvm::report_fatal_error("", true);
    }

    std::vector<double> scale;
    scale.reserve(tosaMultiplier.size());
    for (int i = 0; i < tosaMultiplier.size(); i++) {
        scale.push_back(computeScaleDouble(tosaMultiplier[i], tosaShift[i]));
    }
    return scale;
}

std::vector<int32_t> compute_scale(
    ::llvm::ArrayRef<int32_t> tosaMultiplier, ::llvm::ArrayRef<int8_t> tosaShift, int32_t shift
) {
    if (tosaMultiplier.size() != tosaShift.size()) {
        llvm::errs() << "multiplier and shift arrays have different sizes:" << tosaMultiplier.size()
                     << " and " << tosaShift.size() << "\n";
        llvm::report_fatal_error("", true);
    }

    // Here we compute the scale factor as a floating point value
    // then we transform it into a 32bits integer by multiplying it by 2^shift
    // as required by the hardware
    const int64_t int_multiplier = 1LL << shift;
    std::vector<int32_t> scale;
    for (int i = 0; i < tosaMultiplier.size(); i++) {
        double scale_factor = computeScaleDouble(tosaMultiplier[i], tosaShift[i]);
        double int_scale_factor = scale_factor * int_multiplier;
        if (int_scale_factor > std::numeric_limits<int32_t>::max() ||
            int_scale_factor < std::numeric_limits<int32_t>::min()) {
            llvm::errs() << "scale factor is too large: " << int_scale_factor << "\n";
            llvm::report_fatal_error("", true);
        }
        assert(static_cast<int>(int_scale_factor) <= ((1l << 31) - 1));
        scale.push_back(int_scale_factor);
    }
    return scale;
}

RankedTensorType transposeType(Type origType, const SmallVector<int64_t> &permutation) {
    auto inputType = mlir::cast<RankedTensorType>(origType);
    if (permutation.empty()) {
        return inputType;
    }
    auto inputShape = inputType.getShape();
    assert(inputType.getRank() == permutation.size());
    SmallVector<int64_t> newShape;
    for (int i = 0; i < inputType.getRank(); i++) {
        newShape.push_back(inputShape[permutation[i]]);
    }
    return RankedTensorType::get(newShape, inputType.getElementType());
}

Value transposeValue(
    Value input, const SmallVector<int64_t> &permutation, Location loc, PatternRewriter &rewriter
) {
    if (permutation.empty()) {
        return input;
    }
    auto outputType = transposeType(input.getType(), permutation);
    auto initValue =
        rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    auto transposeOp = rewriter.create<linalg::TransposeOp>(loc, input, initValue, permutation);
    return transposeOp.getResult()[0];
}

Value rescaleValue(Value input, int divFactor, int bias, Location loc, PatternRewriter &rewriter) {
    // Create linalg generic to multiply input by scale factor and add bias
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto outputType = RankedTensorType::get(inputType.getShape(), elementType);
    auto initValue =
        rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType())
            .getResult();
    size_t rank = inputType.getRank();
    SmallVector<AffineMap> maps{2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes{rank, utils::IteratorType::parallel};

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{outputType}, ValueRange{input}, ValueRange{initValue}, maps,
        iteratorTypes, /*rewriter.getStringAttr("parallel"),*/
        /* doc = */ "", /* library_call = */ "",
        [divFactor, bias](OpBuilder &b, Location loc, ValueRange args) {
            // args[0] is input value
            // Multiply by scale factor and add bias
            auto scaled = b.create<arith::DivSIOp>(
                loc, args[0],
                b.create<arith::ConstantOp>(loc, b.getIntegerAttr(args[0].getType(), divFactor))
            );
            auto biased = b.create<arith::AddIOp>(
                loc, scaled,
                b.create<arith::ConstantOp>(loc, b.getIntegerAttr(args[0].getType(), bias))
            );
            SmallVector<Value> yieldValues{biased};
            b.create<linalg::YieldOp>(loc, yieldValues);
        }
    );
    return genericOp.getResult(0);
}

RankedTensorType convertTypeNCHWtoNHWC(Type origType) {
    auto inputType = mlir::cast<RankedTensorType>(origType);
    auto inputShape = inputType.getShape();

    if (inputType.getRank() < 4) {
        return inputType;
    }

    if (inputShape.size() != 4) {
        llvm::report_fatal_error("Unsupported input shape", true);
    }

    return RankedTensorType::get(
        {inputShape[0], inputShape[2], inputShape[3], inputShape[1]}, inputType.getElementType()
    );
}

RankedTensorType convertTypeNHWCtoNCHW(Type origType) {
    auto inputType = mlir::cast<RankedTensorType>(origType);
    auto inputShape = inputType.getShape();

    // TODO: TOSA MAX_RANK is 6, we should handle this case

    if (inputType.getRank() < 4) {
        return inputType;
    }

    if (inputShape.size() != 4) {
        llvm::report_fatal_error("Unsupported input shape", true);
    }

    return RankedTensorType::get(
        {inputShape[0], inputShape[3], inputShape[1], inputShape[2]}, inputType.getElementType()
    );
}

Value convertNHWCtoNCHW(Value input, Location loc, PatternRewriter &rewriter) {

    auto outputType = convertTypeNHWCtoNCHW(input.getType());

    auto initValue =
        rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 3, 1, 2};

    auto transposeOp = rewriter.create<linalg::TransposeOp>(loc, input, initValue, permutation);

    return transposeOp.getResult()[0];
}

Value convertNCHWtoNHWC(Value input, Location loc, PatternRewriter &rewriter) {

    auto outputType = convertTypeNCHWtoNHWC(input.getType());

    auto initValue =
        rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 2, 3, 1};

    auto transposeOp = rewriter.create<linalg::TransposeOp>(loc, input, initValue, permutation);

    return transposeOp.getResult()[0];
}

RankedTensorType convertTypeNHCtoNCH(Type origType) {
    auto inputType = mlir::cast<RankedTensorType>(origType);
    auto inputShape = inputType.getShape();

    if (inputShape.size() != 3) {
        llvm::report_fatal_error("Unsupported input shape", true);
    }

    return RankedTensorType::get(
        {inputShape[0], inputShape[2], inputShape[1]}, inputType.getElementType()
    );
}

Value convertNHCtoNCH(Value input, Location loc, PatternRewriter &rewriter) {

    auto outputType = convertTypeNHCtoNCH(input.getType());

    auto initValue =
        rewriter.create<tensor::EmptyOp>(loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 2, 1};

    auto transposeOp = rewriter.create<linalg::TransposeOp>(loc, input, initValue, permutation);

    return transposeOp.getResult()[0];
}

Value convertNCHtoNHC(Value input, Location loc, PatternRewriter &rewriter) {
    return convertNHCtoNCH(input, loc, rewriter);
}

Value createInitTensorNCHW(Operation *srcOp, PatternRewriter &rewriter) {
    auto nchwType = convertTypeNHWCtoNCHW(srcOp->getResult(0).getType());
    return rewriter
        .create<tensor::EmptyOp>(srcOp->getLoc(), nchwType.getShape(), nchwType.getElementType())
        .getResult();
}

void getTransposeDimSize(
    torq_hl::TransposeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t *transposeDim,
    uint32_t *transposeDimSize, uint32_t *restDimSize
) {
    if (!transposeDimSize || !restDimSize || !transposeDim) {
        llvm::report_fatal_error("Missing input transposeDimSize/restDimSize", true);
        return;
    }

    ArrayRef<int64_t> perm = op.getPerm();

    int rank = out_shape.size();
    if (rank < 2) {
        llvm::report_fatal_error("Transpose rank must be at least 2", true);
        return;
    }

    // transpose has no batch concept, tensor encoding module handls this specific case
    // for rank==4 and perm==[0, 2, 3, 1]
    if (rank == 4 && perm == ArrayRef<int64_t>{0, 2, 3, 1}) {
        return;
    }

    // the other rank tensor hw only support transpose two block of with contiguous dimension
    // for example transpose the first dim [0, 1, 2] -> [1, 2, 0]
    // or the last dim [0, 1, 2] -> [2, 0, 1],
    // or contiguous dims [0, 1, 2, 3] -> [2, 3, 0, 1] or -> [3, 0, 1, 2],
    // for now we only support the first transpose for any rank tensor

    // regarding transpose the last dim to the first, need to group the previous dims together,
    // and compute the strides based on the group to make memory layout as expection
    // for now strides computation does not support this case, need consider from the whole

    uint32_t transpose_dim_size = 1;
    uint32_t rest_dim_size = 1;
    int32_t transpose_dim = -1;

    if (perm[rank - 1] == 0) { // first dim transpose to the last

        bool contiguous = true;
        int32_t last_perm = perm[0];
        for (int i = 1; i < rank - 1; i++) {
            if (perm[i] != last_perm + 1) {
                contiguous = false;
                break;
            }
            last_perm = perm[i];
        }

        if (contiguous) {
            for (int i = 0; i < rank - 1; i++) {
                rest_dim_size *= out_shape[i];
            }
            transpose_dim_size = out_shape[rank - 1];
            transpose_dim = 0;
        }
    }
    else if (perm[0] == rank - 1) { // last dim transpose to the first, doesn't support for now

        bool contiguous = true;
        int32_t last_perm = perm[1];
        for (int i = 2; i < rank; i++) {
            if (perm[i] != last_perm + 1) {
                contiguous = false;
                break;
            }
            last_perm = perm[i];
        }

        // compute size if everything in place
        if (contiguous) {
            for (int i = 1; i < rank; i++) {
                rest_dim_size *= out_shape[i];
            }
            transpose_dim_size = out_shape[0];
            transpose_dim = rank - 1;
        }
    }

    if (transpose_dim_size == 1 && rest_dim_size == 1) {
        return;
    }

    *transposeDimSize = align_ceil(transpose_dim_size, 32);
    *restDimSize = align_ceil(rest_dim_size, 64);
    *transposeDim = transpose_dim;
}

Permutation Permutation::reverse() const {
    Permutation reversePerm(size());
    for (int i = 0; i < size(); i++) {
        // find position of i in permutation
        auto it = std::find(begin(), end(), i);
        assert(it != end() && "Permutation is not a permutation");
        reversePerm[i] = std::distance(begin(), it);
    }
    return reversePerm;
}

const Permutation &Permutation::hwcf2fchw() {
    const static Permutation perm = {3, 2, 0, 1};
    return perm;
}

const Permutation &Permutation::nhwc2nchw() {
    const static Permutation perm = {0, 3, 1, 2};
    return perm;
}

const Permutation &Permutation::hwc2chw() {
    const static Permutation perm = {2, 0, 1};
    return perm;
}

const Permutation &Permutation::nhc2nch() {
    const static Permutation perm = {0, 2, 1};
    return perm;
}

const Permutation &Permutation::none() {
    const static Permutation perm = {};
    return perm;
}

bool supportedByOptimizedTranspose(const ArrayRef<int64_t> &perm) {
#define TORQ_ALWAYS_USE_EK_TRANSPOSE true
#if TORQ_ALWAYS_USE_EK_TRANSPOSE
    // Always use EK (currently unoptimized) transpose for all cases
    // This can bring a 2x performance penalty in some cases but avoids all issues
    // related to the need to align input and output tensors
    return false;
#endif
    return perm.size() < 6;
}

void identifyTransposeDim(
    torq_hl::TransposeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t &transposeDim,
    int32_t &leftDim, int32_t &rightDim
) {
    ArrayRef<int64_t> perm = op.getPerm();

    int rank = out_shape.size();
    if (rank < 2) {
        llvm::report_fatal_error("Transpose rank must be at least 2", true);
        return;
    }

    int idx = 0;
    int i = 0;

    // Splitting the total dimensions into 3 parts
    // * transpose_dim is either a single dimension or a set of sequential
    // dimension that is going to be moved to last
    // * left_dim is that which is left of the tranpose dim and is sequential
    // * right_dim is that which is the right of the transpose dim and is
    // sequential
    while (i < perm.size()) {
        if (perm[i] != idx) {
            if (perm[i] == idx + 1) { // perm:(1, 2, 0), (0, 2, 1), (1, 2, 3, 0),
                // (0, 2, 3, 1), (0, 1, 3, 2) -> the dimension after batch is send
                // to last. Extended to more dim 0, 1, 3, 4, 2
                transposeDim = idx + 1;
            }
            else { // perm: (2, 0, 1), (3, 0, 1, 2), (0, 3, 1, 2), (0, 1, 3, 2)
                // -> the dimension is send to first after batch. Extended to more
                // dim 0, 1, 4, 2, 3
                transposeDim = perm[i];
            }
            leftDim = idx - 1;
            rightDim = idx;
            break;
        }
        i++;
        idx++;
    }
}

void identifyTransposeDim(
    torq_hl::TransposeReshapeOp op, llvm::ArrayRef<int64_t> out_shape, int32_t &transposeDim,
    int32_t &leftDim, int32_t &rightDim
) {
    ArrayRef<int64_t> perm = op.getPerm();

    int rank = out_shape.size();
    if (rank < 2) {
        llvm::report_fatal_error("Transpose rank must be at least 2", true);
        return;
    }

    int idx = 0;
    int i = 0;

    // Splitting the total dimensions into 3 parts
    // * transpose_dim is either a single dimension or a set of sequential
    // dimension that is going to be moved to last
    // * left_dim is that which is left of the tranpose dim and is sequential
    // * right_dim is that which is the right of the transpose dim and is
    // sequential
    while (i < perm.size()) {
        if (perm[i] != idx) {
            if (perm[i] == idx + 1) { // perm:(1, 2, 0), (0, 2, 1), (1, 2, 3, 0),
                // (0, 2, 3, 1), (0, 1, 3, 2) -> the dimension after batch is send
                // to last. Extended to more dim 0, 1, 3, 4, 2
                transposeDim = idx + 1;
            }
            else { // perm: (2, 0, 1), (3, 0, 1, 2), (0, 3, 1, 2), (0, 1, 3, 2)
                // -> the dimension is send to first after batch. Extended to more
                // dim 0, 1, 4, 2, 3
                transposeDim = perm[i];
            }
            leftDim = idx - 1;
            rightDim = idx;
            break;
        }
        i++;
        idx++;
    }
}

void groupContinuousDim(
    llvm::ArrayRef<int64_t> inputShape, int32_t &leftDim, int32_t &transposeDim, int32_t &rightDim,
    int32_t &leftDimSize, int32_t &transposeDimSize, int32_t &rightDimSize
) {

    leftDimSize = 1;
    for (int i = 0; i < leftDim + 1; ++i) {
        leftDimSize *= inputShape[i];
    }
    transposeDimSize = 1;
    for (int i = transposeDim; i < inputShape.size(); ++i) {
        transposeDimSize *= inputShape[i];
    }
    rightDimSize = 1;
    for (int i = rightDim; i < transposeDim; ++i) {
        rightDimSize *= inputShape[i];
    }
}

// Groups the given dimensions that are continuous
// Eg: Size(2, 3, 4, 5) Stride(192, 64, 5, 1) -> Size(6, 20) Stride (3, 5)
llvm::SmallVector<ShapeStrideTuple>
groupAsDenseDims(llvm::SmallVector<int64_t> dataShape, llvm::SmallVector<int64_t> dataStrides) {
    int32_t shapeSize = dataShape.size();
    uint32_t total_strides = 1;
    uint32_t total_idxs = dataShape.back();

    llvm::SmallVector<ShapeStrideTuple> max_entry_splits = {{total_idxs, total_strides}};
    for (int i = shapeSize - 2; i >= 0; --i) {
        total_strides = dataStrides[i];
        if (total_strides != total_idxs) { // If the stride and size doesn't match then its a
            // not a continuous array
            max_entry_splits.push_back({1, total_strides});
        }
        max_entry_splits.back().size *= dataShape[i];
        total_idxs = dataShape[i] * total_strides;
    }
    return max_entry_splits;
}

// Function to create alternative 0's and 1's for each of the input of D2S
//  8bit output:
//  0, 1, 0, 1
//  1, 0, 1, 0
//  16bit output
//  0,0,1,1,0,0,1,1
//  1,1,0,0,1,1,0,0
std::vector<int8_t> genD2SWeights(size_t result_size) {
    const int num_inputs = 2;
    std::vector<int8_t> result;
    result.reserve(result_size * num_inputs);
    for (size_t n = 0; n < num_inputs; ++n) {
        for (size_t i = 0; i < result_size; i++) {
            result.push_back(static_cast<int8_t>((i + n + 1) % 2));
        }
    }
    return result;
}

Value convertScalarToRankedTensor(Value &input, Location loc, PatternRewriter &rewriter) {
    auto constOp = input.getDefiningOp<arith::ConstantOp>();
    auto value = constOp.getValue();
    Value input_new;
    mlir::DenseElementsAttr denseAttr;
    auto elemType = constOp.getType();
    auto tensorType = mlir::RankedTensorType::get({1}, elemType);

    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
        denseAttr = mlir::DenseElementsAttr::get(tensorType, {intAttr.getValue()});
    }
    else if (auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(value)) {
        denseAttr = mlir::DenseElementsAttr::get(tensorType, {floatAttr.getValue()});
    }
    else {
        assert(false && "Cannot convert Scalar to RankedTensor");
    }
    input_new = rewriter.create<arith::ConstantOp>(loc, tensorType, denseAttr);
    return input_new;
}

// Check if the Conv2DOp can be lowered using EK kernel
bool hasEkLowering(mlir::syna::torq_hl::Conv2DOp op) {
    int32_t pad_left = op.getPad()[0];
    int32_t pad_right = op.getPad()[1];
    if (pad_left != 1 || pad_right != 1) {
        // Not supported by HW
        return false;
    }

    int stride = op.getStride()[0]; // FIXME: consider all stride values
    auto weightShape = cast<ShapedType>(op.getWeights().getType()).getShape();
    if (stride != 1 || weightShape[2] != 3 || weightShape[3] != 3) {
        // Not supported by this EK kernel
        return false;
    }
    return true;
}

// FIXME: Copied from
// https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Linalg/IR/LinalgInterfaces.cpp#L141.
// Remove after upgrading to latest MLIR.
std::optional<SmallVector<int64_t>> isaBroadcastOpInterface(linalg::GenericOp genericOp) {
    // Is single Input and Output
    if (genericOp.getInputs().size() != 1 || genericOp.getOutputs().size() != 1)
        return std::nullopt;

    // Is all parallel loops
    for (auto it : genericOp.getIteratorTypesArray()) {
        if (mlir::linalg::isParallelIterator(it) == false)
            return std::nullopt;
    }

    // Is single region, block and yield
    if (!genericOp.getRegion().hasOneBlock())
        return std::nullopt;

    auto &block = genericOp.getRegion().front();
    if (block.getNumArguments() != 2)
        return std::nullopt;

    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1 || yield.getValues()[0] != block.getArgument(0))
        return std::nullopt;

    auto srcTy = genericOp.getDpsInputOperand(0)->get().getType();
    auto dstTy = genericOp.getDpsInitOperand(0)->get().getType();
    if (!isa<MemRefType, RankedTensorType>(srcTy) || !isa<MemRefType, RankedTensorType>(dstTy))
        return std::nullopt;

    // Check output is identity map. Broadcast could additionally be
    // employing permutation of indices and that would be expressible
    // in linalg.generic but is not expressible for named broadcast genericOp.
    auto dstMap = genericOp.getIndexingMapsArray()[1];
    if (!dstMap.isIdentity())
        return std::nullopt;

    SmallVector<int64_t> position;
    auto srcMap = genericOp.getIndexingMapsArray()[0];

    if (srcMap.getResults().size() >= dstMap.getResults().size())
        return std::nullopt;

    // Check input map is monotonically increasing DimIds.
    for (unsigned i = 0; i < srcMap.getNumResults(); ++i) {
        auto expr = llvm::dyn_cast<AffineDimExpr>(srcMap.getResults()[i]);
        if (!expr)
            return std::nullopt;
        int64_t pos = expr.getPosition();
        if (i > 0 && pos <= position[i - 1])
            return std::nullopt;
        position.push_back(expr.getPosition());
    }

    SmallVector<int64_t> broadcastedDims;
    auto numDims = srcMap.getNumDims();
    // This is quadratic but number of items is generally small.
    for (auto dim : llvm::seq<int64_t>(0, numDims)) {
        if (!llvm::is_contained(position, dim))
            broadcastedDims.push_back(dim);
    }
    return broadcastedDims;
}

} // namespace mlir::syna::torq
