// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ConversionUtils.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "torq-conversion-utils"

using namespace mlir;

namespace mlir::syna::torq {

SmallVector<OpFoldResult>
createVector(::llvm::ArrayRef<int64_t> values, PatternRewriter &rewriter) {
    SmallVector<OpFoldResult> result;
    for (int i = 0; i < values.size(); i++) {
        result.push_back(rewriter.getIndexAttr(values[i]));
    }
    return result;
}

Operation *getSingleUser(Value value) {
    if (!value.hasOneUse())
        return {};
    return *value.getUsers().begin();
}

Value createInitTensor(OpBuilder &builder, Location loc, RankedTensorType type, Value shapeSource) {
    SmallVector<Value> dynamicSizes;
    for (auto [i, dim] : llvm::enumerate(type.getShape())) {
        if (ShapedType::isDynamic(dim))
            dynamicSizes.push_back(tensor::DimOp::create(builder, loc, shapeSource, i));
    }
    return tensor::EmptyOp::create(builder, loc, type, dynamicSizes).getResult();
}

std::pair<int32_t, int32_t> getDTypeRange(Type type) {
    /// Check if type is float
    if (isa<FloatType>(type)) {
        constexpr int32_t minusInfinity = 0xff800000;
        constexpr int32_t plusInfinity = 0x7f800000;
        return {minusInfinity, plusInfinity};
    }

    // For integer operations we decide to wraparound as done by the LLVM CPU implementation
    // We could instead saturate by choosing the limits according to the type bitwidth
    // e.g., for int8_t: -128 to 127 but this would generate mismatches with the CPU implementation
    // in case of overflow
    return {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()};
}

bool extractBF16ConstantValue(Value v, double &out) {
    auto constOp = v.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
        return false;
    // Case 1: scalar bf16 float constant.
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        if (!floatAttr.getType().isBF16())
            return false;
        out = floatAttr.getValueAsDouble();
        return true;
    }
    // Cases 2 & 3: dense bf16 tensor constant (0-d or single-element rank-1).
    auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!denseAttr || !denseAttr.isSplat())
        return false;
    auto tensorTy = dyn_cast<RankedTensorType>(denseAttr.getType());
    if (!tensorTy || !tensorTy.getElementType().isBF16() || tensorTy.getNumElements() != 1)
        return false;
    out = denseAttr.getSplatValue<APFloat>().convertToDouble();
    return true;
}

// Reduce a traced arith.constant's attribute to a scalar attribute, unwrapping
// a qualifying dense splat.
static TypedAttr getConstScalarAttr(Attribute attr, SplatPolicy policy) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
        if (policy == SplatPolicy::SingleElement && denseAttr.getNumElements() != 1) {
            return nullptr;
        }
        if (!denseAttr.isSplat()) {
            return nullptr;
        }
        return denseAttr.getSplatValue<TypedAttr>();
    }
    return dyn_cast<TypedAttr>(attr);
}

// If `val` is a block argument of `op`'s body fed by a DPS input, return that
// input; null otherwise.
static Value getMatchingDpsInput(linalg::GenericOp op, Value val) {
    auto blockArg = dyn_cast<BlockArgument>(val);
    if (!blockArg || blockArg.getOwner() != op.getBody()) {
        return nullptr;
    }
    unsigned argIdx = blockArg.getArgNumber();
    // Block args: first N are inputs, last M are outputs (inits)
    auto inputs = op.getDpsInputs();
    if (argIdx < inputs.size()) {
        return inputs[argIdx];
    }
    return nullptr;
}

std::optional<llvm::APFloat>
traceToConstFloat(linalg::GenericOp op, Value val, SplatPolicy policy) {
    assert(
        val.getParentRegion()->isAncestor(&op.getRegion()) && "val must be usable inside op's body"
    );
    if (auto truncOp = val.getDefiningOp<arith::TruncFOp>()) {
        auto value = traceToConstFloat(op, truncOp.getIn(), policy);
        if (!value) {
            return std::nullopt;
        }
        return convertFloatToType(*value, truncOp.getOut().getType());
    }
    if (auto extOp = val.getDefiningOp<arith::ExtFOp>()) {
        auto value = traceToConstFloat(op, extOp.getIn(), policy);
        if (!value) {
            return std::nullopt;
        }
        return convertFloatToType(*value, extOp.getOut().getType());
    }
    if (Value input = getMatchingDpsInput(op, val)) {
        return traceToConstFloat(op, input, policy);
    }
    auto constOp = val.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
        return std::nullopt;
    }
    auto scalarAttr = dyn_cast_or_null<FloatAttr>(getConstScalarAttr(constOp.getValue(), policy));
    if (!scalarAttr) {
        return std::nullopt;
    }
    return scalarAttr.getValue();
}

std::optional<int64_t> traceToConstInt(linalg::GenericOp op, Value val, SplatPolicy policy) {
    assert(
        val.getParentRegion()->isAncestor(&op.getRegion()) && "val must be usable inside op's body"
    );
    if (Value input = getMatchingDpsInput(op, val)) {
        val = input;
    }
    auto constOp = val.getDefiningOp<arith::ConstantOp>();
    if (!constOp) {
        return std::nullopt;
    }
    auto scalarAttr = dyn_cast_or_null<IntegerAttr>(getConstScalarAttr(constOp.getValue(), policy));
    if (!scalarAttr) {
        return std::nullopt;
    }
    return scalarAttr.getInt();
}

std::optional<llvm::APFloat> convertFloatToType(llvm::APFloat value, Type type) {
    auto floatType = dyn_cast<FloatType>(type);
    if (!floatType) {
        return std::nullopt;
    }
    bool losesInfo = false;
    value.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    return value;
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
    return arith::ConstantOp::create(b, loc, attr).getResult();
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
    SmallVector<int64_t> newShape = applyPermutation(inputShape, permutation);
    return RankedTensorType::get(newShape, inputType.getElementType());
}

Value transposeValue(
    Value input, const SmallVector<int64_t> &permutation, Location loc, OpBuilder &builder
) {
    if (permutation.empty()) {
        return input;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(input);
    auto outputType = transposeType(input.getType(), permutation);
    auto initValue =
        tensor::EmptyOp::create(builder, loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    auto transposeOp = linalg::TransposeOp::create(builder, loc, input, initValue, permutation);
    return transposeOp.getResult()[0];
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
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 3, 1, 2};

    auto transposeOp = linalg::TransposeOp::create(rewriter, loc, input, initValue, permutation);

    return transposeOp.getResult()[0];
}

Value convertNCHWtoNHWC(Value input, Location loc, PatternRewriter &rewriter) {

    auto outputType = convertTypeNCHWtoNHWC(input.getType());

    auto initValue =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 2, 3, 1};

    auto transposeOp = linalg::TransposeOp::create(rewriter, loc, input, initValue, permutation);

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
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(), outputType.getElementType())
            .getResult();

    SmallVector<int64_t> permutation = {0, 2, 1};

    auto transposeOp = linalg::TransposeOp::create(rewriter, loc, input, initValue, permutation);

    return transposeOp.getResult()[0];
}

Value convertNCHtoNHC(Value input, Location loc, PatternRewriter &rewriter) {
    return convertNHCtoNCH(input, loc, rewriter);
}

Value createInitTensorNCHW(Operation *srcOp, PatternRewriter &rewriter) {
    auto nchwType = convertTypeNHWCtoNCHW(srcOp->getResult(0).getType());
    return tensor::EmptyOp::create(
               rewriter, srcOp->getLoc(), nchwType.getShape(), nchwType.getElementType()
    )
        .getResult();
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

const Permutation &Permutation::nchw2nhwc() {
    const static Permutation perm = {0, 2, 3, 1};
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

bool isGroupSwapTranspose(ArrayRef<int64_t> perm, int &batchDims, int &rowDims) {
    int rank = perm.size();
    int l = 0;
    while (l < rank && perm[l] == l) {
        l++;
    }
    if (l == rank) {
        // Identity permutation
        return false;
    }
    int r = perm[l] - l;
    int c = rank - l - r;
    if (r < 1 || c < 1) {
        return false;
    }
    for (int i = 0; i < c; i++) {
        if (perm[l + i] != l + r + i) {
            return false;
        }
    }
    for (int i = 0; i < r; i++) {
        if (perm[l + c + i] != l + i) {
            return false;
        }
    }
    batchDims = l;
    rowDims = r;
    return true;
}

bool useBlockTranspose(
    ArrayRef<int64_t> perm, ArrayRef<int64_t> inputShape, int elementSize, int &batchDims,
    int &rowDims
) {
    if (!isGroupSwapTranspose(perm, batchDims, rowDims) || elementSize > 2) {
        return false;
    }
    int64_t inRows = 1; // product of the first group of swapped dims
    int64_t inCols = 1; // product of the second group of swapped dims
    for (int i = batchDims; i < (int)inputShape.size(); i++) {
        (i < batchDims + rowDims ? inRows : inCols) *= inputShape[i];
    }
    return inRows >= 32 && inCols >= 32;
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

// Check if the Conv2DOp can be lowered using EK kernel
bool hasEkLoweringConv(mlir::syna::torq_hl::Conv2DOp op) {
    auto weightShape = cast<ShapedType>(op.getWeights().getType()).getShape();
    int kh = weightShape[2];
    int kw = weightShape[3];
    int stride = op.getStride()[0];
    // Common HW/kernel limitations. The border zeros are forged by the data engine and each axis
    // has its own limit on how wide a border it will forge; see the cap definitions.
    if (stride > 2 || kw > HwInfo::nss_max_kernel_width || kh > HwInfo::nss_max_kernel_height) {
        return false;
    }

    if (weightShape.size() == 5 && weightShape[4] > 4) {
        // Not supported by this EK kernel
        return false;
    }

    return true;
}

// Check if the DepthwiseConv2DOp can be lowered using EK kernel
bool hasEkLoweringConv(mlir::syna::torq_hl::DepthwiseConv2DOp op) {
    if (op.getIsDw1dStride1()) {
        return true;
    }
    auto weightShape = cast<ShapedType>(op.getWeights().getType()).getShape();

    const int kh = weightShape[2];
    const int kw = weightShape[3];

    // Common HW/kernel limitations; the two axes have different limits, see the Conv2DOp
    // overload above for what sets each.
    //
    // isValidConv is fragile: ValidPadToSamePadPass runs first, so a zero pad here means that
    // pass declined this kernel, not that the user wrote a valid conv. Widening its cap flips
    // this to false and drops large 1D kernels back onto the NDL path.
    //
    // A genuine 1D depthwise is exempt from both limits: with one spatial extent degenerate it
    // takes lowerDw1dStride1ToHw, which never reaches convAdjustPadding and so is not bound by
    // the forged halo at all.
    const auto inShape = cast<ShapedType>(op.getInput().getType()).getShape(); // NCHW
    const bool is1D =
        inShape.size() == 4 && ((inShape[2] == 1 && kh == 1) || (inShape[3] == 1 && kw == 1));
    const bool isValidConv = llvm::all_of(op.getPad(), [](int64_t p) { return p == 0; });
    if (!(isValidConv && is1D) &&
        (kw > HwInfo::nss_max_kernel_width || kh > HwInfo::nss_max_kernel_height)) {
        return false;
    }

    if (weightShape.size() == 5 && weightShape[4] != 4) {
        return false;
    }

    return true;
}

// Check if the MaxPool2dOp can be lowered using EK kernel
bool hasEkLoweringMaxPool(mlir::syna::torq_hl::MaxPool2dOp op) {
    auto kernel = op.getKernel();
    int kh = kernel[0];
    int kw = kernel[1];

    auto stride = op.getStride();
    int strideH = stride[0];
    int strideW = stride[1];

    // Stride > 2 and different strides along H and W are not supported by EK kernel
    if (strideH > 2 || strideW > 2 || strideH != strideW) {
        return false;
    }

    // Kernel size is limited by the HW like for convolutions
    if (kw > HwInfo::nss_max_kernel_width || kh > HwInfo::nss_max_kernel_height) {
        return false;
    }
    return true;
}

MaxPoolLowering getMaxPoolLowering(
    ArrayRef<int64_t> kernel, ArrayRef<int64_t> stride, ArrayRef<int64_t> pad, int64_t inputH,
    int64_t inputW
) {
    // The reshape-based 1D lowerings only handle exact non-overlapping, unpadded windows
    bool hasNoPadding = llvm::all_of(pad, [](int64_t p) { return p == 0; });
    if (kernel[1] == 1 && stride[1] == 1 && kernel[0] != 1 && stride[0] == kernel[0] &&
        hasNoPadding && inputH % kernel[0] == 0) {
        return MaxPoolLowering::Vertical1D;
    }
    if (kernel[0] == 1 && stride[0] == 1 && kernel[1] != 1 && stride[1] == kernel[1] &&
        hasNoPadding && inputW % kernel[1] == 0) {
        return MaxPoolLowering::Horizontal1D;
    }
    if (stride[0] == stride[1] && stride[0] <= 2) {
        if (kernel[0] > HwInfo::nss_max_kernel_height || kernel[1] > HwInfo::nss_max_kernel_width) {
            return MaxPoolLowering::None;
        }
        return MaxPoolLowering::Generic;
    }
    return MaxPoolLowering::None;
}

SmallVector<int64_t> getMaxPoolInputPadding(Value input, int hDim, int wDim) {
    if (auto padOp = input.getDefiningOp<tensor::PadOp>()) {
        auto low = padOp.getStaticLow();
        auto high = padOp.getStaticHigh();
        SmallVector<int64_t> pad = {low[wDim], high[wDim], low[hDim], high[hDim]};
        if (llvm::any_of(pad, [](int64_t p) { return ShapedType::isDynamic(p); })) {
            return SmallVector<int64_t>(4, -1);
        }
        return pad;
    }
    // A 1D pooling lifted to 2D can have a unit-dim expand between its padding and the pooling
    if (auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>()) {
        input = expandOp.getSrc();
    }
    if (input.getDefiningOp<tensor::PadOp>() || input.getDefiningOp<tensor::InsertSliceOp>()) {
        return SmallVector<int64_t>(4, -1);
    }
    return SmallVector<int64_t>(4, 0);
}

Value cloneAndReplaceToBody(
    OpBuilder &destBuilder, llvm::SmallVectorImpl<Operation *> &ops, IRMapping &mapping,
    Block *srcBlock, Block *destBlock
) {
    Operation *clonedOp = nullptr;
    if (srcBlock && destBlock) {
        for (auto bArg : srcBlock->getArguments()) {
            mapping.map(bArg, destBlock->getArgument(bArg.getArgNumber()));
        }
    }
    for (auto op : ops) {
        clonedOp = destBuilder.clone(*op, mapping);
    }
    assert(clonedOp->getNumResults() == 1 && "Expected single result for op");
    return clonedOp->getResult(0);
}

// Clones a contiguous segment of fusion-plan operations into the current builder context,
// clone with a zero-initialized tensor to replace the anchor(including runtime dynamic sizes),
// remaps intermediate values through IRMapping, and returns the cloned segment's final value.
// Any source ops consumed by the cloned segment are appended to `opsToDelete` for cleanup.
Value createClonedBlock(
    OpBuilder &builder, FusionPlan &fusionPlan, SmallVectorImpl<Operation *> &opsToFuse,
    int fusionStartIdx, int fusionEndIdx, llvm::SmallVectorImpl<Operation *> &opsToDelete
) {
    Operation *parent = fusionPlan.anchor;
    auto pResultTy = dyn_cast<ShapedType>(parent->getResult(0).getType());

    // For dynamic result shapes, tensor.empty requires runtime index operands for dynamic dims.
    // Walk backward from the anchor op to recover those runtime size values.
    // Current heuristic prioritizes tensor.extract_slice mixed sizes when present.
    SmallVector<Value> indexV;
    {
        SmallVector<Operation *, 8> toAnalyze{parent};
        while (toAnalyze.size() > 0) {
            auto currentOp = toAnalyze.pop_back_val();
            if (auto extractOp = dyn_cast<tensor::ExtractSliceOp>(currentOp)) {
                // Collect dynamic sizes from extract_slice; static sizes are attributes and
                // ignored.
                for (auto s : extractOp.getMixedSizes()) {
                    if (auto sVal = dyn_cast<Value>(s)) {
                        indexV.push_back(sVal);
                    }
                }
                continue;
            }
            for (auto operand : currentOp->getOperands()) {
                // If an operand is dynamically shaped, continue traversing its defining op to
                // locate index values that can materialize the destination tensor shape.
                if (auto opTy = dyn_cast<ShapedType>(operand.getType())) {
                    if (!opTy.hasStaticShape()) {
                        toAnalyze.push_back(operand.getDefiningOp());
                        break;
                    }
                }
            }
            if (!indexV.empty()) {
                // Stop once we have enough runtime sizes for the current clone scenario.
                // FIXME: gather all dynamic size values across all relevant parent paths.
                break;
            }
        }
    }

    auto zeroAttr = builder.getZeroAttr(pResultTy.getElementType());
    auto zeroTensor =
        arith::ConstantOp::create(builder, parent->getLoc(), pResultTy.getElementType(), zeroAttr)
            .getResult();
    // Pass recovered runtime indices into tensor.empty when pResultTy has dynamic dims.
    auto emptyTensor =
        tensor::EmptyOp::create(builder, parent->getLoc(), pResultTy, indexV).getResult();
    zeroTensor =
        linalg::FillOp::create(builder, parent->getLoc(), pResultTy, zeroTensor, emptyTensor)
            .getResult(0);

    llvm::SmallVector<Operation *, 8> opsToMove;
    opsToMove.reserve(fusionEndIdx - fusionStartIdx);
    llvm::DenseMap<Value, Value> valueMap;

    Value lastV;
    IRMapping mapping;
    mapping.map(fusionPlan.anchor->getResult(0), zeroTensor);
    for (int i = fusionStartIdx; i < fusionEndIdx; ++i) {
        auto op = opsToFuse[i];
        opsToMove.push_back(op);

        // We are moving linalg.generic and support ops till that point to produces the bias
        if (isa<linalg::GenericOp, linalg::BroadcastOp>(op)) {
            lastV = cloneAndReplaceToBody(builder, opsToMove, mapping);
            opsToDelete.insert(opsToDelete.end(), opsToMove.begin(), opsToMove.end());
            opsToMove.clear();
        }
    }
    return lastV;
}

DenseTypedElementsAttr returnDenseElementAttr(Value constV) {
    auto op = constV.getDefiningOp<arith::ConstantOp>();
    if (!op) {
        return nullptr;
    }
    return mlir::dyn_cast<DenseTypedElementsAttr>(op.getValue());
}

Attribute returnAttr(Value constV) {
    auto op = constV.getDefiningOp<arith::ConstantOp>();
    if (!op) {
        return nullptr;
    }
    return mlir::dyn_cast<Attribute>(op.getValue());
}

Value getDefaultBiasScale(
    Operation *srcOp, RankedTensorType opType, PatternRewriter &rewriter, int biasDim
) {
    Value biasV;
    if (opType.getElementType().isFloat()) {
        const std::vector<APFloat> bias(opType.getShape()[biasDim], APFloat(0.0f));
        biasV = createFConst(
            rewriter, *srcOp, bias, llvm::ArrayRef<int64_t>{opType.getShape()[biasDim]}
        );
    }
    else {
        const std::vector<int32_t> bias(opType.getShape()[biasDim], 0);
        const std::vector<int32_t> scale(opType.getShape()[biasDim], 1);
        biasV = createI32Const(
            rewriter, *srcOp, interleave(bias, scale), opType.getShape()[biasDim] * 2
        );
    }
    return biasV;
}

} // namespace mlir::syna::torq
