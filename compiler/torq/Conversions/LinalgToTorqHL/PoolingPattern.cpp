// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "linalg-torq-pooling-pattern"

namespace mlir::syna::torq {

template <bool IsNCHW>
static Value applyInputTranspose(
    Value input, RankedTensorType &resultType, Location loc, PatternRewriter &rewriter
) {
    if constexpr (IsNCHW) {
        // NCHW is already in correct format, no transpose needed
        return input;
    }
    else {
        // NHWC needs transpose to NCHW
        if (resultType.getRank() == 4) {
            auto dataPerm = Permutation::nhwc2nchw();
            resultType = transposeType(resultType, dataPerm);
            return transposeValue(input, dataPerm, loc, rewriter);
        }
        return input;
    }
}

template <bool IsNCHW>
static Value applyOutputTranspose(
    Value output, RankedTensorType resultType, Location loc, PatternRewriter &rewriter
) {
    if constexpr (IsNCHW) {
        return output;
    }
    else {
        if (resultType.getRank() == 4) {
            auto dataPerm = Permutation::nhwc2nchw();
            return transposeValue(output, dataPerm.reverse(), loc, rewriter);
        }
        return output;
    }
}

/// Look for a downstream linalg.generic that performs an elementwise mulf or divf
/// by a constant. If found, return the constant scale value and the generic op.
/// For divf, the scale is the reciprocal of the divisor.
// The scale is a double so that it is rounded only once, to the weight type.
static std::pair<double, linalg::GenericOp> tryExtractFusedScale(Value output) {
    auto genericOp = getSingleUser<linalg::GenericOp>(output);
    if (!genericOp || genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1 ||
        genericOp.getNumReductionLoops() != 0)
        return {1.0, nullptr};

    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return {1.0, nullptr};

    auto mulOp = dyn_cast<arith::MulFOp>(yieldOp.getOperand(0).getDefiningOp());
    auto divOp = dyn_cast<arith::DivFOp>(yieldOp.getOperand(0).getDefiningOp());
    if (!mulOp && !divOp)
        return {1.0, nullptr};

    Value lhs = mulOp ? mulOp.getLhs() : divOp.getLhs();
    Value rhs = mulOp ? mulOp.getRhs() : divOp.getRhs();

    bool lhsIsBlockArg =
        isa<BlockArgument>(lhs) && cast<BlockArgument>(lhs).getOwner() == genericOp.getBody();
    bool rhsIsBlockArg =
        isa<BlockArgument>(rhs) && cast<BlockArgument>(rhs).getOwner() == genericOp.getBody();

    arith::ConstantOp constOp = nullptr;
    if (lhsIsBlockArg && !rhsIsBlockArg)
        constOp = rhs.getDefiningOp<arith::ConstantOp>();
    else if (rhsIsBlockArg && !lhsIsBlockArg)
        constOp = lhs.getDefiningOp<arith::ConstantOp>();

    if (!constOp)
        return {1.0, nullptr};

    auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue());
    if (!floatAttr)
        return {1.0, nullptr};

    double value = floatAttr.getValueAsDouble();
    if (divOp) {
        if (value == 0.0)
            return {1.0, nullptr};
        value = 1.0 / value;
    }
    return {value, genericOp};
}

// The integer clamp (minsi and/or maxsi of the data with constants) that is the only user of the
// result of a max pooling, if any: e.g. the clamp of a quantize folded in an int8 max pooling (see
// FoldMaxPoolQuantization), whose maxsi with -128 is removed by the canonicalization. The max
// pooling applies it to its output.
static linalg::GenericOp getMaxPoolOutputClamp(Value output, int32_t &clampMin, int32_t &clampMax) {
    if (!output.hasOneUse() || !isa<IntegerType>(getElementTypeOrSelf(output.getType()))) {
        return {};
    }
    auto clampOp = dyn_cast<linalg::GenericOp>(*output.getUsers().begin());
    if (!clampOp || !linalg::isElementwise(clampOp) || clampOp.getNumDpsInputs() != 1 ||
        clampOp.getResult(0).getType() != output.getType()) {
        return {};
    }
    clampMin = std::numeric_limits<int32_t>::min();
    clampMax = std::numeric_limits<int32_t>::max();
    Value v = clampOp.getBody()->getTerminator()->getOperand(0);
    while (!isa<BlockArgument>(v)) {
        Operation *op = v.getDefiningOp();
        if (!isa<arith::MinSIOp, arith::MaxSIOp>(op)) {
            return {};
        }
        std::optional<int64_t> bound = getConstantIntValue(op->getOperand(1));
        if (!bound) {
            return {};
        }
        if (isa<arith::MinSIOp>(op)) {
            clampMax = std::min<int64_t>(clampMax, *bound);
        }
        else {
            clampMin = std::max<int64_t>(clampMin, *bound);
        }
        v = op->getOperand(0);
    }
    if (v != clampOp.getBody()->getArgument(0)) {
        return {};
    }
    return clampOp;
}

template <typename PoolingOpType, bool IsNCHW>
struct PoolingMaxOpConversionBase : public OpRewritePattern<PoolingOpType> {
  private:
    const bool _markFuseGroups;

  public:
    using OpRewritePattern<PoolingOpType>::OpRewritePattern;
    PoolingMaxOpConversionBase(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<PoolingOpType>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(PoolingOpType srcOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(srcOp)) {
            return rewriter.notifyMatchFailure(srcOp, "Already marked");
        }

        if (srcOp.getInputs().size() != 2 || srcOp.getResults().size() != 1) {
            return rewriter.notifyMatchFailure(srcOp, "Expects 2 inputs and 1 output");
        }
        auto loc = srcOp.getLoc();
        Value input = srcOp.getInputs()[0];
        Value output = srcOp.getResults()[0];

        auto attrStrides = attrValuesAsVec(srcOp.getStrides());
        if (attrStrides.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two strides for PoolingMaxOp"
            );
        }

        const std::vector<int32_t> bias = {0};
        const std::vector<int32_t> scale = {1};

        auto kernel = srcOp.getInputs()[1];
        auto kernelsShape = mlir::cast<RankedTensorType>(kernel.getType()).getShape();
        if (kernelsShape.size() != 2) {
            return rewriter.notifyMatchFailure(
                srcOp, "Expected exactly two kernel sizes for PoolingMaxOp"
            );
        }

        // Leave the maxpools the NSS can't run in linalg so that they run on CSS or Host.
        // The padding isn't folded yet (it must be decided the same way when marking fuse groups):
        // an unknown padding is refused by the lowerings that depend on it.
        auto inputShape = mlir::cast<RankedTensorType>(input.getType()).getShape();
        int64_t inputH = inputShape[IsNCHW ? 2 : 1];
        int64_t inputW = inputShape[IsNCHW ? 3 : 2];
        SmallVector<int64_t> pad = getMaxPoolInputPadding(input, IsNCHW ? 2 : 1, IsNCHW ? 3 : 2);
        if (getMaxPoolLowering(kernelsShape, attrStrides, pad, inputH, inputW) ==
            MaxPoolLowering::None) {
            return rewriter.notifyMatchFailure(srcOp, "Maxpool not supported on the NSS");
        }

        if (_markFuseGroups) {
            // A max pooling whose result only feeds another max pooling (e.g. the stages of a max
            // pooling split in a chain) is marked in the group of that consumer, so that
            // tile-and-fuse fuses both in the same tiles instead of writing the intermediate
            // result to XRAM
            if (output.hasOneUse() && isa<PoolingOpType>(*output.getUsers().begin())) {
                return rewriter.notifyMatchFailure(srcOp, "Marked with its consumer max pooling");
            }
            int32_t clampMin, clampMax;
            if (auto clampOp = getMaxPoolOutputClamp(output, clampMin, clampMax)) {
                output = clampOp.getResult(0);
            }
            Value source = peelPadAndReshapeSource(input);
            while (auto producer = source.getDefiningOp<PoolingOpType>()) {
                if (!producer->getResult(0).hasOneUse()) {
                    break;
                }
                source = peelPadAndReshapeSource(producer.getInputs()[0]);
            }
            if (!markFuseGroupBackward(
                    output, {source, kernel}, rewriter,
                    srcOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        PaddingInfo padInfo = foldBackwardPadding(input, rewriter, IsNCHW, srcOp.getResult(0));

        auto srcResultType = mlir::cast<RankedTensorType>(srcOp.getResult(0).getType());

        input = applyInputTranspose<IsNCHW>(input, srcResultType, loc, rewriter);

        auto elementType = srcResultType.getElementType();
        int32_t outputMin, outputMax;
        Value weightConst;

        if (elementType.isInteger(8)) {
            outputMin = -128;
            outputMax = 127;
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }
        else if (elementType.isInteger(16)) {
            outputMin = -32768;
            outputMax = 32767;
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }
        else if (elementType.isBF16()) {
            outputMin = 0xff800000; // -inf in bf16 (as int32 bits)
            outputMax = 0x7f800000; // +inf in bf16 (as int32 bits)
            // Create BF16 weight with value 1.0
            auto bf16Type = rewriter.getBF16Type();
            auto weightType = RankedTensorType::get({1, 1, 1, 1}, bf16Type);
            auto bf16One = rewriter.getFloatAttr(bf16Type, 1.0);
            auto weightAttr = DenseElementsAttr::get(weightType, bf16One);
            weightConst = arith::ConstantOp::create(rewriter, loc, weightType, weightAttr);
        }
        else {
            outputMin = std::numeric_limits<int32_t>::min();
            outputMax = std::numeric_limits<int32_t>::max();
            const std::vector<int8_t> weight = {1};
            weightConst =
                createI8Const(rewriter, srcOp, weight, llvm::ArrayRef<int64_t>{1, 1, 1, 1});
        }

        int32_t clampMin, clampMax;
        linalg::GenericOp clampOp = getMaxPoolOutputClamp(output, clampMin, clampMax);
        if (clampOp) {
            outputMin = std::max(outputMin, clampMin);
            outputMax = std::min(outputMax, clampMax);
        }

        auto maxpoolOp = torq_hl::MaxPool2dOp::create(
            rewriter, loc, srcResultType, createInitTensor(srcOp, rewriter, srcResultType),
            padInfo.padValue, outputMin, outputMax, attrStrides, padInfo.lrtbPad, kernelsShape,
            weightConst, createI32Const(rewriter, srcOp, interleave(bias, scale)), input,
            /*segment_output=*/false
        );

        Value result =
            applyOutputTranspose<IsNCHW>(maxpoolOp.getOutput(), srcResultType, loc, rewriter);

        if (clampOp) {
            rewriter.replaceOp(clampOp, result);
            rewriter.eraseOp(srcOp);
        }
        else {
            rewriter.replaceOp(output.getDefiningOp(), result);
        }

        return success();
    }
};

using PoolingNhwcMaxOpConversion = PoolingMaxOpConversionBase<linalg::PoolingNhwcMaxOp, false>;
using PoolingNchwMaxOpConversion = PoolingMaxOpConversionBase<linalg::PoolingNchwMaxOp, true>;

namespace {

// Zero-based shape indices; stride/dilation arrays use the kernel axis order.
enum NhwcDim : unsigned { NHWC_BATCH = 0, NHWC_HEIGHT = 1, NHWC_WIDTH = 2, NHWC_CHANNEL = 3 };

enum NchwDim : unsigned { NCHW_BATCH = 0, NCHW_CHANNEL = 1, NCHW_HEIGHT = 2, NCHW_WIDTH = 3 };

enum KernelDim : unsigned { KERNEL_HEIGHT = 0, KERNEL_WIDTH = 1 };

// Compiler-work budgets, not hardware limits. Only position-dependent scales
// need a scan of H*W pixels; constant scales bypass that scan and its limit.
constexpr int64_t kMaxSlidingPoolScalePixels = 65536;
// Bound scale metadata and the number of tasks expanded by EasyKernel.
constexpr size_t kMaxSlidingPoolScaleRegions = 64;

struct PoolScale {
    int32_t multiplier;
    int32_t shift;

    bool operator==(const PoolScale &other) const {
        return multiplier == other.multiplier && shift == other.shift;
    }
};

struct PoolScaleRegion {
    int64_t y, x, height, width;
    PoolScale scale;
};

struct PoolRescaleInfo {
    tosa::ApplyScaleOp apply;
    int32_t zeroPoint, min, max;
};

// Signed index intervals only. Arithmetic that could wrap is left unsupported.
using PoolIndexBounds = std::pair<int64_t, int64_t>;

static std::optional<PoolIndexBounds>
combinePoolIndexBounds(PoolIndexBounds lhs, PoolIndexBounds rhs, bool subtract) {
    bool lowerOverflow = false, upperOverflow = false;
    APInt lower = subtract ? APInt(64, lhs.first).ssub_ov(APInt(64, rhs.second), lowerOverflow)
                           : APInt(64, lhs.first).sadd_ov(APInt(64, rhs.first), lowerOverflow);
    APInt upper = subtract ? APInt(64, lhs.second).ssub_ov(APInt(64, rhs.first), upperOverflow)
                           : APInt(64, lhs.second).sadd_ov(APInt(64, rhs.second), upperOverflow);
    if (lowerOverflow || upperOverflow)
        return std::nullopt;
    return PoolIndexBounds{lower.getSExtValue(), upper.getSExtValue()};
}

static PoolIndexBounds
combineMinMaxPoolIndexBounds(PoolIndexBounds lhs, PoolIndexBounds rhs, bool isMin) {
    if (isMin)
        return {std::min(lhs.first, rhs.first), std::min(lhs.second, rhs.second)};
    return {std::max(lhs.first, rhs.first), std::max(lhs.second, rhs.second)};
}

// Evaluate the original rescale exactly; padded windows may use different divisors.
class PoolScaleEvaluator {
  public:
    PoolScaleEvaluator(tosa::ApplyScaleOp apply, bool nchw)
        : apply(apply), hDim(NHWC_HEIGHT), wDim(NHWC_WIDTH) {
        if (nchw) {
            hDim = NCHW_HEIGHT;
            wDim = NCHW_WIDTH;
        }
    }

    std::optional<PoolScale> scaleAt(int64_t y, int64_t x) {
        h = y;
        w = x;
        cache.clear(); // Index-derived values change at every output position.
        boundsCache.clear();
        auto multiplier = evaluate(apply.getMultiplier());
        auto shift = evaluate(apply.getShift());
        if (!multiplier || !shift || multiplier->isNegative() || shift->getSExtValue() < 0 ||
            shift->getSExtValue() > 62)
            return std::nullopt;
        return PoolScale{
            static_cast<int32_t>(multiplier->getSExtValue()),
            static_cast<int32_t>(shift->getSExtValue())
        };
    }

  private:
    std::optional<APInt> evaluate(Value value);
    std::optional<APInt> evaluateAffine(Operation *op, unsigned bits);
    std::optional<APInt> evaluateUnary(Operation *op, unsigned bits);
    std::optional<APInt> evaluateBinary(Operation *op);
    std::optional<PoolIndexBounds> evaluateIndexBounds(Value value);
    std::optional<PoolIndexBounds>
    evaluateAffineBounds(AffineExpr expr, unsigned numDims, ArrayRef<PoolIndexBounds> operands);

    tosa::ApplyScaleOp apply;
    unsigned hDim, wDim;
    int64_t h = 0, w = 0;
    DenseMap<Value, APInt> cache;
    DenseMap<Value, PoolIndexBounds> boundsCache;
};

std::optional<PoolIndexBounds> PoolScaleEvaluator::evaluateAffineBounds(
    AffineExpr expr, unsigned numDims, ArrayRef<PoolIndexBounds> operands
) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr))
        return PoolIndexBounds{constant.getValue(), constant.getValue()};
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
        return operands[dim.getPosition()];
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr))
        return operands[numDims + symbol.getPosition()];
    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary)
        return std::nullopt;
    auto lhs = evaluateAffineBounds(binary.getLHS(), numDims, operands);
    auto rhs = evaluateAffineBounds(binary.getRHS(), numDims, operands);
    if (!lhs || !rhs)
        return std::nullopt;
    if (expr.getKind() == AffineExprKind::Add)
        return combinePoolIndexBounds(*lhs, *rhs, /*subtract=*/false);
    // Affine subtraction is represented as multiplication by -1 plus addition.
    if (expr.getKind() == AffineExprKind::Mul && rhs->first == rhs->second) {
        bool lowerOverflow = false, upperOverflow = false;
        APInt lower = APInt(64, lhs->first).smul_ov(APInt(64, rhs->first), lowerOverflow);
        APInt upper = APInt(64, lhs->second).smul_ov(APInt(64, rhs->first), upperOverflow);
        if (lowerOverflow || upperOverflow)
            return std::nullopt;
        int64_t a = lower.getSExtValue(), b = upper.getSExtValue();
        return PoolIndexBounds{std::min(a, b), std::max(a, b)};
    }
    return std::nullopt;
}

std::optional<PoolIndexBounds> PoolScaleEvaluator::evaluateIndexBounds(Value value) {
    if (!value.getType().isIndex())
        return std::nullopt;
    if (auto it = boundsCache.find(value); it != boundsCache.end())
        return it->second;
    if (auto exact = evaluate(value)) {
        int64_t constant = exact->getSExtValue();
        return PoolIndexBounds{constant, constant};
    }

    std::optional<PoolIndexBounds> result;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
        auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
        if (!loop || value != loop.getInductionVar())
            return std::nullopt;
        auto lower = getConstIntValue(loop.getLowerBound());
        auto upper = getConstIntValue(loop.getUpperBound());
        auto step = getConstIntValue(loop.getStep());
        if (!lower || !upper || !step || *step <= 0 || *lower >= *upper)
            return std::nullopt;
        // Bound the values reached by the loop. The upper bound is exclusive,
        // and values between iterations are not induction-variable values.
        APInt wideLower(128, *lower, /*isSigned=*/true);
        APInt wideUpper(128, *upper, /*isSigned=*/true);
        APInt wideStep(128, *step, /*isSigned=*/true);
        APInt distance = wideUpper - wideLower;
        APInt tripCount = distance.udiv(wideStep);
        if (!distance.urem(wideStep).isZero())
            tripCount += 1;
        APInt last = wideLower + (tripCount - 1) * wideStep;
        if (!last.isSignedIntN(64))
            return std::nullopt;
        result = PoolIndexBounds{*lower, last.getSExtValue()};
    }
    else if (auto op = value.getDefiningOp()) {
        if (isa<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>(op)) {
            AffineMap map;
            if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
                map = apply.getAffineMap();
            else if (auto min = dyn_cast<affine::AffineMinOp>(op))
                map = min.getAffineMap();
            else
                map = cast<affine::AffineMaxOp>(op).getAffineMap();
            SmallVector<PoolIndexBounds> operands;
            for (Value operand : op->getOperands()) {
                auto bounds = evaluateIndexBounds(operand);
                if (!bounds)
                    return std::nullopt;
                operands.push_back(*bounds);
            }
            for (AffineExpr expr : map.getResults()) {
                auto bounds = evaluateAffineBounds(expr, map.getNumDims(), operands);
                if (!bounds)
                    return std::nullopt;
                if (!result)
                    result = bounds;
                else
                    result = combineMinMaxPoolIndexBounds(
                        *result, *bounds, isa<affine::AffineMinOp>(op)
                    );
            }
        }
        else if (isa<arith::AddIOp, arith::SubIOp, arith::MinSIOp, arith::MaxSIOp>(op)) {
            auto lhs = evaluateIndexBounds(op->getOperand(0));
            auto rhs = evaluateIndexBounds(op->getOperand(1));
            if (!lhs || !rhs)
                return std::nullopt;
            if (isa<arith::AddIOp, arith::SubIOp>(op))
                result = combinePoolIndexBounds(*lhs, *rhs, isa<arith::SubIOp>(op));
            else
                result = combineMinMaxPoolIndexBounds(*lhs, *rhs, isa<arith::MinSIOp>(op));
        }
    }
    if (result)
        boundsCache.try_emplace(value, *result);
    return result;
}

std::optional<APInt> PoolScaleEvaluator::evaluate(Value value) {
    if (auto it = cache.find(value); it != cache.end())
        return it->second;
    auto op = value.getDefiningOp();
    if (!op || !value.getType().isIntOrIndex())
        return std::nullopt;
    unsigned bits = value.getType().isIndex() ? 64 : value.getType().getIntOrFloatBitWidth();
    std::optional<APInt> result;
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        if (auto attr = dyn_cast<IntegerAttr>(constant.getValue()))
            result = attr.getValue().sextOrTrunc(bits);
    }
    else if (auto index = dyn_cast<linalg::IndexOp>(op)) {
        if (index.getDim() == hDim || index.getDim() == wDim)
            result = APInt(bits, index.getDim() == hDim ? h : w);
    }
    else if (isa<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp>(op))
        result = evaluateAffine(op, bits);
    else if (op->getNumOperands() == 1)
        result = evaluateUnary(op, bits);
    else if (op->getNumOperands() == 2)
        result = evaluateBinary(op);
    if (result)
        cache.try_emplace(value, *result);
    return result;
}

std::optional<APInt> PoolScaleEvaluator::evaluateAffine(Operation *op, unsigned bits) {
    AffineMap map;
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op))
        map = apply.getAffineMap();
    else if (auto min = dyn_cast<affine::AffineMinOp>(op))
        map = min.getAffineMap();
    else
        map = cast<affine::AffineMaxOp>(op).getAffineMap();
    SmallVector<Attribute> operands;
    for (Value operand : op->getOperands()) {
        auto evaluated = evaluate(operand);
        if (!evaluated)
            return std::nullopt;
        operands.push_back(IntegerAttr::get(operand.getType(), *evaluated));
    }
    // Preserve global tile offsets and signed floor/ceil division semantics.
    SmallVector<Attribute> folded;
    bool hasPoison = false;
    if (failed(map.constantFold(operands, folded, &hasPoison)) || hasPoison || folded.empty())
        return std::nullopt;
    std::optional<APInt> result;
    const bool isMin = isa<affine::AffineMinOp>(op);
    for (Attribute attr : folded) {
        auto candidate = cast<IntegerAttr>(attr).getValue().sextOrTrunc(bits);
        if (!result || (isMin ? candidate.slt(*result) : candidate.sgt(*result)))
            result = candidate;
    }
    return result;
}

std::optional<APInt> PoolScaleEvaluator::evaluateUnary(Operation *op, unsigned bits) {
    auto operand = evaluate(op->getOperand(0));
    if (!operand)
        return std::nullopt;
    if (isa<arith::IndexCastOp, arith::ExtSIOp>(op))
        return operand->sextOrTrunc(bits);
    if (isa<arith::ExtUIOp, arith::TruncIOp>(op))
        return operand->zextOrTrunc(bits);
    if (isa<math::CountLeadingZerosOp>(op))
        return APInt(bits, operand->countLeadingZeros());
    return std::nullopt;
}

std::optional<APInt> PoolScaleEvaluator::evaluateBinary(Operation *op) {
    auto lhs = evaluate(op->getOperand(0));
    auto rhs = evaluate(op->getOperand(1));
    if (!lhs || !rhs) {
        // Prove the clamp constant over every enclosing-loop iteration, not
        // merely at a sampled IV. This removes interior-tile row dependence.
        if (isa<arith::MinSIOp, arith::MaxSIOp>(op)) {
            auto a = evaluateIndexBounds(op->getOperand(0));
            auto b = evaluateIndexBounds(op->getOperand(1));
            if (a && b) {
                auto bounds = combineMinMaxPoolIndexBounds(*a, *b, isa<arith::MinSIOp>(op));
                if (bounds.first == bounds.second)
                    return APInt(64, bounds.first);
            }
        }
        return std::nullopt;
    }
    if (lhs->getBitWidth() != rhs->getBitWidth())
        return std::nullopt;
    if (isa<arith::AddIOp>(op))
        return *lhs + *rhs;
    if (isa<arith::SubIOp>(op))
        return *lhs - *rhs;
    if (isa<arith::MulIOp>(op))
        return *lhs * *rhs;
    if (isa<arith::MinSIOp>(op))
        return lhs->slt(*rhs) ? *lhs : *rhs;
    if (isa<arith::MaxSIOp>(op))
        return lhs->sgt(*rhs) ? *lhs : *rhs;
    if (isa<arith::DivUIOp>(op) && !rhs->isZero())
        return lhs->udiv(*rhs);
    if (isa<arith::ShLIOp>(op) && rhs->ult(lhs->getBitWidth()))
        return lhs->shl(rhs->getZExtValue());
    return std::nullopt;
}

// Match apply_scale -> optional output zero point -> clamp -> i8 truncation.
static std::optional<PoolRescaleInfo> matchPoolRescale(linalg::GenericOp generic) {
    auto yield = cast<linalg::YieldOp>(generic.getBody()->getTerminator());
    auto trunc = yield.getValues()[0].getDefiningOp<arith::TruncIOp>();
    auto min = trunc ? trunc.getIn().getDefiningOp<arith::MinSIOp>() : nullptr;
    auto max = min ? min.getLhs().getDefiningOp<arith::MaxSIOp>() : nullptr;
    if (!max)
        return std::nullopt;
    auto minValue = getConstIntValue(max.getRhs());
    auto maxValue = getConstIntValue(min.getRhs());
    if (!minValue || !maxValue || *minValue < -128 || *maxValue > 127 || *minValue > *maxValue)
        return std::nullopt;

    Value scaled = max.getLhs();
    int32_t zeroPoint = 0;
    if (auto add = scaled.getDefiningOp<arith::AddIOp>()) {
        auto constant = getConstIntValue(add.getRhs());
        if (!constant || *constant < -128 || *constant > 127)
            return std::nullopt;
        zeroPoint = *constant;
        scaled = add.getLhs();
    }
    auto apply = scaled.getDefiningOp<tosa::ApplyScaleOp>();
    if (!apply || apply.getValue() != generic.getBody()->getArgument(0) ||
        apply.getRoundingMode() != tosa::RoundingMode::SINGLE_ROUND)
        return std::nullopt;
    return PoolRescaleInfo{
        apply, zeroPoint, static_cast<int32_t>(*minValue), static_cast<int32_t>(*maxValue)
    };
}

static LogicalResult appendPoolScaleRun(
    SmallVectorImpl<PoolScaleRegion> &regions, ArrayRef<size_t> previousRow,
    SmallVectorImpl<size_t> &currentRow, int64_t y, int64_t start, int64_t end,
    const PoolScale &scale
) {
    // Only matching rectangles in the preceding row can extend vertically.
    auto previous = llvm::find_if(previousRow, [&](size_t index) {
        const auto &region = regions[index];
        return region.x == start && region.width == end - start && region.scale == scale;
    });
    if (previous != previousRow.end()) {
        ++regions[*previous].height;
        currentRow.push_back(*previous);
    }
    else {
        if (regions.size() == kMaxSlidingPoolScaleRegions)
            return failure();
        currentRow.push_back(regions.size());
        regions.push_back({y, start, 1, end - start, scale});
    }
    return success();
}

static FailureOr<SmallVector<PoolScaleRegion>>
collectPoolScaleRegions(tosa::ApplyScaleOp apply, int64_t height, int64_t width, bool nchwRescale) {
    PoolScaleEvaluator evaluator(apply, nchwRescale);
    if (getConstIntValue(apply.getMultiplier()) && getConstIntValue(apply.getShift())) {
        auto scale = evaluator.scaleAt(0, 0);
        if (!scale)
            return failure();
        return SmallVector<PoolScaleRegion>{{0, 0, height, width, *scale}};
    }
    // Division avoids overflow when checking H*W against the scan budget.
    if (height > kMaxSlidingPoolScalePixels / width)
        return failure();

    SmallVector<PoolScaleRegion> regions;
    SmallVector<size_t> previousRow, currentRow;
    for (int64_t y = 0; y < height; ++y) {
        auto scale = evaluator.scaleAt(y, 0);
        if (!scale)
            return failure();
        int64_t start = 0;
        for (int64_t x = 1; x < width; ++x) {
            auto nextScale = evaluator.scaleAt(y, x);
            if (!nextScale)
                return failure();
            if (*nextScale == *scale)
                continue;
            if (failed(appendPoolScaleRun(regions, previousRow, currentRow, y, start, x, *scale)))
                return failure();
            start = x;
            scale = nextScale;
        }
        if (failed(appendPoolScaleRun(regions, previousRow, currentRow, y, start, width, *scale)))
            return failure();
        previousRow.swap(currentRow);
        currentRow.clear();
    }
    return regions;
}

struct SlidingPoolInfo {
    linalg::PoolingNhwcSumOp pool;
    linalg::TransposeOp transpose;
    RankedTensorType outputType;
    SmallVector<int64_t> kernel, stride, dilation;
};

static bool hasZeroPoolInit(linalg::PoolingNhwcSumOp pool) {
    auto init = pool.getOutputs()[0].getDefiningOp();
    if (auto fill = dyn_cast_or_null<torq_hl::FillOp>(init))
        return fill.getValue() == 0;
    if (auto fill = dyn_cast_or_null<linalg::FillOp>(init))
        return getConstIntValue(fill.getInputs()[0]) == 0;
    return false;
}

static bool hasValidSlidingPoolGeometry(const SlidingPoolInfo &info, ArrayRef<int64_t> in) {
    auto out = info.outputType.getShape();
    const auto &kernel = info.kernel, &stride = info.stride, &dilation = info.dilation;
    // Leave whole-frame pooling to the existing global pattern.
    if (kernel[KERNEL_HEIGHT] == in[NHWC_HEIGHT] && kernel[KERNEL_WIDTH] == in[NHWC_WIDTH])
        return false;
    if (out[NHWC_BATCH] != 1 || in[NHWC_BATCH] != out[NHWC_BATCH] || in[NHWC_HEIGHT] <= 0 ||
        in[NHWC_WIDTH] <= 0 || in[NHWC_CHANNEL] != out[NHWC_CHANNEL] || out[NHWC_CHANNEL] <= 0 ||
        out[NHWC_HEIGHT] <= 0 || out[NHWC_WIDTH] <= 0 || stride.size() != 2 || dilation.size() != 2)
        return false;
    for (KernelDim axis : {KERNEL_HEIGHT, KERNEL_WIDTH}) {
        const NhwcDim spatialDim = axis == KERNEL_HEIGHT ? NHWC_HEIGHT : NHWC_WIDTH;
        const int64_t inputSize = in[spatialDim];
        // Check before multiplying to avoid overflowing the dilated window.
        if (kernel[axis] <= 0 || stride[axis] <= 0 || dilation[axis] <= 0 ||
            kernel[axis] - 1 > (inputSize - 1) / dilation[axis])
            return false;
        const int64_t window = (kernel[axis] - 1) * dilation[axis] + 1;
        if (out[spatialDim] != (inputSize - window) / stride[axis] + 1)
            return false;
    }
    return hasZeroPoolInit(info.pool);
}

static std::optional<SlidingPoolInfo> matchSlidingPool(linalg::GenericOp generic, bool mark) {
    if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 ||
        generic.getNumLoops() != 4 || generic.getNumReductionLoops() != 0 ||
        llvm::any_of(generic.getIndexingMapsArray(), [](AffineMap m) { return !m.isIdentity(); }))
        return std::nullopt;
    Value sum = generic.getInputs()[0];
    auto transpose = sum.getDefiningOp<linalg::TransposeOp>();
    if (transpose) {
        // Fuse only an exclusively owned NHWC -> NCHW transpose.
        if (!transpose->getResult(0).hasOneUse() ||
            transpose.getPermutation() !=
                ArrayRef<int64_t>{NHWC_BATCH, NHWC_CHANNEL, NHWC_HEIGHT, NHWC_WIDTH})
            return std::nullopt;
        sum = transpose.getInput();
    }
    auto pool = sum.getDefiningOp<linalg::PoolingNhwcSumOp>();
    if (!pool || !pool.getResult(0).hasOneUse() || (mark && isMarkedFuseGroup(pool)))
        return std::nullopt;
    auto inputType = dyn_cast<RankedTensorType>(pool.getInputs()[0].getType());
    auto sumType = dyn_cast<RankedTensorType>(pool.getResult(0).getType());
    auto outputType = dyn_cast<RankedTensorType>(generic.getResult(0).getType());
    if (!outputType || outputType.getRank() != 4)
        return std::nullopt;
    if (transpose)
        outputType = transposeType(outputType, Permutation::nchw2nhwc());
    auto kernelType = dyn_cast<RankedTensorType>(pool.getInputs()[1].getType());
    if (!inputType || !sumType || !kernelType || !inputType.hasStaticShape() ||
        !outputType.hasStaticShape() || !kernelType.hasStaticShape() || inputType.getRank() != 4 ||
        kernelType.getRank() != 2 || !inputType.getElementType().isInteger(8) ||
        !sumType.getElementType().isInteger(32) || !outputType.getElementType().isInteger(8) ||
        sumType.getShape() != outputType.getShape())
        return std::nullopt;
    SlidingPoolInfo info{
        pool,
        transpose,
        outputType,
        llvm::to_vector(kernelType.getShape()),
        llvm::to_vector(attrValuesAsVec(pool.getStrides())),
        llvm::to_vector(attrValuesAsVec(pool.getDilations()))
    };
    if (!hasValidSlidingPoolGeometry(info, inputType.getShape()))
        return std::nullopt;
    return info;
}

static Value createSlidingAvgPool(
    linalg::GenericOp generic, SlidingPoolInfo &info, const PoolRescaleInfo &rescale,
    ArrayRef<PoolScaleRegion> regions, PatternRewriter &rewriter
) {
    // Keep one NHWC layer; only EasyKernel expands the scale regions into tasks.
    Value output = info.transpose ? createInitTensor(info.pool, rewriter, info.outputType)
                                  : generic.getOutputs()[0];
    Value weights = createI8Const(rewriter, info.pool, {1}, {1, 1, 1, 1});
    SmallVector<int32_t> biasScales;
    SmallVector<int64_t> scaleRegions;
    for (const auto &region : regions) {
        biasScales.append({0, region.scale.multiplier});
        scaleRegions.append({region.y, region.x, region.height, region.width, region.scale.shift});
    }
    auto regionsAttr =
        regions.size() == 1 ? DenseI64ArrayAttr{} : rewriter.getDenseI64ArrayAttr(scaleRegions);
    Value biasScale = createI32Const(rewriter, info.pool, biasScales);
    auto avg = torq_hl::AvgPool2DOp::create(
        rewriter, generic.getLoc(), info.outputType, output, /*input_zp=*/0, rescale.zeroPoint,
        rescale.min, rescale.max, regions.front().scale.shift, weights, biasScale,
        info.pool.getInputs()[0], /*sliding_pool=*/true, rewriter.getDenseI64ArrayAttr(info.kernel),
        rewriter.getDenseI64ArrayAttr(info.stride), rewriter.getDenseI64ArrayAttr(info.dilation),
        regionsAttr
    );
    output = avg.getOutput();
    if (info.transpose)
        output = transposeValue(output, Permutation::nhwc2nchw(), generic.getLoc(), rewriter);
    return output;
}

} // namespace

// Anchor on the rescale generic: local integer NHWC pooling is deliberately
// legal for CSS fallback, but its scale generic must still get a chance to fuse.
struct SlidingNhwcAveragePoolConversion : public OpRewritePattern<linalg::GenericOp> {
    const bool markFuseGroups;
    SlidingNhwcAveragePoolConversion(MLIRContext *context, bool mark)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/10), markFuseGroups(mark) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp generic, PatternRewriter &rewriter) const override {
        auto info = matchSlidingPool(generic, markFuseGroups);
        if (!info)
            return failure();
        auto rescale = matchPoolRescale(generic);
        if (!rescale)
            return failure();
        auto out = info->outputType.getShape();
        // Index queries still name the rescale generic's loops (NCHW after
        // a transpose); scale rectangles and the AvgPool2D op remain NHWC.
        auto regions = collectPoolScaleRegions(
            rescale->apply, out[NHWC_HEIGHT], out[NHWC_WIDTH], bool(info->transpose)
        );
        if (failed(regions))
            return rewriter.notifyMatchFailure(
                generic, "unsupported sliding pooling rescale or scale analysis limit exceeded"
            );
        if (markFuseGroups) {
            auto group = info->pool->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            if (!group)
                return failure();
            return success(markFuseGroupBackward(
                generic.getResult(0),
                {peelPadAndReshapeSource(info->pool.getInputs()[0]), info->pool.getInputs()[1]},
                rewriter, group
            ));
        }
        rewriter.replaceOp(
            generic, createSlidingAvgPool(generic, *info, *rescale, *regions, rewriter)
        );
        if (info->transpose)
            rewriter.eraseOp(info->transpose);
        rewriter.eraseOp(info->pool);
        return success();
    }
};

struct PoolingNhwcSumOpConversion : public OpRewritePattern<linalg::PoolingNhwcSumOp> {
  public:
    using LinalgOpT = linalg::PoolingNhwcSumOp;
    using TorqOp = torq_hl::AvgPool2DOp;
    using OpRewritePattern<LinalgOpT>::OpRewritePattern;

    PoolingNhwcSumOpConversion(MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups)
        : OpRewritePattern<LinalgOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgOpT linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        // Get the inputs and output of the original operation
        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();
        if (kernelShape.size() != 2 || kernelShape[0] != in_s[1] || kernelShape[1] != in_s[2]) {
            return rewriter.notifyMatchFailure(linalgOp, "Only kernel == whole frame supported");
        }
        int itemsPerChannel = in_s[1] * in_s[2];

        auto dataPerm = Permutation::none();

        bool isInt = outType.getElementType().isInteger();
        if (!isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Only integer pooling supported");
        }

        const int channelDimension = 3;
        int32_t channelCount = in_s[channelDimension];

        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, channelCount, _shift8b, _shift16b, true);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(linalgOp, "Cannot fold forward scale/clamp");
        }

        // Don't fold input rescale here. Input zp is applied after pooling in the ForwardScale
        ScaleInfo scaleInput0;

        if (_markFuseGroups) {
            if (!markFuseGroupBackward(
                    output, {peelPadAndReshapeSource(input0), kernel}, rewriter,
                    linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        // Generate torq_hl op with input in the expected format
        input0 = transposeValue(input0, dataPerm, loc, rewriter);

        // Compute scale and bias vectors
        const std::vector<int32_t> scale(
            channelCount, int32_t(scaleInput0.scale * (1 << scInfo.scaleShift) / itemsPerChannel)
        );
        const std::vector<int32_t> bias(channelCount, -scInfo.bias);

        // Prepare bias (and scale for integer ops)
        auto biasScale = createConst(interleave(bias, scale), rewriter, loc);

        // Prepare weights
        auto weights = createI8Const(
            rewriter, linalgOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
        );

        // Generate torq_hl op with output in the expected format
        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = TorqOp::create(
            rewriter, linalgOp.getLoc(), torqOutType,
            createInitTensor(linalgOp, rewriter, torqOutType), scInfo.zp, scInfo.zp, scInfo.min,
            scInfo.max, scInfo.scaleShift, weights, biasScale, input0, /*sliding_pool=*/false,
            /*kernel=*/nullptr, /*stride=*/nullptr, /*dilation=*/nullptr,
            /*scale_regions=*/nullptr
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        rewriter.replaceOp(output.getDefiningOp(), torqOut);
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

struct PoolingNchwSumOpConversion : public OpRewritePattern<linalg::PoolingNchwSumOp> {
  public:
    using LinalgOpT = linalg::PoolingNchwSumOp;
    using TorqOp = torq_hl::AvgPool2DOp;
    using OpRewritePattern<LinalgOpT>::OpRewritePattern;

    PoolingNchwSumOpConversion(MLIRContext *context, int shift8b, int shift16b, bool markFuseGroups)
        : OpRewritePattern<LinalgOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgOpT linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();
        if (kernelShape.size() != 2 || kernelShape[0] != in_s[2] || kernelShape[1] != in_s[3]) {
            return rewriter.notifyMatchFailure(linalgOp, "Only kernel == whole frame supported");
        }
        int itemsPerChannel = in_s[2] * in_s[3];

        // AvgPool2DOp expects NHWC input; transpose NCHW -> NHWC
        auto dataPerm = Permutation::nhwc2nchw().reverse();

        bool isInt = outType.getElementType().isInteger();
        bool isBF16 = outType.getElementType().isBF16();
        if (!isInt && !isBF16) {
            return rewriter.notifyMatchFailure(linalgOp, "Only integer or bf16 pooling supported");
        }

        const int channelDimension = 1;
        int32_t channelCount = in_s[channelDimension];

        // Try to fold a downstream elementwise mulf/divf generic op into the avgpool weight.
        auto [scaleValue, fusedGenericOp] = tryExtractFusedScale(output);

        // Integer pooling folds the requantization that follows it, so `output` becomes
        // the requantized value. A per-tensor requantization (TFLite AVERAGE_POOL_2D, or
        // MEAN raised to a pool) has one scale, which already includes the 1/(H*W).
        ScaleClampInfo scInfo;
        bool perTensorScale = false;
        if (isInt) {
            scInfo = foldForwardScaleClamp(output, channelCount, _shift8b, _shift16b, true);
            if (!scInfo) {
                scInfo = foldForwardScaleClamp(output, 1, _shift8b, _shift16b, false);
                perTensorScale = static_cast<bool>(scInfo);
            }
            if (!scInfo) {
                return rewriter.notifyMatchFailure(linalgOp, "Cannot fold forward scale/clamp");
            }
        }

        if (_markFuseGroups) {
            // If we found a downstream fused generic, update output so that
            // markFuseGroupBackward walks through it and marks it in the same group.
            if (fusedGenericOp) {
                output = fusedGenericOp.getResult(0);
            }
            if (!markFuseGroupBackward(
                    output, {peelPadAndReshapeSource(input0), kernel}, rewriter,
                    linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        // Transpose input NCHW -> NHWC
        input0 = transposeValue(input0, dataPerm, loc, rewriter);

        Value weights;
        Value biasScale;
        int32_t inputZp = 0;
        int32_t outputZp = 0;
        int32_t outputMin = 0;
        int32_t outputMax = 0;
        int32_t shiftFactor = 0;

        if (isInt) {
            ScaleInfo scaleInput0;
            const std::vector<int32_t> scale(
                channelCount,
                perTensorScale
                    ? scInfo.scaleNpu[0]
                    : int32_t(scaleInput0.scale * (1 << scInfo.scaleShift) / itemsPerChannel)
            );
            const std::vector<int32_t> bias(channelCount, -scInfo.bias);
            biasScale = createConst(interleave(bias, scale), rewriter, loc);
            weights = createI8Const(
                rewriter, linalgOp, std::vector<int8_t>{1}, llvm::ArrayRef<int64_t>{1, 1, 1, 1}
            );
            inputZp = scInfo.zp;
            outputZp = scInfo.zp;
            outputMin = scInfo.min;
            outputMax = scInfo.max;
            shiftFactor = scInfo.scaleShift;
        }
        else {
            // Round the scale to bf16 once. A decimal string with six digits would lose
            // most of the precision of 1 / N for a large N.
            std::vector<llvm::APFloat> weightsData(
                1, *convertFloatToType(APFloat(scaleValue), rewriter.getBF16Type())
            );
            weights = createConst(weightsData, rewriter, loc);
            std::vector<float> biasScaleData{0.0f};
            biasScale = createConst(biasScaleData, rewriter, loc);
            outputMin = 0xff800000;
            outputMax = 0x7f800000;
        }

        auto torqOutType = transposeType(output.getType(), dataPerm);
        auto torqOp = TorqOp::create(
            rewriter, linalgOp.getLoc(), torqOutType,
            createInitTensor(linalgOp, rewriter, torqOutType), inputZp, outputZp, outputMin,
            outputMax, shiftFactor, weights, biasScale, input0, /*sliding_pool=*/false,
            /*kernel=*/nullptr, /*stride=*/nullptr, /*dilation=*/nullptr,
            /*scale_regions=*/nullptr
        );
        auto torqOut = transposeValue(torqOp.getOutput(), dataPerm.reverse(), loc, rewriter);

        if (fusedGenericOp) {
            rewriter.replaceOp(fusedGenericOp, torqOut);
            rewriter.eraseOp(linalgOp);
        }
        else if (isInt) {
            // Replace the folded requantization, then erase the ops it consumed.
            SmallVector<Operation *> folded;
            for (Operation *op = output.getDefiningOp(); op != linalgOp;
                 op = op->getOperand(0).getDefiningOp())
                folded.push_back(op);
            rewriter.replaceOp(folded.front(), torqOut);
            for (Operation *op : llvm::drop_begin(folded))
                rewriter.eraseOp(op);
            rewriter.eraseOp(linalgOp);
        }
        else {
            rewriter.replaceOp(linalgOp, torqOut);
        }
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

/// Convert linalg.pooling_nchw_sum (local average pooling) to DepthwiseConv2D.
///
/// A depthwise conv2d can implement local average pooling because the two
/// operations are mathematically equivalent when the depthwise weights are
/// initialized correctly:
///   AvgPool(x) = 1/(KH·KW) · Σ(xᵢ)
///   DWConv(x)  = Σ(xᵢ · wᵢ) + bias
/// With wᵢ = 1/(KH·KW) and bias = 0, DWConv(x) = AvgPool(x).
///
/// This pattern:
/// 1. Matches linalg.pooling_nchw_sum.
/// 2. Looks for a downstream mulf/divf generic that scales by 1/(KH·KW).
/// 3. Creates depthwise conv2d weights of shape [C, 1, KH, KW] filled with
///    that scale value.
/// 4. Sets bias to zeros and groups = channelCount (making it depthwise).
/// 5. Replaces both the pooling op and the scale generic with a single
///    DepthwiseConv2DOp.
struct PoolingNchwSumOpToDW2DConversion : public OpRewritePattern<linalg::PoolingNchwSumOp> {
    using OpRewritePattern<linalg::PoolingNchwSumOp>::OpRewritePattern;

    PoolingNchwSumOpToDW2DConversion(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::PoolingNchwSumOp>(context), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::PoolingNchwSumOp linalgOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(linalgOp)) {
            return rewriter.notifyMatchFailure(linalgOp, "Already marked");
        }

        const auto loc = linalgOp.getLoc();

        Value input0 = linalgOp.getInputs()[0];
        Value kernel = linalgOp.getInputs()[1];
        Value output = linalgOp.getResultTensors()[0];
        const auto outType = mlir::cast<RankedTensorType>(output.getType());
        RankedTensorType input_type = mlir::cast<RankedTensorType>(input0.getType());
        auto in_s = input_type.getShape();
        auto kernelType = mlir::cast<RankedTensorType>(kernel.getType());
        auto kernelShape = kernelType.getShape();

        // Only match local pooling (kernel != whole frame)
        if (kernelShape.size() != 2 || (kernelShape[0] == in_s[2] && kernelShape[1] == in_s[3])) {
            return rewriter.notifyMatchFailure(linalgOp, "Only local pooling supported");
        }

        // Only bf16 for now
        if (!outType.getElementType().isBF16()) {
            return rewriter.notifyMatchFailure(linalgOp, "Only bf16 supported");
        }

        // Look for downstream elementwise mulf/divf generic that applies a scale.
        auto [scaleValue, fusedGenericOp] = tryExtractFusedScale(output);
        if (!fusedGenericOp) {
            return rewriter.notifyMatchFailure(linalgOp, "No downstream scale generic found");
        }

        // Verify scale roughly matches 1/(KH*KW).  Use a loose tolerance because
        // the constant may be in BF16 (epsilon ~ 7.8e-3).
        float expectedScale = 1.0f / static_cast<float>(kernelShape[0] * kernelShape[1]);
        if (std::abs(scaleValue - expectedScale) > 1e-2f) {
            return rewriter.notifyMatchFailure(linalgOp, "Scale does not match 1/(KH*KW)");
        }

        if (_markFuseGroups) {
            Value fuseOutput = fusedGenericOp ? fusedGenericOp.getResult(0) : output;
            if (!markFuseGroupBackward(
                    fuseOutput, {peelPadAndReshapeSource(input0), kernel}, rewriter,
                    linalgOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
                ))
                return failure();
            return success();
        }

        PaddingInfo padInfo = foldBackwardPadding(input0, rewriter, true, linalgOp.getResult(0));

        int32_t channelCount = static_cast<int32_t>(in_s[1]);

        // Create depthwise conv2d weights [C, 1, KH, KW] filled with scaleValue.
        auto bf16Type = rewriter.getBF16Type();
        auto weightType =
            RankedTensorType::get({channelCount, 1, kernelShape[0], kernelShape[1]}, bf16Type);
        auto weightAttr =
            DenseElementsAttr::get(weightType, rewriter.getFloatAttr(bf16Type, scaleValue));
        auto weights = arith::ConstantOp::create(rewriter, loc, weightType, weightAttr);

        std::vector<float> biasData(channelCount, 0.0f);
        auto biasScale = createConst(biasData, rewriter, loc);

        auto initTensor = createInitTensor(linalgOp, rewriter, outType);

        auto strides = attrValuesAsVec(linalgOp.getStrides());
        auto dilations = attrValuesAsVec(linalgOp.getDilations());

        auto dwOp = torq_hl::DepthwiseConv2DOp::create(
            rewriter, loc, outType, initTensor,
            /*input_zp=*/0, /*weight_zp=*/0, /*output_zp=*/0,
            /*output_min=*/0xff800000, /*output_max=*/0x7f800000, /*shift_factor=*/0,
            /*groups=*/channelCount, padInfo.lrtbPad, strides, dilations,
            torq_hl::VectorizationModeEnum::None, weights, biasScale, input0,
            /*isDw1dStride1=*/false, /*segment_output=*/false, /*nhwc_input=*/false
        );

        rewriter.replaceOp(fusedGenericOp, dwOp.getOutput());
        rewriter.eraseOp(linalgOp);
        return success();
    }

  private:
    const bool _markFuseGroups;
};

void populateLinalgToTorqHLPoolingPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<SlidingNhwcAveragePoolConversion>(context, markFuseGroups);
    patterns.insert<PoolingNchwMaxOpConversion>(context, markFuseGroups);
    patterns.insert<PoolingNhwcSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
    patterns.insert<PoolingNchwSumOpToDW2DConversion>(context, markFuseGroups);
    patterns.insert<PoolingNchwSumOpConversion>(context, 20, 20 /* FIXME */, markFuseGroups);
}

} // namespace mlir::syna::torq
