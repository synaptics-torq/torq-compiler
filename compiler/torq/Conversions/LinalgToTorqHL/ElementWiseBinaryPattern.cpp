// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "OpPatternOptions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#define DEBUG_TYPE "linalg-torq-elementwsie-binary-pattern"

namespace mlir::syna::torq {

// Best rational approximation p/q of x in [0, 1] with q <= maxDen, from the continued fraction
// convergents and the last semiconvergent.
static std::pair<int64_t, int64_t> bestRational(double x, int64_t maxDen) {
    int64_t p0 = 0, q0 = 1, p1 = 1, q1 = 0;
    double v = x;
    for (int i = 0; i < 64; ++i) {
        const int64_t a = static_cast<int64_t>(std::floor(v));
        const int64_t p2 = a * p1 + p0, q2 = a * q1 + q0;
        if (q2 > maxDen) {
            const int64_t k = (maxDen - q0) / q1;
            const int64_t ps = k * p1 + p0, qs = k * q1 + q0;
            if (std::abs(x - static_cast<double>(ps) / qs) <
                std::abs(x - static_cast<double>(p1) / q1)) {
                return {ps, qs};
            }
            return {p1, q1};
        }
        p0 = p1, q0 = q1, p1 = p2, q1 = q2;
        const double frac = v - static_cast<double>(a);
        if (frac < 1e-12) {
            break;
        }
        v = 1.0 / frac;
    }
    return {p1, q1};
}

// ACT scale and right shift with scale / 2^shift as close to `multiplier` as a 32-bit scale allows.
struct ActScale {
    int32_t scale;
    int shift;
};

static std::optional<ActScale> fitActScale(double multiplier) {
    // Right shifts up to this value matched TFLite bit-exactly on SL2610 (cofi_nss_v3 int8).
    constexpr int maxActShift = 46;
    int shift =
        static_cast<int>(std::floor(std::log2(std::numeric_limits<int32_t>::max() / multiplier)));
    shift = std::min(shift, maxActShift);
    const double scale = std::round(std::ldexp(multiplier, shift));
    if (shift < 0 || scale < 1 || scale > std::numeric_limits<int32_t>::max()) {
        return std::nullopt;
    }
    return ActScale{static_cast<int32_t>(scale), shift};
}

// Weights and ACT scale/shift for acc = w0 * in0 + w1 * in1, out = (acc + bias) * scale >> shift.
// The i16 weights only carry the ratio of the two multipliers, as the closest fraction with a
// denominator that keeps the i32 accumulator in range; the 32-bit ACT scale carries the
// magnitude. With weights that carry the magnitude too, the shift is capped where the larger
// weight fills an i16, which leaves an 8-bit add with about 14 bits of precision per operand.
struct RatioScale {
    int16_t weight0, weight1;
    ActScale act;
};

static std::optional<RatioScale>
fitRatioScale(double multiplier0, double multiplier1, double range0, double range1) {
    const double big = std::max(multiplier0, multiplier1);
    const double small = std::min(multiplier0, multiplier1);
    if (!(small > 0.0)) {
        return std::nullopt;
    }
    const bool firstIsBig = multiplier0 >= multiplier1;
    const double ratio = small / big;
    const double bigRange = firstIsBig ? range0 : range1;
    const double smallRange = firstIsBig ? range1 : range0;
    const double maxDen = std::min<double>(
        std::numeric_limits<int16_t>::max(),
        std::floor(std::numeric_limits<int32_t>::max() / (bigRange + ratio * smallRange))
    );
    if (maxDen < 1) {
        return std::nullopt;
    }
    auto [p, q] = bestRational(ratio, static_cast<int64_t>(maxDen));
    if (p <= 0 || q <= 0) {
        return std::nullopt;
    }
    auto act = fitActScale(big / static_cast<double>(q));
    if (!act) {
        return std::nullopt;
    }
    RatioScale fit;
    fit.weight0 = static_cast<int16_t>(firstIsBig ? q : p);
    fit.weight1 = static_cast<int16_t>(firstIsBig ? p : q);
    fit.act = *act;
    return fit;
}

// Weight, bias and ACT scale/shift for acc = w * in + bias, out = acc * scale >> shift, computing
// multiplier * in + offset for an add whose other operand is a scalar folded into `offset`. As in
// fitRatioScale the 32-bit ACT scale carries the magnitude, so the weight only sets the bias
// resolution and takes the largest value that keeps the i32 accumulator in range.
struct ScalarScale {
    int16_t weight;
    int32_t bias;
    ActScale act;
};

static std::optional<ScalarScale> fitScalarScale(double multiplier, double range, double offset) {
    const double magnitude = std::abs(multiplier);
    if (!(magnitude > 0.0)) {
        return std::nullopt;
    }
    const double weight = std::min<double>(
        std::numeric_limits<int16_t>::max(),
        std::floor(std::numeric_limits<int32_t>::max() / (range + std::abs(offset) / magnitude))
    );
    if (weight < 1) {
        return std::nullopt;
    }
    auto act = fitActScale(magnitude / weight);
    if (!act) {
        return std::nullopt;
    }
    ScalarScale fit;
    fit.weight = static_cast<int16_t>(multiplier < 0 ? -weight : weight);
    fit.bias = static_cast<int32_t>(std::round(std::ldexp(offset, act->shift) / act->scale));
    fit.act = *act;
    return fit;
}

// True when alignInputToOutputShape has to materialize `input` at the output shape, which is
// the cost the caller weighs. Compares shapes rather than element counts so that a reshape is
// not mistaken for a copy, and so that operands the alignment cannot handle at all (a mismatch
// in a non-unit axis, e.g. tensor<2x3> against tensor<3x2>) are caught here instead of forming
// a torq_hl.add with mismatched operands.
// Rank padding with leading unit axes is a reshape, and scalars fold into the ACT bias, so
// neither counts.
static bool needsBroadcastMaterialization(Value input, RankedTensorType outType) {
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    if (!inType || !inType.hasStaticShape() || !outType.hasStaticShape()) {
        return false;
    }
    ArrayRef<int64_t> inShape = inType.getShape();
    ArrayRef<int64_t> outShape = outType.getShape();
    if (llvm::all_of(inShape, [](int64_t dim) { return dim == 1; })) {
        return false;
    }
    if (inShape.size() > outShape.size()) {
        return true;
    }
    // Right-aligned, matching alignInputToOutputShape: the leading axes the input lacks are a
    // reshape only when the output has them unit sized.
    const size_t leadingPad = outShape.size() - inShape.size();
    if (llvm::any_of(outShape.take_front(leadingPad), [](int64_t dim) { return dim != 1; })) {
        return true;
    }
    return inShape != outShape.drop_front(leadingPad);
}

struct EltwiseBinaryConvert : public OpRewritePattern<linalg::GenericOp> {
  public:
    enum OpType {
        ADD_OP,
        SUB_OP,
        // Other operations can be added here if needed
    };
    using LinalgEltOpT = linalg::GenericOp;
    using TorqEltOp = torq_hl::AddOp;
    using OpRewritePattern<LinalgEltOpT>::OpRewritePattern;

    bool foldBackwardScalarConstant(
        Value &input, ScaleInfo &scaleInfo, int32_t *scalarValue, PatternRewriter &rewriter
    ) const {

        auto elOp = input.getDefiningOp<Operation *>();
        bool maybeReplace = false;
        if (isa_and_nonnull<tensor::ExpandShapeOp>(elOp) ||
            isa_and_nonnull<tensor::CollapseShapeOp>(elOp)) {
            maybeReplace = true;
        }
        if (auto maybeElemOp = dyn_cast_if_present<linalg::GenericOp>(elOp)) {
            auto &r = maybeElemOp.getRegion();
            if (r.hasOneBlock() && isa<linalg::YieldOp>(r.front().front())) {
                maybeReplace = true;
            }
        }

        // FIXME Replacing expand, collapse or its corresponding linalg.generic op
        // with its input only works when the input is a scalar tensor with dims 1x1x..x1.
        // Probe ahead first to check if the fold will succeed before making
        // destructive IR changes.  If the underlying value is not a constant,
        // leave the IR untouched so the caller can bail out cleanly.
        Value candidateInput = input;
        Operation *candidateElOp = nullptr;
        if (maybeReplace) {
            auto elOpInput = elOp->getOperand(0);
            auto elOpInputShape = dyn_cast<RankedTensorType>(elOpInput.getType()).getShape();
            if (mlir::computeProduct(elOpInputShape) == 1) {
                candidateInput = elOpInput;
                candidateElOp = elOp;
            }
        }

        auto inputType = dyn_cast<RankedTensorType>(candidateInput.getType());
        auto inputShape = inputType.getShape();
        auto inputElementType = inputType.getElementType();

        Value scalarInput = candidateInput;
        ScaleInfo probeScaleInfo = scaleInfo;

        // Check if candidateInput is already a constant before rescale probing
        auto constOp = dyn_cast_if_present<arith::ConstantOp>(scalarInput.getDefiningOp());
        if (!constOp && inputShape.size() <= 1) {
            while (foldBackwardRescale(scalarInput, probeScaleInfo)) {
            }

            while (foldScalarRescale(scalarInput, probeScaleInfo, inputElementType, rewriter)) {
            }
            constOp = dyn_cast<arith::ConstantOp>(scalarInput.getDefiningOp());
        }

        if (constOp && getIntegerConstantValue(constOp, scalarValue)) {
            // Only now commit the destructive IR changes
            input = candidateInput;
            if (candidateElOp &&
                candidateElOp->getResult(0).getType() == candidateInput.getType()) {
                rewriter.replaceOp(candidateElOp, candidateInput);
            }
            scaleInfo = probeScaleInfo;
            return true;
        }
        return false;
    }

    bool broadcastProcessing(Value &input, PatternRewriter &rewriter) const {
        auto bkpInput = input;
        if (!input.hasOneUse()) {
            return false;
        }
        auto selectInput = [](Value &in, Value subst) {
            if (!in.hasOneUse()) {
                return false;
            }
            in = subst;
            return true;
        };

        linalg::GenericOp bcastOp = input.getDefiningOp<linalg::GenericOp>();
        std::optional<SmallVector<int64_t>> bcastDims;
        if (bcastOp) {
            bcastDims = linalg::isaBroadcastOpInterface(bcastOp);
            if (bcastDims) {
                auto success = selectInput(input, bcastOp.getDpsInputOperand(0)->get());
                if (!success) {
                    input = bkpInput;
                    return false;
                }
            }
        }

        auto expandOp = input.getDefiningOp<tensor::ExpandShapeOp>();
        if (expandOp) {
            auto success = selectInput(input, expandOp.getSrc());
            if (!success) {
                input = bkpInput;
                return false;
            }
        }

        auto collapseOp = input.getDefiningOp<tensor::CollapseShapeOp>();
        if (collapseOp) {
            auto success = selectInput(input, collapseOp.getSrc());
            if (!success) {
                input = bkpInput;
                return false;
            }
        }

        auto maybeElemOp = input.getDefiningOp<linalg::GenericOp>();
        if (maybeElemOp) {
            auto &r = maybeElemOp.getRegion();
            if (r.hasOneBlock() && isa<linalg::YieldOp>(r.front().front())) {
                auto success = selectInput(input, maybeElemOp.getOperand(0));
                if (!success) {
                    input = bkpInput;
                    return false;
                }
            }
            else {
                maybeElemOp = nullptr;
            }
        }

        auto inputElementType = dyn_cast<RankedTensorType>(input.getType()).getElementType();

        // linalg.generic as collapse_shape
        if (maybeElemOp) {
            auto maybeElemOpOutputShape =
                dyn_cast<RankedTensorType>(maybeElemOp.getResult(0).getType()).getShape();

            auto createElOp = [&](PatternRewriter &rewriter, Location loc, Type elementType) {
                auto newOutputType =
                    RankedTensorType::get(maybeElemOpOutputShape, inputElementType);

                auto emptyOp = tensor::EmptyOp::create(
                    rewriter, maybeElemOp.getLoc(), newOutputType.getShape(),
                    newOutputType.getElementType()
                );

                auto newOp = linalg::GenericOp::create(
                    rewriter, loc, ArrayRef<Type>{newOutputType}, ValueRange{input},
                    ValueRange{emptyOp},
                    /*indexingMaps=*/
                    ArrayRef<AffineMap>{
                        maybeElemOp.getIndexingMapsArray().front(),
                        maybeElemOp.getIndexingMapsArray().back()
                    },
                    /*iteratorTypes=*/maybeElemOp.getIteratorTypesArray(),
                    [&](OpBuilder &bBuilder, Location bLoc, ValueRange bArgs) {
                        linalg::YieldOp::create(bBuilder, bLoc, bArgs[0]);
                    }
                );

                return newOp;
            };

            auto newOp = createElOp(rewriter, maybeElemOp.getLoc(), inputElementType);

            rewriter.replaceOp(maybeElemOp, newOp->getResults());
            input = newOp->getResult(0);
        }

        // tensor.collapse_shape
        if (collapseOp) {
            auto collapseShape = collapseOp.getResultType().getShape();
            auto newOutputType = RankedTensorType::get(collapseShape, inputElementType);

            auto newCollapseOp = tensor::CollapseShapeOp::create(
                rewriter, collapseOp.getLoc(), newOutputType, input,
                collapseOp.getReassociationIndices()
            );
            rewriter.replaceOp(collapseOp, newCollapseOp.getResult());
            input = newCollapseOp.getResult();
        }

        // tensor.expand_shape
        if (expandOp) {
            auto expandShape = expandOp.getResultType().getShape();
            auto newOutputType = RankedTensorType::get(expandShape, inputElementType);

            auto newExpandOp = tensor::ExpandShapeOp::create(
                rewriter, expandOp.getLoc(), newOutputType, input,
                expandOp.getReassociationIndices()
            );
            rewriter.replaceOp(expandOp, newExpandOp.getResult());
            input = newExpandOp.getResult();
        }

        if (bcastDims) {

            auto dstTy = bcastOp.getDpsInitOperand(0)->get().getType();
            auto src = bcastOp.getDpsInputOperand(0)->get();
            auto bOutputShape = mlir::cast<RankedTensorType>(dstTy).getShape();
            auto bOutputType = RankedTensorType::get(bOutputShape, inputElementType);

            auto op = linalg::BroadcastOp::create(
                rewriter, bcastOp.getLoc(), src, createInitTensor(bcastOp, rewriter, bOutputType),
                *bcastDims
            );
            auto gOp = linalg::generalizeNamedOp(rewriter, op);
            rewriter.replaceOp(bcastOp, gOp->getResults()[0]);

            input = gOp->getResults()[0];
        }

        return true;
    }

    // After backward fold peeled shape casts and broadcasts, re-materialize the input at the
    // add output shape. Handles two cases:
    //   * a strict suffix match where only leading unit dims are missing (use expand_shape), and
    //   * a broadcast where some output dims have no corresponding input axis (use a linalg
    //     broadcast generalized to a linalg.generic so downstream tiling can fuse it).
    bool alignInputToOutputShape(Value &input, Value output, PatternRewriter &rewriter) const {
        auto outTy = cast<RankedTensorType>(output.getType());
        auto inTy = cast<RankedTensorType>(input.getType());
        if (inTy.getShape() == outTy.getShape()) {
            return false;
        }

        // foldBackwardShapeCast peels pure reshapes (expand/collapse, no data
        // reordering) off the operand.  That can leave it element-count-equal to
        // the output but at a rank the broadcast/rank-pad logic below cannot reach
        // (it only *adds* leading dims, inRank <= outRank).  Example: a 1x1-conv
        // result [15,1,1,256] whose reshape to [1,15,256] was folded away.  Since
        // the peeled cast never reordered data, restore the consumer's shape with
        // an equivalent reshape (collapse to 1-D, expand to the output shape).
        if (inTy.hasStaticShape() && outTy.hasStaticShape() &&
            mlir::computeProduct(inTy.getShape()) == mlir::computeProduct(outTy.getShape())) {
            Value flat = input;
            if (inTy.getRank() != 1) {
                SmallVector<ReassociationIndices> toFlat(1);
                for (int64_t i = 0; i < inTy.getRank(); ++i)
                    toFlat[0].push_back(i);
                flat = tensor::CollapseShapeOp::create(
                    rewriter, input.getLoc(),
                    RankedTensorType::get(
                        {mlir::computeProduct(inTy.getShape())}, inTy.getElementType()
                    ),
                    input, toFlat
                );
            }
            if (outTy.getRank() == 1) {
                input = flat;
            }
            else {
                SmallVector<ReassociationIndices> toOut(1);
                for (int64_t i = 0; i < outTy.getRank(); ++i)
                    toOut[0].push_back(i);
                input = tensor::ExpandShapeOp::create(rewriter, input.getLoc(), outTy, flat, toOut)
                            .getResult();
            }
            return true;
        }

        if (inTy.getRank() > outTy.getRank() || inTy.getRank() == 0) {
            return false;
        }

        ArrayRef<int64_t> inShape = inTy.getShape();
        ArrayRef<int64_t> outShape = outTy.getShape();

        // Right-align the input shape with the output shape; missing leading axes are treated
        // as broadcast dimensions. Any axis that matches in size is preserved, axes where the
        // input has size 1 and output has a larger size are also broadcast.
        const int64_t outRank = outTy.getRank();
        const int64_t inRank = inTy.getRank();
        const int64_t leadingPad = outRank - inRank;
        SmallVector<int64_t> broadcastDims;
        SmallVector<int64_t> keptShape;
        for (int64_t d = 0; d < outRank; ++d) {
            int64_t inIdx = d - leadingPad;
            if (inIdx < 0) {
                broadcastDims.push_back(d);
                continue;
            }
            int64_t inDim = inShape[inIdx];
            int64_t outDim = outShape[d];
            if (inDim == outDim) {
                keptShape.push_back(inDim);
            }
            else if (inDim == 1) {
                broadcastDims.push_back(d);
            }
            else {
                return false;
            }
        }

        if (broadcastDims.empty()) {
            // Pure rank-padding (leading unit dims). Use a single expand_shape that folds the
            // missing axes into the first input axis.
            SmallVector<ReassociationIndices> reassociation;
            ReassociationIndices first;
            for (int64_t i = 0; i <= leadingPad; ++i) {
                first.push_back(i);
            }
            reassociation.push_back(first);
            for (int64_t i = leadingPad + 1; i < outRank; ++i) {
                reassociation.push_back({i});
            }
            auto expandOp = tensor::ExpandShapeOp::create(
                rewriter, input.getLoc(), outTy, input, reassociation
            );
            input = expandOp.getResult();
            return true;
        }

        // Need to broadcast. linalg.broadcast expects an input whose rank equals output rank
        // minus the number of broadcast dims, so squeeze the input to that rank first by
        // dropping the broadcast positions that fall inside the input's current shape.
        Value src = input;
        if (static_cast<int64_t>(keptShape.size()) != inRank) {
            auto squeezedTy = RankedTensorType::get(keptShape, inTy.getElementType());
            SmallVector<ReassociationIndices> reassociation;
            ReassociationIndices group;
            int64_t inIdx = 0;
            int64_t keptIdx = 0;
            for (int64_t d = 0; d < outRank; ++d) {
                int64_t srcIdx = d - leadingPad;
                if (srcIdx < 0) {
                    continue;
                }
                if (llvm::is_contained(broadcastDims, d)) {
                    // Unit dim in input that broadcasts; fold into the previous kept axis,
                    // or the next kept axis if there is no previous one.
                    if (!group.empty() || keptIdx == 0) {
                        group.push_back(inIdx);
                    }
                    else {
                        reassociation.back().push_back(inIdx);
                    }
                    ++inIdx;
                    continue;
                }
                group.push_back(inIdx);
                reassociation.push_back(group);
                group.clear();
                ++inIdx;
                ++keptIdx;
            }
            if (!group.empty()) {
                if (reassociation.empty()) {
                    reassociation.push_back(group);
                }
                else {
                    reassociation.back().append(group.begin(), group.end());
                }
            }
            if (static_cast<int64_t>(reassociation.size()) !=
                static_cast<int64_t>(keptShape.size())) {
                return false;
            }
            src = tensor::CollapseShapeOp::create(
                      rewriter, input.getLoc(), squeezedTy, src, reassociation
            )
                      .getResult();
        }

        auto init =
            tensor::EmptyOp::create(rewriter, input.getLoc(), outShape, inTy.getElementType());
        auto bcastOp =
            linalg::BroadcastOp::create(rewriter, input.getLoc(), src, init, broadcastDims);
        auto generalized = linalg::generalizeNamedOp(rewriter, bcastOp);
        input = succeeded(generalized) ? generalized->getResults()[0] : bcastOp.getResults()[0];
        return true;
    }

    static bool isQuantizedOperand(Value input) {
        const unsigned bits = cast<RankedTensorType>(input.getType()).getElementTypeBitWidth();
        return bits == 8 || bits == 16;
    }

    EltwiseBinaryConvert(
        MLIRContext *context, int shift8b, int shift16b, OpType opType, bool markFuseGroups
    )
        : OpRewritePattern<LinalgEltOpT>(context), _shift8b(shift8b), _shift16b(shift16b),
          _opType(opType), _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(LinalgEltOpT eltOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups && isMarkedFuseGroup(eltOp)) {
            return rewriter.notifyMatchFailure(eltOp, "Already marked");
        }

        Operation *binaryOp = getElementwiseBinaryOp(eltOp);
        if (!binaryOp) {
            return rewriter.notifyMatchFailure(eltOp, "Not an elementwise binary op");
        }

        // Get the inputs and output of the original operation
        Value input0 = eltOp.getInputs()[0];
        // Binary ops can actually have only one input if both operands are the same
        Value input1 = eltOp.getInputs()[eltOp.getInputs().size() > 1 ? 1 : 0];
        Value output = eltOp.getResultTensors()[0];
        auto outType = cast<RankedTensorType>(output.getType());

        const auto loc = eltOp.getLoc();

        // TODO:
        // 1. handle arith.AddFOp and floating point operations
        // 2. handle two inputs are identical, rescale pattern is different

        int sign = 1;
        auto opName = "add"; // Default operation name is "add" (for decoration purposes)
        switch (_opType) {
        case ADD_OP:
            if (clACTBasedAdd) {
                return rewriter.notifyMatchFailure(
                    eltOp, "ACT-based add enabled; ALU-based AddOp not expected"
                );
            }
            if (!dyn_cast<arith::AddIOp>(binaryOp)) {
                return rewriter.notifyMatchFailure(eltOp, "Expected arith.addi");
            }
            break;
        case SUB_OP:
            if (clACTBasedSub) {
                return rewriter.notifyMatchFailure(
                    eltOp, "ACT-based sub enabled; ALU-based SubOp not expected"
                );
            }
            if (!dyn_cast<arith::SubIOp>(binaryOp)) {
                return rewriter.notifyMatchFailure(eltOp, "Expected arith.subi");
            }
            sign = -1; // For subtraction, we need to negate the second operand
            opName = "sub";
            break;
        }

        bool isInt = outType.getElementType().isInteger();

        // check if output has rescaleClamp firstly to quik return if it is not present
        const int outChannelCount = 1; // No channels here, only one single scale
        ScaleClampInfo scInfo =
            foldForwardScaleClamp(output, outChannelCount, _shift8b, _shift16b, true);
        if (!scInfo && isInt) {
            return rewriter.notifyMatchFailure(eltOp, "Cannot fold forward scale/clamp");
        }

        // Fold inputs rescale if present (peel shape casts and broadcast wrappers so rescale
        // chains are reachable). Upstream broadcast materialization can wrap a per-channel
        // constant in expand_shape + yield-only broadcast linalg.generic; peeling through both
        // lets foldBackwardRescale recover the original constant and capture its scale.
        ScaleInfo scaleInput0;
        while (foldBackwardShapeCast(input0) || foldBackwardYieldOnlyGeneric(input0) ||
               foldBackwardRescale(input0, scaleInput0)) {
        }
        ScaleInfo scaleInput1;
        while (foldBackwardShapeCast(input1) || foldBackwardYieldOnlyGeneric(input1) ||
               foldBackwardRescale(input1, scaleInput1)) {
        }

        // Fusing approximates both input scales with i16 weights and materializes any
        // broadcast operand at the full output shape (broadcastProcessing below), which for a
        // per-channel bias repeats per op and dominates runtime. Hand those to the unfused
        // lowering: it keeps the operand small and rescales exactly with 32-bit multipliers.
        // Full tensors have nothing to materialize, so fusing stays the cheaper trade.
        auto scaledOutType = cast<RankedTensorType>(output.getType());
        if (outType.getElementType().isInteger(32) &&
            scaledOutType.getElementType().isInteger(16) &&
            (needsBroadcastMaterialization(input0, outType) ||
             needsBroadcastMaterialization(input1, outType))) {
            return rewriter.notifyMatchFailure(
                eltOp, "Broadcast operand: keep i32 op and i16 requantization separate"
            );
        }

        // Compute scale and bias vectors
        const double outputScale = scInfo.scaleDouble[0];
        double multiplier0 = outputScale * scaleInput0.scale;
        double multiplier1 = outputScale * scaleInput1.scale;
        // Grow the shift and round the weights so each i16 weight keeps as many significant
        // bits as it can: at the fixed shift of 12 a multiplier of 0.0197 becomes 80 (a 0.7%
        // error), and even a well conditioned 8-bit multiplier of 0.7245 becomes 2967 instead
        // of 2967.4, which flips outputs that sit near a rounding boundary.
        const bool refineScale = isInt && isQuantizedOperand(input0) && isQuantizedOperand(input1);
        int scaleShift = scInfo.scaleShift;
        if (refineScale) {
            scaleShift = maximizeScaleShift(
                scaleShift, multiplier0, multiplier1, operandRange(input0, scaleInput0.zp),
                operandRange(input1, scaleInput1.zp)
            );
        }
        const int scaleFactor = 1 << scaleShift;

        if (multiplier0 * scaleFactor > std::numeric_limits<int16_t>::max() ||
            multiplier0 * scaleFactor < std::numeric_limits<int16_t>::min() ||
            multiplier1 * scaleFactor > std::numeric_limits<int16_t>::max() ||
            multiplier1 * scaleFactor < std::numeric_limits<int16_t>::min()) {
            return rewriter.notifyMatchFailure(eltOp, "Multiplier overflow");
        }

        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input0, input1}, rewriter,
                eltOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        broadcastProcessing(input0, rewriter);
        broadcastProcessing(input1, rewriter);
        alignInputToOutputShape(input0, output, rewriter);
        alignInputToOutputShape(input1, output, rewriter);

        // Rounding to nearest matters because doubleToInt truncates, which throws away up to a
        // full LSB of the multiplier.
        auto toFixedPoint = [refineScale](double value) {
            return refineScale ? std::round(value) : value;
        };
        int16_t weight0 = doubleToInt<int16_t>(toFixedPoint(multiplier0 * scaleFactor));
        int16_t weight1 = doubleToInt<int16_t>(toFixedPoint(multiplier1 * scaleFactor)) * sign;
        // Deriving the bias from the weight cancels the zero point exactly, which rounding the
        // whole product does not.
        int32_t bias0 = -weight0 * scaleInput0.zp;
        int32_t bias1 = -weight1 * scaleInput1.zp;

        // Scalar operand contribution to the output, before the output zero point.
        double scalarOffset = 0.0;
        int32_t scalarValue0 = 0;
        bool input0IsScalar =
            foldBackwardScalarConstant(input0, scaleInput0, &scalarValue0, rewriter);

        int32_t scalarValue1 = 0;
        bool input1IsScalar =
            foldBackwardScalarConstant(input1, scaleInput1, &scalarValue1, rewriter);

        if (input0IsScalar && input1IsScalar) {
            return rewriter.notifyMatchFailure(eltOp, "don't support both input is scalar for now");
        }

        if (input0IsScalar) {
            weight0 = 0;
            // instead of using the ALU to perform the operation
            // we use the ACT bias to perform the operation:
            //
            // original operation:
            //
            //   output = ( (input0 - zp0) * inputScale0 + sign * (input1 - zp1) * inputScale1 ) *
            //   outputScale + outputZp
            //
            // rewritten operation:
            //
            //   output = (input1 * weight0 + input1 * weight1 + actBias) * actScale / ( 1 <<
            //   actShift ) + actZp
            //
            // where:
            //
            //   weight0 = 0
            //   weight1 = sign * inputScale1 * outputScale * ( 1 << actShift )
            //   actBias = bias0 + bias1
            //   bias0 = ( (input0 - zp0) * inputScale0 * outputScale * ( 1 << actShift ))
            //   bias1 = - sign * zp1 * weight1
            //   actScale = 1
            //   actShift = scaleShift
            //   actZp = outputZp
            scalarOffset = (scalarValue0 - scaleInput0.zp) * scaleInput0.scale * outputScale;
            bias0 = doubleToInt<int32_t>(toFixedPoint(scalarOffset * scaleFactor));

            input0 = input1;
        }
        else if (input1IsScalar) {
            // force weight1 is 0, input0 * weight0 + input1 * 0
            weight1 = 0;

            // Read above comment for the detailed explanation of the formula
            scalarOffset = (scalarValue1 - scaleInput1.zp) * scaleInput1.scale * sign * outputScale;
            bias1 = doubleToInt<int32_t>(toFixedPoint(scalarOffset * scaleFactor));
            input1 = input0;
        }

        // Quantized operands move the magnitude into the 32-bit ACT scale; the formula above with
        // actScale = 1 is the fallback.
        int32_t actScale = 1;
        if (refineScale && input0IsScalar) {
            if (auto fit = fitScalarScale(
                    sign * multiplier1, operandRange(input1, scaleInput1.zp), scalarOffset
                )) {
                weight1 = fit->weight;
                bias0 = fit->bias;
                bias1 = -weight1 * scaleInput1.zp;
                actScale = fit->act.scale;
                scaleShift = fit->act.shift;
            }
        }
        else if (refineScale && input1IsScalar) {
            if (auto fit = fitScalarScale(
                    multiplier0, operandRange(input0, scaleInput0.zp), scalarOffset
                )) {
                weight0 = fit->weight;
                bias0 = -weight0 * scaleInput0.zp;
                bias1 = fit->bias;
                actScale = fit->act.scale;
                scaleShift = fit->act.shift;
            }
        }
        else if (refineScale) {
            if (auto fit = fitRatioScale(
                    multiplier0, multiplier1, operandRange(input0, scaleInput0.zp),
                    operandRange(input1, scaleInput1.zp)
                )) {
                weight0 = fit->weight0;
                weight1 = fit->weight1 * sign;
                bias0 = -weight0 * scaleInput0.zp;
                bias1 = -weight1 * scaleInput1.zp;
                actScale = fit->act.scale;
                scaleShift = fit->act.shift;
            }
        }

        // concatenate weight0 and weight1
        std::vector<int16_t> weights = {weight0, weight1};
        auto torqWeights = createConst(weights, rewriter, loc);

        VectorIntOrFloat bias(1, isInt);
        bias.ints[0] = bias0 + bias1; // TODO: handle the case of floating point operation

        const std::vector<int32_t> scale = {actScale}; // TODO: handle floating point operation

        // Prepare bias (and scale for integer ops)
        auto biasScale = isInt ? createConst(interleave(bias.ints, scale), rewriter, loc)
                               : createConst(bias.floats, rewriter, loc);

        // Generate torq_hl op with output in the expected format
        outType = cast<RankedTensorType>(output.getType());
        auto torqOp = TorqEltOp::create(
            rewriter, loc, outType, createInitTensor(eltOp, rewriter, outType), opName,
            /* input zp not needed */ 0, scInfo.zp, scInfo.min, scInfo.max, scaleShift, torqWeights,
            biasScale, input0, input1
        );

        rewriter.replaceOp(output.getDefiningOp(), torqOp.getOutput());
        return success();
    }

  private:
    const int _shift8b;         // Scale shift for 8-bit integer operations
    const int _shift16b;        // Scale shift for 16-bit integer operations
    const OpType _opType;       // Type of the operation (ADD_OP, SUB_OP, etc.)
    const bool _markFuseGroups; // When true, mark the TI operations, don't convert.
};

void populateLinalgToTorqHLEWBinaryPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    // These patterns are applied after the high-priority patterns
    // This allows to apply the most beneficial patterns first (eg conv2d with bias addition)
    // before the remaining patterns (eg addition)
    // Note: using benefit to control the order of application is not enough since this
    // only works for patterns that are applied to the same op
    // sh8b and sh16b are only floors, maximizeScaleShift grows them per operation.
    // FIXME the 16b floor is unvalidated: maximizeScaleShift only tests shift + 1, and the
    // caller checks the i16 weight bound but not the accumulator, so with 16-bit operands
    // and both multipliers >= 4 a shift of 12 overflows the i32 accumulator undetected.
    // 8-bit operands have 8 more bits of headroom and cannot reach it.
    int sh8b = 12;
    int sh16b = 12;
    patterns.insert<EltwiseBinaryConvert>(
        context, sh8b, sh16b, EltwiseBinaryConvert::ADD_OP, markFuseGroups
    );
    patterns.insert<EltwiseBinaryConvert>(
        context, sh8b, sh16b, EltwiseBinaryConvert::SUB_OP, markFuseGroups
    );
    // bf16 float add/sub is lowered by AddOpPattern (createBf16Add) in the main
    // LinalgToTorqHL conversion pass; no dedicated pre-pass pattern is needed.
}

} // namespace mlir::syna::torq
