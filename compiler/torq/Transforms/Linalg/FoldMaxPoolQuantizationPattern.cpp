// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/QuantPatternUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <limits>

#define DEBUG_TYPE "torq-fold-maxpool-quantization"

namespace mlir::syna::torq {

namespace {

// Get the constant f32 value of v, if any
std::optional<double> getConstantFloat(Value v) {
    FloatAttr attr;
    if (!matchPattern(v, m_Constant(&attr))) {
        return std::nullopt;
    }
    return attr.getValueAsDouble();
}

// Clamp the integer tensor x to [min, max] with an elementwise linalg.generic, in the form the
// maxpool conversion to torq_hl folds in the maxpool (minsi(maxsi(x, min), max))
Value createClamp(PatternRewriter &rewriter, Location loc, Value x, double min, double max) {
    auto type = cast<RankedTensorType>(x.getType());
    Type elemType = type.getElementType();
    auto map = rewriter.getMultiDimIdentityMap(type.getRank());
    Value init = tensor::EmptyOp::create(rewriter, loc, type.getShape(), elemType);
    auto clampOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{type}, ValueRange{x}, ValueRange{init},
        ArrayRef<AffineMap>{map, map},
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value minValue = arith::ConstantOp::create(
                b, loc, b.getIntegerAttr(elemType, static_cast<int64_t>(min))
            );
            Value maxValue = arith::ConstantOp::create(
                b, loc, b.getIntegerAttr(elemType, static_cast<int64_t>(max))
            );
            Value v = arith::MaxSIOp::create(b, loc, args[0], minValue);
            v = arith::MinSIOp::create(b, loc, v, maxValue);
            linalg::YieldOp::create(b, loc, v);
        }
    );
    return clampOp.getResult(0);
}

// Quantize(MaxPool(Pad(Dequantize(x)))) is MaxPool(Pad(x)) on the int8 data when the quantize
// undoes the dequantize: max is monotonic, so it commutes with the (increasing) dequantize and
// quantize, and Quantize(Dequantize(x)) == Clamp(x) when both use the same scale and zero point,
// with the clamp of the quantize: Clamp(x) is x when the quantize clamps to the whole int8 range,
// and an int8 clamp after the maxpool otherwise (the clamp also commutes with max). The padding and
// the initial value of the f32 maxpool (normally -inf) never win as long as they are not larger
// than the lowest dequantized value and every window contains data, so they become -128. This
// removes the f32 maxpool and its quantize and dequantize ops (when the dequantize has no other
// user). The int8 maxpool is lowered to the NSS, and maxpools of the same int8 input can share
// their stages (see ShareMaxPoolStages). Unlike the QDQ maxpool conversion to torq_hl, this works
// when several maxpools read the same dequantize, as in a YOLO SPP block.
template <typename PoolOp> class FoldMaxPoolQuantization : public OpRewritePattern<PoolOp> {
  public:
    using OpRewritePattern<PoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PoolOp poolOp, PatternRewriter &rewriter) const override {
        constexpr bool isNchw = std::is_same_v<PoolOp, linalg::PoolingNchwMaxOp>;
        const int hDim = isNchw ? 2 : 1;
        const int wDim = isNchw ? 3 : 2;

        if (!poolOp.hasPureTensorSemantics() || !poolOp->getResult(0).hasOneUse()) {
            return rewriter.notifyMatchFailure(poolOp, "expected a single use");
        }
        Value input = poolOp.getDpsInputs()[0];
        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[1].getType());
        if (!kernelType || !kernelType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected a static kernel");
        }
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputType || !inputType.getElementType().isF32()) {
            return rewriter.notifyMatchFailure(poolOp, "expected an f32 maxpool");
        }

        // Optional padding, only on H and W, smaller than the kernel
        tensor::PadOp padOp = input.getDefiningOp<tensor::PadOp>();
        std::optional<double> padValue;
        SmallVector<int64_t> low, high;
        Value dequantized = input;
        if (padOp) {
            if (!padOp.getConstantPaddingValue()) {
                return rewriter.notifyMatchFailure(poolOp, "expected a constant padding");
            }
            padValue = getConstantFloat(padOp.getConstantPaddingValue());
            low = SmallVector<int64_t>(padOp.getStaticLow());
            high = SmallVector<int64_t>(padOp.getStaticHigh());
            for (size_t i = 0; i < low.size(); i++) {
                int64_t k = i == hDim   ? kernelType.getDimSize(0)
                            : i == wDim ? kernelType.getDimSize(1)
                                        : 1;
                if (!padValue || ShapedType::isDynamic(low[i]) || ShapedType::isDynamic(high[i]) ||
                    low[i] >= k || high[i] >= k) {
                    return rewriter.notifyMatchFailure(poolOp, "unsupported padding");
                }
            }
            dequantized = padOp.getSource();
        }

        // The output must start from a constant
        auto fillOp = poolOp.getDpsInits()[0].template getDefiningOp<linalg::FillOp>();
        std::optional<double> initValue =
            fillOp ? getConstantFloat(fillOp.getInputs()[0]) : std::nullopt;
        if (!initValue) {
            return rewriter.notifyMatchFailure(poolOp, "expected a constant initial value");
        }

        // The dequantize may have other users (e.g. the other maxpools of a SPP block): it is kept
        // for them
        QuantizedOpChain chain;
        chain.inputDequantOp = dequantized.getDefiningOp<linalg::GenericOp>();
        if (!chain.inputDequantOp ||
            !matchDequantGeneric(chain.inputDequantOp, chain.inputInfo.scale, chain.inputInfo.zp) ||
            failed(matchOutputQuantization(poolOp->getResult(0), rewriter, chain))) {
            return rewriter.notifyMatchFailure(poolOp, "expected dequantize and quantize ops");
        }
        Value x = chain.inputDequantOp.getInputs()[0];
        auto xType = dyn_cast<RankedTensorType>(x.getType());
        auto quantType = dyn_cast<RankedTensorType>(chain.quantOp.getResult(0).getType());
        if (!xType || !quantType || !xType.getElementType().isInteger(8) ||
            quantType.getElementType() != xType.getElementType()) {
            return rewriter.notifyMatchFailure(poolOp, "expected int8 input and output");
        }

        // The quantize must undo the dequantize, up to its clamp. The clamp bounds must be integers
        // so that clamping the integer data gives the same values as clamping before the rounding.
        const double qmin = std::numeric_limits<int8_t>::min();
        const double qmax = std::numeric_limits<int8_t>::max();
        double clampMin = std::max(chain.outputInfo.min, qmin);
        double clampMax = std::min(chain.outputInfo.max, qmax);
        if (chain.inputInfo.scale <= 0 || chain.inputInfo.scale != chain.outputInfo.scale ||
            chain.inputInfo.zp != std::llround(chain.outputInfo.zp) || clampMin > clampMax ||
            clampMin != std::round(clampMin) || clampMax != std::round(clampMax)) {
            return rewriter.notifyMatchFailure(poolOp, "quantize doesn't undo the dequantize");
        }

        // The padding and initial values must not be larger than any dequantized value
        double lowest = (qmin - chain.inputInfo.zp) * chain.inputInfo.scale;
        if (*initValue > lowest || (padValue && *padValue > lowest)) {
            return rewriter.notifyMatchFailure(poolOp, "padding or initial value too large");
        }

        Location loc = poolOp.getLoc();
        rewriter.setInsertionPoint(chain.quantOp);
        Type i8 = xType.getElementType();
        Value minValue =
            arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(i8, qmin));
        Value newInput = x;
        if (padOp) {
            auto paddedType = tensor::PadOp::inferResultType(xType, low, high);
            newInput = tensor::PadOp::create(
                rewriter, loc, paddedType, x, getAsIndexOpFoldResult(rewriter.getContext(), low),
                getAsIndexOpFoldResult(rewriter.getContext(), high), minValue
            );
        }
        Value init = tensor::EmptyOp::create(rewriter, loc, quantType.getShape(), i8);
        init = linalg::FillOp::create(rewriter, loc, ValueRange{minValue}, ValueRange{init})
                   .getResult(0);
        Value kernel = tensor::EmptyOp::create(rewriter, loc, kernelType.getShape(), i8);
        auto newPool = PoolOp::create(
            rewriter, loc, TypeRange{quantType}, ValueRange{newInput, kernel}, ValueRange{init},
            poolOp.getStrides(), poolOp.getDilations()
        );

        Value result = newPool->getResult(0);
        if (clampMin > qmin || clampMax < qmax) {
            result = createClamp(rewriter, loc, result, clampMin, clampMax);
        }

        LLVM_DEBUG(llvm::dbgs() << "Folded quantization around maxpool " << poolOp << "\n");
        rewriter.replaceOp(chain.quantOp, result);
        return success();
    }
};

} // namespace

void populateFoldMaxPoolQuantizationPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<FoldMaxPoolQuantization<linalg::PoolingNchwMaxOp>>(context);
    patterns.add<FoldMaxPoolQuantization<linalg::PoolingNhwcMaxOp>>(context);
}

} // namespace mlir::syna::torq
