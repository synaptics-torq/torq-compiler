// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "torq-fuse-decomposed-gelu-pattern"

namespace mlir::syna::torq {

namespace {

// Constants of the tanh GELU: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kCubeCoefficient = 0.044715;

// The model stores the constants in bf16 (e.g. 0.796875 for sqrt(2 / pi)): accept a 1% error
static bool isNear(double value, double expected) {
    return std::abs(value - expected) <= 0.01 * std::abs(expected);
}

// An operand of one step of the chain: a tensor of the result shape, or a scalar constant
struct ChainOperand {
    Value tensor;
    std::optional<double> scalar;
};

// The generic that computes `value` if its body is a single op of type OpTy, all its loops are
// parallel and it has one result. An intermediate step of the chain must have no other user:
// otherwise it stays alive after the fusion and that part of the GELU is computed twice.
template <typename OpTy>
static OpTy getSingleBodyOp(Value value, linalg::GenericOp &generic, bool intermediate = true) {
    generic = value.getDefiningOp<linalg::GenericOp>();
    if (!generic || (intermediate && !value.hasOneUse()) || generic->getNumResults() != 1 ||
        generic.getNumParallelLoops() != generic.getNumLoops()) {
        return nullptr;
    }
    Block *body = generic.getBody();
    if (body->getOperations().size() != 2) {
        return nullptr;
    }
    auto bodyOp = dyn_cast<OpTy>(&body->front());
    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
    if (!bodyOp || yieldOp.getOperand(0) != bodyOp->getResult(0)) {
        return nullptr;
    }
    return bodyOp;
}

// The tensor or scalar constant that the body value `bodyValue` of `generic` reads. A tensor must
// be an input with the result shape and an identity map, so that the step is elementwise.
static std::optional<ChainOperand> getChainOperand(linalg::GenericOp generic, Value bodyValue) {
    if (auto scalar = traceToConstFloat(generic, bodyValue, SplatPolicy::Any)) {
        return ChainOperand{nullptr, scalar->convertToDouble()};
    }
    auto arg = dyn_cast<BlockArgument>(bodyValue);
    if (!arg || arg.getOwner() != generic.getBody() ||
        arg.getArgNumber() >= generic.getNumDpsInputs()) {
        return std::nullopt;
    }
    OpOperand *input = generic.getDpsInputOperand(arg.getArgNumber());
    if (input->get().getType() != generic->getResult(0).getType() ||
        !generic.getMatchingIndexingMap(input).isIdentity()) {
        return std::nullopt;
    }
    return ChainOperand{input->get(), std::nullopt};
}

// The two operands of `value` if it is computed by a binary elementwise generic with op OpTy
template <typename OpTy>
static std::optional<std::pair<ChainOperand, ChainOperand>>
matchBinaryStep(Value value, bool intermediate = true) {
    linalg::GenericOp generic;
    auto bodyOp = getSingleBodyOp<OpTy>(value, generic, intermediate);
    if (!bodyOp) {
        return std::nullopt;
    }
    auto lhs = getChainOperand(generic, bodyOp.getLhs());
    auto rhs = getChainOperand(generic, bodyOp.getRhs());
    if (!lhs || !rhs) {
        return std::nullopt;
    }
    return std::make_pair(*lhs, *rhs);
}

// The tensor t if `value` is t OpTy c (in any operand order), with the constant c near `expected`
template <typename OpTy> static Value matchStepWithScalar(Value value, double expected) {
    auto operands = matchBinaryStep<OpTy>(value);
    if (!operands) {
        return nullptr;
    }
    auto [lhs, rhs] = *operands;
    if (rhs.tensor) {
        std::swap(lhs, rhs);
    }
    if (!lhs.tensor || !rhs.scalar || !isNear(*rhs.scalar, expected)) {
        return nullptr;
    }
    return lhs.tensor;
}

// The tensors a and b if `value` is a OpTy b
template <typename OpTy>
static std::optional<std::pair<Value, Value>>
matchTensorStep(Value value, bool intermediate = true) {
    auto operands = matchBinaryStep<OpTy>(value, intermediate);
    if (!operands || !operands->first.tensor || !operands->second.tensor) {
        return std::nullopt;
    }
    return std::make_pair(operands->first.tensor, operands->second.tensor);
}

// True if `value` is pow(x, 3), as math.fpowi or math.powf
static bool isPowCubeOf(Value value, Value x) {
    linalg::GenericOp generic;
    if (auto fpowi = getSingleBodyOp<math::FPowIOp>(value, generic)) {
        auto lhs = getChainOperand(generic, fpowi.getLhs());
        auto exponent = traceToConstInt(generic, fpowi.getRhs(), SplatPolicy::Any);
        return lhs && lhs->tensor == x && exponent == 3;
    }
    if (auto powf = getSingleBodyOp<math::PowFOp>(value, generic)) {
        auto lhs = getChainOperand(generic, powf.getLhs());
        auto exponent = getChainOperand(generic, powf.getRhs());
        return lhs && lhs->tensor == x && exponent && exponent->scalar == 3.0;
    }
    return false;
}

// True if `value` is x^3: pow(x, 3), or (x * x) * x as PowToMulPattern makes it
static bool isCubeOf(Value value, Value x) {
    if (isPowCubeOf(value, x)) {
        return true;
    }
    auto cube = matchTensorStep<arith::MulFOp>(value);
    if (!cube) {
        return false;
    }
    auto [square, other] = *cube;
    if (square == x) {
        std::swap(square, other);
    }
    auto squareOperands = matchTensorStep<arith::MulFOp>(square);
    return other == x && squareOperands && squareOperands->first == x &&
           squareOperands->second == x;
}

// True if `value` is sqrt(2 / pi) * (x + 0.044715 * x^3), the input of the tanh
static bool isTanhInputOf(Value value, Value x) {
    Value sum = matchStepWithScalar<arith::MulFOp>(value, kSqrt2OverPi);
    auto sumOperands = sum ? matchTensorStep<arith::AddFOp>(sum) : std::nullopt;
    if (!sumOperands) {
        return false;
    }
    auto [cubeTerm, other] = *sumOperands;
    if (cubeTerm == x) {
        std::swap(cubeTerm, other);
    }
    Value cube = matchStepWithScalar<arith::MulFOp>(cubeTerm, kCubeCoefficient);
    return other == x && cube && isCubeOf(cube, x);
}

// The GELU input x if `halfInput` is 0.5 * x and `onePlusTanh` is 1 + tanh(...) of the tanh GELU
static Value matchGeluFactors(Value halfInput, Value onePlusTanh) {
    Value x = matchStepWithScalar<arith::MulFOp>(halfInput, 0.5);
    Value tanhResult = matchStepWithScalar<arith::AddFOp>(onePlusTanh, 1.0);
    if (!x || !tanhResult) {
        return nullptr;
    }
    linalg::GenericOp tanhGeneric;
    auto tanhOp = getSingleBodyOp<math::TanhOp>(tanhResult, tanhGeneric);
    if (!tanhOp) {
        return nullptr;
    }
    auto tanhInput = getChainOperand(tanhGeneric, tanhOp.getOperand());
    if (!tanhInput || !tanhInput->tensor || !isTanhInputOf(tanhInput->tensor, x)) {
        return nullptr;
    }
    return x;
}

// Body of the tanh GELU, in the form that the ONNX Gelu import makes and GeluOpPattern matches
static void buildTanhGeluBody(OpBuilder &b, Location loc, ValueRange args) {
    Value x = args[0];
    Type type = x.getType();
    auto constant = [&](double value) {
        return arith::ConstantOp::create(b, loc, b.getFloatAttr(type, value));
    };
    Value three = arith::ConstantOp::create(b, loc, b.getI64IntegerAttr(3));
    Value cube = math::FPowIOp::create(b, loc, x, three);
    Value cubeTerm = arith::MulFOp::create(b, loc, cube, constant(kCubeCoefficient));
    Value sum = arith::AddFOp::create(b, loc, x, cubeTerm);
    Value tanhInput = arith::MulFOp::create(b, loc, sum, constant(kSqrt2OverPi));
    Value tanhResult = math::TanhOp::create(b, loc, tanhInput);
    Value onePlusTanh = arith::AddFOp::create(b, loc, tanhResult, constant(1.0));
    Value half = arith::MulFOp::create(b, loc, onePlusTanh, constant(0.5));
    linalg::YieldOp::create(b, loc, ValueRange{arith::MulFOp::create(b, loc, x, half)});
}

// Replaces a tanh GELU that is decomposed into one elementwise generic per step with one generic,
// so that GeluOpPattern can lower it as one op. ONNX models that export the GELU as separate
// Mul, Pow, Add and Tanh nodes give this chain:
//   half = x * 0.5
//   cube = x^3                    (pow(x, 3), or (x * x) * x after PowToMulPattern)
//   tanhInput = (x + cube * 0.044715) * 0.7978
//   gelu = half * (tanh(tanhInput) + 1)
// Each step except the last must have the next one as its only user, so that the whole chain
// becomes dead and is erased.
class FuseDecomposedGeluPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    FuseDecomposedGeluPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("FuseDecomposedGeluPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp srcOp, PatternRewriter &rewriter) const override {
        if (srcOp->getNumResults() != 1 ||
            !getElementTypeOrSelf(srcOp->getResult(0).getType()).isBF16()) {
            return rewriter.notifyMatchFailure(srcOp, "not a bf16 generic with one result");
        }
        auto factors = matchTensorStep<arith::MulFOp>(srcOp->getResult(0), /*intermediate=*/false);
        if (!factors) {
            return rewriter.notifyMatchFailure(srcOp, "not a multiply of two tensors");
        }
        Value x = matchGeluFactors(factors->first, factors->second);
        if (!x) {
            x = matchGeluFactors(factors->second, factors->first);
        }
        if (!x) {
            return rewriter.notifyMatchFailure(srcOp, "not the last step of a tanh GELU");
        }

        auto rank = cast<RankedTensorType>(x.getType()).getRank();
        rewriter.replaceOpWithNewOp<linalg::GenericOp>(
            srcOp, srcOp->getResultTypes(), ValueRange{x}, srcOp.getDpsInits(),
            SmallVector<AffineMap>(2, rewriter.getMultiDimIdentityMap(rank)),
            SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel), buildTanhGeluBody
        );
        return success();
    }
};

} // namespace

void populateFuseDecomposedGeluPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.insert<FuseDecomposedGeluPattern>(context);
}

} // namespace mlir::syna::torq
