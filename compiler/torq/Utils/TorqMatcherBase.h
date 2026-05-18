// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "iree-dialects/Transforms/TransformMatchers.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <type_traits>

namespace mlir::syna::torq {

template <typename T> inline bool matchYieldDagPath(Value value) {
    return value.getDefiningOp<T>() != nullptr;
}

// Checks whether `value`'s defining-op chain follows the order `First -> Second -> Rest...`.
template <typename First, typename Second, typename... Rest>
inline bool matchYieldDagPath(Value value) {
    auto op = value.getDefiningOp<First>();
    if (!op) {
        return false;
    }

    return llvm::any_of(op->getOperands(), [](Value operand) {
        return matchYieldDagPath<Second, Rest...>(operand);
    });
}

template <typename OpTy = linalg::GenericOp>
class TorqStructuredOpMatcher
    : public transform_ext::ConcreteOpMatcher<TorqStructuredOpMatcher<OpTy>, OpTy> {
  public:
    using Base = transform_ext::ConcreteOpMatcher<TorqStructuredOpMatcher<OpTy>, OpTy>;
    using Base::addPredicate;

    // ------------------- IREE-compatible predicates -------------------

    // Checks whether the linalg op's number of loops is at least `rank`.
    // EX: `minRank(2)` matches ops with rank >= 2 (2D+ elementwise, matmul, etc.).
    TorqStructuredOpMatcher &minRank(int64_t rank) {
        return addPredicate([rank](OpTy op) {
            return static_cast<int64_t>(cast<linalg::LinalgOp>(op.getOperation()).getNumLoops()) >=
                   rank;
        });
    }

    // Checks whether every loop iterator is `parallel`.
    TorqStructuredOpMatcher &allParallel() {
        return addPredicate([](OpTy op) {
            auto types = cast<linalg::LinalgOp>(op.getOperation()).getIteratorTypesArray();
            return llvm::all_of(types, [](utils::IteratorType iter) {
                return iter == utils::IteratorType::parallel;
            });
        });
    }

    // Checks whether the number of DPS input operands equals `n`.
    // EX: `numInputs(1)` → unary (sigmoid/gelu/relu); `numInputs(2)` → binary (add/mul) or
    // conv with `ins=[input, filter]`; `numInputs(3)` → conv with bias `ins=[input, filter,
    // bias]`.
    TorqStructuredOpMatcher &numInputs(int64_t n) {
        return addPredicate([n](OpTy op) {
            return cast<linalg::LinalgOp>(op.getOperation()).getNumDpsInputs() == n;
        });
    }

    // Checks whether the number of DPS output (init) operands equals `n`.
    // EX: `numOutputs(1)` → single result tensor; `numOutputs(2)` or more
    // applies only to multi-result `linalg.generic` (e.g., a decomposed softmax yielding max/sum
    // together).
    TorqStructuredOpMatcher &numOutputs(int64_t n) {
        return addPredicate([n](OpTy op) {
            return cast<linalg::LinalgOp>(op.getOperation()).getNumDpsInits() == n;
        });
    }

    // ------------------- Torq-specific predicates -------------------

    // Checks whether the element type of the `pos`-th input satisfies `pred`.
    // EX: `inputElementType(1, [](Type t){ return t.isBF16(); })` matches only when input 1 is
    // bf16.
    TorqStructuredOpMatcher &inputElementType(int64_t pos, llvm::function_ref<bool(Type)> pred) {
        return addPredicate([pos, pred](OpTy op) {
            auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
            auto tensorTy =
                dyn_cast<RankedTensorType>(linalgOp.getDpsInputOperand(pos)->get().getType());
            return tensorTy && pred(tensorTy.getElementType());
        });
    }

    // Checks whether the element type of the `pos`-th output (init) satisfies `pred`.
    // EX: `outputElementType(2, [](Type t){ return t.isF32(); })` matches only when output 2
    // is f32.
    TorqStructuredOpMatcher &outputElementType(int64_t pos, llvm::function_ref<bool(Type)> pred) {
        return addPredicate([pos, pred](OpTy op) {
            auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
            auto tensorTy =
                dyn_cast<RankedTensorType>(linalgOp.getDpsInitOperand(pos)->get().getType());
            return tensorTy && pred(tensorTy.getElementType());
        });
    }

    // Checks whether the body yield's op chain follows the order `YieldOps...`.
    // EX (sigmoid body): `yieldChain<arith::DivFOp, arith::AddFOp, math::ExpOp,
    // arith::NegFOp>()`.
    template <typename... YieldOps> TorqStructuredOpMatcher &yieldChain() {
        static_assert(
            std::is_same_v<OpTy, linalg::GenericOp>,
            "yieldChain<> only applies to linalg::GenericOp"
        );
        return addPredicate([](OpTy op) {
            auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
            if (yieldOp.getValues().empty()) {
                return false;
            }
            return matchYieldDagPath<YieldOps...>(yieldOp.getValues()[0]);
        });
    }
};

} // namespace mlir::syna::torq
