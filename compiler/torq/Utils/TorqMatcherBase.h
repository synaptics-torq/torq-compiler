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
    TorqStructuredOpMatcher &minRank(int64_t rank) {
        return addPredicate([rank](OpTy op) {
            return static_cast<int64_t>(cast<linalg::LinalgOp>(op.getOperation()).getNumLoops()) >=
                   rank;
        });
    }

    TorqStructuredOpMatcher &allParallel() {
        return addPredicate([](OpTy op) {
            auto types = cast<linalg::LinalgOp>(op.getOperation()).getIteratorTypesArray();
            return llvm::all_of(types, [](utils::IteratorType iter) {
                return iter == utils::IteratorType::parallel;
            });
        });
    }

    TorqStructuredOpMatcher &numInputs(int64_t n) {
        return addPredicate([n](OpTy op) {
            return cast<linalg::LinalgOp>(op.getOperation()).getNumDpsInputs() == n;
        });
    }

    TorqStructuredOpMatcher &numOutputs(int64_t n) {
        return addPredicate([n](OpTy op) {
            return cast<linalg::LinalgOp>(op.getOperation()).getNumDpsInits() == n;
        });
    }

    // ------------------- Torq-specific predicates -------------------
    TorqStructuredOpMatcher &inputElementType(int64_t pos, llvm::function_ref<bool(Type)> pred) {
        return addPredicate([pos, pred](OpTy op) {
            auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
            auto tensorTy =
                dyn_cast<RankedTensorType>(linalgOp.getDpsInputOperand(pos)->get().getType());
            return tensorTy && pred(tensorTy.getElementType());
        });
    }

    TorqStructuredOpMatcher &outputElementType(int64_t pos, llvm::function_ref<bool(Type)> pred) {
        return addPredicate([pos, pred](OpTy op) {
            auto linalgOp = cast<linalg::LinalgOp>(op.getOperation());
            auto tensorTy =
                dyn_cast<RankedTensorType>(linalgOp.getDpsInitOperand(pos)->get().getType());
            return tensorTy && pred(tensorTy.getElementType());
        });
    }

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
