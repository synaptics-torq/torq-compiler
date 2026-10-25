// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-share-maxpool-stages"

namespace mlir::syna::torq {

namespace {

// A max pooling op and its parameters, the padding being given by the tensor.pad of its input
struct MaxPool {
    linalg::LinalgOp op;
    int hDim, wDim;                       // Spatial dims of the input
    Value source;                         // Input before padding
    SmallVector<int64_t> kernel;          // [kh, kw]
    SmallVector<int64_t> padLow, padHigh; // [h, w]
    Attribute padValue;                   // Null if not padded
    Attribute initValue;                  // Constant the output is filled with
    bool unitStride = false;
};

std::optional<MaxPool> getMaxPool(Operation *op) {
    MaxPool pool;
    DenseIntElementsAttr strides, dilations;
    if (auto nhwc = dyn_cast<linalg::PoolingNhwcMaxOp>(op)) {
        pool.hDim = 1;
        pool.wDim = 2;
        strides = nhwc.getStrides();
        dilations = nhwc.getDilations();
    }
    else if (auto nchw = dyn_cast<linalg::PoolingNchwMaxOp>(op)) {
        pool.hDim = 2;
        pool.wDim = 3;
        strides = nchw.getStrides();
        dilations = nchw.getDilations();
    }
    else {
        return std::nullopt;
    }
    pool.op = cast<linalg::LinalgOp>(op);
    if (!llvm::all_of(dilations.getValues<int64_t>(), [](int64_t d) { return d == 1; })) {
        return std::nullopt;
    }
    pool.unitStride = llvm::all_of(strides.getValues<int64_t>(), [](int64_t s) { return s == 1; });

    auto kernelType = dyn_cast<RankedTensorType>(pool.op.getDpsInputs()[1].getType());
    if (!kernelType || !kernelType.hasStaticShape() || kernelType.getRank() != 2) {
        return std::nullopt;
    }
    pool.kernel = SmallVector<int64_t>(kernelType.getShape());

    // The output must start from a constant (the lowest value of the type)
    auto fillOp = pool.op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fillOp || !matchPattern(fillOp.getInputs()[0], m_Constant(&pool.initValue))) {
        return std::nullopt;
    }

    Value input = pool.op.getDpsInputs()[0];
    pool.source = input;
    pool.padLow = {0, 0};
    pool.padHigh = {0, 0};
    if (auto padOp = input.getDefiningOp<tensor::PadOp>()) {
        Value padValue = padOp.getConstantPaddingValue();
        if (!padValue || !matchPattern(padValue, m_Constant(&pool.padValue))) {
            return std::nullopt;
        }
        auto low = padOp.getStaticLow();
        auto high = padOp.getStaticHigh();
        for (int i = 0; i < (int)low.size(); i++) {
            if (ShapedType::isDynamic(low[i]) || ShapedType::isDynamic(high[i])) {
                return std::nullopt;
            }
            if (i != pool.hDim && i != pool.wDim && (low[i] != 0 || high[i] != 0)) {
                return std::nullopt;
            }
        }
        pool.source = padOp.getSource();
        pool.padLow = {low[pool.hDim], low[pool.wDim]};
        pool.padHigh = {high[pool.hDim], high[pool.wDim]};
    }
    return pool;
}

// True if all the uses of the result of op are after pos (op can be moved after pos)
bool usesAfter(Operation *op, Operation *pos) {
    for (Operation *user : op->getUsers()) {
        Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user);
        if (!ancestor || !pos->isBeforeInBlock(ancestor)) {
            return false;
        }
    }
    return true;
}

// True if pool b can be computed from the output of pool a: a max over a window of size k_b is a
// max over windows of size k_a of a window of size k_b - k_a + 1, padded with p_b - p_a (at most
// k_b - k_a on each side so that windows partially in the padding see the same data)
bool canReuse(const MaxPool &a, const MaxPool &b) {
    linalg::LinalgOp aLinalg = a.op, bLinalg = b.op;
    Operation *aOp = aLinalg, *bOp = bLinalg;
    if (aOp->getName() != bOp->getName() || aOp->getBlock() != bOp->getBlock() || !a.unitStride) {
        return false;
    }
    // a must come first, or b must be movable after a
    if (!aOp->isBeforeInBlock(bOp) && !usesAfter(bOp, aOp)) {
        return false;
    }
    if (a.kernel[0] > b.kernel[0] || a.kernel[1] > b.kernel[1] || a.kernel == b.kernel) {
        return false;
    }
    if (a.initValue != b.initValue || getElementTypeOrSelf(aOp->getResult(0).getType()) !=
                                          getElementTypeOrSelf(a.source.getType())) {
        return false;
    }
    // Both paddings must use the same value, and b must have one if more padding is needed
    if (a.padValue && b.padValue != a.padValue) {
        return false;
    }
    bool needsPad = false;
    for (int i = 0; i < 2; i++) {
        int64_t maxPad = b.kernel[i] - a.kernel[i];
        int64_t low = b.padLow[i] - a.padLow[i];
        int64_t high = b.padHigh[i] - a.padHigh[i];
        if (a.padLow[i] >= a.kernel[i] || a.padHigh[i] >= a.kernel[i] || low < 0 || high < 0 ||
            low > maxPad || high > maxPad) {
            return false;
        }
        needsPad |= low > 0 || high > 0;
    }
    return !needsPad || b.padValue;
}

} // namespace

void shareMaxPoolStages(FunctionOpInterface funcOp, IRRewriter &rewriter) {
    llvm::MapVector<Value, SmallVector<MaxPool>> poolsBySource;
    funcOp.walk([&](Operation *op) {
        if (auto pool = getMaxPool(op)) {
            poolsBySource[pool->source].push_back(*pool);
        }
    });

    for (auto &entry : poolsBySource) {
        SmallVector<MaxPool> &pools = entry.second;
        // Decide on the original parameters before changing any pool: a pool rewritten to read
        // another one keeps computing the same result
        SmallVector<std::pair<size_t, size_t>> reuses;
        for (size_t b = 0; b < pools.size(); b++) {
            std::optional<size_t> best;
            for (size_t a = 0; a < pools.size(); a++) {
                if (canReuse(pools[a], pools[b]) &&
                    (!best || pools[a].kernel[0] * pools[a].kernel[1] >
                                  pools[*best].kernel[0] * pools[*best].kernel[1])) {
                    best = a;
                }
            }
            if (best) {
                reuses.push_back({*best, b});
            }
        }

        // Rewrite the smaller pools first, so that a pool is moved after the pool it reads once
        // that one is at its final position
        llvm::stable_sort(reuses, [&](auto x, auto y) {
            return pools[x.second].kernel[0] * pools[x.second].kernel[1] <
                   pools[y.second].kernel[0] * pools[y.second].kernel[1];
        });
        for (auto [a, b] : reuses) {
            const MaxPool &pa = pools[a];
            const MaxPool &pb = pools[b];
            linalg::LinalgOp aOp = pa.op, bOp = pb.op;
            Location loc = bOp->getLoc();
            if (!aOp->isBeforeInBlock(bOp)) {
                if (!usesAfter(bOp, aOp)) {
                    continue;
                }
                rewriter.moveOpAfter(bOp, aOp);
            }
            rewriter.setInsertionPoint(bOp);

            Value input = aOp->getResult(0);
            auto inputType = cast<RankedTensorType>(input.getType());
            SmallVector<int64_t> low(inputType.getRank(), 0), high(inputType.getRank(), 0);
            low[pb.hDim] = pb.padLow[0] - pa.padLow[0];
            low[pb.wDim] = pb.padLow[1] - pa.padLow[1];
            high[pb.hDim] = pb.padHigh[0] - pa.padHigh[0];
            high[pb.wDim] = pb.padHigh[1] - pa.padHigh[1];
            if (llvm::any_of(low, [](int64_t p) { return p != 0; }) ||
                llvm::any_of(high, [](int64_t p) { return p != 0; })) {
                auto resultType = tensor::PadOp::inferResultType(inputType, low, high);
                Value padValue =
                    arith::ConstantOp::create(rewriter, loc, cast<TypedAttr>(pb.padValue));
                input = tensor::PadOp::create(
                    rewriter, loc, resultType, input,
                    getAsIndexOpFoldResult(rewriter.getContext(), low),
                    getAsIndexOpFoldResult(rewriter.getContext(), high), padValue
                );
            }

            Value kernel = bOp.getDpsInputs()[1];
            auto kernelType = cast<RankedTensorType>(kernel.getType());
            Value newKernel = tensor::EmptyOp::create(
                rewriter, loc,
                SmallVector<int64_t>{
                    pb.kernel[0] - pa.kernel[0] + 1, pb.kernel[1] - pa.kernel[1] + 1
                },
                kernelType.getElementType()
            );

            LLVM_DEBUG(
                llvm::dbgs() << "Computing maxpool " << pb.kernel[0] << "x" << pb.kernel[1]
                             << " from maxpool " << pa.kernel[0] << "x" << pa.kernel[1] << "\n"
            );
            rewriter.modifyOpInPlace(bOp, [&]() {
                bOp.getDpsInputOperand(0)->set(input);
                bOp.getDpsInputOperand(1)->set(newKernel);
            });
        }
    }
}

} // namespace mlir::syna::torq
