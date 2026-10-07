// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RaiseIntMeanToPooling
// ---------------------
// TOSA imports an int8 TFLite MEAN over H and W (global average pooling) as
//   w = generic: extsi(x) - input_zp, identity apply_scale    i8 [N, H, W, C] -> i32
//   s = reduce(w) over H and W, with unit-dim expand_shapes   -> i32 [N, 1, 1, C]
//   q = generic: apply_scale(s, m, shift) + output_zp, clamp  -> i8 [N, 1, 1, C]
// The reductions then run as separate int32 ops with the widened tensor in memory.
//
// This raise rewrites the chain as a whole-frame pooling_nhwc_sum over the int8
// input followed by the same requantization. The input zero point moves after the
// sum, sum(x - zp) = sum(x) - H*W*zp, as the subtraction the forward scale fold
// reads as a bias. The pool then lowers to the average pool kernel.

#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::syna::torq {

namespace {

// Returns the element type of the single result when `op` is an elementwise generic
// on tensors with one input and one init. The maps must be identities, because the
// caller reads the dims of the input as the dims of the result.
static Type getUnaryElementwiseResultType(linalg::GenericOp op) {
    if (!op.hasPureTensorSemantics() || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        !linalg::isElementwise(op) ||
        !llvm::all_of(op.getIndexingMapsArray(), [](AffineMap map) { return map.isIdentity(); }))
        return {};
    return getElementTypeOrSelf(op.getResultTypes()[0]);
}

// Returns the int8 [N, H, W, C] input of `widen` when it computes
// extsi(x) - zp followed by an optional identity apply_scale (multiplier 1 << shift).
static Value matchWiden(linalg::GenericOp widen, int64_t &zp) {
    Type resultType = getUnaryElementwiseResultType(widen);
    if (!resultType || !resultType.isInteger(32))
        return {};
    Value input = widen.getDpsInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() != 4 || !inputType.getElementType().isInteger(8))
        return {};

    Value value = cast<linalg::YieldOp>(widen.getBody()->getTerminator()).getValues()[0];
    if (auto applyScale = value.getDefiningOp<tosa::ApplyScaleOp>()) {
        auto multiplier = getConstIntValue(applyScale.getMultiplier());
        auto shift = getConstIntValue(applyScale.getShift());
        if (!multiplier || !shift || *shift < 0 || *shift > 30 || *multiplier != (1 << *shift))
            return {};
        value = applyScale.getValue();
    }
    zp = 0;
    if (auto sub = value.getDefiningOp<arith::SubIOp>()) {
        auto subZp = getConstIntValue(sub.getRhs());
        if (!subZp)
            return {};
        zp = *subZp;
        value = sub.getLhs();
    }
    auto extend = value.getDefiningOp<arith::ExtSIOp>();
    if (!extend || extend.getIn() != widen.getBody()->getArgument(0))
        return {};
    return input;
}

// True for a linalg.reduce that sums its single i32 input into a zero fill.
static bool isZeroInitSum(linalg::ReduceOp reduce) {
    if (reduce.getNumDpsInputs() != 1 || reduce.getNumDpsInits() != 1)
        return false;
    auto fill = reduce.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fill || getConstIntValue(fill.getInputs()[0]) != 0)
        return false;
    Block &body = reduce.getCombiner().front();
    auto add = body.getTerminator()->getOperand(0).getDefiningOp<arith::AddIOp>();
    return add && body.getOperations().size() == 2 &&
           llvm::is_contained(body.getArguments(), add.getLhs()) &&
           llvm::is_contained(body.getArguments(), add.getRhs());
}

// Follows `value` [N, H, W, C] forward through single-use summing reduces and
// unit-dim expand_shapes. `dims` holds the original dim of each current dim, or
// -1 for an inserted unit dim. Returns the last value of the chain, and sets
// `reducesHW` when the chain sums exactly H and W into [N, 1, 1, C].
static Value followSumChain(Value value, bool &reducesHW) {
    SmallVector<int64_t> dims = {0, 1, 2, 3};
    SmallVector<int64_t> reduced;
    while (value.hasOneUse()) {
        Operation *user = *value.getUsers().begin();
        if (auto reduce = dyn_cast<linalg::ReduceOp>(user)) {
            if (!isZeroInitSum(reduce))
                break;
            SmallVector<int64_t> kept;
            for (auto [index, dim] : llvm::enumerate(dims)) {
                if (llvm::is_contained(reduce.getDimensions(), index))
                    reduced.push_back(dim);
                else
                    kept.push_back(dim);
            }
            dims = kept;
            value = reduce->getResult(0);
            continue;
        }
        auto expand = dyn_cast<tensor::ExpandShapeOp>(user);
        if (!expand)
            break;
        ArrayRef<int64_t> shape = expand.getResultType().getShape();
        SmallVector<int64_t> expanded;
        for (auto [source, group] : llvm::enumerate(expand.getReassociationIndices())) {
            // Each group holds the source dim and unit dims around it.
            int64_t sourceSize = expand.getSrcType().getDimSize(source);
            if (llvm::count_if(group, [&](int64_t i) { return shape[i] == sourceSize; }) != 1)
                return {};
            for (int64_t index : group)
                expanded.push_back(shape[index] == sourceSize ? dims[source] : -1);
        }
        dims = expanded;
        value = expand.getResult();
    }
    llvm::sort(reduced);
    reducesHW = dims == SmallVector<int64_t>{0, -1, -1, 3} && reduced == SmallVector<int64_t>{1, 2};
    return value;
}

// True for an elementwise i32 -> i8 requantization whose body applies a constant
// scalar apply_scale to its input, the form the forward scale fold reads.
static bool isScalarRequant(linalg::GenericOp requant) {
    Type resultType = getUnaryElementwiseResultType(requant);
    if (!resultType || !resultType.isInteger(8))
        return false;
    tosa::ApplyScaleOp applyScale;
    requant.getBody()->walk([&](tosa::ApplyScaleOp op) { applyScale = op; });
    return applyScale && applyScale.getValue() == requant.getBody()->getArgument(0) &&
           getConstIntValue(applyScale.getMultiplier()) && getConstIntValue(applyScale.getShift());
}

struct RaiseIntMeanToPooling : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::GenericOp widen, PatternRewriter &rewriter) const override {
        int64_t zp;
        Value input = matchWiden(widen, zp);
        if (!input)
            return rewriter.notifyMatchFailure(widen, "not an int8 -> int32 widening");

        bool reducesHW = false;
        Value sum = followSumChain(widen.getResult(0), reducesHW);
        if (!reducesHW)
            return rewriter.notifyMatchFailure(widen, "not a sum over H and W");

        auto requant = dyn_cast_or_null<linalg::GenericOp>(
            sum.hasOneUse() ? *sum.getUsers().begin() : nullptr
        );
        if (!requant || !isScalarRequant(requant))
            return rewriter.notifyMatchFailure(widen, "sum is not requantized to int8");

        // The requantization init is defined after the widening.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(requant);
        Location loc = requant.getLoc();
        auto inputShape = cast<RankedTensorType>(input.getType()).getShape();
        auto sumType = cast<RankedTensorType>(sum.getType());
        Value zero = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
        Value init = linalg::FillOp::create(
                         rewriter, loc, ValueRange{zero},
                         ValueRange{tensor::EmptyOp::create(
                             rewriter, loc, sumType.getShape(), sumType.getElementType()
                         )}
        )
                         .getResult(0);
        Value window = tensor::EmptyOp::create(
            rewriter, loc, ArrayRef<int64_t>{inputShape[1], inputShape[2]}, sumType.getElementType()
        );
        auto unit = rewriter.getI64VectorAttr({1, 1});
        Value pool = linalg::PoolingNhwcSumOp::create(
                         rewriter, loc, TypeRange{sumType}, ValueRange{input, window},
                         ValueRange{init}, unit, unit
        )
                         .getResult(0);

        auto raised = linalg::GenericOp::create(
            rewriter, loc, requant.getResultTypes(), ValueRange{pool}, requant.getDpsInits(),
            requant.getIndexingMapsArray(), requant.getIteratorTypesArray()
        );
        rewriter.inlineRegionBefore(
            requant.getRegion(), raised.getRegion(), raised.getRegion().end()
        );

        int64_t frameZp = inputShape[1] * inputShape[2] * zp;
        if (frameZp != 0) {
            Block *body = raised.getBody();
            rewriter.setInsertionPointToStart(body);
            Value in = body->getArgument(0);
            Value bias =
                arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(frameZp));
            auto unbiased = arith::SubIOp::create(rewriter, loc, in, bias);
            rewriter.replaceAllUsesExcept(in, unbiased.getResult(), unbiased);
        }
        rewriter.replaceOp(requant, raised.getResults());
        return success();
    }
};

} // namespace

void populateRaiseIntMeanToPoolingPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseIntMeanToPooling>(context);
}

} // namespace mlir::syna::torq
