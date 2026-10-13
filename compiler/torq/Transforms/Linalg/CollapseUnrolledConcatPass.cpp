// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// CollapseUnrolledConcat Pass
//===----------------------------------------------------------------------===//
//
// Collapses "unrolled stack" concats back into their source tensor.
//
// Some model emitters lower `stack(unbind(V, d))` into a single
// `tensor.concat dim(d)` with one operand per element of V, each a unit
// `tensor.extract_slice` of V. That is an exact identity on V over
// the slice run, but it bloats the graph (one extract per element) and the
// CPU codegen materializes it as one small strided copy per element into a
// local buffer before the final copy out.
//
// Each maximal run of two or more consecutive unit slices of a common source
// (ascending starts, step 1) is rewritten to the source tensor itself when
// the run covers the whole source extent on the concat dim, or to a single
// extract_slice of that extent otherwise. Non-slice operands keep their
// place and order. The match is shape-proven: static types only, unit
// extent + zero offset + stride 1 on every dim except the concat dim, so
// the rewrite fires only when it is value-identical.
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-collapse-unrolled-concat"

namespace mlir::syna::torq {

enum class CollapseUnrolledConcatMode { Enable, Disable };

static llvm::cl::opt<CollapseUnrolledConcatMode> clCollapseUnrolledConcatMode(
    "torq-collapse-unrolled-concat-mode",
    llvm::cl::desc("Choose how to handle high-fan-in concats of unit slices"),
    llvm::cl::values(
        clEnumValN(
            CollapseUnrolledConcatMode::Enable, "enable",
            "Rewrite each maximal run of consecutive unit extract_slices of a "
            "common source feeding a concat (>= 32 operands) into the source "
            "tensor (full cover) or a single extract of the run's extent. "
            "Value-preserving normalization; on by default"
        ),
        clEnumValN(
            CollapseUnrolledConcatMode::Disable, "disable",
            "Leave such concats as is (one extract_slice and one copy per "
            "element)"
        )
    ),
    llvm::cl::init(CollapseUnrolledConcatMode::Enable)
);

namespace {

// Only consider concats with at least this many operands: the unrolled-stack
// artifact shows up at high fan-in (one operand per element of the stack),
// and small concats are legitimate data flow.
constexpr int64_t kMinFanin = 32;

// A unit-slice classification of one concat operand.
struct UnitSlice {
    Value source;
    int64_t start; // offset on the concat dim
};

// True when `slice` extracts a single element at `start` along dim `d` of its
// static source: unit size on d, zero offset, full size, stride 1 on every
// other dim, same rank as the source.
std::optional<UnitSlice> classifyUnitSlice(Value operand, tensor::ExtractSliceOp slice, int64_t d) {
    auto srcTy = dyn_cast<RankedTensorType>(slice.getSource().getType());
    auto dstTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!srcTy || !dstTy || !srcTy.hasStaticShape())
        return std::nullopt;
    if (srcTy.getRank() != dstTy.getRank() || d < 0 || d >= dstTy.getRank())
        return std::nullopt;
    ArrayRef<int64_t> strides = slice.getStaticStrides();
    ArrayRef<int64_t> offsets = slice.getStaticOffsets();
    ArrayRef<int64_t> sizes = slice.getStaticSizes();
    if (!llvm::all_of(strides, [](int64_t s) { return s == 1; }))
        return std::nullopt;
    for (int64_t k = 0; k < dstTy.getRank(); ++k) {
        int64_t srcDim = srcTy.getDimSize(k);
        if (k == d) {
            if (sizes[k] != 1)
                return std::nullopt;
        }
        else {
            if (offsets[k] != 0 || sizes[k] != srcDim)
                return std::nullopt;
        }
    }
    int64_t start = offsets[d];
    if (start < 0 || start >= srcTy.getDimSize(d))
        return std::nullopt;
    // The operand type must be exactly the source type with dim d -> 1.
    if (dstTy.getElementType() != srcTy.getElementType())
        return std::nullopt;
    for (int64_t k = 0; k < dstTy.getRank(); ++k) {
        if (dstTy.getDimSize(k) != (k == d ? 1 : srcTy.getDimSize(k)))
            return std::nullopt;
    }
    return UnitSlice{slice.getSource(), start};
}

class CollapseUnrolledConcatPattern : public OpRewritePattern<tensor::ConcatOp> {
  public:
    explicit CollapseUnrolledConcatPattern(MLIRContext *ctx)
        : OpRewritePattern<tensor::ConcatOp>(ctx) {
        setDebugName("CollapseUnrolledConcatPattern");
    }

    LogicalResult
    matchAndRewrite(tensor::ConcatOp concat, PatternRewriter &rewriter) const override {
        auto resultTy = dyn_cast<RankedTensorType>(concat.getResult().getType());
        if (!resultTy || !resultTy.hasStaticShape())
            return failure();
        ValueRange operands = concat.getOperands();
        if (operands.size() < kMinFanin)
            return failure();
        int64_t d = concat.getDim();
        if (d < 0 || d >= resultTy.getRank())
            return failure();

        // Classify each operand; a "run" is a maximal stretch of adjacent
        // unit slices of the same source with consecutive starts.
        SmallVector<std::optional<UnitSlice>, 8> classified(operands.size());
        for (auto [i, operand] : llvm::enumerate(operands)) {
            auto slice = operand.getDefiningOp<tensor::ExtractSliceOp>();
            if (!slice)
                continue;
            classified[i] = classifyUnitSlice(operand, slice, d);
        }

        SmallVector<Value> newOperands;
        newOperands.reserve(operands.size());
        bool collapsedAny = false;
        int64_t i = 0;
        while (i < operands.size()) {
            if (!classified[i]) {
                newOperands.push_back(operands[i]);
                ++i;
                continue;
            }
            Value source = classified[i]->source;
            int64_t start = classified[i]->start;
            int64_t j = i + 1;
            while (j < operands.size() && classified[j] && classified[j]->source == source &&
                   classified[j]->start == start + 1) {
                start = classified[j]->start;
                ++j;
            }
            if (j - i < 2) {
                newOperands.push_back(operands[i]);
                ++i;
                continue;
            }
            collapsedAny = true;
            auto srcTy = cast<RankedTensorType>(source.getType());
            int64_t s0 = classified[i]->start;
            int64_t s1 = start + 1; // exclusive
            Value replacement;
            if (s0 == 0 && s1 == srcTy.getDimSize(d)) {
                replacement = source;
            }
            else {
                SmallVector<int64_t> sizes(srcTy.getRank());
                for (int64_t k = 0; k < srcTy.getRank(); ++k)
                    sizes[k] = srcTy.getDimSize(k);
                sizes[d] = s1 - s0;
                auto partialTy =
                    RankedTensorType::get(sizes, srcTy.getElementType(), srcTy.getEncoding());
                SmallVector<OpFoldResult> off(srcTy.getRank(), rewriter.getIndexAttr(0)),
                    sz(srcTy.getRank()), st(srcTy.getRank(), rewriter.getIndexAttr(1));
                for (int64_t k = 0; k < srcTy.getRank(); ++k)
                    sz[k] = rewriter.getIndexAttr(sizes[k]);
                off[d] = rewriter.getIndexAttr(s0);
                replacement = tensor::ExtractSliceOp::create(
                                  rewriter, concat.getLoc(), partialTy, source, off, sz, st
                )
                                  .getResult();
            }
            newOperands.push_back(replacement);
            i = j;
        }
        if (!collapsedAny)
            return failure();

        tensor::ConcatOp newConcat = tensor::ConcatOp::create(
            rewriter, concat.getLoc(), resultTy, d, ValueRange(newOperands)
        );
        rewriter.replaceOp(concat, newConcat.getResult());

        LLVM_DEBUG(
            llvm::dbgs() << "[torq-collapse-unrolled-concat] collapsed concat dim(" << d << "): "
                         << operands.size() << " -> " << newOperands.size() << " operands\n"
        );
        return success();
    }
};

class CollapseUnrolledConcatPass
    : public impl::CollapseUnrolledConcatBase<CollapseUnrolledConcatPass> {
  public:
    using CollapseUnrolledConcatBase::CollapseUnrolledConcatBase;

    void runOnOperation() override {
        if (clCollapseUnrolledConcatMode != CollapseUnrolledConcatMode::Enable)
            return;
        auto funcOp = getOperation();
        RewritePatternSet patterns(&getContext());
        patterns.insert(std::make_unique<CollapseUnrolledConcatPattern>(&getContext()));
        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);
        if (failed(applyPatternsGreedily(funcOp, frozenPatterns)))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCollapseUnrolledConcatPass() {
    return std::make_unique<CollapseUnrolledConcatPass>();
}

} // namespace mlir::syna::torq
