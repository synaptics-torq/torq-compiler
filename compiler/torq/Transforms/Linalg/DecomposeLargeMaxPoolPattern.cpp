// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-large-maxpool"

namespace mlir::syna::torq {

namespace {

// Split a kernel size k into the sizes of a chain of stride-1 windows covering the same k
// elements: sum(k_i - 1) == k - 1, in stageCount stages. The sizes are balanced and kept odd when k
// is odd (so that each stage can use symmetric "same" padding).
SmallVector<int64_t> splitPoolKernel(int64_t k, int64_t stageCount) {
    int64_t span = k - 1;
    // Distribute the span in units of 2 when possible to keep the stage sizes odd (some stages may
    // get size 1 when the kernel is smaller in this dimension)
    int64_t unit = span % 2 == 0 ? 2 : 1;
    int64_t units = span / unit;
    SmallVector<int64_t> sizes;
    for (int64_t i = 0; i < stageCount; i++) {
        int64_t stageUnits = units / stageCount + (i < units % stageCount ? 1 : 0);
        sizes.push_back(stageUnits * unit + 1);
    }
    return sizes;
}

// Distribute a total padding over the stages of a pooling chain, at most kernelSizes[i] - 1 per
// stage. The center position of each stage is filled first so that a symmetric "same" padding is
// split into symmetric "same" paddings.
SmallVector<int64_t> splitPoolPadding(int64_t pad, ArrayRef<int64_t> kernelSizes, bool isBegin) {
    SmallVector<int64_t> pads;
    for (int64_t k : kernelSizes) {
        int64_t center = isBegin ? (k - 1) / 2 : k - 1 - (k - 1) / 2;
        pads.push_back(std::min(pad, center));
        pad -= pads.back();
    }
    for (auto [i, k] : llvm::enumerate(kernelSizes)) {
        int64_t extra = std::min(pad, k - 1 - pads[i]);
        pads[i] += extra;
        pad -= extra;
    }
    assert(pad == 0 && "padding larger than the kernel");
    return pads;
}

// A max pooling with a kernel larger than what the NSS supports is replaced by a chain of max
// poolings with smaller kernels: a max over a k-wide window is a max over overlapping stride-1
// windows of sizes k_i with sum(k_i - 1) == k - 1 (e.g. 9 = 5 + 5 - 1, 13 = 7 + 7 - 1), the last
// stage applying the stride. Each stage pads its input with the padding value of the original
// pooling (lower than any data) and at most k_i - 1 elements on each side, so windows partially in
// the padding see the same data as the original one.
// Done on linalg, before tiling and the conversion to torq_hl: the stages are ordinary max poolings
// for tile-and-fuse (a chain of single-use max poolings is fused in one group) and the frontend.
template <typename PoolOp> class DecomposeLargeMaxPool : public OpRewritePattern<PoolOp> {
  public:
    using OpRewritePattern<PoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PoolOp poolOp, PatternRewriter &rewriter) const override {
        constexpr bool isNchw = std::is_same_v<PoolOp, linalg::PoolingNchwMaxOp>;
        const int hDim = isNchw ? 2 : 1;
        const int wDim = isNchw ? 3 : 2;
        const int64_t maxKh = HwInfo::nss_max_kernel_height;
        const int64_t maxKw = HwInfo::nss_max_kernel_width;

        auto strides = SmallVector<int64_t>(poolOp.getStrides().template getValues<int64_t>());
        if (!poolOp.hasPureTensorSemantics() || strides[0] != strides[1] || strides[0] > 2 ||
            !llvm::all_of(poolOp.getDilations().template getValues<int64_t>(), [](int64_t d) {
                return d == 1;
            })) {
            return rewriter.notifyMatchFailure(poolOp, "expected equal strides <= 2, dilation 1");
        }
        Value input = poolOp.getDpsInputs()[0];
        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[1].getType());
        auto outputType = dyn_cast<RankedTensorType>(poolOp->getResult(0).getType());
        if (!kernelType || !kernelType.hasStaticShape() || !outputType ||
            !outputType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected static shapes");
        }
        int64_t kh = kernelType.getDimSize(0), kw = kernelType.getDimSize(1);
        if (kh <= maxKh && kw <= maxKw) {
            return rewriter.notifyMatchFailure(poolOp, "kernel supported by the NSS");
        }

        // The output must start from a constant, reused for the intermediate results
        auto fillOp = poolOp.getDpsInits()[0].template getDefiningOp<linalg::FillOp>();
        if (!fillOp) {
            return rewriter.notifyMatchFailure(poolOp, "expected a filled output");
        }

        // Padding of the original pooling, smaller than the kernel
        Value source = input;
        Value padValue;
        SmallVector<int64_t> low(4, 0), high(4, 0);
        if (auto padOp = input.getDefiningOp<tensor::PadOp>()) {
            padValue = padOp.getConstantPaddingValue();
            auto staticLow = padOp.getStaticLow();
            auto staticHigh = padOp.getStaticHigh();
            for (int i = 0; i < 4; i++) {
                low[i] = staticLow[i];
                high[i] = staticHigh[i];
                bool spatial = i == hDim || i == wDim;
                if (ShapedType::isDynamic(low[i]) || ShapedType::isDynamic(high[i]) ||
                    (!spatial && (low[i] || high[i]))) {
                    return rewriter.notifyMatchFailure(poolOp, "unsupported padding");
                }
            }
            if (!padValue || low[hDim] >= kh || high[hDim] >= kh || low[wDim] >= kw ||
                high[wDim] >= kw) {
                return rewriter.notifyMatchFailure(poolOp, "unsupported padding");
            }
            source = padOp.getSource();
        }
        auto sourceType = dyn_cast<RankedTensorType>(source.getType());
        if (!sourceType || !sourceType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected a static input");
        }

        int64_t stageCount =
            std::max(llvm::divideCeil(kh - 1, maxKh - 1), llvm::divideCeil(kw - 1, maxKw - 1));
        auto khs = splitPoolKernel(kh, stageCount);
        auto kws = splitPoolKernel(kw, stageCount);
        auto padTop = splitPoolPadding(low[hDim], khs, true);
        auto padBottom = splitPoolPadding(high[hDim], khs, false);
        auto padLeft = splitPoolPadding(low[wDim], kws, true);
        auto padRight = splitPoolPadding(high[wDim], kws, false);

        Location loc = poolOp.getLoc();
        Type elemType = outputType.getElementType();
        Type kernelElemType = kernelType.getElementType();
        Value stageInput = source;
        SmallVector<int64_t> shape(sourceType.getShape());
        for (int64_t i = 0; i < stageCount; i++) {
            bool isLast = i == stageCount - 1;
            int64_t s = isLast ? strides[0] : 1;
            if (padTop[i] || padBottom[i] || padLeft[i] || padRight[i]) {
                SmallVector<int64_t> stageLow(4, 0), stageHigh(4, 0);
                stageLow[hDim] = padTop[i];
                stageHigh[hDim] = padBottom[i];
                stageLow[wDim] = padLeft[i];
                stageHigh[wDim] = padRight[i];
                auto paddedType = tensor::PadOp::inferResultType(
                    cast<RankedTensorType>(stageInput.getType()), stageLow, stageHigh
                );
                stageInput = tensor::PadOp::create(
                    rewriter, loc, paddedType, stageInput,
                    getAsIndexOpFoldResult(rewriter.getContext(), stageLow),
                    getAsIndexOpFoldResult(rewriter.getContext(), stageHigh), padValue
                );
            }
            shape[hDim] = (shape[hDim] + padTop[i] + padBottom[i] - khs[i]) / s + 1;
            shape[wDim] = (shape[wDim] + padLeft[i] + padRight[i] - kws[i]) / s + 1;

            Value init;
            if (isLast) {
                if (RankedTensorType::get(shape, elemType) != outputType) {
                    return rewriter.notifyMatchFailure(poolOp, "unexpected output shape");
                }
                init = poolOp.getDpsInits()[0];
            }
            else {
                init = tensor::EmptyOp::create(rewriter, loc, shape, elemType);
                init = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(), ValueRange{init})
                           .getResult(0);
            }
            Value kernel = tensor::EmptyOp::create(rewriter, loc, {khs[i], kws[i]}, kernelElemType);
            auto stage = PoolOp::create(
                rewriter, loc, TypeRange{init.getType()}, ValueRange{stageInput, kernel},
                ValueRange{init}, rewriter.getI64VectorAttr({s, s}), poolOp.getDilations()
            );
            stageInput = stage->getResult(0);
        }
        rewriter.replaceOp(poolOp, stageInput);
        return success();
    }
};

} // namespace

void populateDecomposeLargeMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<DecomposeLargeMaxPool<linalg::PoolingNchwMaxOp>>(context);
    patterns.add<DecomposeLargeMaxPool<linalg::PoolingNhwcMaxOp>>(context);
}

} // namespace mlir::syna::torq
