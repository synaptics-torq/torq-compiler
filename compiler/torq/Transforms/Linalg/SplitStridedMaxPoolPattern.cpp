// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-split-strided-maxpool"

namespace mlir::syna::torq {

namespace {

// The NSS only supports equal strides of 1 or 2 and the exact 1D max poolings (kernel == stride
// along one axis, kernel 1 and stride 1 along the other, no padding). Along an axis, a max over a
// window of size k with stride s >= 1 and k >= s is a max over s consecutive windows of size
// k - s + 1: [s*o, s*o + k - 1] is the union of [s*o + j, s*o + j + k - s] for j in [0, s). So a
// khxkw max pooling with strides [sh, sw] is:
// - a (kh-sh+1)x(kw-sw+1) stride-1 max pooling of the same (padded) input, skipped when it is 1x1;
// - then a vertical shx1 max pooling with strides [sh, 1] if sh > 1;
// - then a horizontal 1xsw max pooling with strides [1, sw] if sw > 1 (lowered with the vertical
//   kernel between two transposes when that is faster, see SwapHorizontalMaxPool).
// The 1D stages are exact (kernel == stride, no padding) when the stride-1 result size is a
// multiple of the stride, i.e. when (D - k) is a multiple of s with D the padded input size, which
// also makes the output sizes match.
// For example a 5x5 stride-3 max pooling becomes 3x3 stride 1, 3x1 stride [3, 1] and 1x3 stride
// [1, 3], and a 3x3 stride-[2, 1] one 2x3 stride 1 and 2x1 stride [2, 1].
// Only for the max poolings that the NSS doesn't support, other than equal strides of 4 (see
// SplitStride4MaxPool).
template <typename PoolOp> class SplitStridedMaxPool : public OpRewritePattern<PoolOp> {
  public:
    using OpRewritePattern<PoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PoolOp poolOp, PatternRewriter &rewriter) const override {
        constexpr bool isNchw = std::is_same_v<PoolOp, linalg::PoolingNchwMaxOp>;
        const int hDim = isNchw ? 2 : 1;
        const int wDim = isNchw ? 3 : 2;

        if (!poolOp.hasPureTensorSemantics() ||
            !llvm::all_of(poolOp.getDilations().template getValues<int64_t>(), [](int64_t d) {
                return d == 1;
            })) {
            return rewriter.notifyMatchFailure(poolOp, "expected dilations 1");
        }
        auto strides = SmallVector<int64_t>(poolOp.getStrides().template getValues<int64_t>());
        int64_t sh = strides[0], sw = strides[1];
        if (sh == sw && (sh <= 2 || sh == 4)) {
            return rewriter.notifyMatchFailure(poolOp, "strides handled by other patterns");
        }
        Value input = poolOp.getDpsInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[1].getType());
        auto outputType = dyn_cast<RankedTensorType>(poolOp->getResult(0).getType());
        if (!inputType || !kernelType || !outputType || !inputType.hasStaticShape() ||
            !kernelType.hasStaticShape() || !outputType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected static shapes");
        }
        int64_t kh = kernelType.getDimSize(0), kw = kernelType.getDimSize(1);
        int64_t inH = inputType.getDimSize(hDim), inW = inputType.getDimSize(wDim);
        SmallVector<int64_t> pad = getMaxPoolInputPadding(input, hDim, wDim);
        if (getMaxPoolLowering({kh, kw}, strides, pad, inH, inW) != MaxPoolLowering::None) {
            return rewriter.notifyMatchFailure(poolOp, "maxpool supported by the NSS");
        }
        if (kh < sh || kw < sw) {
            return rewriter.notifyMatchFailure(poolOp, "windows don't cover the input");
        }
        if ((inH - kh) % sh != 0 || (inW - kw) % sw != 0) {
            return rewriter.notifyMatchFailure(
                poolOp, "padded input size not a multiple of the stride"
            );
        }
        // The stride-1 stage reads the padding, which must be smaller than its kernel (no padding
        // at all when it is skipped)
        int64_t kh1 = kh - sh + 1, kw1 = kw - sw + 1;
        if (llvm::any_of(pad, [](int64_t p) { return p < 0; }) || pad[0] >= kw1 || pad[1] >= kw1 ||
            pad[2] >= kh1 || pad[3] >= kh1) {
            return rewriter.notifyMatchFailure(poolOp, "unsupported padding");
        }
        if ((kh1 > 1 || kw1 > 1) &&
            getMaxPoolLowering({kh1, kw1}, {1, 1}, pad, inH, inW) != MaxPoolLowering::Generic) {
            return rewriter.notifyMatchFailure(poolOp, "stride-1 stage not supported by the NSS");
        }

        // The output must start from a constant, reused for the intermediate results
        auto fillOp = poolOp.getDpsInits()[0].template getDefiningOp<linalg::FillOp>();
        if (!fillOp) {
            return rewriter.notifyMatchFailure(poolOp, "expected a filled output");
        }

        struct Stage {
            int64_t kh, kw, sh, sw;
            SmallVector<int64_t> shape;
        };
        SmallVector<Stage> stages;
        if (kh1 > 1 || kw1 > 1) {
            stages.push_back({kh1, kw1, 1, 1, {}});
        }
        if (sh > 1) {
            stages.push_back({sh, 1, sh, 1, {}});
        }
        if (sw > 1) {
            stages.push_back({1, sw, 1, sw, {}});
        }
        SmallVector<int64_t> shape(outputType.getShape());
        shape[hDim] = inH;
        shape[wDim] = inW;
        for (Stage &stage : stages) {
            shape[hDim] = (shape[hDim] - stage.kh) / stage.sh + 1;
            shape[wDim] = (shape[wDim] - stage.kw) / stage.sw + 1;
            stage.shape = shape;
        }
        if (stages.size() < 2 || shape != SmallVector<int64_t>(outputType.getShape())) {
            return rewriter.notifyMatchFailure(poolOp, "unexpected output shape");
        }

        Location loc = poolOp.getLoc();
        Type elemType = outputType.getElementType();
        Type kernelElemType = kernelType.getElementType();
        Value stageInput = input;
        for (auto [i, stage] : llvm::enumerate(stages)) {
            Value init;
            if (i == stages.size() - 1) {
                init = poolOp.getDpsInits()[0];
            }
            else {
                init = tensor::EmptyOp::create(rewriter, loc, stage.shape, elemType);
                init = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(), ValueRange{init})
                           .getResult(0);
            }
            Value kernel =
                tensor::EmptyOp::create(rewriter, loc, {stage.kh, stage.kw}, kernelElemType);
            auto stageOp = PoolOp::create(
                rewriter, loc, TypeRange{init.getType()}, ValueRange{stageInput, kernel},
                ValueRange{init}, rewriter.getI64VectorAttr({stage.sh, stage.sw}),
                poolOp.getDilations()
            );
            stageInput = stageOp->getResult(0);
        }
        rewriter.replaceOp(poolOp, stageInput);
        return success();
    }
};

} // namespace

void populateSplitStridedMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<SplitStridedMaxPool<linalg::PoolingNchwMaxOp>>(context);
    patterns.add<SplitStridedMaxPool<linalg::PoolingNhwcMaxOp>>(context);
}

} // namespace mlir::syna::torq
