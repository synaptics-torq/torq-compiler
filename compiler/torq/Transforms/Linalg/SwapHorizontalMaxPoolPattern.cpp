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

#define DEBUG_TYPE "torq-swap-horizontal-maxpool"

namespace mlir::syna::torq {

namespace {

// The horizontal 1D maxpool kernel (kernel 1xk, stride k along W) reduces along the innermost
// dimension one output at a time, about one cycle per input element, while the vertical one (kx1,
// stride k along H) reduces whole vectors of W. A horizontal 1D max pooling is replaced with a
// transpose swapping H and W, the vertical max pooling and a transpose back. Only for elements of
// at most 16 bits (the transposes of 32-bit elements are as slow as the horizontal kernel) and an
// input height of at least 4 (the input height becomes the vector length of the vertical kernel;
// below 4 the swap is slower).
// Done on linalg, before the NHWC to NCHW conversion: the transposes compose with the layout
// transposes of the NHWC max poolings (OptimizeTransposeLayoutPass), so an NHWC horizontal max
// pooling becomes NHWC -> NCWH, the vertical max pooling and NCWH -> NHWC, with no more
// transposes than the original one.
template <typename PoolOp> class SwapHorizontalMaxPool : public OpRewritePattern<PoolOp> {
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
        Value input = poolOp.getDpsInputs()[0];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto kernelType = dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[1].getType());
        auto outputType = dyn_cast<RankedTensorType>(poolOp->getResult(0).getType());
        if (!inputType || !kernelType || !outputType || !inputType.hasStaticShape() ||
            !kernelType.hasStaticShape() || !outputType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected static shapes");
        }
        auto strides = SmallVector<int64_t>(poolOp.getStrides().template getValues<int64_t>());
        SmallVector<int64_t> kernel(kernelType.getShape());
        int64_t inH = inputType.getDimSize(hDim), inW = inputType.getDimSize(wDim);
        SmallVector<int64_t> pad = getMaxPoolInputPadding(input, hDim, wDim);
        if (getMaxPoolLowering(kernel, strides, pad, inH, inW) != MaxPoolLowering::Horizontal1D) {
            return rewriter.notifyMatchFailure(poolOp, "not a horizontal 1D max pooling");
        }
        constexpr int64_t minHeight = 4;
        if (inputType.getElementTypeBitWidth() > 16 || inH < minHeight) {
            return rewriter.notifyMatchFailure(poolOp, "the horizontal kernel is faster");
        }
        auto fillOp = poolOp.getDpsInits()[0].template getDefiningOp<linalg::FillOp>();
        if (!fillOp) {
            return rewriter.notifyMatchFailure(poolOp, "expected a filled output");
        }

        Location loc = poolOp.getLoc();
        SmallVector<int64_t> perm = {0, 1, 2, 3};
        std::swap(perm[hDim], perm[wDim]);
        auto transpose = [&](Value value) {
            auto type = cast<RankedTensorType>(value.getType());
            SmallVector<int64_t> shape(type.getShape());
            std::swap(shape[hDim], shape[wDim]);
            Value init = tensor::EmptyOp::create(rewriter, loc, shape, type.getElementType());
            return linalg::TransposeOp::create(rewriter, loc, value, init, perm)->getResult(0);
        };

        Value swappedInput = transpose(input);
        SmallVector<int64_t> swappedShape(outputType.getShape());
        std::swap(swappedShape[hDim], swappedShape[wDim]);
        Type elemType = outputType.getElementType();
        Value init = tensor::EmptyOp::create(rewriter, loc, swappedShape, elemType);
        init = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(), ValueRange{init})
                   .getResult(0);
        Value verticalKernel =
            tensor::EmptyOp::create(rewriter, loc, {kernel[1], 1}, kernelType.getElementType());
        auto vertical = PoolOp::create(
            rewriter, loc, TypeRange{init.getType()}, ValueRange{swappedInput, verticalKernel},
            ValueRange{init}, rewriter.getI64VectorAttr({strides[1], 1}), poolOp.getDilations()
        );
        rewriter.replaceOp(poolOp, transpose(vertical->getResult(0)));
        return success();
    }
};

} // namespace

void populateSwapHorizontalMaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<SwapHorizontalMaxPool<linalg::PoolingNchwMaxOp>>(context);
    patterns.add<SwapHorizontalMaxPool<linalg::PoolingNhwcMaxOp>>(context);
}

} // namespace mlir::syna::torq
