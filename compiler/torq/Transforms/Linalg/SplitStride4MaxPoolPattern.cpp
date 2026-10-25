// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-split-stride4-maxpool"

namespace mlir::syna::torq {

namespace {

// The NSS only supports strides of 1 and 2. A kxk stride-4 max pooling is a kxk stride-2 max
// pooling of the same (padded) input followed by a 1x1 stride-2 max pooling: output o of the
// second one is output 2*o of the first one, whose window starts at input 4*o, the window of the
// stride-4 pooling. The output sizes also match: floor(floor(D / 2) / 2) == floor(D / 4) with D the
// padded input size minus the kernel size.
// Done on linalg, before tiling and the conversion to torq_hl, so that both poolings are handled
// like any other stride-2 max pooling.
template <typename PoolOp> class SplitStride4MaxPool : public OpRewritePattern<PoolOp> {
  public:
    using OpRewritePattern<PoolOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(PoolOp poolOp, PatternRewriter &rewriter) const override {
        constexpr bool isNchw = std::is_same_v<PoolOp, linalg::PoolingNchwMaxOp>;
        const int hDim = isNchw ? 2 : 1;
        const int wDim = isNchw ? 3 : 2;

        if (!poolOp.hasPureTensorSemantics() ||
            !llvm::all_of(
                poolOp.getStrides().template getValues<int64_t>(), [](int64_t s) { return s == 4; }
            ) ||
            !llvm::all_of(poolOp.getDilations().template getValues<int64_t>(), [](int64_t d) {
                return d == 1;
            })) {
            return rewriter.notifyMatchFailure(poolOp, "expected strides 4 and dilations 1");
        }
        Value input = poolOp.getDpsInputs()[0];
        Value kernel = poolOp.getDpsInputs()[1];
        auto inputType = dyn_cast<RankedTensorType>(input.getType());
        auto kernelType = dyn_cast<RankedTensorType>(kernel.getType());
        auto outputType = dyn_cast<RankedTensorType>(poolOp->getResult(0).getType());
        if (!inputType || !kernelType || !outputType || !inputType.hasStaticShape() ||
            !kernelType.hasStaticShape() || !outputType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(poolOp, "expected static shapes");
        }
        // Only when the NSS supports the stride-2 maxpool: otherwise the stride-4 maxpool is left
        // unchanged for CSS or Host. A stride-2 maxpool with a kernel larger than the HW limits is
        // then split in a chain of smaller maxpools (DecomposeLargeMaxPool), each converted to the
        // NSS only if it supports it.
        SmallVector<int64_t> kernelSize(kernelType.getShape());
        SmallVector<int64_t> pad = getMaxPoolInputPadding(input, hDim, wDim);
        bool isLargeKernel = kernelSize[0] > HwInfo::nss_max_kernel_height ||
                             kernelSize[1] > HwInfo::nss_max_kernel_width;
        if (!isLargeKernel &&
            getMaxPoolLowering(
                kernelSize, {2, 2}, pad, inputType.getDimSize(hDim), inputType.getDimSize(wDim)
            ) != MaxPoolLowering::Generic) {
            return rewriter.notifyMatchFailure(poolOp, "stride-2 maxpool not supported");
        }

        // The output must start from a constant, reused for the intermediate result
        auto fillOp = poolOp.getDpsInits()[0].template getDefiningOp<linalg::FillOp>();
        if (!fillOp) {
            return rewriter.notifyMatchFailure(poolOp, "expected a filled output");
        }

        SmallVector<int64_t> shape(outputType.getShape());
        shape[hDim] = (inputType.getDimSize(hDim) - kernelType.getDimSize(0)) / 2 + 1;
        shape[wDim] = (inputType.getDimSize(wDim) - kernelType.getDimSize(1)) / 2 + 1;
        if ((shape[hDim] - 1) / 2 + 1 != outputType.getDimSize(hDim) ||
            (shape[wDim] - 1) / 2 + 1 != outputType.getDimSize(wDim)) {
            return rewriter.notifyMatchFailure(poolOp, "unexpected output shape");
        }

        Location loc = poolOp.getLoc();
        Type elemType = outputType.getElementType();
        auto stride2 = rewriter.getI64VectorAttr({2, 2});
        Value init = tensor::EmptyOp::create(rewriter, loc, shape, elemType);
        init = linalg::FillOp::create(rewriter, loc, fillOp.getInputs(), ValueRange{init})
                   .getResult(0);
        auto stage1 = PoolOp::create(
            rewriter, loc, TypeRange{init.getType()}, ValueRange{input, kernel}, ValueRange{init},
            stride2, poolOp.getDilations()
        );
        Value unitKernel =
            tensor::EmptyOp::create(rewriter, loc, {1, 1}, kernelType.getElementType());
        rewriter.replaceOpWithNewOp<PoolOp>(
            poolOp, TypeRange{outputType}, ValueRange{stage1->getResult(0), unitKernel},
            ValueRange{poolOp.getDpsInits()[0]}, stride2, poolOp.getDilations()
        );
        return success();
    }
};

} // namespace

void populateSplitStride4MaxPoolPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<SplitStride4MaxPool<linalg::PoolingNchwMaxOp>>(context);
    patterns.add<SplitStride4MaxPool<linalg::PoolingNhwcMaxOp>>(context);
}

} // namespace mlir::syna::torq
