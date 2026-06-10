// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// DecomposeConvToSpaceToDepth Pass
//===----------------------------------------------------------------------===//
//
// Decomposes linalg.conv_2d_nchw_fchw ops with large kernels (kH > 7 or
// kW > 7 or stride > 2) into an equivalent SpaceToDepth reshape followed by a convolution
// with a smaller kernel that fits within the HW limit (max 7x7, stride <= 2).
//
// Block sizes bH and bW are chosen independently per axis as the largest
// divisor of (kDim, stride, inputDim) such that kDim/b <= 7 and stride/b <= 2.
// The pattern fires only when at least one axis requires decomposition.
//
// PRECONDITIONS (per axis, independently for H and W):
//   - kDim % b == 0  AND  stride % b == 0  AND  inputDim % b == 0
//   - kDim / b <= 7   AND  stride / b <= 2
//   - dilation == 1  (dilated convolutions not supported)
//
// TRANSFORMATION:
//
//   BEFORE:
//     input  : [N, C,   H,   W  ]   (NCHW)
//     filter : [F, C,   kH,  kW ]   (FCHW)
//     conv stride [sH, sW]
//     output : [N, F,   oH,  oW ]
//
//   AFTER:
//     s2d_input  = SpaceToDepth(input,  bH, bW) : [N, C*bH*bW, H/bH,  W/bW ]
//     s2d_filter = SpaceToDepth(filter, bH, bW) : [F, C*bH*bW, kH/bH, kW/bW]
//     conv(s2d_input, s2d_filter, stride=[sH/bH, sW/bW]) : [N, F, oH, oW]
//
// SpaceToDepth is implemented as expand -> transpose -> collapse:
//   [N, C, H, W]
//     -> expand  [N, C, H/bH, bH, W/bW, bW]
//     -> transpose perm [0,1,3,5,2,4]
//        -> [N, C, bH, bW, H/bH, W/bW]
//     -> collapse [N, C*bH*bW, H/bH, W/bW]
//
// The same transform applied to both input and filter ensures the channel
// ordering is consistent and the convolution result is numerically identical.
//
// EXAMPLE:
//   input   [1, 1, 40, 80],  filter [64, 1, 4, 10],  stride [1, 2]
//   bH = 1  (kH=4 already <= 7, no height decomposition)
//   bW = 2  (kW=10 > 7; largest b dividing kW=10, sW=2, W=80 with kW/b<=7)
//   s2d_input  [1, 2, 40, 40],  s2d_filter [64, 2, 4, 5],  stride [1, 1]
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/LayoutTransformUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-decompose-conv-to-space-to-depth"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Chooses the largest block size `b` that can be folded into channels.
/// A valid `b` must divide the kernel size, stride, and input size, and after
/// decomposition the reduced kernel and stride must fit the HW limits
/// [kernel size <=7 and stride <= 2]:
///   kDim % b == 0, stride % b == 0, inputDim % b == 0
///   kDim / b <= maxKernel, stride / b <= 2
/// Returns 1 when no useful decomposition is possible.
/// Example: if kDim = 10, stride = 4, inputDim = 80, maxKernel = 7,
/// then b = 2 is valid because 10 % 2 == 0, 4 % 2 == 0, 80 % 2 == 0,
/// kDim / b = 5 <= 7, and stride / b = 2 <= 2.
static int64_t selectBlock(int64_t kDim, int64_t stride, int64_t inputDim, int64_t maxKernel) {
    if (kDim <= maxKernel && stride <= 2)
        return 1; // already within HW limit, no decomposition needed

    int64_t best = 1;
    // Try divisors of kDim in descending order
    for (int64_t b = kDim; b >= 2; --b) {
        if (kDim % b != 0)
            continue;
        if (stride % b != 0)
            continue;
        if (inputDim % b != 0)
            continue;
        if (kDim / b > maxKernel)
            continue; // reduction not sufficient
        if (stride / b > 2)
            continue; // reduction not sufficient (only s1 & s2 are supported by hardware)
        best = b;
        break;
    }
    return best;
}

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

/// Rewrites linalg.conv_2d_nchw_fchw with a large kernel by inserting
/// SpaceToDepth on the input and reusing the same transform on the filter.
///
/// Both input  [N, C,  H,  W ]  and filter [F, C, kH, kW] are 4-D NCHW-like
/// tensors, so torq::getSpaceToDepth() applies identically to both:
///   input  -> [N, C*bH*bW, H/bH,  W/bW ]
///   filter -> [F, C*bH*bW, kH/bH, kW/bW]
/// The channel ordering is consistent, so the convolution result is unchanged.
/// No constant-only restriction — the filter can be any Value.
struct DecomposeConvWithSpaceToDepthPattern : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv2DNchwFchwOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        Value input = convOp.getInputs()[0];   // [N, C, H, W]    NCHW
        Value filter = convOp.getInputs()[1];  // [F, C, kH, kW]  FCHW
        Value output = convOp.getOutputs()[0]; // [N, F, oH, oW]  NFHW

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());

        ArrayRef<int64_t> inShape = inputType.getShape();   // N C H W
        ArrayRef<int64_t> fltShape = filterType.getShape(); // F C kH kW

        if (inShape.size() != 4 || fltShape.size() != 4)
            return rewriter.notifyMatchFailure(convOp, "unexpected tensor rank");

        const int64_t H = inShape[2];
        const int64_t W = inShape[3];
        const int64_t kH = fltShape[2];
        const int64_t kW = fltShape[3];

        // Strides and dilations
        auto stridesAttr = convOp.getStrides();
        auto dilationsAttr = convOp.getDilations();
        int64_t sH = stridesAttr.getValues<int64_t>()[0];
        int64_t sW = stridesAttr.getValues<int64_t>()[1];
        int64_t dH = dilationsAttr.getValues<int64_t>()[0];
        int64_t dW = dilationsAttr.getValues<int64_t>()[1];

        // Only handle unit dilation — dilated convs require separate algorithm.
        // TODO: expandthefilter by inserting zeros to support dilation > 1.
        if (dH != 1 || dW != 1)
            return rewriter.notifyMatchFailure(convOp, "dilation != 1 not supported");

        // the max kernel size supported by the hardware is 7x7
        int64_t maxK = 7;

        // Determine per-axis block sizes independently.
        // Width axis: absorb sW into bW.
        int64_t bW = selectBlock(kW, sW, W, maxK);
        // Height axis: absorb sH into bH.
        int64_t bH = selectBlock(kH, sH, H, maxK);

        if (bH == 1 && bW == 1)
            return rewriter.notifyMatchFailure(
                convOp, "no S2D decomposition possible (kernel already within HW limit "
                        "or divisibility conditions not met)"
            );

        LLVM_DEBUG(
            llvm::dbgs() << "[S2D] Decomposing conv: kH=" << kH << " kW=" << kW << " sH=" << sH
                         << " sW=" << sW << " -> bH=" << bH << " bW=" << bW << "\n"
        );

        // ------------------------------------------------------------------ //
        // 1. SpaceToDepth on input:  [N, C, H, W] -> [N, C*bH*bW, H/bH, W/bW]
        // 2. SpaceToDepth on filter: [F, C, kH, kW] treated as [n=F, c=C, h=kH, w=kW]
        //                         -> [F, C*bH*bW, kH/bH, kW/bW]
        //
        // Both use getSpaceToDepth which interprets dim layout as [N, C, H, W].
        // The channel ordering (c * bH*bW + bh * bW + bw) is identical for both,
        // so the convolution dot product is unchanged.
        // ------------------------------------------------------------------ //
        Value s2dInput = getSpaceToDepth(input, bH, bW, rewriter);
        Value s2dFilter = getSpaceToDepth(filter, bH, bW, rewriter);

        setCompileTimeConstAttr(s2dFilter.getDefiningOp());

        // ------------------------------------------------------------------ //
        // 3. New conv with reduced kernel and absorbed stride.
        //    stride: [sH/bH, sW/bW]   output shape unchanged: [N, F, oH, oW]
        // ------------------------------------------------------------------ //
        int64_t newSH = sH / bH;
        int64_t newSW = sW / bW;

        auto newStridesAttr = rewriter.getI64TensorAttr({newSH, newSW});

        auto newConv = linalg::Conv2DNchwFchwOp::create(
            rewriter, loc, outputType, ValueRange{s2dInput, s2dFilter}, ValueRange{output},
            newStridesAttr, dilationsAttr
        );

        rewriter.replaceOp(convOp, newConv);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct DecomposeConvToSpaceToDepthPass
    : public impl::DecomposeConvToSpaceToDepthBase<DecomposeConvToSpaceToDepthPass> {

    void runOnOperation() override {
        auto func = getOperation();
        MLIRContext *ctx = func->getContext();

        RewritePatternSet patterns(ctx);
        patterns.add<DecomposeConvWithSpaceToDepthPattern>(ctx);

        if (failed(applyPatternsGreedily(func, std::move(patterns))))
            return signalPassFailure();
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDecomposeConvToSpaceToDepthPass() {
    return std::make_unique<DecomposeConvToSpaceToDepthPass>();
}

} // namespace mlir::syna::torq
