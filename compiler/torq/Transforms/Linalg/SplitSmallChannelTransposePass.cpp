// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// SplitSmallChannelTranspose Pass
//===----------------------------------------------------------------------===//
//
// The block transpose kernel needs at least 32 input rows and 32 input columns.
// A layout transpose with fewer than 32 channels, such as the NHWC->NCHW
// transpose of an RGB input ([1, H, W, 3] -> [1, 3, H, W]), falls back to the
// generalized kernel, which moves one element per cycle.
//
// A (N, C) -> (C, N) transpose with N a multiple of 32 is also two transposes
// with only reshapes between them. Viewing the input as (N/32, 32*C):
//
//   (N/32, 32*C) -T1-> (32*C, N/32) = (32, C*N/32) -T2-> (C*N/32, 32) = (C, N)
//
// Both have at least 32 rows and columns, so both use the block kernel. Keeping
// the row dims (here H) separate so that tile-and-fuse can still tile them:
//
//   [B, R..., W, C] -expand-> [B, R..., W/32, 32, C]
//     T1 -> [B, 32, C, R..., W/32]
//     T2 -> [B, C, R..., W/32, 32]
//   -collapse-> [B, C, R..., W]
//
// When W is not a multiple of 32, W is split in blocks of b > 32 elements instead, b a divisor of
// W (e.g. 100 for 300). T2 then has b rows: the block kernel transposes the first rows in blocks
// of 32 and the generalized kernel the remaining b % 32. selectWBlockSize picks b by an estimate
// of the cycles of both transposes.
//
// T1 is emitted as a copy linalg.generic: the canonicalizer folds two adjacent
// linalg.transpose ops back into one, also inside each tile after tile-and-fuse.
// GenericToTransposeBroadcastOpConversion lowers the generic to torq_hl.transpose.
//
// Only transposes feeding a convolution or pooling (directly or through a
// tensor.pad) are split, so that both transposes join its fuse group.
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-split-small-channel-transpose"

namespace mlir::syna::torq {

namespace {

static llvm::cl::opt<bool> clSplitSmallChannelTranspose(
    "torq-enable-small-channel-transpose-split",
    llvm::cl::desc("Split small-channel layout transposes into two block transposes"),
    llvm::cl::init(true)
);

// Rows and columns of a block of the block transpose kernel
constexpr int64_t kBlockSize = 32;

// Estimated cycles of a 2D transpose of rows x cols elements. TransposePattern transposes
// [0, rows32) x [0, cols32) with the block kernel and the rest with the generalized one (all of
// it when rows or cols is below a block). The generalized kernel moves about one element per
// cycle and the block kernel about a block row of 32 bytes per cycle.
static int64_t estimateTransposeCycles(int64_t rows, int64_t cols, int64_t elementBytes) {
    int64_t total = rows * cols;
    if (rows < kBlockSize || cols < kBlockSize)
        return total;
    int64_t vectorSize = kBlockSize / elementBytes;
    int64_t blockElements = (rows - rows % kBlockSize) * (cols - cols % vectorSize);
    return total - blockElements + blockElements / vectorSize;
}

// Chooses the size b of the W blocks: [..., W, C] is viewed as [..., W/b, b, C] and transposed by
// T1 (rows R*W/b, cols b*C) and T2 (rows b, cols C*R*W/b), where R is the product of the other row
// dims. b divides W and is at least a block, so that T2 can use the block kernel. Returns the b
// with the lowest estimated cost, or nullopt if no split is at least twice as fast as the single
// transpose, which the generalized kernel does one element (two if C is even) per cycle.
static std::optional<int64_t>
selectWBlockSize(int64_t r, int64_t w, int64_t c, int64_t elementBytes) {
    int64_t singleCycles = r * w * c / (c % 2 == 0 ? 2 : 1);
    // Blocks of 32 have no tail in T2, and give T1 the most rows when the tiles are short
    if (w % kBlockSize == 0 && r * w / kBlockSize >= kBlockSize)
        return kBlockSize;
    std::optional<int64_t> best;
    int64_t bestCycles = singleCycles / 2;
    for (int64_t b = kBlockSize; b <= w; b++) {
        if (w % b != 0)
            continue;
        int64_t cycles = estimateTransposeCycles(r * w / b, b * c, elementBytes) +
                         estimateTransposeCycles(b, c * r * w / b, elementBytes);
        if (cycles < bestCycles) {
            best = b;
            bestCycles = cycles;
        }
    }
    return best;
}

// True if `op` is a convolution or pooling, reached directly or through a tensor.pad
static bool feedsSlidingWindowOp(linalg::TransposeOp op) {
    Value result = op.getResult()[0];
    if (!result.hasOneUse())
        return false;
    Operation *user = *result.getUsers().begin();
    if (auto padOp = dyn_cast<tensor::PadOp>(user)) {
        if (!padOp.getResult().hasOneUse())
            return false;
        user = *padOp.getResult().getUsers().begin();
    }
    return isa<linalg::ConvolutionOpInterface>(user);
}

class SplitSmallChannelTransposePattern : public OpRewritePattern<linalg::TransposeOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::TransposeOp op, PatternRewriter &rewriter) const override {
        auto inType = dyn_cast<RankedTensorType>(op.getInput().getType());
        auto outType = dyn_cast<RankedTensorType>(op.getInit().getType());
        if (!inType || !outType || !inType.hasStaticShape())
            return rewriter.notifyMatchFailure(op, "expected a static tensor");
        int64_t elementBits = inType.getElementTypeBitWidth();
        if (elementBits != 8 && elementBits != 16)
            return rewriter.notifyMatchFailure(op, "the block kernel needs 8 or 16-bit elements");

        // The input is [B..., R..., C...] and the output [B..., C..., R...]
        ArrayRef<int64_t> perm = op.getPermutation();
        int batchDims = 0, rowDims = 0;
        if (!isGroupSwapTranspose(perm, batchDims, rowDims))
            return rewriter.notifyMatchFailure(op, "not a swap of two groups of dims");
        ArrayRef<int64_t> inShape = inType.getShape();
        int rank = inShape.size();
        int colDims = rank - batchDims - rowDims;
        int64_t rows = 1, cols = 1;
        for (int i = batchDims; i < rank; i++)
            (i < batchDims + rowDims ? rows : cols) *= inShape[i];
        int64_t w = inShape[batchDims + rowDims - 1];
        if (cols >= kBlockSize)
            return rewriter.notifyMatchFailure(op, "already uses the block kernel");
        if (!feedsSlidingWindowOp(op))
            return rewriter.notifyMatchFailure(op, "does not feed a convolution or pooling");
        std::optional<int64_t> blockSize =
            selectWBlockSize(rows / w, w, cols, inType.getElementTypeBitWidth() / 8);
        if (!blockSize)
            return rewriter.notifyMatchFailure(op, "no W block size makes the split worth it");
        int64_t b = *blockSize;

        Location loc = op.getLoc();
        Type elementType = inType.getElementType();

        // Expand W into [W/b, b]: [B..., R'..., Wk, Wj, C...]
        int wkDim = batchDims + rowDims - 1;
        int wjDim = wkDim + 1;
        SmallVector<int64_t> expandedShape(inShape.begin(), inShape.begin() + wkDim);
        expandedShape.append({w / b, b});
        expandedShape.append(inShape.begin() + wkDim + 1, inShape.end());
        SmallVector<ReassociationIndices> expandReassoc;
        for (int i = 0; i < rank; i++) {
            if (i < wkDim)
                expandReassoc.push_back({i});
            else if (i == wkDim)
                expandReassoc.push_back({wkDim, wjDim});
            else
                expandReassoc.push_back({i + 1});
        }
        Value expanded = tensor::ExpandShapeOp::create(
            rewriter, loc, RankedTensorType::get(expandedShape, elementType), op.getInput(),
            expandReassoc
        );

        // T1: [B..., R'..., Wk, Wj, C...] -> [B..., Wj, C..., R'..., Wk]
        SmallVector<int64_t> perm1;
        for (int i = 0; i < batchDims; i++)
            perm1.push_back(i);
        perm1.push_back(wjDim);
        for (int i = 0; i < colDims; i++)
            perm1.push_back(wjDim + 1 + i);
        for (int i = batchDims; i <= wkDim; i++)
            perm1.push_back(i);
        SmallVector<int64_t> shape1;
        for (int64_t d : perm1)
            shape1.push_back(expandedShape[d]);
        Value init1 = tensor::EmptyOp::create(rewriter, loc, shape1, elementType);
        auto t1 = linalg::TransposeOp::create(rewriter, loc, expanded, init1, perm1);
        FailureOr<linalg::GenericOp> t1Generic = linalg::generalizeNamedOp(rewriter, t1);
        if (failed(t1Generic))
            return rewriter.notifyMatchFailure(op, "could not generalize the first transpose");

        // T2: [B..., Wj, C..., R'..., Wk] -> [B..., C..., R'..., Wk, Wj]
        SmallVector<int64_t> perm2;
        for (int i = 0; i < batchDims; i++)
            perm2.push_back(i);
        for (int i = batchDims + 1; i <= rank; i++)
            perm2.push_back(i);
        perm2.push_back(batchDims);
        SmallVector<int64_t> shape2;
        for (int64_t d : perm2)
            shape2.push_back(shape1[d]);
        Value init2 = tensor::EmptyOp::create(rewriter, loc, shape2, elementType);
        auto t2 = linalg::TransposeOp::create(rewriter, loc, t1Generic->getResult(0), init2, perm2);

        // Collapse [Wk, Wj] back into W: [B..., C..., R..., W]
        SmallVector<ReassociationIndices> collapseReassoc;
        for (int i = 0; i < rank - 1; i++)
            collapseReassoc.push_back({i});
        collapseReassoc.push_back({rank - 1, rank});
        Value collapsed = tensor::CollapseShapeOp::create(
            rewriter, loc, outType, t2.getResult()[0], collapseReassoc
        );

        rewriter.replaceOp(op, collapsed);
        return success();
    }
};

struct SplitSmallChannelTransposePass
    : public impl::SplitSmallChannelTransposeBase<SplitSmallChannelTransposePass> {

    void runOnOperation() override {
        if (!clSplitSmallChannelTranspose)
            return;
        auto func = getOperation();
        MLIRContext *ctx = func->getContext();

        RewritePatternSet patterns(ctx);
        patterns.add<SplitSmallChannelTransposePattern>(ctx);

        if (failed(applyPatternsGreedily(func, std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSplitSmallChannelTransposePass() {
    return std::make_unique<SplitSmallChannelTransposePass>();
}

} // namespace mlir::syna::torq
