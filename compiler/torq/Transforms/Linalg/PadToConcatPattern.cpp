// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

enum class PadToConcatMode { Enable, Disable };

static llvm::cl::opt<PadToConcatMode> clPadToConcatMode(
    "torq-pad-to-concat-mode",
    llvm::cl::desc("Choose how to handle a tensor pad with a constant padding value"),
    llvm::cl::values(
        clEnumValN(
            PadToConcatMode::Enable, "enable",
            "A pad that appends a scalar constant along one axis is the concatenation of its "
            "source and a constant-filled tensor, so rewrite it as that concat instead of a "
            "fill-and-copy kernel. Value-preserving; on by default"
        ),
        clEnumValN(PadToConcatMode::Disable, "disable", "Leave constant-value pads as is")
    ),
    llvm::cl::init(PadToConcatMode::Enable)
);

namespace {

/// The constant the pad's value region yields, or null if the region is not
/// exactly one tensor.yield of a scalar arith.constant of the element type.
Attribute padValueConstant(tensor::PadOp pad, Type elem) {
    Block &block = *pad.getBody();
    if (block.getOperations().size() != 1)
        return nullptr;
    auto yield = dyn_cast<tensor::YieldOp>(block.front());
    if (!yield)
        return nullptr;
    auto constant = dyn_cast_or_null<arith::ConstantOp>(yield.getValue().getDefiningOp());
    if (!constant || constant.getValueAttr().getType() != elem)
        return nullptr;
    return constant.getValueAttr();
}

// Reshapes inserted by convolution/pooling raising must not hide an input pad.
// Conservatively preserve pads through any expand/collapse chain.
bool feedsConvolution(Value value) {
    return llvm::any_of(value.getUsers(), [](Operation *user) {
        if (isa<linalg::ConvolutionOpInterface>(user))
            return true;
        return isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(user) &&
               feedsConvolution(user->getResult(0));
    });
}

/// Rewrites a constant-value pad: a single trailing pad on one axis becomes
/// a concat of the source and a constant-filled tensor.
LogicalResult padToConcat(tensor::PadOp pad, PatternRewriter &rewriter) {
    auto sourceTy = cast<RankedTensorType>(pad.getSource().getType());
    auto resultTy = dyn_cast<RankedTensorType>(pad.getResult().getType());
    if (!resultTy || !sourceTy.hasStaticShape() || !resultTy.hasStaticShape())
        return failure();
    unsigned rank = sourceTy.getRank();
    if (rank != resultTy.getRank())
        return failure();
    // A nofold pad is kept materialized on purpose.
    if (pad.getNofold())
        return failure();
    // Convolution and pooling lowering absorbs input pads into hardware padding.
    // A concat hides that padding and can also block producer segmentation fusion.
    if (feedsConvolution(pad.getResult()))
        return failure();
    int64_t padAxis = -1;
    int64_t padLen = 0;
    for (unsigned i = 0; i < rank; ++i) {
        // kDynamic amounts are negative, so this also declines dynamic pads.
        int64_t lo = pad.getStaticLow()[i];
        int64_t hi = pad.getStaticHigh()[i];
        if (lo != 0 || hi < 0)
            return failure();
        if (hi == 0)
            continue;
        if (padAxis != -1)
            return failure(); // more than one padded axis
        padAxis = i;
        padLen = hi;
    }

    Attribute value = padValueConstant(pad, resultTy.getElementType());
    if (!value)
        return failure();

    // No trailing pad: the all-zero case is canonicalized away upstream
    // (PadOp::fold), so it never needs rewriting here.
    if (padAxis == -1)
        return failure();

    Location loc = pad.getLoc();
    SmallVector<int64_t> fillShape(sourceTy.getShape());
    fillShape[padAxis] = padLen;
    auto fillTy = RankedTensorType::get(fillShape, resultTy.getElementType());
    auto fillAttr = DenseElementsAttr::get(fillTy, {value});
    Value fill = arith::ConstantOp::create(rewriter, loc, fillTy, fillAttr);
    Value concat = tensor::ConcatOp::create(
        rewriter, loc, resultTy, padAxis, ValueRange{pad.getSource(), fill}
    );
    rewriter.replaceOp(pad, concat);
    LLVM_DEBUG(
        llvm::dbgs() << "[torq-pad-to-concat] " << sourceTy << " padded " << padLen << " on axis "
                     << padAxis << " to a concat with a constant-filled " << fillTy << "\n"
    );
    return success();
}

/// Rewrites a constant-value tensor.pad that appends a scalar constant along
/// one axis into a tensor.concat of the source and a constant-filled tensor.
class PadToConcatPattern : public OpRewritePattern<tensor::PadOp> {
  public:
    explicit PadToConcatPattern(MLIRContext *ctx)
        : OpRewritePattern<tensor::PadOp>(ctx, /*benefit=*/2) {
        setDebugName("PadToConcatPattern");
    }

    LogicalResult matchAndRewrite(tensor::PadOp pad, PatternRewriter &rewriter) const override {
        return padToConcat(pad, rewriter);
    }
};

} // namespace

void populatePadToConcatPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    if (clPadToConcatMode != PadToConcatMode::Enable)
        return;
    patterns.add<PadToConcatPattern>(ctx);
}

} // namespace mlir::syna::torq
