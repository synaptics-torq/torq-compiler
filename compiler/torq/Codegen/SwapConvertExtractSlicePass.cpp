// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SwapConvertExtractSlicePass
// ---------------------------
// TorqHLTileLargeInputsPass tiles a TorqHL op whose operand was already
// encoded (torq_hl.convert'ed) into its target memory space/layout, producing:
//   %converted = torq_hl.convert out(%init : Tc) in(%x : Tx) {...} -> Tc
//   %slice = tensor.extract_slice %converted[offsets][sizes][strides] : Tc to Ts
// i.e. extract_slice(convert). Every tile therefore requires the WHOLE tensor
// to have been converted (e.g. loaded into LRAM) up front, even though each
// tile only ever reads one slice of it.
//
// This pass swaps the pattern to convert(extract_slice): it slices the
// pre-conversion source %x the same way, then converts only that slice, so
// only the region actually read by a given tile is ever converted at once.
// Other uses of %converted (if any) are left untouched; if it ends up with no
// remaining uses it is cleaned up as dead code by the usual canonicalizer.

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "torq-swap-convert-extract-slice"

namespace mlir::syna::torq {

namespace {

class SwapConvertExtractSlice : public OpRewritePattern<tensor::ExtractSliceOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::ExtractSliceOp sliceOp, PatternRewriter &rewriter) const override {
        auto convertOp = sliceOp.getSource().getDefiningOp<torq_hl::ConvertOp>();
        if (!convertOp)
            return failure();

        if (getEncodingMemorySpace(convertOp.getInit().getType()) != torq_hl::MemorySpace::Lram) {
            // Even if the below algorithm works with XRAM encoding also we still need to profile it
            // before enabling it. So for now we only support LRAM encoding.
            return failure();
        }
        Location loc = sliceOp.getLoc();

        // convert is shape-preserving (encoding-only), so the same slice region
        // applies unchanged to its pre-conversion input.
        Value slicedInput = tensor::ExtractSliceOp::create(
            rewriter, loc, convertOp.getInput(), sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
            sliceOp.getMixedStrides()
        );
        auto ty = cast<RankedTensorType>(sliceOp.getType());
        Value newInit = tensor::EmptyOp::create(
            rewriter, loc, ty.getShape(), ty.getElementType(), convertOp.getEncodingAttr()
        );

        auto newConvert = torq_hl::ConvertOp::create(
            rewriter, loc, newInit.getType(), newInit, slicedInput, convertOp.getRequirementsAttr(),
            convertOp.getEncodingAttr()
        );

        rewriter.replaceOp(sliceOp, newConvert.getResult(0));
        return success();
    }
};

class SwapConvertExtractSlicePass
    : public impl::SwapConvertExtractSliceBase<SwapConvertExtractSlicePass> {
  public:
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<SwapConvertExtractSlice>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSwapConvertExtractSlicePass() {
    return std::make_unique<SwapConvertExtractSlicePass>();
}

} // namespace mlir::syna::torq
