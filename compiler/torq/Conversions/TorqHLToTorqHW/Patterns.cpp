// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

class CopyPattern : public OpRewritePattern<memref::CopyOp> {
  public:
    using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter &rewriter) const override {

        auto sourceMemSpace = getEncodingMemorySpace(op.getSource().getType());
        auto targetMemSpace = getEncodingMemorySpace(op.getTarget().getType());

        if ((sourceMemSpace == torq_hl::MemorySpace::Lram &&
             (targetMemSpace == torq_hl::MemorySpace::Dtcm ||
              targetMemSpace == torq_hl::MemorySpace::Itcm)) ||
            (targetMemSpace == torq_hl::MemorySpace::Lram &&
             (sourceMemSpace == torq_hl::MemorySpace::Dtcm ||
              sourceMemSpace == torq_hl::MemorySpace::Itcm))) {

            auto copyTaskOp = rewriter.create<torq_hw::NssTaskOp>(op.getLoc());

            rewriter.createBlock(&copyTaskOp.getBody());

            rewriter.create<torq_hw::CDMAStartOp>(
                op.getLoc(), op.getTarget(), op.getSource(), nullptr, nullptr
            );
            rewriter.create<torq_hw::CDMAWaitOp>(op.getLoc());
            ;

            rewriter.eraseOp(op);

            return success();
        }
        else {
            return op.emitError("memref.copy not between lram and dtcm/itcm memrefs not supported");
        }
    }
};

static FailureOr<DmaNdlAttr> createNdl(
    MLIRContext *ctx, ArrayRef<int64_t> strideBytes, ArrayRef<int64_t> shape,
    int64_t contiguousElementsSizeBytes
) {

    // write out the contiguous blocks with appropriate strides
    SmallVector<DmaDimAttr> ndlDims;
    ndlDims.reserve(shape.size());

    // TODO: check if we can write more than one byte here
    ndlDims.push_back(DmaDimAttr::get(ctx, contiguousElementsSizeBytes, 1));

    for (int i = shape.size() - 1; i >= 0; i--) {
        ndlDims.push_back(DmaDimAttr::get(ctx, shape[i], strideBytes[i]));
    }

    return DmaNdlAttr::get(ctx, ndlDims);
}

/*

    DMA OUT block works as follows:

    Inputs:

    READ NDL = LRAM_BASE { count = LRAM_LEN, stride = 1 }
    WRITE_NDL = XRAM_BASE { count = XRAM_COUNT_0, stride = XRAM_STRIDE_0 }, { count = XRAM_COUNT_1,
   stride = XRAM_STRIDE_1 }, { count = XRAM_COUNT_2, stride = XRAM_STRIDE_2 }

    Process:

    assert(XRAM_COUNT_0 * XRAM_COUNT_1 * XRAM_COUNT_2 == LRAM_LEN) ----> Will hang next DMA OUT if
   XRAM_COUNT_0 * XRAM_COUNT_1 * XRAM_COUNT_2 < LRAM_LEN

    lram_address = LRAM_BASE
    xram_address = XRAM_BASE

    for (int d1 = 0 ; d1 < XRAM_COUNT_2; d1++) {
        for (int d2 = 0 ; d2 < XRAM_COUNT_1; d2++) {
                memcpy(xram_address, lram_address, XRAM_COUNT_0);
                lram_address += XRAM_COUNT_0;
                xram_address += XRAM_STRIDE1;
        }
        xram_address += XRAM_STRIDE2;
    }

    TODO: Does it support a fourth dimension in the WRITE_NDL?

*/

class StoreOpPattern : public OpRewritePattern<torq_hl::StoreOp> {
  public:
    using OpRewritePattern<torq_hl::StoreOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::StoreOp op, PatternRewriter &rewriter) const override {
        auto ctx = rewriter.getContext();
        auto outputType = op.getOutput().getType();
        auto inputType = op.getInput().getType();

        if (inputType.getShape() != outputType.getShape()) {
            return op.emitError("StoreOp input and output shapes must be equal");
        }

        if (inputType.getElementType() != outputType.getElementType()) {
            return op.emitError("StoreOp input and output element types must be equal");
        }

        if (getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Xram ||
            getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Lram) {
            return op.emitError("StoreOp input must be lram and output must be xram");
        }

        // read elements contiguously from LRAM
        int totalCount = 1;
        for (auto count : op.getShape()) {
            totalCount *= count;
        }
        auto totalInputSizeBytes = totalCount * op.getElementSizeBytes();
        auto readNdl = DmaNdlAttr::get(ctx, DmaDimAttr::get(ctx, totalInputSizeBytes, 1));

        // write the data to XRAM with the appropriate strides
        auto maybeWriteNdl =
            createNdl(ctx, op.getOutputStridesBytes(), op.getShape(), op.getElementSizeBytes());
        if (failed(maybeWriteNdl)) {
            return op.emitError("cannot create write NDL for output type");
        }

        auto taskOp = rewriter.create<torq_hw::NssTaskOp>(op.getLoc());
        rewriter.createBlock(&taskOp.getBody());
        rewriter.create<torq_hw::DmaOutCfgOp>(
            op.getLoc(), op.getInput(), op.getOutput(), readNdl, *maybeWriteNdl, nullptr, nullptr
        );
        rewriter.create<torq_hw::DmaOutStartOp>(op.getLoc());
        rewriter.create<torq_hw::DmaOutWaitOp>(op.getLoc());
        rewriter.replaceOp(op, taskOp);

        return success();
    }
};

class LoadOpPattern : public OpRewritePattern<torq_hl::LoadOp> {
  public:
    using OpRewritePattern<torq_hl::LoadOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(torq_hl::LoadOp op, PatternRewriter &rewriter) const override {
        auto ctx = rewriter.getContext();
        auto outputType = op.getOutput().getType();
        auto inputType = op.getInput().getType();

        if (inputType.getShape() != outputType.getShape()) {
            return op.emitError("input and output shapes must be equal");
        }

        if (inputType.getElementType() != outputType.getElementType()) {
            return op.emitError("input and output element types must be equal");
        }

        if (getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Lram ||
            getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Xram) {
            return op.emitError("input must be xram and output must be lram");
        }

        // read the data to XRAM with the appropriate strides
        auto maybeReadNdl =
            createNdl(ctx, op.getInputStridesBytes(), op.getShape(), op.getElementSizeBytes());

        if (failed(maybeReadNdl)) {
            return op.emitError("cannot create read NDL for input type");
        }

        // write elements contiguously to LRAM
        int totalCount = 1;
        for (auto count : op.getShape()) {
            totalCount *= count;
        }
        auto totalInputSizeBytes = totalCount * op.getElementSizeBytes();
        auto writeNdl = DmaNdlAttr::get(ctx, DmaDimAttr::get(ctx, totalInputSizeBytes, 1));

        auto taskOp = rewriter.create<torq_hw::NssTaskOp>(op.getLoc());
        rewriter.createBlock(&taskOp.getBody());
        rewriter.create<torq_hw::DmaInCfgOp>(
            op.getLoc(), op.getInput(), op.getOutput(), *maybeReadNdl, writeNdl, nullptr, nullptr
        );
        rewriter.create<torq_hw::DmaInStartOp>(op.getLoc());
        rewriter.create<torq_hw::DmaInWaitOp>(op.getLoc());
        rewriter.replaceOp(op, taskOp);

        return success();
    }
};

template <typename OperationT> static FailureOr<APInt> getSliceId(OperationT op) {

    auto nssInvocation = getNssInvocation(op);

    auto maybeExecutorId = getExecutorId(op.getInvocation(), nssInvocation);

    if (!maybeExecutorId) {
        return op.emitError("must have executor id");
    }

    return APInt(64, *maybeExecutorId);
}

class StartProgramOpPattern : public OpRewritePattern<torq_hl::StartProgramOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(torq_hl::StartProgramOp op, PatternRewriter &rewriter) const override {

        auto programType = cast<torq_hl::InvocationType>(op.getInvocation().getType());

        auto taskOp = rewriter.create<torq_hw::NssTaskOp>(op.getLoc());
        rewriter.createBlock(&taskOp.getBody());

        if (programType.getExecutor() == torq_hl::Executor::Slice) {

            if (op.getCodeSections().size() != 1) {
                return op.emitError("Slice executor must have one code section");
            }

            auto maybeSliceId = getSliceId(op);

            if (failed(maybeSliceId)) {
                return failure();
            }

            rewriter.create<SliceStartOp>(
                op.getLoc(), op.getInvocation(), op.getCodeSections()[0], *maybeSliceId,
                op.getArgs()
            );
        }
        else if (programType.getExecutor() == torq_hl::Executor::CSS) {

            if (op.getCodeSections().size() != 2) {
                return op.emitError("CSS executor must have one code section");
            }

            rewriter.create<CSSStartOp>(
                op.getLoc(), op.getInvocation(), op.getCodeSections()[0], op.getCodeSections()[1],
                op.getArgs(), nullptr, nullptr
            );
        }
        else {
            return op.emitError("Unsupported executor type for StartProgramOp: ")
                   << torq_hl::stringifyExecutor(programType.getExecutor());
        }

        rewriter.eraseOp(op);

        return success();
    }
};

class WaitProgramOpPattern : public OpRewritePattern<torq_hl::WaitProgramOp> {
  public:
    using OpRewritePattern::OpRewritePattern;
    LogicalResult
    matchAndRewrite(torq_hl::WaitProgramOp waitOp, PatternRewriter &rewriter) const override {

        auto taskOp = rewriter.create<torq_hw::NssTaskOp>(waitOp.getLoc());
        rewriter.createBlock(&taskOp.getBody());

        if (waitOp.getInvocation().getType().getExecutor() == torq_hl::Executor::Slice) {

            auto maybeSliceId = getSliceId(waitOp);

            if (failed(maybeSliceId)) {
                return failure();
            }

            rewriter.create<SliceWaitOp>(waitOp.getLoc(), *maybeSliceId);
        }
        else if (waitOp.getInvocation().getType().getExecutor() == torq_hl::Executor::CSS) {
            rewriter.create<CSSWaitOp>(waitOp.getLoc());
        }
        else {
            return waitOp.emitError("unsupported executor type: ")
                   << torq_hl::stringifyExecutor(waitOp.getInvocation().getType().getExecutor());
        }

        rewriter.eraseOp(waitOp);
        return success();
    }
};

#ifdef TORQHL_GENERIC_OP
void populateGenericOpPatterns(MLIRContext *ctx, RewritePatternSet &patterns);
#endif // TORQHL_GENERIC_OP

void populateNssTaskPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    patterns.add<LoadOpPattern>(ctx);
    patterns.add<StoreOpPattern>(ctx);
    patterns.add<StartProgramOpPattern>(ctx);
    patterns.add<WaitProgramOpPattern>(ctx);
    patterns.add<CopyPattern>(ctx);
#ifdef TORQHL_GENERIC_OP
    populateGenericOpPatterns(ctx, patterns);
#endif // TORQHL_GENERIC_OP
}

} // namespace mlir::syna::torq
