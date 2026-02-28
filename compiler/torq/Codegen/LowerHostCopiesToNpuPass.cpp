// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torq/Utils/EncodingUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-host-copies-to-npu"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// This pattern finds torq_hl.host_copy operations that can be replaced
// by a pair of torq_hl.load and torq_hl.store operations that copy the
// data through LRAM. This allows to avoid going back to the host for
// copying data between two buffers in XRAM which can be done much more
// efficiently by the DMA engine of the NPU.
class XramToXramHostCopyPattern : public OpRewritePattern<torq_hl::HostCopyOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::HostCopyOp copyOp, PatternRewriter &rewriter) const override {

        auto inputType = copyOp.getInput().getType();
        auto outputType = copyOp.getOutput().getType();

        // we only lower XRAM to XRAM operations (XRAM to LRAM copies must be done by
        // the host as they are used to setup the LRAM for the first NSS block execution)
        if (getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Xram ||
            getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Xram) {
            return failure();
        }

        // NPU DMA engine can process only up to 4 dimension for strided copies,
        // so we can't do the copy in LRAM if the input or output buffer has more than 4 dimensions
        // here we look at the strides on the host copy operation and not the shape of input/output
        // types because the types because some of those dimensions may be contiguous
        // and therefore the load/store op can ignore them
        if (copyOp.getInputStridesBytes().size() > 4 || copyOp.getOutputStridesBytes().size() > 4) {
            return failure();
        }

        // compute how much memory we need in LRAM to store the elements of the input buffer
        // this size excludes all the padding bytes between elements which we don't need to
        // copy to LRAM
        auto elementSizeBytes =
            llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(), 8);
        auto totalElementSizeBytes = elementSizeBytes * inputType.getNumElements();

        // if the total size of the elements is larger than the available memory for tiling in
        // LRAM, we can't do the copy in LRAM with a single load/store pair
        if (totalElementSizeBytes > TorqHw::get().getAvailableMemoryForTiling()) {
            return failure();
        }

        // create a temporary dense buffer in LRAM to copy the data
        auto tempBufferType = MemRefType::get(
            inputType.getShape(), inputType.getElementType(), nullptr,
            createDenseEncoding(inputType, torq_hl::MemorySpace::Lram)
        );

        auto tempBuffer =
            rewriter.create<memref::AllocOp>(copyOp.getLoc(), tempBufferType, ValueRange{});

        // create a torq copy to copy from the source buffer in XRAM to the temporary buffer in LRAM
        // this will be a torq_hl::LoadOp
        if (failed(createTorqCopy(rewriter, copyOp.getLoc(), copyOp.getInput(), tempBuffer))) {
            copyOp->emitError("failed to create torq copy from XRAM to LRAM");
            llvm::report_fatal_error("failed to create torq copy from XRAM to LRAM");
        }

        // create a torq copy to copy from the temporary buffer in LRAM to the destination buffer in
        // XRAM this will be a torq_hl::StoreOp
        if (failed(createTorqCopy(rewriter, copyOp.getLoc(), tempBuffer, copyOp.getOutput()))) {
            copyOp->emitError("failed to create torq copy from LRAM to XRAM");
            llvm::report_fatal_error("failed to create torq copy from LRAM to XRAM");
        }

        // erase the original host copy operation
        copyOp.erase();

        LLVM_DEBUG({ llvm::dbgs() << "Replace host copy with pair of load and store ops\n"; });

        return success();
    }
};

class LowerHostCopiesToNpuPass : public LowerHostCopiesToNpuBase<LowerHostCopiesToNpuPass> {
  public:
    LowerHostCopiesToNpuPass() = default;
    LowerHostCopiesToNpuPass(const LowerHostCopiesToNpuPass &pass) {}

    void runOnOperation() override;
};

void LowerHostCopiesToNpuPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<XramToXramHostCopyPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerHostCopiesToNpuPass() {
    return std::make_unique<LowerHostCopiesToNpuPass>();
}

} // namespace mlir::syna::torq
