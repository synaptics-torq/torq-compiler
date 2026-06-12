// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torq/Utils/EncodingUtils.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

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
    XramToXramHostCopyPattern(MLIRContext *ctx, const DenseMap<Operation *, int64_t> &lramBudget)
        : OpRewritePattern(ctx), lramBudget_(lramBudget) {}

  private:
    const DenseMap<Operation *, int64_t> &lramBudget_;
    static LogicalResult
    lowerSingleCopy(PatternRewriter &rewriter, Location loc, Value input, Value output) {
        // create a temporary dense buffer in LRAM to copy the data
        auto inputType = cast<MemRefType>(input.getType());
        auto tempBufferType = MemRefType::get(
            inputType.getShape(), inputType.getElementType(), nullptr,
            createDenseEncoding(inputType, torq_hl::MemorySpace::Lram)
        );

        auto tempBuffer = memref::AllocOp::create(rewriter, loc, tempBufferType, ValueRange{});

        // create a torq copy to copy from the source buffer in XRAM to the temporary buffer in
        // LRAM this will be a torq_hl::LoadOp
        if (failed(createTorqCopy(rewriter, loc, input, tempBuffer))) {
            emitError(loc, "failed to create torq copy from XRAM to LRAM");
            return failure();
        }

        // create a torq copy to copy from the temporary buffer in LRAM to the destination buffer
        // in XRAM this will be a torq_hl::StoreOp
        if (failed(createTorqCopy(rewriter, loc, tempBuffer, output))) {
            emitError(loc, "failed to create torq copy from LRAM to XRAM");
            return failure();
        }
        return success();
    }

    // Recursive case 1: both buffers are dense and the total size exceeds the LRAM budget.
    // Flatten to 1D and iterate over fixed-size contiguous chunks.
    static LogicalResult copyDenseMemRefsInChunks(
        PatternRewriter &rewriter, Location loc, Value input, Value output,
        int64_t elementSizeBytes, int64_t maxChunkSizeBytes
    ) {
        auto inputType = cast<MemRefType>(input.getType());
        auto outputType = cast<MemRefType>(output.getType());
        int64_t rank = inputType.getRank();
        int64_t maxChunkElements = maxChunkSizeBytes / elementSizeBytes;
        int64_t totalElements = inputType.getNumElements();

        if (maxChunkElements <= 0) {
            return failure();
        }

        Value flatInput = input;
        Value flatOutput = output;

        if (rank > 1) {
            SmallVector<ReassociationIndices> reassociation(1);
            reassociation[0].reserve(rank);
            for (int64_t i = 0; i < rank; ++i)
                reassociation[0].push_back(i);

            auto flatInputType =
                memref::CollapseShapeOp::computeCollapsedType(inputType, reassociation);
            auto flatOutputType =
                memref::CollapseShapeOp::computeCollapsedType(outputType, reassociation);

            flatInput =
                memref::CollapseShapeOp::create(rewriter, loc, flatInputType, input, reassociation)
                    .getResult();
            flatOutput = memref::CollapseShapeOp::create(
                             rewriter, loc, flatOutputType, output, reassociation
            )
                             .getResult();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "[LowerHostCopiesToNpu] Dense tiling into "
                         << llvm::divideCeil(totalElements, maxChunkElements)
                         << " chunk(s) of up to " << maxChunkElements << " elements\n";
        });

        for (int64_t offset = 0; offset < totalElements; offset += maxChunkElements) {
            int64_t currChunkElements = std::min(maxChunkElements, totalElements - offset);
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu]   chunk offset=" << offset
                             << " size=" << currChunkElements << "\n";
            });

            SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(offset)};
            SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(currChunkElements)};
            SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};

            auto inputChunk =
                memref::SubViewOp::create(rewriter, loc, flatInput, offsets, sizes, strides);
            auto outputChunk =
                memref::SubViewOp::create(rewriter, loc, flatOutput, offsets, sizes, strides);

            if (failed(lowerSingleCopy(rewriter, loc, inputChunk, outputChunk)))
                return failure();
        }
        return success();
    }

    // Recursive case 2: at least one buffer is non-dense and the total size exceeds the LRAM
    // budget. Slice off one row at a time along the first non-unit dimension and recurse so that
    // each row is independently checked against the budget on the next call.
    static LogicalResult copyNonDenseMemRefsByRows(
        PatternRewriter &rewriter, Location loc, Value input, Value output,
        int64_t maxChunkSizeBytes
    ) {
        auto inputType = cast<MemRefType>(input.getType());
        int64_t rank = inputType.getRank();
        auto shape = inputType.getShape();

        // Find the first dimension with extent > 1 to split over.
        auto nonUnitDim = llvm::find_if(shape, [&](int64_t dim) { return dim > 1; });
        if (nonUnitDim == shape.end()) {
            // All dimensions are unit, but the total size exceeds the budget which shouldn't happen
            // since the base case should have handled it. Just return failure since we don't know
            // how to split further.
            assert(false && "unexpectedly found non-dense memref with only unit dimensions");
        }
        int64_t dimIdx = std::distance(shape.begin(), nonUnitDim);
        int64_t dimShape = *nonUnitDim;

        LLVM_DEBUG({
            llvm::dbgs() << "[LowerHostCopiesToNpu] Non-dense split over dim" << dimIdx
                         << " (size=" << dimShape << ")\n";
        });

        auto elementSizeBytes =
            llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(), 8);
        auto chunkSize = std::accumulate(
            shape.begin() + dimIdx + 1, shape.end(), elementSizeBytes, std::multiplies<int64_t>()
        );

        // Per-row sizes: extent 1 on the split dimension, full extent on all others.
        SmallVector<int64_t> rowSizeInts(shape.begin(), shape.end());

        SmallVector<OpFoldResult> sizes, strides;
        for (int64_t s : rowSizeInts)
            sizes.push_back(rewriter.getIndexAttr(s));
        for (int64_t i = 0; i < rank; ++i)
            strides.push_back(rewriter.getIndexAttr(1));

        int rowSize = maxChunkSizeBytes / chunkSize;
        SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
        for (int64_t row = 0; row < dimShape; row += rowSize) {
            offsets[dimIdx] = rewriter.getIndexAttr(row);
            sizes[dimIdx] = rewriter.getIndexAttr(std::min<int64_t>(rowSize, dimShape - row));

            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu]   non-dense row=" << row << "/" << dimShape
                             << "\n";
            });

            auto inputSlice =
                memref::SubViewOp::create(rewriter, loc, input, offsets, sizes, strides);
            auto outputSlice =
                memref::SubViewOp::create(rewriter, loc, output, offsets, sizes, strides);

            if (failed(
                    copyMemRefsRecursive(rewriter, loc, inputSlice, outputSlice, maxChunkSizeBytes)
                ))
                return failure();
        }
        return success();
    }

    // Recursively copy input to output through LRAM, splitting along the outermost non-unit
    // dimension until each sub-region fits within maxChunkSizeBytes.
    //
    // Base case: the region fits in LRAM → lowerSingleCopy handles it directly (works for both
    // dense and non-dense layouts via strided DMA).
    //
    // Recursive case 1 (both dense, too big): copyDenseMemRefsInChunks.
    // Recursive case 2 (non-dense, too big): copyNonDenseMemRefsByRows.
    static LogicalResult copyMemRefsRecursive(
        PatternRewriter &rewriter, Location loc, Value input, Value output,
        int64_t maxChunkSizeBytes
    ) {
        auto inputType = cast<MemRefType>(input.getType());
        auto outputType = cast<MemRefType>(output.getType());

        auto elementSizeBytes =
            llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(), 8);

        if (maxChunkSizeBytes < elementSizeBytes) {
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu] Cannot copy: LRAM budget ("
                             << maxChunkSizeBytes << ") smaller than one element ("
                             << elementSizeBytes << ")\n";
            });
            return failure();
        }

        // Does not fit — need static shape to slice
        if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu] Cannot handle non-static shape ("
                             << inputType << " / " << outputType << ")\n";
            });
            return failure();
        }

        int64_t totalElementSizeBytes = elementSizeBytes * inputType.getNumElements();

        // Base case: the entire region fits in LRAM.
        // lowerSingleCopy handles both dense and non-dense layouts via strided DMA.
        if (totalElementSizeBytes <= maxChunkSizeBytes) {
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu] Fits in LRAM (" << totalElementSizeBytes
                             << " bytes), lowering as single copy\n";
            });
            return lowerSingleCopy(rewriter, loc, input, output);
        }

        if (isDenseInMemory(inputType) && isDenseInMemory(outputType)) {
            return copyDenseMemRefsInChunks(
                rewriter, loc, input, output, elementSizeBytes, maxChunkSizeBytes
            );
        }

        return copyNonDenseMemRefsByRows(rewriter, loc, input, output, maxChunkSizeBytes);
    }

    LogicalResult convertToLramCopy(
        torq_hl::HostCopyOp copyOp, int64_t maxChunkSizeBytes, PatternRewriter &rewriter
    ) const {
        auto inputType = copyOp.getInput().getType();
        // compute sizes for debug logging only; the recursive helper re-derives them per slice
        auto elementSizeBytes =
            llvm::divideCeil(inputType.getElementType().getIntOrFloatBitWidth(), 8);
        auto totalElementSizeBytes = elementSizeBytes * inputType.getNumElements();

        LLVM_DEBUG({
            llvm::dbgs() << "[LowerHostCopiesToNpu] elementSizeBytes=" << elementSizeBytes
                         << " totalElementSizeBytes=" << totalElementSizeBytes
                         << " maxChunkSizeBytes=" << maxChunkSizeBytes << "\n";
        });

        return copyMemRefsRecursive(
            rewriter, copyOp.getLoc(), copyOp.getInput(), copyOp.getOutput(), maxChunkSizeBytes
        );
    }

  public:
    LogicalResult
    matchAndRewrite(torq_hl::HostCopyOp copyOp, PatternRewriter &rewriter) const override {

        auto inputType = copyOp.getInput().getType();
        auto outputType = copyOp.getOutput().getType();

        LLVM_DEBUG({
            llvm::dbgs() << "[LowerHostCopiesToNpu] Visiting host_copy: input=" << inputType
                         << " output=" << outputType << "\n";
        });

        // we only lower XRAM to XRAM operations (XRAM to LRAM copies must be done by
        // the host as they are used to setup the LRAM for the first NSS block execution)
        if (getEncodingMemorySpace(inputType) != torq_hl::MemorySpace::Xram ||
            getEncodingMemorySpace(outputType) != torq_hl::MemorySpace::Xram) {
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu] Skipping: not an XRAM->XRAM copy\n";
            });
            return failure();
        }

        // NPU DMA engine can process only up to 4 dimension for strided copies,
        // so we can't do the copy in LRAM if the input or output buffer has more than 4 dimensions
        // here we look at the strides on the host copy operation and not the shape of input/output
        // types because the types because some of those dimensions may be contiguous
        // and therefore the load/store op can ignore them
        if (copyOp.getInputStridesBytes().size() > 4 || copyOp.getOutputStridesBytes().size() > 4) {
            LLVM_DEBUG({
                llvm::dbgs() << "[LowerHostCopiesToNpu] Skipping: too many stride dimensions ("
                             << copyOp.getInputStridesBytes().size() << " input, "
                             << copyOp.getOutputStridesBytes().size() << " output)\n";
            });
            return failure();
        }

        // Use the LRAM budget computed for this specific op (total available minus all LRAM
        // allocations live at this point in the IR), so we do not over-commit LRAM that is
        // already reserved by surrounding ops.  Fall back to the hardware limit if the op
        // was not found in the budget map (should not happen in normal flow).
        auto budgetIt = lramBudget_.find(copyOp.getOperation());
        if (budgetIt == lramBudget_.end()) {
            LLVM_DEBUG({
                llvm::dbgs(
                ) << "[LowerHostCopiesToNpu] Warning: host copy op not found in LRAM budget map, "
                  << "falling back to hardware limit\n";
                copyOp.dump();
                llvm::dbgs() << "\nBudget map contains:" << lramBudget_.size() << " entries\n";
                for (const auto &entry : lramBudget_) {
                    entry.first->dump();
                    llvm::dbgs() << " budget: " << entry.second << "\n";
                }
            });
            return failure();
        }
        if (failed(convertToLramCopy(copyOp, budgetIt->second, rewriter))) {
            return failure();
        }

        // erase the original host copy operation
        rewriter.eraseOp(copyOp);

        LLVM_DEBUG({
            llvm::dbgs() << "[LowerHostCopiesToNpu] Replaced host_copy with load/store pair(s)\n";
        });

        return success();
    }
};

// Compute the memory budget for each HostCopyOp in the function. This is currently based on the
// allocations that maybe needed for the next operation. This is used to decide when we
// need to swap out values to stay within the memory budget. The default is set to half the
// available memory. This is a conservative estimate, and we are seeing if a larger memory than the
// default memory can be calculated and safely used. The memory space for which we want to compute
// the budget can be specified with the `memorySpace` parameter, only tested with LRAM for now.
DenseMap<Operation *, int64_t>
computeMemoryBudgetForHostCopy(mlir::FunctionOpInterface funcOp, const int64_t availableBytes) {
    DenseMap<Operation *, int64_t> lramBudget;

    Operation *lastHostCopyOp = nullptr;

    // The Allocator would allocate if the sum of all tensor sizes of single op is less than LRAM,
    // so we check the next kernel op after the HostCopy to see which needs to be alive and keep
    // only those. For now we accept only the inputs since this can be later used for parallel DMA
    // IN. Rest can be swapped out by the allocator.
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<torq_hl::HostCopyOp>(op)) {
            lastHostCopyOp = op;
            lramBudget[lastHostCopyOp] = availableBytes / 2; // Default to half the size, useful if
                                                             // no kernelOp follows the HostCopyOp
            return WalkResult::skip();
        }
        if (isa<torq_hl::KernelInterface>(op) && lastHostCopyOp) {
            auto dpsOp = cast<DestinationStyleOpInterface>(op);
            int64_t totalAllocBytes = 0;
            for (auto operand : dpsOp.getDpsInputOperands()) {
                if (auto memrefType = dyn_cast<MemRefType>(operand->get().getType())) {
                    if (getEncodingMemorySpace(memrefType) == torq_hl::MemorySpace::Lram) {
                        totalAllocBytes += getEncodedTotalSizeBytes(memrefType);
                    }
                }
            }
            lramBudget[lastHostCopyOp] =
                std::max(availableBytes - totalAllocBytes, availableBytes / 2);
            lastHostCopyOp = nullptr;
            return WalkResult::skip();
        }
        return WalkResult::advance();
    });

    return lramBudget;
}

class LowerHostCopiesToNpuPass : public impl::LowerHostCopiesToNpuBase<LowerHostCopiesToNpuPass> {
  public:
    LowerHostCopiesToNpuPass(const LowerHostCopiesToNpuOptions &options) {
        this->lramSize = options.lramSize;
    }
    LowerHostCopiesToNpuPass(const LowerHostCopiesToNpuPass &pass) {
        this->lramSize = pass.lramSize;
    }

    void runOnOperation() override;
};

void LowerHostCopiesToNpuPass::runOnOperation() {
    assert(lramSize > 0 && "LRAM size must be a positive integer");
    auto funcOp = getOperation();

    LLVM_DEBUG({
        llvm::dbgs() << "Running LowerHostCopiesToNpuPass on function " << funcOp.getName()
                     << " with lramSize " << lramSize << "\n";
    });
    // Walk the function in program order, tracking cumulative LRAM allocations and
    // deallocations, and record the remaining LRAM budget at each HostCopyOp.  This
    // budget is used by XramToXramHostCopyPattern so that maxChunkSizeBytes reflects
    // the memory actually available at that point rather than the raw hardware limit.
    auto lramBudget = computeMemoryBudgetForHostCopy(funcOp, lramSize);
    LLVM_DEBUG({
        llvm::dbgs() << "[LowerHostCopiesToNpu] Running on function: " << funcOp.getName() << "\n";
    });

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<XramToXramHostCopyPattern>(ctx, lramBudget);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerHostCopiesToNpuPass(int64_t lramSize
) {
    return std::make_unique<LowerHostCopiesToNpuPass>(LowerHostCopiesToNpuOptions{lramSize});
}

} // namespace mlir::syna::torq
