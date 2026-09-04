// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TilingUtils.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-hl-tile"

namespace mlir::syna::torq {

static llvm::cl::opt<int64_t> clTorqHLTileInputSizeThreshold(
    "torq-hl-tile-input-size-threshold",
    llvm::cl::desc("Tile TorqHL ops whose any single input tensor element count exceeds this "
                   "value; 0 disables the pass"),
    llvm::cl::init(0)
);

static llvm::cl::opt<int64_t> clTorqHLTileDimSize(
    "torq-hl-tile-dim-size",
    llvm::cl::desc("Fixed tile size applied to each parallel domain by TorqHLTileLargeInputs"),
    llvm::cl::init(0)
);

namespace {

/// Returns true if \\p op is a TorqHL compute op whose any single DPS input
/// tensor exceeds \\p threshold elements.
static bool isTorqHLLargeInputCandidate(Operation *op, int64_t threshold) {
    if (!isa<
            torq_hl::Conv2DOp, torq_hl::DepthwiseConv2DOp, torq_hl::Conv1DOp, torq_hl::MatMulOp,
            torq_hl::FullyConnectedOp, torq_hl::FMAOp, torq_hl::AvgPool2DOp, torq_hl::MaxPool2dOp>(
            op
        ))
        return false;

    auto dpsOp = cast<DestinationStyleOpInterface>(op);
    for (auto [idx, input] : llvm::enumerate(dpsOp.getDpsInputs())) {
        auto tensorTy = dyn_cast<RankedTensorType>(input.getType());
        if (!tensorTy) {
            LLVM_DEBUG(
                llvm::dbgs() << "isTorqHLLargeInputCandidate: " << op->getName() << " input[" << idx
                             << "] not a tensor, skipping\n"
            );
            continue;
        }
        if (!tensorTy.hasStaticShape()) {
            LLVM_DEBUG(
                llvm::dbgs() << "isTorqHLLargeInputCandidate: " << op->getName() << " input[" << idx
                             << "] dynamic shape, skipping\n"
            );
            continue;
        }
        int64_t numElements = tensorTy.getNumElements();
        LLVM_DEBUG(
            llvm::dbgs() << "isTorqHLLargeInputCandidate: " << op->getName() << " input[" << idx
                         << "] numElements=" << numElements << " threshold=" << threshold << "\n"
        );
        auto convertOp = input.getDefiningOp<torq_hl::ConvertOp>();
        if (!convertOp)
            continue;

        if (getEncodingMemorySpace(convertOp.getInit().getType()) != torq_hl::MemorySpace::Lram)
            continue;
        if (numElements > threshold)
            return true;
    }
    return false;
}

class TorqHLTileLargeInputsPass
    : public impl::TorqHLTileLargeInputsBase<TorqHLTileLargeInputsPass> {
  public:
    mlir::OpPassManager localPm_;

    TorqHLTileLargeInputsPass() {
        localPm_ = mlir::OpPassManager(func::FuncOp::getOperationName());
        localPm_.addPass(mlir::createCanonicalizerPass());
    }

    TorqHLTileLargeInputsPass(const TorqHLTileLargeInputsPass &pass)
        : TorqHLTileLargeInputsBase(pass), localPm_(pass.localPm_) {}

    void runOnOperation() override {
        if (clTorqHLTileInputSizeThreshold == 0 || clTorqHLTileDimSize == 0) {
            LLVM_DEBUG(
                llvm::dbgs() << "TorqHLTileLargeInputs: disabled (threshold="
                             << clTorqHLTileInputSizeThreshold << " dimSize=" << clTorqHLTileDimSize
                             << "), skipping\n"
            );
            return;
        }

        LLVM_DEBUG(
            llvm::dbgs() << "TorqHLTileLargeInputs: threshold=" << clTorqHLTileInputSizeThreshold
                         << " dimSize=" << clTorqHLTileDimSize << "\n"
        );

        FunctionOpInterface funcOp = getOperation();
        IRRewriter rewriter(&getContext());
        DenseSet<Operation *> existingLoops;
        funcOp.walk([&](scf::ForOp forOp) { existingLoops.insert(forOp.getOperation()); });

        SmallVector<TilingInterface> candidates = collectCandidates(funcOp);
        LLVM_DEBUG(
            llvm::dbgs() << "TorqHLTileLargeInputs: " << candidates.size()
                         << " candidate(s) found\n"
        );
        for (TilingInterface tiOp : candidates)
            tileAndPeelCandidate(rewriter, tiOp);

        simplifyAndUnrollNewLoops(funcOp, rewriter, existingLoops);
        LLVM_DEBUG(llvm::dbgs() << "TorqHLTileLargeInputs: DONE\n");
    }

  private:
    static bool isOutputChannelDim(TilingInterface tiOp, unsigned i, unsigned rank) {
        Operation *o = tiOp.getOperation();
        if (isa<torq_hl::Conv2DOp, torq_hl::DepthwiseConv2DOp>(o))
            return i == 1;
        return true;
    }

    SmallVector<TilingInterface> collectCandidates(FunctionOpInterface funcOp) {
        SmallVector<TilingInterface> candidates;
        funcOp.walk([&](TilingInterface tiOp) {
            LLVM_DEBUG(
                llvm::dbgs() << "TorqHLTileLargeInputs: walk sees " << tiOp->getName() << "\n"
            );
            if (isTorqHLLargeInputCandidate(tiOp, clTorqHLTileInputSizeThreshold)) {
                LLVM_DEBUG({
                    llvm::dbgs() << "TorqHLTileLargeInputs: candidate " << tiOp->getName() << "\n";
                    tiOp->dump();
                });
                candidates.push_back(tiOp);
            }
        });
        return candidates;
    }

    SmallVector<OpFoldResult>
    computeTileSizes(TilingInterface tiOp, ArrayRef<Range> iterDomain, OpBuilder &builder) {
        auto loopTypes = tiOp.getLoopIteratorTypes();
        SmallVector<OpFoldResult> tileSizes(iterDomain.size(), builder.getIndexAttr(0));
        for (auto [i, type] : llvm::enumerate(loopTypes)) {
            if (type != utils::IteratorType::parallel ||
                !isOutputChannelDim(tiOp, i, iterDomain.size()))
                continue;
            std::optional<int64_t> domainSize = getConstantIntValue(iterDomain[i].size);
            if (!domainSize)
                continue;
            int64_t tileSize = std::min(clTorqHLTileDimSize.getValue(), *domainSize);
            if (tileSize < *domainSize)
                tileSizes[i] = builder.getIndexAttr(tileSize);
        }
        return tileSizes;
    }

    void tileAndPeelCandidate(IRRewriter &rewriter, TilingInterface tiOp) {
        OpBuilder builder(tiOp->getContext());
        SmallVector<Range> iterDomain = tiOp.getIterationDomain(builder);
        SmallVector<OpFoldResult> tileSizes = computeTileSizes(tiOp, iterDomain, builder);
        if (llvm::all_of(tileSizes, [](OpFoldResult ofr) {
                auto intVal = getConstantIntValue(ofr);
                return intVal && *intVal == 0;
            }))
            return;

        scf::SCFTilingOptions options;
        options.setTileSizes(tileSizes);
        rewriter.setInsertionPoint(tiOp);
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(rewriter, tiOp, options);
        if (failed(tilingResult)) {
            tiOp->emitWarning("TorqHLTileLargeInputs: tileUsingSCF failed, skipping op");
            return;
        }

        SmallVector<scf::ForOp> forOps;
        for (Operation *loop : tilingResult->loops)
            if (auto forOp = dyn_cast<scf::ForOp>(loop))
                forOps.push_back(forOp);

        // Replace uses before peeling so the epilogue loop's result is retained.
        rewriter.replaceOp(tiOp, tilingResult->replacements);
        linalg::peelLoops(rewriter, forOps);
    }

    static SmallVector<scf::ForOp>
    collectNewLoops(FunctionOpInterface funcOp, const DenseSet<Operation *> &existingLoops) {
        SmallVector<scf::ForOp> newLoops;
        funcOp.walk([&](scf::ForOp forOp) {
            if (!existingLoops.contains(forOp.getOperation()))
                newLoops.push_back(forOp);
        });
        return newLoops;
    }

    void simplifyAndUnrollNewLoops(
        FunctionOpInterface funcOp, IRRewriter &rewriter, const DenseSet<Operation *> &existingLoops
    ) {
        SmallVector<scf::ForOp> newLoops = collectNewLoops(funcOp, existingLoops);
        for (scf::ForOp forOp : newLoops)
            rewriteAffineOpInLoop(rewriter, forOp);
        (void)runPipeline(localPm_, funcOp);

        newLoops = collectNewLoops(funcOp, existingLoops);
        for (scf::ForOp forOp : newLoops)
            (void)mlir::loopUnrollFull(forOp);
        (void)runPipeline(localPm_, funcOp);
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTorqHLTileLargeInputsPass() {
    return std::make_unique<TorqHLTileLargeInputsPass>();
}

} // namespace mlir::syna::torq
