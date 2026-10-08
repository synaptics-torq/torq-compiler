// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/LramAllocation.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ProfilingUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "torq-prefetch-constant-loads"

static llvm::cl::opt<bool> clPrefetchConstantLoads(
    "torq-constant-prefetch",
    llvm::cl::desc("Load constants into LRAM while the program before their consumer runs"),
    llvm::cl::init(true)
);

namespace mlir::syna::torq {

// Defined in CompileNSSInvocationsPass.cpp; set via --torq-dma-in-mtu / --torq-dma-out-mtu.
extern llvm::cl::opt<unsigned> clDmaInMtu;
extern llvm::cl::opt<unsigned> clDmaOutMtu;

// The torq_hl.const that `value` views, with the view ops on the way (nearest first).
static torq_hl::ConstOp getViewedConst(Value value, SmallVectorImpl<Operation *> &viewOps) {
    while (auto viewOp = value.getDefiningOp<ViewLikeOpInterface>()) {
        // a view op may produce results other than the view itself (see getViewBase)
        if (value != viewOp.getViewDest())
            return {};
        viewOps.push_back(viewOp);
        value = viewOp.getViewSource();
    }
    return value.getDefiningOp<torq_hl::ConstOp>();
}

// Estimated cycles the NSS spends on `op`: the DMA of a load, store or copy, or
// the slice tasks of the program a start_program launches. A CSS program counts as
// 0, since nothing estimates it.
static int64_t estimateNssCycles(Operation *op) {
    if (auto loadOp = dyn_cast<torq_hl::LoadOp>(op)) {
        uint64_t bytes = transferBytes(loadOp.getShape(), loadOp.getElementSizeBytes());
        return estimateDmaCycles(bytes, dmaThroughputBytesPerCycle(clDmaInMtu));
    }
    if (auto storeOp = dyn_cast<torq_hl::StoreOp>(op)) {
        uint64_t bytes = transferBytes(storeOp.getShape(), storeOp.getElementSizeBytes());
        return estimateDmaCycles(bytes, dmaThroughputBytesPerCycle(clDmaOutMtu));
    }
    if (auto copyOp = dyn_cast<memref::CopyOp>(op))
        return estimateCdmaCycles(cdmaTransferBytes(copyOp.getSource()));
    if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
        auto invocation = startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();
        Operation *program = invocation ? invocation.getProgram().getDefiningOp() : nullptr;
        if (!program)
            return 0;
        int64_t cycles = 0;
        program->walk([&](torq_hw::SliceTaskOp task) { cycles += estimateSliceTaskCycles(task); });
        return cycles;
    }
    return 0;
}

// A run of wait_program ops. The NSS idles there until the last of the programs
// ends, so a load moved in front of the run is free for up to `slack` cycles.
struct WaitRun {
    Operation *firstWait;
    int64_t index;
    int64_t slack;
};

// The wait runs of `body`, with the idle time of each one in the NSS order: loads,
// stores and copies keep the NSS busy, and a started program runs on its slice
// until its wait.
static SmallVector<WaitRun> findWaitRuns(Block &body, const DenseMap<Operation *, int64_t> &index) {
    SmallVector<WaitRun> runs;
    DenseMap<Value, int64_t> programEnd;
    int64_t time = 0, runStart = 0;
    for (Operation &op : body) {
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
            programEnd[startOp.getInvocation()] = time + estimateNssCycles(&op);
            continue;
        }
        if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
            if (!isa_and_present<torq_hl::WaitProgramOp>(op.getPrevNode())) {
                runs.push_back({&op, index.lookup(&op), 0});
                runStart = time;
            }
            time = std::max(time, programEnd.lookup(waitOp.getInvocation()));
            runs.back().slack = time - runStart;
            continue;
        }
        time += estimateNssCycles(&op);
    }
    return runs;
}

// A constant load and the ops that move with it: the views it reads that come
// after the new position, its alloc, and the load itself.
struct ConstLoad {
    SmallVector<Operation *> ops;
    int64_t allocIndex, loadIndex;
    int64_t bytes, cycles;
};

// Constant loads that may move: the alloc is used first by the load, and the load
// is not already behind a start_program (issued while a program runs).
static SmallVector<ConstLoad>
findConstLoads(Block &body, const DenseMap<Operation *, int64_t> &index) {
    SmallVector<ConstLoad> loads;
    for (auto loadOp : body.getOps<torq_hl::LoadOp>()) {
        auto allocOp = loadOp.getOutput().getDefiningOp<memref::AllocOp>();
        if (!allocOp || getEncodingMemorySpace(allocOp.getType()) != torq_hl::MemorySpace::Lram)
            continue;
        SmallVector<Operation *> viewOps;
        if (!getViewedConst(loadOp.getInput(), viewOps))
            continue;
        if (llvm::any_of(allocOp->getUsers(), [&](Operation *user) {
                return user != loadOp && index.lookup(user) < index.lookup(loadOp);
            }))
            continue;
        Operation *prev = allocOp->getPrevNode();
        while (prev && !isa<torq_hl::StartProgramOp, torq_hl::WaitProgramOp>(prev))
            prev = prev->getPrevNode();
        if (!isa_and_present<torq_hl::WaitProgramOp>(prev))
            continue;

        ConstLoad load;
        load.ops.append(viewOps.rbegin(), viewOps.rend());
        load.ops.push_back(allocOp);
        load.ops.push_back(loadOp);
        load.allocIndex = index.lookup(allocOp);
        load.loadIndex = index.lookup(loadOp);
        load.bytes = getEncodedTotalSizeBytes(allocOp.getType());
        load.cycles = estimateNssCycles(loadOp);
        loads.push_back(std::move(load));
    }
    return loads;
}

// True when every operand of the ops of `load` is defined in front of `wait`, other
// than the operands these ops produce.
static bool operandsDefinedBefore(const ConstLoad &load, Operation *wait) {
    return llvm::all_of(load.ops, [&](Operation *op) {
        return llvm::all_of(op->getOperands(), [&](Value operand) {
            Operation *def = operand.getDefiningOp();
            return !def || llvm::is_contained(load.ops, def) || def->isBeforeInBlock(wait);
        });
    });
}

namespace {

// Moves the LRAM load of a constant (weights, bias, scale) in front of an earlier
// wait_program, so the NSS runs that DMA while the program computes instead of
// after it. A load moves in front of the nearest earlier wait, or further back (at
// most kMaxDepth waits) to one where the NSS idles long enough to hide all of it.
// The idle time comes from the DMA and slice cycle estimates. A load moves only
// while the LRAM that its earlier alloc keeps live fits. The real allocator checks
// the result, and a tighter budget is tried when it fails or swaps more bytes than
// before. Added only to the real pipeline, so the tile-fit probe does not count on it.
class PrefetchConstantLoadsPass
    : public impl::PrefetchConstantLoadsBase<PrefetchConstantLoadsPass> {
  public:
    PrefetchConstantLoadsPass() = default;
    PrefetchConstantLoadsPass(const PrefetchConstantLoadsPass &pass) {}

    void runOnOperation() override {
        auto funcOp = getOperation();
        if (!clPrefetchConstantLoads || !funcOp.getFunctionBody().hasOneBlock())
            return;
        std::optional<int64_t> baseSwapBytes = lramSwapBytes(funcOp);
        if (!baseSwapBytes)
            return;
        Block &body = funcOp.getFunctionBody().front();

        // Live LRAM bytes after each op of the body.
        DenseMap<Operation *, int64_t> index;
        SmallVector<int64_t> live;
        int64_t current = 0;
        for (Operation &op : body) {
            if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
                if (getEncodingMemorySpace(allocOp.getType()) == torq_hl::MemorySpace::Lram)
                    current += getEncodedTotalSizeBytes(allocOp.getType());
            }
            else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
                auto type = cast<MemRefType>(deallocOp.getMemref().getType());
                if (getEncodingMemorySpace(type) == torq_hl::MemorySpace::Lram)
                    current -= getEncodedTotalSizeBytes(type);
            }
            index[&op] = live.size();
            live.push_back(current);
        }

        SmallVector<ConstLoad> loads = findConstLoads(body, index);
        SmallVector<WaitRun> waitRuns = findWaitRuns(body, index);
        if (loads.empty() || waitRuns.empty())
            return;

        const int64_t usable = TorqHw::get().getLramSize() - HwInfo::nss_max_program_size * 2;
        constexpr size_t kMaxDepth = 8;
        for (int percent : {100, 90, 75, 50}) {
            int64_t budget = usable * percent / 100;
            SmallVector<int64_t> planned = live;
            SmallVector<int64_t> slack =
                llvm::map_to_vector(waitRuns, [](const WaitRun &run) { return run.slack; });
            SmallVector<std::pair<const ConstLoad *, Operation *>> moves;
            for (const ConstLoad &load : loads) {
                // The wait runs before the alloc, nearest first.
                auto it = llvm::partition_point(waitRuns, [&](const WaitRun &run) {
                    return run.index < load.allocIndex;
                });
                // The nearest wait run, or an earlier one (at most kMaxDepth back)
                // that has the idle time to hide all of the load.
                std::optional<size_t> nearest, chosen;
                size_t first = it - waitRuns.begin();
                for (size_t run = first; run-- > 0 && first - run <= kMaxDepth;) {
                    int64_t from = waitRuns[run].index;
                    // The alloc is live from allocIndex already, so the move adds
                    // its bytes only in front of it.
                    auto range = ArrayRef(planned).slice(from, load.allocIndex - from);
                    if (*llvm::max_element(range) + load.bytes > budget ||
                        !operandsDefinedBefore(load, waitRuns[run].firstWait))
                        break;
                    if (!nearest)
                        nearest = run;
                    if (slack[run] >= load.cycles) {
                        chosen = run;
                        break;
                    }
                }
                if (!chosen)
                    chosen = nearest;
                LLVM_DEBUG(
                    llvm::dbgs() << "prefetch: load at " << load.loadIndex << " (" << load.bytes
                                 << " B, " << load.cycles << " cycles) -> "
                                 << (chosen ? waitRuns[*chosen].index : -1) << "\n"
                );
                if (!chosen)
                    continue;
                slack[*chosen] -= load.cycles;
                for (int64_t i = waitRuns[*chosen].index; i < load.allocIndex; ++i)
                    planned[i] += load.bytes;
                moves.emplace_back(&load, waitRuns[*chosen].firstWait);
            }
            if (moves.empty())
                return;

            // Each moved op goes back in front of the next op that stays, if the
            // allocator rejects the result.
            DenseSet<Operation *> moved;
            for (auto [load, wait] : moves)
                moved.insert(load->ops.begin(), load->ops.end());
            SmallVector<std::pair<Operation *, Operation *>> undo;
            for (Operation &op : body) {
                if (!moved.contains(&op))
                    continue;
                Operation *anchor = op.getNextNode();
                while (moved.contains(anchor))
                    anchor = anchor->getNextNode();
                undo.emplace_back(&op, anchor);
            }

            // A view that another load already moved further up stays there.
            for (auto [load, wait] : moves) {
                for (Operation *op : load->ops) {
                    if (wait->isBeforeInBlock(op))
                        op->moveBefore(wait);
                }
            }
            // The moves must not make the allocator swap more bytes than before.
            std::optional<int64_t> swapBytes = lramSwapBytes(funcOp);
            if (swapBytes && *swapBytes <= *baseSwapBytes)
                return;
            for (auto [op, anchor] : undo)
                op->moveBefore(anchor);
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPrefetchConstantLoadsPass() {
    return std::make_unique<PrefetchConstantLoadsPass>();
}

} // namespace mlir::syna::torq
