// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Codegen/PrefetchPassUtils.h"
#include "torq/Codegen/LramAllocation.h"
#include "torq/Codegen/Passes.h"

#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <set>

#define DEBUG_TYPE "torq-prefetch-inputs"

namespace mlir::syna::torq {

// Defined in CompileNSSInvocationsPass.cpp.
extern llvm::cl::opt<unsigned> clDmaInMtu;
extern llvm::cl::opt<unsigned> clDmaOutMtu;

/// True if the alloc lives in LRAM (the buffers we care about prefetching).
bool isLramAlloc(memref::AllocOp allocOp) {
    return getEncodingMemorySpace(allocOp.getType()) == torq_hl::MemorySpace::Lram;
}

/// Byte footprint of a statically-shaped LRAM alloc (0 for dynamic shapes,
/// which we conservatively never touch).
int64_t lramAllocBytes(memref::AllocOp allocOp) {
    auto memrefType = allocOp.getType();
    if (!memrefType.hasStaticShape()) {
        return 0;
    }
    return getEncodedTotalSizeBytes(memrefType);
}

/// Follow view/reshape ops back to the value that actually owns the storage.
Value resolveMemRefBase(Value memref) {
    while (Operation *defOp = memref.getDefiningOp()) {
        if (isDerivedMemRefOperation(defOp)) {
            memref = getDerivedMemRefBase(defOp).get();
            continue;
        }
        break;
    }
    return memref;
}

/// Return the underlying LRAM alloc backing `memref`, or null if there is none.
memref::AllocOp getBaseAlloc(Value memref) {
    Value base = resolveMemRefBase(memref);
    return base.getDefiningOp<memref::AllocOp>();
}

/// Return the first wait_program (after `startOp`) that waits on the same
/// invocation, or null. The IR is not mutated during planning, so walking the
/// block by IR order here is stable.
/// Collect the programs of the block in program order, in one scan: a wait completes
/// the program of the last start on its invocation.
SmallVector<Program> collectPrograms(ArrayRef<Operation *> order) {
    SmallVector<Program> programs;
    // The program of each started invocation that has no wait yet.
    llvm::DenseMap<Value, size_t> pending;
    for (Operation *op : order) {
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
            pending[startOp.getInvocation()] = programs.size();
            programs.push_back(Program{startOp, {}});
        }
        else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
            auto it = pending.find(waitOp.getInvocation());
            if (it != pending.end()) {
                programs[it->second].waitOp = waitOp;
                pending.erase(it);
            }
        }
    }
    // defensive: a start without a wait is not a program to schedule
    llvm::erase_if(programs, [](const Program &program) { return !program.waitOp; });
    LLVM_DEBUG(
        llvm::dbgs() << "[prefetch] collectPrograms found " << programs.size() << " programs\n"
    );
    return programs;
}

/// Ops that are pure setup and can always be relocated earlier without changing
/// semantics: view/reshape ops, constants, bindings, and the program-setup ops
/// (create_invocation just wires a program to an executor; program is
/// IsolatedFromAbove).
bool isPureProducer(Operation *op) {
    return isa<
        memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp, torq_hl::ConstOp,
        torq_hl::MapBindingOp, torq_hl::CreateInvocationOp, torq_hl::ProgramOp>(op);
}

/// Peak simultaneous LRAM bytes, deallocs already in place. Mirrors
/// checkPeakMemoryUsage: sum live allocations, no contiguity, no extra alignment.
static int64_t peakLramBytes(FunctionOpInterface funcOp) {
    int64_t current = 0, peak = 0;
    for (Operation &op : funcOp.getFunctionBody().front()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
            int64_t bytes = isLramAlloc(allocOp) ? lramAllocBytes(allocOp) : 0;
            if (bytes <= 0) {
                continue;
            }
            current += bytes;
            peak = std::max(peak, current);
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(&op)) {
            memref::AllocOp base = getBaseAlloc(deallocOp.getMemref());
            int64_t bytes = (base && isLramAlloc(base)) ? lramAllocBytes(base) : 0;
            if (bytes > 0) {
                current -= bytes;
            }
        }
    }
    return peak;
}

/// Cheap whole-order probe: clone, run AddDeallocation, then ask the global LRAM planner
/// whether this order fits without inserted swaps. Orders needing swaps proceed to the
/// real allocation measurement below, where their cost is compared against the baseline.
///
///  - capacity: peak live bytes within the usable size. AssignAddressesPass treats
///    going over as a compile error.
bool orderPlacesInLram(FunctionOpInterface funcOp, int64_t poolSize) {
    Operation *clonedOp = funcOp->clone();
    auto clone = cast<FunctionOpInterface>(clonedOp);

    PassManager pm(funcOp.getContext(), clone->getName().getStringRef());
    pm.addPass(createAddDeallocationPass());

    bool fits = false;
    if (succeeded(pm.run(clone))) {
        int reservedBytes = HwInfo::nss_max_program_size * 2;
        int64_t usable = (poolSize / 8) * 8 - reservedBytes;
        int64_t peak = peakLramBytes(clone);
        bool withinCapacity = peak <= usable;
        bool placeable = withinCapacity && lramPlacementFitsWithoutSwaps(clone, usable);
        fits = withinCapacity && placeable;
        LLVM_DEBUG(
            llvm::dbgs() << "[prefetch] peak " << peak << "/" << usable << " B -> "
                         << (!withinCapacity ? "OVER CAPACITY"
                             : placeable     ? "fits"
                                             : "needs swaps")
                         << "\n"
        );
    }
    clonedOp->erase();
    return fits;
}

/// Ground truth for a candidate order: run the allocation tail of the NSS pipeline,
/// the same builder the real pipeline uses, on a clone, and time what comes out,
/// swaps included. A failed allocation is not an error here, so its diagnostics are
/// swallowed.
AllocOutcome measureRealAllocation(FunctionOpInterface funcOp, int64_t poolSize) {
    AllocOutcome outcome;
    Operation *clonedOp = funcOp->clone();
    auto clone = cast<FunctionOpInterface>(clonedOp);
    {
        ScopedDiagnosticHandler swallow(funcOp.getContext(), [](Diagnostic &) {
            return success();
        });
        PassManager pm(funcOp.getContext(), clone->getName().getStringRef());
        addNssLramAllocationPasses(pm, /*isRealPipeline=*/true, poolSize);
        outcome.ok = succeeded(pm.run(clone));
    }
    if (outcome.ok) {
        outcome.simulatedMakespan = simulateStream(clone);
    }
    clonedOp->erase();
    return outcome;
}

int64_t simulateStream(FunctionOpInterface funcOp) {
    llvm::DenseMap<Operation *, int64_t> programCycles;
    computeProgramCycles(funcOp, programCycles);
    Block &block = funcOp.getFunctionBody().front();
    llvm::DenseSet<Value> codeBuffers;
    for (Operation &op : block) {
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(&op)) {
            for (Value code : startOp.getCodeSections()) {
                codeBuffers.insert(resolveMemRefBase(code));
            }
        }
    }
    auto transferCycles = [](Operation *op, ArrayRef<int64_t> shape, int64_t elementSize,
                             unsigned mtu) -> int64_t {
        if (auto attr = op->getAttrOfType<IntegerAttr>(kAsyncDurationAttr)) {
            return attr.getInt();
        }
        return estimateDmaCycles(
            transferBytes(shape, elementSize), dmaThroughputBytesPerCycle(mtu)
        );
    };
    int64_t t = 0, dmaFree = 0;
    llvm::DenseMap<Value, int64_t> endOfInvocation;
    for (Operation &op : block) {
        if (auto loadOp = dyn_cast<torq_hl::LoadOp>(&op)) {
            int64_t cycles =
                transferCycles(&op, loadOp.getShape(), loadOp.getElementSizeBytes(), clDmaInMtu);
            if (codeBuffers.contains(resolveMemRefBase(loadOp.getOutput()))) {
                cycles = std::min<int64_t>(cycles, 90);
            }
            t = std::max(t, dmaFree) + cycles;
            dmaFree = t;
        }
        else if (auto storeOp = dyn_cast<torq_hl::StoreOp>(&op)) {
            t = std::max(t, dmaFree) +
                transferCycles(&op, storeOp.getShape(), storeOp.getElementSizeBytes(), clDmaOutMtu);
            dmaFree = t;
        }
        else if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(&op)) {
            torq_hl::ProgramOp programOp = getProgramOp(startOp);
            int64_t cycles = programOp ? programCycles.lookup(programOp.getOperation()) : 0;
            endOfInvocation[startOp.getInvocation()] = t + cycles;
            t += 1;
        }
        else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(&op)) {
            t = std::max(t, endOfInvocation.lookup(waitOp.getInvocation()));
        }
    }
    return std::max(t, dmaFree);
}

/// Rewrite the block so its ops appear in exactly the sequence `order`.
void applyOrderToBlock(ArrayRef<Operation *> order) {
    for (size_t i = 1; i < order.size(); ++i) {
        order[i]->moveAfter(order[i - 1]);
    }
}

/// Trace a start_program back to the torq_hl.program it launches, or null.
torq_hl::ProgramOp getProgramOp(torq_hl::StartProgramOp startOp) {
    Value invocation = startOp.getInvocation();
    auto createOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>();
    if (!createOp) {
        return {};
    }
    return createOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();
}

/// Fill `programCycles` with each program's compute-cycle estimate, read straight
/// from the torq_hw.slice_profiling annotations written into every slice program body (max_cycle
/// per slice task). No clone or re-lowering: the slice programs are torq_hw with cycle annotations
/// by the time this pass runs. Returns false if no annotation is present, in which case there
/// is no window budget to fill and the caller leaves the block alone.
bool computeProgramCycles(
    FunctionOpInterface funcOp, llvm::DenseMap<Operation *, int64_t> &programCycles
) {
    bool sawProfiling = false;
    funcOp->walk([&](torq_hl::ProgramOp programOp) {
        int64_t cycles = 0;
        programOp.walk([&](torq_hw::SliceProfilingOp prof) {
            cycles += static_cast<int64_t>(prof.getMaxCycle());
            sawProfiling = true;
        });
        programCycles[programOp.getOperation()] = cycles;
    });
    return sawProfiling;
}

/// DMA transfer cycles for one load, read from the "torq-async-duration-cycles"
/// attribute. Returns 0 if the load is not annotated
int64_t dmaCyclesForLoad(torq_hl::LoadOp loadOp) {
    if (auto attr = loadOp->getAttrOfType<IntegerAttr>(kAsyncDurationAttr)) {
        return attr.getInt();
    }
    return 0;
}

} // namespace mlir::syna::torq
