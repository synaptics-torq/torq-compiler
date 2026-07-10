// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// EliminateRedundantLramLoadsPass
// -------------------------------
// After OutlineSlicePrograms each slice kernel (torq_hl.start_program) is a
// self-contained invocation whose operands are DMA-loaded into fresh LRAM
// buffers. Consecutive kernels in one region therefore re-issue a torq_hl.load
// for the SAME read-only source (a torq_hl.const, or an input subview) into a
// new buffer every time, even though a prior load already holds that value in
// LRAM. Each redundant load is an XRAM->LRAM DMA transaction.
//
// This pass performs a conservative, block-local redundant-load elimination:
// within a block, if a source memref is loaded more than once and neither the
// source's base buffer nor the reused destination buffer is written by any
// NON-load op in that block, the later loads are dropped and their consumers
// reuse the first load's LRAM buffer.
//
// Writes are detected via DestinationStyleOpInterface (DPS inits), the explicit
// torq_hl.store/host_copy outputs, MemoryEffects writes, and — for
// torq_hl.start_program — by mapping each arg to the target program's block
// argument and checking whether that argument is written inside the program
// body. Anything not understood is treated conservatively as a write, so the
// transform never reuses a buffer whose contents may have changed.
//
// LRAM-size guard. Reusing a buffer keeps it resident from its first load
// to its last reuse instead of streaming it on demand. For a large read-only
// operand the repeated load is really the address allocator reloading from the
// original source into an otherwise-empty LRAM, so
// removing it can push the peak concurrent LRAM usage past lramSize
// and make address assignment fail. To stay safe the pass models per-block peak
// LRAM usage (buffer size over its live interval) and accepts a reuse only when
// it does not push the peak above the block's own baseline (bounded below by the
// usable LRAM size (lramSize), derived from the hardware LRAM size). See filterByLramSize.

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "torq-eliminate-redundant-lram-loads"

using namespace mlir::iree_compiler;

static llvm::cl::opt<bool> clDisableEliminateRedundantLramLoads(
    "torq-disable-eliminate-redundant-lram-loads",
    llvm::cl::desc("Disable the redundant-LRAM-load elimination pass"), llvm::cl::init(false)
);

namespace mlir::syna::torq {

namespace {

// Follow view-like producers (memref.subview / cast / reinterpret_cast) down to
// the root memref value, so that aliasing views resolve to the same base.
static Value getBaseBuffer(Value v) {
    while (auto viewOp = v.getDefiningOp<ViewLikeOpInterface>()) {
        Value src = viewOp.getViewSource();
        if (src == v)
            break;
        v = src;
    }
    return v;
}

// Map a value used inside a program body back to the index of the program's
// block argument it originates from (following views). std::nullopt if it is
// not derived from a block argument of `body`.
static std::optional<unsigned> blockArgIndexOf(Value v, Block &body) {
    v = getBaseBuffer(v);
    if (auto ba = dyn_cast<BlockArgument>(v))
        if (ba.getOwner() == &body)
            return ba.getArgNumber();
    return std::nullopt;
}

// Append the memref values a single op writes: DPS inits (KernelInterface),
// torq_hl.store/load/host_copy outputs, and explicit MemoryEffects writes. An op
// that is not understood conservatively contributes all of its memref operands,
// so a write is never missed.
static void collectWrittenValues(Operation *op, SmallVectorImpl<Value> &out) {
    bool understood = false;
    if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
        for (OpOperand &init : dps.getDpsInitsMutable())
            out.push_back(init.get());
        understood = true;
    }
    if (auto st = dyn_cast<torq_hl::StoreOp>(op)) {
        out.push_back(st.getOutput());
        understood = true;
    }
    if (auto ld = dyn_cast<torq_hl::LoadOp>(op)) {
        out.push_back(ld.getOutput());
        understood = true;
    }
    if (auto hc = dyn_cast<torq_hl::HostCopyOp>(op)) {
        out.push_back(hc.getOutput());
        understood = true;
    }
    if (auto eff = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        eff.getEffects(effects);
        for (auto &e : effects)
            if (isa<MemoryEffects::Write>(e.getEffect()) && e.getValue())
                out.push_back(e.getValue());
        understood = true;
    }
    if (!understood)
        for (Value o : op->getOperands())
            if (isa<MemRefType>(o.getType()))
                out.push_back(o);
}

// Indices of a program's block arguments that the program body may write.
static llvm::DenseSet<unsigned> getWrittenProgramArgs(torq_hl::ProgramOp prog) {
    llvm::DenseSet<unsigned> written;
    if (prog.getBody().empty())
        return written;
    Block &body = prog.getBody().front();

    SmallVector<Value> vals;
    body.walk([&](Operation *op) {
        vals.clear();
        collectWrittenValues(op, vals);
        for (Value v : vals)
            if (auto idx = blockArgIndexOf(v, body))
                written.insert(*idx);
    });
    return written;
}

// Append the memref values written by a single op, excluding a torq_hl.load's
// own destination (that establishes residency, it does not clobber an existing
// resident buffer). A torq_hl.start_program writes the args that map to its
// program's written block arguments.
static void appendWrites(
    Operation *op, llvm::DenseMap<Operation *, llvm::DenseSet<unsigned>> &progCache,
    SmallVectorImpl<Value> &out
) {
    if (isa<torq_hl::LoadOp>(op))
        return;
    if (auto sp = dyn_cast<torq_hl::StartProgramOp>(op)) {
        auto inv = sp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();
        torq_hl::ProgramOp prog =
            inv ? inv.getProgram().getDefiningOp<torq_hl::ProgramOp>() : nullptr;
        OperandRange args = sp.getArgs();
        if (!prog) { // cannot analyse: conservatively treat every arg as written
            for (Value a : args)
                out.push_back(a);
            return;
        }
        auto it = progCache.find(prog);
        if (it == progCache.end())
            it = progCache.insert({prog, getWrittenProgramArgs(prog)}).first;
        const llvm::DenseSet<unsigned> &wargs = it->second;
        for (unsigned i = 0; i < args.size(); ++i)
            if (wargs.contains(i))
                out.push_back(args[i]);
        return;
    }
    collectWrittenValues(op, out);
}

// Two torq_hl.load ops read identical data only if they read the SAME source
// value (same subview identity, hence same offset/strides) with the same load
// attributes. Keying residency by the source's *base* buffer would wrongly merge
// distinct subviews of one binding (different offsets = different data).
static bool sameLoadShape(torq_hl::LoadOp a, torq_hl::LoadOp b) {
    return a.getElementSizeBytes() == b.getElementSizeBytes() &&
           a.getInputStridesBytes() == b.getInputStridesBytes() && a.getShape() == b.getShape() &&
           a.getUnsafe() == b.getUnsafe();
}

// A candidate reuse: `load`'s destination `dst` can be replaced by the already
// resident buffer `keep` (both hold identical data). Recorded during the
// correctness scan and applied later, after the lramSize filter.
struct ReuseCandidate {
    torq_hl::LoadOp load;
    Value keep;
    Value dst;
};

// The bytes an LRAM buffer occupies over its live interval [begin, end] in
// block-op order. Used to estimate peak concurrent LRAM usage.
struct LramInterval {
    int begin;
    int end;
    int64_t size;
};

// Last op index (in `idx`) that uses `v` or any view derived from it.
static int computeLastUseIndex(Value v, const llvm::DenseMap<Operation *, int> &idx) {
    int last = -1;
    SmallVector<Value> worklist{v};
    llvm::DenseSet<Value> seen;
    while (!worklist.empty()) {
        Value cur = worklist.pop_back_val();
        for (Operation *user : cur.getUsers()) {
            if (auto it = idx.find(user); it != idx.end())
                last = std::max(last, it->second);
            if (isa<ViewLikeOpInterface>(user))
                for (Value res : user->getResults())
                    if (seen.insert(res).second)
                        worklist.push_back(res);
        }
    }
    return last;
}

// Peak concurrent bytes across all live intervals (sweep over begin/end events).
static int64_t computePeakLramUsage(const llvm::DenseMap<Value, LramInterval> &intervals) {
    SmallVector<std::pair<int, int64_t>> events;
    events.reserve(intervals.size() * 2);
    for (auto &kv : intervals) {
        events.push_back({kv.second.begin, kv.second.size});
        events.push_back({kv.second.end + 1, -kv.second.size});
    }
    llvm::sort(events, [](auto &a, auto &b) { return a.first < b.first; });
    int64_t cur = 0, peak = 0;
    for (auto &e : events) {
        cur += e.second;
        peak = std::max(peak, cur);
    }
    return peak;
}

// From the data-flow-valid `candidates`, drop any reuse that would raise the
// block's peak LRAM usage above the ceiling max(`lramSize`, baseline peak).
// A reuse extends the resident buffer's live interval to the reuse point. Using
// the baseline peak as a floor keeps reuses the block already tolerates: a block
// whose baseline exceeds lramSize still compiles because the allocator streams
// the excess, and the extended buffer is not pinned across the spanned region so
// it stays swappable there. Large buffers whose reuse spans other large operands
// raise the peak and are left to stream (reloaded), matching the address
// allocator's own reload from the source.
static SmallVector<ReuseCandidate>
filterByLramSize(Block &block, ArrayRef<ReuseCandidate> candidates, int64_t lramSize) {
    // Index every op and collect the LRAM buffers allocated in the block.
    llvm::DenseMap<Operation *, int> idx;
    SmallVector<memref::AllocOp> lramAllocs;
    int n = 0;
    for (Operation &op : block) {
        idx[&op] = n++;
        if (auto alloc = dyn_cast<memref::AllocOp>(&op))
            if (getEncodingMemorySpace(alloc.getType()) == torq_hl::MemorySpace::Lram)
                lramAllocs.push_back(alloc);
    }

    // Baseline live interval of each LRAM buffer.
    llvm::DenseMap<Value, LramInterval> intervals;
    for (memref::AllocOp alloc : lramAllocs) {
        int begin = idx[alloc];
        int end = std::max(begin, computeLastUseIndex(alloc.getResult(), idx));
        intervals[alloc.getResult()] = {begin, end, getEncodedTotalSizeBytes(alloc.getType())};
    }

    int64_t ceiling = std::max<int64_t>(lramSize, computePeakLramUsage(intervals));

    SmallVector<ReuseCandidate> kept;
    for (const ReuseCandidate &c : candidates) {
        auto keepIt = intervals.find(c.keep);
        auto dstIt = intervals.find(c.dst);
        if (keepIt == intervals.end() || dstIt == intervals.end()) {
            kept.push_back(c); // not LRAM-tracked: lramSize irrelevant
            continue;
        }
        // Tentatively fold dst into keep (keep stays resident until dst's last
        // use, dst's buffer disappears); accept only if the peak stays <= ceiling.
        int oldEnd = keepIt->second.end;
        LramInterval dstInterval = dstIt->second;
        keepIt->second.end = std::max(oldEnd, dstInterval.end);
        intervals.erase(c.dst);

        if (computePeakLramUsage(intervals) <= ceiling) {
            kept.push_back(c);
        }
        else {
            keepIt->second.end = oldEnd; // revert: this buffer keeps streaming
            intervals[c.dst] = dstInterval;
        }
    }
    return kept;
}

// Apply accepted reuses: drop each redundant load, point its consumers at the
// resident buffer, and delete the now-dead destination alloc. Returns the count.
static unsigned applyReuses(ArrayRef<ReuseCandidate> candidates) {
    for (const ReuseCandidate &c : candidates) {
        torq_hl::LoadOp load = c.load;
        Value dst = c.dst;
        Operation *allocOp = dst.getDefiningOp();
        load.erase();
        dst.replaceAllUsesWith(c.keep);
        if (allocOp && isa<memref::AllocOp>(allocOp) && allocOp->use_empty())
            allocOp->erase();
        // Tag the now-shared resident buffer. A reuse keeps this read-only buffer
        // resident across the consumers it merged; if that later makes LRAM address
        // assignment fail, AssignLramAddresses reverts the reuse for the tagged
        // buffer (reloads the source per consumer). Safe to revert because the
        // reuse was only recorded for read-only buffers (never written in-block).
        if (Operation *keepAlloc = c.keep.getDefiningOp())
            keepAlloc->setAttr("torq.lram_reuse", UnitAttr::get(keepAlloc->getContext()));
    }
    return candidates.size();
}

// Order-aware, block-local redundant-load elimination.
//
// `resident` maps a source VALUE (exact subview/const identity) to the load that
// already brought it into LRAM. A later load of the SAME source value with the
// same load attributes is a reuse candidate whose consumers can share the
// resident buffer, PROVIDED:
//   1. The resident buffer's base is never written anywhere in the block
//      (whole-block check), so the value stays valid for every future reader
//      regardless of op ordering. Safe even for an in-place binding written by a
//      trailing store: the store targets the XRAM source, not the LRAM buffer.
//   2. The source's underlying buffer has not been written since residency was
//      established (forward-scan invalidation), so a reload after an in-place
//      write starts fresh instead of reusing a stale buffer.
//
// Data-flow-valid candidates are then filtered against lramSize
// (filterByLramSize) so a reuse never makes address assignment run out of
// LRAM, and the survivors are applied (applyReuses). A non-positive `lramSize`
// disables the lramSize filter (reuse whenever data flow allows it).
static unsigned dedupBlock(
    Block &block, llvm::DenseMap<Operation *, llvm::DenseSet<unsigned>> &progCache, int64_t lramSize
) {
    // Whole-block set of written base buffers (used for resident-buffer safety).
    llvm::DenseSet<Value> writtenBases;
    {
        SmallVector<Value> w;
        for (Operation &op : block) {
            w.clear();
            appendWrites(&op, progCache, w);
            for (Value v : w)
                writtenBases.insert(getBaseBuffer(v));
        }
    }

    // Phase 1: collect reuse candidates (correctness only, no IR mutation yet).
    llvm::DenseMap<Value, torq_hl::LoadOp> resident; // source value -> its load
    SmallVector<ReuseCandidate> candidates;
    SmallVector<Value> writes;

    for (Operation &op : block) {
        // Invalidate residency for any source whose underlying buffer is written
        // here (the source data, or an aliasing view of it, has changed).
        writes.clear();
        appendWrites(&op, progCache, writes);
        if (!writes.empty()) {
            SmallVector<Value> toDrop;
            for (Value v : writes) {
                Value wb = getBaseBuffer(v);
                for (auto &kv : resident)
                    if (getBaseBuffer(kv.first) == wb)
                        toDrop.push_back(kv.first);
            }
            for (Value k : toDrop)
                resident.erase(k);
        }

        auto ld = dyn_cast<torq_hl::LoadOp>(&op);
        if (!ld)
            continue;

        Value src = ld.getInput();
        Value dst = ld.getOutput();

        auto it = resident.find(src);
        if (it != resident.end() && it->second.getOutput().getType() == dst.getType() &&
            sameLoadShape(ld, it->second)) {
            candidates.push_back({ld, it->second.getOutput(), dst});
        }
        else if (!writtenBases.contains(getBaseBuffer(dst))) {
            // Only anchor on a buffer that is never written.
            resident[src] = ld;
        }
    }

    // Phase 2: drop reuses that would exceed lramSize. Phase 3: apply.
    if (lramSize > 0 && !candidates.empty())
        candidates = filterByLramSize(block, candidates, lramSize);
    return applyReuses(candidates);
}

class EliminateRedundantLramLoadsPass
    : public impl::EliminateRedundantLramLoadsBase<EliminateRedundantLramLoadsPass> {
  public:
    EliminateRedundantLramLoadsPass() = default;
    EliminateRedundantLramLoadsPass(const EliminateRedundantLramLoadsPass &) {}

    void runOnOperation() override {
        if (clDisableEliminateRedundantLramLoads)
            return;

        // lramSize (usable LRAM) = total LRAM minus the reserved NSS program pool,
        // matching the address allocator (AssignLramAddresses / VirtualMemory).
        int64_t lramSize =
            static_cast<int64_t>(TorqHw::get().getLramSize()) - HwInfo::nss_max_program_size * 2;

        FunctionOpInterface funcOp = getOperation();

        SmallVector<Block *> blocks;
        funcOp->walk([&](Block *b) { blocks.push_back(b); });

        llvm::DenseMap<Operation *, llvm::DenseSet<unsigned>> progCache;
        unsigned total = 0;
        for (Block *b : blocks)
            total += dedupBlock(*b, progCache, lramSize);

        LLVM_DEBUG(llvm::dbgs() << "EliminateRedundantLramLoads: removed " << total << " loads\n");
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createEliminateRedundantLramLoadsPass() {
    return std::make_unique<EliminateRedundantLramLoadsPass>();
}

} // namespace mlir::syna::torq
