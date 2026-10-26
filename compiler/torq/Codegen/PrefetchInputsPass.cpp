// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Software pipelining for the NSS control stream: overlap DMA with slice compute and
// run independent programs on both slices at once.
//
// The stream is in order: start_program is asynchronous (the slice computes while the
// stream goes on), wait_program blocks, and loads and stores block on the one DMA
// engine. The block becomes a dependency graph (StreamGraph): SSA, start_program ->
// wait_program, and every pair of LRAM or XRAM accesses to overlapping bytes with one
// write (a program holds its buffers until its wait). listSchedule then reissues it
// in order on a simulated clock: a free slice takes the oldest ready program, the DMA
// engine takes the oldest ready load or store if it would start before a running
// program ends, and otherwise the stream waits for the program that ends first. LRAM
// is tracked with the allocator's address-space policy; loads for programs further
// ahead may only fill part of it and must leave room for the largest program before
// their consumer.
//
// A schedule the real allocator places without swaps is kept. If none does (the
// input order itself is over LRAM), every schedule is priced after allocation, swaps
// included, and the fastest is kept if it beats the input order.
//
// See doc/dev-manual/prefetch_list_scheduler.md.

#include "PassesDetail.h"

#include "torq/Codegen/LramAddressSpace.h"
#include "torq/Codegen/PrefetchPassUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <optional>
#include <set>

#define DEBUG_TYPE "torq-prefetch-inputs"

namespace mlir::syna::torq {

static llvm::cl::opt<int> clListCapPercent(
    "torq-list-cap-percent",
    llvm::cl::desc("percent of usable LRAM the list scheduler may keep live; an input order that "
                   "already fills LRAM cannot be beaten under a smaller cap"),
    llvm::cl::init(100)
);

static llvm::cl::opt<int> clListPrefetchPercent(
    "torq-list-prefetch-percent",
    llvm::cl::desc("percent of the list scheduler's cap that loads for programs beyond the next "
                   "few may use"),
    llvm::cl::init(40)
);

static llvm::cl::opt<int> clListUrgentStarts(
    "torq-list-urgent-starts",
    llvm::cl::desc("loads for this many upcoming programs may use the full cap"), llvm::cl::init(4)
);

namespace {

/// The element range of its base buffer that a memref view covers, or the whole
/// buffer when the layout is not static.
struct Extent {
    Value base;
    int64_t lo = 0, hi = std::numeric_limits<int64_t>::max();
};

Extent extentOf(Value view) {
    Extent e;
    e.base = resolveMemRefBase(view);
    auto type = dyn_cast<MemRefType>(view.getType());
    if (!type || !type.hasStaticShape()) {
        return e;
    }
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(type.getStridesAndOffset(strides, offset)) || ShapedType::isDynamic(offset)) {
        return e;
    }
    int64_t hi = offset;
    for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
        if (ShapedType::isDynamic(stride) || stride < 0) {
            return e;
        }
        if (size == 0) {
            return e;
        }
        hi += (size - 1) * stride;
    }
    e.lo = offset;
    e.hi = hi;
    return e;
}

bool overlaps(const Extent &a, const Extent &b) {
    return a.base == b.base && a.lo <= b.hi && b.lo <= a.hi;
}

struct Access {
    Extent extent;
    bool write;
};

/// What a program touches in LRAM, from its start_program.
SmallVector<Access> programAccesses(torq_hl::StartProgramOp startOp) {
    SmallVector<Access> accesses;
    std::optional<ArrayAttr> kinds = startOp.getArgAccesses();
    for (auto [i, arg] : llvm::enumerate(startOp.getArgs())) {
        if (!isa<MemRefType>(arg.getType())) {
            continue;
        }
        bool write = true;
        if (kinds && i < kinds->size()) {
            if (auto kind = dyn_cast<torq_hl::ArgAccessBitfieldAttr>((*kinds)[i])) {
                write = bitEnumContainsAny(kind.getValue(), torq_hl::ArgAccessBitfield::Write);
            }
        }
        accesses.push_back({extentOf(arg), write});
    }
    for (Value code : startOp.getCodeSections()) {
        accesses.push_back({extentOf(code), false});
    }
    return accesses;
}

torq_hl::CreateInvocationOp invocationOf(torq_hl::StartProgramOp startOp) {
    return startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();
}

enum class NodeKind : uint8_t { Pure, Load, Store, Start, Wait, Barrier };

struct ListScheduleParams {
    int64_t cap;          // live LRAM bytes the schedule may hold
    int64_t prefetchCap;  // live bytes a load for a far program may push it to
    int64_t urgentStarts; // how many upcoming programs count as near
};

/// The block as a dependency graph. Pure ops (allocs, views, invocations, program
/// bodies, constants) are not scheduled; each is emitted just before the first
/// action that needs it, which is also where the allocator will see its buffer born.
class StreamGraph {
  public:
    StreamGraph(
        ArrayRef<Operation *> order, const llvm::DenseMap<Operation *, int64_t> &startCompute
    );

    size_t size() const { return ops_.size(); }

    SmallVector<Operation *> ops_;
    SmallVector<NodeKind> kind_;
    SmallVector<int64_t> dur_;
    SmallVector<SmallVector<unsigned>> actionPreds_; // through pure ops
    SmallVector<SmallVector<unsigned>> actionSuccs_;
    SmallVector<SmallVector<unsigned>> pureClosure_; // pure ops an action needs, sorted
    SmallVector<int> waitOf_, startOf_;
    SmallVector<int64_t> allocBytes_; // LRAM allocs with action users
    SmallVector<unsigned> allocUserCount_;
    SmallVector<SmallVector<unsigned>> userAllocs_; // action -> allocs it last-uses for
    SmallVector<SmallVector<Value>> startBuffers_;  // start -> bases of args and code
    SmallVector<Value> loadDst_;                    // load -> base of its destination
    SmallVector<int> firstConsumer_;                // load -> first start using its buffer
    Operation *terminator_ = nullptr;

  private:
    llvm::DenseMap<Operation *, unsigned> index_;
};

StreamGraph::StreamGraph(
    ArrayRef<Operation *> order, const llvm::DenseMap<Operation *, int64_t> &startCompute
) {
    for (Operation *op : order) {
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            terminator_ = op;
            continue;
        }
        index_[op] = ops_.size();
        ops_.push_back(op);
    }
    size_t n = ops_.size();
    kind_.assign(n, NodeKind::Barrier);
    dur_.assign(n, 0);
    waitOf_.assign(n, -1);
    startOf_.assign(n, -1);
    allocBytes_.assign(n, 0);
    allocUserCount_.assign(n, 0);
    userAllocs_.resize(n);
    startBuffers_.resize(n);
    loadDst_.resize(n);

    llvm::DenseSet<Value> codeBuffers;
    for (Operation *op : ops_) {
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
            for (Value code : startOp.getCodeSections()) {
                codeBuffers.insert(resolveMemRefBase(code));
            }
        }
    }
    auto isSlice = [](Value invocation) {
        auto type = dyn_cast<torq_hl::InvocationType>(invocation.getType());
        return type && type.getExecutor() == torq_hl::Executor::Slice;
    };
    llvm::DenseMap<Value, unsigned> startOfInvocation;
    for (unsigned i = 0; i < n; ++i) {
        Operation *op = ops_[i];
        if (isa<memref::AllocOp, arith::ConstantOp>(op) || isPureProducer(op)) {
            kind_[i] = NodeKind::Pure;
        }
        else if (auto loadOp = dyn_cast<torq_hl::LoadOp>(op)) {
            kind_[i] = NodeKind::Load;
            dur_[i] = dmaCyclesForLoad(loadOp);
            loadDst_[i] = resolveMemRefBase(loadOp.getOutput());
            // A code slot is sized for the largest program; the compiled code that
            // really moves is a few hundred bytes.
            if (codeBuffers.contains(loadDst_[i])) {
                dur_[i] = std::min<int64_t>(dur_[i], 90);
            }
        }
        else if (auto storeOp = dyn_cast<torq_hl::StoreOp>(op)) {
            kind_[i] = NodeKind::Store;
            if (auto attr = storeOp->getAttrOfType<IntegerAttr>("torq-async-duration-cycles")) {
                dur_[i] = attr.getInt();
            }
        }
        else if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
            if (isSlice(startOp.getInvocation()) &&
                startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>()) {
                kind_[i] = NodeKind::Start;
                dur_[i] = startCompute.lookup(op);
                startOfInvocation[startOp.getInvocation()] = i;
                for (Value arg : startOp.getArgs()) {
                    if (isa<MemRefType>(arg.getType())) {
                        startBuffers_[i].push_back(resolveMemRefBase(arg));
                    }
                }
                for (Value code : startOp.getCodeSections()) {
                    startBuffers_[i].push_back(resolveMemRefBase(code));
                }
            }
        }
        else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
            auto it = startOfInvocation.find(waitOp.getInvocation());
            if (isSlice(waitOp.getInvocation()) && it != startOfInvocation.end() &&
                waitOf_[it->second] < 0) {
                kind_[i] = NodeKind::Wait;
                waitOf_[it->second] = i;
                startOf_[i] = it->second;
            }
        }
    }
    // A start without a wait is not a program we can reason about.
    for (unsigned i = 0; i < n; ++i) {
        if (kind_[i] == NodeKind::Start && waitOf_[i] < 0) {
            kind_[i] = NodeKind::Barrier;
        }
    }

    SmallVector<llvm::SmallSetVector<unsigned, 4>> preds(n);
    // SSA
    for (unsigned i = 0; i < n; ++i) {
        ops_[i]->walk([&](Operation *nested) {
            for (Value operand : nested->getOperands()) {
                Operation *def = operand.getDefiningOp();
                if (!def) {
                    continue;
                }
                auto it = index_.find(def);
                if (it != index_.end() && it->second < i) {
                    preds[i].insert(it->second);
                }
            }
        });
    }
    for (unsigned i = 0; i < n; ++i) {
        if (kind_[i] == NodeKind::Wait) {
            preds[i].insert(startOf_[i]);
        }
    }
    // Barriers keep their place among the actions.
    int lastBarrier = -1;
    SmallVector<unsigned> sinceBarrier;
    for (unsigned i = 0; i < n; ++i) {
        if (kind_[i] == NodeKind::Pure) {
            continue;
        }
        if (kind_[i] == NodeKind::Barrier) {
            for (unsigned a : sinceBarrier) {
                preds[i].insert(a);
            }
            if (lastBarrier >= 0) {
                preds[i].insert(lastBarrier);
            }
            lastBarrier = i;
            sinceBarrier.clear();
            continue;
        }
        if (lastBarrier >= 0) {
            preds[i].insert(lastBarrier);
        }
        sinceBarrier.push_back(i);
    }
    // Memory: two accesses of overlapping ranges of one buffer, one of them a
    // write, keep their order; a program holds its buffers until its wait.
    struct Acc {
        unsigned op;
        int64_t lo, hi;
        bool write;
    };
    llvm::MapVector<Value, SmallVector<Acc>> byBase;
    auto addAcc = [&](unsigned i, Value view, bool write) {
        Extent e = extentOf(view);
        byBase[e.base].push_back({i, e.lo, e.hi, write});
    };
    for (unsigned i = 0; i < n; ++i) {
        Operation *op = ops_[i];
        if (kind_[i] == NodeKind::Load) {
            auto loadOp = cast<torq_hl::LoadOp>(op);
            addAcc(i, loadOp.getInput(), false);
            addAcc(i, loadOp.getOutput(), true);
        }
        else if (kind_[i] == NodeKind::Store) {
            auto storeOp = cast<torq_hl::StoreOp>(op);
            addAcc(i, storeOp.getInput(), false);
            addAcc(i, storeOp.getOutput(), true);
        }
        else if (kind_[i] == NodeKind::Start) {
            auto startOp = cast<torq_hl::StartProgramOp>(op);
            for (const Access &a : programAccesses(startOp)) {
                byBase[a.extent.base].push_back({i, a.extent.lo, a.extent.hi, a.write});
            }
        }
    }
    for (auto &entry : byBase) {
        SmallVector<Acc> &accs = entry.second;
        for (size_t b = 0; b < accs.size(); ++b) {
            for (size_t a = 0; a < b; ++a) {
                const Acc &x = accs[a], &y = accs[b];
                if (x.op == y.op || !(x.write || y.write) || x.lo > y.hi || y.lo > x.hi) {
                    continue;
                }
                unsigned src = x.op;
                if (kind_[src] == NodeKind::Start && static_cast<unsigned>(waitOf_[src]) < y.op) {
                    src = waitOf_[src];
                }
                if (src < y.op) {
                    preds[y.op].insert(src);
                }
            }
        }
    }

    firstConsumer_.assign(n, -1);
    {
        llvm::DenseMap<Value, unsigned> firstStartOfBuffer;
        for (unsigned i = 0; i < n; ++i) {
            if (kind_[i] == NodeKind::Start) {
                for (Value b : startBuffers_[i]) {
                    firstStartOfBuffer.try_emplace(b, i);
                }
            }
        }
        for (unsigned i = 0; i < n; ++i) {
            if (kind_[i] == NodeKind::Load) {
                auto it = firstStartOfBuffer.find(loadDst_[i]);
                if (it != firstStartOfBuffer.end() && it->second > i) {
                    firstConsumer_[i] = it->second;
                }
            }
        }
    }

    // Action-level view.
    actionPreds_.resize(n);
    actionSuccs_.resize(n);
    pureClosure_.resize(n);
    SmallVector<SmallVector<unsigned>> pureActionPreds(n), purePureClosure(n);
    for (unsigned i = 0; i < n; ++i) {
        llvm::SmallSetVector<unsigned, 8> ap, pc;
        for (unsigned p : preds[i]) {
            if (kind_[p] == NodeKind::Pure) {
                ap.insert(pureActionPreds[p].begin(), pureActionPreds[p].end());
                pc.insert(purePureClosure[p].begin(), purePureClosure[p].end());
                pc.insert(p);
            }
            else {
                ap.insert(p);
            }
        }
        SmallVector<unsigned> apv(ap.begin(), ap.end()), pcv(pc.begin(), pc.end());
        llvm::sort(pcv);
        if (kind_[i] == NodeKind::Pure) {
            pureActionPreds[i] = std::move(apv);
            purePureClosure[i] = std::move(pcv);
        }
        else {
            actionPreds_[i] = std::move(apv);
            pureClosure_[i] = std::move(pcv);
        }
    }
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned p : actionPreds_[i]) {
            actionSuccs_[p].push_back(i);
        }
    }

    // LRAM lifetimes: born at the alloc, dead after the last action that uses the
    // buffer (a program uses its buffers until its wait).
    llvm::DenseMap<Value, unsigned> allocOfBase;
    for (unsigned i = 0; i < n; ++i) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(ops_[i])) {
            if (isLramAlloc(allocOp) && lramAllocBytes(allocOp) > 0) {
                allocOfBase[allocOp.getResult()] = i;
            }
        }
    }
    SmallVector<llvm::SmallSetVector<unsigned, 4>> users(n);
    auto use = [&](unsigned user, Value view) {
        auto it = allocOfBase.find(resolveMemRefBase(view));
        if (it != allocOfBase.end()) {
            users[it->second].insert(user);
        }
    };
    for (unsigned i = 0; i < n; ++i) {
        Operation *op = ops_[i];
        if (kind_[i] == NodeKind::Load) {
            use(i, cast<torq_hl::LoadOp>(op).getOutput());
        }
        else if (kind_[i] == NodeKind::Store) {
            use(i, cast<torq_hl::StoreOp>(op).getInput());
        }
        else if (kind_[i] == NodeKind::Start) {
            auto startOp = cast<torq_hl::StartProgramOp>(op);
            for (Value v : startOp.getArgs()) {
                if (isa<MemRefType>(v.getType())) {
                    use(waitOf_[i], v);
                }
            }
            for (Value v : startOp.getCodeSections()) {
                use(waitOf_[i], v);
            }
        }
        else if (kind_[i] == NodeKind::Barrier) {
            for (Value v : op->getOperands()) {
                if (isa<MemRefType>(v.getType())) {
                    use(i, v);
                }
            }
        }
    }
    for (auto [base, a] : allocOfBase) {
        if (users[a].empty()) {
            continue;
        }
        allocBytes_[a] = lramAllocBytes(cast<memref::AllocOp>(ops_[a]));
        allocUserCount_[a] = users[a].size();
        for (unsigned u : users[a]) {
            userAllocs_[u].push_back(a);
        }
    }
}

/// Makespan of `order` under the scheduler's own timing model (loads and stores
/// block the stream, a program runs from its start until its wait), with the
/// executor ids the IR has now. Used to price the input order.
int64_t simulateOrder(const StreamGraph &g, ArrayRef<Operation *> order) {
    llvm::DenseMap<Operation *, unsigned> idx;
    for (unsigned i = 0; i < g.size(); ++i) {
        idx[g.ops_[i]] = i;
    }
    int64_t t = 0, dmaFree = 0;
    llvm::DenseMap<Operation *, int64_t> endOfStart;
    for (Operation *op : order) {
        auto it = idx.find(op);
        if (it == idx.end()) {
            continue;
        }
        unsigned i = it->second;
        switch (g.kind_[i]) {
        case NodeKind::Load:
        case NodeKind::Store:
            t = std::max(t, dmaFree) + g.dur_[i];
            dmaFree = t;
            break;
        case NodeKind::Start:
            endOfStart[op] = t + g.dur_[i];
            t += 1;
            break;
        case NodeKind::Wait:
            t = std::max(t, endOfStart.lookup(g.ops_[g.startOf_[i]]));
            break;
        default:
            break;
        }
    }
    return std::max(t, dmaFree);
}

/// Schedule the graph. Returns the new order of the block (terminator last) and
/// the slice of every program start.
SmallVector<Operation *> listSchedule(
    const StreamGraph &g, const ListScheduleParams &params,
    llvm::DenseMap<Operation *, unsigned> &sliceOfStart, int64_t &predictedMakespan,
    int64_t &peakLive
) {
    size_t n = g.size();
    SmallVector<bool> appended(n, false);
    SmallVector<unsigned> missing(n, 0);
    SmallVector<unsigned> remainingUsers(g.allocUserCount_.begin(), g.allocUserCount_.end());
    // Ready actions by kind, oldest first, so a step never walks the whole ready set
    // (thousands of weight loads are ready from the start on a large function).
    std::set<unsigned> readyStarts, readyDmas, readyWaits, readyBarriers, unstartedStarts;
    auto readySetOf = [&](unsigned i) -> std::set<unsigned> & {
        switch (g.kind_[i]) {
        case NodeKind::Start:
            return readyStarts;
        case NodeKind::Load:
        case NodeKind::Store:
            return readyDmas;
        case NodeKind::Wait:
            return readyWaits;
        default:
            return readyBarriers;
        }
    };
    size_t actions = 0;
    for (unsigned i = 0; i < n; ++i) {
        if (g.kind_[i] == NodeKind::Pure) {
            continue;
        }
        ++actions;
        missing[i] = g.actionPreds_[i].size();
        if (missing[i] == 0) {
            readySetOf(i).insert(i);
        }
        if (g.kind_[i] == NodeKind::Start) {
            unstartedStarts.insert(i);
        }
    }
    SmallVector<Operation *> out;
    out.reserve(n + 1);
    int64_t live = 0;
    peakLive = 0;
    int64_t t = 0, dmaFree = 0;
    constexpr unsigned kSlices = HwInfo::slice_count;
    int64_t sliceEnd[kSlices] = {};
    int sliceProg[kSlices];
    std::fill(std::begin(sliceProg), std::end(sliceProg), -1);
    llvm::DenseMap<unsigned, unsigned> progSlice;

    // Addresses, with the planner's policy, so a schedule that only fits in bytes
    // but not in contiguous ranges is not admitted.
    LramAddressSpace space(params.cap);
    llvm::DenseMap<unsigned, int64_t> offsetOf;
    auto alignedBytes = [&](unsigned a) {
        return static_cast<int64_t>(align_ceil((uint32_t)g.allocBytes_[a], 8u));
    };
    auto birthBytes = [&](unsigned i) {
        int64_t bytes = 0;
        for (unsigned p : g.pureClosure_[i]) {
            if (!appended[p]) {
                bytes += g.allocBytes_[p];
            }
        }
        return bytes;
    };
    // True if every buffer `i` would create gets an address now, and the bytes
    // leave `reserve` free under `limit`.
    auto fits = [&](unsigned i, int64_t limit, int64_t reserve) {
        if (live + birthBytes(i) + reserve > limit) {
            return false;
        }
        SmallVector<std::pair<int64_t, int64_t>> trial;
        bool ok = true;
        for (unsigned p : g.pureClosure_[i]) {
            if (appended[p] || g.allocBytes_[p] == 0) {
                continue;
            }
            int64_t size = alignedBytes(p);
            int64_t offset = space.allocate(size);
            if (offset < 0) {
                ok = false;
                break;
            }
            trial.push_back({offset, size});
        }
        for (auto [offset, size] : trial) {
            space.release(offset, size);
        }
        return ok;
    };
    auto append = [&](unsigned i) {
        for (unsigned p : g.pureClosure_[i]) {
            if (!appended[p]) {
                appended[p] = true;
                out.push_back(g.ops_[p]);
                live += g.allocBytes_[p];
                if (g.allocBytes_[p] > 0) {
                    int64_t offset = space.allocate(alignedBytes(p));
                    if (offset >= 0) {
                        offsetOf[p] = offset;
                    }
                }
            }
        }
        peakLive = std::max(peakLive, live);
        appended[i] = true;
        out.push_back(g.ops_[i]);
        readySetOf(i).erase(i);
        unstartedStarts.erase(i);
        for (unsigned s : g.actionSuccs_[i]) {
            if (--missing[s] == 0) {
                readySetOf(s).insert(s);
            }
        }
        for (unsigned a : g.userAllocs_[i]) {
            if (--remainingUsers[a] == 0 && appended[a]) {
                live -= g.allocBytes_[a];
                auto it = offsetOf.find(a);
                if (it != offsetOf.end()) {
                    space.release(it->second, alignedBytes(a));
                    offsetOf.erase(it);
                }
            }
        }
    };

    // Oldest-first candidates a step looks at; the policy takes the first that fits.
    constexpr int kMaxCandidates = 32;
    size_t done = 0;
    while (done < actions) {
        // Barriers go as soon as they can.
        if (!readyBarriers.empty()) {
            append(*readyBarriers.begin());
            ++done;
            continue;
        }

        int freeSlice = -1;
        bool anyRunning = false;
        int64_t earliestEnd = std::numeric_limits<int64_t>::max();
        int earliestSlice = -1;
        for (unsigned s = 0; s < kSlices; ++s) {
            if (sliceProg[s] < 0) {
                if (freeSlice < 0) {
                    freeSlice = s;
                }
            }
            else {
                anyRunning = true;
                if (sliceEnd[s] < earliestEnd) {
                    earliestEnd = sliceEnd[s];
                    earliestSlice = s;
                }
            }
        }

        // (1) A free slice takes the oldest ready program that fits.
        if (freeSlice >= 0) {
            unsigned start = UINT_MAX, forced = UINT_MAX;
            int examined = 0;
            for (unsigned i : readyStarts) {
                if (++examined > kMaxCandidates) {
                    break;
                }
                if (forced == UINT_MAX) {
                    forced = i;
                }
                if (fits(i, params.cap, 0)) {
                    start = i;
                    break;
                }
            }
            if (start == UINT_MAX && !anyRunning) {
                start = forced;
            }
            if (start != UINT_MAX) {
                append(start);
                ++done;
                sliceEnd[freeSlice] = t + g.dur_[start];
                sliceProg[freeSlice] = start;
                progSlice[start] = freeSlice;
                sliceOfStart[g.ops_[start]] = freeSlice;
                t += 1;
                continue;
            }
        }

        // (2) The DMA engine takes the oldest ready transfer that fits, if it would
        // start before any running program ends.
        llvm::DenseSet<Value> urgent;
        {
            int64_t k = 0;
            for (unsigned s : unstartedStarts) {
                if (k++ >= params.urgentStarts) {
                    break;
                }
                urgent.insert(g.startBuffers_[s].begin(), g.startBuffers_[s].end());
            }
        }
        // Largest footprint among the next upcoming programs, as a running max, so
        // each load's reserve is one lookup.
        SmallVector<std::pair<unsigned, int64_t>> upcomingMax; // (start, max so far)
        {
            int64_t runningMax = 0;
            for (unsigned s : unstartedStarts) {
                if (upcomingMax.size() >= 64) {
                    break;
                }
                runningMax = std::max(runningMax, birthBytes(s));
                upcomingMax.push_back({s, runningMax});
            }
        }
        unsigned dma = UINT_MAX, forcedDma = UINT_MAX;
        int examinedDmas = 0;
        for (unsigned i : readyDmas) {
            if (++examinedDmas > kMaxCandidates) {
                break;
            }
            if (forcedDma == UINT_MAX) {
                forcedDma = i;
            }
            int64_t limit = params.cap;
            int64_t reserve = 0;
            if (g.kind_[i] == NodeKind::Load) {
                if (!urgent.contains(g.loadDst_[i])) {
                    limit = params.prefetchCap;
                }
                // Leave room for the largest program that starts before this
                // load's consumer; the buffer stays live across all of them.
                int consumer = g.firstConsumer_[i];
                auto it = llvm::partition_point(upcomingMax, [&](const auto &e) {
                    return consumer < 0 || e.first < static_cast<unsigned>(consumer);
                });
                if (it == upcomingMax.end() && upcomingMax.size() >= 64 &&
                    (consumer < 0 || upcomingMax.back().first < static_cast<unsigned>(consumer))) {
                    reserve = params.cap; // too far ahead to reason about
                }
                else if (it != upcomingMax.begin()) {
                    reserve = std::prev(it)->second;
                }
            }
            if (g.kind_[i] == NodeKind::Store || fits(i, limit, reserve)) {
                dma = i;
                break;
            }
        }
        bool issueDma = dma != UINT_MAX && (!anyRunning || std::max(t, dmaFree) <= earliestEnd);
        unsigned waitPick = UINT_MAX;
        if (!issueDma && anyRunning) {
            int w = g.waitOf_[sliceProg[earliestSlice]];
            if (readyWaits.count(w)) {
                waitPick = w;
            }
            else if (!readyWaits.empty()) {
                waitPick = *readyWaits.begin();
            }
        }
        if (!issueDma && waitPick == UINT_MAX) {
            if (dma == UINT_MAX) {
                dma = forcedDma;
            }
            issueDma = dma != UINT_MAX;
        }
        if (issueDma) {
            append(dma);
            ++done;
            t = std::max(t, dmaFree);
            dmaFree = t + g.dur_[dma];
            t = dmaFree;
            continue;
        }
        if (waitPick != UINT_MAX) {
            unsigned s = progSlice.lookup(g.startOf_[waitPick]);
            append(waitPick);
            ++done;
            t = std::max(t, sliceEnd[s]);
            sliceProg[s] = -1;
            continue;
        }
        // Nothing fits the policy: take the oldest ready action.
        unsigned oldest = UINT_MAX;
        for (std::set<unsigned> *set : {&readyStarts, &readyDmas, &readyWaits}) {
            if (!set->empty()) {
                oldest = std::min(oldest, *set->begin());
            }
        }
        if (oldest == UINT_MAX) {
            break; // cannot happen on a DAG
        }
        append(oldest);
        ++done;
    }
    for (unsigned i = 0; i < n; ++i) {
        if (!appended[i]) {
            out.push_back(g.ops_[i]);
        }
    }
    if (g.terminator_) {
        out.push_back(g.terminator_);
    }
    predictedMakespan = std::max(t, dmaFree);
    return out;
}

} // namespace

class PrefetchInputsPass : public impl::PrefetchInputsBase<PrefetchInputsPass> {
  public:
    using impl::PrefetchInputsBase<PrefetchInputsPass>::PrefetchInputsBase;

    PrefetchInputsPass(const PrefetchInputsOptions &options) { this->lramSize = options.lramSize; }
    PrefetchInputsPass(const PrefetchInputsPass &pass) { this->lramSize = pass.lramSize; }

    void runOnOperation() override;
};

void PrefetchInputsPass::runOnOperation() {
    FunctionOpInterface funcOp = getOperation();
    Block &block = funcOp.getFunctionBody().front();

    // TOTAL LRAM size, not the usable budget: the allocator reserves the NSS program
    // slots itself. Falls back to full LRAM for a standalone pass run.
    int64_t poolSize =
        this->lramSize > 0 ? this->lramSize : static_cast<int64_t>(TorqHw::get().getLramSize());

    SmallVector<Operation *> originalOrder;
    for (Operation &op : block) {
        originalOrder.push_back(&op);
    }

    SmallVector<Program> programs = collectPrograms(originalOrder);
    LLVM_DEBUG(
        llvm::dbgs() << "[prefetch] " << funcOp.getName() << ": " << programs.size()
                     << " programs\n"
    );
    if (programs.size() < 2) {
        return; // nothing to overlap
    }

    // Compute cycles come from the slice_profiling annotations already in the IR.
    llvm::DenseMap<Operation *, int64_t> programCycles;
    if (!computeProgramCycles(funcOp, programCycles)) {
        LLVM_DEBUG(llvm::dbgs() << "[prefetch] no cost model; skipping block\n");
        return;
    }
    llvm::DenseMap<Operation *, int64_t> startCompute;
    for (const Program &program : programs) {
        int64_t cycles = 0;
        if (torq_hl::ProgramOp programOp = getProgramOp(torq_hl::StartProgramOp(program.startOp))) {
            cycles = programCycles.lookup(programOp.getOperation());
        }
        torq_hl::StartProgramOp startOp = program.startOp;
        startCompute[startOp.getOperation()] = cycles;
    }

    // Executor ids as they came in, so a rejected schedule can put them back.
    llvm::DenseMap<Operation *, uint64_t> inputIds;
    for (const Program &program : programs) {
        if (auto inv = invocationOf(program.startOp)) {
            if (inv.getExecutorId()) {
                inputIds[inv.getOperation()] = inv.getExecutorId()->getZExtValue();
            }
        }
    }
    auto restoreInputIds = [&]() {
        for (auto [op, id] : inputIds) {
            cast<torq_hl::CreateInvocationOp>(op).setExecutorId(APInt(64, id));
        }
    };
    auto applyIds = [&](const llvm::DenseMap<Operation *, unsigned> &sliceOfStart) {
        for (auto [startOp, slice] : sliceOfStart) {
            if (auto inv = invocationOf(cast<torq_hl::StartProgramOp>(startOp))) {
                inv.setExecutorId(APInt(64, slice));
            }
        }
    };

    StreamGraph graph(originalOrder, startCompute);
    const int64_t inputPredicted = simulateOrder(graph, originalOrder);

    // What a schedule has to beat. The prediction cannot see the swaps the allocator
    // adds, so it is the bar only when the input order places without them. Otherwise
    // the bar is the input order's real cost after allocation (efficientnet_b0 SiLU:
    // 116k cycles predicted, 272k allocated). The pass measures it once, and only if necessary.
    std::optional<int64_t> inputBar;
    auto inputCost = [&]() {
        if (!inputBar) {
            inputBar = inputPredicted;
            if (!orderPlacesInLram(funcOp, poolSize)) {
                AllocOutcome input = measureRealAllocation(funcOp, poolSize);
                if (input.ok)
                    inputBar = std::max(inputPredicted, input.simulatedMakespan);
            }
        }
        return *inputBar;
    };
    const int64_t usable = poolSize - HwInfo::nss_max_program_size * 2;
    const int64_t capPercents[] = {
        clListCapPercent, clListCapPercent * 85 / 100, clListCapPercent * 70 / 100
    };

    // The best schedule that only fits with the allocator's swaps, priced after allocation.
    struct Candidate {
        SmallVector<Operation *> order;
        llvm::DenseMap<Operation *, unsigned> sliceOfStart;
        int64_t makespan;
    };
    std::optional<Candidate> best;

    // Pass 0 takes the first schedule that places without swaps. Pass 1, only if none
    // did, prices each one after allocation, its swaps included.
    for (int pass = 0; pass < 2; ++pass) {
        for (int64_t capPercent : capPercents) {
            ListScheduleParams params;
            params.cap = usable * capPercent / 100;
            params.prefetchCap = params.cap * clListPrefetchPercent / 100;
            params.urgentStarts = clListUrgentStarts;
            llvm::DenseMap<Operation *, unsigned> sliceOfStart;
            int64_t predicted = 0, peak = 0;
            SmallVector<Operation *> scheduled =
                listSchedule(graph, params, sliceOfStart, predicted, peak);
            if (scheduled.size() != originalOrder.size() ||
                (predicted >= inputPredicted && predicted >= inputCost())) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[prefetch] cap " << capPercent << "%: predicted " << predicted
                                 << " cyc, input " << inputPredicted << " cyc; skipped\n"
                );
                continue;
            }
            applyIds(sliceOfStart);
            applyOrderToBlock(scheduled);
            if (pass == 0) {
                bool placed = orderPlacesInLram(funcOp, poolSize);
                LLVM_DEBUG(
                    llvm::dbgs() << "[prefetch] cap " << capPercent << "%: predicted " << predicted
                                 << " cyc vs input " << inputPredicted << ", peak " << peak
                                 << " B, " << (placed ? "placed" : "does not place") << "\n"
                );
                if (placed) {
                    return;
                }
            }
            else {
                AllocOutcome outcome = measureRealAllocation(funcOp, poolSize);
                LLVM_DEBUG(
                    llvm::dbgs() << "[prefetch] cap " << capPercent << "%, after allocation: "
                                 << (outcome.ok ? std::to_string(outcome.simulatedMakespan)
                                                : std::string("FAILED"))
                                 << " cyc\n"
                );
                if (outcome.ok && (!best || outcome.simulatedMakespan < best->makespan)) {
                    best = Candidate{scheduled, sliceOfStart, outcome.simulatedMakespan};
                }
            }
            applyOrderToBlock(originalOrder);
            restoreInputIds();
        }
    }
    if (!best) {
        return;
    }

    // Keep it only if it beats the input order, priced the same way.
    AllocOutcome input = measureRealAllocation(funcOp, poolSize);
    LLVM_DEBUG(
        llvm::dbgs() << "[prefetch] after allocation: schedule " << best->makespan << " vs input "
                     << (input.ok ? std::to_string(input.simulatedMakespan) : std::string("FAILED"))
                     << " cyc\n"
    );
    if (input.ok && input.simulatedMakespan <= best->makespan) {
        return;
    }
    applyIds(best->sliceOfStart);
    applyOrderToBlock(best->order);
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPrefetchInputsPass(int64_t lramSize) {
    return std::make_unique<PrefetchInputsPass>(PrefetchInputsOptions{lramSize});
}

} // namespace mlir::syna::torq
