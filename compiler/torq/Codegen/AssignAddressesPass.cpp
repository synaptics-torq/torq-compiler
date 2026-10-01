// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/Pool.h"
#include "torq/Codegen/VirtualMemory.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>
#include <queue>
#include <tuple>

#define DEBUG_TYPE "torq-assign-addresses"

static llvm::cl::opt<int> clXramStartAddress(
    "torq-xram-start-address", llvm::cl::desc("XRAM Start Address"),
    llvm::cl::init(0x100000) // 1MB
);

// Reduce read-only LRAM load residency in functions whose address allocation would
// otherwise fail. Two actions share this one trigger (see runOnOperation): a
// whole-function clone is allocated first, and only when that clone fails do we
// (1) revert load-CSE reuses (tagged "torq.lram_reuse") that keep a shared buffer
// resident, and (2) narrow read-only full-buffer loads (a weight loaded whole and
// then sliced per-invocation with memref.subview) into per-use sub-loads, so only
// the used slice is resident. Functions that already fit are left untouched (both
// transforms would split one shared load into more concurrent loads and can make
// allocation worse).
static llvm::cl::opt<bool> clFailureDrivenRecovery(
    "torq-failure-driven-recovery",
    llvm::cl::desc("In functions whose LRAM allocation would fail, revert load-CSE reuses and "
                   "narrow read-only weight loads to reduce peak residency"),
    llvm::cl::init(true)
);

llvm::cl::opt<int> clMaxNssProgramsSize(
    "torq-max-nss-programs-size", llvm::cl::desc("Size of the XRAM reserved for NSS programs"),
    llvm::cl::init(mlir::syna::torq::HwInfo::xram_nss_programs_size)
);

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

const std::string OUT_OF_MEMORY_MESSAGE = "failed to allocate LRAM addresses";

// check if we peak memory usage exceeds the pool size before ever trying to find addresses
// (we may later fail allocation even if peak usage is below the pool size)
static LogicalResult
checkPeakMemoryUsage(FunctionOpInterface funcOp, torq_hl::MemorySpace memSpace, int memSize) {

    int currentMemUsage = 0;
    int peakMemUsage = 0;
    int overflowMemUsage = 0;
    Operation *overflowAlloc{};
    Operation *peakAlloc{};
    DenseSet<Value> overflowAllocatedValues;
    DenseSet<Value> peakAllocatedValues;

    DenseSet<Value> allocatedValues;
    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {

            if (getEncodingMemorySpace(allocOp.getType()) != memSpace) {
                continue;
            }
            allocatedValues.insert(allocOp);
            auto reserveSize = getEncodedTotalSizeBytes(allocOp.getType());
            currentMemUsage += reserveSize;
            if (currentMemUsage > memSize && !overflowAlloc) {
                overflowMemUsage = currentMemUsage;
                overflowAlloc = &op;
                overflowAllocatedValues = allocatedValues;
            }
            if (currentMemUsage > peakMemUsage) {
                peakMemUsage = currentMemUsage;
                peakAlloc = &op;
                peakAllocatedValues = allocatedValues;
            }
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
            if (getEncodingMemorySpace(deallocOp.getMemref().getType()) != memSpace) {
                continue;
            }
            allocatedValues.erase(deallocOp.getMemref());
            auto memRefType = mlir::cast<MemRefType>(deallocOp.getMemref().getType());
            auto releaseSize = getEncodedTotalSizeBytes(memRefType);
            currentMemUsage -= releaseSize;
        }
    }

    if (peakMemUsage <= memSize) {
        return success();
    }

    // Memory overflow
    llvm::errs() << "First overflow at operation : ";
    overflowAlloc->dump();
    llvm::errs() << "Memory allocated at first overflow:\n";
    for (auto value : overflowAllocatedValues) {
        auto type = cast<MemRefType>(value.getType());
        llvm::dbgs() << "   size: " << getEncodedTotalSizeBytes(type) << " value: ";
        value.dump();
    }
    llvm::errs() << "\nPeak operation at operation : ";
    peakAlloc->dump();
    llvm::errs() << "\nMemory allocated at peak:\n";
    for (auto value : peakAllocatedValues) {
        auto type = cast<MemRefType>(value.getType());
        llvm::dbgs() << "   size: " << getEncodedTotalSizeBytes(type) << " value: ";
        value.dump();
    }

    llvm::errs() << "Peak memory usage of " << peakMemUsage << " bytes at: ";
    peakAlloc->dump();
    return overflowAlloc->emitError() << "Memory usage: " << overflowMemUsage << " exceeds "
                                      << stringifyEnum(memSpace) << " size " << memSize << " bytes";
}

static LogicalResult setDerivedMemrefAddress(Operation *op) {

    assert(isDerivedMemRefOperation(op) && "expected a derived memref op");
    auto viewOp = cast<ViewLikeOpInterface>(op);

    // A reinterpret_cast shares its source's address, so it must not add an offset of its
    // own (that would have to be folded in here, which is unsupported).
    if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(op)) {
        for (int i = 0; i < reinterpretCast.getResultRank(); i++) {
            if (reinterpretCast.isDynamicOffset(i) || reinterpretCast.isDynamicSize(i)) {
                return reinterpretCast.emitOpError("dynamic offsets or sizes not supported");
            }
            if (reinterpretCast.getStaticOffset(i) != 0) {
                return reinterpretCast.emitOpError("non-zero static offsets not supported");
            }
        }
    }

    if (auto memorySpaceCast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
        auto srcSpace = getEncodingMemorySpace(memorySpaceCast.getSource().getType());
        auto dstSpace = getEncodingMemorySpace(memorySpaceCast.getDest().getType());

        // casts with different spaces are not supported for the moment
        if (srcSpace != dstSpace) {
            return memorySpaceCast.emitOpError(
                "memory space cast with different source and destination spaces not supported"
            );
        }
    }

    auto maybeBaseAddress = getAddress(viewOp.getViewSource());
    if (!maybeBaseAddress) {
        return op->emitOpError("source does not have an address assigned");
    }
    if (failed(setAddress(op, *maybeBaseAddress))) {
        return op->emitOpError("failed to set address");
    }

    return success();
}

// op's index in the function body, the time axis of the placement.
using OpIdx = int;
// index into the buffer list
using BufferIdx = int;

namespace {

// The half-open op ranges over which a buffer must be resident.
class ResidentRanges {
    // sorted and disjoint, so the range ends ascend as well
    SmallVector<std::pair<OpIdx, OpIdx>> ranges_;

    const std::pair<OpIdx, OpIdx> *firstPast(OpIdx opIdx) const {
        return llvm::partition_point(ranges_, [&](const std::pair<OpIdx, OpIdx> &range) {
            return range.second <= opIdx;
        });
    }

  public:
    // Add [start, end), merging it with the ranges it overlaps or touches.
    void add(OpIdx start, OpIdx end) {
        auto *first = llvm::partition_point(ranges_, [&](const std::pair<OpIdx, OpIdx> &range) {
            return range.second < start;
        });
        auto *last = first;
        for (; last != ranges_.end() && last->first <= end; ++last) {
            start = std::min(start, last->first);
            end = std::max(end, last->second);
        }
        ranges_.erase(first, last);
        ranges_.insert(first, {start, end});
    }

    bool contains(OpIdx opIdx) const {
        const auto *range = firstPast(opIdx);
        return range != ranges_.end() && range->first <= opIdx;
    }

    // The first op strictly after `opIdx` in a range, or INT_MAX.
    OpIdx nextAfter(OpIdx opIdx) const {
        const auto *range = firstPast(opIdx + 1);
        return range == ranges_.end() ? INT_MAX : std::max(range->first, opIdx + 1);
    }
};

// One LRAM buffer to place.
struct LramBuffer {
    // its alloc receives the address
    Value value;
    // byte size, rounded up to a multiple of 8
    int64_t size = 0;
    // lifetime [birth, death): the alloc's op and its dealloc's, or the op count
    // if it is never freed
    OpIdx birth = 0;
    OpIdx death = 0;
    // every op that has the buffer as an operand, and every [start_program, wait_program+1)
    // of an async program that accesses it
    ResidentRanges resident;
};

} // namespace

// Collect one LramBuffer per LRAM allocation in `funcOp`'s body.
static SmallVector<LramBuffer> collectLramBuffers(FunctionOpInterface funcOp) {

    Block &body = funcOp.getFunctionBody().front();
    const int opCount = body.getOperations().size();

    llvm::MapVector<Value, LramBuffer> allocToBuffer;
    // each start_program's op and accessed buffers, until its wait_program
    llvm::DenseMap<Value, std::pair<OpIdx, SmallVector<Value>>> openPrograms;
    for (auto [opIdx, op] : llvm::enumerate(body.getOperations())) {
        if (auto alloc = dyn_cast<memref::AllocOp>(&op)) {
            if (getEncodingMemorySpace(alloc.getType()) != torq_hl::MemorySpace::Lram) {
                continue;
            }
            LramBuffer buffer{
                .value = alloc.getResult(),
                .size = align_ceil((uint32_t)getEncodedTotalSizeBytes(alloc.getType()), 8u),
                .birth = (OpIdx)opIdx,
                .death = opCount,
                .resident = {},
            };
            allocToBuffer.insert({buffer.value, std::move(buffer)});
        }
        else if (auto dealloc = dyn_cast<memref::DeallocOp>(&op)) {
            auto it = allocToBuffer.find(dealloc.getMemref());
            if (it != allocToBuffer.end()) {
                it->second.death = (OpIdx)opIdx;
            }
        }
        else if (!isDerivedMemRefOperation(&op)) {
            // a view op does not force residency; the op that consumes the view does
            for (Value operand : op.getOperands()) {
                if (!isa<MemRefType>(operand.getType())) {
                    continue;
                }
                auto it = allocToBuffer.find(getViewBase(operand));
                if (it != allocToBuffer.end()) {
                    it->second.resident.add((OpIdx)opIdx, (OpIdx)opIdx + 1);
                }
            }
            if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(&op)) {
                // only the buffers the program accesses stay resident until the wait;
                // convertVirtualToPhysicalMemRefs pins the same set
                SmallVector<torq_hl::AsyncAccess> accesses;
                startOp.getAsyncAccesses(accesses);
                auto &[startPos, buffers] = openPrograms[startOp.getInvocation()];
                startPos = (OpIdx)opIdx;
                for (auto &access : accesses) {
                    buffers.push_back(access.buffer);
                }
            }
            else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(&op)) {
                auto it = openPrograms.find(waitOp.getInvocation());
                assert(it != openPrograms.end() && "wait_program without a matching start_program");
                for (Value buffer : it->second.second) {
                    auto ivIt = allocToBuffer.find(getViewBase(buffer));
                    if (ivIt != allocToBuffer.end()) {
                        ivIt->second.resident.add(it->second.first, (OpIdx)opIdx + 1);
                    }
                }
                openPrograms.erase(it);
            }
        }
    }

    return llvm::to_vector(allocToBuffer.values());
}

// A compaction forced by re-planning: the op it applies at, and the failing op whose pinned
// buffers it places first.
struct ForcedCompaction {
    OpIdx opIdx;
    OpIdx failingOpIdx;
};

namespace {

// A two-ended address space over [0, usable): blocks smaller than largeBufferBytes place
// bottom-up and the rest top-down (the Pool allocator's policy), so the small buffers
// churning between the large ones never fragment the space the large ones need.
class AddressSpace {
    static constexpr int64_t largeBufferBytes = 8 * 1024;

    // the free ranges; IntervalMap merges adjacent ones on insert, so release() never has to
    using FreeRanges = llvm::IntervalMap<int64_t, char, 16, llvm::IntervalMapHalfOpenInfo<int64_t>>;
    FreeRanges::Allocator allocator_;
    FreeRanges free_;

    // Remove [offset, offset + size), which lies at one end of `range`, from it.
    static void claim(FreeRanges::iterator &range, int64_t offset, int64_t size) {
        if (range.stop() - range.start() == size) {
            range.erase();
        }
        else if (offset == range.start()) {
            range.setStart(offset + size);
        }
        else {
            range.setStop(offset);
        }
    }

  public:
    explicit AddressSpace(int64_t usable) : free_(allocator_) { free_.insert(0, usable, 0); }

    // Returns the placed offset, or -1 if no free range fits `size`.
    int64_t allocate(int64_t size) {
        if (size < largeBufferBytes) {
            for (auto it = free_.begin(); it != free_.end(); ++it) {
                if (it.stop() - it.start() >= size) {
                    int64_t offset = it.start();
                    claim(it, offset, size);
                    return offset;
                }
            }
        }
        else {
            for (auto it = free_.end(); it != free_.begin();) {
                --it;
                if (it.stop() - it.start() >= size) {
                    int64_t offset = it.stop() - size;
                    claim(it, offset, size);
                    return offset;
                }
            }
        }
        return -1;
    }

    void release(int64_t offset, int64_t size) { free_.insert(offset, offset + size, 0); }
};

// The buffers resident in an address space: which ones, at what offset, how many bytes.
class Occupancy {
    ArrayRef<LramBuffer> buffers_;
    const int64_t usable_;
    AddressSpace space_;
    llvm::BitVector resident_;
    SmallVector<int64_t> offsets_;
    int64_t bytes_ = 0;

  public:
    Occupancy(ArrayRef<LramBuffer> buffers, int64_t usable)
        : buffers_(buffers), usable_(usable), space_(usable), resident_(buffers.size()),
          offsets_(buffers.size()) {}

    int64_t usable() const { return usable_; }
    int64_t bytes() const { return bytes_; }
    bool isResident(BufferIdx idx) const { return resident_.test(idx); }
    auto residents() const { return resident_.set_bits(); }
    int64_t offset(BufferIdx idx) const { return offsets_[idx]; }

    // Make `idx` resident; when no free range fits it, changes nothing and returns false.
    bool place(BufferIdx idx) {
        int64_t offset = space_.allocate(buffers_[idx].size);
        if (offset < 0) {
            return false;
        }
        resident_.set(idx);
        offsets_[idx] = offset;
        bytes_ += buffers_[idx].size;
        return true;
    }

    void release(BufferIdx idx) {
        space_.release(offsets_[idx], buffers_[idx].size);
        resident_.reset(idx);
        bytes_ -= buffers_[idx].size;
    }

    SmallVector<int64_t> takeOffsets() { return std::move(offsets_); }
};

// Why a placement failed.
struct PlacementFailure {
    enum class Cause { BytePressure, Fragmentation };
    Cause cause = Cause::BytePressure;
    // the op whose buffer could not be placed
    OpIdx opIdx = -1;
    // the buffers pinned there, the only ones left resident
    SmallVector<BufferIdx> pinned;
};

// A placement of every buffer: its offset from the LRAM base, and the swaps that make
// the layout fit.
struct Placement {
    SmallVector<int64_t> offsets;
    SwapSchedule schedule;
};

// Places each buffer at its birth, frees it at its death and places a spilled one again at
// its next use; see place() for how a placement failure is resolved.
class PlacementSimulation {
    // At an equal op, deaths are processed first, then reloads, then births: space freed
    // there is available to the buffers placed there.
    enum class EventKind { Death, Reload, Birth };
    struct Event {
        OpIdx opIdx;
        EventKind kind;
        BufferIdx buffer;
    };
    struct EventLater { // inverted for the min-heap
        bool operator()(const Event &a, const Event &b) const {
            return std::tie(b.opIdx, b.kind, b.buffer) < std::tie(a.opIdx, a.kind, a.buffer);
        }
    };

    ArrayRef<LramBuffer> buffers_;
    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
    Occupancy occupancy_;
    SwapSchedule schedule_;
    PlacementFailure failure_;

    PlacementSimulation(ArrayRef<LramBuffer> buffers, int64_t usable)
        : buffers_(buffers), occupancy_(buffers, usable) {
        for (auto [i, buffer] : llvm::enumerate(buffers_)) {
            events_.push({buffer.birth, EventKind::Birth, (BufferIdx)i});
            events_.push({buffer.death, EventKind::Death, (BufferIdx)i});
        }
    }

    // Schedule the spill of the released `idx` at `opIdx`, and its reload at its next use.
    void scheduleSpill(BufferIdx idx, OpIdx opIdx);
    void spill(BufferIdx victim, OpIdx opIdx);
    // Re-place every buffer not pinned at `opIdx`: those pinned at `pinnedFirstAt` first (so
    // they take the extremes and leave the middle contiguous), then largest first. A buffer
    // that no longer fits is spilled. False when nothing was movable.
    bool compact(OpIdx opIdx, OpIdx pinnedFirstAt);
    // Record that `idx` cannot be placed at `opIdx`: every resident buffer is pinned there.
    void recordFailure(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause);
    // Among the resident buffers not pinned at `opIdx`; -1 if there is none.
    BufferIdx farthestNextUseBuffer(OpIdx opIdx) const;
    // Spill the buffer needed last to make room for `idx`: any victim costs one reload at
    // its next use, and this one keeps the freed space available the longest before that
    // reload comes due. False when every resident buffer is pinned at `opIdx`.
    bool spillVictim(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause);
    // Place `idx` at `opIdx`. Spill under byte pressure, compact once, and then
    // spill under fragmentation. False when nothing more can be spilled.
    bool place(BufferIdx idx, OpIdx opIdx);
    // Sweep the events, compacting at each of `forcedCompactions` (ascending) as well.
    // False when the buffers pinned at some op cannot coexist.
    bool sweep(ArrayRef<ForcedCompaction> forcedCompactions);

  public:
    // Place `buffers` in [0, usable), or return nullopt and describe in `failure` why no
    // placement exists.
    static std::optional<Placement>
    run(ArrayRef<LramBuffer> buffers, int64_t usable, ArrayRef<ForcedCompaction> forcedCompactions,
        PlacementFailure &failure) {
        PlacementSimulation simulation(buffers, usable);
        if (!simulation.sweep(forcedCompactions)) {
            failure = simulation.failure_;
            return std::nullopt;
        }
        return Placement{simulation.occupancy_.takeOffsets(), std::move(simulation.schedule_)};
    }
};

} // namespace

void PlacementSimulation::scheduleSpill(BufferIdx idx, OpIdx opIdx) {
    schedule_[opIdx].push_back({Swap::Kind::Spill, {buffers_[idx].value}});
    OpIdx reloadOpIdx = buffers_[idx].resident.nextAfter(opIdx);
    if (reloadOpIdx != INT_MAX) {
        events_.push({reloadOpIdx, EventKind::Reload, idx});
    }
}

void PlacementSimulation::spill(BufferIdx victim, OpIdx opIdx) {
    occupancy_.release(victim);
    scheduleSpill(victim, opIdx);
}

bool PlacementSimulation::compact(OpIdx opIdx, OpIdx pinnedFirstAt) {
    SmallVector<BufferIdx> movable;
    for (BufferIdx candidate : occupancy_.residents()) {
        if (!buffers_[candidate].resident.contains(opIdx)) {
            movable.push_back(candidate);
        }
    }
    if (movable.empty()) {
        return false;
    }
    llvm::sort(movable, [&](BufferIdx a, BufferIdx b) {
        bool aFirst = buffers_[a].resident.contains(pinnedFirstAt);
        bool bFirst = buffers_[b].resident.contains(pinnedFirstAt);
        if (aFirst != bFirst) {
            return aFirst;
        }
        if (buffers_[a].size != buffers_[b].size) {
            return buffers_[a].size > buffers_[b].size;
        }
        return a < b;
    });
    for (BufferIdx moved : movable) {
        occupancy_.release(moved);
    }
    SmallVector<Value> relocated;
    for (BufferIdx moved : movable) {
        if (!occupancy_.place(moved)) {
            scheduleSpill(moved, opIdx);
            continue;
        }
        relocated.push_back(buffers_[moved].value);
    }
    if (!relocated.empty()) {
        schedule_[opIdx].push_back({Swap::Kind::Relocate, std::move(relocated)});
    }
    return true;
}

void PlacementSimulation::recordFailure(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause) {
    failure_.cause = cause;
    failure_.opIdx = opIdx;
    auto residents = occupancy_.residents();
    failure_.pinned.assign(residents.begin(), residents.end());
    LLVM_DEBUG({
        llvm::dbgs() << "LRAM plan infeasible ("
                     << (cause == PlacementFailure::Cause::Fragmentation ? "fragmentation"
                                                                         : "byte pressure")
                     << ") at op " << opIdx << ": cannot place buffer " << idx << " ("
                     << buffers_[idx].size << " bytes), resident " << occupancy_.bytes() << " of "
                     << occupancy_.usable() << " bytes\n";
        for (BufferIdx pinned : residents) {
            llvm::dbgs() << "  pinned buffer " << pinned << " size " << buffers_[pinned].size
                         << " offset " << occupancy_.offset(pinned) << "\n";
        }
    });
}

BufferIdx PlacementSimulation::farthestNextUseBuffer(OpIdx opIdx) const {
    BufferIdx result = -1;
    OpIdx farthest = -1;
    for (BufferIdx i : occupancy_.residents()) {
        if (buffers_[i].resident.contains(opIdx)) {
            continue;
        }
        OpIdx nextUse = buffers_[i].resident.nextAfter(opIdx);
        if (nextUse > farthest) {
            farthest = nextUse;
            result = i;
        }
    }
    return result;
}

bool PlacementSimulation::spillVictim(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause) {
    BufferIdx victim = farthestNextUseBuffer(opIdx);
    if (victim < 0) {
        recordFailure(idx, opIdx, cause);
        return false;
    }
    spill(victim, opIdx);
    return true;
}

bool PlacementSimulation::place(BufferIdx idx, OpIdx opIdx) {
    while (occupancy_.bytes() + buffers_[idx].size > occupancy_.usable()) {
        if (!spillVictim(idx, opIdx, PlacementFailure::Cause::BytePressure)) {
            return false;
        }
    }
    bool placed = occupancy_.place(idx);
    if (!placed && compact(opIdx, opIdx)) {
        placed = occupancy_.place(idx);
    }
    while (!placed) {
        if (!spillVictim(idx, opIdx, PlacementFailure::Cause::Fragmentation)) {
            return false;
        }
        placed = occupancy_.place(idx);
    }
    return true;
}

bool PlacementSimulation::sweep(ArrayRef<ForcedCompaction> forcedCompactions) {
    while (!events_.empty()) {
        Event event = events_.top();
        events_.pop();
        // a forced compaction precedes the events of the op it applies at
        while (!forcedCompactions.empty() && forcedCompactions.front().opIdx <= event.opIdx) {
            compact(forcedCompactions.front().opIdx, forcedCompactions.front().failingOpIdx);
            forcedCompactions = forcedCompactions.drop_front();
        }
        if (event.kind == EventKind::Death) {
            // not resident when it was spilled with no reload before dying
            if (occupancy_.isResident(event.buffer)) {
                occupancy_.release(event.buffer);
            }
        }
        else if (!place(event.buffer, event.opIdx)) {
            return false;
        }
    }
    return true;
}

// Assign each allocation its address; the views derived from it inherit theirs. Forward
// order suffices: a view always follows the value it derives from.
static LogicalResult commitLramLayout(
    FunctionOpInterface funcOp, int64_t reservedBase, ArrayRef<LramBuffer> buffers,
    ArrayRef<int64_t> offsets
) {
    for (auto [buffer, offset] : llvm::zip_equal(buffers, offsets)) {
        if (failed(setAddress(buffer.value, reservedBase + offset))) {
            return buffer.value.getDefiningOp()->emitOpError("failed to set LRAM address");
        }
    }
    for (Operation &op : funcOp.getFunctionBody().front()) {
        if (!isDerivedMemRefOperation(&op)) {
            continue;
        }
        auto type = dyn_cast<MemRefType>(op.getResult(0).getType());
        if (!type || getEncodingMemorySpace(type) != torq_hl::MemorySpace::Lram) {
            continue;
        }
        if (failed(setDerivedMemrefAddress(&op))) {
            return failure();
        }
    }
    return success();
}

static LogicalResult
allocateAddresses(FunctionOpInterface funcOp, Pool &pool, torq_hl::MemorySpace memSpace) {

    auto status = checkPeakMemoryUsage(funcOp, memSpace, pool.usableSize());

    if (failed(status)) {
        return status;
    }

    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {

            if (getEncodingMemorySpace(allocOp.getType()) != memSpace) {
                continue;
            }

            FailureOr<int64_t> maybeAddr = pool.allocate(op.getResult(0));

            if (failed(maybeAddr)) {
                return op.emitError(
                    "Failed to allocate " + stringifyMemorySpace(memSpace) + " buffer of size " +
                    std::to_string(getEncodedTotalSizeBytes(allocOp.getType()))
                );
            }

            if (failed(setAddress(&op, *maybeAddr))) {
                return op.emitError("Failed to set " + stringifyMemorySpace(memSpace) + " address");
            }
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
            if (getEncodingMemorySpace(deallocOp.getMemref().getType()) != memSpace) {
                continue;
            }
            pool.free(deallocOp.getMemref());
        }
        else if (isDerivedMemRefOperation(&op)) {

            auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

            if (getEncodingMemorySpace(type) != memSpace) {
                continue;
            }

            if (failed(setDerivedMemrefAddress(&op))) {
                return failure();
            }
        }
    }

    return success();
}

// Where to force a compaction so that a pinned obstacle of `failure` stops splitting the
// space: the last op before the failure at which the largest obstacle was still movable
// (resident but not pinned). Ops already forced are skipped, obstacle by obstacle. Returns
// nullopt when nothing is left to try.
static std::optional<OpIdx> lastMovablePosition(
    ArrayRef<LramBuffer> buffers, const PlacementFailure &failure, ArrayRef<ForcedCompaction> forced
) {
    SmallVector<BufferIdx> obstacles(failure.pinned);
    llvm::sort(obstacles, [&](BufferIdx a, BufferIdx b) {
        if (buffers[a].size != buffers[b].size) {
            return buffers[a].size > buffers[b].size;
        }
        return a < b;
    });
    for (BufferIdx obstacle : obstacles) {
        for (OpIdx opIdx = failure.opIdx - 1; opIdx >= buffers[obstacle].birth; opIdx--) {
            if (!buffers[obstacle].resident.contains(opIdx)) {
                bool tried = llvm::any_of(forced, [&](const ForcedCompaction &compaction) {
                    return compaction.opIdx == opIdx;
                });
                if (!tried) {
                    return opIdx;
                }
                break; // its latest movable op was already tried
            }
        }
    }
    return std::nullopt;
}

// The offsets and swaps that place `buffers` in [0, usable), or nullopt when none exist.
// With `replan`, a fragmentation failure forces a compaction where lastMovablePosition says
// and the placement is planned again, once per attempt.
static std::optional<Placement>
planPlacement(ArrayRef<LramBuffer> buffers, int64_t usable, bool replan) {
    // each attempt adds one forced compaction
    constexpr int maxReplans = 8;
    SmallVector<ForcedCompaction> forcedCompactions;
    for (int attempt = 0; attempt <= maxReplans; attempt++) {
        PlacementFailure failure;
        auto placement = PlacementSimulation::run(buffers, usable, forcedCompactions, failure);
        if (placement) {
            return placement;
        }
        if (!replan || failure.cause != PlacementFailure::Cause::Fragmentation) {
            return std::nullopt;
        }
        auto compactAt = lastMovablePosition(buffers, failure, forcedCompactions);
        if (!compactAt) {
            return std::nullopt;
        }
        LLVM_DEBUG(
            llvm::dbgs() << "re-planning with a forced compaction before op " << *compactAt
                         << " for op " << failure.opIdx << "\n"
        );
        ForcedCompaction compaction{*compactAt, failure.opIdx};
        forcedCompactions.insert(
            llvm::upper_bound(
                forcedCompactions, compaction,
                [](const ForcedCompaction &a, const ForcedCompaction &b) {
                    return a.opIdx < b.opIdx;
                }
            ),
            compaction
        );
    }
    return std::nullopt;
}

// Lay out every LRAM allocation over its whole lifetime; when that does not fit, schedule
// swaps, perform them with convertVirtualToPhysicalMemRefs and lay out the rewritten IR.
// Infeasibility (the buffers pinned at some op cannot coexist) fails without a
// diagnostic, so runOnOperation emits the OUT_OF_MEMORY_MESSAGE tile-and-fuse's probe
// shrinks the tile on; any other diagnostic aborts the compile.
static LogicalResult allocateLramAddresses(FunctionOpInterface funcOp, bool replan) {

    // we reserve space for two NSS programs at the begininng of LRAM
    // (one block for the currently executing program and one for the
    // next program to be executed that will be loaded by the current one)
    int64_t reservedBase = HwInfo::nss_max_program_size * 2;
    const int64_t usable = TorqHw::get().getLramSize() - reservedBase;

    SmallVector<LramBuffer> buffers = collectLramBuffers(funcOp);
    std::optional<Placement> placement = planPlacement(buffers, usable, replan);
    if (!placement) {
        return failure();
    }
    if (placement->schedule.empty()) {
        return commitLramLayout(funcOp, reservedBase, buffers, placement->offsets);
    }

    if (failed(convertVirtualToPhysicalMemRefs(
            funcOp, usable, torq_hl::MemorySpace::Lram, placement->schedule
        ))) {
        return failure();
    }

    // TODO: the swap schedule planPlacement returns should always fit once
    // convertVirtualToPhysicalMemRefs applies it, so reaching this is a bug in one of
    // them (reproduced by keras_app layer_inceptionresnetv2_conv2d_258 on SL2610).
    // Until it is fixed, fail as an infeasible plan so tile-and-fuse's probe tries a
    // smaller tile instead of aborting the compile.
    buffers = collectLramBuffers(funcOp);
    placement = planPlacement(buffers, usable, /*replan=*/false);
    if (!placement || !placement->schedule.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "LRAM plan infeasible after swaps\n");
        return failure();
    }
    return commitLramLayout(funcOp, reservedBase, buffers, placement->offsets);
}

namespace {

struct AddressAssignment : public llvm::ilist_node<AddressAssignment> {
    int64_t address;
    int64_t size;

    AddressAssignment(int64_t address, int64_t size) : address(address), size(size) {}
};

} // namespace

static std::pair<int64_t, llvm::ilist<AddressAssignment>::iterator> findEmptyRegion(
    llvm::ilist<AddressAssignment> &liveAllocations, int64_t startAddress, int64_t size
) {

    int64_t currentAddress = startAddress;

    for (auto it = liveAllocations.begin(); it != liveAllocations.end(); ++it) {
        if (it->address - currentAddress >= size) {
            return {currentAddress, it};
        }
        currentAddress = it->address + it->size;
    }

    return {currentAddress, liveAllocations.end()};
}

static FailureOr<int64_t>
assignAddressesToXramAllocs(FunctionOpInterface funcOp, int64_t startAddress) {

    llvm::ilist<AddressAssignment> liveAllocations;
    llvm::DenseMap<Value, AddressAssignment *> liveAllocationsMap;

    int64_t lastUsedAddress = startAddress;

    // Track which memory regions are in use via liveAllocations (a sorted list of
    // allocated address ranges). For each AllocOp, find the first gap large enough
    // to fit the allocation, insert it into the live list, and record the mapping
    // from value to allocation. When DeallocOps are encountered, remove their
    // corresponding allocations from the live list. This maintains a fragmented
    // memory map that can reuse space as allocations are deallocated.

    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {

            auto value = allocOp.getResult();

            if (getEncodingMemorySpace(value.getType()) != torq_hl::MemorySpace::Xram) {
                continue;
            }

            int64_t allocSize = align_ceil(getEncodedTotalSizeBytes(value.getType()), 4);

            auto [allocAddr, it] = findEmptyRegion(liveAllocations, startAddress, allocSize);

            if (allocAddr + allocSize > HwInfo::xram_size) {
                return allocOp.emitError("XRAM size exceeded while assigning addresses");
            }

            lastUsedAddress = std::max(lastUsedAddress, allocAddr + allocSize);

            LLVM_DEBUG({
                llvm::dbgs() << "Processing alloc: ";
                allocOp.dump();

                llvm::dbgs() << "Assigned address " << allocAddr << " to allocation of size "
                             << allocSize << "\n";
                llvm::dbgs() << "Live allocations:\n";
                for (auto &alloc : liveAllocations) {
                    llvm::dbgs() << "   address: " << alloc.address << " size: " << alloc.size
                                 << "\n";
                }
                llvm::dbgs() << "\n";

                llvm::dbgs() << "Last used address: " << lastUsedAddress << "\n";
            });

            // append the allocation at the right place
            auto allocation = liveAllocations.insert(it, {allocAddr, allocSize});
            liveAllocationsMap.try_emplace(value, &*allocation);

            setXramAddress(&op, allocAddr);
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {

            auto it = liveAllocationsMap.find(deallocOp.getMemref());
            if (it != liveAllocationsMap.end()) {

                LLVM_DEBUG({
                    llvm::dbgs() << "Processing dealloc: ";
                    deallocOp.dump();
                });

                liveAllocations.erase(it->second->getIterator());
                liveAllocationsMap.erase(it);
            }
        }
    }

    return lastUsedAddress;
}

static LogicalResult allocateXramAddresses(FunctionOpInterface funcOp) {

    int64_t startAddress = clXramStartAddress;

    startAddress = align_ceil(startAddress, HwInfo::xram_page_size);

    // initializeed data pass: pack CreateInvocationOp code sections and ConstOp at
    // 4-byte alignment so the whole region can later be mapped as a single SG
    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (auto createInvocationOp = dyn_cast<syna::torq_hl::CreateInvocationOp>(op)) {

            auto codeSection = createInvocationOp.getCodeSection();

            if (!codeSection) {
                continue;
            }

            auto type = mlir::cast<MemRefType>(codeSection.getType());

            createInvocationOp.setXramCodeAddress(startAddress);

            startAddress = align_ceil(startAddress + getEncodedTotalSizeBytes(type), 4);

            if (startAddress > HwInfo::xram_size) {
                return createInvocationOp.emitError("XRAM size exceeded while assigning addresses");
            }
        }
        else if (auto programCodeOp = dyn_cast<syna::torq_hl::ProgramCodeOp>(op)) {

            // The program code is shared across all invocations of the program, so
            // it is placed in XRAM exactly once here.
            auto type = mlir::cast<MemRefType>(programCodeOp.getCode().getType());

            setXramAddress(&op, startAddress);

            startAddress = align_ceil(startAddress + getEncodedTotalSizeBytes(type), 4);

            if (startAddress > HwInfo::xram_size) {
                return programCodeOp.emitError("XRAM size exceeded while assigning addresses");
            }
        }
        else if (isa<syna::torq_hl::ConstOp>(op)) {

            assert(op.getNumResults() == 1 && "Expected a single result for alloc-like operations");

            auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

            if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
                continue;
            }

            setXramAddress(&op, startAddress);

            startAddress = align_ceil(startAddress + getEncodedTotalSizeBytes(type), 4);

            if (startAddress > HwInfo::xram_size) {
                return op.emitError("XRAM size exceeded while assigning addresses");
            }
        }
    }

    // uninitialized data pass: allocate addresses for Xram memref.alloc operations
    // while aggressively reusing addresses as soon as a memref is de-allocated
    auto maybeNextStartAddress = assignAddressesToXramAllocs(funcOp, startAddress);

    if (failed(maybeNextStartAddress)) {
        return failure();
    }

    startAddress = *maybeNextStartAddress;

    // reseve a chunk of memory at the end of the model, but before the bindings
    // where we will be able to store NSS programs
    int64_t nssProgramBase = startAddress;

    Builder builder(funcOp.getContext());
    funcOp->setAttr("torq-nss-program-base", builder.getI64IntegerAttr(nssProgramBase));

    startAddress = startAddress + clMaxNssProgramsSize;

    // binding pass: place each MapBindingOp on its own page(s) after the body
    // so that each binding can be IOMMU-mapped independently without sharing a
    // page with the body or any other binding
    int64_t bindingBase = align_ceil(startAddress, HwInfo::xram_page_size);

    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (!isa<syna::torq_hl::MapBindingOp>(op)) {
            continue;
        }

        assert(op.getNumResults() == 1 && "Expected a single result for MapBindingOp");

        auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
            continue;
        }

        int64_t addr = align_ceil(bindingBase, HwInfo::xram_page_size);

        setXramAddress(&op, addr);

        // round the size up to a page so the next allocation starts on a fresh page
        bindingBase = addr + align_ceil(getEncodedTotalSizeBytes(type), HwInfo::xram_page_size);

        if (bindingBase > HwInfo::xram_size) {
            return op.emitError("XRAM size exceeded while assigning binding addresses");
        }
    }

    // derived memref pass: subview/reinterpret_cast/reshape etc. inherit their
    // address from their source, so run this after the body and binding passes
    // have assigned addresses to all possible sources
    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (!isDerivedMemRefOperation(&op)) {
            continue;
        }

        auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
            continue;
        }

        if (failed(setDerivedMemrefAddress(&op))) {
            return failure();
        }
    }

    reserveXramArea(funcOp, bindingBase);

    // The body pass above only visits top-level ops, which is where program_code
    // is expected to live. Guard against a nested program_code silently missing an
    // XRAM address (which would later drop its binary from the serialized output).
    if (funcOp
            .walk([&](syna::torq_hl::ProgramCodeOp programCodeOp) {
                if (!getXramAddress(programCodeOp.getOperation())) {
                    programCodeOp.emitError(
                        "program_code must be at the function-body top level to be "
                        "assigned an XRAM address"
                    );
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            })
            .wasInterrupted()) {
        return failure();
    }

    return success();
}

// compute the backward slice of operations that the given operation depends on
// we can remove this once we rebase on a new version of MLIR that includes
// https://github.com/llvm/llvm-project/commit/2dd3d3852d16cab2c3a032223fc751db750a78f2
static void getBackwardSlice(Operation *op, SetVector<Operation *> *slice) {
    for (auto operand : op->getOperands()) {
        if (isa<BlockArgument>(operand)) {
            continue;
        }
        auto definingOp = operand.getDefiningOp();
        if (definingOp && !slice->contains(definingOp)) {
            getBackwardSlice(definingOp, slice);
        }
    }
    slice->insert(op);
}

// True if `t`'s layout is contiguous row-major (a non-zero offset is allowed). Only
// such a subview can be reloaded from its source as a plain contiguous DMA; a
// subview that slices inner dimensions is strided (has gaps) and is left alone.
static bool isContiguousRowMajor(MemRefType t) {
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(t.getStridesAndOffset(strides, offset)))
        return false;
    int64_t expected = 1;
    ArrayRef<int64_t> shape = t.getShape();
    for (int d = (int)shape.size() - 1; d >= 0; --d) {
        if (strides[d] != expected)
            return false;
        expected *= shape[d];
    }
    return true;
}

// If `alloc` is written by exactly one torq_hl.load (as its output) and otherwise
// only read via memref.subview (an optional dealloc aside), return that load and
// collect the subviews and dealloc. Return nullptr otherwise. Such a buffer is a
// read-only value loaded whole and then sliced.
static torq_hl::LoadOp matchWholeBufferReadOnlyLoad(
    memref::AllocOp alloc, SmallVectorImpl<memref::SubViewOp> &subs, memref::DeallocOp &dealloc
) {
    torq_hl::LoadOp load = nullptr;
    for (Operation *user : alloc.getResult().getUsers()) {
        if (auto ld = dyn_cast<torq_hl::LoadOp>(user)) {
            if (load || ld.getOutput() != alloc.getResult())
                return nullptr;
            load = ld;
        }
        else if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
            subs.push_back(sv);
        }
        else if (auto dc = dyn_cast<memref::DeallocOp>(user)) {
            dealloc = dc;
        }
        else {
            return nullptr;
        }
    }
    return load;
}

// For a read-only buffer loaded whole and then sliced per-invocation with
// memref.subview (a weight), replace each subview use with a fresh load of only the
// matching sub-region from the original XRAM source, then drop the now-dead
// whole-buffer load. This keeps only the used slice resident instead of the whole
// weight, reducing peak LRAM residency.
static void narrowReadOnlyWeightLoads(FunctionOpInterface funcOp) {
    // A root larger than the usable LRAM can never be resident whole; the allocator
    // already streams its subviews, and splitting it into per-use loads only adds
    // concurrent loads. Only narrow roots that could be resident (so keeping them
    // whole is the waste we remove).
    int64_t budget =
        static_cast<int64_t>(TorqHw::get().getLramSize()) - HwInfo::nss_max_program_size * 2;

    SmallVector<memref::AllocOp> roots;
    funcOp->walk([&](memref::AllocOp alloc) { roots.push_back(alloc); });

    IRRewriter rewriter(funcOp->getContext());
    for (memref::AllocOp alloc : roots) {
        auto rootType = dyn_cast<MemRefType>(alloc.getType());
        if (!rootType || getEncodingMemorySpace(rootType) != torq_hl::MemorySpace::Lram)
            continue;
        int64_t rootSize = getEncodedTotalSizeBytes(rootType);
        if (rootSize > budget)
            continue;

        SmallVector<memref::SubViewOp> subs;
        memref::DeallocOp dealloc = nullptr;
        torq_hl::LoadOp load = matchWholeBufferReadOnlyLoad(alloc, subs, dealloc);
        if (!load || subs.empty())
            continue;

        Value xramSrc = load.getInput();
        auto xramType = dyn_cast<MemRefType>(xramSrc.getType());
        if (!xramType || xramType.getShape() != rootType.getShape())
            continue; // load is not a straight whole-buffer copy

        bool narrowedAll = true;
        for (memref::SubViewOp sv : subs) {
            auto svType = cast<MemRefType>(sv.getType());
            // contiguous LRAM buffer of the subview's logical shape
            auto newType = MemRefType::get(
                svType.getShape(), svType.getElementType(), MemRefLayoutAttrInterface(),
                rootType.getMemorySpace()
            );
            int64_t svSize = getEncodedTotalSizeBytes(newType);
            // Skip subviews that save nothing or are strided (see isContiguousRowMajor).
            if (svSize >= rootSize || !isContiguousRowMajor(svType)) {
                narrowedAll = false;
                continue;
            }

            // Load the matching sub-region of the XRAM source (same offsets/sizes/
            // strides, applied to the identically-shaped source) into a fresh buffer.
            rewriter.setInsertionPoint(sv);
            auto xsv = memref::SubViewOp::create(
                rewriter, sv.getLoc(), xramSrc, sv.getMixedOffsets(), sv.getMixedSizes(),
                sv.getMixedStrides()
            );
            auto newAlloc = memref::AllocOp::create(rewriter, sv.getLoc(), newType);
            torq_hl::LoadOp::create(
                rewriter, sv.getLoc(), newAlloc.getResult(), xsv.getResult(),
                SmallVector<int64_t>{}, SmallVector<int64_t>{}, svSize, /*unsafe=*/true
            );
            rewriter.replaceAllUsesWith(sv.getResult(), newAlloc.getResult());
            rewriter.eraseOp(sv);
        }

        // If every subview was narrowed, the whole-buffer load is now dead.
        if (narrowedAll) {
            if (dealloc)
                rewriter.eraseOp(dealloc);
            rewriter.eraseOp(load);
            rewriter.eraseOp(alloc);
        }
    }
}

// Revert load-CSE reuses (tagged "torq.lram_reuse" by EliminateRedundantLramLoads)
// that keep a read-only buffer resident and shared across consumers. When slicing
// is on, such a shared buffer stays pinned by an in-flight start_program while other
// programs need LRAM, so address assignment runs out of space. Give each consumer
// its own fresh load of the same source, so each copy is short-lived and swappable
// (this restores the pre-reuse per-use load for the tagged buffers only). Safe:
// the reuse was recorded only for buffers never written in-block (read-only).
static void revertSharedReadOnlyReuses(FunctionOpInterface funcOp) {
    SmallVector<memref::AllocOp> tagged;
    funcOp->walk([&](memref::AllocOp alloc) {
        if (alloc->hasAttr("torq.lram_reuse"))
            tagged.push_back(alloc);
    });

    IRRewriter rewriter(funcOp->getContext());
    for (memref::AllocOp alloc : tagged) {
        // The buffer must be written by exactly one torq_hl.load (its fill) and
        // otherwise only read; collect the reading consumer ops.
        torq_hl::LoadOp load = nullptr;
        memref::DeallocOp dealloc = nullptr;
        SmallVector<Operation *> consumers;
        bool ok = true;
        for (Operation *user : alloc.getResult().getUsers()) {
            if (auto ld = dyn_cast<torq_hl::LoadOp>(user)) {
                if (ld.getOutput() == alloc.getResult()) {
                    if (load) { // a second writer: not the read-only pattern
                        ok = false;
                        break;
                    }
                    load = ld;
                }
                else {
                    consumers.push_back(user);
                }
            }
            else if (auto dc = dyn_cast<memref::DeallocOp>(user)) {
                dealloc = dc;
            }
            else {
                consumers.push_back(user);
            }
        }
        if (!ok || !load || consumers.empty())
            continue;

        // The clones re-read the original load's XRAM source. AddDeallocation
        // already placed that source's dealloc right after the (single) original
        // load; moving the reads to the later consumers would leave them reading a
        // freed buffer. Find the source buffer's dealloc so we can sink it past the
        // reloaded reads below.
        Value srcBase = getViewBase(load.getInput());
        memref::DeallocOp srcDealloc = nullptr;
        for (Operation *u : srcBase.getUsers())
            if (auto dc = dyn_cast<memref::DeallocOp>(u))
                srcDealloc = dc;

        // Give each consumer its own fresh load from the same source, and free that
        // copy as soon as the consumer is done with it, so it is not resident across
        // the other programs (which is what made the shared buffer overflow LRAM).
        // The source (load's input) dominates the original load, hence every later
        // consumer.
        for (Operation *consumer : consumers) {
            rewriter.setInsertionPoint(consumer);
            auto newAlloc = memref::AllocOp::create(rewriter, load.getLoc(), alloc.getType());
            Operation *newLoad = rewriter.clone(*load.getOperation());
            cast<torq_hl::LoadOp>(newLoad).getOutputMutable().set(newAlloc.getResult());
            consumer->replaceUsesOfWith(alloc.getResult(), newAlloc.getResult());

            // Deallocate the copy after the consumer finishes with it. A
            // start_program keeps its operands live until the matching wait_program,
            // so free after that; any other consumer is done at its own point.
            Operation *deallocAfter = consumer;
            if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(consumer))
                for (Operation *u : startOp.getInvocation().getUsers())
                    if (isa<torq_hl::WaitProgramOp>(u))
                        deallocAfter = u;
            rewriter.setInsertionPointAfter(deallocAfter);
            memref::DeallocOp::create(rewriter, load.getLoc(), newAlloc.getResult());
        }

        // Keep the source buffer live until the last reloaded read. Sink its
        // dealloc past the latest consumer (its ancestor in the dealloc's block),
        // so the clones never read freed memory.
        if (srcDealloc) {
            Block *db = srcDealloc->getBlock();
            Operation *latest = nullptr;
            for (Operation *consumer : consumers)
                if (Operation *anc = db->findAncestorOpInBlock(*consumer))
                    if (!latest || latest->isBeforeInBlock(anc))
                        latest = anc;
            if (latest && srcDealloc->isBeforeInBlock(latest))
                srcDealloc->moveAfter(latest);
        }

        // Original shared load / alloc are now dead.
        if (dealloc)
            rewriter.eraseOp(dealloc);
        rewriter.eraseOp(load);
        rewriter.eraseOp(alloc);
    }
}

// True if `funcOp`'s LRAM addresses can be assigned without narrowing. The trial
// runs on a throwaway clone so the real IR is untouched, and its diagnostics are
// suppressed.
static bool lramAllocationSucceeds(FunctionOpInterface funcOp) {
    Operation *clone = funcOp->clone();
    bool ok;
    {
        const uint64_t ownerThreadId = llvm::get_threadid();
        mlir::ScopedDiagnosticHandler swallow(
            funcOp->getContext(),
            [ownerThreadId](mlir::Diagnostic &) -> LogicalResult {
                // Swallow only the diagnostics from this trial. The handler is on the
                // shared context. Swallowing everything would hide another thread's error.
                if (llvm::get_threadid() != ownerThreadId)
                    return llvm::failure();
                return llvm::success();
            }
        );
        // without re-planning, so recovery applies whenever the plain plan fails and
        // re-planning handles what the recovered IR still cannot place
        ok = succeeded(allocateLramAddresses(cast<FunctionOpInterface>(clone), false));
    }
    clone->erase();
    return ok;
}

namespace {

// Reduce peak LRAM residency in functions whose address allocation would otherwise
// fail. Added only to the real pipeline: the tile-fit probe cannot see the DMA this
// costs, so it would pick a tile that only fits because of it.
class RecoverLramResidencyPass : public impl::RecoverLramResidencyBase<RecoverLramResidencyPass> {
  public:
    RecoverLramResidencyPass() = default;
    RecoverLramResidencyPass(const RecoverLramResidencyPass &pass) {}

    void runOnOperation() override {
        auto funcOp = getOperation();

        if (!clFailureDrivenRecovery || lramAllocationSucceeds(funcOp))
            return;

        revertSharedReadOnlyReuses(funcOp);
        narrowReadOnlyWeightLoads(funcOp);
    }
};

class AssignLramAddressesPass : public impl::AssignLramAddressesBase<AssignLramAddressesPass> {
  public:
    AssignLramAddressesPass() = default;
    AssignLramAddressesPass(const AssignLramAddressesPass &pass) {}
    AssignLramAddressesPass(const AssignLramAddressesOptions &options) {
        this->replan = options.replan;
    }

    // TODO(sflur): except for the emitError below, that tile-and-fuse expects,
    // this pass should not report errors that tile-and-fuse can not suppress.
    // Those will appear to the user as real errors, when tile-and-fuse is
    // looking for the right tile size (so they are not real errors).
    void runOnOperation() override {
        auto funcOp = getOperation();

        LogicalResult result = llvm::failure();

        // When tile and fuse runs the pipeline it expects to see
        // OUT_OF_MEMORY_MESSAGE as the first error, if the tile does not fit in
        // memory. Any errors after that one will be suppressed. The diagHandler
        // below catches all the errors/warnings the pass emits, and re-emits
        // them after emitting OUT_OF_MEMORY_MESSAGE.
        llvm::SmallVector<mlir::InFlightDiagnostic> inFlightDiags;

        const uint64_t ownerThreadId = llvm::get_threadid();

        { // This scopes delimits diagHandler, so at the end we can actually
          // re-emit the errors without it catching them again.
            mlir::ScopedDiagnosticHandler diagHandler(
                funcOp->getContext(),
                [&, ownerThreadId](mlir::Diagnostic &diag) -> LogicalResult {
                    // Re-emit only the diagnostics from this function. The handler is shared.
                    if (llvm::get_threadid() != ownerThreadId)
                        return llvm::failure();

                    InFlightDiagnostic inFlightDiag =
                        getContext().getDiagEngine().emit(diag.getLocation(), diag.getSeverity());

                    inFlightDiag << diag.str();

                    for (Diagnostic &note : diag.getNotes())
                        inFlightDiag.attachNote(note.getLocation()) << note.str();

                    inFlightDiags.emplace_back(std::move(inFlightDiag));

                    return llvm::success();
                }
            );

            result = allocateLramAddresses(funcOp, replan);
        } // destruct diagHandler;

        if (failed(result)) {
            // TileAndFusePass relies on this message to identify memory overflow
            emitError(funcOp->getLoc()) << OUT_OF_MEMORY_MESSAGE;
            signalPassFailure();
        }

        // re-emit all the errors/warnings (after OUT_OF_MEMORY_MESSAGE).
        if (!inFlightDiags.empty()) {
            for (auto &inFlightDiag : inFlightDiags) {
                inFlightDiag.report();
            }
        }
    }
};

class AssignDtcmItcmXramAddressesPass
    : public impl::AssignDtcmItcmXramAddressesBase<AssignDtcmItcmXramAddressesPass> {
  public:
    AssignDtcmItcmXramAddressesPass() = default;
    AssignDtcmItcmXramAddressesPass(const AssignDtcmItcmXramAddressesPass &pass) {}

    void runOnOperation() override {
        auto funcOp = getOperation();

        Pool dtcmPool(HwInfo::dtcm_size - HwInfo::css_reserved_dtcm_size, 0, 4);

        if (failed(allocateAddresses(funcOp, dtcmPool, syna::torq_hl::MemorySpace::Dtcm))) {
            emitError(funcOp->getLoc()) << "Failed to allocate DTCM addresses";
            return signalPassFailure();
        }

        // FIXME: here we all allocations to be at the begining of the pool
        // so that we end up using only address 0 for the ITCM which is the
        // only addrees where the compiled dispatches can currently run
        Pool itcmPool(HwInfo::itcm_size, 0, 4, 8 * 1024);

        if (failed(allocateAddresses(funcOp, itcmPool, syna::torq_hl::MemorySpace::Itcm))) {
            emitError(funcOp->getLoc()) << "Failed to allocate ITCM addresses";
            return signalPassFailure();
        };
        if (failed(allocateXramAddresses(funcOp))) {
            llvm::errs() << "Failed to allocate XRAM addresses\n";
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createRecoverLramResidencyPass() {
    return std::make_unique<RecoverLramResidencyPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignLramAddressesPass(bool replan) {
    return std::make_unique<AssignLramAddressesPass>(AssignLramAddressesOptions{replan});
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignDtcmItcmXramAddressesPass() {
    return std::make_unique<AssignDtcmItcmXramAddressesPass>();
}

} // namespace mlir::syna::torq
