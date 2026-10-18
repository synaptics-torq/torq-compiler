// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/LramAddressSpace.h"
#include "torq/Codegen/LramAllocation.h"
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

    // A reinterpret view shares its source's address, so it must not add an
    // offset of its own (that would have to be folded in here, which is
    // unsupported). The offset is a single entry (not one per dimension);
    // sizes and strides must be static so the view's geometry cannot depend
    // on runtime values the resolver cannot see. This applies to both
    // memref.reinterpret_cast and the element-reinterpreting
    // torq_hl.reinterpret_cast (the bufferized form of torq_hl.bitcast).
    if (isa<memref::ReinterpretCastOp, syna::torq_hl::ReinterpretCastOp>(op)) {
        auto reinterpretCast = cast<OffsetSizeAndStrideOpInterface>(op);
        if (reinterpretCast.isDynamicOffset(0)) {
            return op->emitOpError("dynamic offsets not supported");
        }
        if (llvm::any_of(
                reinterpretCast.getStaticSizes(),
                [](int64_t size) { return ShapedType::isDynamic(size); }
            ) ||
            llvm::any_of(reinterpretCast.getStaticStrides(), [](int64_t stride) {
                return ShapedType::isDynamic(stride);
            })) {
            return op->emitOpError("dynamic sizes or strides not supported");
        }
        if (reinterpretCast.getStaticOffset(0) != 0) {
            return op->emitOpError("non-zero static offsets not supported");
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

// The buffers resident in an address space: which ones, at what offset, how many bytes.
class Occupancy {
    ArrayRef<LramBuffer> buffers_;
    const int64_t usable_;
    LramAddressSpace space_;
    llvm::BitVector resident_;
    SmallVector<int64_t> offsets_;
    SmallVector<int64_t> birthOffsets_;
    int64_t bytes_ = 0;

  public:
    Occupancy(ArrayRef<LramBuffer> buffers, int64_t usable)
        : buffers_(buffers), usable_(usable), space_(usable), resident_(buffers.size()),
          offsets_(buffers.size()), birthOffsets_(buffers.size(), -1) {}

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
        if (birthOffsets_[idx] < 0) {
            birthOffsets_[idx] = offset;
        }
        bytes_ += buffers_[idx].size;
        return true;
    }

    void release(BufferIdx idx) {
        space_.release(offsets_[idx], buffers_[idx].size);
        resident_.reset(idx);
        bytes_ -= buffers_[idx].size;
    }

    SmallVector<int64_t> takeBirthOffsets() { return std::move(birthOffsets_); }
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

// A placement of every buffer: its offset from the LRAM base at its birth, and the swaps
// that make the layout fit, each with the offsets its reloads land at.
struct Placement {
    SmallVector<int64_t> birthOffsets;
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
    // The function body the buffers were collected from, so a failure can name
    // the op at a given index and its source line.
    Block *body_;
    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
    Occupancy occupancy_;
    SwapSchedule schedule_;
    PlacementFailure failure_;

    PlacementSimulation(ArrayRef<LramBuffer> buffers, int64_t usable, Block *body)
        : buffers_(buffers), body_(body), occupancy_(buffers, usable) {
        for (auto [i, buffer] : llvm::enumerate(buffers_)) {
            events_.push({buffer.birth, EventKind::Birth, (BufferIdx)i});
            events_.push({buffer.death, EventKind::Death, (BufferIdx)i});
        }
    }

    // Release the resident `idx` and schedule its swap-out at `opIdx`.
    void swapOut(BufferIdx idx, OpIdx opIdx);
    // Schedule the swap-in of `idx` at `opIdx`, at the offset it is placed at.
    void scheduleSwapIn(BufferIdx idx, OpIdx opIdx);
    // Queue the reload of the released `idx` at its next use after `opIdx`.
    void scheduleReload(BufferIdx idx, OpIdx opIdx);
    // Swap `victim` out at `opIdx` and reload it at its next use.
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
        PlacementFailure &failure, Block *body) {
        PlacementSimulation simulation(buffers, usable, body);
        if (!simulation.sweep(forcedCompactions)) {
            failure = simulation.failure_;
            return std::nullopt;
        }
        return Placement{simulation.occupancy_.takeBirthOffsets(), std::move(simulation.schedule_)};
    }
};

} // namespace

void PlacementSimulation::swapOut(BufferIdx idx, OpIdx opIdx) {
    occupancy_.release(idx);
    schedule_[opIdx].push_back({Swap::Kind::Out, buffers_[idx].value});
}

void PlacementSimulation::scheduleSwapIn(BufferIdx idx, OpIdx opIdx) {
    schedule_[opIdx].push_back({Swap::Kind::In, buffers_[idx].value, occupancy_.offset(idx)});
}

void PlacementSimulation::scheduleReload(BufferIdx idx, OpIdx opIdx) {
    OpIdx reloadOpIdx = buffers_[idx].resident.nextAfter(opIdx);
    if (reloadOpIdx != INT_MAX) {
        events_.push({reloadOpIdx, EventKind::Reload, idx});
    }
}

void PlacementSimulation::spill(BufferIdx victim, OpIdx opIdx) {
    swapOut(victim, opIdx);
    scheduleReload(victim, opIdx);
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
    // every moved buffer is out before any comes back, so they land in the freed space
    for (BufferIdx moved : movable) {
        swapOut(moved, opIdx);
    }
    for (BufferIdx moved : movable) {
        if (occupancy_.place(moved)) {
            scheduleSwapIn(moved, opIdx);
        }
        else {
            scheduleReload(moved, opIdx);
        }
    }
    return true;
}

void PlacementSimulation::recordFailure(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause) {
    failure_.cause = cause;
    failure_.opIdx = opIdx;
    auto residents = occupancy_.residents();
    failure_.pinned.assign(residents.begin(), residents.end());
    LLVM_DEBUG({
        // Name the op at `opIdx` and its source line so the failure can be
        // located in the IR, not just by its index in the function body.
        std::string opDesc;
        if (body_ && opIdx >= 0 && opIdx < (int)body_->getOperations().size()) {
            Operation &op = *std::next(body_->getOperations().begin(), opIdx);
            opDesc = " (op " + op.getName().getStringRef().str();
            if (auto loc = dyn_cast<FileLineColLoc>(op.getLoc()))
                opDesc += " at " + loc.getFilename().getValue().str() + ":" +
                          std::to_string(loc.getLine());
            opDesc += ")";
        }
        llvm::dbgs() << "LRAM plan infeasible ("
                     << (cause == PlacementFailure::Cause::Fragmentation ? "fragmentation"
                                                                         : "byte pressure")
                     << ") at op " << opIdx << opDesc << ": cannot place buffer " << idx << " ("
                     << buffers_[idx].size << " bytes, " << buffers_[idx].value << "), resident "
                     << occupancy_.bytes() << " of " << occupancy_.usable() << " bytes\n";
        for (BufferIdx pinned : residents) {
            llvm::dbgs() << "  pinned buffer " << pinned << " size " << buffers_[pinned].size
                         << " offset " << occupancy_.offset(pinned) << " ("
                         << buffers_[pinned].value << ")\n";
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
        else {
            if (!place(event.buffer, event.opIdx)) {
                return false;
            }
            // the swap-outs place() scheduled at this op come first, so the walk frees the
            // space before it reloads into it
            if (event.kind == EventKind::Reload) {
                scheduleSwapIn(event.buffer, event.opIdx);
            }
        }
    }
    return true;
}

// Assign each allocation its address; the views derived from it inherit theirs. Forward
// order suffices: a view always follows the value it derives from.
static LogicalResult commitLramLayout(
    FunctionOpInterface funcOp, int64_t reservedBase, ArrayRef<LramBuffer> buffers,
    ArrayRef<int64_t> birthOffsets
) {
    for (auto [buffer, offset] : llvm::zip_equal(buffers, birthOffsets)) {
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
planPlacement(ArrayRef<LramBuffer> buffers, int64_t usable, bool replan, Block *body) {
    // each attempt adds one forced compaction; a stack of repeated layers needs a few per layer
    constexpr int maxReplans = 256;
    SmallVector<ForcedCompaction> forcedCompactions;
    for (int attempt = 0; attempt <= maxReplans; attempt++) {
        PlacementFailure failure;
        auto placement =
            PlacementSimulation::run(buffers, usable, forcedCompactions, failure, body);
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

// Bytes of the buffers that `swaps`, scheduled at one op, swap out and back in there.
static int64_t bytesSwappedOutAndIn(ArrayRef<Swap> swaps) {
    llvm::DenseSet<Value> swappedOut;
    int64_t bytes = 0;
    for (const Swap &swap : swaps) {
        if (swap.kind == Swap::Kind::Out)
            swappedOut.insert(swap.buffer);
        else if (swappedOut.contains(swap.buffer))
            bytes += getEncodedTotalSizeBytes(cast<MemRefType>(swap.buffer.getType()));
    }
    return bytes;
}

// Bytes that `schedule` stores to XRAM only to load them straight back, at the same
// op. A spill reloads its buffer at a later op.
static int64_t relocatedBytes(const SwapSchedule &schedule) {
    int64_t bytes = 0;
    for (const auto &opSwaps : schedule)
        bytes += bytesSwappedOutAndIn(opSwaps.second);
    return bytes;
}

// The tile-fit probe mark of the module of `funcOp`: null outside a probe, true in a
// strict probe.
static BoolAttr getTileFitProbeAttr(FunctionOpInterface funcOp) {
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    return moduleOp ? moduleOp->getAttrOfType<BoolAttr>(kTileFitProbeAttr) : BoolAttr();
}

bool lramPlacementFitsWithoutSwaps(FunctionOpInterface funcOp, int64_t usableBytes) {
    SmallVector<LramBuffer> buffers = collectLramBuffers(funcOp);
    std::optional<Placement> placement =
        planPlacement(buffers, usableBytes, /*replan=*/true, &funcOp.getFunctionBody().front());
    return placement && placement->schedule.empty();
}

// Lay out every LRAM allocation over its whole lifetime; when that does not fit, schedule
// swaps and perform them with convertVirtualToPhysicalMemRefs, which places the buffers it
// swaps in where the plan put them. Infeasibility (the buffers pinned at some op cannot
// coexist) fails without a diagnostic, so runOnOperation emits the OUT_OF_MEMORY_MESSAGE
// tile-and-fuse's probe shrinks the tile on; any other diagnostic aborts the compile.
// Without `allowRelocation`, a plan that relocates buffers fails the same way.
static LogicalResult allocateLramAddresses(
    FunctionOpInterface funcOp, bool replan, bool allowRelocation, int64_t lramSize = 0
) {

    // we reserve space for two NSS programs at the begininng of LRAM
    // (one block for the currently executing program and one for the
    // next program to be executed that will be loaded by the current one)
    int64_t reservedBase = HwInfo::nss_max_program_size * 2;
    int64_t poolSize = lramSize > 0 ? lramSize : static_cast<int64_t>(TorqHw::get().getLramSize());
    const int64_t usable = poolSize - reservedBase;

    SmallVector<LramBuffer> buffers = collectLramBuffers(funcOp);
    std::optional<Placement> placement =
        planPlacement(buffers, usable, replan, &funcOp.getFunctionBody().front());
    if (!placement) {
        return failure();
    }
    if (placement->schedule.empty()) {
        return commitLramLayout(funcOp, reservedBase, buffers, placement->birthOffsets);
    }
    if (int64_t bytes = relocatedBytes(placement->schedule)) {
        if (!allowRelocation) {
            LLVM_DEBUG(llvm::dbgs() << "LRAM plan relocates buffers, rejected\n");
            return failure();
        }
        // tell the tile-fit probe; never mark the IR of a real compile
        if (getTileFitProbeAttr(funcOp)) {
            Builder builder(funcOp->getContext());
            funcOp->setAttr(kLramPlanRelocatesAttr, builder.getI64IntegerAttr(bytes));
        }
    }

    if (failed(convertVirtualToPhysicalMemRefs(
            funcOp, usable, torq_hl::MemorySpace::Lram, placement->schedule, reservedBase
        ))) {
        return failure();
    }
    // the swapped-in buffers have their addresses; the allocations the plan started from
    // get theirs here, with every view
    return commitLramLayout(funcOp, reservedBase, buffers, placement->birthOffsets);
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

// The top-level op of `funcOp` at which each XRAM allocation is first used, through
// derived views (deallocations do not count). Allocations are often hoisted far
// ahead of their first use; starting the live range at the alloc op would keep the
// buffer reserved across everything in between and block address reuse.
static llvm::DenseMap<Operation *, SmallVector<memref::AllocOp>>
findXramAllocStarts(FunctionOpInterface funcOp) {
    Block &body = funcOp.getFunctionBody().front();
    llvm::DenseMap<Operation *, int> order;
    int index = 0;
    for (Operation &op : body)
        order[&op] = index++;

    llvm::DenseMap<Operation *, SmallVector<memref::AllocOp>> starts;
    for (auto allocOp : body.getOps<memref::AllocOp>()) {
        if (getEncodingMemorySpace(allocOp.getType()) != torq_hl::MemorySpace::Xram)
            continue;

        Operation *first = nullptr;
        SmallVector<Value> worklist{allocOp.getResult()};
        llvm::DenseSet<Value> seen;
        while (!worklist.empty()) {
            Value value = worklist.pop_back_val();
            for (Operation *user : value.getUsers()) {
                if (isa<memref::DeallocOp>(user))
                    continue;
                if (isDerivedMemRefOperation(user))
                    for (Value result : user->getResults())
                        if (seen.insert(result).second)
                            worklist.push_back(result);
                Operation *top = body.findAncestorOpInBlock(*user);
                if (top && (!first || order[top] < order[first]))
                    first = top;
            }
        }
        // never start before the allocation itself (or at all, if it is unused)
        if (!first || order[first] < order[allocOp])
            first = allocOp;
        starts[first].push_back(allocOp);
    }
    return starts;
}

static FailureOr<int64_t>
assignAddressesToXramAllocs(FunctionOpInterface funcOp, int64_t startAddress) {

    llvm::ilist<AddressAssignment> liveAllocations;
    llvm::DenseMap<Value, AddressAssignment *> liveAllocationsMap;

    int64_t lastUsedAddress = startAddress;

    // Track which memory regions are in use via liveAllocations (a sorted list of
    // allocated address ranges). Each XRAM allocation becomes live at its first use
    // (findXramAllocStarts): find the first gap large enough to fit it, insert it
    // into the live list, and record the mapping from value to allocation. When
    // DeallocOps are encountered, remove their corresponding allocations from the
    // live list. This maintains a fragmented memory map that can reuse space as
    // allocations are deallocated.

    auto starts = findXramAllocStarts(funcOp);

    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (auto it = starts.find(&op); it != starts.end()) {
            for (memref::AllocOp allocOp : it->second) {

                auto value = allocOp.getResult();

                int64_t allocSize = align_ceil(getEncodedTotalSizeBytes(value.getType()), 4);

                auto [allocAddr, pos] = findEmptyRegion(liveAllocations, startAddress, allocSize);

                if (allocAddr + allocSize > HwInfo::xram_size) {
                    return allocOp.emitError("XRAM size exceeded while assigning addresses");
                }

                lastUsedAddress = std::max(lastUsedAddress, allocAddr + allocSize);

                LLVM_DEBUG({
                    llvm::dbgs() << "Processing alloc: ";
                    allocOp.dump();
                    llvm::dbgs() << "Assigned address " << allocAddr << " to allocation of size "
                                 << allocSize << "\n";
                    llvm::dbgs() << "Last used address: " << lastUsedAddress << "\n";
                });

                // append the allocation at the right place
                auto allocation = liveAllocations.insert(pos, {allocAddr, allocSize});
                liveAllocationsMap.try_emplace(value, &*allocation);

                setXramAddress(allocOp, allocAddr);
            }
        }

        if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {

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
// programs need LRAM, so address assignment runs out of space. Give each group of
// consumers that run at the same time its own fresh load of the same source, so
// each copy is short-lived and swappable (this restores the pre-reuse per-use load
// for the tagged buffers only). Safe: the reuse was recorded only for buffers never
// written in-block (read-only).
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

        // A start_program keeps its operands live until the matching wait_program;
        // any other consumer is done at its own point.
        auto getDoneOp = [](Operation *consumer) {
            if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(consumer))
                for (Operation *u : startOp.getInvocation().getUsers())
                    if (isa<torq_hl::WaitProgramOp>(u) && u->getBlock() == consumer->getBlock())
                        return u;
            return consumer;
        };

        // Consumers that run at the same time keep sharing one copy: the slices of an
        // scf.forall, which FoldConvert made read one buffer (see
        // hoistSliceInvariantLoads), all start before the first of them is waited for.
        // A private copy per slice needs more LRAM at that point, not less. So a
        // consumer joins the current group when it starts before the group is done.
        struct ConsumerGroup {
            SmallVector<Operation *> members;
            Operation *done;
        };
        SmallVector<ConsumerGroup> groups;
        Block *block = consumers.front()->getBlock();
        bool oneBlock =
            llvm::all_of(consumers, [&](Operation *c) { return c->getBlock() == block; });
        if (oneBlock) {
            llvm::sort(consumers, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
            consumers.erase(llvm::unique(consumers), consumers.end());
        }
        for (Operation *consumer : consumers) {
            Operation *done = getDoneOp(consumer);
            Operation *groupDone = groups.empty() ? nullptr : groups.back().done;
            if (!oneBlock || !groupDone || !consumer->isBeforeInBlock(groupDone)) {
                groups.push_back({{consumer}, done});
                continue;
            }
            groups.back().members.push_back(consumer);
            if (groupDone->isBeforeInBlock(done))
                groups.back().done = done;
        }

        // Give each group its own fresh load from the same source, and free that copy
        // as soon as the group is done with it, so it is not resident across the other
        // programs (which is what made the shared buffer overflow LRAM). The source
        // (load's input) dominates the original load, hence every later consumer.
        for (ConsumerGroup &group : groups) {
            rewriter.setInsertionPoint(group.members.front());
            auto newAlloc = memref::AllocOp::create(rewriter, load.getLoc(), alloc.getType());
            Operation *newLoad = rewriter.clone(*load.getOperation());
            cast<torq_hl::LoadOp>(newLoad).getOutputMutable().set(newAlloc.getResult());
            for (Operation *consumer : group.members)
                consumer->replaceUsesOfWith(alloc.getResult(), newAlloc.getResult());

            rewriter.setInsertionPointAfter(group.done);
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

// True if a plain (no re-planning) LRAM placement exists for `funcOp`. This is the
// only way allocateLramAddresses can fail with allowRelocation=true: the swap
// conversion applies the plan and always succeeds, and the commit only records
// offsets on the allocations the plan came from. The check is therefore the pure
// plan over the collected buffers -- it never mutates IR, so it runs on `funcOp`
// itself. The trial used to clone the whole function and run the pass on the copy;
// the clone carries every nested program region, which took hours on whole-net
// single-dispatch functions for a bit the plan already computes.
static bool lramAllocationSucceeds(FunctionOpInterface funcOp) {
    const int64_t usable = TorqHw::get().getLramSize() - HwInfo::nss_max_program_size * 2;
    std::optional<Placement> placement = planPlacement(
        collectLramBuffers(funcOp), usable, /*replan=*/false, &funcOp.getFunctionBody().front()
    );
    return placement.has_value();
}

std::optional<int64_t> lramSwapBytes(FunctionOpInterface funcOp) {
    const int64_t usable = TorqHw::get().getLramSize() - HwInfo::nss_max_program_size * 2;
    std::optional<Placement> placement = planPlacement(
        collectLramBuffers(funcOp), usable, /*replan=*/false, &funcOp.getFunctionBody().front()
    );
    if (!placement)
        return std::nullopt;
    int64_t bytes = 0;
    for (const auto &[opIdx, swaps] : placement->schedule) {
        // one entry is one DMA, so a relocation counts its buffer out and back in
        for (const Swap &swap : swaps) {
            bytes += getEncodedTotalSizeBytes(cast<MemRefType>(swap.buffer.getType()));
        }
    }
    return bytes;
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

        if (lramAllocationSucceeds(funcOp))
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
        this->lramSize = options.lramSize;
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

            // only a strict tile-fit probe rejects relocations; it never re-plans
            BoolAttr probe = getTileFitProbeAttr(funcOp);
            bool allowRelocation = replan || !probe || !probe.getValue();
            result = allocateLramAddresses(funcOp, replan, allowRelocation, lramSize);
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

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAssignLramAddressesPass(bool replan, int64_t lramSize) {
    return std::make_unique<AssignLramAddressesPass>(AssignLramAddressesOptions{replan, lramSize});
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignDtcmItcmXramAddressesPass() {
    return std::make_unique<AssignDtcmItcmXramAddressesPass>();
}

} // namespace mlir::syna::torq
