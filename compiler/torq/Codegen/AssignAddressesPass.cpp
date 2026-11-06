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
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/HwInfo.h"
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
#include "llvm/Support/MathExtras.h"
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

    // ascending, disjoint [start, end) ranges
    ArrayRef<std::pair<OpIdx, OpIdx>> ranges() const { return ranges_; }
};

// One LRAM buffer to place.
struct LramBuffer {
    // its alloc receives the address
    TypedValue<MemRefType> value;
    int64_t bytes = 0;
    // lifetime [birth, death): the alloc's op and its dealloc's, or the op count
    // if it is never freed
    OpIdx birth = 0;
    OpIdx death = 0;
    // every op that has the buffer as an operand, and every [start_program, wait_program+1)
    // of an async program that accesses it
    ResidentRanges resident;
    // ascending by op; an access through a view is an access of the buffer
    struct Access {
        OpIdx opIdx;
        torq_hl::ArgAccessBitfield access;
    };
    SmallVector<Access> accesses;
    // the load from outside LRAM that fills the buffer, see isFill
    struct Fill {
        OpIdx opIdx;
        Value source;
    };
    std::optional<Fill> fill;

    // addresses are 8-byte aligned
    int64_t alignedSize() const { return align_ceil((uint32_t)bytes, 8u); }

    // Live contents: something reads them before the next whole overwrite.
    bool contentsLiveAt(OpIdx opIdx) const {
        const auto *next = llvm::partition_point(accesses, [&](const Access &access) {
            return access.opIdx < opIdx;
        });
        return next != accesses.end() &&
               bitEnumContainsAny(next->access, torq_hl::ArgAccessBitfield::Read);
    }

    // over [from, to)
    bool writtenBetween(OpIdx from, OpIdx to) const {
        return accessedBetween(from, to, torq_hl::ArgAccessBitfield::Write);
    }
    bool readBetween(OpIdx from, OpIdx to) const {
        return accessedBetween(from, to, torq_hl::ArgAccessBitfield::Read);
    }

  private:
    bool accessedBetween(OpIdx from, OpIdx to, torq_hl::ArgAccessBitfield how) const {
        return llvm::any_of(accesses, [&](const Access &access) {
            return from <= access.opIdx && access.opIdx < to &&
                   bitEnumContainsAny(access.access, how);
        });
    }
};

} // namespace

// A load fills its buffer when it is the buffer's only write and nothing writes the source
// after it: the source holds the contents from then on.
static bool isFill(
    const LramBuffer &buffer, const LramBuffer::Fill &fill,
    const llvm::DenseMap<Value, OpIdx> &lastSourceWrite
) {
    bool onlyWrite = llvm::none_of(buffer.accesses, [&](const LramBuffer::Access &access) {
        return access.opIdx != fill.opIdx &&
               bitEnumContainsAny(access.access, torq_hl::ArgAccessBitfield::Write);
    });
    auto sourceWrite = lastSourceWrite.find(getViewBase(fill.source));
    bool sourceKept = sourceWrite == lastSourceWrite.end() || sourceWrite->second < fill.opIdx;
    return onlyWrite && sourceKept;
}

// Collect one LramBuffer per LRAM allocation in `funcOp`'s body.
static SmallVector<LramBuffer> collectLramBuffers(FunctionOpInterface funcOp) {

    Block &body = funcOp.getFunctionBody().front();
    const int opCount = body.getOperations().size();

    llvm::MapVector<Value, LramBuffer> allocToBuffer;
    // each start_program's op and accessed buffers, until its wait_program
    llvm::DenseMap<Value, std::pair<OpIdx, SmallVector<Value>>> openPrograms;
    // keyed by root: a write to one root must leave every other unchanged, so distinct
    // roots are assumed not to alias. Not recursive: a write nested in a region is not
    // detected.
    llvm::DenseMap<Value, OpIdx> lastSourceWrite;
    // the last load into each buffer from outside LRAM
    llvm::MapVector<Value, LramBuffer::Fill> fillCandidates;
    for (auto [opIdx, op] : llvm::enumerate(body.getOperations())) {
        if (auto alloc = dyn_cast<memref::AllocOp>(&op)) {
            if (getEncodingMemorySpace(alloc.getType()) != torq_hl::MemorySpace::Lram) {
                continue;
            }
            LramBuffer buffer{
                .value = alloc.getResult(),
                .bytes = getEncodedTotalSizeBytes(alloc.getType()),
                .birth = (OpIdx)opIdx,
                .death = opCount,
                .resident = {},
                .accesses = {},
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
            for (const auto &[access, root] : getRootAccesses(&op, std::nullopt)) {
                if (getEncodingMemorySpace(root.getType()) != torq_hl::MemorySpace::Lram) {
                    if (bitEnumContainsAny(access, torq_hl::ArgAccessBitfield::Write)) {
                        lastSourceWrite[root] = (OpIdx)opIdx;
                    }
                    continue;
                }
                // Every LRAM memref is a body memref.alloc or a view of one: bindings
                // and constants live in XRAM, the LRAM program slots are globals created
                // after this pass, and views keep their source's memory space (a cross-space
                // memory_space_cast is rejected when addresses are set).
                auto it = allocToBuffer.find(root);
                assert(it != allocToBuffer.end() && "LRAM root is not a body allocation");
                it->second.accesses.push_back({(OpIdx)opIdx, access});
            }
            if (auto load = dyn_cast<torq_hl::LoadOp>(&op)) {
                auto it = allocToBuffer.find(load.getOutput());
                Value source = getViewBase(load.getInput());
                // a program's code section is loaded once: torq-size-slice-code-sections
                // later shrinks that one load to the compiled code's size
                if (it != allocToBuffer.end() &&
                    getEncodingMemorySpace(cast<MemRefType>(source.getType())) !=
                        torq_hl::MemorySpace::Lram &&
                    !source.getDefiningOp<torq_hl::CreateInvocationOp>()) {
                    fillCandidates[it->first] = {(OpIdx)opIdx, load.getInput()};
                }
            }
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

    for (const auto &[value, load] : fillCandidates) {
        LramBuffer &buffer = allocToBuffer[value];
        if (isFill(buffer, load, lastSourceWrite)) {
            buffer.fill = load;
        }
    }

    return llvm::to_vector(allocToBuffer.values());
}

// Runs after convertVirtualToPhysicalMemRefs cloned the reloads: moving an op before it
// would shift the op indices the schedule refers to.
static void sinkSourceDeallocs(ArrayRef<LramBuffer> buffers) {
    for (const LramBuffer &buffer : buffers) {
        if (!buffer.fill) {
            continue;
        }
        Value source = buffer.fill->source;
        Operation *sourceDealloc = nullptr;
        for (Operation *user : getViewBase(source).getUsers()) {
            if (isa<memref::DeallocOp>(user)) {
                sourceDealloc = user;
            }
        }
        if (!sourceDealloc) {
            continue;
        }
        // a load nested in a region counts at its ancestor in the dealloc's block
        Block *block = sourceDealloc->getBlock();
        Operation *lastLoad = nullptr;
        for (Operation *user : source.getUsers()) {
            Operation *ancestor = block->findAncestorOpInBlock(*user);
            if (isa<torq_hl::LoadOp>(user) && ancestor &&
                (!lastLoad || lastLoad->isBeforeInBlock(ancestor))) {
                lastLoad = ancestor;
            }
        }
        if (lastLoad && sourceDealloc->isBeforeInBlock(lastLoad)) {
            sourceDealloc->moveAfter(lastLoad);
        }
    }
}

// A compaction forced by re-planning: the op it applies at, and the failing op whose pinned
// buffers it places first.
struct ForcedCompaction {
    OpIdx opIdx;
    OpIdx failingOpIdx;
    // the failing buffer once every position is tried: the compaction places it itself
    BufferIdx placeFirst = -1;
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
        assert(!resident_.test(idx) && "placing a resident buffer");
        int64_t offset = space_.allocate(buffers_[idx].alignedSize());
        if (offset < 0) {
            return false;
        }
        resident_.set(idx);
        offsets_[idx] = offset;
        if (birthOffsets_[idx] < 0) {
            birthOffsets_[idx] = offset;
        }
        bytes_ += buffers_[idx].alignedSize();
        return true;
    }

    void release(BufferIdx idx) {
        space_.release(offsets_[idx], buffers_[idx].alignedSize());
        resident_.reset(idx);
        bytes_ -= buffers_[idx].alignedSize();
    }

    SmallVector<int64_t> takeBirthOffsets() { return std::move(birthOffsets_); }
};

// Why a placement failed.
struct PlacementFailure {
    enum class Cause { BytePressure, Fragmentation };
    Cause cause = Cause::BytePressure;
    OpIdx opIdx = -1;
    BufferIdx buffer = -1;
    // the buffers pinned there, the only ones left resident
    SmallVector<BufferIdx> pinned;
    // the op at `opIdx`, for the diagnostic (null if out of range)
    Operation *op = nullptr;
    // the size of the buffer that could not be placed, the bytes resident (all pinned) then,
    // and the usable LRAM
    int64_t bytes = 0;
    int64_t residentBytes = 0;
    int64_t usableBytes = 0;

    StringRef causeName() const {
        return cause == Cause::Fragmentation ? "fragmentation" : "byte pressure";
    }
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
    // At an equal op, use ends and deaths are processed first, then reloads, then births:
    // space freed there is available to the buffers placed there.
    enum class EventKind { UseEnd, Death, Reload, Birth };
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
    // The ops of the function body the buffers were collected from, by index, so a
    // failure can name the op at a given index.
    ArrayRef<Operation *> ops_;
    std::priority_queue<Event, std::vector<Event>, EventLater> events_;
    Occupancy occupancy_;
    SwapSchedule schedule_;
    PlacementFailure failure_;
    // per buffer, the op after its latest transfer to or from a copy outside LRAM: a store,
    // its fill, or a swap-in repeating the fill
    SmallVector<std::optional<OpIdx>> copyHoldsFrom_;
    // filled buffers deferred at their birth, waiting for free space before their first use
    SmallVector<BufferIdx> deferredFills_;

    // With `spillFilledAfterUse`, a filled buffer is resident only while in use: it is
    // spilled at the end of each use after its fill and reloaded at the next.
    PlacementSimulation(
        ArrayRef<LramBuffer> buffers, int64_t usable, ArrayRef<Operation *> ops,
        bool spillFilledAfterUse
    )
        : buffers_(buffers), ops_(ops), occupancy_(buffers, usable),
          copyHoldsFrom_(buffers.size()) {
        for (auto [i, buffer] : llvm::enumerate(buffers_)) {
            events_.push({buffer.birth, EventKind::Birth, (BufferIdx)i});
            events_.push({buffer.death, EventKind::Death, (BufferIdx)i});
            if (!buffer.fill) {
                continue;
            }
            OpIdx filledAt = buffer.fill->opIdx + 1;
            copyHoldsFrom_[i] = filledAt;
            if (spillFilledAfterUse) {
                for (auto [start, end] : buffer.resident.ranges()) {
                    // use ranges past the fill's own, which ends at filledAt, and before
                    // the death, which releases the buffer itself
                    if (end > filledAt && end < buffer.death) {
                        events_.push({end, EventKind::UseEnd, (BufferIdx)i});
                    }
                }
            }
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
    // Leave the filled `idx` out from its birth, its fill cancelled, and reload it at its first
    // use unless placeDeferredFills places it before.
    void deferFill(BufferIdx idx);
    // Place the deferred fills that free space holds at `opIdx`, before its events.
    void placeDeferredFills(OpIdx opIdx);
    // Place `placeFirst` (-1 for none), then move every buffer not pinned at `opIdx`; one
    // that no longer fits is spilled. False when nothing was movable.
    bool compact(OpIdx opIdx, OpIdx pinnedFirstAt, BufferIdx placeFirst);
    // Record that `idx` cannot be placed at `opIdx`: every resident buffer is pinned there.
    void recordFailure(BufferIdx idx, OpIdx opIdx, PlacementFailure::Cause cause);
    // Whether nothing has read `idx` since its latest fill ran, as of `opIdx`.
    bool fillUnread(BufferIdx idx, OpIdx opIdx) const;
    // Among the resident buffers not pinned at `opIdx`; -1 if there is none.
    BufferIdx farthestNextUseBuffer(OpIdx opIdx) const;
    // Spill the buffer needed last to make room for `idx`: a victim costs one reload at its
    // next use, and this one keeps the freed space available the longest before that reload
    // comes due. False when every resident buffer is pinned at `opIdx`.
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
        PlacementFailure &failure, ArrayRef<Operation *> ops, bool spillFilledAfterUse) {
        PlacementSimulation simulation(buffers, usable, ops, spillFilledAfterUse);
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
    const LramBuffer &buffer = buffers_[idx];
    bool held = copyHoldsFrom_[idx] && !buffer.writtenBetween(*copyHoldsFrom_[idx], opIdx);
    bool store = buffer.contentsLiveAt(opIdx) && !held;
    if (store) {
        copyHoldsFrom_[idx] = opIdx;
    }
    bool cancelsFill = fillUnread(idx, opIdx);
    schedule_[opIdx].push_back(
        {Swap::Kind::Out, buffer.value, -1, store, /*fill=*/false, cancelsFill}
    );
}

void PlacementSimulation::scheduleSwapIn(BufferIdx idx, OpIdx opIdx) {
    const LramBuffer &buffer = buffers_[idx];
    bool load = buffer.contentsLiveAt(opIdx);
    // a filled buffer is never written after its fill, so its only copy is the source
    bool fill = load && buffer.fill;
    if (fill) {
        copyHoldsFrom_[idx] = opIdx;
    }
    schedule_[opIdx].push_back({Swap::Kind::In, buffer.value, occupancy_.offset(idx), load, fill});
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

bool PlacementSimulation::compact(OpIdx opIdx, OpIdx pinnedFirstAt, BufferIdx placeFirst) {
    SmallVector<BufferIdx> movable;
    for (BufferIdx candidate : occupancy_.residents()) {
        if (!buffers_[candidate].resident.contains(opIdx)) {
            movable.push_back(candidate);
        }
    }
    if (movable.empty()) {
        return false;
    }
    // those pinned at `pinnedFirstAt` go first so they do not split the space the others
    // leave for the buffer that failed there
    llvm::sort(movable, [&](BufferIdx a, BufferIdx b) {
        bool aFirst = buffers_[a].resident.contains(pinnedFirstAt);
        bool bFirst = buffers_[b].resident.contains(pinnedFirstAt);
        if (aFirst != bFirst) {
            return aFirst;
        }
        if (buffers_[a].alignedSize() != buffers_[b].alignedSize()) {
            return buffers_[a].alignedSize() > buffers_[b].alignedSize();
        }
        return a < b;
    });
    // every moved buffer is out before any comes back, so they land in the freed space
    for (BufferIdx moved : movable) {
        swapOut(moved, opIdx);
    }
    if (placeFirst >= 0 && occupancy_.place(placeFirst)) {
        scheduleSwapIn(placeFirst, opIdx);
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
    failure_.buffer = idx;
    auto residents = occupancy_.residents();
    failure_.pinned.assign(residents.begin(), residents.end());
    failure_.op = opIdx >= 0 && opIdx < (OpIdx)ops_.size() ? ops_[opIdx] : nullptr;
    failure_.bytes = buffers_[idx].alignedSize();
    failure_.residentBytes = occupancy_.bytes();
    failure_.usableBytes = occupancy_.usable();
    LLVM_DEBUG({
        // Name the op at `opIdx` and its location so the failure can be found
        // in the IR, not just by its index in the function body. Print the
        // location itself (not just FileLineColLoc): imported models (TFLite,
        // ONNX, torch-mlir) carry FusedLoc/NameLoc locations. After fusion and
        // tiling a FusedLoc can hold dozens of locations, so cut it short.
        constexpr size_t maxLocChars = 200;
        std::string opDesc;
        if (Operation *op = failure_.op) {
            std::string loc;
            llvm::raw_string_ostream(loc) << op->getLoc();
            if (loc.size() > maxLocChars)
                loc = loc.substr(0, maxLocChars) + "...";
            opDesc = " (op " + op->getName().getStringRef().str() + " at " + loc + ")";
        }
        llvm::dbgs() << "LRAM plan infeasible (" << failure_.causeName() << ") at op " << opIdx
                     << opDesc << ": cannot place buffer " << idx << " ("
                     << buffers_[idx].alignedSize() << " bytes, " << buffers_[idx].value
                     << "), resident " << occupancy_.bytes() << " of " << occupancy_.usable()
                     << " bytes\n";
        for (BufferIdx pinned : residents) {
            llvm::dbgs() << "  pinned buffer " << pinned << " size "
                         << buffers_[pinned].alignedSize() << " offset "
                         << occupancy_.offset(pinned) << " (" << buffers_[pinned].value << ")\n";
        }
    });
}

bool PlacementSimulation::fillUnread(BufferIdx idx, OpIdx opIdx) const {
    const LramBuffer &buffer = buffers_[idx];
    // the fill has run once the source holds the contents
    return buffer.fill && *copyHoldsFrom_[idx] <= opIdx &&
           !buffer.readBetween(*copyHoldsFrom_[idx], opIdx);
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

void PlacementSimulation::deferFill(BufferIdx idx) {
    const LramBuffer &buffer = buffers_[idx];
    // at the alloc: convertVirtualToPhysicalMemRefs registers the buffer swapped out
    schedule_[buffer.birth].push_back(
        {Swap::Kind::Out, buffer.value, -1, /*transfers=*/false, /*fill=*/false,
         /*cancelsFill=*/true}
    );
    deferredFills_.push_back(idx);
    scheduleReload(idx, buffer.fill->opIdx);
}

void PlacementSimulation::placeDeferredFills(OpIdx opIdx) {
    SmallVector<BufferIdx> stillDeferred;
    for (BufferIdx idx : deferredFills_) {
        const LramBuffer &buffer = buffers_[idx];
        // a forced compaction or the reload at its first use has placed it
        if (occupancy_.isResident(idx) || opIdx >= buffer.resident.nextAfter(buffer.fill->opIdx)) {
            continue;
        }
        // a swap-in at or before the fill carries no load: the next access is the fill itself
        if (opIdx <= buffer.fill->opIdx || !occupancy_.place(idx)) {
            stillDeferred.push_back(idx);
            continue;
        }
        scheduleSwapIn(idx, opIdx);
    }
    deferredFills_ = std::move(stillDeferred);
}

bool PlacementSimulation::place(BufferIdx idx, OpIdx opIdx) {
    while (occupancy_.bytes() + buffers_[idx].alignedSize() > occupancy_.usable()) {
        if (!spillVictim(idx, opIdx, PlacementFailure::Cause::BytePressure)) {
            return false;
        }
    }
    bool placed = occupancy_.place(idx);
    if (!placed && compact(opIdx, opIdx, /*placeFirst=*/-1)) {
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
    OpIdx lastOpIdx = -1;
    while (!events_.empty()) {
        Event event = events_.top();
        events_.pop();
        if (event.opIdx != lastOpIdx) {
            // the space the earlier ops released is free from here
            placeDeferredFills(event.opIdx);
            lastOpIdx = event.opIdx;
        }
        // a forced compaction precedes the events of the op it applies at
        while (!forcedCompactions.empty() && forcedCompactions.front().opIdx <= event.opIdx) {
            const ForcedCompaction &forced = forcedCompactions.front();
            BufferIdx placeFirst = forced.placeFirst;
            // an earlier compaction for the same op may have placed it already; at its alloc
            // it is not born yet, its Birth event comes after this compaction; and a swap-in
            // at or before a fill carries no load, which leaves a deferred fill unfilled
            if (placeFirst >= 0) {
                const LramBuffer &first = buffers_[placeFirst];
                OpIdx bornAt = first.fill ? first.fill->opIdx : first.birth;
                if (bornAt >= forced.opIdx || occupancy_.isResident(placeFirst)) {
                    placeFirst = -1;
                }
            }
            compact(forced.opIdx, forced.failingOpIdx, placeFirst);
            forcedCompactions = forcedCompactions.drop_front();
        }
        if (event.kind == EventKind::UseEnd) {
            // not resident when pressure spilled it during the use
            if (occupancy_.isResident(event.buffer)) {
                spill(event.buffer, event.opIdx);
            }
        }
        else if (event.kind == EventKind::Death) {
            // not resident when it was spilled with no reload before dying
            if (occupancy_.isResident(event.buffer)) {
                occupancy_.release(event.buffer);
            }
        }
        else if (event.kind == EventKind::Birth && buffers_[event.buffer].fill) {
            // a fill takes free space or waits: deferring it to the first use is free, an
            // eviction or a compaction is not
            if (!occupancy_.place(event.buffer)) {
                deferFill(event.buffer);
            }
        }
        else {
            // a forced compaction, or free space before a deferred fill's first use, placed
            // it already
            if (event.kind == EventKind::Reload && occupancy_.isResident(event.buffer)) {
                continue;
            }
            if (!place(event.buffer, event.opIdx)) {
                return false;
            }
            // the swap-outs place() scheduled at this op come first, so
            // convertVirtualToPhysicalMemRefs frees the space before it reloads into it
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
        if (offset < 0) {
            // never resident: its fill was deferred and nothing used it
            continue;
        }
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
        if (buffers[a].alignedSize() != buffers[b].alignedSize()) {
            return buffers[a].alignedSize() > buffers[b].alignedSize();
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

// Force one more compaction for `failure`: at the last op where its largest obstacle was
// still movable, or, once every such op is forced, by letting the compactions already
// forced for its op place the buffer themselves. False when nothing more can be forced.
static bool forceCompaction(
    ArrayRef<LramBuffer> buffers, const PlacementFailure &failure,
    SmallVectorImpl<ForcedCompaction> &forcedCompactions
) {
    if (auto compactAt = lastMovablePosition(buffers, failure, forcedCompactions)) {
        LLVM_DEBUG(
            llvm::dbgs() << "re-planning with a forced compaction before op " << *compactAt
                         << " for op " << failure.opIdx << "\n"
        );
        ForcedCompaction compaction{*compactAt, failure.opIdx};
        forcedCompactions.insert(
            llvm::upper_bound(
                forcedCompactions, compaction.opIdx,
                [](OpIdx opIdx, const ForcedCompaction &forced) { return opIdx < forced.opIdx; }
            ),
            compaction
        );
        return true;
    }
    bool changed = false;
    for (ForcedCompaction &compaction : forcedCompactions) {
        if (compaction.failingOpIdx == failure.opIdx && compaction.placeFirst < 0) {
            compaction.placeFirst = failure.buffer;
            changed = true;
        }
    }
    LLVM_DEBUG(if (changed) {
        llvm::dbgs() << "re-planning with the forced compactions for op " << failure.opIdx
                     << " placing its buffer\n";
    });
    return changed;
}

// The offsets and swaps that place `buffers` in [0, usable), or nullopt when none exist.
// With `replan`, a fragmentation failure forces a compaction as forceCompaction says and
// the placement is planned again, once per attempt. On failure, `lastFailure` (if given)
// describes the last attempt's.
static std::optional<Placement> planPlacement(
    ArrayRef<LramBuffer> buffers, int64_t usable, bool replan, Block *body,
    bool spillFilledAfterUse, PlacementFailure *lastFailure = nullptr
) {
    // each attempt adds one forced compaction; a stack of repeated layers needs a few per layer
    constexpr int maxReplans = 256;
    SmallVector<Operation *> ops(llvm::make_pointer_range(*body));
    SmallVector<ForcedCompaction> forcedCompactions;
    PlacementFailure failure;
    for (int attempt = 0; attempt <= maxReplans; attempt++) {
        failure = PlacementFailure();
        auto placement = PlacementSimulation::run(
            buffers, usable, forcedCompactions, failure, ops, spillFilledAfterUse
        );
        if (placement) {
            return placement;
        }
        if (!replan || failure.cause != PlacementFailure::Cause::Fragmentation) {
            break;
        }
        if (!forceCompaction(buffers, failure, forcedCompactions)) {
            break;
        }
    }
    if (lastFailure) {
        *lastFailure = std::move(failure);
    }
    return std::nullopt;
}

// True if a torq_hl.load fills `buffer`: it holds a copy of data in XRAM.
static bool isLoadedBuffer(Value buffer) {
    return llvm::any_of(buffer.getUsers(), [&](Operation *user) {
        auto loadOp = dyn_cast<torq_hl::LoadOp>(user);
        return loadOp && loadOp.getOutput() == buffer;
    });
}

// Bytes of the computed buffers that `swaps`, scheduled at one op, swap out and load back
// at a later op.
static int64_t bytesSpilledComputed(ArrayRef<Swap> swaps) {
    llvm::DenseSet<Value> swappedIn;
    for (const Swap &swap : swaps)
        if (swap.kind == Swap::Kind::In)
            swappedIn.insert(swap.buffer);
    int64_t bytes = 0;
    for (const Swap &swap : swaps) {
        if (swap.kind == Swap::Kind::Out && !swappedIn.contains(swap.buffer) &&
            !isLoadedBuffer(swap.buffer))
            bytes += getEncodedTotalSizeBytes(swap.buffer.getType());
    }
    return bytes;
}

// The bytes the swaps of a schedule transfer.
struct SwapTraffic {
    int64_t transferred = 0;
    // the part of `transferred` by buffers swapped out and back in at one op, counted once
    // per buffer: they store to XRAM only to load straight back, where a spill reloads at
    // a later op
    int64_t relocated = 0;
    // the bytes of the computed buffers the schedule spills, which a tile-fit probe marked
    // kTileFitProbeCountSpillsAttr avoids like relocations
    int64_t spilledComputed = 0;
};

static SwapTraffic swapTraffic(const SwapSchedule &schedule) {
    SwapTraffic traffic;
    llvm::SmallDenseSet<Value> cancelledFills;
    for (const auto &[opIdx, swaps] : schedule) {
        llvm::SmallDenseSet<Value> swappedOut;
        for (const Swap &swap : swaps) {
            int64_t bytes = swap.transfers ? getEncodedTotalSizeBytes(swap.buffer.getType()) : 0;
            if (swap.cancelsFill) {
                cancelledFills.insert(swap.buffer);
            }
            else if (swap.fill && cancelledFills.erase(swap.buffer)) {
                // the rewrite moves the cancelled load here; later swap-ins clone it
                bytes = 0;
            }
            traffic.transferred += bytes;
            if (swap.kind == Swap::Kind::Out) {
                swappedOut.insert(swap.buffer);
            }
            else if (swappedOut.contains(swap.buffer)) {
                traffic.relocated += bytes;
            }
        }
        traffic.spilledComputed += bytesSpilledComputed(swaps);
    }
    return traffic;
}

static std::optional<Placement> planCheapestPlacement(
    ArrayRef<LramBuffer> buffers, int64_t usable, bool replan, Block *body,
    PlacementFailure *lastFailure = nullptr
) {
    std::optional<Placement> plain =
        planPlacement(buffers, usable, replan, body, /*spillFilledAfterUse=*/false, lastFailure);
    if (!replan) {
        // the probe plans plainly: a tile must not fit only by the spills' traffic
        return plain;
    }
    std::optional<Placement> spilled =
        planPlacement(buffers, usable, replan, body, /*spillFilledAfterUse=*/true);
    if (!plain || !spilled) {
        return plain ? plain : spilled;
    }
    int64_t plainBytes = swapTraffic(plain->schedule).transferred;
    int64_t spilledBytes = swapTraffic(spilled->schedule).transferred;
    LLVM_DEBUG(
        llvm::dbgs() << "plain plan transfers " << plainBytes
                     << " bytes, spill-after-use plan transfers " << spilledBytes << " bytes\n"
    );
    return spilledBytes < plainBytes ? spilled : plain;
}

// The tile-fit probe mark of the module of `funcOp`: null outside a probe, true in a
// strict probe.
static BoolAttr getTileFitProbeAttr(FunctionOpInterface funcOp) {
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    return moduleOp ? moduleOp->getAttrOfType<BoolAttr>(kTileFitProbeAttr) : BoolAttr();
}

// LRAM begins with two NSS program slots: one for the running program, one for the next,
// which the running one loads.
constexpr int64_t kReservedLramBytes = HwInfo::nss_max_program_size * 2;

// The LRAM a plan places into, out of a pool of `poolSize` (the whole LRAM when 0): past
// the program slots, rounded down to 8 bytes. A tile-fit probe's reduced budget can give a
// pool that isn't a multiple of 8, and buffers placed top-down from its end would then get
// unaligned addresses.
static int64_t usableLramBytes(int64_t poolSize) {
    if (poolSize <= 0) {
        poolSize = TorqHw::get().getLramSize();
    }
    return static_cast<int64_t>(llvm::alignDown(poolSize, 8)) - kReservedLramBytes;
}

std::optional<int64_t> lramSwapBytes(FunctionOpInterface funcOp, int64_t poolSize) {
    std::optional<Placement> placement = planCheapestPlacement(
        collectLramBuffers(funcOp), usableLramBytes(poolSize), /*replan=*/true,
        &funcOp.getFunctionBody().front()
    );
    if (!placement) {
        return std::nullopt;
    }
    return swapTraffic(placement->schedule).transferred;
}

// Lay out every LRAM allocation over its whole lifetime; when that does not fit, schedule
// swaps and perform them with convertVirtualToPhysicalMemRefs, which places the buffers it
// swaps in where the plan put them. Infeasibility (the buffers pinned at some op cannot
// coexist) fails without a diagnostic, so runOnOperation emits the OUT_OF_MEMORY_MESSAGE
// tile-and-fuse's probe shrinks the tile on; any other diagnostic aborts the compile.
// Without `allowRelocation`, a plan that relocates buffers fails the same way. When no
// plan exists, `placementFailure` describes why, for the diagnostic.
static LogicalResult allocateLramAddresses(
    FunctionOpInterface funcOp, bool replan, bool allowRelocation,
    PlacementFailure &placementFailure, int64_t lramSize = 0
) {
    const int64_t reservedBase = kReservedLramBytes;
    const int64_t usable = usableLramBytes(lramSize);

    SmallVector<LramBuffer> buffers = collectLramBuffers(funcOp);
    std::optional<Placement> placement = planCheapestPlacement(
        buffers, usable, replan, &funcOp.getFunctionBody().front(), &placementFailure
    );
    if (!placement) {
        return failure();
    }
    if (placement->schedule.empty()) {
        return commitLramLayout(funcOp, reservedBase, buffers, placement->birthOffsets);
    }
    SwapTraffic traffic = swapTraffic(placement->schedule);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    bool countSpills = moduleOp && moduleOp->hasAttr(kTileFitProbeCountSpillsAttr);
    if (int64_t bytes = traffic.relocated + (countSpills ? traffic.spilledComputed : 0)) {
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

    // has no failure path today; lramSwapBytes treats a successful plan as a successful
    // allocation and relies on that (see convertVirtualToPhysicalMemRefs)
    if (failed(convertVirtualToPhysicalMemRefs(
            funcOp, usable, torq_hl::MemorySpace::Lram, placement->schedule, reservedBase
        ))) {
        return failure();
    }
    sinkSourceDeallocs(buffers);
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

namespace {

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
        // why no LRAM placement exists, when that is why the allocation failed
        PlacementFailure placementFailure;

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
            // a probe module marked kTileFitReplanAttr plans like a real compile
            auto moduleOp = funcOp->getParentOfType<ModuleOp>();
            bool planAsReal = replan || (moduleOp && moduleOp->hasAttr(kTileFitReplanAttr));
            bool allowRelocation = planAsReal || !probe || !probe.getValue();
            result = allocateLramAddresses(
                funcOp, planAsReal, allowRelocation, placementFailure, lramSize
            );
        } // destruct diagHandler;

        if (failed(result)) {
            // TileAndFusePass relies on this message to identify memory overflow; it
            // compares only the message, so the note does not change that
            InFlightDiagnostic diag = emitError(funcOp->getLoc()) << OUT_OF_MEMORY_MESSAGE;
            if (Operation *op = placementFailure.op) {
                diag.attachNote(op->getLoc())
                    << "no LRAM placement exists at this '" << op->getName() << "': cannot place a "
                    << placementFailure.bytes << "-byte buffer in " << placementFailure.usableBytes
                    << " usable bytes, of which " << placementFailure.residentBytes
                    << " are pinned here (" << placementFailure.causeName() << ")";
            }
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

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createAssignLramAddressesPass(bool replan, int64_t lramSize) {
    return std::make_unique<AssignLramAddressesPass>(AssignLramAddressesOptions{replan, lramSize});
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignDtcmItcmXramAddressesPass() {
    return std::make_unique<AssignDtcmItcmXramAddressesPass>();
}

} // namespace mlir::syna::torq
