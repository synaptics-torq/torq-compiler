#include <torq/Codegen/VirtualMemory.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "torq-virtual-memory"

namespace mlir::syna::torq {

namespace {

// Kept for debugging: print the LRAM/XRAM swap-in/swap-out statistics collected
// during convertVirtualToPhysicalMemRefs (see SwapStatistics below).
static llvm::cl::opt<bool> clPrintStatistics(
    "torq-vm-print-statistics", llvm::cl::desc("Print virtual memory statistics"),
    llvm::cl::init(false)
);

class VirtualAlias;
class VirtualBuffer;
class VirtualMemory;

// A memref of the converted memory space: a buffer, or a view of one.
class VirtualObject {
    VirtualMemory &vm_;
    // the memref in the input IR
    Value value_;
    // the memref holding the object now: value_ until the first swap-out, a fresh alloc
    // (a fresh view, for an alias) after each swap-in, null while swapped out
    Value physical_;
    int pinCount_ = 0;
    SmallVector<VirtualAlias *> aliases_;

  protected:
    void setPhysical(Value physical) { physical_ = physical; }
    void invalidateAliases();

  public:
    VirtualObject(VirtualMemory &vm, Value value, Value physical)
        : vm_(vm), value_(value), physical_(physical) {}
    virtual ~VirtualObject() = default;
    VirtualObject(const VirtualObject &) = delete;
    VirtualObject &operator=(const VirtualObject &) = delete;

    VirtualMemory &vm() const { return vm_; }
    Value value() const { return value_; }
    bool isSwappedOut() const { return !physical_; }
    bool isInPlace() const { return physical_ == value_; }
    Value physical() const {
        assert(physical_ && "the object is swapped out");
        return physical_;
    }

    // A pinned object stays resident: an op's operands while it is processed, and the
    // buffers a start_program accesses until its wait_program.
    bool isPinned() const { return pinCount_ > 0; }
    virtual void pin() { pinCount_++; }
    virtual void unpin() {
        assert(pinCount_ > 0 && "unpinning an object that is not pinned");
        pinCount_--;
    }

    void addAlias(VirtualAlias *alias) { aliases_.push_back(alias); }

    virtual VirtualBuffer &root() = 0;
};

class VirtualBuffer : public VirtualObject {
    const int64_t size_;

    // Copy of the contents in the swap space, kept across a swap-in and freed when the
    // buffer is written: while it exists it is up to date, so a swap-out skips the store.
    Value swappedOutValue_;

  public:
    VirtualBuffer(VirtualMemory &vm, Value value)
        : VirtualObject(vm, value, /*physical=*/value),
          size_(getEncodedTotalSizeBytes(cast<MemRefType>(value.getType()))) {}

    VirtualBuffer &root() override { return *this; }

    int64_t size() const { return size_; }

    // Free the retained swap copy; called before the contents change.
    void freeSwappedOutValue(IRRewriter &rewriter, Location loc) {
        assert(!isSwappedOut() && "the swap copy holds the only contents");
        if (!swappedOutValue_) {
            return;
        }
        memref::DeallocOp::create(rewriter, loc, swappedOutValue_);
        swappedOutValue_ = nullptr;
    }

    void swapIn(IRRewriter &rewriter, Location loc, int64_t address);

    void swapOut(IRRewriter &rewriter, Location loc);

    // Rewrite the buffer's dealloc to whatever storage is left, or erase it.
    void deallocate(memref::DeallocOp deallocOp, IRRewriter &rewriter);
};

// A view of a virtual object.
class VirtualAlias : public VirtualObject {
    VirtualObject &parent_;
    OpOperand &parentOperand_;

  public:
    VirtualAlias(VirtualMemory &vm, Value value, VirtualObject &parent)
        // the view is usable as is while its parent is in place; once the parent has
        // moved, the view must be cloned onto the parent's physical value
        : VirtualObject(vm, value, parent.isInPlace() ? value : Value()), parent_(parent),
          parentOperand_(getDerivedMemRefBase(value.getDefiningOp())) {
        parent.addAlias(this);
    }

    VirtualBuffer &root() override { return parent_.root(); }

    void pin() override {
        VirtualObject::pin();
        parent_.pin();
    }

    void unpin() override {
        VirtualObject::unpin();
        parent_.unpin();
    }

    void invalidate() {
        setPhysical(nullptr);
        invalidateAliases();
    }

    void swapIn(IRRewriter &rewriter) {
        if (parent_.isSwappedOut()) {
            llvm::report_fatal_error("LRAM plan: a view is used while its buffer is swapped out");
        }
        Operation *view = rewriter.clone(*value().getDefiningOp());
        view->getOpOperand(parentOperand_.getOperandNumber()).set(parent_.physical());
        setPhysical(view->getResult(0));

        LLVM_DEBUG({
            llvm::dbgs() << "Swapped in alias ";
            value().dump();
        });
    }
};

void VirtualObject::invalidateAliases() {
    for (VirtualAlias *alias : aliases_) {
        alias->invalidate();
    }
}

// Swap traffic accounting for the LRAM/XRAM spill buffers, printed by
// printStatistics when --torq-vm-print-statistics is set. Debugging only; does not
// affect codegen.
struct SwapStatistics {
    int swapOuts = 0;
    int storesSkipped = 0;
    int dropped = 0;
    int64_t droppedBytes = 0;
    int loadsSkipped = 0;
    int64_t storedBytes = 0;
    int64_t loadedBytes = 0;
};

// The virtual objects of one memory space and the swap space their buffers spill to.
class VirtualMemory {
    llvm::MapVector<Value, std::unique_ptr<VirtualObject>> objects_;

  public:
    const int64_t capacity;
    const torq_hl::MemorySpace memorySpace;
    const torq_hl::MemorySpace swapMemSpace;
    const int64_t addressBase;
    SwapStatistics stats;

    // ordered (op index, contents fully overwritten) accesses per buffer
    DenseMap<Value, SmallVector<std::pair<unsigned, bool>>> bufferAccessSchedule;

    // index (into the processed op list) of the op being processed
    unsigned currentOpIndex = 0;

    VirtualMemory(int64_t capacity, torq_hl::MemorySpace memorySpace, int64_t addressBase)
        : capacity(capacity), memorySpace(memorySpace),
          swapMemSpace(
              memorySpace == torq_hl::MemorySpace::Lram ? torq_hl::MemorySpace::Xram
                                                        : torq_hl::MemorySpace::Lram
          ),
          addressBase(addressBase) {}

    VirtualObject &object(Value virtualValue) {
        auto it = objects_.find(virtualValue);
        assert(it != objects_.end() && "virtual object not found");
        return *it->second;
    }

    VirtualBuffer &buffer(Value virtualValue) {
        VirtualBuffer &root = object(virtualValue).root();
        assert(root.value() == virtualValue && "not a buffer");
        return root;
    }

    // Register an allocation; it is resident from the start.
    void addBuffer(Value virtualValue) {
        assert(!objects_.contains(virtualValue) && "virtual object already registered");
        objects_.try_emplace(virtualValue, std::make_unique<VirtualBuffer>(*this, virtualValue));
        assert(residentBytes() <= capacity && "resident bytes exceed the capacity");
    }

    void addAlias(Value virtualValue) {
        assert(!objects_.contains(virtualValue) && "virtual object already registered");
        VirtualObject &parent = object(getDerivedMemRefBase(virtualValue.getDefiningOp()).get());
        objects_.try_emplace(
            virtualValue, std::make_unique<VirtualAlias>(*this, virtualValue, parent)
        );
    }

    int64_t residentBytes() const {
        int64_t bytes = 0;
        for (const auto &[value, object] : objects_) {
            VirtualBuffer &root = object->root();
            if (&root == object.get() && !root.isSwappedOut()) {
                bytes += root.size();
            }
        }
        return bytes;
    }

    // Whether the buffer's current contents are dead: no access at or after the op being
    // processed observes them, because the next one overwrites them whole or none remains.
    bool contentsDead(Value rootValue) const {
        auto it = bufferAccessSchedule.find(rootValue);
        if (it == bufferAccessSchedule.end()) {
            return true;
        }
        const auto *access = llvm::lower_bound(
            it->second, currentOpIndex,
            [](const std::pair<unsigned, bool> &access, unsigned opIndex) {
                return access.first < opIndex;
            }
        );
        return access == it->second.end() || access->second;
    }
};

void VirtualBuffer::swapIn(IRRewriter &rewriter, Location loc, int64_t address) {
    assert(isSwappedOut() && "the buffer is resident");

    auto physicalAlloc =
        memref::AllocOp::create(rewriter, loc, cast<MemRefType>(value().getType()));
    if (failed(setAddress(physicalAlloc.getOperation(), address))) {
        llvm::report_fatal_error("LRAM plan: a swapped-in buffer cannot carry an address");
    }

    if (swappedOutValue_) {
        // overwriting the alignment bytes of the new buffer as well
        torq_hl::LoadOp::create(
            rewriter, loc, physicalAlloc, swappedOutValue_, SmallVector<int64_t>{},
            SmallVector<int64_t>{}, getEncodedTotalSizeBytes(physicalAlloc.getType()), true
        );
        vm().stats.loadedBytes += size();
    }
    else {
        // the contents were dropped at swap-out: nothing to load
        vm().stats.loadsSkipped++;
    }

    setPhysical(physicalAlloc);
    assert(vm().residentBytes() <= vm().capacity && "resident bytes exceed the capacity");

    LLVM_DEBUG({
        llvm::dbgs() << "Swapped in buffer ";
        value().dump();
    });
}

void VirtualBuffer::swapOut(IRRewriter &rewriter, Location loc) {
    assert(!isSwappedOut() && "the buffer is swapped out");
    assert(!isPinned() && "the buffer is pinned");

    LLVM_DEBUG({
        llvm::dbgs() << "Swapping out buffer ";
        value().dump();
    });

    if (vm().contentsDead(value())) {
        // nothing observes the contents: drop them, and the swap-in loads nothing
        freeSwappedOutValue(rewriter, loc);
        vm().stats.dropped++;
        vm().stats.droppedBytes += size();
    }
    else if (swappedOutValue_) {
        // the swap copy is still up to date
        vm().stats.storesSkipped++;
    }
    else {
        // otherwise create the swap copy (an existing one is still up to date and kept as is)
        auto physicalType = cast<MemRefType>(value().getType());
        auto swappedOutType = createMemRefTypeWithEncoding(
            physicalType,
            cloneEncodingWithNewMemorySpace(getEncoding(physicalType), vm().swapMemSpace)
        );
        swappedOutValue_ = memref::AllocOp::create(rewriter, loc, swappedOutType);

        // overwriting the alignment bytes of the swap copy as well
        torq_hl::StoreOp::create(
            rewriter, loc, swappedOutValue_, physical(), SmallVector<int64_t>{},
            SmallVector<int64_t>{}, getEncodedTotalSizeBytes(swappedOutType), true
        );
        vm().stats.storedBytes += size();
    }

    invalidateAliases();
    memref::DeallocOp::create(rewriter, loc, physical());
    setPhysical(nullptr);
    vm().stats.swapOuts++;
}

void VirtualBuffer::deallocate(memref::DeallocOp deallocOp, IRRewriter &rewriter) {
    rewriter.setInsertionPoint(deallocOp);
    Value target;
    if (isSwappedOut()) {
        target = swappedOutValue_;
    }
    else {
        freeSwappedOutValue(rewriter, deallocOp.getLoc());
        target = physical();
    }
    swappedOutValue_ = nullptr;
    setPhysical(nullptr);
    if (target) {
        deallocOp.getMemrefMutable().set(target);
    }
    else {
        rewriter.eraseOp(deallocOp);
    }
}

// Precompute, for every allocation in `memorySpace`, the ordered list of the
// function's accesses to it, so eviction can ask whether any remaining access
// observes a buffer's current contents.
static void buildBufferAccessSchedule(
    ArrayRef<Operation *> ops, torq_hl::MemorySpace memorySpace,
    DenseMap<Value, SmallVector<std::pair<unsigned, bool>>> &schedule
) {
    for (auto [index, op] : llvm::enumerate(ops)) {
        if (isa<memref::DeallocOp>(op) || isDerivedMemRefOperation(op)) {
            continue;
        }

        for (const auto &[root, access] : collectRootAccesses(op, memorySpace)) {
            schedule[root].emplace_back(index, access == torq_hl::ArgAccessBitfield::Write);
        }
    }
}

struct OperationMemoryInfo {
    SmallVector<OpOperand *> operands;
    SmallVector<OpResult> results;
    SmallVector<torq_hl::AsyncAccess> accesses;
};

static OperationMemoryInfo
collectOperationMemoryInfo(Operation *op, torq_hl::MemorySpace memorySpace) {
    OperationMemoryInfo info;
    for (auto &operand : op->getOpOperands()) {
        auto type = dyn_cast<MemRefType>(operand.get().getType());
        if (type && getEncodingMemorySpace(type) == memorySpace) {
            info.operands.push_back(&operand);
        }
    }

    for (auto &access : collectBufferAccesses(op)) {
        if (getEncodingMemorySpace(access.buffer.getType()) == memorySpace) {
            info.accesses.push_back(access);
        }
    }

    for (auto result : op->getResults()) {
        auto type = dyn_cast<MemRefType>(result.getType());
        if (type && getEncodingMemorySpace(type) == memorySpace) {
            info.results.push_back(result);
        }
    }
    return info;
}

static bool processDeallocationOrAlias(
    Operation *op, VirtualMemory &vm, IRRewriter &rewriter, torq_hl::MemorySpace memorySpace
) {
    if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        if (getEncodingMemorySpace(deallocOp.getMemref().getType()) == memorySpace) {
            vm.buffer(deallocOp.getMemref()).deallocate(deallocOp, rewriter);
        }
        return true;
    }

    if (isDerivedMemRefOperation(op)) {
        auto &baseMemRef = getDerivedMemRefBase(op);
        if (getEncodingMemorySpace(cast<MemRefType>(baseMemRef.get().getType())) == memorySpace) {
            vm.addAlias(op->getResult(0));
        }
        return true;
    }
    return false;
}

static void materializeOperandsAndAllocateResults(
    Operation *op, const OperationMemoryInfo &info, VirtualMemory &vm, IRRewriter &rewriter
) {
    rewriter.setInsertionPoint(op);
    for (auto *operand : info.operands) {
        VirtualObject &object = vm.object(operand->get());
        if (!object.isSwappedOut()) {
            continue;
        }
        // the plan reloads a buffer before the op that uses it; only a view can still be
        // swapped out here, when its buffer moved, and it is re-created on the new value
        if (&object == &object.root()) {
            llvm::report_fatal_error("LRAM plan: a buffer is used while it is swapped out");
        }
        static_cast<VirtualAlias &>(object).swapIn(rewriter);
    }
    for (auto result : info.results) {
        vm.addBuffer(result);
    }
}

// Pin `operands` and point them at their physical values.
static SmallVector<Value>
rewriteOperandsToPhysicalMemory(ArrayRef<OpOperand *> operands, VirtualMemory &vm) {
    SmallVector<Value> pinnedValues;
    for (auto *operand : operands) {
        Value virtualValue = operand->get();
        VirtualObject &object = vm.object(virtualValue);
        object.pin();
        pinnedValues.push_back(virtualValue);
        operand->set(object.physical());
    }
    return pinnedValues;
}

// Unpin `op`'s operands, except that a start_program keeps the buffers it accesses pinned
// until its wait_program (the planner pins the same set), and a wait_program releases them.
static void updateOperandPinLifetimes(
    Operation *op, ArrayRef<Value> pinnedValues, ArrayRef<torq_hl::AsyncAccess> accesses,
    VirtualMemory &vm, llvm::MapVector<Value, SmallVector<Value>> &invocationToVirtual
) {
    if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
        SmallVector<Value> &programPinned = invocationToVirtual[startOp.getInvocation()];
        for (Value value : pinnedValues) {
            bool accessed = llvm::any_of(accesses, [value](const torq_hl::AsyncAccess &access) {
                return access.buffer == value;
            });
            if (accessed) {
                programPinned.push_back(value);
            }
            else {
                vm.object(value).unpin();
            }
        }
        return;
    }

    for (Value value : pinnedValues) {
        vm.object(value).unpin();
    }
    if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
        for (Value value : invocationToVirtual[waitOp.getInvocation()]) {
            vm.object(value).unpin();
        }
        invocationToVirtual.erase(waitOp.getInvocation());
    }
}

// Perform the swaps scheduled at `op`, in the planner's order.
static void performScheduledSwaps(
    Operation *op, ArrayRef<Swap> swaps, VirtualMemory &vm, IRRewriter &rewriter
) {
    rewriter.setInsertionPoint(op);
    for (const Swap &swap : swaps) {
        if (swap.kind == Swap::Kind::Out) {
            vm.buffer(swap.buffer).swapOut(rewriter, op->getLoc());
        }
        else {
            vm.buffer(swap.buffer).swapIn(rewriter, op->getLoc(), vm.addressBase + swap.offset);
        }
    }
}

// Debug aid (--torq-vm-print-statistics): summarize the spill traffic the scheduler
// generated, to see how much LRAM pressure a model puts on the swap space.
static void printStatistics(const SwapStatistics &stats) {
    if (!clPrintStatistics) {
        return;
    }
    llvm::dbgs() << "Total swap outs: " << stats.swapOuts << "\n";
    llvm::dbgs() << "Total swap-out stores skipped (clean buffers): " << stats.storesSkipped
                 << "\n";
    llvm::dbgs() << "Swap outs dropped (contents dead): " << stats.dropped << " ("
                 << stats.droppedBytes << " bytes)\n";
    llvm::dbgs() << "Swap-in loads skipped (dropped contents): " << stats.loadsSkipped << "\n";
    llvm::dbgs() << "Swap-out bytes stored: " << stats.storedBytes << "\n";
    llvm::dbgs() << "Swap-in bytes loaded: " << stats.loadedBytes << "\n";
}

} // namespace

LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, int capacityBytes, torq_hl::MemorySpace memorySpace,
    const SwapSchedule &schedule, int64_t addressBase
) {
    VirtualMemory vm(capacityBytes, memorySpace, addressBase);
    IRRewriter rewriter(funcOp);

    // Snapshot the original operations: the schedule's op indices refer to it, and swapping
    // inserts new operations while we walk.
    SmallVector<Operation *> ops;
    for (auto &op : funcOp.getFunctionBody().getOps()) {
        ops.push_back(&op);
    }

    // Future accesses determine whether an evicted buffer must be copied to swap memory.
    buildBufferAccessSchedule(ops, memorySpace, vm.bufferAccessSchedule);

    // Keep StartProgramOp operands pinned until the corresponding WaitProgramOp.
    llvm::MapVector<Value, SmallVector<Value>> invocationToVirtual;
    for (auto [opIndex, op] : llvm::enumerate(ops)) {
        // the scheduled swap-outs judge the deadness of a buffer's contents relative to this op
        vm.currentOpIndex = opIndex;

        if (auto it = schedule.find((int)opIndex); it != schedule.end()) {
            performScheduledSwaps(op, it->second, vm, rewriter);
        }

        LLVM_DEBUG({
            llvm::dbgs() << "------------\n";
            llvm::dbgs() << "Processing operation: ";
            op->dump();
        });

        // Deallocations update allocator state; aliases only register their virtual parent.
        if (processDeallocationOrAlias(op, vm, rewriter, memorySpace)) {
            continue;
        }

        // Inspect virtual operands and results before any operand is rewritten.
        OperationMemoryInfo memoryInfo = collectOperationMemoryInfo(op, memorySpace);

        materializeOperandsAndAllocateResults(op, memoryInfo, vm, rewriter);

        // Keep operands resident during execution and point the op at their physical values.
        SmallVector<Value> pinnedOperands =
            rewriteOperandsToPhysicalMemory(memoryInfo.operands, vm);

        // A write makes any retained swap-space copy stale.
        for (const torq_hl::AsyncAccess &access : memoryInfo.accesses) {
            if (bitEnumContainsAny(access.access, torq_hl::ArgAccessBitfield::Write)) {
                vm.object(access.buffer).root().freeSwappedOutValue(rewriter, op->getLoc());
            }
        }

        updateOperandPinLifetimes(op, pinnedOperands, memoryInfo.accesses, vm, invocationToVirtual);
    }

    printStatistics(vm.stats);
    return success();
}

} // namespace mlir::syna::torq
