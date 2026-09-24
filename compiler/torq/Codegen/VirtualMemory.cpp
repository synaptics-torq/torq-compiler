#include <torq/Codegen/VirtualMemory.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-virtual-memory"

namespace mlir::syna::torq {

namespace {

static llvm::cl::opt<bool> clAnnotateVirtualBufferIds(
    "torq-vm-annotate-virtual-buffer-ids", llvm::cl::desc("Annotate virtual buffer IDs"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clPrintStatistics(
    "torq-vm-print-statistics", llvm::cl::desc("Print virtual memory statistics"),
    llvm::cl::init(false)
);

static const std::string VIRTUAL_OBJECT_ID_ATTR_NAME = "torq-virtual-buffer-id";

class VirtualBuffer;
class VirtualAlias;
class VirtualObject;
class VirtualMemory;

class PhysicalObject {
    Value value_;

  public:
    PhysicalObject(const PhysicalObject &) = delete;
    PhysicalObject &operator=(const PhysicalObject &) = delete;
    PhysicalObject(PhysicalObject &&) = delete;
    PhysicalObject &operator=(PhysicalObject &&) = delete;

    VirtualMemory &vm();

    bool isPinned();

    Value value() const { return value_; }

    virtual VirtualObject &virtualObject() = 0;

    PhysicalObject(Value value) : value_(value) {}

    virtual ~PhysicalObject() = default;
};

class PhysicalBuffer : public PhysicalObject {
    VirtualBuffer &virtualBuffer_;

  public:
    VirtualBuffer &virtualBuffer() { return virtualBuffer_; }

    virtual VirtualObject &virtualObject() override;

    int size();

    PhysicalBuffer(Value value, VirtualBuffer &virtualBuffer)
        : PhysicalObject{value}, virtualBuffer_(virtualBuffer) {}
};

class PhysicalAlias : public PhysicalObject {

    VirtualAlias &virtualAlias_;

  public:
    PhysicalObject &parent();

    VirtualAlias &virtualAlias();

    virtual VirtualObject &virtualObject() override;

    PhysicalAlias(Value value, VirtualAlias &virtualAlias)
        : PhysicalObject{value}, virtualAlias_(virtualAlias) {}
};

class VirtualObject {
    int id_;
    VirtualMemory &vm_;
    Value value_;
    int pinCount_{0};
    SmallVector<std::unique_ptr<VirtualAlias>> aliases_;

  public:
    VirtualObject(const VirtualObject &) = delete;
    VirtualObject &operator=(const VirtualObject &) = delete;
    VirtualObject(VirtualObject &&) = delete;
    VirtualObject &operator=(VirtualObject &&) = delete;

    virtual VirtualBuffer &root() = 0;

    SmallVector<std::unique_ptr<VirtualAlias>> &aliases() { return aliases_; }

    int id() const { return id_; }

    VirtualMemory &vm() const { return vm_; }

    Value value() const { return value_; }

    virtual PhysicalObject &physicalObject() = 0;

    virtual bool isSwappedOut() = 0;

    int isPinned() const { return pinCount_ > 0; }

    int pinCount() const { return pinCount_; }

    virtual void pin() { pinCount_++; }

    virtual void unpin() {
        assert(pinCount_ > 0 && "Unpinning a non-pinned object");
        pinCount_--;
    }

    virtual PhysicalObject *swapIn(IRRewriter &rewriter, Location loc) = 0;

    VirtualAlias &addAlias(Value virtualValue) {
        auto &aliasPtr =
            aliases_.emplace_back(std::make_unique<VirtualAlias>(virtualValue, *this, vm()));
        return *aliasPtr;
    }

    VirtualObject(VirtualMemory &vm, Value value);

    virtual ~VirtualObject() = default;
};

class VirtualBuffer : public VirtualObject {
    int size_;
    std::optional<PhysicalBuffer *> maybePhysicalBuffer_;

    // Copy of the buffer data in the swap space. Retained across swap in and
    // freed when the swapped-in buffer is written. So when non-null, it is up
    // to date and swapping out can skip the store.
    Value swappedOutValue_;

  public:
    virtual VirtualBuffer &root() override { return *this; }

    // free the retained swapped out buffer; called before the contents change
    void freeSwappedOutValue(IRRewriter &rewriter, Location loc) {
        assert(!isSwappedOut() && "the swapped out buffer holds the only copy");
        if (!swappedOutValue_) {
            return;
        }
        memref::DeallocOp::create(rewriter, loc, swappedOutValue_);
        swappedOutValue_ = nullptr;
    }

    virtual void pin() override;

    virtual void unpin() override;

    int size() const { return size_; }

    Value swappedOutValue() const { return swappedOutValue_; }

    virtual PhysicalObject &physicalObject() override { return physicalBuffer(); }

    PhysicalBuffer &physicalBuffer() {
        assert(maybePhysicalBuffer_.has_value() && "Physical buffer is not allocated");
        return *(maybePhysicalBuffer_.value());
    }

    virtual bool isSwappedOut() override { return !maybePhysicalBuffer_.has_value(); }

    virtual PhysicalObject *swapIn(IRRewriter &rewriter, Location loc) override;

    void swapOut(IRRewriter &rewriter, Location loc);

    void initialize();

    void dump() {
        llvm::dbgs() << "Buffer id:" << id() << " size: " << size_ << " value: ";
        value().dump();
    }

    VirtualBuffer(Value value, VirtualMemory &vm) : VirtualObject{vm, value} {

        size_ = getEncodedTotalSizeBytes(cast<MemRefType>(value.getType()));

        if (clAnnotateVirtualBufferIds) {
            // set the id attribute on the operation that generated the value
            auto opResult = dyn_cast<OpResult>(value);

            if (opResult) {

                SmallVector<int64_t> resultIds(opResult.getOwner()->getNumResults());

                // if an operator returns multiple results, the attribute may already be set
                auto existingAttr =
                    opResult.getOwner()->getAttrOfType<ArrayAttr>(VIRTUAL_OBJECT_ID_ATTR_NAME);
                if (existingAttr) {
                    for (auto prevId : existingAttr.getValue()) {
                        resultIds.push_back(cast<IntegerAttr>(prevId).getSInt());
                    }
                }

                // set the id for the current result
                resultIds[opResult.getResultNumber()] = id();

                // update the attribute
                IRRewriter rewriter = IRRewriter(opResult.getContext());
                opResult.getOwner()->setAttr(
                    VIRTUAL_OBJECT_ID_ATTR_NAME, rewriter.getIndexArrayAttr(resultIds)
                );
            }
        }
    }
};

class VirtualAlias : public VirtualObject {

    std::optional<PhysicalAlias> maybePhysicalAlias;
    VirtualObject &parent_;
    OpOperand &parentOperand_;

  public:
    virtual VirtualBuffer &root() override { return parent().root(); }

    VirtualObject &parent() { return parent_; }

    virtual void pin() override {
        VirtualObject::pin();
        parent_.pin();
    }

    virtual void unpin() override {
        VirtualObject::unpin();
        parent_.unpin();
    }

    void invalidate() {
        maybePhysicalAlias = std::nullopt;

        // make sure all the aliases pointing to this alias are invalidated
        for (auto &alias : aliases()) {
            alias->invalidate();
        }
    }

    virtual PhysicalObject *swapIn(IRRewriter &rewriter, Location loc) override {

        LLVM_DEBUG({
            llvm::dbgs() << "Swapping in virtual alias ";
            value().dump();
        });

        if (parent().isSwappedOut()) {
            parent().swapIn(rewriter, loc);
        }

        auto opResult = dyn_cast<OpResult>(value());
        auto physicalValueOp = rewriter.clone(*(opResult.getOwner()));

        // create a new copy of the operation with using the right current physical value
        auto &opOperand = physicalValueOp->getOpOperand(parentOperand_.getOperandNumber());
        opOperand.set(parent().physicalObject().value());

        maybePhysicalAlias.emplace(physicalValueOp->getResult(0), *this);

        LLVM_DEBUG({
            llvm::dbgs() << "Swapped in to physical alias ";
            physicalObject().value().dump();
        });

        return &(maybePhysicalAlias.value());
    }

    virtual PhysicalObject &physicalObject() override {
        assert(!isSwappedOut() && "Cannot get physical object of a swapped out alias");
        return maybePhysicalAlias.value();
    }

    virtual bool isSwappedOut() override { return !maybePhysicalAlias.has_value(); }

    VirtualAlias(Value value, VirtualObject &parent, VirtualMemory &vm)
        : VirtualObject{vm, value}, parent_(parent),
          parentOperand_(getDerivedMemRefBase(value.getDefiningOp())) {

        // initially the virtual alias point to the virtual value if the parent is not
        // and was never before swapped out
        if (!parent.isSwappedOut() && parent.physicalObject().value() == parent.value()) {
            maybePhysicalAlias.emplace(value, *this);
        }
    }
};

// This class is used to track the currently active physical buffers and theirs state
class PhysicalMemory {

    llvm::MapVector<VirtualBuffer *, std::unique_ptr<PhysicalBuffer>> physicalBuffers_;
    llvm::MapVector<PhysicalBuffer *, int> pinnedBuffers_;
    int totalPinnedSize_ = 0;
    int capacity_;
    int residentBytes_ = 0;
    int swapOutCount_ = 0;
    int skippedSwapOutStoreCount_ = 0;
    int droppedSwapOutCount_ = 0;
    int64_t droppedSwapOutBytes_ = 0;
    int skippedSwapInLoadCount_ = 0;
    int64_t swapOutStoredBytes_ = 0;
    int64_t swapInLoadedBytes_ = 0;

  public:
    explicit PhysicalMemory(int capacity) : capacity_(capacity) {}

    int totalPinnedSize() const { return totalPinnedSize_; }

    int totalPhysicalBufferSize() const { return residentBytes_; }

    int usableSize() const { return capacity_; }

    int swapOutCount() const { return swapOutCount_; }

    void countSwapOut() { swapOutCount_++; }

    void noteSkippedSwapOutStore() { skippedSwapOutStoreCount_++; }

    int skippedSwapOutStoreCount() const { return skippedSwapOutStoreCount_; }

    void noteDroppedSwapOut(int64_t bytes) {
        droppedSwapOutCount_++;
        droppedSwapOutBytes_ += bytes;
    }

    int droppedSwapOutCount() const { return droppedSwapOutCount_; }

    int64_t droppedSwapOutBytes() const { return droppedSwapOutBytes_; }

    void noteSkippedSwapInLoad() { skippedSwapInLoadCount_++; }

    int skippedSwapInLoadCount() const { return skippedSwapInLoadCount_; }

    void noteSwapOutStore(int64_t bytes) { swapOutStoredBytes_ += bytes; }

    int64_t swapOutStoredBytes() const { return swapOutStoredBytes_; }

    void noteSwapInLoad(int64_t bytes) { swapInLoadedBytes_ += bytes; }

    int64_t swapInLoadedBytes() const { return swapInLoadedBytes_; }

    void pin(PhysicalBuffer &object) {
        auto pinCount = pinnedBuffers_[&object];

        if (pinCount == 0) {
            totalPinnedSize_ += object.size();
        }

        pinnedBuffers_[&object]++;
    }

    void unpin(PhysicalBuffer &object) {
        assert(pinnedBuffers_.contains(&object) && "Buffer not pinned");

        auto pinCount = pinnedBuffers_[&object];

        if (pinCount == 1) {
            pinnedBuffers_.erase(&object);
            totalPinnedSize_ -= object.size();
        }
        else {
            pinnedBuffers_[&object] = pinCount - 1;
        }
    }

    int pinCount(PhysicalBuffer &object) {
        if (pinnedBuffers_.contains(&object)) {
            return pinnedBuffers_[&object];
        }
        return 0;
    }

    PhysicalBuffer *add(VirtualBuffer &obj, Value value) {
        assert(!physicalBuffers_.contains(&obj) && "Physical buffer already exists for this value");
        assert(obj.size() + residentBytes_ <= capacity_ && "resident bytes exceed the capacity");

        residentBytes_ += obj.size();

        return physicalBuffers_.try_emplace(&obj, std::make_unique<PhysicalBuffer>(value, obj))
            .first->second.get();
    }

    void remove(VirtualBuffer &obj) {
        assert(physicalBuffers_.contains(&obj) && "Physical buffer not found");
        residentBytes_ -= physicalBuffers_[&obj]->size();
        physicalBuffers_.erase(&obj);
    }

    void dump() {

        llvm::dbgs() << "Current memory usage " << totalPhysicalBufferSize() << "\n";

        llvm::dbgs() << "Active virtual allocations:\n";
        for (auto &[value, buf] : physicalBuffers_) {
            auto virtualValue = buf->virtualObject().value();
            llvm::dbgs() << "   - size " << buf->size() << " ";
            llvm::dbgs() << "pinned " << pinnedBuffers_.contains(buf.get()) << " ";
            virtualValue.dump();
        }
    }

    void miniDump() {
        llvm::dbgs() << "Current memory usage " << totalPhysicalBufferSize() << "\n";
    }
};

// This class is used to track the currently active virtual objects
class VirtualObjects {
    llvm::MapVector<Value, std::unique_ptr<VirtualBuffer>> virtualBuffers_;
    llvm::MapVector<Value, VirtualObject *> virtualObjects_;
    VirtualMemory &vm_;

  public:
    VirtualObjects(VirtualMemory &vm) : vm_(vm) {}

    void addBuffer(Value value) {
        assert(!virtualBuffers_.contains(value) && "Virtual buffer already exists for this value");

        virtualBuffers_.try_emplace(value, std::make_unique<VirtualBuffer>(value, vm_));
        virtualBuffers_[value]->initialize();
        virtualObjects_.try_emplace(value, virtualBuffers_[value].get());
    }

    void addAlias(Value value) {
        auto &baseOpOperand = getDerivedMemRefBase(value.getDefiningOp());
        auto &parentObject = getVirtualObject(baseOpOperand.get());
        auto &alias = parentObject.addAlias(value);
        virtualObjects_.try_emplace(value, &alias);
    }

    VirtualObject &getVirtualObject(Value virtualValue) {
        auto it = virtualObjects_.find(virtualValue);
        assert(it != virtualObjects_.end() && "virtual object not found");
        return *(it->second);
    }

    VirtualBuffer &getVirtualBuffer(Value virtualValue) {
        auto it = virtualBuffers_.find(virtualValue);
        assert(it != virtualBuffers_.end() && "virtual buffer not found");
        return *(it->second);
    }

    Value getPhysicalValue(Value virtualValue) {
        auto &virtualObject = getVirtualObject(virtualValue);
        assert(!virtualObject.isSwappedOut() && "virtual object is swapped out");
        return virtualObject.physicalObject().value();
    }
};

// Virtual memory is a class that helps convert all memrefs of a given memory space
// from a virtual buffers that are allocated as if the memory space is unlimited to
// physical buffers that are allocated in a memory space of limited size.
//
// Memrefs in the virtual space are represented by VirtualObjects and can be
// either VirtualBuffers (that represent actual memory buffers) or VirtualAliases that
// are an alias to (part of) a virtual buffer or another virtual alias.
//
// Memref in the physical space are represented in a similar way by PhysicalObjects
// (either PhysicalBuffer and PhysicalAlias).
//
// State of virtual and physical objects changes during the execution of a program.
//
// At any given point in time VirtualObjects can be in two states: swapped in or
// swapped out. When they are swapped in they can be accesses in target memory space
// through the associated physical buffer (resp physical alias). When they are
// swapped out their data is in a buffer in the swap out memory space and they need
// to be swapped to a new physical buffer (or alias ) to be accessed.
//
// Physical objects (and implicitely the virtual objects associated to them) can be
// pinned: an op's operands while it is processed, and the buffers a start_program
// accesses until its wait_program.

class VirtualMemory {

  public:
    VirtualMemory(int capacity, torq_hl::MemorySpace memorySpace)
        : virtualObjects(*this), physicalMemory(capacity), memorySpace(memorySpace),
          swapMemSpace(
              memorySpace == torq_hl::MemorySpace::Lram ? torq_hl::MemorySpace::Xram
                                                        : torq_hl::MemorySpace::Lram
          ) {}

    int getNextId() { return nextId++; }

    // deallocates the virtual buffer: updates the internal state of the virtual
    // memory system and returns the value the existing deallocate operation must
    // target (a retained swappedOutValue_ gets its own deallocation). A null
    // return means nothing needs deallocation (the contents were dropped at swap
    // out) and the deallocate operation must be erased.
    Value deallocate(Value virtualValue, IRRewriter &rewriter) {

        LLVM_DEBUG({
            llvm::dbgs() << "Deallocate virtual value ";
            virtualValue.dump();
        });

        auto &virtualBuffer = virtualObjects.getVirtualBuffer(virtualValue);

        Value physicalValue;
        if (virtualBuffer.isSwappedOut()) {
            physicalValue = virtualBuffer.swappedOutValue();
        }
        else {
            virtualBuffer.freeSwappedOutValue(rewriter, virtualValue.getLoc());
            auto &physicalObject = virtualBuffer.physicalBuffer();
            physicalValue = physicalObject.value();
            physicalMemory.remove(virtualBuffer);
        }

        LLVM_DEBUG({ physicalMemory.miniDump(); });

        return physicalValue;
    }

    void addAlias(Value virtualValue) {

        LLVM_DEBUG({
            llvm::dbgs() << "Allocate alias for virtual value ";
            virtualValue.dump();
        });

        virtualObjects.addAlias(virtualValue);

        LLVM_DEBUG({ physicalMemory.miniDump(); });
    }

    void addAllocation(Value virtualValue) {

        LLVM_DEBUG({
            llvm::dbgs() << "Allocate for virtual value ";
            virtualValue.dump();
        });

        virtualObjects.addBuffer(virtualValue);

        LLVM_DEBUG({ physicalMemory.miniDump(); });
    }

    void freeSwappedOutValue(Value virtualValue, IRRewriter &rewriter, Location loc) {
        virtualObjects.getVirtualObject(virtualValue).root().freeSwappedOutValue(rewriter, loc);
    }

    // inserts IR to copy the swapped out virtual buffer back to the the memory space into
    // a new physical buffer and updates the virtual memory system internal state
    Value swapIn(Value virtualValue, IRRewriter &rewriter, Location loc) {

        LLVM_DEBUG({
            llvm::dbgs() << "Swap-in virtual value ";
            virtualValue.dump();
        });

        auto &virtualObject = virtualObjects.getVirtualObject(virtualValue);
        virtualObject.swapIn(rewriter, loc);

        LLVM_DEBUG({ physicalMemory.miniDump(); });

        return virtualObject.physicalObject().value();
    }

    // Swap out a scheduled buffer; VirtualBuffer::swapOut asserts it is resident and unpinned.
    void swapOut(Value virtualValue, IRRewriter &rewriter, Location loc) {
        virtualObjects.getVirtualBuffer(virtualValue).swapOut(rewriter, loc);
        physicalMemory.countSwapOut();
    }

    bool isSwappedOut(Value virtualValue) {
        return virtualObjects.getVirtualObject(virtualValue).isSwappedOut();
    }

    Value getPhysicalValue(Value virtualValue) {
        return virtualObjects.getVirtualObject(virtualValue).physicalObject().value();
    }

    // mark a virtual value not to be swappable (in use)
    void pin(Value virtualValue) {

        LLVM_DEBUG({
            llvm::dbgs() << "Pinning virtual value ";
            virtualValue.dump();
        });

        virtualObjects.getVirtualObject(virtualValue).pin();

        LLVM_DEBUG({ physicalMemory.miniDump(); });
    }

    // unmark a virtual value not to be swappable (in use)
    void unpin(Value virtualValue) {
        LLVM_DEBUG({
            llvm::dbgs() << "Unpinning virtual value ";
            virtualValue.dump();
        });

        virtualObjects.getVirtualObject(virtualValue).unpin();

        LLVM_DEBUG({ physicalMemory.miniDump(); });
    }

    // returns whether the buffer's current contents are dead: no access at or
    // after the op being processed observes them, either because the next
    // access fully overwrites them or because no access remains
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

  public:
    VirtualObjects virtualObjects;
    PhysicalMemory physicalMemory;
    const torq_hl::MemorySpace memorySpace;
    const torq_hl::MemorySpace swapMemSpace;

    // ordered (op index, previous contents discarded) accesses per allocation
    DenseMap<Value, SmallVector<std::pair<unsigned, bool>>> bufferAccessSchedule;

    // index (into the processed op list) of the op currently being processed
    unsigned currentOpIndex = 0;

  private:
    int nextId = 0;
};

VirtualMemory &PhysicalObject::vm() { return virtualObject().vm(); }

bool PhysicalObject::isPinned() { return virtualObject().isPinned(); }

VirtualObject &PhysicalBuffer::virtualObject() { return virtualBuffer(); }

int PhysicalBuffer::size() { return virtualBuffer().size(); }

PhysicalObject &PhysicalAlias::parent() { return virtualAlias_.parent().physicalObject(); }

VirtualAlias &PhysicalAlias::virtualAlias() { return virtualAlias_; }

VirtualObject &PhysicalAlias::virtualObject() { return virtualAlias(); }

VirtualObject::VirtualObject(VirtualMemory &vm, Value value)
    : id_(vm.getNextId()), vm_(vm), value_(value) {}

void VirtualBuffer::pin() {
    if (pinCount() == 0) {
        vm().physicalMemory.pin(physicalBuffer());
    }

    VirtualObject::pin();
}

void VirtualBuffer::unpin() {
    VirtualObject::unpin();

    if (pinCount() == 0) {
        vm().physicalMemory.unpin(physicalBuffer());
    }
}

PhysicalObject *VirtualBuffer::swapIn(IRRewriter &rewriter, Location loc) {
    assert(isSwappedOut() && "Cannot swap in a non-swapped out buffer");

    LLVM_DEBUG({
        llvm::dbgs() << "Swapping in virtual buffer ";
        value().dump();
    });

    // create a buffer where to swap in the value
    auto physicalValueOp =
        memref::AllocOp::create(rewriter, loc, cast<MemRefType>(value().getType()));

    if (clAnnotateVirtualBufferIds) {
        physicalValueOp->setAttr(VIRTUAL_OBJECT_ID_ATTR_NAME, rewriter.getIndexArrayAttr({id()}));
    }

    if (swappedOutValue_) {
        // swap in the value by overwriting all the eventual alignment bytes in the newly
        // allocated physical buffer
        torq_hl::LoadOp::create(
            rewriter, loc, physicalValueOp, swappedOutValue_, SmallVector<int64_t>{},
            SmallVector<int64_t>{}, getEncodedTotalSizeBytes(physicalValueOp.getType()), true
        );
        vm().physicalMemory.noteSwapInLoad(size());
    }
    else {
        // the contents were dropped at swap out: there is nothing to load
        vm().physicalMemory.noteSkippedSwapInLoad();
    }

    maybePhysicalBuffer_ = vm().physicalMemory.add(*this, physicalValueOp);

    LLVM_DEBUG({
        llvm::dbgs() << "Swapped in to physical buffer ";
        physicalBuffer().value().dump();
    });

    return *maybePhysicalBuffer_;
}

void VirtualBuffer::swapOut(IRRewriter &rewriter, Location loc) {
    assert(!isSwappedOut() && "cannot swap out already swapped out buffer");
    assert(!isPinned() && "cannot swap out pinned buffer");

    LLVM_DEBUG({
        llvm::dbgs() << "Swapping out buffer ";
        value().dump();
    });

    // the contents are dead: free the retained swapped out buffer, skip the
    // store, and later swap back in without loading
    if (vm().contentsDead(value())) {

        LLVM_DEBUG(llvm::dbgs() << "Buffer contents are dead, dropping them\n");

        freeSwappedOutValue(rewriter, loc);
        vm().physicalMemory.noteDroppedSwapOut(size());
    }
    // the retained swapped out buffer is still up to date: skip the store
    else if (swappedOutValue_) {

        LLVM_DEBUG(llvm::dbgs() << "Buffer contents unchanged, skipping the store\n");

        vm().physicalMemory.noteSkippedSwapOutStore();
    }
    else {
        auto physicalValueType = cast<MemRefType>(value().getType());

        // allocate a memref where to swap out the value
        auto physicalValueEncoding = getEncoding(physicalValueType);
        auto swappedOutEncoding =
            cloneEncodingWithNewMemorySpace(physicalValueEncoding, vm().swapMemSpace);

        auto swappedOutType = createMemRefTypeWithEncoding(physicalValueType, swappedOutEncoding);
        auto swappedOutAllocOp = memref::AllocOp::create(rewriter, loc, swappedOutType);

        swappedOutValue_ = swappedOutAllocOp;

        if (clAnnotateVirtualBufferIds) {
            rewriter.modifyOpInPlace(swappedOutAllocOp, [&]() {
                swappedOutAllocOp->setAttr("torq-swap-out-buffer", rewriter.getUnitAttr());
            });

            swappedOutAllocOp->setAttr(
                VIRTUAL_OBJECT_ID_ATTR_NAME, rewriter.getIndexArrayAttr({id()})
            );
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Created swapped out buffer ";
            swappedOutAllocOp.dump();
        });

        // swap out the value by overwriting all the eventual alignment bytes in the newly
        // allocated swap out buffer
        torq_hl::StoreOp::create(
            rewriter, loc, swappedOutValue_, (*maybePhysicalBuffer_)->value(),
            SmallVector<int64_t>{}, SmallVector<int64_t>{},
            getEncodedTotalSizeBytes(swappedOutType), true
        );
        vm().physicalMemory.noteSwapOutStore(size());
    }

    // make sure all the aliases pointing to the physical buffer are invalidated
    for (auto &alias : aliases()) {
        alias->invalidate();
    }

    // deallocate the buffer we just swapped out
    memref::DeallocOp::create(rewriter, loc, (*maybePhysicalBuffer_)->value());
    vm().physicalMemory.remove(*this);
    maybePhysicalBuffer_ = std::nullopt;
}

void VirtualBuffer::initialize() {
    // initially we use the virtual value as the physical value of this virtual buffer
    maybePhysicalBuffer_ = vm().physicalMemory.add(*this, value());
}

// Collect the buffer accesses `op` may perform, normalized to AsyncAccess.
static void collectBufferAccesses(Operation *op, SmallVectorImpl<torq_hl::AsyncAccess> &accesses) {

    if (auto asyncAccessOp = dyn_cast<torq_hl::AsyncAccessOpInterface>(op)) {
        asyncAccessOp.getAsyncAccesses(accesses);
        return;
    }

    if (auto memEffectsOp = dyn_cast<MemoryEffectOpInterface>(op)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffectsOp.getEffects(effects);

        // a write effect not attached to a value could touch anything: fall back
        // to the conservative path below
        bool hasUnattributedWrite = llvm::any_of(effects, [](const auto &effect) {
            return isa<MemoryEffects::Write>(effect.getEffect()) && !effect.getValue();
        });

        if (!hasUnattributedWrite) {
            SetVector<Value> accessedValues;
            for (auto &effect : effects) {
                Value value = effect.getValue();
                if (value && isa<MemRefType>(value.getType())) {
                    accessedValues.insert(value);
                }
            }
            for (auto value : accessedValues) {
                auto access = torq_hl::getValueAccessFromEffects(effects, value);
                if (access != torq_hl::ArgAccessBitfield::None) {
                    accesses.push_back({access, cast<TypedValue<MemRefType>>(value)});
                }
            }
            return;
        }
    }

    // no usable effect information: assume the op may read and write any operand
    for (auto operand : op->getOperands()) {
        if (isa<MemRefType>(operand.getType())) {
            accesses.push_back(
                {torq_hl::ArgAccessBitfield::Read | torq_hl::ArgAccessBitfield::Write,
                 cast<TypedValue<MemRefType>>(operand)}
            );
        }
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

        SmallVector<torq_hl::AsyncAccess> accesses;
        collectBufferAccesses(op, accesses);

        // aggregate per root allocation: the previous contents are discarded
        // only if the whole allocation is accessed Write-without-Read and no
        // other access of the op observes it
        llvm::MapVector<Value, bool> rootDiscards;
        for (auto &access : accesses) {
            if (getEncodingMemorySpace(access.buffer.getType()) != memorySpace) {
                continue;
            }
            Value root = getViewBase(access.buffer);
            bool discards =
                access.access == torq_hl::ArgAccessBitfield::Write && access.buffer == root;
            auto [it, inserted] = rootDiscards.try_emplace(root, discards);
            if (!inserted) {
                it->second = it->second && discards;
            }
        }

        for (auto &[root, discards] : rootDiscards) {
            schedule[root].emplace_back(index, discards);
        }
    }
}

struct OperationMemoryInfo {
    SmallVector<OpOperand *> operands;
    SmallVector<OpResult> results;
    SmallVector<torq_hl::AsyncAccess> accesses;
    int resultsSize = 0;
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

    SmallVector<torq_hl::AsyncAccess> accesses;
    collectBufferAccesses(op, accesses);
    for (auto &access : accesses) {
        if (getEncodingMemorySpace(access.buffer.getType()) == memorySpace) {
            info.accesses.push_back(access);
        }
    }

    for (auto result : op->getResults()) {
        auto type = dyn_cast<MemRefType>(result.getType());
        if (type && getEncodingMemorySpace(type) == memorySpace) {
            info.results.push_back(result);
            info.resultsSize += getEncodedTotalSizeBytes(type);
        }
    }
    return info;
}

static bool processDeallocationOrAlias(
    Operation *op, VirtualMemory &vm, IRRewriter &rewriter, torq_hl::MemorySpace memorySpace
) {
    if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        if (getEncodingMemorySpace(deallocOp.getMemref().getType()) == memorySpace) {
            rewriter.setInsertionPoint(deallocOp);
            Value physicalValue = vm.deallocate(deallocOp.getMemref(), rewriter);
            if (physicalValue) {
                deallocOp.getMemrefMutable().set(physicalValue);
            }
            else {
                rewriter.eraseOp(deallocOp);
            }
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

// The swapped-out operands among `operands`; a set, since several operands may be the
// same value.
static SetVector<Value> collectOperandsToSwapIn(ArrayRef<OpOperand *> operands, VirtualMemory &vm) {
    SetVector<Value> toSwapIn;
    for (auto *operand : operands) {
        if (vm.isSwappedOut(operand->get())) {
            toSwapIn.insert(operand->get());
        }
    }
    return toSwapIn;
}

static void swapInOperandsAndAllocateResults(
    Operation *op, const OperationMemoryInfo &info, const SetVector<Value> &toSwapIn,
    VirtualMemory &vm, IRRewriter &rewriter
) {
    rewriter.setInsertionPoint(op);
    for (Value value : toSwapIn) {
        vm.swapIn(value, rewriter, op->getLoc());
    }
    for (auto result : info.results) {
        vm.addAllocation(result);
    }
}

// Pin `operands` and point them at their physical values.
static SmallVector<Value>
rewriteOperandsToPhysicalMemory(ArrayRef<OpOperand *> operands, VirtualMemory &vm) {
    SmallVector<Value> pinnedValues;
    for (auto *operand : operands) {
        Value virtualValue = operand->get();
        vm.pin(virtualValue);
        pinnedValues.push_back(virtualValue);
        operand->set(vm.getPhysicalValue(virtualValue));
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
                vm.unpin(value);
            }
        }
        return;
    }

    for (Value value : pinnedValues) {
        vm.unpin(value);
    }
    if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
        for (Value value : invocationToVirtual[waitOp.getInvocation()]) {
            vm.unpin(value);
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
        for (Value buffer : swap.buffers) {
            vm.swapOut(buffer, rewriter, op->getLoc());
        }
        if (swap.kind == Swap::Kind::Spill) {
            continue;
        }
        // relocated buffers swap back in only once all of them are out, so they land in
        // the fully freed space in the scheduled order
        for (Value buffer : swap.buffers) {
            vm.swapIn(buffer, rewriter, op->getLoc());
        }
    }
}

static void printStatistics(const VirtualMemory &vm) {
    if (!clPrintStatistics) {
        return;
    }
    llvm::dbgs() << "Total swap outs: " << vm.physicalMemory.swapOutCount() << "\n";
    llvm::dbgs() << "Total swap-out stores skipped (clean buffers): "
                 << vm.physicalMemory.skippedSwapOutStoreCount() << "\n";
    llvm::dbgs() << "Swap outs dropped (contents dead): " << vm.physicalMemory.droppedSwapOutCount()
                 << " (" << vm.physicalMemory.droppedSwapOutBytes() << " bytes)\n";
    llvm::dbgs() << "Swap-in loads skipped (dropped contents): "
                 << vm.physicalMemory.skippedSwapInLoadCount() << "\n";
    llvm::dbgs() << "Swap-out bytes stored: " << vm.physicalMemory.swapOutStoredBytes() << "\n";
    llvm::dbgs() << "Swap-in bytes loaded: " << vm.physicalMemory.swapInLoadedBytes() << "\n";
}

} // namespace

LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, int capacityBytes, torq_hl::MemorySpace memorySpace,
    const SwapSchedule &schedule
) {
    VirtualMemory vm(capacityBytes, memorySpace);
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

        SetVector<Value> operandsToSwapIn = collectOperandsToSwapIn(memoryInfo.operands, vm);
        swapInOperandsAndAllocateResults(op, memoryInfo, operandsToSwapIn, vm, rewriter);

        // Keep operands resident during execution and point the op at their physical values.
        SmallVector<Value> pinnedOperands =
            rewriteOperandsToPhysicalMemory(memoryInfo.operands, vm);

        // A write makes any retained swap-space copy stale.
        for (const torq_hl::AsyncAccess &access : memoryInfo.accesses) {
            if (bitEnumContainsAny(access.access, torq_hl::ArgAccessBitfield::Write)) {
                vm.freeSwappedOutValue(access.buffer, rewriter, op->getLoc());
            }
        }

        updateOperandPinLifetimes(op, pinnedOperands, memoryInfo.accesses, vm, invocationToVirtual);
    }

    printStatistics(vm);
    return success();
}

} // namespace mlir::syna::torq
