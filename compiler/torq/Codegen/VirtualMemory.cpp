#include <torq/Codegen/VirtualMemory.h>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
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

    virtual void touch() = 0;

    VirtualMemory &vm();

    bool isPinned();

    Value value() const { return value_; }

    virtual VirtualObject &virtualObject() = 0;

    PhysicalObject(Value value) : value_(value) {}

    virtual ~PhysicalObject() = default;
};

class PhysicalBuffer : public PhysicalObject {
    VirtualBuffer &virtualBuffer_;
    int address_;

  public:
    VirtualBuffer &virtualBuffer() { return virtualBuffer_; }

    int address() const { return address_; }

    virtual VirtualObject &virtualObject() override;

    int size();

    virtual void touch() override;

    PhysicalBuffer(Value value, VirtualBuffer &virtualBuffer, int address)
        : PhysicalObject{value}, virtualBuffer_(virtualBuffer), address_(address) {}
};

class PhysicalAlias : public PhysicalObject {

    VirtualAlias &virtualAlias_;

  public:
    PhysicalObject &parent();

    VirtualAlias &virtualAlias();

    virtual VirtualObject &virtualObject() override;

    virtual void touch() override;

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

    virtual FailureOr<PhysicalObject *>
    swapIn(IRRewriter &rewriter, Location loc, bool allowDefragment) = 0;

    void touch() { physicalObject().touch(); }

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
    Value swappedOutValue_;

  public:
    virtual VirtualBuffer &root() override { return *this; }

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

    virtual FailureOr<PhysicalObject *>
    swapIn(IRRewriter &rewriter, Location loc, bool allowDefragment) override;

    void swapOut(IRRewriter &rewriter, Location loc);

    LogicalResult initialize();

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

    virtual FailureOr<PhysicalObject *>
    swapIn(IRRewriter &rewriter, Location loc, bool allowDefragment) override {

        LLVM_DEBUG({
            llvm::dbgs() << "Swapping in virtual alias ";
            value().dump();
        });

        if (parent().isSwappedOut()) {
            if (failed(parent().swapIn(rewriter, loc, allowDefragment))) {
                return failure();
            }
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

    VirtualMemory &vm_;
    DenseMap<VirtualBuffer *, std::unique_ptr<PhysicalBuffer>> physicalBuffers_;
    SetVector<PhysicalBuffer *> lastUsedPhysicalBuffer_;
    DenseMap<PhysicalBuffer *, int> pinnedBuffers_;
    int totalPinnedSize_ = 0;
    Pool &pool_;
    int defragCount_ = 0;
    int swapOutCount_ = 0;

  public:
    PhysicalMemory(VirtualMemory &vm, Pool &pool) : vm_(vm), pool_(pool) {}

    int totalPinnedSize() const { return totalPinnedSize_; }

    int totalPhysicalBufferSize() const { return pool_.usedSize(); }

    int defragCount() const { return defragCount_; }

    int swapOutCount() const { return swapOutCount_; }

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

    LogicalResult defragment(IRRewriter &rewriter, Location loc) {

        LLVM_DEBUG({
            llvm::dbgs() << "Defragment memory\n";
            llvm::dbgs() << "Memory before defragmentation:\n";
            dump();
        });

        // swap out all the non pinned active values to XRAM
        SmallVector<VirtualBuffer *> toMove;
        for (auto &[value, buf] : physicalBuffers_) {

            if (pinnedBuffers_.contains(buf.get())) {
                continue;
            }

            toMove.push_back(&(buf->virtualBuffer()));
        }

        // swap out all the non pinned active values to XRAM
        for (auto vBuf : toMove) {
            vBuf->swapOut(rewriter, loc);
        }

        // swap in all the values, this should make things more compact
        // (if no buffers are currently pinned it will remove all spaces)
        for (auto vBuf : toMove) {
            if (failed(vBuf->swapIn(rewriter, loc, false))) {
                return failure();
            };
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Defragmentation done\n";
            llvm::dbgs() << "Memory after defragmentation:\n";
            dump();
        });

        defragCount_++;

        return success();
    }

    FailureOr<PhysicalBuffer *> add(VirtualBuffer &obj, Value value, bool allowDefragment) {
        assert(!physicalBuffers_.contains(&obj) && "Physical buffer already exists for this value");

        assert(
            obj.size() + totalPhysicalBufferSize() <= pool_.usableSize() &&
            "Not enough memory to add physical buffer"
        );

        auto maybeAddr = pool_.allocate(value);

        // cannot allocate address, we know there is enough space but it is maybe fragmented
        if (failed(maybeAddr)) {

            if (!allowDefragment) {
                LLVM_DEBUG({
                    llvm::dbgs() << "Allocation failed and cannot defragment, giving up\n";
                });

                dump();
                return failure();
            }

            LLVM_DEBUG({ llvm::dbgs() << "Allocation failed, trying to defragment\n"; });

            IRRewriter rewriter(value.getContext());
            rewriter.setInsertionPoint(value.getDefiningOp());
            if (failed(defragment(rewriter, value.getLoc()))) {
                return failure();
            }

            maybeAddr = pool_.allocate(value);

            // we still fail to find an address, the pinned buffers prevent us from successfully
            // defragment
            if (failed(maybeAddr)) {
                LLVM_DEBUG({
                    llvm::dbgs() << "Allocation failed after defragmentation, need to give up\n";
                });
                dump();
                return failure();
            }
        }

        auto [it, inserted] = physicalBuffers_.try_emplace(
            &obj, std::make_unique<PhysicalBuffer>(value, obj, *maybeAddr)
        );
        auto &physicalBuffer = *it->second;
        lastUsedPhysicalBuffer_.insert(&physicalBuffer);

        return &physicalBuffer;
    }

    void remove(VirtualBuffer &obj) {
        assert(physicalBuffers_.contains(&obj) && "Physical buffer not found");
        auto &physicalBuffer = *physicalBuffers_[&obj];
        lastUsedPhysicalBuffer_.remove(&physicalBuffer);
        pool_.free(physicalBuffer.value());
        physicalBuffers_.erase(&obj);
    }

    void touch(PhysicalBuffer &object) {
        lastUsedPhysicalBuffer_.remove(&object);
        lastUsedPhysicalBuffer_.insert(&object);
    }

    LogicalResult freeSpace(int space, IRRewriter &rewriter, Location loc) {

        int nextUnpinnedValue = 0;

        LLVM_DEBUG({
            llvm::dbgs() << "Freeing space, need " << space
                         << " current usage: " << totalPhysicalBufferSize() << "/"
                         << pool_.usableSize() << "( free "
                         << (pool_.usableSize() - totalPhysicalBufferSize()) << ")\n";
        });

        while (totalPhysicalBufferSize() + space > pool_.usableSize()) {

            if (nextUnpinnedValue >= lastUsedPhysicalBuffer_.size()) {
                return failure();
            }

            auto physicalBuffer = lastUsedPhysicalBuffer_[nextUnpinnedValue];

            // skip this value because it is pinned
            if (pinnedBuffers_.contains(physicalBuffer)) {

                LLVM_DEBUG({
                    llvm::dbgs() << "Skipping pinned buffer ";
                    physicalBuffer->virtualObject().value().dump();
                });

                nextUnpinnedValue++;
                continue;
            }

            LLVM_DEBUG({
                llvm::dbgs() << "Swapping out buffer ";
                physicalBuffer->virtualObject().value().dump();
            });

            physicalBuffer->virtualBuffer().swapOut(rewriter, loc);

            swapOutCount_++;
        }

        return success();
    }

    void dump() {

        llvm::dbgs() << "Current memory usage " << totalPhysicalBufferSize() << "\n";

        llvm::dbgs() << "LRU:\n";
        for (auto buf : lastUsedPhysicalBuffer_) {
            llvm::dbgs() << "   - ";
            buf->virtualObject().value().dump();
        }

        llvm::dbgs() << "Active virtual allocations:\n";
        for (auto &[value, buf] : physicalBuffers_) {
            auto virtualValue = buf->virtualObject().value();
            llvm::dbgs() << "   - size " << buf->size() << " ";
            llvm::dbgs() << "address " << buf->address() << " ";
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
    DenseMap<Value, std::unique_ptr<VirtualBuffer>> virtualBuffers_;
    DenseMap<Value, VirtualObject *> virtualObjects_;
    VirtualMemory &vm_;

  public:
    VirtualObjects(VirtualMemory &vm) : vm_(vm) {}

    LogicalResult addBuffer(Value value) {
        assert(!virtualBuffers_.contains(value) && "Virtual buffer already exists for this value");

        virtualBuffers_.try_emplace(value, std::make_unique<VirtualBuffer>(value, vm_));

        auto &virtualBuffer = virtualBuffers_[value];
        if (failed(virtualBuffer->initialize())) {
            virtualBuffers_.erase(value);
            return failure();
        }

        virtualObjects_.try_emplace(value, virtualBuffers_[value].get());

        return success();
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
// pinned. This prevents them from being swapped out during memory compaction (freeing
// some space to allow the next allocation to happen) and physical memory defragmentation

class VirtualMemory {

  public:
    VirtualMemory(Pool &pool, torq_hl::MemorySpace memorySpace)
        : virtualObjects(*this), physicalMemory(*this, pool), memorySpace(memorySpace),
          swapMemSpace(
              memorySpace == torq_hl::MemorySpace::Lram ? torq_hl::MemorySpace::Xram
                                                        : torq_hl::MemorySpace::Lram
          ) {}

    int getNextId() { return nextId++; }

    // deallocates the virtual buffer, this doesn't actually introduce the deallocate
    // operation in the IR but just updates the internal state of the virtual memory system
    Value deallocate(Value virtualValue) {

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

    LogicalResult addAllocation(Value virtualValue) {

        LLVM_DEBUG({
            llvm::dbgs() << "Allocate for virtual value ";
            virtualValue.dump();
        });

        auto ret = virtualObjects.addBuffer(virtualValue);

        if (failed(ret)) {
            return failure();
        }

        LLVM_DEBUG({ physicalMemory.miniDump(); });

        return success();
    }

    // updates the last used timestamp of the virtual value
    void touch(Value virtualValue) {

        LLVM_DEBUG({
            llvm::dbgs() << "Touch virtual value ";
            virtualValue.dump();
        });

        virtualObjects.getVirtualObject(virtualValue).touch();

        LLVM_DEBUG({ physicalMemory.miniDump(); });
    }

    // inserts IR to copy the swapped out virtual buffer back to the the memory space into
    // a new physical buffer and updates the virtual memory system internal state
    FailureOr<Value> swapIn(Value virtualValue, IRRewriter &rewriter, Location loc) {

        LLVM_DEBUG({
            llvm::dbgs() << "Swap-in virtual value ";
            virtualValue.dump();
        });

        auto &virtualObject = virtualObjects.getVirtualObject(virtualValue);

        auto ret = virtualObject.swapIn(rewriter, loc, true);

        if (failed(ret)) {
            return failure();
        }

        LLVM_DEBUG({ physicalMemory.miniDump(); });

        return virtualObject.physicalObject().value();
    }

    // swaps out physical buffers from the memory space until the given amount of space is available
    // physical buffers are swapped out in order of usage (last used is swapped out last).
    // the function doesn't swap out pinned buffers and fails if it is not possible to free the
    // requested space
    LogicalResult freeSpace(int space, IRRewriter &rewriter, Location loc) {

        LLVM_DEBUG({ llvm::dbgs() << "Freeing " << space << " bytes\n"; });

        if (failed(physicalMemory.freeSpace(space, rewriter, loc))) {
            return failure();
        }

        LLVM_DEBUG({ physicalMemory.miniDump(); });

        return success();
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

  public:
    VirtualObjects virtualObjects;
    PhysicalMemory physicalMemory;
    const torq_hl::MemorySpace memorySpace;
    const torq_hl::MemorySpace swapMemSpace;

  private:
    int nextId = 0;
};

VirtualMemory &PhysicalObject::vm() { return virtualObject().vm(); }

bool PhysicalObject::isPinned() { return virtualObject().isPinned(); }

VirtualObject &PhysicalBuffer::virtualObject() { return virtualBuffer(); }

int PhysicalBuffer::size() { return virtualBuffer().size(); }

void PhysicalBuffer::touch() { vm().physicalMemory.touch(*this); }

PhysicalObject &PhysicalAlias::parent() { return virtualAlias_.parent().physicalObject(); }

void PhysicalAlias::touch() { parent().touch(); }

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

FailureOr<PhysicalObject *>
VirtualBuffer::swapIn(IRRewriter &rewriter, Location loc, bool allowDefragment) {
    assert(isSwappedOut() && "Cannot swap in a non-swapped out buffer");

    LLVM_DEBUG({
        llvm::dbgs() << "Swapping in virtual buffer ";
        value().dump();
    });

    // create a buffer where to swap in the value
    auto physicalValueOp = rewriter.create<memref::AllocOp>(loc, value().getType(), ValueRange{});

    if (clAnnotateVirtualBufferIds) {
        physicalValueOp->setAttr(VIRTUAL_OBJECT_ID_ATTR_NAME, rewriter.getIndexArrayAttr({id()}));
    }

    // swap in the value by overwriting all the eventual alignment bytes in the newly allocated
    // physical buffer
    rewriter.create<torq_hl::LoadOp>(
        loc, physicalValueOp, swappedOutValue_, SmallVector<int64_t>{}, SmallVector<int64_t>{},
        getEncodedTotalSizeBytes(physicalValueOp.getType()), true
    );

    // deallocate the buffer from which we just swapped in the data
    rewriter.create<memref::DeallocOp>(loc, swappedOutValue_);

    // drop reference to the swapped out value
    swappedOutValue_ = nullptr;

    // create the physical buffer
    auto ret = vm().physicalMemory.add(*this, physicalValueOp, allowDefragment);

    if (failed(ret)) {
        return failure();
    }

    maybePhysicalBuffer_ = ret.value();

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

    auto physicalValueType = cast<MemRefType>(value().getType());

    // allocate a memref where to swap out the value
    auto physicalValueEncoding = getEncoding(physicalValueType);
    auto swappedOutEncoding =
        cloneEncodingWithNewMemorySpace(physicalValueEncoding, vm().swapMemSpace);

    auto swappedOutType = createMemRefTypeWithEncoding(physicalValueType, swappedOutEncoding);
    auto swappedOutAllocOp = rewriter.create<memref::AllocOp>(loc, swappedOutType, ValueRange{});

    swappedOutValue_ = swappedOutAllocOp;

    if (clAnnotateVirtualBufferIds) {
        rewriter.modifyOpInPlace(swappedOutAllocOp, [&]() {
            swappedOutAllocOp->setAttr("torq-swap-out-buffer", rewriter.getUnitAttr());
        });

        swappedOutAllocOp->setAttr(VIRTUAL_OBJECT_ID_ATTR_NAME, rewriter.getIndexArrayAttr({id()}));
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Created swapped out buffer ";
        swappedOutAllocOp.dump();
    });

    // swap out the value by overwriting all the eventual alignment bytes in the newly allocated
    // swap out buffer
    rewriter.create<torq_hl::StoreOp>(
        loc, swappedOutValue_, (*maybePhysicalBuffer_)->value(), SmallVector<int64_t>{},
        SmallVector<int64_t>{}, getEncodedTotalSizeBytes(swappedOutType), true
    );

    // make sure all the aliases pointing to the physical buffer are invalidated
    for (auto &alias : aliases()) {
        alias->invalidate();
    }

    // deallocate the buffer we just swapped out
    rewriter.create<memref::DeallocOp>(loc, (*maybePhysicalBuffer_)->value());
    vm().physicalMemory.remove(*this);
    maybePhysicalBuffer_ = std::nullopt;
}

LogicalResult VirtualBuffer::initialize() {

    // initially we use the virtual value as the physical value of this virtual buffer
    auto ret = vm().physicalMemory.add(*this, value(), true);

    if (failed(ret)) {
        return failure();
    }

    maybePhysicalBuffer_ = ret.value();

    return success();
}

} // namespace

// go over the full function and replace virtual values with physical values
// ensuring that when too many physical values are active, the least recently
// used are swapped out. Perform defragmentation of active values if necessary.
LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, Pool &pool, torq_hl::MemorySpace memorySpace
) {

    VirtualMemory vm(pool, memorySpace);

    IRRewriter rewriter(funcOp);

    // find all the operations we need to process
    SmallVector<Operation *> ops;
    for (auto &op : funcOp.getFunctionBody().getOps()) {
        ops.push_back(&op);
    }

    DenseMap<Value, SmallVector<Value>> invocationToVirtual;

    // process every operation that we found to map any memref operand or result from virtual
    // to physical value. Use pinning and swap-in to make sure all operands are present before
    // the operation and there is enough space to allocate the result
    for (auto op : ops) {

        LLVM_DEBUG({
            llvm::dbgs() << "------------\n";
            llvm::dbgs() << "Processing operation: ";
            op->dump();
        });

        // special case for memory deallocations
        if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {

            if (getEncodingMemorySpace(deallocOp.getMemref().getType()) != memorySpace) {
                continue;
            }

            // update the dealloc to deallocate the current value (it may be swapped out or not)
            deallocOp.getMemrefMutable().set(vm.deallocate(deallocOp.getMemref()));

            continue;
        }

        // special case for operation that create an alias of an allocation
        if (isDerivedMemRefOperation(op)) {

            auto &baseMemRef = getDerivedMemRefBase(op);

            if (getEncodingMemorySpace(cast<MemRefType>(baseMemRef.get().getType())) !=
                memorySpace) {
                continue;
            }

            vm.addAlias(op->getResult(0));

            // we don't need to swap in the contents not touch the buffer since
            // these operations have no side effects and we may need to use this value much later

            continue;
        }

        // find all the memref operands needed for the operation
        SmallVector<OpOperand *> memrefOperands;
        for (auto &opOperand : op->getOpOperands()) {
            auto operand = opOperand.get();
            auto resultType = dyn_cast<MemRefType>(operand.getType());
            if (!resultType || getEncodingMemorySpace(resultType) != memorySpace) {
                continue;
            }
            memrefOperands.push_back(&opOperand);
        }

        // find all the memref outputs of the operation
        SmallVector<OpResult> memrefResults;
        int resultsSize = 0;
        for (auto result : op->getResults()) {
            auto resultType = dyn_cast<MemRefType>(result.getType());
            if (!resultType || getEncodingMemorySpace(resultType) != memorySpace) {
                continue;
            }
            memrefResults.push_back(result);
            resultsSize += getEncodedTotalSizeBytes(resultType);
        }

        // find operands that need to be swapped in and compute how much space we need to do so
        DenseSet<Value>
            toSwapIn; // use a set because multiple opOperands may point to the same value
        SmallVector<Value> pinnedValues;
        for (auto &opOperand : memrefOperands) {
            auto operand = opOperand->get();
            // if currently swapped out, add it to the list
            if (vm.isSwappedOut(operand)) {
                toSwapIn.insert(opOperand->get());
            }
            else {
                // pin the value to prevent it being swapped out when freeing space for the values
                // we need to swap in
                vm.pin(operand);
                pinnedValues.push_back(operand);
            }
        }

        // compute the total size of the operands we need to swap in
        int swapInSize = 0;
        for (auto v : toSwapIn) {
            auto memRefType = cast<MemRefType>(v.getType());
            swapInSize += getEncodedTotalSizeBytes(memRefType);
        }

        rewriter.setInsertionPoint(op);

        // make room to swap in operands and allocate results
        if (failed(vm.freeSpace(swapInSize + resultsSize, rewriter, op->getLoc()))) {

            LLVM_DEBUG({
                op->dump();
                llvm::dbgs() << "Operands:\n";
                for (auto &opOperand : memrefOperands) {
                    llvm::dbgs(
                    ) << "  - size "
                      << getEncodedTotalSizeBytes(cast<MemRefType>(opOperand->get().getType()))
                      << " bytes";
                    llvm::dbgs() << " to swap in " << vm.isSwappedOut(opOperand->get()) << " ";
                    opOperand->get().dump();
                }
                llvm::dbgs() << "Results sizes:\n";
                for (auto result : memrefResults) {
                    llvm::dbgs() << "  - size "
                                 << getEncodedTotalSizeBytes(cast<MemRefType>(result.getType()))
                                 << " bytes";
                    result.dump();
                }
            });

            return op->emitError("unable to free enough space for results and operands");
        }

        // unpin all the pinned virtual allocations to allow defragmentation
        for (auto pinnedValue : pinnedValues) {
            vm.unpin(pinnedValue);
        }

        // swap in all the operands that are currently swapped out (this will cause defragmentation
        // if necessary)
        for (auto virtualValue : toSwapIn) {
            auto maybePhysicalValue = vm.swapIn(virtualValue, rewriter, op->getLoc());

            if (failed(maybePhysicalValue)) {
                llvm::dbgs() << "Failed to swap in operand for operation: ";
                virtualValue.dump();
                return op->emitError("unable to swap in operand");
            }
        }

        // allocate all the results (this will cause defragmentation if necessary)
        for (auto result : memrefResults) {
            if (failed(vm.addAllocation(result))) {
                llvm::dbgs() << "Failed to allocate result for operation: ";
                result.dump();
                return op->emitError(
                    "unable to allocate space for result # " +
                    std::to_string(result.getResultNumber())
                );
            }
        }

        // pin all the memref operands and replace virtual with physical values
        pinnedValues.clear();
        for (auto memRefOperand : memrefOperands) {
            vm.pin(memRefOperand->get());
            pinnedValues.push_back(memRefOperand->get());
            memRefOperand->set(vm.getPhysicalValue(memRefOperand->get()));
        }

        // special case for start op: save the list of pinned virtual values
        // so that we don't touch them till the corresponding wait is executed
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {

            invocationToVirtual[startOp.getInvocation()] = pinnedValues;
        }
        else {

            // unpin all the operations we have pinned
            for (auto pinnedValue : pinnedValues) {
                vm.unpin(pinnedValue);
            }

            // special case for wait op: unpin all the arguments that were pinned for the start
            // operation
            if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
                for (auto arg : invocationToVirtual.at(waitOp.getInvocation())) {
                    vm.unpin(arg);
                }

                invocationToVirtual.erase(waitOp.getInvocation());
            }
        }
    }

    if (clPrintStatistics) {
        llvm::dbgs() << "Total defragmentations: " << vm.physicalMemory.defragCount() << "\n";
        llvm::dbgs() << "Total swap outs: " << vm.physicalMemory.swapOutCount() << "\n";
    }

    return success();
}

} // namespace mlir::syna::torq