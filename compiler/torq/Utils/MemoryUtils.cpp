#include "MemoryUtils.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::syna {

int64_t getElementSizeBytes(ShapedType shapedType) {
    return getScalarSizeBytes(shapedType.getElementType());
}

int64_t getScalarSizeBytes(Type type) { return torq::div_ceil(type.getIntOrFloatBitWidth(), 8); }

size_t getShapeTypeDataSize(mlir::ShapedType type) {
    auto shape = type.getShape();
    if (shape.empty()) {
        return 0;
    }
    auto elementType = type.getElementType();
    auto elementSize = getScalarSizeBytes(elementType);
    auto numElements = shape[0];
    for (int i = 1; i < shape.size(); i++) {

        if (shape[i] == ShapedType::kDynamic) {
            llvm::report_fatal_error("Unsupported dynamic shape");
        }

        numElements *= shape[i];
    }
    return numElements * elementSize;
}

static int64_t getMemRefTypeOffsetBytes(MemRefType memRefType) {
    if (auto stridesAttr = mlir::dyn_cast_if_present<StridedLayoutAttr>(memRefType.getLayout())) {

        auto offset = stridesAttr.getOffset();

        if (offset == ShapedType::kDynamic) {
            llvm::report_fatal_error("Unsupported dynamic offsets");
        }

        if (offset != 0 && !hasDenseEncoding(memRefType)) {

            // we need to figure out where in memory the offset is pointing to

            auto encoding = getEncoding(memRefType);

            assert(
                memRefType.getRank() == encoding.getCounts().size() &&
                "rank changing subviews not supported"
            );

            // compute the natural strides of the original alloc using the counts on the encoding
            SmallVector<int64_t> naturalStrides(memRefType.getRank());
            naturalStrides[memRefType.getRank() - 1] = 1;
            for (int i = memRefType.getRank() - 2; i >= 0; i--) {
                naturalStrides[i] = naturalStrides[i + 1] * encoding.getCounts()[i + 1];
            }

            // find the coordinates in the backing buffer that the offset is pointing to
            SmallVector<int64_t> coords;
            coords.push_back(offset / naturalStrides[0]);

            for (int i = 1; i < memRefType.getRank(); i++) {
                coords.push_back((offset % naturalStrides[i - 1]) / naturalStrides[i]);
            }

            // compute the offset in the backing buffer memory using the strides of the encoding
            int realOffset = 0;
            for (int i = 0; i < memRefType.getRank(); i++) {
                realOffset += coords[i] * stridesAttr.getStrides()[i];
            }

            return realOffset * getElementSizeBytes(memRefType);
        }

        return offset * getElementSizeBytes(memRefType);
    }

    return 0;
}

static const std::string LRAM_ADDRESS_ATTR_NAME = "lram_address";
static const std::string DTCM_ADDRESS_ATTR_NAME = "dtcm_address";
static const std::string ITCM_ADDRESS_ATTR_NAME = "itcm_address";
static const std::string XRAM_ADDRESS_ATTR_NAME = "xram_address";

static void setOperationAddress(Operation *op, const std::string attrName, int64_t address) {
    OpBuilder builder(op->getContext());
    return op->setAttr(attrName, builder.getI64IntegerAttr(address));
}

static std::optional<int64_t>
getOperationAddress(Operation *op, const std::string attrName, int64_t offset) {

    if (!op) {
        // return no address if this is a function argument
        return std::nullopt;
    }

    if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {

        auto maybeAddress = waitOp.getResultAddresses();
        if (!maybeAddress) {
            return std::nullopt;
        }

        // return the address of the result
        return maybeAddress.value()[cast<OpResult>(op->getResult(0)).getResultNumber()] + offset;
    }
    else if (op->hasAttrOfType<IntegerAttr>(attrName)) {
        return op->getAttrOfType<IntegerAttr>(attrName).getInt() + offset;
    }

    return std::nullopt;
}

static std::optional<int64_t>
getValueAddress(Value value, const std::string attrName, int64_t offset = 0) {

    if (!value.getDefiningOp()) {
        return std::nullopt;
    }

    if (auto waitOp = value.getDefiningOp<torq_hl::WaitProgramOp>()) {

        auto maybeAddress = waitOp.getResultAddresses();
        if (!maybeAddress) {
            return std::nullopt;
        }

        return maybeAddress.value()[cast<OpResult>(value).getResultNumber()] + offset;
    }

    return getOperationAddress(value.getDefiningOp(), attrName, offset);
}

void setLramAddress(Operation *op, int64_t address) {
    setOperationAddress(op, LRAM_ADDRESS_ATTR_NAME, address);
}

std::optional<int64_t> getLramAddress(Value value, int64_t offset) {

    if (auto createInvocationOp = value.getDefiningOp<torq_hl::CreateInvocationOp>()) {

        // invocation have a lram address only if they are running on a slice
        if (createInvocationOp.getProgram().getType().getExecutor() != torq_hl::Executor::Slice) {
            return std::nullopt;
        }

        auto sectionAddresses = createInvocationOp.getExecutorCodeAddresses();
        auto sectionIndex = cast<OpResult>(value).getResultNumber() - 1;

        if (!sectionAddresses || sectionAddresses->size() <= sectionIndex) {
            return std::nullopt;
        }

        return (*sectionAddresses)[sectionIndex] + offset;
    }

    return getValueAddress(value, LRAM_ADDRESS_ATTR_NAME, offset);
}

void setXramAddress(Operation *op, int64_t address) {
    return setOperationAddress(op, XRAM_ADDRESS_ATTR_NAME, address);
}

std::optional<int64_t> getXramAddress(Value value, int64_t offset) {

    if (auto createInvocationOp = value.getDefiningOp<torq_hl::CreateInvocationOp>()) {
        auto sectionAddresses = createInvocationOp.getXramCodeAddresses();
        auto sectionIndex = cast<OpResult>(value).getResultNumber() - 1;

        if (!sectionAddresses || sectionAddresses->size() <= sectionIndex) {
            return std::nullopt;
        }

        return (*sectionAddresses)[sectionIndex] + offset;
    }

    return getValueAddress(value, XRAM_ADDRESS_ATTR_NAME, offset);
}

std::optional<int64_t> getXramAddress(Operation *op, int64_t offset) {
    return getOperationAddress(op, XRAM_ADDRESS_ATTR_NAME, offset);
}

void setItcmAddress(Operation *op, int64_t address) {
    setOperationAddress(op, ITCM_ADDRESS_ATTR_NAME, address);
}

std::optional<int64_t> getItcmAddress(Operation *op, int64_t offset) {
    return getOperationAddress(op, ITCM_ADDRESS_ATTR_NAME, offset);
}

std::optional<int64_t> getItcmAddress(Value value, int64_t offset) {
    return getValueAddress(value, ITCM_ADDRESS_ATTR_NAME, offset);
}

void setDtcmAddress(Operation *op, int64_t address) {
    setOperationAddress(op, DTCM_ADDRESS_ATTR_NAME, address);
}

std::optional<int64_t> getDtcmAddress(Operation *op, int64_t offset) {
    return getOperationAddress(op, DTCM_ADDRESS_ATTR_NAME, offset);
}

std::optional<int64_t> getDtcmAddress(Value value, int64_t offset) {
    return getValueAddress(value, DTCM_ADDRESS_ATTR_NAME, offset);
}

int64_t reserveXramArea(Operation *funcOp, int64_t size) {

    auto attrName = "torq-next-free-xram-address";

    // get the next free address in XRAM
    auto nextFreeXramAddressAttr = cast_if_present<IntegerAttr>(funcOp->getAttr(attrName));

    auto startAddress =
        nextFreeXramAddressAttr ? nextFreeXramAddressAttr.getValue().getZExtValue() : 0;

    // reserve the area
    funcOp->setAttr(
        attrName, IntegerAttr::get(IntegerType::get(funcOp->getContext(), 64), startAddress + size)
    );

    return startAddress;
}

LogicalResult setAddress(Operation *op, int64_t address) {

    if (!isa<memref::AllocOp>(op) && !isDerivedMemRefOperation(op)) {
        op->emitError("cannot have an address");
        return failure();
    }

    if (auto memrefType = dyn_cast<MemRefType>(op->getResult(0).getType())) {
        auto memSpace = getEncodingMemorySpace(memrefType);
        switch (memSpace) {
        case torq_hl::MemorySpace::Lram:
            setLramAddress(op, address);
            break;
        case torq_hl::MemorySpace::Dtcm:
            setDtcmAddress(op, address);
            break;
        case torq_hl::MemorySpace::Itcm:
            setItcmAddress(op, address);
            break;
        case torq_hl::MemorySpace::Xram:
            setXramAddress(op, address);
            break;
        default:
            return failure();
        }
    }

    return success();
}

LogicalResult setAddress(Value value, int64_t address) {

    if (isa<memref::AllocOp>(value.getDefiningOp())) {
        return setAddress(value.getDefiningOp(), address);
    }

    return failure();
}

static std::optional<int64_t>
getCssAddress(Value value, int64_t offset, TypedValue<torq_hl::InvocationType> invocation) {
    auto memrefType = dyn_cast<MemRefType>(value.getType());

    if (!memrefType) {
        return std::nullopt;
    }

    auto memSpace = getEncodingMemorySpace(memrefType);

    std::optional<int64_t> addr = getDataStartAddress(value, offset, invocation);

    int64_t baseAddress = 0;

    switch (memSpace) {
    case torq_hl::MemorySpace::Dtcm:
        baseAddress = mlir::syna::torq::HwInfo::css_dtcm_base_address;
        break;
    case torq_hl::MemorySpace::Itcm:
        baseAddress = mlir::syna::torq::HwInfo::css_itcm_base_address;
        break;
    default:
        return std::nullopt;
    }

    if (!addr) {
        return std::nullopt;
    }

    return baseAddress + addr.value();
}

std::optional<int64_t> getExecutorDataStartAddress(
    torq_hl::Executor executor, Value value, int64_t offset,
    TypedValue<torq_hl::InvocationType> invocation
) {

    auto type = cast<MemRefType>(value.getType());

    switch (executor) {
    case torq_hl::Executor::CSS:
        return getCssAddress(value, offset, invocation);

    case torq_hl::Executor::Host:
        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
            return std::nullopt;
        }
        return getDataStartAddress(value, offset, invocation);

    case torq_hl::Executor::Slice:
        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Lram) {
            return std::nullopt;
        }
        return getDataStartAddress(value, offset, invocation);

    case torq_hl::Executor::NSS:
        return getDataStartAddress(value, offset, invocation);

    default:
        llvm::report_fatal_error("unsupported executor");
    }
}

std::optional<int64_t>
getAddress(Value value, int64_t offset, TypedValue<torq_hl::InvocationType> invocation) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {

        auto type = cast<MemRefType>(value.getType());

        if (!invocation) {
            return std::nullopt; // no invocation provided
        }

        auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>();

        if (!createInvocationOp) {
            return std::nullopt; // not an invocation argument
        }

        if (blockArg.getOwner()->getParentOp() != createInvocationOp.getProgram().getDefiningOp()) {
            return std::nullopt; // this is an argument that is not an argument of the invoked
                                 // program
        }

        auto argAddresses = createInvocationOp.getExecutorArgsAddresses();

        if (!argAddresses) {
            return std::nullopt; // no addresses available
        }

        if (invocation.getType().getExecutor() == torq_hl::Executor::CSS) {
            // if the executor is CSS the addresses we get are remapped into a single address space
            // so we can't use them directly, this is not useful in practice anyways
            return std::nullopt;
        }

        // the address in getExecutorArgsAddresses is a start address, not a base address
        auto baseAddress =
            (*argAddresses)[blockArg.getArgNumber()] - getMemRefTypeOffsetBytes(type);

        return baseAddress + offset;
    }
    else {

        auto memRefType = dyn_cast<MemRefType>(value.getType());

        if (!memRefType) {
            return std::nullopt; // not a memref type
        }

        auto memorySpace = getEncodingMemorySpace(memRefType);

        switch (memorySpace) {
        case torq_hl::MemorySpace::Lram:
            return getLramAddress(value, offset);
        case torq_hl::MemorySpace::Dtcm:
            return getDtcmAddress(value, offset);
        case torq_hl::MemorySpace::Itcm:
            return getItcmAddress(value, offset);
        case torq_hl::MemorySpace::Xram:
            return getXramAddress(value, offset);
        default:
            return std::nullopt; // unsupported memory space
        }
    }
}

std::optional<int64_t>
getDataStartAddress(Value value, int64_t offset, TypedValue<torq_hl::InvocationType> invocation) {

    MemRefType type = cast<MemRefType>(value.getType());
    return getAddress(value, offset + getMemRefTypeOffsetBytes(type), invocation);
}

bool isDerivedMemRefOperation(Operation *op) {
    return isa<
        memref::SubViewOp, memref::ExpandShapeOp, memref::ReinterpretCastOp,
        memref::MemorySpaceCastOp, memref::CollapseShapeOp>(op);
}

OpOperand &getDerivedMemRefBase(Operation *op) {

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        return subviewOp.getSourceMutable();
    }
    else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
        return expandShapeOp.getSrcMutable();
    }
    else if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {
        return collapseShapeOp.getSrcMutable();
    }
    else if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(op)) {

        return reinterpretCast.getSourceMutable();
    }
    else if (auto memorySpaceCast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
        return memorySpaceCast.getSourceMutable();
    }
    else {
        llvm::report_fatal_error("not a derived memref operation");
    }
}

int getAlignmentByType(int bytes, mlir::Type type) {
    return bytes / std::ceil(1.0 * type.getIntOrFloatBitWidth() / 8);
}

} // namespace mlir::syna
