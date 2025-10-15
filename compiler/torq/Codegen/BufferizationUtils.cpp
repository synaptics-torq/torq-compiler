#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-torq-bufferization-utils"

namespace mlir::syna::torq {

static LogicalResult computeStrides(
    MemRefType fromType, MemRefType toType, SmallVector<int64_t> &fromStridesBytes,
    SmallVector<int64_t> &toStridesBytes, SmallVector<int64_t> &shape,
    int64_t &contiguousElementSizeBytes
) {

    if (fromType.getRank() != toType.getRank()) {
        return failure();
    }

    if (fromType.getElementType() != toType.getElementType()) {
        return failure();
    }

    auto fromStrides = getEncodedStridesElements(fromType);
    auto toStrides = getEncodedStridesElements(toType);

    // find the contiguous part of both input and output memrefs
    int contiguousSizeElements = 1;
    int lastContiguousDim = fromType.getRank();

    for (int i = fromType.getRank() - 1; i >= 0; --i) {

        // if input and output strides differ we need to
        // stop because we can't read/write contiguously anymore
        if (fromStrides[i] != toStrides[i]) {
            break;
        }

        // if the stride is not natural we need to stop
        // because we can't read/write contiguously anymore
        if (fromStrides[i] != contiguousSizeElements) {
            break;
        }

        contiguousSizeElements *= fromType.getShape()[i];
        lastContiguousDim = i;
    }

    // create the input/outputs strides and shape (these may be 0-length vectors
    // if the tensors are contiguous)
    int elementSize = (fromType.getElementType().getIntOrFloatBitWidth() + 7) / 8;
    for (int i = 0; i < lastContiguousDim; i++) {

        // FIXME: we can further optimize this by merging contiguos dimension
        // before lastContiguousDim

        // skip 1-dimensions as they don't affect the copy
        if (fromType.getShape()[i] == 1) {
            continue;
        }

        fromStridesBytes.push_back(fromStrides[i] * elementSize);
        toStridesBytes.push_back(toStrides[i] * elementSize);
        shape.push_back(fromType.getShape()[i]);
    }

    contiguousElementSizeBytes = contiguousSizeElements * elementSize;

    return success();
}

LogicalResult createHostCopyOp(OpBuilder &builder, Location loc, Value from, Value to) {

    auto fromType = cast<MemRefType>(from.getType());
    auto toType = cast<MemRefType>(to.getType());

    SmallVector<int64_t> fromStridesBytes, toStridesBytes, shape;
    int64_t contiguousElementsSizeBytes;

    if (failed(computeStrides(
            fromType, toType, fromStridesBytes, toStridesBytes, shape, contiguousElementsSizeBytes
        ))) {
        return failure();
    }

    // Create a host copy operation
    builder.create<torq_hl::HostCopyOp>(
        loc, to, from,
        /*inputStridesBytes=*/fromStridesBytes,
        /*outputStridesBytes=*/toStridesBytes,
        /*shape=*/shape,
        /*elementSizeBytes=*/contiguousElementsSizeBytes
    );

    return success();
}

static LogicalResult createStoreOp(OpBuilder &builder, Location loc, Value from, Value to) {

    auto fromType = mlir::cast<MemRefType>(from.getType());
    auto toType = mlir::cast<MemRefType>(to.getType());

    // if the source is not contiguous we need first convert to a contiguous buffer
    // and the store that buffer into the destination
    if (!isDenseInMemory(fromType)) {
        // create a new empty alloc with contiguous layout
        auto fromContiguousEncoding = createDenseEncoding(fromType, torq_hl::MemorySpace::Lram);
        auto contiguousFromType = MemRefType::get(
            fromType.getShape(), fromType.getElementType(), nullptr, fromContiguousEncoding
        );
        auto contiguousFrom =
            builder.create<memref::AllocOp>(loc, contiguousFromType, ValueRange{});

        // convert the non-contiguous input data into the contiguous buffer
        builder.create<torq_hl::ConvertOp>(
            loc, TypeRange{}, contiguousFrom, from,
            /*requirements=*/nullptr, /*encoding=*/fromContiguousEncoding
        );

        from = contiguousFrom;
        fromType = contiguousFromType;
    }

    // compute the strides required to perform the copy
    SmallVector<int64_t> fromStridesBytes, toStridesBytes, shape;
    int64_t contiguousElementsSizeBytes;

    if (failed(computeStrides(
            fromType, toType, fromStridesBytes, toStridesBytes, shape, contiguousElementsSizeBytes
        ))) {
        return failure();
    }

    // if the output has too many strides we need to first copy the data to a
    // contiguous buffer and then do an host copy into the destination
    if (toStridesBytes.size() > 4) {

        // create a new empty alloc with contiguous layout
        auto toContiguousEncoding = createDenseEncoding(toType, torq_hl::MemorySpace::Xram);
        auto contiguousToType = MemRefType::get(
            fromType.getShape(), fromType.getElementType(), nullptr, toContiguousEncoding
        );
        auto contiguousTo = builder.create<memref::AllocOp>(loc, contiguousToType, ValueRange{});

        // store the data to the contiguous buffer
        if (failed(createStoreOp(builder, loc, from, contiguousTo))) {
            return failure();
        }

        // do an host copy of the contiguous buffer into the destination
        if (failed(createHostCopyOp(builder, loc, contiguousTo, to))) {
            return failure();
        }

        return success();
    }
    else {
        // store the contiguous buffer into the destination
        builder.create<torq_hl::StoreOp>(
            loc, to, from,
            /*outputStridesBytes=*/builder.getDenseI64ArrayAttr(toStridesBytes),
            /*shape=*/builder.getDenseI64ArrayAttr(shape),
            /*elementSizeBytes=*/builder.getI64IntegerAttr(contiguousElementsSizeBytes),
            /*unsafe=*/builder.getBoolAttr(false)
        );
    }

    return success();
}

static LogicalResult createLoadOp(OpBuilder &builder, Location loc, Value from, Value to) {

    auto fromType = mlir::cast<MemRefType>(from.getType());
    auto toType = mlir::cast<MemRefType>(to.getType());

    auto contiguousTo = to;
    auto contiguousToType = toType;

    // if the destination is not dense we need to first load into a non-contiguous buffer
    if (!isDenseInMemory(toType)) {

        // create a new empty alloc with contiguous layout
        auto toContiguousEncoding = createDenseEncoding(toType, torq_hl::MemorySpace::Lram);

        contiguousToType = MemRefType::get(
            fromType.getShape(), fromType.getElementType(), nullptr, toContiguousEncoding
        );

        contiguousTo = builder.create<memref::AllocOp>(loc, contiguousToType, ValueRange{});
    }

    // compute the strides required to perform the load
    SmallVector<int64_t> fromStridesBytes, toStridesBytes, shape;
    int64_t contiguousElementsSizeBytes;

    if (failed(computeStrides(
            fromType, contiguousToType, fromStridesBytes, toStridesBytes, shape,
            contiguousElementsSizeBytes
        ))) {
        return failure();
    }

    // if the source has too many strides we need to first remove them with an host copy
    if (fromStridesBytes.size() > 4) {

        // create a new empty alloc with contiguous layout
        auto fromContiguousEncoding = createDenseEncoding(fromType, torq_hl::MemorySpace::Xram);

        auto contiguousFromType = MemRefType::get(
            fromType.getShape(), fromType.getElementType(), nullptr, fromContiguousEncoding
        );
        auto contiguousFrom =
            builder.create<memref::AllocOp>(loc, contiguousFromType, ValueRange{});

        // do an host copy of source data into the contiguous XRAM buffer
        if (failed(createHostCopyOp(builder, loc, contiguousFrom, from))) {
            return failure();
        }

        // load the data from the contiguous buffer in XRAM into the contiguous destination LRAM
        if (failed(createLoadOp(builder, loc, contiguousFrom, contiguousTo))) {
            return failure();
        }
    }
    else {

        // perform the copy
        builder.create<torq_hl::LoadOp>(
            loc, contiguousTo, from,
            /*inputStridesBytes=*/builder.getDenseI64ArrayAttr(fromStridesBytes),
            /*shape=*/builder.getDenseI64ArrayAttr(shape),
            /*elementSizeBytes=*/builder.getI64IntegerAttr(contiguousElementsSizeBytes),
            /*unsafe=*/builder.getBoolAttr(false)
        );
    }

    // convert the contiguous input data into the non-contiguous buffer if necessary
    if (contiguousTo != to) {

        builder.create<torq_hl::ConvertOp>(
            loc, TypeRange{}, to, contiguousTo,
            /*requirements=*/nullptr, /* encoding=*/getEncoding(toType)
        );
    }

    return success();
}

LogicalResult createLramToLramCopy(OpBuilder &builder, Location loc, Value from, Value to) {
    builder.create<torq_hl::ConvertOp>(
        loc, TypeRange{}, to, from,
        /* requirements = */ nullptr, /* encoding= */ getEncoding(cast<ShapedType>(to.getType()))
    );
    return success();
}

LogicalResult createTorqCopy(OpBuilder &builder, Location loc, Value from, Value to) {

    auto fromType = mlir::cast<MemRefType>(from.getType());
    auto toType = mlir::cast<MemRefType>(to.getType());

    bool fromLram = getEncodingMemorySpace(fromType) == torq_hl::MemorySpace::Lram;
    bool toLram = getEncodingMemorySpace(toType) == torq_hl::MemorySpace::Lram;
    bool fromXram = getEncodingMemorySpace(fromType) == torq_hl::MemorySpace::Xram;
    bool toXram = getEncodingMemorySpace(toType) == torq_hl::MemorySpace::Xram;

    if (fromLram && toXram) {
        return createStoreOp(builder, loc, from, to);
    }
    else if (fromXram && toLram) {
        return createLoadOp(builder, loc, from, to);
    }
    else if (fromXram && toXram) {
        // FIXME: we may want to lower this to a NPU based copy later but we will
        // need to tile it correctly to fit every chunk in LRAM
        return createHostCopyOp(builder, loc, from, to);
    }
    else if (fromLram && toLram) {
        return createLramToLramCopy(builder, loc, from, to);
    }
    else {
        builder.create<memref::CopyOp>(loc, from, to);
        return success();
    }
}

Attribute getTorqMemSpaceAttr(TensorType t) {
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(t)) {
        Attribute enc = rankedTensorType.getEncoding();
        return enc;
    }
    return Attribute();
}

FailureOr<Value> createTorqAllocation(
    OpBuilder &builder, Location loc, MemRefType memRef, ValueRange dynamicSizes, unsigned alignment
) {

    if (!hasDenseEncoding(memRef) && !memRef.getLayout().isIdentity()) {
        llvm::report_fatal_error("Unsupported non-dense encoding");
    }

    return builder.create<memref::AllocOp>(loc, memRef, /*size=*/nullptr).getResult();
}

} // namespace mlir::syna::torq
