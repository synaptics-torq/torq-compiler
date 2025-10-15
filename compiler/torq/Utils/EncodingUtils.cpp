#include "EncodingUtils.h"
#include "MemoryUtils.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::syna {

// return the strides for a shape with the given encoding alignment
static SmallVector<int64_t>
getAlignedStrides(const ArrayRef<int64_t> &shape, const ArrayRef<int64_t> &align) {
    int rank = shape.size();

    if (rank == 0) {
        return {};
    }

    assert(align.size() == rank);

    SmallVector<int64_t> strides(rank);

    strides[rank - 1] = torq::align_ceil(1, std::abs(align[rank - 1] ? align[rank - 1] : 1));

    for (int i = rank - 2; i >= 0; i--) {
        auto alignment = align[i];
        if (alignment > 0) {
            // Alignment is expressed in data items
            strides[i] = torq::align_ceil(shape[i + 1] * strides[i + 1], alignment);
        }
        else if (alignment < 0) {
            // Alignment is expressed in elements of this dimension
            strides[i] = torq::align_ceil(shape[i + 1], -alignment) * strides[i + 1];
        }
        else {
            // No alignment required
            strides[i] = shape[i + 1] * strides[i + 1];
        }
    }

    return strides;
}

torq_hl::MemorySpace getEncodingMemorySpace(ShapedType type) {
    return getEncoding(type).getMemSpace();
}

bool hasDenseEncoding(ShapedType type) {

    auto encoding = getEncoding(type);

    if (encoding.getStrides().empty()) {
        return true;
    }

    // there are strides so we need to check if they are dense
    int naturalStride = 1;

    for (int i = type.getRank() - 1; i >= 0; i--) {
        if (encoding.getStrides()[i] != naturalStride) {
            return false;
        }
        auto dimSize = type.getShape()[i];

        if (dimSize == ShapedType::kDynamic) {
            llvm::report_fatal_error("Dynamic dimensions not supported in dense encoding check");
        }

        naturalStride *= dimSize;
    }

    return true;
}

bool isDenseInMemory(ShapedType type) {
    if (!hasDenseEncoding(type)) {
        return false;
    }

    if (auto memRefType = dyn_cast<MemRefType>(type)) {
        if (!memRefType.getLayout().isIdentity()) {
            return false;
        }
    }

    return true;
}

SmallVector<int64_t> getEncodedStridesElements(ShapedType type) {

    // if it is a memref type and there is a non identity layout we need to handle this
    // specially
    if (auto memRefType = dyn_cast<MemRefType>(type)) {

        // the encoding is dense so memref strides directly map to backing buffer strides
        if (hasDenseEncoding(type)) {

            // the memref has a strided layout, we return the strided layout
            if (auto stridesAttr =
                    mlir::dyn_cast_if_present<StridedLayoutAttr>(memRefType.getLayout())) {
                return SmallVector<int64_t>(stridesAttr.getStrides());
            }
            // the memref has an identity layout, we can compute the strides
            else if (memRefType.getLayout().isIdentity()) {

                SmallVector<int64_t> strides{};
                auto rank = memRefType.getRank();

                if (rank == 0) {
                    return {};
                }

                strides.resize(rank);
                strides[rank - 1] = 1;

                auto shapes = memRefType.getShape();

                for (int i = strides.size() - 2; i >= 0; i--) {

                    if (shapes[i + 1] == ShapedType::kDynamic) {
                        llvm::report_fatal_error(
                            "Dynamic dimensions not supported in stride computation"
                        );
                    }

                    strides[i] = strides[i + 1] * shapes[i + 1];
                }

                return strides;
            }
            else {
                type.dump();
                assert(
                    false && "only identity or strided layout memrefs with dense encoding supported"
                );
            }
        }
        else if (!memRefType.getLayout().isIdentity()) {

            // FIXME: all this is very ugly, we should revisit the whole encoding system
            auto [memRefStrides, memRefOffset] = getStridesAndOffset(memRefType);

            auto encoding = getEncoding(type);

            if (memRefType.getRank() != encoding.getCounts().size()) {
                type.dump();
                llvm::report_fatal_error("Non-rank preserving subviews not supported");
            }

            // we know the encoding has strides otherwise we would have entered the previous branch
            assert(!encoding.getStrides().empty());

            // since the memref is backed by a buffer with the given encoding we know it must
            // be subview of an allocation with the same counts as the encoding
            //
            // we can then compute the strides the memref would have if it was a subview with
            // slicing strides 1 for each dimension and check that the case
            //
            // if that's the case we know that the strides of the subview are the same as the
            // strides of the backing buffer
            SmallVector<int64_t> naturalStrides(type.getRank());
            naturalStrides[type.getRank() - 1] = 1;

            for (int i = type.getRank() - 2; i >= 0; i--) {
                naturalStrides[i] = naturalStrides[i + 1] * encoding.getCounts()[i + 1];

                if (naturalStrides[i] != memRefStrides[i]) {
                    type.dump();
                    llvm::report_fatal_error(Twine(
                        "Non-stride subview with non-dense encoding not supported found stride " +
                        std::to_string(naturalStrides[i]) + " expected stride" +
                        std::to_string(memRefStrides[i])
                    ));
                }
            }

            // the strides used to access the elements are the strides of the backing buffer
            return SmallVector<int64_t>(encoding.getStrides());
        }

        // fallthrough dense memrefs with any encoding are handled as other shaped types
    }

    auto encoding = getEncoding(type);

    // dense encoding
    if (encoding.getStrides().empty()) {
        SmallVector<int64_t> strides(type.getRank());
        int64_t stride = 1;
        for (int i = type.getRank() - 1; i >= 0; i--) {
            strides[i] = stride;

            auto dimSize = type.getShape()[i];

            if (dimSize == ShapedType::kDynamic) {
                llvm::report_fatal_error("Dynamic dimensions not supported in stride computation");
            }

            stride *= dimSize;
        }
        return strides;
    }

    return {encoding.getStrides().begin(), encoding.getStrides().end()};
}

int64_t getEncodedDataSizeElements(ShapedType type) {

    auto strides = getEncodedStridesElements(type);

    int lastElement = 0;

    for (int i = 0; i < type.getRank(); i++) {
        auto dimSize = type.getShape()[i];
        if (dimSize == ShapedType::kDynamic) {
            llvm::report_fatal_error("Dynamic dimensions not supported in data size computation");
        }
        lastElement += (dimSize - 1) * strides[i];
    }

    return lastElement + 1;
}

int64_t getEncodedTotalSizeElements(ShapedType type) {
    auto encoding = getEncoding(type);

    if (type.getRank() == 0) {
        return 1 + encoding.getPadding();
    }
    // we don't have information on the backing buffer size
    // in the encoding
    else if (encoding.getStrides().empty()) {

        // if the type is a strided memref we need to compute the size
        // based on the stride of the first dimension, this is the actual
        // size in memory of the backing buffer
        if (auto memrefType = dyn_cast<MemRefType>(type)) {

            if (!type.hasStaticShape()) {
                llvm::report_fatal_error("Dynamic shapes not supported in total size computation");
            }

            SmallVector<int64_t> strides;
            int64_t offset;
            auto ret = getStridesAndOffset(memrefType, strides, offset);

            if (failed(ret)) {
                llvm::report_fatal_error("Failed to get memref strides and offset");
            }

            // the backing buffer total size is dim(bb, 0) * stride(0) + padding
            // we don't know dim(bb, 0) but we know it must be at least dim(memref, 0)
            // so we can approximate the total size as dim(memref, 0) * stride(0) + padding
            // the memref however doesn't start at the beginning of the backing buffer
            // but at offset, so we need to compute the offset from the beginning of
            // of the dim(0) element and then subtract it from the lower
            // bound on the total size
            return strides[0] * type.getShape()[0] + encoding.getPadding() - (offset % strides[0]);
        }

        return type.getNumElements() + encoding.getPadding();
    }
    else {
        auto strides = getEncodedStridesElements(type);

        if (!type.hasStaticShape()) {
            llvm::report_fatal_error("Dynamic shapes not supported in total size computation");
        }

        return strides[0] * type.getShape()[0] + encoding.getPadding();
    }
}

SmallVector<int64_t> getEncodedStridesBytes(ShapedType type) {
    auto elementSize = getElementSizeBytes(type);
    return llvm::to_vector(llvm::map_range(getEncodedStridesElements(type), [&](int64_t v) {
        return v * elementSize;
    }));
}

int64_t getEncodedTotalSizeBytes(ShapedType type) {
    return getEncodedTotalSizeElements(type) * getElementSizeBytes(type);
}

int64_t getEncodedDataSizeBytes(ShapedType type) {
    return getEncodedDataSizeElements(type) * getElementSizeBytes(type);
}

bool checkTypeMatchesEncodingRequirements(
    ShapedType type, torq_hl::TensorEncodingRequirementsAttr requirements
) {
    return checkTypeMatchesEncodingRequirements(type, EncodingRequirements::fromAttr(requirements));
}

bool checkTypeMatchesEncodingRequirements(ShapedType type, EncodingRequirements requirements) {
    auto encoding = getEncoding(type);

    // check memory space
    if (encoding.getMemSpace() != requirements.memorySpace) {
        return false;
    }

    // check padding
    if (requirements.paddingAlign > 0) {
        if (getEncodedTotalSizeElements(type) <
            torq::align_ceil(getEncodedTotalSizeElements(type), requirements.paddingAlign)) {
            return false;
        }
    }

    // check if the type must be dense
    if (requirements.onlyDense) {
        if (!hasDenseEncoding(type)) {
            return false;
        }
    }

    // check strides
    auto strides = getEncodedStridesElements(type);

    if (!requirements.stridesAlign.empty()) {
        if (strides.size() != type.getRank()) {
            return false;
        }

        for (int i = 0; i < type.getRank(); i++) {
            auto align = requirements.stridesAlign[i];
            if (align > 0) {
                // Alignment is expressed in elements
                if (strides[i] % align != 0) {
                    return false;
                }
            }
            else if (align < 0) {
                // Alignment is expressed in data items of this dimension
                if (strides[i] % (-align * strides[i + 1]) != 0) {
                    return false;
                }
            }
            // align == 0 means no alignment required
        }
    }

    return true;
}

torq_hl::TensorEncodingAttr getDefaultEncoding(ShapedType type) {
    return torq_hl::TensorEncodingAttr::get(
        type.getContext(), torq_hl::MemorySpace::Xram, /* shape = */ {}, /* strides = */ {},
        /* padding = */ 0
    );
}

torq_hl::TensorEncodingAttr getEncoding(ShapedType type) {
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
        auto enc = dyn_cast_or_null<torq_hl::TensorEncodingAttr>(rankedTensorType.getEncoding());
        if (enc) {
            return enc;
        }
    }
    else if (auto memRefType = dyn_cast<MemRefType>(type)) {
        auto enc = dyn_cast_or_null<torq_hl::TensorEncodingAttr>(memRefType.getMemorySpace());
        if (enc) {
            return enc;
        }
    }

    return getDefaultEncoding(type);
}

torq_hl::TensorEncodingAttr
cloneEncodingWithNewMemorySpace(torq_hl::TensorEncodingAttr enc, torq_hl::MemorySpace memorySpace) {
    auto encoding = torq_hl::TensorEncodingAttr::get(
        enc.getContext(), memorySpace, enc.getCounts(), enc.getStrides(), enc.getPadding()
    );

    return encoding;
}

torq_hl::TensorEncodingAttr
createAlignedEncoding(ShapedType shapedType, const EncodingRequirements &requirements) {

    if (!shapedType.hasStaticShape()) {
        llvm::report_fatal_error("Dynamic shapes not supported in aligned encoding creation");
    }

    if (requirements.stridesAlign.empty()) {
        auto padding = 0;

        if (requirements.paddingAlign > 0) {
            padding = torq::align_ceil(shapedType.getNumElements(), requirements.paddingAlign) -
                      shapedType.getNumElements();
        }

        return torq_hl::TensorEncodingAttr::get(
            shapedType.getContext(), requirements.memorySpace, /* counts = */ {},
            /* strides = */ {}, padding
        );
    }
    else {
        assert(requirements.stridesAlign.size() == shapedType.getRank());
        auto alignedStrides = getAlignedStrides(shapedType.getShape(), requirements.stridesAlign);
        auto dataSize = alignedStrides[0] * shapedType.getShape()[0];

        auto padding = 0;

        if (requirements.paddingAlign > 0) {
            padding = torq::align_ceil(dataSize, requirements.paddingAlign) - dataSize;
        }

        return torq_hl::TensorEncodingAttr::get(
            shapedType.getContext(), requirements.memorySpace, shapedType.getShape(),
            alignedStrides, padding
        );
    }
}

torq_hl::TensorEncodingAttr
createDenseEncoding(ShapedType type, torq_hl::MemorySpace memorySpace, int64_t padding) {
    return torq_hl::TensorEncodingAttr::get(
        type.getContext(), memorySpace, /* counts = */ {}, /* strides = */ {}, padding
    );
}

MemRefType
createMemRefTypeWithEncoding(MemRefType baseType, torq_hl::TensorEncodingAttr encodingAttr) {
    if (!encodingAttr) {
        encodingAttr = getDefaultEncoding(baseType);
    }

    return MemRefType::get(
        baseType.getShape(), baseType.getElementType(), baseType.getLayout(), encodingAttr
    );
}

MemRefType createMemRefTypeWithMemorySpace(MemRefType baseType, torq_hl::MemorySpace memorySpace) {

    auto encoding = getEncoding(baseType);
    auto newEncoding = cloneEncodingWithNewMemorySpace(encoding, memorySpace);

    return createMemRefTypeWithEncoding(baseType, newEncoding);
}

RankedTensorType createRankedTensorTypeWithEncoding(
    RankedTensorType baseType, torq_hl::TensorEncodingAttr encodingAttr
) {
    return RankedTensorType::get(baseType.getShape(), baseType.getElementType(), encodingAttr);
}

Value convertTensorToEncoding(
    OpBuilder &builder, TypedValue<RankedTensorType> value,
    torq_hl::TensorEncodingAttr encodingAttr,
    const std::optional<EncodingRequirements> requirements, Value initValue
) {

    assert((encodingAttr || requirements) && "either encoding or requirements must be set");

    if (encodingAttr == nullptr) {
        encodingAttr = createAlignedEncoding(value.getType(), *requirements);
    }

    auto origEncoding = getEncoding(value.getType());

    if (origEncoding == encodingAttr) {
        return value;
    }

    auto origMemorySpace = origEncoding.getMemSpace();
    auto newMemorySpace = encodingAttr.getMemSpace();

    Value intermediateValue = value;

    // insert an intermediate conversion if we know its necesssary
    bool fromXram = origMemorySpace == torq_hl::MemorySpace::Xram;
    bool fromCss = origMemorySpace == torq_hl::MemorySpace::Dtcm ||
                   origMemorySpace == torq_hl::MemorySpace::Itcm;
    bool toCss = newMemorySpace == torq_hl::MemorySpace::Dtcm ||
                 newMemorySpace == torq_hl::MemorySpace::Itcm;
    bool toXram = newMemorySpace == torq_hl::MemorySpace::Xram;

    if ((fromXram && toCss) || (fromCss && toXram)) {
        auto intermediateEncoding =
            createDenseEncoding(value.getType(), torq_hl::MemorySpace::Lram);
        intermediateValue =
            convertTensorToEncoding(builder, value, intermediateEncoding, std::nullopt, initValue);
    }

    auto dstType = createRankedTensorTypeWithEncoding(value.getType(), encodingAttr);
    Value emptyValue = initValue
                           ? initValue
                           : builder.create<tensor::EmptyOp>(value.getLoc(), dstType, ValueRange{});

    if (requirements) {

        return builder
            .create<torq_hl::ConvertOp>(
                value.getLoc(), dstType, emptyValue, intermediateValue,
                /* requirements = */ requirements->toAttr(value.getContext()),
                /* encoding = */ nullptr
            )
            .getResult(0);
    }
    else {
        return builder
            .create<torq_hl::ConvertOp>(
                value.getLoc(), dstType, emptyValue, intermediateValue,
                /* requirements = */ nullptr, /* encoding = */ encodingAttr
            )
            .getResult(0);
    }
}

Value convertTensorToType(
    OpBuilder &builder, TypedValue<RankedTensorType> value, RankedTensorType targetType,
    Value initValue
) {

    assert(value.getType().getElementType() == targetType.getElementType());
    assert(value.getType().getShape() == targetType.getShape());

    auto origEncoding = getEncoding(value.getType());

    if (value.getType() == targetType) {
        return value;
    }

    auto targetEncoding = getEncoding(targetType);

    auto origMemorySpace = origEncoding.getMemSpace();
    auto newMemorySpace = targetEncoding.getMemSpace();

    Value intermediateValue = value;

    // insert an intermediate conversion if we know its necesssary
    bool fromXram = origMemorySpace == torq_hl::MemorySpace::Xram;
    bool fromCss = origMemorySpace == torq_hl::MemorySpace::Dtcm ||
                   origMemorySpace == torq_hl::MemorySpace::Itcm;
    bool toCss = newMemorySpace == torq_hl::MemorySpace::Dtcm ||
                 newMemorySpace == torq_hl::MemorySpace::Itcm;
    bool toXram = newMemorySpace == torq_hl::MemorySpace::Xram;

    if ((fromXram && toCss) || (fromCss && toXram)) {
        auto intermediateEncoding =
            createDenseEncoding(value.getType(), torq_hl::MemorySpace::Lram);
        intermediateValue =
            convertTensorToEncoding(builder, value, intermediateEncoding, std::nullopt, initValue);
    }

    Value emptyValue = builder.create<tensor::EmptyOp>(value.getLoc(), targetType, ValueRange{});

    return builder
        .create<torq_hl::ConvertOp>(
            value.getLoc(), targetType, emptyValue, intermediateValue,
            /* requirements = */ nullptr, /* encoding = */ targetEncoding
        )
        .getResult(0);
}

} // namespace mlir::syna
