#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::syna {

struct EncodingRequirements {
    torq_hl::MemorySpace memorySpace;
    SmallVector<int64_t> stridesAlign{};
    int64_t paddingAlign = 0;
    bool onlyDense = false;

    torq_hl::TensorEncodingRequirementsAttr toAttr(MLIRContext *ctx) const {
        return torq_hl::TensorEncodingRequirementsAttr::get(
            ctx, memorySpace, stridesAlign, paddingAlign, onlyDense
        );
    }

    static EncodingRequirements fromAttr(torq_hl::TensorEncodingRequirementsAttr attr) {
        return {
            attr.getMemSpace(),
            SmallVector<int64_t>(attr.getStridesAlign().begin(), attr.getStridesAlign().end()),
            attr.getPaddingAlign(), attr.getOnlyDense()
        };
    }
};

// returns the memory space of a given type
torq_hl::MemorySpace getEncodingMemorySpace(ShapedType type);

// return true if the given type has a dense encoding
bool hasDenseEncoding(ShapedType type);

// return true if the elements of the given type are contiguous in memory
// a type must have a dense encoding to be contiguous and if it is a
// memref it must also have an identity layout
bool isDenseInMemory(ShapedType type);

// return the strides of a given type used to access the elements
// in the backing memory
SmallVector<int64_t> getEncodedStridesElements(ShapedType type);

// return the total size of the backing memory in elements of the given type
int64_t getEncodedTotalSizeElements(ShapedType type);

// return the size of the backing memory without trailing padding data
int64_t getEncodedDataSizeElements(ShapedType type);

// return the strides of a given type used to access the elements
// in the backing memory
SmallVector<int64_t> getEncodedStridesBytes(ShapedType type);

// return the total size of the backing memory in elements of the given type
int64_t getEncodedTotalSizeBytes(ShapedType type);

// return the size of the backing memory without trailing padding data
int64_t getEncodedDataSizeBytes(ShapedType type);

// return true if the given type matches the given encoding requirements
bool checkTypeMatchesEncodingRequirements(
    ShapedType type, torq_hl::TensorEncodingRequirementsAttr requirements
);

// return true if the given type matches the given encoding requirements
bool checkTypeMatchesEncodingRequirements(ShapedType type, EncodingRequirements requirements);

// return the default encoding attribute
torq_hl::TensorEncodingAttr getDefaultEncoding(ShapedType type);

// return an encoding attribute for the given type, if the type has no encoding
// the the default encoding is returned
torq_hl::TensorEncodingAttr getEncoding(ShapedType type);

// clone encoding attributes but select the desired memory space
torq_hl::TensorEncodingAttr
cloneEncodingWithNewMemorySpace(torq_hl::TensorEncodingAttr enc, torq_hl::MemorySpace memorySpace);

// return an encoding with the required alignment for the given tensor type
torq_hl::TensorEncodingAttr
createAlignedEncoding(ShapedType shapedType, const EncodingRequirements &requirements);

// return a dense encoding (natural strides) with the given memory space and padding
torq_hl::TensorEncodingAttr
createDenseEncoding(ShapedType type, torq_hl::MemorySpace memorySpace, int64_t padding = 0);

// returns a new MemRefType with the same shape and element type of the original type but
// with the new encoding and strides specified in the encoding
MemRefType
createMemRefTypeWithEncoding(MemRefType baseType, torq_hl::TensorEncodingAttr encodingAttr);

// returns a new MemRefType with the same layout and encoding, except for the memory space
MemRefType createMemRefTypeWithMemorySpace(MemRefType baseType, torq_hl::MemorySpace memorySpace);

// returns a new RankedTensorType with the same shape and element type of the original type but
// with the new encoding
RankedTensorType createRankedTensorTypeWithEncoding(
    RankedTensorType baseType, torq_hl::TensorEncodingAttr encodingAttr
);

// create IR that converts a given Value to a value with given encoding, the inserted conversion
// will use the encodingAttr or the provided requirements as attributes, when the encodingAttr is
// nullptr a new encoding will be created that satisfies the requirements
// If the initValue is not provided the result will be stored in a new empty tensor
Value convertTensorToEncoding(
    OpBuilder &builder, TypedValue<RankedTensorType> value,
    torq_hl::TensorEncodingAttr encodingAttr,
    const std::optional<EncodingRequirements> requirements = std::nullopt,
    Value initValue = Value(0)
);

// creates IR that converts a given Value to a value with the given targetType
// the targetType must have the same shape and element type as the original value
// If the initValue is not provided the result will be stored in a new empty tensor
Value convertTensorToType(
    OpBuilder &builder, TypedValue<RankedTensorType> value, RankedTensorType targetType,
    Value initValue = Value(0)
);

} // namespace mlir::syna
