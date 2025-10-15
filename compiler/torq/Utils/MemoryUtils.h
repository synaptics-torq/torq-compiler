#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::syna {

// Return the size in bytes required to store an element of the given type
int64_t getElementSizeBytes(ShapedType shapedType);

// Return the size in bytes required to store a scalar of the given type
int64_t getScalarSizeBytes(Type type);

// returns the data size with tensor shapeType in the case before bufferization
size_t getShapeTypeDataSize(mlir::ShapedType type);

void setLramAddress(Operation *op, int64_t address);
std::optional<int64_t> getLramAddress(Value value, int64_t offset = 0);

void setXramAddress(Operation *op, int64_t address);
std::optional<int64_t> getXramAddress(Value value, int64_t offset = 0);
std::optional<int64_t> getXramAddress(Operation *op, int64_t offset = 0);

void setItcmAddress(Operation *op, int64_t address);
std::optional<int64_t> getItcmAddress(Value value, int64_t offset = 0);

void setDtcmAddress(Operation *op, int64_t address);
std::optional<int64_t> getDtcmAddress(Value value, int64_t offset = 0);

// Reserve an XRAM area of the given size, return the address of the reserved area
// The returned address is guaranteed not to overlap with any other reserved area
// in the same
int64_t reserveXramArea(Operation *funcOp, int64_t size);

// Set the address of the value produced by the operation (works only with single result operations)
// )
LogicalResult setAddress(Operation *op, int64_t address);

// Set the address of the value by setting the corresponding attribute on the operation generating
// the value
LogicalResult setAddress(Value value, int64_t address);

// return the address of the first entry of a memref as seen by the executor, if it is accessible
std::optional<int64_t> getExecutorDataStartAddress(
    torq_hl::Executor executor, Value value, int64_t offset = 0,
    TypedValue<torq_hl::InvocationType> invocation = nullptr
);

// attribute used to store resolved addesses on operations that use values with addresses
const std::string RESOLVED_ADDRESSES_ATTR_NAME = "torq.resolved_addresses";

// Returns the address of the given value when it is accesses within the given program invocation
std::optional<int64_t> getAddress(
    Value value, int64_t offset = 0, TypedValue<torq_hl::InvocationType> invocation = nullptr
);

// Returns the address of the first byte of the given value (this is the base address plus the
// offset in the layout)
std::optional<int64_t> getDataStartAddress(
    Value value, int64_t offset = 0, TypedValue<torq_hl::InvocationType> invocation = nullptr
);

// Returns true if the operation is creating an alias of another existing memref
bool isDerivedMemRefOperation(Operation *op);

// Returns the operand of op from for which the operations is creating an alias
OpOperand &getDerivedMemRefBase(Operation *op);

// Returns the alignment possible with the bytes based on the type
int getAlignmentByType(int bytes, mlir::Type type);

} // namespace mlir::syna
