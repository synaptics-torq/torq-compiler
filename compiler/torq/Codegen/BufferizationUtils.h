#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::syna::torq {

// Custom function used during bufferization to copy a buffer.
LogicalResult createTorqCopy(OpBuilder &builder, Location loc, Value from, Value to);

// Custom function used during bufferization to get the memory space attribute of a tensor.
// Here we return the full enoding attribute of the tensor if present, the createTorqAllocation()
// function will then use this attribute to infer the actual layout and memspace of the memref
Attribute getTorqMemSpaceAttr(TensorType t);

// Custom function used during bufferization to allocate memory for a tensor.
// Here we use the encoding in the memSpace attribute to infer the actual layout of the memref.
FailureOr<Value> createTorqAllocation(
    OpBuilder &builder, Location loc, MemRefType memRef, ValueRange dynamicSizes, unsigned alignment
);

} // namespace mlir::syna::torq