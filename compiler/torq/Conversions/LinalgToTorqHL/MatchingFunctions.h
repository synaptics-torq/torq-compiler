#pragma once

#include "mlir/IR/Operation.h"
#include <string>

namespace mlir::syna::torq {

bool isTorqCastOp(
    Operation *op, std::string &opName, std::string &failReason, bool *isUnsigned = nullptr
);

bool isTorqAbsOp(Operation *op, std::string &failReason);

bool isTorqCeilOp(Operation *op, std::string &failReason);

bool isTorqClampOp(
    Operation *op, int32_t &minIntValue, int32_t &maxIntValue, float &minFloatValue,
    float &maxFloatValue, std::string &failReason
);

bool isTorqFloorOp(Operation *op, std::string &failReason);

bool isTorqMatMul(Operation *op, std::string &failReason);

bool isTorqNegateOp(Operation *op, std::string &failReason);

bool isLogicNotOp(Operation *op, std::string &failReason);

// Check if the operation is a linalg.reduce sum, max or min that can execute on
// Torq: bf16, f32 or 8/16/32-bit integer input with an output of the same type, or
// a bf16 sum with f32 output.
bool isTorqReduceOp(Operation *op, std::string &failReason);

// Like isTorqReduceOp but for a linalg.generic form of a reduce sum, max or min
// (single input, one reduction loop), the shape the reduction tiling pass produces
// for its partial-reduction tiles.
bool isTorqGenericReduceOp(Operation *op, std::string &failReason);

// True if starting the accumulation of `combineOp` from the reduction init
// `init` changes nothing: `init` is a splat of the identity of `combineOp` or a
// tensor.empty, seen through slices, casts and tiling loops.
bool isIdentityReduceInit(Value init, Operation *combineOp);

} // namespace mlir::syna::torq