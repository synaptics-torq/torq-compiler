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

// Check if the operation is a reduce sum that can execute on Torq.
// Only supports bf16 input -> bf16/f32 output or integer types.
// Does NOT support other reduce operations (max, min, etc.) - those are handled differently.
bool isTorqReduceSumOp(Operation *op, std::string &failReason);

} // namespace mlir::syna::torq