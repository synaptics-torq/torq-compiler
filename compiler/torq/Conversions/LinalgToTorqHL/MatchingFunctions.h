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

} // namespace mlir::syna::torq