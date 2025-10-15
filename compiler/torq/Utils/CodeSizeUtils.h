#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::syna::torq {

LogicalResult getTotalNssCodeSize(mlir::FunctionOpInterface funcOp, int &size);

int getCodeSize(ArrayRef<Operation *> operations);

int getCodeSize(Block *block);

int getCodeOffset(Block *block);

} // namespace mlir::syna::torq