#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>
#include <torq/Codegen/Pool.h>
#include <torq/Dialect/TorqHL/TorqHLOps.h>

namespace mlir::syna::torq {

LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, Pool &pool, torq_hl::MemorySpace memorySpace
);

}