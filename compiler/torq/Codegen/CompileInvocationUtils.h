#pragma once

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Serialization/DescGen.h"

namespace mlir::syna::torq {

FailureOr<torq_hl::StartProgramOp> findStartProgramOp(torq_hl::CreateInvocationOp createInvocationOp
);

LogicalResult updateCode(DescGen &_npu, uint32_t xramAddress, SmallVector<int8_t> &code);

} // namespace mlir::syna::torq