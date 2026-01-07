#pragma once

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"

namespace mlir::syna::torq {

torq_hl::Executor getTargetExecutor(Operation *op);
torq_hl::Executor getTargetExecutor(Operation *op, torq_hl::Executor defaultExecutor);
NamedAttribute getTargetExecutorAttr(MLIRContext *ctx, torq_hl::Executor executor);
void setTargetExecutorAttr(Operation *op, torq_hl::Executor executor);
void setCompileTimeConstAttr(Operation *op);
bool isCompileTimeConst(Operation *op);

} // namespace mlir::syna::torq