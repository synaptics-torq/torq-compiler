#pragma once

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"

namespace mlir::syna::torq {

torq_hl::Executor getTargetExecutor(Operation *op);
torq_hl::Executor getTargetExecutor(Operation *op, torq_hl::Executor defaultExecutor);
NamedAttribute getTargetExecutorAttr(MLIRContext *ctx, torq_hl::Executor executor);
void setTargetExecutorAttr(Operation *op, torq_hl::Executor executor);
void removeCompileTimeConst(Operation *op, RewriterBase &rewriter);
FailureOr<Value> createCompileTimeConstOp(Operation *op, RewriterBase &rewriter);
bool isCompileTimeConst(Operation *op);
SmallVector<Value> collectAllCompileTimeConstOps(Operation *op);
void setCompileTimeConstAttr(Operation *op);
bool isCompileTimeConstAttr(Operation *op);
void removeCompileTimeConstAttr(Operation *op);

} // namespace mlir::syna::torq