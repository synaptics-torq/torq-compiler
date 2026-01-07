#include "torq/Utils/ExecutorAssignment.h"
#include "mlir/IR/Operation.h"

namespace mlir::syna::torq {

static std::string EXECUTOR_ATTR_NAME = "torq-executor";
static std::string COMPILE_TIME_CONST_ATTR_NAME = "torq-compile-time-const";

torq_hl::Executor getTargetExecutor(Operation *op) {
    auto executor = op->getAttr(EXECUTOR_ATTR_NAME);

    if (!executor) {
        return torq_hl::Executor::Slice;
    }

    return cast<torq_hl::ExecutorAttr>(executor).getValue();
}

torq_hl::Executor getTargetExecutor(Operation *op, torq_hl::Executor defaultExecutor) {
    auto executor = op->getAttr(EXECUTOR_ATTR_NAME);

    if (!executor) {
        return defaultExecutor;
    }

    return cast<torq_hl::ExecutorAttr>(executor).getValue();
}

NamedAttribute getTargetExecutorAttr(MLIRContext *ctx, torq_hl::Executor executor) {
    return NamedAttribute(
        StringAttr::get(ctx, EXECUTOR_ATTR_NAME), torq_hl::ExecutorAttr::get(ctx, executor)
    );
}

void setTargetExecutorAttr(Operation *op, torq_hl::Executor executor) {
    op->setAttr(EXECUTOR_ATTR_NAME, torq_hl::ExecutorAttr::get(op->getContext(), executor));
}

void setCompileTimeConstAttr(Operation *op) {
    op->setAttr(COMPILE_TIME_CONST_ATTR_NAME, BoolAttr::get(op->getContext(), true));
}

bool isCompileTimeConst(Operation *op) {
    auto attr = op->getAttr(COMPILE_TIME_CONST_ATTR_NAME);
    if (!attr)
        return false;
    return mlir::cast<BoolAttr>(attr).getValue();
}
} // namespace mlir::syna::torq