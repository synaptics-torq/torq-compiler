#include "torq/Utils/ExecutorAssignment.h"
#include "mlir/IR/Operation.h"

namespace mlir::syna::torq {

static std::string EXECUTOR_ATTR_NAME = "torq-executor";

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

bool locHas(Location loc, StringRef key) {
    if (auto n = dyn_cast<NameLoc>(loc))
        return n.getName().str() == key || locHas(n.getChildLoc(), key);
    if (auto f = dyn_cast<FusedLoc>(loc)) {
        for (auto sub : f.getLocations())
            if (locHas(sub, key))
                return true;
    }
    return false;
}

} // namespace mlir::syna::torq