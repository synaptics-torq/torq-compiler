#include "torq/Utils/ExecutorAssignment.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

Operation *getDefiningOpForBlockArg(BlockArgument bArg) {
    auto parentOp = bArg.getParentBlock()->getParentOp();

    // scf.forall block args are [induction vars..., shared_outs...]; handle separately
    // since operand indices don't map directly to block argument indices.
    if (auto forallOp = dyn_cast<scf::ForallOp>(parentOp)) {
        unsigned numInductionVars = forallOp.getRank();
        if (bArg.getArgNumber() >= numInductionVars) {
            unsigned outputIndex = bArg.getArgNumber() - numInductionVars;
            if (outputIndex < forallOp.getOutputs().size())
                return forallOp.getOutputs()[outputIndex].getDefiningOp();
        }
        return nullptr;
    }

    if (parentOp->getNumOperands() <= bArg.getArgNumber())
        return nullptr;

    return parentOp->getOperand(bArg.getArgNumber()).getDefiningOp();
}

void setCompileTimeConstAttr(Operation *op) {
    if (!op)
        return;

    op->setAttr(COMPILE_TIME_CONST_ATTR_NAME, BoolAttr::get(op->getContext(), true));
    setTargetExecutorAttr(op, torq_hl::Executor::Host);
    SmallVector<Operation *, 4> worklist{op};
    llvm::DenseSet<Operation *> visited;
    SmallVector<Operation *, 4> opsOfInterest;
    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        visited.insert(currentOp);

        // Collect all ops nested inside the regions of `op` into `result`.
        // This ensures that ops inside scf.for / scf.forall bodies are also marked
        // as host executor when the parent loop is a compile-time-const op.
        for (Region &region : currentOp->getRegions()) {
            for (Block &block : region.getBlocks()) {
                for (Operation &nestedOp : block.getOperations()) {
                    if (visited.insert(&nestedOp).second) {
                        opsOfInterest.push_back(&nestedOp);
                        worklist.push_back(&nestedOp);
                    }
                }
            }
        }

        for (Value operand : currentOp->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!defOp) {
                auto bArg = dyn_cast<BlockArgument>(operand);
                if (!bArg)
                    continue;

                defOp = getDefiningOpForBlockArg(bArg);
                if (!defOp)
                    continue;
            }

            if (!visited.insert(defOp).second)
                continue;

            opsOfInterest.push_back(defOp);
            worklist.push_back(defOp);
        }
    }
    for (Operation *op : opsOfInterest) {
        if (op->hasAttr(COMPILE_TIME_CONST_ATTR_NAME) && op->hasOneUse()) {
            op->removeAttr(COMPILE_TIME_CONST_ATTR_NAME);
        }
        setTargetExecutorAttr(op, torq_hl::Executor::Host);
    }
}

void removeCompileTimeConstAttr(Operation *op) { op->removeAttr(COMPILE_TIME_CONST_ATTR_NAME); }

bool isCompileTimeConst(Operation *op) {
    auto attr = op->getAttr(COMPILE_TIME_CONST_ATTR_NAME);
    if (!attr)
        return false;
    return mlir::cast<BoolAttr>(attr).getValue();
}
} // namespace mlir::syna::torq
