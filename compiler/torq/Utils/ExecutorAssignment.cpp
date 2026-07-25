#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"

using namespace mlir::iree_compiler;
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

FailureOr<Value> createCompileTimeConstOp(Operation *maybeConstOp, RewriterBase &rewriter) {
    if (!maybeConstOp)
        return failure();

    // The requested value is already behind a compile-time-constant boundary.
    if (auto existing = dyn_cast<torq_hl::CompileInputToConstOp>(maybeConstOp))
        return existing.getOutput();

    SmallVector<Operation *, 4> worklist{maybeConstOp};
    llvm::DenseSet<Operation *> visited;
    DenseSet<Operation *> opsOfInterest;
    SmallVector<std::pair<OpOperand *, Value>> boundaryBypasses;
    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        visited.insert(currentOp);

        // Collect all ops nested inside the regions of `currentOp` into `opsOfInterest`.
        // This ensures that ops inside scf.for / scf.forall bodies are also marked
        // as host executor when the parent loop is a compile-time-const op.
        for (Region &region : currentOp->getRegions()) {
            for (Block &block : region.getBlocks()) {
                for (Operation &nestedOp : block.getOperations()) {
                    if (visited.insert(&nestedOp).second) {
                        opsOfInterest.insert(&nestedOp);
                        worklist.push_back(&nestedOp);
                    }
                }
            }
        }

        for (OpOperand &operand : currentOp->getOpOperands()) {
            // Walk through any existing boundaries to the real producer so this cone
            // is a clean host chain; record the operand so it can be rewired below.
            Value definingValue = operand.get();
            while (auto boundary = definingValue.getDefiningOp<torq_hl::CompileInputToConstOp>()) {
                definingValue = boundary.getInput();
            }
            if (definingValue != operand.get())
                boundaryBypasses.emplace_back(&operand, definingValue);

            Operation *defOp = definingValue.getDefiningOp();
            if (!defOp) {
                auto bArg = dyn_cast<BlockArgument>(definingValue);
                if (!bArg)
                    continue;

                defOp = getDefiningOpForBlockArg(bArg);
                if (!defOp)
                    continue;
            }

            // the value depends on the input, we cannot compute this
            if (isa<IREE::TensorExt::DispatchTensorLoadOp, IREE::HAL::InterfaceBindingSubspanOp>(
                    defOp
                )) {
                return failure();
            }

            if (!visited.insert(defOp).second)
                continue;

            opsOfInterest.insert(defOp);
            worklist.push_back(defOp);
        }
    }
    // Rewire in-cone operands past existing boundaries in place. The boundary and
    // its other consumers stay untouched, avoiding nested boundaries without
    // deleting shared IR during dialect conversion.
    for (auto &bypass : boundaryBypasses) {
        OpOperand *operand = bypass.first;
        Value definingValue = bypass.second;
        rewriter.modifyOpInPlace(operand->getOwner(), [&]() { operand->set(definingValue); });
    }
    setTargetExecutorAttr(maybeConstOp, torq_hl::Executor::Host);
    for (Operation *op : opsOfInterest) {
        // Bypassed boundaries can remain when they were collected as region-nested
        // ops; leave their executor assignment untouched.
        if (isa<torq_hl::CompileInputToConstOp>(op))
            continue;
        setTargetExecutorAttr(op, torq_hl::Executor::Host);
    }
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(maybeConstOp);
    auto constOp = torq_hl::CompileInputToConstOp::create(
        rewriter, maybeConstOp->getLoc(), maybeConstOp->getResult(0).getType(),
        maybeConstOp->getResult(0)
    );
    return constOp.getOutput();
}

void removeCompileTimeConst(Operation *op, RewriterBase &rewriter) {
    auto constOp = dyn_cast<torq_hl::CompileInputToConstOp>(op);
    if (!constOp)
        return;

    Value input = constOp.getInput();

    rewriter.replaceOp(op, ValueRange{input});
}

bool isCompileTimeConst(Operation *op) { return isa<torq_hl::CompileInputToConstOp>(op); }

SmallVector<Value> collectAllCompileTimeConstOps(Operation *op) {
    SmallVector<Value> valuesToProcess;
    op->walk([&](Operation *op) {
        if (!isCompileTimeConst(op)) {
            return WalkResult::advance();
        }

        Operation *compileOp = op;
        while (auto tOp = dyn_cast_or_null<torq_hl::CompileInputToConstOp>(compileOp)) {
            compileOp = tOp.getInput().getDefiningOp();
        }
        if (compileOp) {
            valuesToProcess.push_back(compileOp->getResult(0));
        }
        return WalkResult::advance();
    });
    return valuesToProcess;
}

void setCompileTimeConstAttr(Operation *op) {
    op->setAttr(COMPILE_TIME_CONST_ATTR_NAME, BoolAttr::get(op->getContext(), true));
}

bool isCompileTimeConstAttr(Operation *op) { return op->hasAttr(COMPILE_TIME_CONST_ATTR_NAME); }
void removeCompileTimeConstAttr(Operation *op) { op->removeAttr(COMPILE_TIME_CONST_ATTR_NAME); }

} // namespace mlir::syna::torq
