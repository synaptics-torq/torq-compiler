#include "torq/Utils/InvocationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-invocation-utils"

namespace mlir::syna::torq {

namespace {

struct ProgramExecutor {

    ProgramExecutor(Operation *rootOp, WalkExecutionOptions options)
        : rootOp(rootOp), options(options) {}

    // operation where to start the execution
    Operation *rootOp;

    // callbacks
    WalkExecutionOptions options;

    DenseMap<InvocationValue, SmallVector<Value>> returnedValues;

    LogicalResult execute() {
        IRMapping mapping;
        return processBlock(&(rootOp->getRegion(0).getBlocks().front()), nullptr, mapping);
    }

  private:
    LogicalResult processStartProgramOp(torq_hl::StartProgramOp startOp, const IRMapping &mapping) {
        auto startedInvocation = cast<InvocationValue>(mapping.lookup(startOp.getInvocation()));
        auto startedInvocationOp = startedInvocation.getDefiningOp<torq_hl::CreateInvocationOp>();

        if (!startedInvocationOp) {
            return startOp.emitError("Expected an invocation created by a create_invocation op or "
                                     "block argument pointing to one");
        }

        // notify the listener that we are starting the invocation
        if (options.onStart) {
            if (failed(options.onStart(startOp, startedInvocation, mapping))) {
                return failure();
            }
        }

        auto startedProgramOp =
            startedInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        // if the invocation is a program we can process the operations inside, if it's not (e.g. a
        // torq.css_program) we cannot recurse in it as it is already compiled
        if (startedProgramOp) {

            if (options.walkInto) {
                if (!options.walkInto(startOp, startedInvocation, mapping)) {
                    return success();
                }
            }

            auto block = &startedProgramOp->getRegion(0).getBlocks().front();

            IRMapping startedMapping;

            for (int i = 0; i < block->getNumArguments(); ++i) {
                startedMapping.map(block->getArgument(i), mapping.lookup(startOp.getArgs()[i]));
            }

            auto maybeReturnedValues = processBlock(block, startedInvocation, startedMapping);

            // recursively process the program
            if (failed(maybeReturnedValues)) {
                return failure();
            }

            returnedValues[startedInvocation] = *maybeReturnedValues;
        }

        return success();
    }

    LogicalResult processWaitProgramOp(torq_hl::WaitProgramOp waitOp, IRMapping &mapping) {
        // find the operation that defines the invocation
        auto waitedInvocation = cast<InvocationValue>(mapping.lookup(waitOp.getInvocation()));
        auto waitedInvocationOp = waitedInvocation.getDefiningOp<torq_hl::CreateInvocationOp>();

        if (!waitedInvocationOp) {
            return waitOp.emitError("Expected an invocation created by a create_invocation op or "
                                    "block argument pointing to one");
        }

        SmallVector<Value> waitReturnValues;

        if (waitOp.getNumResults() > 0) {
            auto maybeWaitReturnValues = returnedValues.find(waitedInvocation);

            if (maybeWaitReturnValues == returnedValues.end()) {
                return waitOp.emitError("Cannot skip operation that returns values");
            }

            waitReturnValues = maybeWaitReturnValues->second;
            returnedValues.erase(waitedInvocation);
        }

        // notify the listener that the invocation has finished
        if (options.onFinish) {
            if (failed(options.onFinish(waitOp, waitedInvocation, waitReturnValues))) {
                return failure();
            }
        }

        for (int i = 0; i < waitOp.getNumResults(); i++) {
            mapping.map(waitOp.getResult(i), waitReturnValues[i]);
        }

        returnedValues.erase(waitedInvocation);

        return success();
    }

    FailureOr<SmallVector<Value>>
    processBlock(Block *block, InvocationValue currentInvocation, IRMapping &mapping) {

        for (auto &op : block->getOperations()) {

            LLVM_DEBUG({
                llvm::dbgs() << "Processing operation:\n";
                op.dump();
            });

            if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {
                if (failed(processStartProgramOp(startOp, mapping))) {
                    return failure();
                }
            }
            else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {
                if (failed(processWaitProgramOp(waitOp, mapping))) {
                    return failure();
                }
            }
            else if (auto returnOp = dyn_cast<torq_hl::ReturnOp>(op)) {
                SmallVector<Value> returnedValues;

                for (auto returnValue : returnOp.getOutputs()) {
                    returnedValues.push_back(mapping.lookup(returnValue));
                }

                return returnedValues;
            }
            else {
                if (options.onExecute) {
                    if (failed(options.onExecute(&op, currentInvocation, mapping))) {
                        return failure();
                    }
                }

                for (auto result : op.getResults()) {
                    mapping.map(result, result);
                }
            }
        }

        return SmallVector<Value>();
    }
};
} // namespace

Value getCurrentValue(Value value, InvocationValue invocation, InvocationArguments arguments) {

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {

        auto programOp = getProgramOp(invocation);

        if (!programOp) {
            return nullptr;
        }

        if (blockArg.getOwner()->getParentOp() != programOp) {
            return nullptr;
        }

        assert(
            arguments.size() == blockArg.getOwner()->getNumArguments() &&
            "Block argument should have a corresponding argument in the invocation"
        );

        return arguments[blockArg.getArgNumber()];
    }

    return value;
}

torq_hl::ProgramOp getProgramOp(InvocationValue invocation) {
    auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>();

    if (!createInvocationOp) {
        return nullptr;
    }

    auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    if (!programOp) {
        return nullptr;
    }

    return programOp;
}

torq_hl::StartProgramOp getStartOp(InvocationValue invocation) {
    for (auto user : invocation.getUsers()) {
        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(user)) {
            return startOp;
        }
    }

    return nullptr;
}

torq_hl::WaitProgramOp getWaitOp(InvocationValue invocation) {

    for (auto user : invocation.getUsers()) {
        if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(user)) {
            return waitOp;
        }
    }

    return nullptr;
}

LogicalResult walkExecution(Operation *op, WalkExecutionOptions options) {
    return ProgramExecutor(op, options).execute();
}

InvocationValue getNssInvocation(Operation *op) {

    auto parentOp = op->getParentOfType<torq_hl::ProgramOp>();

    assert(parentOp && "operation should be inside a program");
    assert(
        parentOp.getProgram().getType().getExecutor() == torq_hl::Executor::NSS &&
        "program should be NSS"
    );
    assert(parentOp.getProgram().hasOneUse() && "program should have only one use");

    return cast<torq_hl::CreateInvocationOp>(*(parentOp.getProgram().user_begin())).getInvocation();
}

std::optional<int64_t>
getExecutorId(InvocationValue invocation, InvocationValue contextInvocation) {

    if (auto blockArg = dyn_cast<BlockArgument>(invocation)) {

        auto contextInvocationOp = contextInvocation.getDefiningOp<torq_hl::CreateInvocationOp>();

        if (!contextInvocationOp) {
            return std::nullopt;
        }

        auto contextProgramOp =
            contextInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        if (!contextProgramOp) {
            return std::nullopt;
        }

        if (blockArg.getOwner()->getParentOp() != contextProgramOp) {
            return std::nullopt;
        }

        auto maybeArgs = contextInvocationOp.getExecutorArgsAddresses();

        if (!maybeArgs || blockArg.getArgNumber() >= maybeArgs->size()) {
            return std::nullopt;
        }

        return (*maybeArgs)[blockArg.getArgNumber()];
    }
    else if (auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>()) {

        if (!createInvocationOp.getExecutorId()) {
            return std::nullopt; // no executor id available
        }

        return createInvocationOp.getExecutorId()->getZExtValue();
    }

    return std::nullopt;
}

} // namespace mlir::syna::torq