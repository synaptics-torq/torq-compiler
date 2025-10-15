#pragma once

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

using InvocationValue = TypedValue<torq_hl::InvocationType>;
using InvocationArguments = SmallVector<Value>;
using InvocationReturns = SmallVector<Value>;

struct WalkExecutionOptions {

    // whether the invocation should be walked into
    std::function<bool(torq_hl::StartProgramOp, InvocationValue, const IRMapping &)> walkInto;

    // an invocation is started
    std::function<LogicalResult(torq_hl::StartProgramOp, InvocationValue, const IRMapping &)>
        onStart;

    // an invocation has been waited for
    std::function<LogicalResult(torq_hl::WaitProgramOp, InvocationValue, InvocationReturns)>
        onFinish;

    // an operation is executed in a given invocation
    std::function<LogicalResult(Operation *, InvocationValue, const IRMapping &)> onExecute;
};

// simulate the execution of the program following any start_program operation
// and notify the events object about the execution events
LogicalResult walkExecution(Operation *op, WalkExecutionOptions options);

// Returns the NSS invocation associated with the given operation
InvocationValue getNssInvocation(Operation *op);

// Returns the program corresponding to the invocation
torq_hl::ProgramOp getProgramOp(InvocationValue invocation);

// Returns the start operation associated with invocation
torq_hl::StartProgramOp getStartOp(InvocationValue invocation);

// Returns the wait operation associated with invocation
torq_hl::WaitProgramOp getWaitOp(InvocationValue invocation);

std::optional<int64_t> getExecutorId(InvocationValue invocation, InvocationValue contextInvocation);

} // namespace mlir::syna::torq
