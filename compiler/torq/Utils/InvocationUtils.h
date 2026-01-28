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

// Returns value of a given program block argument when invoked by the given invocation
FailureOr<Value> getInvocationArgument(InvocationValue invocation, BlockArgument arg);

// Returns the executor id for the given invocation when evaluated within the contextInvocation
std::optional<int64_t> getExecutorId(InvocationValue invocation, InvocationValue contextInvocation);

// Returns the address of the given value when it is accesses within the given program invocation
std::optional<int64_t>
getAddress(Value value, int64_t offset = 0, InvocationValue invocation = nullptr);

// Returns the address of the first byte of the given value (this is the base address plus the
// offset in the layout), the optional invocation parameter can be used to resolve the address
// within the given invocation context
std::optional<int64_t>
getDataStartAddress(Value value, int64_t offset = 0, InvocationValue invocationContext = nullptr);

// return the address of the first entry of a memref as seen by the executor, if it is accessible
std::optional<int64_t> getExecutorDataStartAddress(
    torq_hl::Executor executor, Value value, int64_t offset = 0,
    InvocationValue invocation = nullptr
);

} // namespace mlir::syna::torq
