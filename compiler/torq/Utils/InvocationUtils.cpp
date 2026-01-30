#include "torq/Utils/InvocationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
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

            // we don't go through descriptors based invocations
            if (isa<torq_hl::DescriptorOp>(startedInvocation.getDefiningOp())) {
                return success();
            }

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

            // we don't go through descriptors based invocations
            if (isa<torq_hl::DescriptorOp>(waitedInvocation.getDefiningOp())) {
                return success();
            }

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
            else if (auto nextOp = dyn_cast<torq_hl::NextOp>(op)) {

                Block *targetBlock = nextOp.getDest();

                IRMapping nextMapping;

                for (int i = 0; i < targetBlock->getNumArguments(); ++i) {
                    nextMapping.map(
                        targetBlock->getArgument(i), mapping.lookup(nextOp.getArguments()[i])
                    );
                }

                auto maybeReturnedValues =
                    processBlock(targetBlock, currentInvocation, nextMapping);

                if (failed(maybeReturnedValues)) {
                    return failure();
                }

                return *maybeReturnedValues;
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

static Value simplifyBlockArguments(Value value) {

    auto blockArg = dyn_cast<BlockArgument>(value);

    // not a block argument, nothing to simplify
    if (!blockArg) {
        return value;
    }

    // if the block argument is from the entry block, we cannot simplify it further
    if (blockArg.getOwner() == &blockArg.getOwner()->getParent()->front()) {
        return value;
    }

    // find the operation that jumps to this block
    auto prevBlock = blockArg.getOwner()->getPrevNode();

    auto nextOp = dyn_cast<torq_hl::NextOp>(prevBlock->getTerminator());

    if (!nextOp) {
        return nullptr;
    }

    if (nextOp.getArguments().size() <= blockArg.getArgNumber()) {
        return nullptr;
    }

    // simplify the argument passed to the block
    return simplifyBlockArguments(nextOp.getArguments()[blockArg.getArgNumber()]);
}

std::optional<int64_t>
getExecutorId(InvocationValue invocation, InvocationValue contextInvocation) {

    invocation = dyn_cast<InvocationValue>(simplifyBlockArguments(invocation));

    if (!invocation) {
        return std::nullopt;
    }

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

        auto maybeArgs = contextInvocationOp.getInvocationArgs();

        if (!maybeArgs || blockArg.getArgNumber() >= maybeArgs->size()) {
            return std::nullopt;
        }

        auto args = *maybeArgs;

        auto invocationAttr = dyn_cast<torq_hl::InvocationAttr>(args[blockArg.getArgNumber()]);

        if (!invocationAttr) {
            return std::nullopt;
        }

        return invocationAttr.getExecutorId();
    }
    else if (auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>()) {

        if (!createInvocationOp.getExecutorId()) {
            return std::nullopt; // no executor id available
        }

        return createInvocationOp.getExecutorId()->getZExtValue();
    }

    return std::nullopt;
}

static Attribute getInvocationArgumentAttr(BlockArgument blockArg, InvocationValue invocation) {
    if (!invocation) {
        return nullptr; // no invocation provided
    }

    auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>();

    if (!createInvocationOp) {
        return nullptr; // not an invocation argument
    }

    if (blockArg.getOwner()->getParentOp() != createInvocationOp.getProgram().getDefiningOp()) {
        return nullptr; // this is an argument that is not an argument of the invoked
                        // program
    }

    auto args = createInvocationOp.getInvocationArgs();

    if (!args) {
        return nullptr; // no addresses available
    }

    if (blockArg.getArgNumber() >= args->size()) {
        return nullptr; // argument index out of bounds
    }

    return (*args)[blockArg.getArgNumber()];
}

std::optional<int64_t>
getAddressFromInvocationArg(BlockArgument blockArg, int64_t offset, InvocationValue invocation) {

    if (invocation.getType().getExecutor() == torq_hl::Executor::CSS) {
        // if the executor is CSS the addresses we get are remapped into a single address space
        // so we can't use them directly, this is not useful in practice anyways
        return std::nullopt;
    }

    auto argAttr = getInvocationArgumentAttr(blockArg, invocation);

    if (!argAttr) {
        return std::nullopt; // no argument attribute found
    }

    auto argBuffer = dyn_cast<torq_hl::BufferAttr>(argAttr);

    if (!argBuffer) {
        return std::nullopt; // argument is not an address
    }

    // the address in the invocation arguments is a start address, not a base address
    auto type = argBuffer.getMemrefType();
    auto baseAddress = argBuffer.getAddress() - getMemRefTypeOffsetBytes(type);

    return baseAddress + offset;
}

static std::optional<int64_t> getAddressFromGetBlockOp(
    torq_hl::GetBlockOp getBlockOp, int64_t offset, InvocationValue invocation
) {

    auto blockInvocation = simplifyBlockArguments(getBlockOp.getInvocation());

    // the invocation in the get_block op is an argument of the current invocation
    if (auto blockArg = dyn_cast<BlockArgument>(blockInvocation)) {

        auto argAttr = getInvocationArgumentAttr(blockArg, invocation);

        if (!argAttr) {
            return std::nullopt; // no argument attribute found
        }

        auto argInvocation = dyn_cast<torq_hl::InvocationAttr>(argAttr);

        if (!argInvocation) {
            return std::nullopt; // argument is not an invocation
        }

        return argInvocation.getBlockAddresses()[getBlockOp.getBlockIndex().getZExtValue()] +
               offset;
    }
    // the invocation in the get_block op is value produced by a create_invocation op
    else if (auto blockCreateInvocationOp =
                 blockInvocation.getDefiningOp<torq_hl::CreateInvocationOp>()) {

        auto xramCodeAddresses = blockCreateInvocationOp.getXramCodeAddresses();

        if (!xramCodeAddresses) {
            return std::nullopt; // no addresses available
        }

        if (getBlockOp.getBlockIndex().getZExtValue() >= xramCodeAddresses->size()) {
            return std::nullopt; // block index out of bounds
        }

        return (*xramCodeAddresses)[getBlockOp.getBlockIndex().getZExtValue()] + offset;
    }
    else {
        return std::nullopt; // we don't support any other case
    }
}

std::optional<int64_t> getAddress(Value value, int64_t offset, InvocationValue invocation) {

    // the value may be a block argument, if this is not the first block
    // we need to find how this argument was set (from the next op) and
    // find either a value or block arg of the entry block
    value = simplifyBlockArguments(value);

    if (!value) {
        return std::nullopt;
    }

    // special cases where the value is coming from a block argument or a get_block op
    // in this case the value doesn't carry the address information directly
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        return getAddressFromInvocationArg(blockArg, offset, invocation);
    }
    else if (auto getBlockOp = dyn_cast<torq_hl::GetBlockOp>(value.getDefiningOp())) {
        return getAddressFromGetBlockOp(getBlockOp, offset, invocation);
    }

    auto memRefType = dyn_cast<MemRefType>(value.getType());

    if (!memRefType) {
        return std::nullopt; // not a memref type
    }

    auto memorySpace = getEncodingMemorySpace(memRefType);

    switch (memorySpace) {
    case torq_hl::MemorySpace::Lram:
        return getLramAddress(value, offset);
    case torq_hl::MemorySpace::Dtcm:
        return getDtcmAddress(value, offset);
    case torq_hl::MemorySpace::Itcm:
        return getItcmAddress(value, offset);
    case torq_hl::MemorySpace::Xram:
        return getXramAddress(value, offset);
    default:
        return std::nullopt; // unsupported memory space
    }
}

std::optional<int64_t>
getDataStartAddress(Value value, int64_t offset, InvocationValue invocation) {

    MemRefType type = cast<MemRefType>(value.getType());
    return getAddress(value, offset + getMemRefTypeOffsetBytes(type), invocation);
}

static std::optional<int64_t>
getCssAddress(Value value, int64_t offset, TypedValue<torq_hl::InvocationType> invocation) {
    auto memrefType = dyn_cast<MemRefType>(value.getType());

    if (!memrefType) {
        return std::nullopt;
    }

    auto memSpace = getEncodingMemorySpace(memrefType);

    std::optional<int64_t> addr = getDataStartAddress(value, offset, invocation);

    int64_t baseAddress = 0;

    switch (memSpace) {
    case torq_hl::MemorySpace::Dtcm:
        baseAddress = HwInfo::css_dtcm_base_address;
        break;
    case torq_hl::MemorySpace::Itcm:
        baseAddress = HwInfo::css_itcm_base_address;
        break;
    default:
        return std::nullopt;
    }

    if (!addr) {
        return std::nullopt;
    }

    return baseAddress + addr.value();
}

std::optional<int64_t> getExecutorDataStartAddress(
    torq_hl::Executor executor, Value value, int64_t offset, InvocationValue invocation
) {

    auto type = cast<MemRefType>(value.getType());

    switch (executor) {
    case torq_hl::Executor::CSS:
        return getCssAddress(value, offset, invocation);

    case torq_hl::Executor::Host:
        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
            return std::nullopt;
        }
        return getDataStartAddress(value, offset, invocation);

    case torq_hl::Executor::Slice:
        if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Lram) {
            return std::nullopt;
        }
        return getDataStartAddress(value, offset, invocation);

    case torq_hl::Executor::NSS:
        return getDataStartAddress(value, offset, invocation);

    default:
        llvm::report_fatal_error("unsupported executor");
    }
}

} // namespace mlir::syna::torq