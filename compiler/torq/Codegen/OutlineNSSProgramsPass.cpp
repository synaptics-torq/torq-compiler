// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Codegen/OutlineUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <deque>

#define DEBUG_TYPE "torq-segment-nss-program"

static llvm::cl::opt<int> clNssTaskSize(
    "torq-nss-task-size",
    llvm::cl::desc(
        "Select the max size of NSS programs in number of operations (0 = unbounded, default)"
    ),
    llvm::cl::init(0)
);

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class OutlineNSSProgramsPass : public impl::OutlineNSSProgramsBase<OutlineNSSProgramsPass> {
  public:
    OutlineNSSProgramsPass() = default;
    OutlineNSSProgramsPass(const OutlineNSSProgramsPass &pass) {}

    void runOnOperation() override;
};

static bool isNssManagedMemory(BaseMemRefType memRefType) {
    auto memSpace = getEncodingMemorySpace(memRefType);
    return memSpace == torq_hl::MemorySpace::Lram || memSpace == torq_hl::MemorySpace::Dtcm ||
           memSpace == torq_hl::MemorySpace::Itcm;
}

static bool isNssOperation(Operation &op) {

    if (llvm::isa<torq_hl::StoreOp, torq_hl::LoadOp>(op)) {
        return true;
    }

    if (auto startOp = llvm::dyn_cast<torq_hl::StartProgramOp>(op)) {
        auto executor = startOp.getInvocation().getType().getExecutor();

        return executor == torq_hl::Executor::Slice || executor == torq_hl::Executor::CSS;
    }

    if (auto waitOp = llvm::dyn_cast<torq_hl::WaitProgramOp>(op)) {

        auto executor = waitOp.getInvocation().getType().getExecutor();

        return executor == torq_hl::Executor::Slice || executor == torq_hl::Executor::CSS;
    }
    else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        assert(false && "alloc operations should have been converted at this point");
    }
    else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        assert(false && "alloc operations should have been converted at this point");
    }
    else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        return isNssManagedMemory(copyOp.getSource().getType()) ||
               isNssManagedMemory(copyOp.getTarget().getType());
    }

    return false;
}

static bool isActionOperation(Operation &op) {

    // NSS operations are not actions
    if (isNssOperation(op)) {
        return false;
    }

    // Host actions defined in the FBS are:
    // HostCopyParams, StartNSSParams, WaitNSSParams, StartHostParams, WaitHostParams, AllocParams,
    // DeallocParams
    return isa<
        torq_hl::HostCopyOp, torq_hl::StartProgramOp, torq_hl::WaitProgramOp, memref::AllocOp,
        memref::DeallocOp, func::ReturnOp>(op);
}

static int getOperationSize(Operation *op) {

    // all these operations do not generate any bitstream instruction so
    // we can treat them as size 0 for the purpose of splitting programs
    if (isa<memref::AllocOp, memref::DeallocOp, torq_hl::GetBlockOp, memref::GetGlobalOp>(op)) {
        return 0;
    }

    if (isDerivedMemRefOperation(op)) {
        return 0;
    }

    return 1;
}

struct NssProgramOutliningInfo {
    // all NSS operations we want to outline together in the same program
    SmallVector<Operation *> nssProgramOps;

    // all outputs of the program (outputs of nssProgramOps that are used outside the program)
    SmallVector<Value> programOutputs;

    // all the inputs of the program (operands of nssProgramOps that are defined outside the program
    // and not defined by inlinedInputsOps)
    SmallVector<Value> programInputs;

    // all the operations producing inputs required by nssProgramOps that we want to inline in the
    // NSS program
    SmallVector<Operation *> inlinedInputsOps;

    // all operations we want to outline together in the same program (nssProgramOps +
    // inlinedInputsOps)
    SmallVector<Operation *> outlinedOps;
};

static SmallVector<NssProgramOutliningInfo, 0> findPrograms(FunctionOpInterface funcOp) {
    SmallVector<NssProgramOutliningInfo, 0> outliningInfo;

    bool createNewProgram = true;
    int programSize = 0;

    // the maxium size of a program is the NSS task size minus one
    // to keep space for a load operation to load the next program
    int maxProgramSize = clNssTaskSize - 1;

    if (clNssTaskSize != 0 && maxProgramSize <= 0) {
        funcOp.emitError("NSS max program size must be greater than 1 or 0 for unbounded");
        return {};
    }

    for (Operation &op : funcOp.getCallableRegion()->getOps()) {

        // if we have an operation that is not an NSS operation, we mark
        // that we need to break the current program at the next nss operation
        if (!isNssOperation(op)) {
            createNewProgram = true;
            continue;
        }

        auto operationSize = getOperationSize(&op);

        // if we need to break or the operation doesn't fit in the current program, start a new one
        if (createNewProgram ||
            (clNssTaskSize > 0 && programSize + operationSize > maxProgramSize)) {
            outliningInfo.push_back({});
            createNewProgram = false;
            programSize = 0;
        }

        programSize += operationSize;

        // add the operation to the current program
        outliningInfo.back().nssProgramOps.push_back(&op);
    }

    return outliningInfo;
}

static DenseSet<Value> getNssProgramInputs(SmallVector<Operation *> programOps) {
    // find all the values used by the program that are defined outside the program
    DenseSet<Value> programValues;
    DenseSet<Value> inputs;

    for (Operation *op : programOps) {
        op->walk([&](Operation *nestedOp) {
            for (Value result : nestedOp->getResults()) {
                programValues.insert(result);
            }

            for (Value operand : nestedOp->getOperands()) {
                if (!programValues.contains(operand)) {
                    inputs.insert(operand);
                }
            }
        });
    }

    return inputs;
}

static SetVector<Operation *>
getInlineableOperations(SmallVector<Value> inputs, DenseSet<Value> &inlineableInputs) {

    // these are the operations we can inline, in topological order
    SetVector<Operation *> inlineableOperations;
    DenseSet<Operation *> visited;

    std::deque<Operation *> worklist;

    DenseSet<Value> originalInputs(inputs.begin(), inputs.end());

    // add all defining operations of any input to the worklist to start the traversal
    for (auto input : inputs) {
        auto definingOp = input.getDefiningOp();

        // if the input doesn't have a defining operation, it means it's a block argument
        // block arguments are not inlineable, so we skip it
        if (!definingOp) {
            continue;
        }

        worklist.push_back(definingOp);
    }

    while (!worklist.empty()) {

        Operation *op = worklist.front();
        worklist.pop_front();

        if (visited.contains(op)) {
            continue;
        }

        bool inlined = false;

        // get_global operations can be inlined
        if (isa<memref::GetGlobalOp>(op)) {

            inlineableOperations.insert(op);
            inlined = true;
        }
        else if (isDerivedMemRefOperation(op)) {

            bool allVisitedOperandsInlineable = true;
            bool anyOperandNotVisited = false;

            for (Value operand : op->getOperands()) {

                auto operandDefiningOp = operand.getDefiningOp();

                if (!operandDefiningOp) {
                    // if the operand doesn't have a defining operation, it means it's a block
                    // argument block arguments are not inlineable, so we consider this operand not
                    // inlineable and break
                    allVisitedOperandsInlineable = false;

                    break;
                }
                // we don't have the info on the operand defining operation yet, so we schedule it
                // to be visited
                else if (!visited.contains(operandDefiningOp)) {
                    worklist.push_back(operandDefiningOp);
                    anyOperandNotVisited = true;
                }
                // if the operand defining operation is not inlineable, we consider this operation
                // not inlineable
                else if (!inlineableOperations.contains(operandDefiningOp)) {
                    allVisitedOperandsInlineable = false;
                    break;
                }
            }

            // if we have any operand that is not visited, we cannot consider the
            // input inlineable yet, we need to wait for visiting all its operands first
            // if we found any non inlineable operand we consider this input
            // not inlineable
            if (anyOperandNotVisited && allVisitedOperandsInlineable) {
                worklist.push_back(op);
                continue;
            }

            // if all the operands are inlineable, we can inline this operation as well
            if (allVisitedOperandsInlineable) {
                inlineableOperations.insert(op);
                inlined = true;
            }
        }

        if (inlined) {
            for (Value result : op->getResults()) {
                if (originalInputs.contains(result)) {
                    inlineableInputs.insert(result);
                }
            }
        }

        visited.insert(op);
    }

    return inlineableOperations;
}

static SmallVector<NssProgramOutliningInfo, 0> createNSSOutliningInfo(FunctionOpInterface funcOp) {

    // find the operations we want to put in different NSS programs
    auto outliningInfo = findPrograms(funcOp);

    for (auto &programInfo : outliningInfo) {

        auto outlinedOpsWithNested = getOutlinedOps(programInfo.nssProgramOps);

        // find all inputs to the program
        auto inputs = getOutlineOpsInputs(programInfo.nssProgramOps, outlinedOpsWithNested);

        // find all outputs of the program
        programInfo.programOutputs =
            getOutlineOpsOutputs(programInfo.nssProgramOps, outlinedOpsWithNested);

        // find all operations producing inputs that can be inlined
        DenseSet<Value> inlineableInputs;
        auto inlineableOperations = getInlineableOperations(inputs, inlineableInputs);
        programInfo.inlinedInputsOps.append(
            inlineableOperations.begin(), inlineableOperations.end()
        );

        // put togheter all the operation we need to outline in the same program (first the inlined
        // input operations, then the program operations to make sure all the inputs are defined
        // before being used)
        programInfo.outlinedOps.append(
            programInfo.inlinedInputsOps.begin(), programInfo.inlinedInputsOps.end()
        );
        programInfo.outlinedOps.append(
            programInfo.nssProgramOps.begin(), programInfo.nssProgramOps.end()
        );

        // collect all the inputs to the program that we couldn't inline
        for (Value input : inputs) {
            if (!inlineableInputs.contains(input)) {
                programInfo.programInputs.push_back(input);
            }
        }
    }

    return outliningInfo;
}

static FailureOr<torq_hl::StartProgramOp> outlineOperations(
    int index, OpBuilder &builder, Operation *funcOp, NssProgramOutliningInfo outliningInfo,
    Value &lramProgramAlloc, Value &lramNextProgramAlloc,
    torq_hl::StartProgramOp previousInvocationStart
) {

    auto ctx = funcOp->getContext();

    auto invocationType = torq_hl::InvocationType::get(ctx, torq_hl::Executor::NSS);

    if (previousInvocationStart) {
        // if we have a previous invocation start, we need to insert the program before its start
        // so that we will be able to copy it
        builder.setInsertionPoint(previousInvocationStart);
    }

    SmallVector<Operation *> opsToOutline;
    opsToOutline.append(
        outliningInfo.inlinedInputsOps.begin(), outliningInfo.inlinedInputsOps.end()
    );
    opsToOutline.append(outliningInfo.nssProgramOps.begin(), outliningInfo.nssProgramOps.end());

    auto maybeOutliningResults = outlineProgram(
        builder, std::string("nss_program_") + std::to_string(index), torq_hl::Executor::NSS,
        opsToOutline, /*destinationStyle=*/false, outliningInfo.programInputs,
        outliningInfo.programOutputs
    );

    if (failed(maybeOutliningResults)) {
        return failure();
    }

    auto outliningResults = maybeOutliningResults.value();

    auto programOp = outliningResults.program;

    auto &programBlock = programOp.getBody().front();

    // add two arguments to the program with the two LRAM allocations, one is the
    // program area, the other one is the next program area
    auto currentProgramArg =
        programBlock.insertArgument(0u, lramProgramAlloc.getType(), programOp.getLoc());
    auto nextProgramArg =
        programBlock.insertArgument(1u, lramNextProgramAlloc.getType(), programOp.getLoc());

    // add one argument to the program for the current invocation
    // auto currentInvocationArg =
    programBlock.insertArgument(2u, invocationType, programOp.getLoc());

    // make the program return the next program area as first result
    programBlock.getTerminator()->insertOperands(0, ValueRange{nextProgramArg, currentProgramArg});

    // Use programOp's location if known
    Location loc = programOp->getLoc();
    if (mlir::isa<mlir::UnknownLoc>(loc) && outliningInfo.nssProgramOps.empty()) {
        // fall back to func op's loc if the program op is unknown
        // This can happen for very first program which only includes the host copy
        loc = funcOp->getLoc();
    }

    auto programBlockType = MemRefType::get({HwInfo::nss_max_program_size}, builder.getI8Type());

    // create the invocation
    auto createInvocationOp = torq_hl::CreateInvocationOp::create(
        builder, loc, TypeRange{invocationType, programBlockType}, programOp.getName(),
        programOp.getProgram(), nullptr, nullptr, nullptr, nullptr
    );

    // create the load operation that will load the program into the LRAM
    if (previousInvocationStart) {

        // add the load operation at the end of the previous program
        auto previousInvocation =
            previousInvocationStart.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

        auto previousInvocationProgram =
            previousInvocation.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        auto &previousProgramBlock = previousInvocationProgram.getBody().front();

        // add a new argument to the previous program for the next program invocation
        auto nextInvocation = previousProgramBlock.addArgument(invocationType, loc);

        builder.setInsertionPoint(previousProgramBlock.getTerminator());

        // load the entry block of the next program into the LRAM area
        auto prevInvocationNextProgramAreaArg = previousProgramBlock.getArgument(1);

        auto nextInvocationBlock = torq_hl::GetBlockOp::create(
            builder, loc, programBlockType, nextInvocation, builder.getIndexAttr(0)
        );

        if (failed(
                createTorqCopy(builder, loc, nextInvocationBlock, prevInvocationNextProgramAreaArg)
            )) {
            llvm::report_fatal_error("failed to create copy to LRAM");
        }

        // add the next invocation code block to the previous invocation start to fill the argument
        previousInvocationStart.getArgsMutable().append(createInvocationOp.getInvocation());
    }
    else {
        // create the invocation and the host copy operation

        auto firstCodeBlock = torq_hl::GetBlockOp::create(
            builder, loc, programBlockType, createInvocationOp.getInvocation(),
            builder.getIndexAttr(0)
        );

        torq_hl::HostCopyOp::create(
            builder, loc, lramProgramAlloc, firstCodeBlock,
            /*inputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*outputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*shape=*/builder.getDenseI64ArrayAttr({}),
            /*elementSizeBytes=*/builder.getI64IntegerAttr(HwInfo::nss_max_program_size)
        );
    }

    if (!outliningInfo.nssProgramOps.empty()) {
        builder.setInsertionPoint(outliningInfo.nssProgramOps.front());
    }

    // add the start and wait operations

    SmallVector<Value> startInputs;

    // the program receives the two LRAM allocations as first two inputs
    startInputs.push_back(lramProgramAlloc);
    startInputs.push_back(lramNextProgramAlloc);

    // the program receives the current invocation as third input
    startInputs.push_back(createInvocationOp.getInvocation());

    for (auto input : outliningResults.inputs) {
        startInputs.push_back(input);
    }

    auto startOp = torq_hl::StartProgramOp::create(
        builder, loc,
        /* invocation = */ createInvocationOp.getInvocation(),
        /* code_sections = */ ValueRange{lramProgramAlloc},
        /* args = */ startInputs
    );

    SmallVector<Type> waitOutputTypes;

    // the first two outputs are the next and next-next program areas
    waitOutputTypes.push_back(lramNextProgramAlloc.getType());
    waitOutputTypes.push_back(lramProgramAlloc.getType());

    for (auto output : outliningResults.outputs) {
        waitOutputTypes.push_back(output.getType());
    }

    SmallVector<Value> waitOutputs;

    auto waitOp =
        torq_hl::WaitProgramOp::create(builder, loc, waitOutputTypes, startOp.getInvocation());

    // update the next and current LRAM allocations for the next program
    lramProgramAlloc = waitOp.getOutputs()[0];
    lramNextProgramAlloc = waitOp.getOutputs()[1];

    // replace all usages of the outputs with the results of the wait operation
    for (auto [output, result] :
         llvm::zip(outliningResults.outputs, waitOp.getOutputs().drop_front(2))) {
        output.replaceAllUsesWith(result);
    }

    // erase the original operations that have been outlined
    for (auto op : llvm::reverse(outliningInfo.nssProgramOps)) {
        op->erase();
    }

    return startOp;
}

static SmallVector<Value> getLramCodeAreas(FunctionOpInterface funcOp, OpBuilder &builder) {

    builder.setInsertionPointToStart(&(funcOp.getCallableRegion()->front()));

    SmallVector<Value> values;

    // get the globals mmeref that will be used to load the current and next program
    // these have been created in the create globals pass
    for (int i = 0; i < 2; i++) {
        std::string globalName = (llvm::Twine("__program_slot") + llvm::Twine(i)).str();

        auto globalNameAttr = builder.getStringAttr(globalName);
        auto slotOp =
            SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(funcOp, globalNameAttr);
        auto getSlotOp =
            memref::GetGlobalOp::create(builder, funcOp.getLoc(), slotOp.getType(), globalNameAttr);

        values.push_back(getSlotOp.getResult());
    }

    return values;
}

static void moveDeclarationsToBeginning(FunctionOpInterface funcOp, OpBuilder &builder) {

    // Collect all get_global operations for host managed memory and move them to the beginning of
    // the function
    SmallVector<Operation *> nonActionOps;
    for (auto &op : funcOp.getCallableRegion()->getOps()) {
        if (!isNssOperation(op) && !isActionOperation(op)) {
            nonActionOps.push_back(&op);
        }
    }

    // Move the collected non-action operations to the beginning of the function
    auto &firstBlock = funcOp.getCallableRegion()->front();
    for (auto op : llvm::reverse(nonActionOps)) {
        op->moveBefore(&firstBlock, firstBlock.begin());
    }
}

void OutlineNSSProgramsPass::runOnOperation() {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();
    OpBuilder builder(ctx);

    // move all operations that are not directly actions
    // to be taken by either host or NSS to the beginning of the function
    // so that we don't introduce unwanted NSS program breaks when
    // outlining the programs. By definition all operations that
    // are not actions and are not NSS operations will not be executed by the
    // runtime nor NSS and therefore we can freely move them without changing
    // the semantics of the program (we need to keep their relative order to not
    // break dependencies between them)
    //
    // This includes memref.get_global operations, slice descriptors and
    // programs, memref.subview and other view-like operations on
    // global allocations and constants.
    moveDeclarationsToBeginning(funcOp, builder);

    // find operations we want to outline into NSS programs
    auto nssProgramsOps = createNSSOutliningInfo(funcOp);

    // add an empty program at the beginning that will be
    // uploaded via host_copy and will be used to DMA the
    // first program (which is more efficient)
    if (!nssProgramsOps.empty()) {
        nssProgramsOps.insert(nssProgramsOps.begin(), NssProgramOutliningInfo{});
    }

    // create the two allocations used to load programs (current program and next program)
    auto programAllocs = getLramCodeAreas(funcOp, builder);

    // outline the nss programs
    torq_hl::StartProgramOp previousStartProgram;

    Value currentLramBlock = programAllocs[0];
    Value nextLramBlock = programAllocs[1];

    for (auto [idx, nssProgramOps] : llvm::enumerate(nssProgramsOps)) {
        auto maybePreviousStart = outlineOperations(
            idx, builder, funcOp, nssProgramOps, currentLramBlock, nextLramBlock,
            previousStartProgram
        );
        if (failed(maybePreviousStart)) {
            return signalPassFailure();
        }
        previousStartProgram = maybePreviousStart.value();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineNSSProgramsPass() {
    return std::make_unique<OutlineNSSProgramsPass>();
}

} // namespace mlir::syna::torq
