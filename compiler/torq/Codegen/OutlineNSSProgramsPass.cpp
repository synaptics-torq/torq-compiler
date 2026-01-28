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

class OutlineNSSProgramsPass : public OutlineNSSProgramsBase<OutlineNSSProgramsPass> {
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
    else if (isDerivedMemRefOperation(&op)) {
        auto memRefType = mlir::cast<MemRefType>(op.getResult(0).getType());
        return isNssManagedMemory(memRefType);
    }
    else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        return isNssManagedMemory(allocOp.getResult().getType());
    }
    else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        return isNssManagedMemory(deallocOp.getMemref().getType());
    }
    else if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        return isNssManagedMemory(copyOp.getSource().getType()) ||
               isNssManagedMemory(copyOp.getTarget().getType());
    }

    return false;
}

static int getOperationSize(Operation *op) {

    // allocations and deallocations do not serialize
    if (isa<memref::AllocOp, memref::DeallocOp>(op)) {
        return 0;
    }

    return 1;
}

static SmallVector<SmallVector<Operation *>> findPrograms(FunctionOpInterface funcOp) {
    SmallVector<SmallVector<Operation *>> nssProgramsOps;

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
            nssProgramsOps.push_back({});
            createNewProgram = false;
            programSize = 0;
        }

        programSize += operationSize;

        // add the operation to the current program
        nssProgramsOps.back().push_back(&op);
    }

    return nssProgramsOps;
}

static FailureOr<torq_hl::StartProgramOp> outlineOperations(
    int index, OpBuilder &builder, Operation *funcOp, SmallVector<Operation *> &programOps,
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

    auto maybeOutliningResults = outlineProgram(
        builder, std::string("nss_program_") + std::to_string(index), torq_hl::Executor::NSS,
        programOps, /*destinationStyle=*/false
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
    if (mlir::isa<mlir::UnknownLoc>(loc) && programOps.empty()) {
        // fall back to func op's loc if the program op is unknown
        // This can happen for very first program which only includes the host copy
        loc = funcOp->getLoc();
    }

    auto programBlockType = MemRefType::get({HwInfo::nss_max_program_size}, builder.getI8Type());

    // create the invocation
    auto createInvocationOp = builder.create<torq_hl::CreateInvocationOp>(
        loc, TypeRange{invocationType, programBlockType}, programOp.getName(),
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

        auto nextInvocationBlock = builder.create<torq_hl::GetBlockOp>(
            loc, programBlockType, nextInvocation, builder.getIndexAttr(0)
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

        auto firstCodeBlock = builder.create<torq_hl::GetBlockOp>(
            loc, programBlockType, createInvocationOp.getInvocation(), builder.getIndexAttr(0)
        );

        builder.create<torq_hl::HostCopyOp>(
            loc, lramProgramAlloc, firstCodeBlock,
            /*inputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*outputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*shape=*/builder.getDenseI64ArrayAttr({}),
            /*elementSizeBytes=*/builder.getI64IntegerAttr(HwInfo::nss_max_program_size)
        );
    }

    if (!programOps.empty()) {
        builder.setInsertionPoint(programOps.front());
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

    auto startOp = builder.create<torq_hl::StartProgramOp>(
        loc,
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
        builder.create<torq_hl::WaitProgramOp>(loc, waitOutputTypes, startOp.getInvocation());

    // update the next and current LRAM allocations for the next program
    lramProgramAlloc = waitOp.getOutputs()[0];
    lramNextProgramAlloc = waitOp.getOutputs()[1];

    // replace all usages of the outputs with the results of the wait operation
    for (auto [output, result] :
         llvm::zip(outliningResults.outputs, waitOp.getOutputs().drop_front(2))) {
        output.replaceAllUsesWith(result);
    }

    // erase the original operations that have been outlined
    for (auto op : llvm::reverse(programOps)) {
        op->erase();
    }

    return startOp;
}

static SmallVector<Value> createLramCodeAreas(FunctionOpInterface funcOp, OpBuilder &builder) {

    auto programSectionType = MemRefType::get({HwInfo::nss_max_program_size}, builder.getI8Type());
    auto lramProgramSectionType =
        createMemRefTypeWithMemorySpace(programSectionType, torq_hl::MemorySpace::Lram);

    builder.setInsertionPointToStart(&(funcOp.getCallableRegion()->front()));

    SmallVector<Value> values;

    // allocate two memrefs that will be used to load the current and next program
    // the address has been reserved during the address allocation phase
    for (int i = 0; i < 2; i++) {
        auto allocOp = builder.create<memref::AllocOp>(funcOp.getLoc(), lramProgramSectionType);
        setLramAddress(allocOp, i * HwInfo::nss_max_program_size);
        values.push_back(allocOp.getResult());
    }

    return values;
}

void OutlineNSSProgramsPass::runOnOperation() {
    auto funcOp = getOperation();
    auto ctx = funcOp.getContext();
    OpBuilder builder(ctx);

    // find operations we want to outline into NSS programs
    auto nssProgramsOps = findPrograms(funcOp);

    // add an empty program at the beginning that will be
    // uploaded via host_copy and will be used to DMA the
    // first program (which is more efficient)
    if (!nssProgramsOps.empty()) {
        nssProgramsOps.insert(nssProgramsOps.begin(), SmallVector<Operation *>{});
    }

    // create the two allocations used to load programs (current program and next program)
    auto programAllocs = createLramCodeAreas(funcOp, builder);

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
