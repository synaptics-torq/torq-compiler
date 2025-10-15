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
    llvm::cl::desc("Select the max size of NSS programs in number of operations"), llvm::cl::init(4)
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

    if (maxProgramSize <= 0) {
        funcOp.emitError("NSS max program size must be greater than 1");
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
        if (createNewProgram || programSize + operationSize > maxProgramSize) {
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

static int64_t getProgramSize(torq_hl::ProgramOp programOp) {
    // FIXME: we should compute the actual size to reduce the amount of data we DMA
    return HwInfo::nss_max_program_size;
}

static FailureOr<torq_hl::StartProgramOp> outlineOperations(
    int index, OpBuilder &builder, Operation *funcOp, SmallVector<Operation *> &programOps,
    Value lramProgramAlloc, torq_hl::StartProgramOp previousInvocationStart,
    int64_t xramProgramAddress
) {

    auto ctx = funcOp->getContext();
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

    // Use programOp's location if known
    Location loc = programOp->getLoc();
    if (mlir::isa<mlir::UnknownLoc>(loc) && programOps.empty()) {
        // fall back to func op's loc if the program op is unknown
        // This can happen for very first program which only includes the host copy
        loc = funcOp->getLoc();
    }
    // create the invocation
    auto invocationType = torq_hl::InvocationType::get(ctx, torq_hl::Executor::NSS);
    auto programSize = getProgramSize(programOp);
    auto programSectionType = MemRefType::get({programSize}, builder.getI8Type());
    auto codeSectionAddresses = builder.getDenseI64ArrayAttr({xramProgramAddress});
    auto createInvocationOp = builder.create<torq_hl::CreateInvocationOp>(
        loc, TypeRange{invocationType, programSectionType}, programOp.getName(),
        programOp.getProgram(), nullptr, nullptr, codeSectionAddresses, nullptr
    );

    auto xramProgram = createInvocationOp.getCodeSections()[0];

    // create the load operation that will load the program into the LRAM
    if (previousInvocationStart) {

        // add the arguments to the previous invocation start
        previousInvocationStart.getArgsMutable().append(lramProgramAlloc);
        previousInvocationStart.getArgsMutable().append(xramProgram);

        // add the load operation at the end of the previous program
        auto previousInvocation =
            previousInvocationStart.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();
        auto previousInvocationProgram =
            previousInvocation.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        // add an argument to the block that hosts the load
        auto &previousProgramBlock = previousInvocationProgram.getBody().front();
        auto dstArg = previousProgramBlock.addArgument(lramProgramAlloc.getType(), loc);
        auto srcArg = previousProgramBlock.addArgument(programSectionType, loc);

        // add the load operation
        builder.setInsertionPoint(previousProgramBlock.getTerminator());

        if (failed(createTorqCopy(builder, loc, srcArg, dstArg))) {
            llvm::report_fatal_error("failed to create copy to LRAM");
        }
    }
    else {
        // create the invocation and the host copy operation

        // to be efficient the copy operation copies a scalar of size programSize
        // this is ok because we know input and output memrefs are contiguous
        builder.create<torq_hl::HostCopyOp>(
            loc, lramProgramAlloc, xramProgram,
            /*inputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*outputStridesBytes=*/builder.getDenseI64ArrayAttr({}),
            /*shape=*/builder.getDenseI64ArrayAttr({}),
            /*elementSizeBytes=*/builder.getI64IntegerAttr(programSize)
        );
    }

    if (!programOps.empty()) {
        builder.setInsertionPoint(programOps.front());
    }

    // add the start and wait operations
    auto startOp = builder.create<torq_hl::StartProgramOp>(
        loc,
        /* invocation = */ createInvocationOp.getInvocation(),
        /* code_sections = */ ValueRange{lramProgramAlloc},
        /* args = */ outliningResults.inputs
    );

    SmallVector<Type> outputTypes;
    for (auto output : outliningResults.outputs) {
        outputTypes.push_back(output.getType());
    }

    auto waitOp = builder.create<torq_hl::WaitProgramOp>(loc, outputTypes, startOp.getInvocation());

    // replace all usages of the outputs with the results of the wait operation
    for (auto [output, result] : llvm::zip(outliningResults.outputs, waitOp.getOutputs())) {
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

    // reserve the XRAM area for the code sections
    auto startAddress =
        reserveXramArea(funcOp, HwInfo::nss_max_program_size * nssProgramsOps.size());

    // outline the nss programs
    torq_hl::StartProgramOp previousStartProgram;
    for (auto [idx, nssProgramOps] : llvm::enumerate(nssProgramsOps)) {
        auto maybePreviousStart = outlineOperations(
            idx, builder, funcOp, nssProgramOps, programAllocs[idx % 2], previousStartProgram,
            startAddress
        );
        if (failed(maybePreviousStart)) {
            return signalPassFailure();
        }
        previousStartProgram = maybePreviousStart.value();
        startAddress += HwInfo::nss_max_program_size;
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineNSSProgramsPass() {
    return std::make_unique<OutlineNSSProgramsPass>();
}

} // namespace mlir::syna::torq
