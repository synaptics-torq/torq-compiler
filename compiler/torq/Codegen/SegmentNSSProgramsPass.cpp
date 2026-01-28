// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-segment-nss-programs"

namespace mlir::syna::torq {

static llvm::cl::opt<int> clNssBlockSize(
    "torq-nss-block-size",
    llvm::cl::desc("Select the max size of NSS program block in number of operations"),
    llvm::cl::init(4)
);

namespace {

class SegmentNSSProgramsPass : public SegmentNSSProgramsBase<SegmentNSSProgramsPass> {
  public:
    using SegmentNSSProgramsBase::SegmentNSSProgramsBase;
    void runOnOperation() override;
};

static int getOperationSize(Operation *op) {

    // some operations are not serialized into the NSS program
    if (isa<memref::AllocOp, memref::DeallocOp, torq_hl::ReturnOp>(op)) {
        return 0;
    }

    return 1;
}

static SmallVector<Operation *> computeSplitPoints(Block &block) {

    SmallVector<Operation *> splitPoints;

    int programSize = 0;

    for (auto &op : block.getOperations()) {

        auto operationSize = getOperationSize(&op);

        if (programSize + operationSize > clNssBlockSize) {
            splitPoints.push_back(&op);
            programSize = 0;
        }

        programSize += operationSize;
    }

    return splitPoints;
}

// Helper to compute live-ins for a block: values used in the block but defined elsewhere.
static void appendLiveIns(Block *blk, SmallVectorImpl<Value> &liveIns) {
    DenseSet<Value> seen;

    for (auto val : liveIns) {
        seen.insert(val);
    }

    for (Operation &innerOp : blk->getOperations()) {
        for (Value operand : innerOp.getOperands()) {
            // If operand is defined outside this block or is a block argument of a different block,
            // it is a live-in.
            bool definedHere = false;
            if (auto defOp = operand.getDefiningOp()) {
                definedHere = (defOp->getBlock() == blk);
            }
            else if (auto barg = dyn_cast<BlockArgument>(operand)) {
                definedHere = (barg.getOwner() == blk);
            }

            if (!definedHere && !seen.contains(operand)) {
                seen.insert(operand);
                liveIns.push_back(operand);
            }
        }
    }
}

static int64_t getBlockSize(Block *blk) {
    // FIXME: we should compute the actual size to reduce the amount of data we DMA
    return HwInfo::nss_max_program_size;
}

static LogicalResult splitProgram(torq_hl::ProgramOp programOp) {

    // Do not re-split an already splitted program
    if (programOp.getBody().getBlocks().size() != 1) {
        return success();
    }

    // Split the single region into blocks of at most 4 operations.
    // Ensure each block has the required arguments such that values used
    // within the block are available via block arguments and forwarded
    // through branches.
    auto &region = programOp.getBody();

    if (region.empty())
        return success();

    Block &entry = region.front();

    // Compute places where we need to split the original block
    auto splitPoints = computeSplitPoints(entry);

    if (splitPoints.empty()) {
        return success();
    }

    Location loc = programOp.getLoc();
    Block *currentBlock = &entry;

    auto activeLramCodeArea = programOp.getBody().getArgument(0);
    auto inactiveLramCodeArea = programOp.getBody().getArgument(1);

    // Find pre-fetching of next program
    torq_hl::ReturnOp originalTerminator;
    torq_hl::LoadOp originalLoadNextProgramOp;

    for (auto &use : inactiveLramCodeArea.getUses()) {
        if (auto retOp = dyn_cast<torq_hl::ReturnOp>(use.getOwner())) {
            if (originalTerminator) {
                return programOp.emitOpError("multiple returns using inactive LRAM code area");
            }

            originalTerminator = retOp;
        }
        else if (auto loadOp = dyn_cast<torq_hl::LoadOp>(use.getOwner())) {

            if (originalLoadNextProgramOp) {
                return programOp.emitOpError(
                    "multiple loads of next program using inactive LRAM code area"
                );
            }

            originalLoadNextProgramOp = loadOp;
        }
        else {
            return programOp.emitOpError("unexpected use of inactive LRAM code area");
        }
    }

    SmallVector<Value> codeBlockArgs;

    OpBuilder builder(programOp);
    auto programBlockType = MemRefType::get({HwInfo::nss_max_program_size}, builder.getI8Type());

    // create block arguments for all the segments of the program
    for (int i = 0; i < splitPoints.size(); ++i) {
        codeBlockArgs.push_back(currentBlock->addArgument(programBlockType, loc));
    }

    for (auto splitPoint : splitPoints) {

        // Split the current block at the boundary; trailing ops (including terminator if present)
        // move to the new block.
        Block *newBlock = currentBlock->splitBlock(Block::iterator(splitPoint));

        // Compute the arguments needed by the new block.
        SmallVector<Value> blockArgs;

        // first live-ins are always the LRAM code areas
        blockArgs.push_back(inactiveLramCodeArea);
        blockArgs.push_back(activeLramCodeArea);

        appendLiveIns(newBlock, blockArgs);

        // Add block arguments to the new block for each live-in and replace uses within newBlock.
        for (Value v : blockArgs) {
            auto arg = newBlock->addArgument(v.getType(), loc);
            v.replaceUsesWithIf(arg, [&](OpOperand &useOp) {
                return useOp.getOwner()->getBlock() == newBlock;
            });
        }

        // add block arguments for all the remaining code sections
        SmallVector<Value> newCodeBlockArgs;

        for (int i = 1; i < codeBlockArgs.size(); ++i) {
            blockArgs.push_back(codeBlockArgs[i]);
            newCodeBlockArgs.push_back(newBlock->addArgument(programBlockType, loc));
        }

        OpBuilder builder(programOp);
        builder.setInsertionPointToEnd(currentBlock);

        // Load the next block into the inactive LRAM area
        if (failed(createTorqCopy(builder, loc, codeBlockArgs[0], inactiveLramCodeArea))) {
            return programOp.emitOpError("failed to create copy to LRAM");
        }

        // Terminate the current block with a branch to the new block, passing live-ins.
        builder.create<torq_hl::NextOp>(loc, blockArgs, inactiveLramCodeArea, newBlock);

        // Advance to the newly created block and continue.
        currentBlock = newBlock;

        // Update the active/inactive LRAM code area arguments for the next block
        activeLramCodeArea = newBlock->getArgument(0);
        inactiveLramCodeArea = newBlock->getArgument(1);

        // Update the code block arguments for the next block
        codeBlockArgs = newCodeBlockArgs;
    }

    // Fixup the original load used by load of the next program to use the final inactive LRAM area.
    if (originalLoadNextProgramOp) {
        originalLoadNextProgramOp.getOutputMutable().set(inactiveLramCodeArea);
    }

    // Fixup the original terminator to use the final inactive LRAM area.
    originalTerminator.getOperation()->setOperand(0, inactiveLramCodeArea);
    originalTerminator.getOperation()->setOperand(1, activeLramCodeArea);

    // change the output of the create_invocation to return all the sections of the code
    auto createInvocationOp =
        dyn_cast<torq_hl::CreateInvocationOp>(programOp.getProgram().getUses().begin()->getOwner());

    // return all the new code sections as result of the create_invocation
    SmallVector<Type> resultTypes;
    for (auto type : createInvocationOp.getResultTypes()) {
        resultTypes.push_back(type);
    }

    for (int i = 0; i < splitPoints.size(); i++) {
        resultTypes.push_back(programBlockType);
    }

    builder.setInsertionPoint(createInvocationOp);

    auto newCreateInvocationOp = builder.create<torq_hl::CreateInvocationOp>(
        createInvocationOp.getLoc(), resultTypes, createInvocationOp->getOperands(),
        createInvocationOp->getAttrs()
    );

    for (auto [oldResult, newResult] :
         llvm::zip(createInvocationOp.getResults(), newCreateInvocationOp.getResults())) {
        oldResult.replaceAllUsesWith(newResult);
    }

    createInvocationOp.erase();

    // pass the code sections to the start_program operation
    torq_hl::StartProgramOp startProgramOp;
    for (auto &use : newCreateInvocationOp.getInvocation().getUses()) {
        if (auto spOp = dyn_cast<torq_hl::StartProgramOp>(use.getOwner())) {
            startProgramOp = spOp;
            break;
        }
    }

    if (!startProgramOp) {
        return newCreateInvocationOp.emitOpError(
            "could not find start_program operation for segmented program invocation"
        );
    }

    for (auto codeSections : newCreateInvocationOp.getCodeSections().drop_front(1)) {
        startProgramOp.getArgsMutable().append(codeSections);
    }

    // add all the code areas that will be used to execute the program
    int blockCount = region.getBlocks().size();
    for (int i = 1; i < blockCount; ++i) {
        startProgramOp.getCodeSectionsMutable().append(startProgramOp.getArgs()[i % 2]);
    }

    return success();
}

LogicalResult processProgramOp(torq_hl::ProgramOp programOp) {

    // Split the program into smaller blocks, if necessary
    if (failed(splitProgram(programOp))) {
        return failure();
    }

    // Annotate the program with the block sizes and offsets

    Builder builder(programOp.getContext());

    auto blockCount = programOp.getBody().getBlocks().size();
    SmallVector<int32_t> blockSize(blockCount, HwInfo::nss_max_program_size);

    programOp->setAttr("torq_block_size", builder.getI32ArrayAttr(blockSize));

    return success();
}

void SegmentNSSProgramsPass::runOnOperation() {
    auto funcOp = getOperation();

    SmallVector<torq_hl::ProgramOp> programOps;

    for (auto &op : funcOp.getFunctionBody().getOps()) {
        if (auto programOp = dyn_cast<torq_hl::ProgramOp>(op)) {

            if (programOp.getProgram().getType().getExecutor() != torq_hl::Executor::NSS) {
                continue;
            }

            programOps.push_back(programOp);
        }
    }

    for (auto programOp : programOps) {
        if (failed(processProgramOp(programOp))) {
            return signalPassFailure();
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSegmentNSSProgramsPass() {
    return std::make_unique<SegmentNSSProgramsPass>();
}

} // namespace mlir::syna::torq
