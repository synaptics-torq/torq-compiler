// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-outline-slice-tasks"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// This pattern unrolls a scf.forall operation by replicating the loop body n-1 times.
// The forall loop is used to parallelize the execution of an operation on multiple HW slices
// so its body will be terminated by single Wait and a single Store operation, in order.
// Since by definition all the iterations are independent, it is safe to move all the Wait and Store
// operations to the end of the unrolled loop to parallelize execution without further analysis.
// Code derived from generateUnrolledLoop in mlir/lib/Dialect/SCF/Utils/Utils.cpp
class ForallOpPattern : public OpRewritePattern<scf::ForallOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(scf::ForallOp forallOp, PatternRewriter &rewriter) const override {

        // extract loop information
        llvm::ArrayRef<int64_t> upper = forallOp.getStaticUpperBound();
        assert(upper.size() == 1 && "expected single static upper bound in forall op");
        auto unrollFactor = upper[0];

        auto builder = OpBuilder(forallOp);

        // track the original operations in the body loop
        SmallVector<Operation *> originalOps;
        for (auto &op : forallOp.getBody()->getOperations()) {
            originalOps.push_back(&op);
        }

        // create an operand map for each unrolled iteration
        SmallVector<IRMapping> operandMaps(unrollFactor - 1);

        // insert in the operand map the induction variable
        if (!forallOp.getInductionVar(0).use_empty()) {
            for (unsigned i = 1; i < unrollFactor; i++) {
                Value ivUnroll =
                    rewriter.create<arith::ConstantOp>(forallOp.getLoc(), rewriter.getIndexAttr(i));
                operandMaps[i - 1].map(forallOp.getInductionVar(0), ivUnroll);
            }
        }

        SmallVector<Operation *> operationsToClone;

        // scan the original operations and accumulate operations until a wait,
        // when a wait is found create unrollFactor - 1 copies of the accumulated operations
        // and continue
        for (auto op : originalOps) {
            if (isa<torq_hl::WaitProgramOp>(op) || op->getBlock()->getTerminator() == op) {
                builder.setInsertionPoint(op);
                for (unsigned i = 1; i < unrollFactor; i++) {
                    // Clone the original loop body operations
                    for (auto prevOp : operationsToClone) {
                        builder.clone(*prevOp, operandMaps[i - 1]);
                    }
                }
                operationsToClone.clear();
            }
            operationsToClone.push_back(op);
        }

        // Loop now has a single iteration so we can remove it
        forallOp.setStaticUpperBound(1);
        if (failed(forallOp.promoteIfSingleIteration(rewriter))) {
            assert(false && "Unrolling of scf.forall failed");
        }

        return success();
    }
};

// creates a program containing the specified operation and substitute the operation with a call to
// the program
static void outlineOp(int idx, Operation *op, OpBuilder builder) {

    auto loc = op->getLoc();
    auto ctx = builder.getContext();

    // FIXME: here we should defer computing the size until we compile the Program
    // The size should be enough to store a CFG/SYN task and the required NDLs
    int size = 0xA00;

    // allocate some lram that will contain the code

    // create the program
    builder.setInsertionPoint(op);
    auto programType = torq_hl::ProgramType::get(ctx, torq_hl::Executor::Slice);
    std::string programName =
        "slice_program_" + op->getName().getStringRef().str() + "_" + std::to_string(idx);
    auto programOp =
        builder.create<torq_hl::ProgramOp>(loc, programType, builder.getStringAttr(programName));

    // create the body of the program
    Block &body = programOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&body);

    // add all the arguments to the program body
    IRMapping map;
    for (auto operand : op->getOperands()) {
        map.map(operand, body.addArgument(operand.getType(), loc));
    }

    // clone the operation into the program body mapping the operands to the new block arguments
    builder.clone(*op, map);

    // add a return operation to the program body that returns nothing
    builder.create<torq_hl::ReturnOp>(loc, ValueRange{});

    // create the invocation
    builder.setInsertionPoint(op);
    auto invocationType = torq_hl::InvocationType::get(ctx, torq_hl::Executor::Slice);
    auto programSectionType = MemRefType::get({size}, builder.getI8Type());
    auto createInvocationOp = builder.create<torq_hl::CreateInvocationOp>(
        loc, TypeRange{invocationType, programSectionType}, programOp.getName(),
        programOp.getProgram(), nullptr, nullptr, nullptr, nullptr
    );

    // move the code from xram to lram
    auto programSectionLramCodeType = MemRefType::get(
        {size}, builder.getI8Type(), nullptr,
        createDenseEncoding(programSectionType, torq_hl::MemorySpace::Lram)
    );
    auto lramCodeSection =
        builder.create<memref::AllocOp>(loc, programSectionLramCodeType, nullptr);

    if (failed(
            createTorqCopy(builder, loc, createInvocationOp.getCodeSections()[0], lramCodeSection)
        )) {
        llvm::report_fatal_error("failed to create copy to LRAM");
    }

    // add the start and wait operations
    auto startOp = builder.create<torq_hl::StartProgramOp>(
        loc,
        /* bound_program = */ createInvocationOp.getInvocation(),
        /* code_sections = */ ValueRange{lramCodeSection},
        /* args = */ op->getOperands()
    );

    builder.create<torq_hl::WaitProgramOp>(loc, TypeRange{}, startOp.getInvocation());

    op->erase();
}

static void outlineSlicePrograms(Operation *op) {

    SmallVector<Operation *> toOutline;

    op->walk([&](Operation *op) {
        if (isa<scf::ForallOp, func::FuncOp>(op)) {
            return WalkResult::advance();
        }

        // FIXME: kernels should implement an interface to simplify this logic
        if (isa<DestinationStyleOpInterface>(op) && !isa<torq_hl::CallProgramOp>(op)) {
            toOutline.push_back(op);
        }

        // do not recurse into nested operations
        return WalkResult::skip();
    });

    OpBuilder builder(op->getContext());

    for (auto [idx, op] : llvm::enumerate(toOutline)) {
        outlineOp(idx, op, builder);
    }
}

static LogicalResult unrollLoops(Operation *op) {

    RewritePatternSet unrollPatterns(op->getContext());

    unrollPatterns.add<ForallOpPattern>(op->getContext());

    return applyPatternsAndFoldGreedily(op, std::move(unrollPatterns));
}

// assign an executor_id to each start_program operation
static LogicalResult
scheduleSliceTasks(Region &region, torq_hl::Executor executor, int executorCount) {

    SmallVector<bool> sliceBusy(executorCount, false);

    for (auto &op : region.getOps()) {

        if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {

            auto invocationOp =
                startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

            if (!invocationOp) {
                return op.emitError() << "must use an invocation created by a create_invocation op";
            }

            if (!invocationOp.getExecutorId()) {

                // schedule the program on the first available slice

                auto it = llvm::find(sliceBusy, false);

                if (it == sliceBusy.end()) {
                    return op.emitError() << "all slices busy, cannot allocate a executor_id";
                }

                int availableSlice = std::distance(sliceBusy.begin(), it);

                invocationOp.setExecutorId(APInt(64, availableSlice));
                sliceBusy[availableSlice] = true;
            }
            else {

                // mark the executor being used as busy
                auto executorId = invocationOp.getExecutorId()->getZExtValue();

                if (sliceBusy[executorId]) {
                    return op.emitError() << "executor is already busy";
                }

                sliceBusy[executorId] = true;
            }
        }

        else if (auto sliceWaitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {

            auto invocationOp =
                sliceWaitOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

            if (!invocationOp) {
                return op.emitError() << "must use an invocation created by a create_invocation op";
            }

            sliceBusy[invocationOp.getExecutorId()->getZExtValue()] = false;
        }
    }

    return success();
}

class OutlineSliceTasksPass : public OutlineSliceTasksBase<OutlineSliceTasksPass> {
  public:
    OutlineSliceTasksPass() = default;
    OutlineSliceTasksPass(const OutlineSliceTasksPass &pass) {}

    void runOnOperation() override;
};

void OutlineSliceTasksPass::runOnOperation() {

    // TODO: we should divide this in three passes

    outlineSlicePrograms(getOperation());

    if (failed(unrollLoops(getOperation()))) {
        return signalPassFailure();
    }

    if (failed(scheduleSliceTasks(
            getOperation().getFunctionBody(), torq_hl::Executor::Slice, HwInfo::slice_count
        ))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineSliceProgramsPass() {
    return std::make_unique<OutlineSliceTasksPass>();
}

} // namespace mlir::syna::torq
