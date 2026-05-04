// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-outline-slice-programs"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// creates a program containing the specified operation and substitute the operation with a call to
// the program
static void outlineOp(int idx, Operation *op, OpBuilder builder) {

    auto loc = op->getLoc();
    auto ctx = builder.getContext();

    // FIXME: here we should defer computing the size until we compile the Program
    // The size should be enough to store a CFG/SYN task and the required NDLs
    // Note: this has to be kept in sync with the size in getProgramSize()
    int size = 0xA00 * 2;

    // allocate some lram that will contain the code

    // create the program
    builder.setInsertionPoint(op);
    auto programType = torq_hl::ProgramType::get(ctx, torq_hl::Executor::Slice);
    std::string programName =
        "slice_program_" + op->getName().getStringRef().str() + "_" + std::to_string(idx);
    auto programOp = torq_hl::ProgramOp::create(
        builder, loc, programType, builder.getStringAttr(programName), nullptr
    );

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
    torq_hl::ReturnOp::create(builder, loc, ValueRange{});

    // create the invocation
    builder.setInsertionPoint(op);
    auto invocationType = torq_hl::InvocationType::get(ctx, torq_hl::Executor::Slice);
    auto programSectionType = MemRefType::get({size}, builder.getI8Type());
    auto createInvocationOp = torq_hl::CreateInvocationOp::create(
        builder, loc, TypeRange{invocationType, programSectionType}, programOp.getName(),
        programOp.getProgram(), nullptr, nullptr, nullptr, nullptr
    );

    // move the code from xram to lram
    auto programSectionLramCodeType = MemRefType::get(
        {size}, builder.getI8Type(), nullptr,
        createDenseEncoding(programSectionType, torq_hl::MemorySpace::Lram)
    );
    auto lramCodeSection =
        memref::AllocOp::create(builder, loc, programSectionLramCodeType, nullptr);

    if (failed(
            createTorqCopy(builder, loc, createInvocationOp.getCodeSections()[0], lramCodeSection)
        )) {
        llvm::report_fatal_error("failed to create copy to LRAM");
    }

    // add the start and wait operations
    auto startOp = torq_hl::StartProgramOp::create(
        builder, loc,
        /* bound_program = */ createInvocationOp.getInvocation(),
        /* code_sections = */ ValueRange{lramCodeSection},
        /* args = */ op->getOperands()
    );

    torq_hl::WaitProgramOp::create(builder, loc, TypeRange{}, startOp.getInvocation());

    op->erase();
}

static void outlineSlicePrograms(Operation *op) {

    SmallVector<Operation *> toOutline;

    op->walk([&](Operation *op) {
        if (isa<torq_hl::KernelInterface>(op)) {
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

class OutlineSliceProgramsPass : public impl::OutlineSliceProgramsBase<OutlineSliceProgramsPass> {
  public:
    OutlineSliceProgramsPass() = default;
    OutlineSliceProgramsPass(const OutlineSliceProgramsPass &pass) {}

    void runOnOperation() override;
};

void OutlineSliceProgramsPass::runOnOperation() { outlineSlicePrograms(getOperation()); }

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createOutlineSliceProgramsPass() {
    return std::make_unique<OutlineSliceProgramsPass>();
}

} // namespace mlir::syna::torq
