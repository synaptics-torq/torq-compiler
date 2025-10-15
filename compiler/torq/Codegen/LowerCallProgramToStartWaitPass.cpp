// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-map-bindings"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class CallProgramPattern : public OpRewritePattern<torq_hl::CallProgramOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hl::CallProgramOp callOp, PatternRewriter &rewriter) const override {

        auto ctx = callOp.getContext();

        // find the program called by the call op
        auto importProgramOp = callOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();

        if (!importProgramOp) {
            return rewriter.notifyMatchFailure(importProgramOp, "Not an import program");
        }

        auto executor = callOp.getProgram().getType().getExecutor();
        auto invocationType = torq_hl::InvocationType::get(ctx, executor);

        SmallVector<Type> invocationResultsTypes = {invocationType};

        if (executor == torq_hl::Executor::CSS) {

            // figure out the size of the invocation code and arguments
            auto program = SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableOp>(
                callOp, importProgramOp.getNameAttr()
            );

            if (!program) {
                return callOp.emitError() << "Symbol not found: " << importProgramOp.getName();
            }

            auto binaryOp = SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableBinaryOp>(
                program, rewriter.getStringAttr("code")
            );

            if (!binaryOp) {
                return callOp.emitError() << "Program not found in " << importProgramOp.getName();
            }

            auto codeVectorType = dyn_cast<VectorType>(binaryOp.getDataAttr().getType());
            auto codeSectionType =
                MemRefType::get(codeVectorType.getShape(), codeVectorType.getElementType());
            int64_t argsNumber = callOp.getInputs().size() + callOp.getInits().size();
            auto argsAddressesSectionType =
                MemRefType::get({argsNumber + 1}, rewriter.getI32Type());
            invocationResultsTypes =
                TypeRange{invocationType, codeSectionType, argsAddressesSectionType};
        }

        // create op to represent the invocation
        auto createInvocationOp = rewriter.create<torq_hl::CreateInvocationOp>(
            callOp.getLoc(), invocationResultsTypes, importProgramOp.getName(), callOp.getProgram(),
            nullptr, nullptr, nullptr, nullptr
        );

        SmallVector<Value> startOpCodeSections;

        if (executor == torq_hl::Executor::CSS) {

            // convert the code to a LRAM tensor and then to an ITCM tensor
            TypedValue<MemRefType> code =
                cast<TypedValue<MemRefType>>(createInvocationOp.getCodeSections()[0]);
            auto lramCode = rewriter.create<memref::AllocOp>(
                callOp.getLoc(),
                createMemRefTypeWithMemorySpace(code.getType(), torq_hl::MemorySpace::Lram)
            );

            if (failed(createTorqCopy(rewriter, callOp.getLoc(), code, lramCode))) {
                llvm::report_fatal_error("failed to create copy to LRAM");
            }

            auto itcmCode = rewriter.create<memref::AllocOp>(
                callOp.getLoc(),
                createMemRefTypeWithMemorySpace(code.getType(), torq_hl::MemorySpace::Itcm)
            );

            if (failed(createTorqCopy(rewriter, callOp.getLoc(), lramCode, itcmCode))) {
                llvm::report_fatal_error("failed to create copy to ITCM");
            }

            startOpCodeSections.push_back(itcmCode);

            // convert the arguments adddresses section to a LRAM tensor and then to an ITCM tensor
            TypedValue<MemRefType> argsAddresses =
                cast<TypedValue<MemRefType>>(createInvocationOp.getCodeSections()[1]);
            auto lramArgsAddresses = rewriter.create<memref::AllocOp>(
                callOp.getLoc(),
                createMemRefTypeWithMemorySpace(argsAddresses.getType(), torq_hl::MemorySpace::Lram)
            );

            if (failed(createTorqCopy(rewriter, callOp.getLoc(), argsAddresses, lramArgsAddresses)
                )) {
                llvm::report_fatal_error("failed to create copy to LRAM");
            }

            auto dtcmArgsAddresses = rewriter.create<memref::AllocOp>(
                callOp.getLoc(),
                createMemRefTypeWithMemorySpace(argsAddresses.getType(), torq_hl::MemorySpace::Dtcm)
            );

            if (failed(
                    createTorqCopy(rewriter, callOp.getLoc(), lramArgsAddresses, dtcmArgsAddresses)
                )) {
                llvm::report_fatal_error("failed to create copy to DTCM");
            }

            startOpCodeSections.push_back(dtcmArgsAddresses);
        }

        // the arguments to the start are the operands of the call without the initial program
        // operand
        auto argValues = callOp.getOperands().slice(1, callOp.getOperands().size() - 1);

        // create a start_program op
        auto startOp = rewriter.create<torq_hl::StartProgramOp>(
            callOp.getLoc(),
            /* invocation = */ createInvocationOp.getInvocation(),
            /* code_sections = */ startOpCodeSections,
            /* args = */ argValues
        );

        // create a wait_program op
        rewriter.create<torq_hl::WaitProgramOp>(
            callOp.getLoc(), TypeRange{}, startOp.getInvocation()
        );

        rewriter.eraseOp(callOp);

        return success();
    }
};

class LowerCallProgramToStartWaitPass
    : public LowerCallProgramToStartWaitBase<LowerCallProgramToStartWaitPass> {
  public:
    LowerCallProgramToStartWaitPass() = default;
    LowerCallProgramToStartWaitPass(const LowerCallProgramToStartWaitPass &pass) {}

    void runOnOperation() override;
};

void LowerCallProgramToStartWaitPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<CallProgramPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerCallProgramToStartWaitPass() {
    return std::make_unique<LowerCallProgramToStartWaitPass>();
}

} // namespace mlir::syna::torq
