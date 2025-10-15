// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#define DEBUG_TYPE "torq-outline-cpu-tasks"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class OutlineCpuProgramsPass : public OutlineCpuProgramsBase<OutlineCpuProgramsPass> {
  public:
    using OutlineCpuProgramsBase<OutlineCpuProgramsPass>::OutlineCpuProgramsBase;

    void runOnOperation() override;
};

static ModuleOp
createExecutable(IRRewriter &rewriter, Location loc, torq_hl::Executor executor, std::string name) {

    auto executorName = torq_hl::stringifyExecutor(executor);
    auto executableOp = rewriter.create<IREE::HAL::ExecutableOp>(loc, name);

    rewriter.setInsertionPointToStart(&executableOp.getBody().front());
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::get(
        rewriter.getContext(), rewriter.getStringAttr("llvm-" + executorName),
        rewriter.getStringAttr(executorName)
    );
    auto variantOp = rewriter.create<IREE::HAL::ExecutableVariantOp>(loc, executorName, targetAttr);
    rewriter.setInsertionPointToStart(&variantOp.getBody().front());
    auto moduleOp = rewriter.create<ModuleOp>(loc);

    return moduleOp;
}

static Operation *outlineProgram(
    IRRewriter &rewriter, std::string name, torq_hl::ProgramOp programOp, ModuleOp moduleOp
) {

    auto loc = programOp.getLoc();

    rewriter.setInsertionPointToStart(moduleOp.getBody());

    // create the function that will host the outlined operations
    // the function has no input/outputs and instead uses flow.dispatch.tensor.load and
    // flow.dispatch.tensor.store operations to interoperate with the IREE LLVM codegen pipeline
    auto funcOp = rewriter.create<func::FuncOp>(
        loc, name, rewriter.getFunctionType(TypeRange{}, TypeRange{})
    );

    // set the translation strategy of the function to the default CPU strategy
    // FIXME: we should probably run the configuration pipeline instead
    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(),
        mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault
    );

    succeeded(setTranslationInfo(funcOp, translationInfo));

    rewriter.setInsertionPointToStart(funcOp.addEntryBlock());

    auto returnOp = programOp.getBody().front().getTerminator();

    // first we add all binding subspans as LLVMCPU codegen from IREE expects them to be at the
    // beginning of the function

    int outputCount = returnOp->getNumOperands();

    auto programArgs = programOp.getBody().front().getArguments();
    SmallVector<Value> outputSubspanOps;
    SmallVector<Value> inputSubspanOps;

    // create subspans for all arguments (both inputs and inits)
    for (auto [idx, input] : llvm::enumerate(programArgs)) {

        auto accessType = idx < outputCount ? IREE::Flow::TensorAccess::ReadWrite
                                            : IREE::Flow::TensorAccess::ReadOnly;

        auto inputType = cast<RankedTensorType>(input.getType());

        RankedTensorType bindingType;

        // use i8 tensors when the input type is i1 as the CPU pipeline doesn't support i1
        // bindings
        if (inputType.getElementType().isInteger(1)) {
            bindingType = RankedTensorType::get(
                inputType.getShape(), rewriter.getI8Type(), inputType.getEncoding()
            );
        }
        else {
            bindingType = inputType;
        }

        auto dispatchTensorType = IREE::Flow::DispatchTensorType::get(accessType, bindingType);

        auto subspanOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
            loc, dispatchTensorType, APInt(64, 0), APInt(64, idx),
            IREE::HAL::DescriptorType::StorageBuffer, nullptr, ValueRange{},
            rewriter.getIndexAttr(4)
        );

        if (idx < outputCount) {
            outputSubspanOps.push_back(subspanOp.getResult());
        }

        inputSubspanOps.push_back(subspanOp.getResult());
    }

    // load all arguments (both inputs and inits that have at least one use)
    IRMapping map;

    for (auto [idx, input] : llvm::enumerate(programArgs)) {

        // inits that are never read have 0-uses, non need to load them
        if (input.use_empty()) {
            continue;
        }

        auto inputType = cast<RankedTensorType>(input.getType());

        // if the input type is i1 we need to load it as i8 and then truncate it
        if (inputType.getElementType().isInteger(1)) {

            auto boundType =
                cast<IREE::Flow::DispatchTensorType>(inputSubspanOps[idx].getType()).getBoundType();

            auto tensorLoadOp = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
                loc, cast<RankedTensorType>(boundType), inputSubspanOps[idx], ValueRange{}
            );

            auto tensorCastOp =
                rewriter.create<arith::TruncIOp>(loc, inputType, tensorLoadOp.getResult());

            map.map(input, tensorCastOp.getResult());
        }
        else {

            auto tensorLoadOp = rewriter.create<IREE::Flow::DispatchTensorLoadOp>(
                loc, cast<RankedTensorType>(input.getType()), inputSubspanOps[idx], ValueRange{}
            );

            map.map(input, tensorLoadOp);
        }
    }

    // clone all the operations into the task except the return operation
    for (auto &op : programOp.getBody().front()) {

        if (isa<torq_hl::ReturnOp>(op)) {
            continue;
        }

        rewriter.clone(op, map);
    }

    // perform a store operation for all the outputs
    SmallVector<Value> funcOutputs;
    for (auto [idx, output] : llvm::enumerate(returnOp->getOperands())) {
        auto outputType = cast<RankedTensorType>(output.getType());

        // if the output type is i1 we need to extend it to i8 before storing it
        if (outputType.getElementType().isInteger(1)) {

            auto boundType = cast<IREE::Flow::DispatchTensorType>(outputSubspanOps[idx].getType())
                                 .getBoundType();

            auto tensorCastOp = rewriter.create<arith::ExtUIOp>(
                loc, cast<RankedTensorType>(boundType), map.lookup(output)
            );

            rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
                loc, tensorCastOp, outputSubspanOps[idx], ValueRange{}
            );
        }
        else {
            rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
                loc, map.lookup(output), outputSubspanOps[idx], ValueRange{}
            );
        }
    }

    // add a return operation to finish the function
    rewriter.create<func::ReturnOp>(loc, ValueRange{});

    // replace the program operation with a import program operation that uses the original program
    rewriter.setInsertionPoint(programOp);
    rewriter.replaceOpWithNewOp<torq_hl::ImportProgramOp>(
        programOp, programOp.getType(), programOp.getName()
    );

    return funcOp;
}

void OutlineCpuProgramsPass::runOnOperation() {

    SmallVector<func::FuncOp> funcOps;

    // create a list of all the functions where we should analyze
    for (auto op : getOperation().getOps<func::FuncOp>()) {
        funcOps.push_back(op);
    }

    IRRewriter rewriter(getOperation());

    for (auto funcOp : funcOps) {

        // create a list of the programs we want to outline
        SmallVector<torq_hl::ProgramOp> cssProgramOps;
        SmallVector<torq_hl::ProgramOp> hostProgramOps;
        funcOp.walk([&](torq_hl::ProgramOp programOp) {
            if (programOp.getType().getExecutor() == torq_hl::Executor::CSS) {
                cssProgramOps.push_back(programOp);
            }
            else if (programOp.getType().getExecutor() == torq_hl::Executor::Host) {
                hostProgramOps.push_back(programOp);
            }
        });

        // outline each program CSS program in a dedicated executable (each will be compiled to
        // distinct bare metal executable)
        for (auto programOp : cssProgramOps) {
            rewriter.setInsertionPoint(funcOp);
            auto moduleOp = createExecutable(
                rewriter, funcOp.getLoc(), torq_hl::Executor::CSS, programOp.getName().str()
            );
            outlineProgram(rewriter, "main", programOp, moduleOp);
        }

        // outline each host program as a function in the same executable (they will be jointly
        // compiled into a single host shared lib)
        if (hostProgramOps.size() > 0) {
            rewriter.setInsertionPoint(funcOp);
            auto hostCode =
                createExecutable(rewriter, funcOp.getLoc(), torq_hl::Executor::Host, "host_code");
            for (auto programOp : hostProgramOps) {
                outlineProgram(rewriter, programOp.getName().str(), programOp, hostCode);
            }
        }
    }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOutlineCpuProgramsPass() {
    return std::make_unique<OutlineCpuProgramsPass>();
}

} // namespace mlir::syna::torq
