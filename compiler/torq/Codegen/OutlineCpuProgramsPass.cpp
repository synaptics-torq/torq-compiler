// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/IR/Attributes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
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

class OutlineCpuProgramsPass : public impl::OutlineCpuProgramsBase<OutlineCpuProgramsPass> {
  public:
    using OutlineCpuProgramsBase<OutlineCpuProgramsPass>::OutlineCpuProgramsBase;

    void runOnOperation() override;
};

static ModuleOp
createExecutable(IRRewriter &rewriter, Location loc, torq_hl::Executor executor, std::string name) {

    auto executorName = torq_hl::stringifyExecutor(executor);
    auto executableOp = IREE::HAL::ExecutableOp::create(rewriter, loc, name);

    rewriter.setInsertionPointToStart(&executableOp.getBody().front());
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::get(
        rewriter.getContext(), rewriter.getStringAttr("llvm-" + executorName),
        rewriter.getStringAttr(executorName)
    );
    auto variantOp =
        IREE::HAL::ExecutableVariantOp::create(rewriter, loc, executorName, targetAttr);
    rewriter.setInsertionPointToStart(&variantOp.getBody().front());
    auto moduleOp = ModuleOp::create(rewriter, loc);

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
    auto funcOp = func::FuncOp::create(
        rewriter, loc, name, rewriter.getFunctionType(TypeRange{}, TypeRange{})
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

    SmallVector<IREE::HAL::PipelineBindingAttr> bindingAttrs;
    for (int i = 0; i < programArgs.size(); ++i) {
        bindingAttrs.push_back(IREE::HAL::PipelineBindingAttr::get(
            rewriter.getContext(), IREE::HAL::DescriptorType::StorageBuffer,
            (i < outputCount ? IREE::HAL::DescriptorFlags::None
                             : IREE::HAL::DescriptorFlags::ReadOnly)
        ));
    }

    auto layout =
        IREE::HAL::PipelineLayoutAttr::get(rewriter.getContext(), bindingAttrs, 0, std::nullopt);

    // create subspans for all arguments (both inputs and inits)
    for (auto [idx, input] : llvm::enumerate(programArgs)) {

        auto accessType = idx < outputCount ? IREE::TensorExt::TensorAccess::ReadWrite
                                            : IREE::TensorExt::TensorAccess::ReadOnly;

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

        auto dispatchTensorType = IREE::TensorExt::DispatchTensorType::get(accessType, bindingType);

        auto subspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
            rewriter, loc, dispatchTensorType, layout, APInt(64, idx), nullptr, ValueRange{},
            rewriter.getIndexAttr(4),
            IREE::HAL::DescriptorFlagsAttr::get(
                rewriter.getContext(), idx < outputCount ? IREE::HAL::DescriptorFlags::None
                                                         : IREE::HAL::DescriptorFlags::ReadOnly
            )
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
                cast<IREE::TensorExt::DispatchTensorType>(inputSubspanOps[idx].getType())
                    .getBoundType();

            auto tensorLoadOp = IREE::TensorExt::DispatchTensorLoadOp::create(
                rewriter, loc, cast<RankedTensorType>(boundType), inputSubspanOps[idx], ValueRange{}
            );

            auto tensorCastOp =
                arith::TruncIOp::create(rewriter, loc, inputType, tensorLoadOp.getResult());

            map.map(input, tensorCastOp.getResult());
        }
        else {

            auto tensorLoadOp = IREE::TensorExt::DispatchTensorLoadOp::create(
                rewriter, loc, cast<RankedTensorType>(input.getType()), inputSubspanOps[idx],
                ValueRange{}
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

            auto boundType =
                cast<IREE::TensorExt::DispatchTensorType>(outputSubspanOps[idx].getType())
                    .getBoundType();

            auto tensorCastOp = arith::ExtUIOp::create(
                rewriter, loc, cast<RankedTensorType>(boundType), map.lookup(output)
            );

            IREE::TensorExt::DispatchTensorStoreOp::create(
                rewriter, loc, tensorCastOp, outputSubspanOps[idx], ValueRange{}
            );
        }
        else {
            IREE::TensorExt::DispatchTensorStoreOp::create(
                rewriter, loc, map.lookup(output), outputSubspanOps[idx], ValueRange{}
            );
        }
    }

    // add a return operation to finish the function
    func::ReturnOp::create(rewriter, loc, ValueRange{});

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
