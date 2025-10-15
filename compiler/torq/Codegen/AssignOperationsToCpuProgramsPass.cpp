// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/OutlineUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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

#define DEBUG_TYPE "torq-assign-operations-to-cpu-programs"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

struct OutliningGroup {
    Operation *root;
    SmallVector<Operation *> toOutline;
    torq_hl::Executor executor;
    std::string name;
};

class AssignOperationsToCpuProgramsPass
    : public AssignOperationsToCpuProgramsBase<AssignOperationsToCpuProgramsPass> {
  public:
    using AssignOperationsToCpuProgramsBase::AssignOperationsToCpuProgramsBase;

    AssignOperationsToCpuProgramsPass(const AssignOperationsToCpuProgramsOptions &options) {
        this->disableCSS.setValue(options.disableCSS);
        this->disableHost.setValue(options.disableHost);
    }

    void runOnOperation() override;

  private:
    FailureOr<SmallVector<OutliningGroup>>
    createOutliningGroups(Region *region, int nextProgramId = 0);
};

FailureOr<SmallVector<OutliningGroup>>
AssignOperationsToCpuProgramsPass::createOutliningGroups(Region *region, int nextProgramId) {

    // find all the operation that we don't want to outline
    DenseSet<Operation *> noOutlineOps;

    for (auto &op : region->getOps()) {

        if (isa<IREE::Flow::DispatchTensorLoadOp>(op)) {

            noOutlineOps.insert(&op);

            // exclude all operations that are used perform the load
            SetVector<Operation *> upstreamOps;
            getBackwardSlice(&op, &upstreamOps);
            noOutlineOps.insert(upstreamOps.begin(), upstreamOps.end());
        }
        else if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(op)) {

            noOutlineOps.insert(&op);

            // exclude all operations use the resulting value of the store
            SetVector<Operation *> downstreamOps;
            getForwardSlice(&op, &downstreamOps);
            noOutlineOps.insert(downstreamOps.begin(), downstreamOps.end());

            // exclude all the operations that are used to compute the target of the store
            if (auto targetOp = storeOp.getTarget().getDefiningOp()) {
                SetVector<Operation *> upstreamOps;
                getBackwardSlice(targetOp, &upstreamOps);
                noOutlineOps.insert(upstreamOps.begin(), upstreamOps.end());
                noOutlineOps.insert(targetOp);
            }
        }
        // these operations do not need to be executed on CSS (they are no-op or can be done with
        // the DMA engine or will be folded away when we unroll loops)
        else if (isa<arith::ConstantOp, tensor::EmptyOp, tensor::InsertSliceOp,
                     tensor::ExtractSliceOp, func::ReturnOp, tensor::ParallelInsertSliceOp,
                     tensor::CollapseShapeOp, tensor::ExpandShapeOp, affine::AffineApplyOp,
                     affine::AffineMaxOp, affine::AffineMinOp>(op)) {
            noOutlineOps.insert(&op);
        }
        else if (op.getDialect()->getNamespace() == torq_hl::TorqHLDialect::getDialectNamespace()) {
            noOutlineOps.insert(&op);
        }
        else if (op.getDialect()->getNamespace() ==
                 bufferization::BufferizationDialect::getDialectNamespace()) {
            noOutlineOps.insert(&op);
        }
        else if (auto applyOp = dyn_cast<affine::AffineApplyOp>(op)) {

            // if the operands of the applyOp are not iteration variables of an SCF, then
            // consider this operation for outlining, otherwise do not consider it as when
            // we will unroll the loop to run it on NSS we will be able to constant fold
            // this operation
            for (auto operand : applyOp->getOperands()) {
                if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                    if (blockArg.getOwner()->getParentOp()->getDialect()->getNamespace() !=
                        scf::SCFDialect::getDialectNamespace()) {
                        continue;
                    }
                }
            }

            noOutlineOps.insert(&op);
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Ops that can't be outlined:\n";
        for (auto op : noOutlineOps) {
            op->dump();
        }
        llvm::dbgs() << "\n";
    });

    // create an outlining group for each operator that we can outline
    SmallVector<OutliningGroup> groups;

    for (auto &op : region->getOps()) {

        if (noOutlineOps.contains(&op)) {
            continue;
        }
        else if (op.getDialect()->getNamespace() == scf::SCFDialect::getDialectNamespace()) {

            for (auto &nestedRegion : op.getRegions()) {
                auto maybeGroups = createOutliningGroups(&nestedRegion, nextProgramId);
                if (failed(maybeGroups)) {
                    return failure();
                }
                groups.append(maybeGroups->begin(), maybeGroups->end());
                nextProgramId += maybeGroups->size();
            }
        }
        else {

            torq_hl::Executor executor;

            if (!disableCSS) {
                executor = getTargetExecutor(&op, torq_hl::Executor::CSS);
            }
            else if (!disableHost) {
                executor = getTargetExecutor(&op, torq_hl::Executor::Host);
            }
            else {
                return op.emitError("Unable to find executor for op");
            }

            std::string executableName = llvm::formatv(
                "{0}_{1}_{2}", torq_hl::stringifyExecutor(executor), op.getName(), nextProgramId
            );

            groups.push_back({&op, {&op}, executor, executableName});

            nextProgramId++;
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Ops to outline:\n";
        for (auto group : groups) {
            group.root->dump();
        }
        llvm::dbgs() << "\n";
    });

    return groups;
}

static void fuseScalarConstantsInGroups(SmallVector<OutliningGroup> &groups) {

    for (auto &group : groups) {
        // copy all the scalar constant used in the operation in the program (this is an
        // optimization to reduce arguments)
        group.root->walk([&](Operation *nestedOp) {
            for (auto &opOperand : nestedOp->getOpOperands()) {
                auto operand = opOperand.get();

                // skip tensors, we don't want to bake them into the program
                // TODO: maybe small tensors we could copy them
                if (isa<RankedTensorType>(operand.getType())) {
                    continue;
                }

                auto constOp = operand.getDefiningOp<arith::ConstantOp>();

                // skip non-constant operands and constants defined within the operation
                if (constOp && !group.root->isAncestor(constOp)) {

                    // clone the constant so that we can outline it without
                    // having to return its value
                    if (!constOp.getResult().hasOneUse()) {
                        IRRewriter rewriter(constOp);
                        rewriter.setInsertionPoint(constOp);
                        constOp = cast<arith::ConstantOp>(rewriter.clone(*constOp));
                        opOperand.set(constOp.getResult());
                    }

                    group.toOutline.insert(group.toOutline.begin(), constOp);
                }
            }
        });
    }
}

static void fuseVectorEmptyInGroups(SmallVector<OutliningGroup> &groups) {

    for (auto &group : groups) {
        // move any tensor empty operator inside the group, this way when
        // we compile the program we can do an in-place operation on the
        // outputs
        group.root->walk([&](Operation *nestedOp) {
            for (auto &opOperand : nestedOp->getOpOperands()) {
                auto emptyOp = opOperand.get().getDefiningOp<tensor::EmptyOp>();

                if (!emptyOp) {
                    continue;
                }

                // skip empty op defined within the operation
                if (!group.root->isAncestor(emptyOp)) {

                    // clone the empty op so that we can outline it without
                    // having to return its value
                    if (!emptyOp.getResult().hasOneUse()) {
                        IRRewriter rewriter(emptyOp);
                        rewriter.setInsertionPoint(emptyOp);
                        emptyOp = cast<tensor::EmptyOp>(rewriter.clone(*emptyOp));
                        opOperand.set(emptyOp.getResult());
                    }

                    group.toOutline.insert(group.toOutline.begin(), emptyOp);
                }
            }
        });
    }
}

static void convertScalarInOutsToTensors(SmallVector<OutliningGroup> &groups) {
    for (auto &group : groups) {

        DenseSet<Operation *> outlinedOps;
        for (auto op : group.toOutline) {
            op->walk([&](Operation *nestedOp) { outlinedOps.insert(nestedOp); });
        }

        DenseSet<Value> scalarInputs;
        DenseSet<Value> scalarOutputs;

        for (auto toOutlineOp : group.toOutline) {

            // walk over the operation and its nested operations
            toOutlineOp->walk([&](Operation *op) {
                // find all the scalar inputs
                for (auto operand : op->getOperands()) {

                    // skip non-scalar operands
                    if (!isa<IntegerType, FloatType, IndexType>(operand.getType())) {
                        continue;
                    }

                    // skip all block arguments of operations in the same group
                    if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
                        if (outlinedOps.contains(blockArg.getOwner()->getParentOp())) {
                            continue;
                        }
                    }

                    // skip all scalars produced by operations in the same group
                    if (outlinedOps.contains(operand.getDefiningOp())) {
                        continue;
                    }

                    scalarInputs.insert(operand);
                }

                // find all the scalar outputs
                for (auto result : op->getResults()) {

                    // skip non-scalar results
                    if (!isa<IntegerType, FloatType, IndexType>(result.getType())) {
                        continue;
                    }

                    // skip all scalars consumed only by operations in the same group
                    bool allUsesInGroup = true;
                    for (auto &use : result.getUses()) {
                        if (!outlinedOps.contains(use.getOwner())) {
                            allUsesInGroup = false;
                            break;
                        }
                    }
                    if (allUsesInGroup) {
                        continue;
                    }

                    scalarOutputs.insert(result);
                }
            });
        }

        for (auto scalarInput : scalarInputs) {

            auto tensorType = RankedTensorType::get({}, scalarInput.getType());

            OpBuilder rewriter(scalarInput.getDefiningOp());
            rewriter.setInsertionPointAfter(scalarInput.getDefiningOp());

            // we use tensor.insert to be symmetric with the output conversion
            auto emptyOp =
                rewriter.create<tensor::EmptyOp>(scalarInput.getLoc(), tensorType, ValueRange{});
            auto toTensorOp = rewriter.create<tensor::InsertOp>(
                scalarInput.getLoc(), tensorType, scalarInput, emptyOp.getResult(), ValueRange{}
            );

            auto toScalarOp = rewriter.create<tensor::ExtractOp>(
                scalarInput.getLoc(), scalarInput.getType(), toTensorOp.getResult(), ValueRange{}
            );

            // insert at the beginning of the group so that it is outlined before
            // their users
            group.toOutline.insert(group.toOutline.begin(), toScalarOp);

            scalarInput.replaceAllUsesExcept(toScalarOp.getResult(), toTensorOp);
        }

        for (auto scalarOutput : scalarOutputs) {

            auto tensorType = RankedTensorType::get({}, scalarOutput.getType());

            IRRewriter rewriter(scalarOutput.getDefiningOp());
            rewriter.setInsertionPointAfter(scalarOutput.getDefiningOp());

            auto emptyTensor =
                rewriter.create<tensor::EmptyOp>(scalarOutput.getLoc(), tensorType, ValueRange{});

            // here we use insert op instead of from_elements to avoid
            // an error during compilation with LLVMCPU backend
            auto toTensorOp = rewriter.create<tensor::InsertOp>(
                scalarOutput.getLoc(), tensorType, scalarOutput, emptyTensor.getResult(),
                ValueRange{}
            );

            auto toScalarOp = rewriter.create<tensor::ExtractOp>(
                scalarOutput.getLoc(), scalarOutput.getType(), toTensorOp.getResult(), ValueRange{}
            );

            // insert at the end of the group so that it is outlined after their producer
            group.toOutline.push_back(emptyTensor);
            group.toOutline.push_back(toTensorOp);

            scalarOutput.replaceAllUsesExcept(toScalarOp.getResult(), toTensorOp);
        }
    }
}

static LogicalResult outlineGroup(OutliningGroup &group) {

    IRRewriter rewriter(group.root);

    rewriter.setInsertionPoint(group.root);

    torq_hl::MemorySpace executorMemorySpace;

    if (group.executor == torq_hl::Executor::CSS) {
        executorMemorySpace = torq_hl::MemorySpace::Dtcm;
    }
    else if (group.executor == torq_hl::Executor::Host) {
        executorMemorySpace = torq_hl::MemorySpace::Xram;
    }
    else {
        llvm::report_fatal_error("Unsupported executor for outlining");
    }

    // create an program with all the operations we want to outline (this copies the operations)
    rewriter.setInsertionPoint(group.root);

    auto maybeOutlineResult = outlineProgram(
        rewriter, group.name, group.executor, group.toOutline, /*destinationStyle=*/true
    );

    if (failed(maybeOutlineResult)) {
        return failure();
    }

    // find the list of all the inputs that are take from the inits
    DenseSet<Value> initInputs;
    for (auto it : maybeOutlineResult->outputToInput) {
        initInputs.insert(it.second);
    }

    // convert all the inputs to a format suitable for the executor if necessary
    SmallVector<Value> executorInputs;
    DenseMap<Value, Value> inputToExecutorInput;
    for (auto input : maybeOutlineResult->inputs) {

        auto tensorInput = dyn_cast<TypedValue<RankedTensorType>>(input);

        if (!tensorInput) {
            group.root->dump();
            llvm::report_fatal_error("Unsupported non-tensor input during outlining");
        }

        // we need to convert the input to a *dense* encoding with the correct memory space so
        // that the executor can use it
        auto executorEncoding = createDenseEncoding(tensorInput.getType(), executorMemorySpace);
        Value executorInput = convertTensorToEncoding(rewriter, tensorInput, executorEncoding);

        // add the converted input to the list of inputs that we will use in the call
        // we skip inputs that are remapped to inits
        if (!initInputs.contains(input)) {
            executorInputs.push_back(executorInput);
        }

        // add the converted input to the map input -> converted input that will be
        // used to populate the inits
        inputToExecutorInput.try_emplace(input, executorInput);
    }

    SmallVector<Value> executorInits;
    SmallVector<Type> executorOutputTypes;

    // create init tensors in executor memory space where the outputs of the program will be stored
    // (we know that by construction these are not read)
    for (auto output : maybeOutlineResult->outputs) {

        // TODO: handle non-tensor inputs
        auto rankedTensorType = dyn_cast<RankedTensorType>(output.getType());

        if (!rankedTensorType) {
            llvm::dbgs() << "In program:\n";
            maybeOutlineResult->program.dump();

            llvm::report_fatal_error("Unsupported tensor ouput during outlining");
        }

        Value initTensor;

        auto maybeInput = maybeOutlineResult->outputToInput.find(output);

        // if there is an input that can be used as init for this output, use it
        if (maybeInput != maybeOutlineResult->outputToInput.end()) {
            initTensor = inputToExecutorInput[maybeInput->second];
        }

        // otherwise create an empty tensor suitable for the executor
        else {
            auto initEncoding = createDenseEncoding(rankedTensorType, executorMemorySpace);
            auto initType = createRankedTensorTypeWithEncoding(rankedTensorType, initEncoding);
            initTensor =
                rewriter.create<tensor::EmptyOp>(group.root->getLoc(), initType, ValueRange{});
        }

        executorInits.push_back(initTensor);
        executorOutputTypes.push_back(initTensor.getType());
    }

    // call the program
    auto callOp = rewriter.create<torq_hl::CallProgramOp>(
        group.root->getLoc(), executorOutputTypes, maybeOutlineResult->program, executorInits,
        executorInputs
    );

    // we need to convert back the outputs to the original encoding
    for (auto [idx, output] : llvm::enumerate(maybeOutlineResult->outputs)) {
        auto rankedTensorType = dyn_cast<RankedTensorType>(output.getType());

        if (!rankedTensorType) {
            llvm::report_fatal_error("Unsupported non-tensor output during outlining");
        }

        Value origEncodingOutputValue = convertTensorToType(
            rewriter, cast<TypedValue<RankedTensorType>>(callOp.getResult(idx)), rankedTensorType
        );

        output.replaceAllUsesWith(origEncodingOutputValue);
    }

    // delete all operations that were outlined and are no longer used
    // we delete in reverse order to ensure any use within the group is
    // deleted before the operation that produces it
    for (auto op : llvm::reverse(group.toOutline)) {
        if (op->use_empty()) {
            rewriter.eraseOp(op);
        }
    }

    return success();
}

void AssignOperationsToCpuProgramsPass::runOnOperation() {

    for (auto funcOp : getOperation().getOps<func::FuncOp>()) {

        // we create groups of operations that will be outlined in a single program
        auto maybeGroups = createOutliningGroups(&funcOp.getFunctionBody());

        if (failed(maybeGroups)) {
            return signalPassFailure();
        }

        // we add to each program all the scalar constant parameters so that we dont
        // need to passe them as arguments
        fuseScalarConstantsInGroups(*maybeGroups);

        // we add to each program all the empty op so that when bufferizing the
        // operations we know which values are actually empty and we try to re-use
        // the output buffers
        fuseVectorEmptyInGroups(*maybeGroups);

        // transform any remaining scalar input / output to/from tensors
        // so that all program arguments are tensors
        convertScalarInOutsToTensors(*maybeGroups);

        // we do the actual outlining of each group
        for (auto &group : *maybeGroups) {

            if (group.executor == torq_hl::Executor::CSS && disableCSS) {
                group.root->emitError(
                    "Cannot execute operation on CSS as disabled by command line option"
                );
                return signalPassFailure();
            }
            else if (group.executor == torq_hl::Executor::Host && disableHost) {
                group.root->emitError(
                    "Cannot execute operation on Host as disabled by command line option"
                );
                return signalPassFailure();
            }

            if (failed(outlineGroup(group))) {
                return signalPassFailure();
            }
        }
    }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createAssignOperationsToCpuProgramsPass(bool disableCss, bool disableHost) {
    return std::make_unique<AssignOperationsToCpuProgramsPass>(
        AssignOperationsToCpuProgramsOptions{disableCss, disableHost}
    );
}

} // namespace mlir::syna::torq
