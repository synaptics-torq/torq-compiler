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
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#define DEBUG_TYPE "torq-assign-operations-to-cpu-programs"

using namespace ::mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

struct OutliningGroup {
    Operation *root;
    SmallVector<Operation *> toOutline;
    torq_hl::Executor executor;
    std::string name;
    int programId;

    InFlightDiagnostic emitError(Location loc, const Twine &message) {
        auto diag = mlir::emitError(loc, message);

        diag.attachNote(root->getLoc())
            .append("Program root operation")
            .appendOp(*root, OpPrintingFlags());

        diag.attachNote(std::nullopt).append("Operations included in outlined program:");

        for (auto op : toOutline) {
            diag.attachNote(std::nullopt).appendOp(*op, OpPrintingFlags());
        }

        return diag;
    }
};

class AssignOperationsToCpuProgramsPass
    : public impl::AssignOperationsToCpuProgramsBase<AssignOperationsToCpuProgramsPass> {
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

        if (isa<IREE::TensorExt::DispatchTensorLoadOp>(op)) {

            noOutlineOps.insert(&op);

            // exclude all operations that are used perform the load
            SetVector<Operation *> upstreamOps;
            if (failed(getBackwardSlice(&op, &upstreamOps))) {
                return failure();
            }
            noOutlineOps.insert(upstreamOps.begin(), upstreamOps.end());
        }
        else if (auto storeOp = dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(op)) {

            noOutlineOps.insert(&op);

            // exclude all operations use the resulting value of the store
            SetVector<Operation *> downstreamOps;
            getForwardSlice(&op, &downstreamOps);
            noOutlineOps.insert(downstreamOps.begin(), downstreamOps.end());

            // exclude all the operations that are used to compute the target of the store
            if (auto targetOp = storeOp.getTarget().getDefiningOp()) {
                SetVector<Operation *> upstreamOps;
                if (failed(getBackwardSlice(targetOp, &upstreamOps))) {
                    return failure();
                }
                noOutlineOps.insert(upstreamOps.begin(), upstreamOps.end());
                noOutlineOps.insert(targetOp);
            }
        }
        // these operations do not need to be executed on CSS (they are no-op or can be done with
        // the DMA engine or will be folded away when we unroll loops)
        else if (isa<arith::ConstantOp, tensor::EmptyOp, tensor::InsertSliceOp,
                     tensor::ExtractSliceOp, func::ReturnOp, tensor::ParallelInsertSliceOp,
                     tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::ReshapeOp,
                     affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp>(op)) {
            noOutlineOps.insert(&op);
        }
        // Leave concat on the default Slice/NSS path unless it was explicitly
        // reassigned to CSS/Host. The later NSS lowering can handle these ops,
        // while outlining them here forces an unnecessary CSS fallback.
        else if (isa<tensor::ConcatOp>(op) && getTargetExecutor(&op) == torq_hl::Executor::Slice) {
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

            groups.push_back({&op, {&op}, executor, executableName, nextProgramId});

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

            SmallVector<Operation *> scalarInputUsers(scalarInput.getUsers());
            computeTopologicalSorting(scalarInputUsers);

            // Insert just before the top most user.
            OpBuilder rewriter(scalarInputUsers.back());

            Value insertionValue = scalarInput;

            Type scalarInputType = scalarInput.getType();
            bool isIndex = isa<IndexType>(scalarInputType);
            if (isIndex) {
                scalarInputType = rewriter.getI32Type();
                insertionValue = arith::IndexCastOp::create(
                                     rewriter, scalarInput.getLoc(), scalarInputType, scalarInput
                )
                                     .getResult();
            }

            auto tensorType = RankedTensorType::get({}, scalarInputType);

            // we use tensor.insert to be symmetric with the output conversion
            auto emptyOp =
                tensor::EmptyOp::create(rewriter, scalarInput.getLoc(), tensorType, ValueRange{});

            tensor::InsertOp toTensorOp = tensor::InsertOp::create(
                rewriter, scalarInput.getLoc(), tensorType, insertionValue, emptyOp.getResult(),
                ValueRange{}
            );

            Operation *toScalarOp = tensor::ExtractOp::create(
                rewriter, scalarInput.getLoc(), scalarInputType, toTensorOp.getResult(),
                ValueRange{}
            );

            // insert at the beginning of the group so that it is outlined before
            // their users
            group.toOutline.insert(group.toOutline.begin(), toScalarOp);

            if (isIndex) {
                toScalarOp = arith::IndexCastUIOp::create(
                    rewriter, scalarInput.getLoc(), rewriter.getIndexType(),
                    toScalarOp->getResult(0)
                );

                group.toOutline.insert(group.toOutline.begin() + 1, toScalarOp);
            }

            scalarInput.replaceUsesWithIf(toScalarOp->getResult(0), [&](OpOperand &operand) {
                return outlinedOps.contains(operand.getOwner());
            });

            if (isIndex)
                outlinedOps.insert(insertionValue.getDefiningOp());
            else
                outlinedOps.insert(toTensorOp);
        }

        for (auto scalarOutput : scalarOutputs) {

            SmallVector<Operation *> scalarOutputUsers(scalarOutput.getUsers());
            mlir::computeTopologicalSorting(scalarOutputUsers);

            // Insert just before the top most user.
            OpBuilder rewriter(scalarOutputUsers.back());

            Value insertionValue = scalarOutput;

            Type scalarOutputType = scalarOutput.getType();
            bool isIndex = isa<IndexType>(scalarOutputType);
            if (isIndex) {
                scalarOutputType = rewriter.getI32Type();

                auto indexCastOp = arith::IndexCastOp::create(
                    rewriter, scalarOutput.getLoc(), scalarOutputType, scalarOutput
                );

                // This will make sure we don't replace the input when we do scalarOutput.replace...
                // below.
                outlinedOps.insert(indexCastOp);

                insertionValue = indexCastOp.getResult();

                group.toOutline.push_back(indexCastOp);
            }

            auto tensorType = RankedTensorType::get({}, scalarOutputType);

            auto emptyTensor =
                tensor::EmptyOp::create(rewriter, scalarOutput.getLoc(), tensorType, ValueRange{});

            // here we use insert op instead of from_elements to avoid
            // an error during compilation with LLVMCPU backend
            auto toTensorOp = tensor::InsertOp::create(
                rewriter, scalarOutput.getLoc(), tensorType, insertionValue,
                emptyTensor.getResult(), ValueRange{}
            );
            // This will make sure we don't replace the input when we do scalarOutput.replace...
            // below.
            if (insertionValue == scalarOutput)
                outlinedOps.insert(toTensorOp);

            Operation *toScalarOp = tensor::ExtractOp::create(
                rewriter, scalarOutput.getLoc(), scalarOutputType, toTensorOp.getResult(),
                ValueRange{}
            );

            if (isIndex) {
                toScalarOp = arith::IndexCastUIOp::create(
                    rewriter, scalarOutput.getLoc(), rewriter.getIndexType(),
                    toScalarOp->getResult(0)
                );
            }

            // insert at the end of the group so that it is outlined after their producer
            group.toOutline.push_back(emptyTensor);
            group.toOutline.push_back(toTensorOp);

            scalarOutput.replaceUsesWithIf(toScalarOp->getResult(0), [&](OpOperand &operand) {
                return !outlinedOps.contains(operand.getOwner());
            });
        }
    }
}

static LogicalResult outlineGroup(OutliningGroup &group) {

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
    IRRewriter rewriter(group.root);

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
            return group.emitError(
                input.getLoc(), "Failed to outline program due to non tensor input value"
            );
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
            return group.emitError(
                output.getLoc(), "Failed to outline program due to non tensor output value"
            );
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
                tensor::EmptyOp::create(rewriter, group.root->getLoc(), initType, ValueRange{});
        }

        executorInits.push_back(initTensor);
        executorOutputTypes.push_back(initTensor.getType());
    }

    // call the program
    auto callOp = torq_hl::CallProgramOp::create(
        rewriter, group.root->getLoc(), executorOutputTypes, maybeOutlineResult->program,
        executorInits, executorInputs
    );

    // we need to convert back the outputs to the original encoding
    for (auto [idx, output] : llvm::enumerate(maybeOutlineResult->outputs)) {
        auto rankedTensorType = dyn_cast<RankedTensorType>(output.getType());

        if (!rankedTensorType) {
            return group.emitError(
                output.getLoc(), "Failed to convert back non-ranked tensor value"
            );
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

static void fallbackGroupToExecutor(SmallVector<OutliningGroup> &groups, bool disableHost) {

    for (auto &group : groups) {

        // if the group contains any operation that uses floating point
        // we fall back to Host as CSS does not support floating point
        for (auto op : group.toOutline) {
            auto ret = op->walk([&](Operation *nestedOp) {
                for (auto operand : nestedOp->getOperands()) {
                    if (isa<FloatType>(operand.getType())) {
                        return WalkResult::interrupt();
                    }
                }

                for (auto result : nestedOp->getResults()) {
                    if (isa<FloatType>(result.getType())) {
                        return WalkResult::interrupt();
                    }
                }

                return WalkResult::advance();
            });

            if (ret.wasInterrupted() && group.executor != torq_hl::Executor::Host) {

                if (disableHost) {
                    group.emitError(
                        group.root->getLoc(),
                        "Group requires floating point operations but Host execution is disabled"
                    );
                    return;
                }

                group.executor = torq_hl::Executor::Host;

                group.name = llvm::formatv(
                    "{0}_{1}_{2}_fallback", torq_hl::stringifyExecutor(group.executor),
                    group.root->getName(), group.programId
                );

                break;
            }
        }
    }
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

        // change executor based on operations in the group
        // fallbackGroupToExecutor(*maybeGroups, disableHost);

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
