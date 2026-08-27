// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-resolve-invocation-addresses"

namespace mlir::syna::torq {

namespace {

class ResolveInvocationArgumentsPass
    : public impl::ResolveInvocationArgumentsBase<ResolveInvocationArgumentsPass> {
  public:
    ResolveInvocationArgumentsPass() = default;
    ResolveInvocationArgumentsPass(const ResolveInvocationArgumentsPass &pass) {}

    void runOnOperation() override;
};

// this function adds to the StartProgramOp that is being analyzed the attributes
// executor_code_address and invocation_args addresses of
// the code sections and arguments when the invocation is called.
//
// The addresses cannot directly be found by inspecting the operands because
// the start op may be inside another program and the arguments are arguments to
// the program.
//
// The code below assumes that it is possible to statically know which allocations
// correspond to which arguments and code sections.
static LogicalResult updateCreateInvocation(
    AddressResolver &resolver, torq_hl::StartProgramOp startProgramOp,
    InvocationValue currentInvocation, const IRMapping &mapping
) {

    // in case the invocation is a argument of the current program
    // get the actual invocation value that was passed to the current
    // program (this is outside the current program)
    auto invocationValue = cast<InvocationValue>(mapping.lookup(startProgramOp.getInvocation()));

    // get the create_invocation operation that created this invocation
    auto createInvocationOp = invocationValue.getDefiningOp<torq_hl::CreateInvocationOp>();

    if (!createInvocationOp) {
        return startProgramOp.emitError() << "invocation must be from a CreateInvocationOp";
    }

    // update the address of the code sections
    SmallVector<int64_t> executorCodeSectionAddresses;
    for (auto [idx, section] : llvm::enumerate(startProgramOp.getCodeSections())) {

        auto sectionValue = mapping.lookup(section);

        auto executorAddress = resolver.getAddress(sectionValue);

        if (!executorAddress) {
            return startProgramOp.emitError() << "code section #" << idx << " must have address";
        }

        executorCodeSectionAddresses.push_back(*executorAddress);
    }

    createInvocationOp.setExecutorCodeAddresses(executorCodeSectionAddresses);

    auto executor = createInvocationOp.getProgram().getType().getExecutor();

    // update the address for all the arguments
    SmallVector<Attribute> argAttrs;

    for (auto [idx, arg] : llvm::enumerate(startProgramOp.getArgs())) {

        // find the actual value of this argument
        auto argValue = mapping.lookup(arg);

        // we are passing an invocation as argument to the program
        if (auto invocationArg = dyn_cast<TypedValue<torq_hl::InvocationType>>(argValue)) {

            if (auto sourceInvocationOp =
                    invocationArg.getDefiningOp<torq_hl::CreateInvocationOp>()) {

                torq_hl::Executor executor = invocationArg.getType().getExecutor();

                auto maybeExecutorId = sourceInvocationOp.getExecutorId();

                int executorId = -1;

                if (maybeExecutorId) {
                    executorId = (*maybeExecutorId).getZExtValue();
                }

                auto maybeCodeSectionAddresses = sourceInvocationOp.getXramCodeAddresses();

                if (!maybeCodeSectionAddresses) {
                    return startProgramOp.emitError()
                           << "argument #" << idx
                           << " has no valid code section addresses, see: " << argValue;
                }

                argAttrs.push_back(torq_hl::InvocationAttr::get(
                    startProgramOp.getContext(), executor, executorId, *maybeCodeSectionAddresses
                ));
            }
            else if (auto descriptorOp = invocationArg.getDefiningOp<torq_hl::DescriptorOp>()) {

                torq_hl::Executor executor = invocationArg.getType().getExecutor();

                int executorId = descriptorOp.getExecutorId().getZExtValue();

                argAttrs.push_back(torq_hl::InvocationAttr::get(
                    startProgramOp.getContext(), executor, executorId,
                    descriptorOp.getXramCodeAddresses()
                ));
            }
            else {
                return startProgramOp.emitError()
                       << "argument #" << idx << " must be an create_invocation op";
            }
        }
        else if (auto memRefArg = dyn_cast<TypedValue<MemRefType>>(argValue)) {

            // A data-dependent (dynamic) memory offset — e.g. from a slice whose
            // start index is a runtime value (stablehlo.dynamic_slice) — cannot be
            // resolved to a static address on Torq. Reject it with a clear
            // diagnostic instead of aborting deeper in address resolution.
            if (memRefHasDynamicOffset(memRefArg.getType())) {
                return startProgramOp.emitError()
                       << "argument #" << idx
                       << " has a data-dependent (dynamic) memory offset, which Torq does not "
                          "support (e.g. a slice with a runtime start index); see: "
                       << argValue;
            }

            auto maybeStartAddress = resolver.getDataStartAddress(argValue);

            if (!maybeStartAddress) {
                return startProgramOp.emitError()
                       << "argument #" << idx << " has no valid addresses "
                       << torq_hl::stringifyExecutor(executor) << ", see: " << argValue;
            }

            argAttrs.push_back(torq_hl::BufferAttr::get(
                startProgramOp.getContext(), memRefArg.getType(), *maybeStartAddress
            ));
        }
        else {
            return startProgramOp.emitError() << "argument #" << idx << " has unsupported type "
                                              << argValue.getType() << ", see: " << argValue;
        }
    }

    auto invocationArgs = ArrayAttr::get(startProgramOp.getContext(), argAttrs);

    createInvocationOp.setInvocationArgsAttr(invocationArgs);

    return success();
}

static LogicalResult updateWait(
    AddressResolver &resolver, torq_hl::WaitProgramOp waitProgramOp,
    InvocationValue currentInvocation, InvocationReturns returnValues
) {

    SmallVector<int64_t> executorArgsAddresses;
    for (auto returnValue : returnValues) {

        auto returnValueAddress = resolver.getAddress(returnValue);

        if (returnValueAddress) {
            executorArgsAddresses.push_back(*returnValueAddress);
        }
        else {
            executorArgsAddresses.push_back(-1);
        }
    }

    waitProgramOp.setResultAddresses(executorArgsAddresses);

    return success();
}

void ResolveInvocationArgumentsPass::runOnOperation() {
    auto funcOp = getOperation();

    WalkExecutionOptions options;
    AddressResolver resolver;

    // simulate the whole function and update each invocation when we encounter a start_program op
    // and we know the actual addresses of the arguments and code sections
    options.onStart = [&](torq_hl::StartProgramOp startProgramOp, InvocationValue invocation,
                          const IRMapping &mapping) {
        return updateCreateInvocation(resolver, startProgramOp, invocation, mapping);
    };
    options.onFinish = [&](torq_hl::WaitProgramOp waitProgramOp, InvocationValue invocation,
                           InvocationReturns returnValues) {
        return updateWait(resolver, waitProgramOp, invocation, returnValues);
    };

    // Only recurse into NSS programs. CSS/Slice/Host programs are always
    // leaf programs — guaranteed by the program lowering pass which ensures
    // non-NSS executors never contain nested StartProgramOp/WaitProgramOp pairs.
    options.walkInto = [&](torq_hl::StartProgramOp startProgramOp, InvocationValue invocation,
                           const mlir::IRMapping &) {
        return startProgramOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS;
    };

    if (failed(walkExecution(funcOp, options))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createResolveInvocationArgumentsPass() {
    return std::make_unique<ResolveInvocationArgumentsPass>();
}

} // namespace mlir::syna::torq
