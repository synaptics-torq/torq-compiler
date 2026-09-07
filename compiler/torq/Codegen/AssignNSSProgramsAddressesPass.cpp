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
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/MemoryUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-assign-nss-programs-addresses"

extern llvm::cl::opt<int> clMaxNssProgramsSize;

namespace mlir::syna::torq {

LogicalResult updateCodeSectionSizes(torq_hl::CreateInvocationOp createInvocationOp);
LogicalResult resolveGetBlockReadAddresses(torq_hl::CreateInvocationOp createInvocationOp);
LogicalResult updateGetBlockOperations(torq_hl::CreateInvocationOp createInvocationOp);

namespace {

class AssignNSSProgramsAddressesPass
    : public impl::AssignNSSProgramsAddressesBase<AssignNSSProgramsAddressesPass> {
  public:
    using AssignNSSProgramsAddressesBase::AssignNSSProgramsAddressesBase;
    void runOnOperation() override;
};

FailureOr<SmallVector<torq_hl::CreateInvocationOp>> findNSSInvocations(FunctionOpInterface funcOp) {

    SmallVector<torq_hl::CreateInvocationOp> nssInvocations;

    for (auto invocationOp : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {

        if (invocationOp.getProgram().getType().getExecutor() != torq_hl::Executor::NSS) {
            continue;
        }

        auto programOp = invocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        if (!programOp) {
            return invocationOp.emitError(
                "NSS program operand is not defined by a torq_hl::ProgramOp"
            );
        }

        nssInvocations.push_back(invocationOp);
    }

    return nssInvocations;
}

// Lay the programs out contiguously in XRAM starting from nssProgramBase, using their block_sizes,
// and set each invocation's xram_code_addresses. Returns the end of the laid-out range.
int64_t assignXramCodeAddresses(
    ArrayRef<torq_hl::CreateInvocationOp> nssInvocations, int64_t nssProgramBase
) {

    int64_t nextNssProgramAddress = nssProgramBase;

    for (auto invocationOp : nssInvocations) {

        auto programOp = invocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        auto blockSizes = programOp.getBlockSizes();

        assert(blockSizes && "block_sizes missing: SegmentNSSPrograms must run first");

        nextNssProgramAddress = llvm::alignTo(nextNssProgramAddress, 4);

        SmallVector<int64_t> addresses;

        for (auto blockSize : *blockSizes) {
            addresses.push_back(nextNssProgramAddress);
            nextNssProgramAddress += blockSize;
        }

        invocationOp.setXramCodeAddresses(addresses);
    }

    return nextNssProgramAddress;
}

// Rewrite the InvocationAttr entries in the started invocation's invocation_args
// that refer to NSS invocations.
void resolveNssInvocationArgs(torq_hl::StartProgramOp startProgramOp) {

    auto targetInvocationOp =
        startProgramOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();
    if (!targetInvocationOp || !targetInvocationOp.getInvocationArgs()) {
        return;
    }

    SmallVector<Attribute> argAttrs(targetInvocationOp.getInvocationArgs()->getValue());

    for (auto [argAttr, arg] : llvm::zip_equal(argAttrs, startProgramOp.getArgs())) {
        auto invocationArg = dyn_cast<TypedValue<torq_hl::InvocationType>>(arg);
        if (!invocationArg || invocationArg.getType().getExecutor() != torq_hl::Executor::NSS) {
            continue;
        }
        auto argInvocationOp = invocationArg.getDefiningOp<torq_hl::CreateInvocationOp>();
        assert(argInvocationOp && "NSS invocation argument not defined by create_invocation");
        auto invocationAttr = cast<torq_hl::InvocationAttr>(argAttr);
        argAttr = torq_hl::InvocationAttr::get(
            startProgramOp.getContext(), torq_hl::Executor::NSS, invocationAttr.getExecutorId(),
            *argInvocationOp.getXramCodeAddresses()
        );
    }

    targetInvocationOp.setInvocationArgsAttr(ArrayAttr::get(startProgramOp.getContext(), argAttrs));
}

void AssignNSSProgramsAddressesPass::runOnOperation() {

    auto nssInvocations = findNSSInvocations(getOperation());

    if (failed(nssInvocations)) {
        signalPassFailure();
        return;
    }

    // find the base address for the NSS programs that was reserved in the AssignAddressesPass
    int64_t nssProgramBase =
        getOperation()->getAttrOfType<IntegerAttr>("torq-nss-program-base").getInt();

    // Assign provisional addresses from the upper-bound block sizes so the programs
    // can be compiled. The compiled size of each block is address-independent, so
    // the real addresses can be assigned afterwards.
    assignXramCodeAddresses(nssInvocations.value(), nssProgramBase);

    for (auto invocationOp : nssInvocations.value()) {
        if (failed(resolveGetBlockReadAddresses(invocationOp))) {
            signalPassFailure();
            return;
        }
    }

    for (auto invocationOp : nssInvocations.value()) {
        if (failed(updateCodeSectionSizes(invocationOp))) {
            signalPassFailure();
            return;
        }
    }

    // the final assignment, now from the measured sizes
    int64_t nextNssProgramAddress = assignXramCodeAddresses(nssInvocations.value(), nssProgramBase);

    int64_t maxNssProgramAddress = nssProgramBase + clMaxNssProgramsSize;

    if (nextNssProgramAddress > maxNssProgramAddress) {

        getOperation().emitError(
            "Not enough space available for NSS programs (required " +
            std::to_string(nextNssProgramAddress - nssProgramBase) + " bytes, available " +
            std::to_string(maxNssProgramAddress - nssProgramBase) +
            " bytes) (this can be increased with the --torq-max-nss-programs-size option)"
        );

        signalPassFailure();
        return;
    }

    // NSS invocations appear as arguments only at function level (a program loads
    // the next program's blocks), so a flat walk covers every placeholder
    for (auto startProgramOp : getOperation().getFunctionBody().getOps<torq_hl::StartProgramOp>()) {
        resolveNssInvocationArgs(startProgramOp);
    }

    for (auto invocationOp : nssInvocations.value()) {
        if (failed(updateGetBlockOperations(invocationOp))) {
            signalPassFailure();
            return;
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignNSSProgramsAddressesPass() {
    return std::make_unique<AssignNSSProgramsAddressesPass>();
}

} // namespace mlir::syna::torq
