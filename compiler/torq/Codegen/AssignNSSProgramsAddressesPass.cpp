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

LogicalResult measureBlockSizes(torq_hl::CreateInvocationOp createInvocationOp);
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
// and set each invocation's xram_code_address. Returns the end of the laid-out range.
int64_t assignXramCodeAddresses(
    ArrayRef<torq_hl::CreateInvocationOp> nssInvocations, int64_t nssProgramBase
) {

    int64_t nextNssProgramAddress = nssProgramBase;

    for (auto invocationOp : nssInvocations) {

        auto programOp = invocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        assert(
            programOp.getBlockSizes() && "block_sizes missing: SegmentNSSPrograms must run first"
        );

        nextNssProgramAddress = llvm::alignTo(nextNssProgramAddress, 4);

        invocationOp.setXramCodeAddress(nextNssProgramAddress);

        nextNssProgramAddress += programOp.getBlockOffsets().back();
    }

    return nextNssProgramAddress;
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
        if (failed(measureBlockSizes(invocationOp))) {
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
