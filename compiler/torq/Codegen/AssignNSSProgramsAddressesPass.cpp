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

void AssignNSSProgramsAddressesPass::runOnOperation() {

    auto nssInvocations = findNSSInvocations(getOperation());

    if (failed(nssInvocations)) {
        signalPassFailure();
        return;
    }

    Builder builder(getOperation().getContext());

    // find the base address for the NSS programs that was reserved in the AssignAddressesPass
    int64_t nssProgramBase =
        getOperation()->getAttrOfType<IntegerAttr>("torq-nss-program-base").getInt();

    int64_t maxNssProgramAddress = nssProgramBase + clMaxNssProgramsSize;

    int64_t nextNssProgramAddress = nssProgramBase;

    for (auto invocationOp : nssInvocations.value()) {

        auto programOp = invocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        auto blockSizesAttr = programOp.getBlockSizes();

        if (!blockSizesAttr) {
            programOp.emitError("NSS program missing block sizes");
            signalPassFailure();
            return;
        }

        int programSize = 0;

        for (auto blockSize : *blockSizesAttr) {
            programSize += blockSize;
        }

        auto programAddress = llvm::alignTo(nextNssProgramAddress, 4);

        nextNssProgramAddress = programAddress + programSize;

        SmallVector<int64_t> addresses;

        int programOffset = 0;
        for (auto blockSize : *blockSizesAttr) {
            addresses.push_back(programAddress + programOffset);
            programOffset += blockSize;
        }

        invocationOp.setXramCodeAddresses(addresses);
    }

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
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignNSSProgramsAddressesPass() {
    return std::make_unique<AssignNSSProgramsAddressesPass>();
}

} // namespace mlir::syna::torq
