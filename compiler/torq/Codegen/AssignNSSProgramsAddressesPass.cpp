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
#include "torq/Utils/MemoryUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-assign-nss-programs-addresses"

namespace mlir::syna::torq {

namespace {

class AssignNSSProgramsAddressesPass
    : public AssignNSSProgramsAddressesBase<AssignNSSProgramsAddressesPass> {
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

    for (auto invocationOp : nssInvocations.value()) {

        auto programOp = invocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        auto blockSizesAttr = programOp->getAttrOfType<ArrayAttr>("torq_block_size");

        if (!blockSizesAttr) {
            programOp.emitError("NSS program missing 'torq_block_size' attribute");
            signalPassFailure();
            return;
        }

        int programSize = 0;

        for (auto blockSizeAttr : blockSizesAttr) {
            programSize += cast<IntegerAttr>(blockSizeAttr).getInt();
        }

        auto programAddress = reserveXramArea(getOperation(), programSize);

        SmallVector<int64_t> addresses;

        int programOffset = 0;
        for (auto blockSizeAttr : blockSizesAttr) {
            addresses.push_back(programAddress + programOffset);
            programOffset += cast<IntegerAttr>(blockSizeAttr).getInt();
        }

        invocationOp.setXramCodeAddresses(addresses);
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignNSSProgramsAddressesPass() {
    return std::make_unique<AssignNSSProgramsAddressesPass>();
}

} // namespace mlir::syna::torq
