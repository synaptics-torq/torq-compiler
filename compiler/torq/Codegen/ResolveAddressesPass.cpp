// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-resolve-addresses"

namespace mlir::syna::torq {

namespace {

class ResolveAddressesPass : public impl::ResolveAddressesBase<ResolveAddressesPass> {
  public:
    ResolveAddressesPass() = default;
    ResolveAddressesPass(const ResolveAddressesPass &pass) {}

    void runOnOperation() override;
};

template <typename OpT> static LogicalResult resolveDmaCfg(AddressResolver &resolver, OpT op) {
    if (op.getReadAddressAttr() && op.getWriteAddressAttr()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeReadAddress = resolver.getDataStartAddress(op.getRead(), nssInvocation);

    if (!maybeReadAddress) {
        return op.emitError("unable to resolve read address of value of ") << op.getRead();
    }

    auto maybeWriteAddress = resolver.getDataStartAddress(op.getWrite(), nssInvocation);

    if (!maybeWriteAddress) {
        return op.emitError("unable to resolve write address of ") << op.getWrite();
    }

    op.setReadAddress(*maybeReadAddress);
    op.setWriteAddress(*maybeWriteAddress);

    return success();
}

static LogicalResult resolveCDMAStart(AddressResolver &resolver, torq_hw::CDMAStartOp op) {
    if (op.getDestAddress() && op.getSrcAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeSrcAddress = resolver.getCdmaDataStartAddress(op.getSrc(), nssInvocation);

    if (!maybeSrcAddress) {
        return op.emitError("unable to resolve src address");
    }

    auto maybeDestAddress = resolver.getCdmaDataStartAddress(op.getDest(), nssInvocation);

    if (!maybeDestAddress) {
        return op.emitError("unable to resolve dest address");
    }

    op.setSrcAddress(*maybeSrcAddress);
    op.setDestAddress(*maybeDestAddress);

    return success();
}

static LogicalResult resolveCSSStart(AddressResolver &resolver, torq_hw::CSSStartOp op) {
    if (op.getProgramAddress() && op.getArgAddressesAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = resolver.getExecutorDataStartAddress(
        torq_hl::Executor::CSS, op.getProgram(), nssInvocation
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve program address");
    }

    auto maybeArgAddress = resolver.getExecutorDataStartAddress(
        torq_hl::Executor::CSS, op.getArgsAddresses(), nssInvocation
    );

    if (!maybeArgAddress) {
        return op.emitError("unable to resolve addresses buffer addresss");
    }

    op.setProgramAddress(*maybeProgramAddress);
    op.setArgAddressesAddress(*maybeArgAddress);

    return success();
}

static LogicalResult resolveSliceStart(AddressResolver &resolver, torq_hw::SliceStartOp op) {
    if (op.getProgramAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = resolver.getExecutorDataStartAddress(
        torq_hl::Executor::Slice, op.getProgram(), nssInvocation
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve program address");
    }

    op.setProgramAddress(*maybeProgramAddress);

    return success();
}

static LogicalResult resolveNext(AddressResolver &resolver, torq_hl::NextOp op) {
    if (op.getLramAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = resolver.getExecutorDataStartAddress(
        torq_hl::Executor::NSS, op.getLramArea(), nssInvocation
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve lram area address");
    }

    op.setLramAddress(*maybeProgramAddress);

    return success();
}

void ResolveAddressesPass::runOnOperation() {
    auto funcOp = getOperation();

    AddressResolver resolver;
    LogicalResult result = success();

    auto walkResult = funcOp.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
            .Case<torq_hw::DmaInCfgOp, torq_hw::DmaOutCfgOp>([&](auto dmaOp) {
                result = resolveDmaCfg(resolver, dmaOp);
            })
            .Case<torq_hw::CDMAStartOp>([&](auto cdmaOp) {
                result = resolveCDMAStart(resolver, cdmaOp);
            })
            .Case<torq_hw::CSSStartOp>([&](auto cssOp) {
                result = resolveCSSStart(resolver, cssOp);
            })
            .Case<torq_hw::SliceStartOp>([&](auto sliceOp) {
                result = resolveSliceStart(resolver, sliceOp);
            })
            .Case<torq_hl::NextOp>([&](auto nextOp) { result = resolveNext(resolver, nextOp); });

        if (failed(result))
            return WalkResult::interrupt();
        return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
        return signalPassFailure();
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createResolveAddressesPass() {
    return std::make_unique<ResolveAddressesPass>();
}

} // namespace mlir::syna::torq
