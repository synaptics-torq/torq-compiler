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

static FailureOr<int64_t> getCdmaDataStartAddress(
    Value value, TypedValue<torq_hl::InvocationType> invocation, AddressCache *cache
) {

    auto memrefType = dyn_cast<MemRefType>(value.getType());
    auto memSpace = getEncodingMemorySpace(memrefType);

    std::optional<int64_t> addr = getDataStartAddress(value, 0, invocation, cache);

    int64_t baseAddress = 0;

    switch (memSpace) {
    case torq_hl::MemorySpace::Lram:
        baseAddress = mlir::syna::torq::HwInfo::cdma_lram_base_address;
        break;
    case torq_hl::MemorySpace::Dtcm:
        baseAddress = mlir::syna::torq::HwInfo::cdma_dtcm_base_address;
        break;
    case torq_hl::MemorySpace::Itcm:
        baseAddress = mlir::syna::torq::HwInfo::cdma_itcm_base_address;
        break;
    default:
        return failure();
    }

    if (!addr) {
        return failure();
    }

    return baseAddress + addr.value();
}

static FailureOr<int64_t> getNdmaDataStartAddress(
    Value value, TypedValue<torq_hl::InvocationType> invocation, AddressCache *cache
) {

    auto address = getDataStartAddress(value, 0, invocation, cache);

    if (!address) {
        return failure();
    }

    return address.value();
}

template <typename OpT> static LogicalResult resolveDmaCfg(OpT op, AddressCache *cache) {
    if (op.getReadAddressAttr() && op.getWriteAddressAttr()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeReadAddress = getNdmaDataStartAddress(op.getRead(), nssInvocation, cache);

    if (failed(maybeReadAddress)) {
        return op.emitError("unable to resolve read address of value of ") << op.getRead();
    }

    auto maybeWriteAddress = getNdmaDataStartAddress(op.getWrite(), nssInvocation, cache);

    if (failed(maybeWriteAddress)) {
        return op.emitError("unable to resolve write address of ") << op.getWrite();
    }

    op.setReadAddress(*maybeReadAddress);
    op.setWriteAddress(*maybeWriteAddress);

    return success();
}

static LogicalResult resolveCDMAStart(torq_hw::CDMAStartOp op, AddressCache *cache) {
    if (op.getDestAddress() && op.getSrcAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeSrcAddress = getCdmaDataStartAddress(op.getSrc(), nssInvocation, cache);

    if (failed(maybeSrcAddress)) {
        return op.emitError("unable to resolve src address");
    }

    auto maybeDestAddress = getCdmaDataStartAddress(op.getDest(), nssInvocation, cache);

    if (failed(maybeDestAddress)) {
        return op.emitError("unable to resolve dest address");
    }

    op.setSrcAddress(*maybeSrcAddress);
    op.setDestAddress(*maybeDestAddress);

    return success();
}

static LogicalResult resolveCSSStart(torq_hw::CSSStartOp op, AddressCache *cache) {
    if (op.getProgramAddress() && op.getArgAddressesAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = getExecutorDataStartAddress(
        torq_hl::Executor::CSS, op.getProgram(), 0, nssInvocation, cache
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve program address");
    }

    auto maybeArgAddress = getExecutorDataStartAddress(
        torq_hl::Executor::CSS, op.getArgsAddresses(), 0, nssInvocation, cache
    );

    if (!maybeArgAddress) {
        return op.emitError("unable to resolve addresses buffer addresss");
    }

    op.setProgramAddress(*maybeProgramAddress);
    op.setArgAddressesAddress(*maybeArgAddress);

    return success();
}

static LogicalResult resolveSliceStart(torq_hw::SliceStartOp op, AddressCache *cache) {
    if (op.getProgramAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = getExecutorDataStartAddress(
        torq_hl::Executor::Slice, op.getProgram(), 0, nssInvocation, cache
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve program address");
    }

    op.setProgramAddress(*maybeProgramAddress);

    return success();
}

static LogicalResult resolveNext(torq_hl::NextOp op, AddressCache *cache) {
    if (op.getLramAddress()) {
        return success();
    }

    auto nssInvocation = getNssInvocation(op);

    auto maybeProgramAddress = getExecutorDataStartAddress(
        torq_hl::Executor::NSS, op.getLramArea(), 0, nssInvocation, cache
    );

    if (!maybeProgramAddress) {
        return op.emitError("unable to resolve lram area address");
    }

    op.setLramAddress(*maybeProgramAddress);

    return success();
}

void ResolveAddressesPass::runOnOperation() {
    auto funcOp = getOperation();

    AddressCache cache;
    LogicalResult result = success();

    auto walkResult = funcOp.walk([&](Operation *op) {
        TypeSwitch<Operation *>(op)
            .Case<torq_hw::DmaInCfgOp, torq_hw::DmaOutCfgOp>([&](auto dmaOp) {
                result = resolveDmaCfg(dmaOp, &cache);
            })
            .Case<torq_hw::CDMAStartOp>([&](auto cdmaOp) {
                result = resolveCDMAStart(cdmaOp, &cache);
            })
            .Case<torq_hw::CSSStartOp>([&](auto cssOp) { result = resolveCSSStart(cssOp, &cache); })
            .Case<torq_hw::SliceStartOp>([&](auto sliceOp) {
                result = resolveSliceStart(sliceOp, &cache);
            })
            .Case<torq_hl::NextOp>([&](auto nextOp) { result = resolveNext(nextOp, &cache); });

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
