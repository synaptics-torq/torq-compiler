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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "torq-resolve-addresses"

namespace mlir::syna::torq {

namespace {

class ResolveAddressesPass : public ResolveAddressesBase<ResolveAddressesPass> {
  public:
    ResolveAddressesPass() = default;
    ResolveAddressesPass(const ResolveAddressesPass &pass) {}

    void runOnOperation() override;
};

static FailureOr<int64_t>
getCdmaDataStartAddress(Value value, TypedValue<torq_hl::InvocationType> invocation) {

    auto memrefType = dyn_cast<MemRefType>(value.getType());
    auto memSpace = getEncodingMemorySpace(memrefType);

    std::optional<int64_t> addr = getDataStartAddress(value, 0, invocation);

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

static FailureOr<int64_t>
getNdmaDataStartAddress(Value value, TypedValue<torq_hl::InvocationType> invocation) {

    auto address = getDataStartAddress(value, 0, invocation);

    if (!address) {
        return failure();
    }

    return address.value();
}

template <typename OpT> class ResolveDmaCfg : public OpRewritePattern<OpT> {
  public:
    using OpRewritePattern<OpT>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const override {

        if (op.getReadAddressAttr() && op.getWriteAddressAttr()) {
            return failure();
        }

        auto nssInvocation = getNssInvocation(op);

        auto maybeReadAddress = getNdmaDataStartAddress(op.getRead(), nssInvocation);

        if (failed(maybeReadAddress)) {
            return op.emitError("unable to resolve read address");
        }

        auto maybeWriteAddress = getNdmaDataStartAddress(op.getWrite(), nssInvocation);

        if (failed(maybeWriteAddress)) {
            return op.emitError("unable to resolve write address");
        }

        rewriter.modifyOpInPlace(op, [&]() {
            op.setReadAddress(*maybeReadAddress);
            op.setWriteAddress(*maybeWriteAddress);
        });

        return success();
    }
};

class ResolveCDMAStart : public OpRewritePattern<torq_hw::CDMAStartOp> {
  public:
    using OpRewritePattern<torq_hw::CDMAStartOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hw::CDMAStartOp op, PatternRewriter &rewriter) const override {

        if (op.getDestAddress() && op.getSrcAddress()) {
            return failure();
        }

        auto nssInvocation = getNssInvocation(op);

        auto maybeSrcAddress = getCdmaDataStartAddress(op.getSrc(), nssInvocation);

        if (failed(maybeSrcAddress)) {
            return op.emitError("unable to resolve src address");
        }

        auto maybeDestAddress = getCdmaDataStartAddress(op.getDest(), nssInvocation);

        if (failed(maybeDestAddress)) {
            return op.emitError("unable to resolve dest address");
        }

        rewriter.modifyOpInPlace(op, [&]() {
            op.setSrcAddress(*maybeSrcAddress);
            op.setDestAddress(*maybeDestAddress);
        });

        return success();
    }
};

class ResolveCSSStart : public OpRewritePattern<torq_hw::CSSStartOp> {
  public:
    using OpRewritePattern<torq_hw::CSSStartOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(torq_hw::CSSStartOp op, PatternRewriter &rewriter) const override {

        if (op.getProgramAddress() && op.getArgAddressesAddress()) {
            return failure();
        }

        auto nssInvocation = getNssInvocation(op);

        auto maybeProgramAddress =
            getExecutorDataStartAddress(torq_hl::Executor::CSS, op.getProgram(), 0, nssInvocation);

        if (!maybeProgramAddress) {
            return op.emitError("unable to resolve program address");
        }

        auto maybeArgAddress = getExecutorDataStartAddress(
            torq_hl::Executor::CSS, op.getArgsAddresses(), 0, nssInvocation
        );

        if (!maybeArgAddress) {
            return op.emitError("unable to resolve addresses buffer addresss");
        }

        rewriter.modifyOpInPlace(op, [&]() {
            op.setProgramAddress(*maybeProgramAddress);
            op.setArgAddressesAddress(*maybeArgAddress);
        });

        return success();
    }
};

void ResolveAddressesPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = getOperation().getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<ResolveDmaCfg<torq_hw::DmaInCfgOp>>(ctx);
    patterns.add<ResolveDmaCfg<torq_hw::DmaOutCfgOp>>(ctx);
    patterns.add<ResolveCDMAStart>(ctx);
    patterns.add<ResolveCSSStart>(ctx);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        funcOp.emitError("Failed to apply ResolveAddressesVisitor pattern");
        signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createResolveAddressesPass() {
    return std::make_unique<ResolveAddressesPass>();
}

} // namespace mlir::syna::torq
