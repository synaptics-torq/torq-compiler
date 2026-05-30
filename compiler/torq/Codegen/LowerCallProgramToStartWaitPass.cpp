// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-map-bindings"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// Stage a CSS code section by copying it XRAM->LRAM->ITCM at the current
// rewriter insertion point. Returns the ITCM-side memref SSA value.
static Value
stageCssCodeSection(RewriterBase &rewriter, Location loc, TypedValue<MemRefType> code) {

    auto lramCode = memref::AllocOp::create(
        rewriter, loc, createMemRefTypeWithMemorySpace(code.getType(), torq_hl::MemorySpace::Lram)
    );

    if (failed(createTorqCopy(rewriter, loc, code, lramCode))) {
        llvm::report_fatal_error("failed to create copy to LRAM");
    }

    auto itcmCode = memref::AllocOp::create(
        rewriter, loc, createMemRefTypeWithMemorySpace(code.getType(), torq_hl::MemorySpace::Itcm)
    );

    if (failed(createTorqCopy(rewriter, loc, lramCode, itcmCode))) {
        llvm::report_fatal_error("failed to create copy to ITCM");
    }

    return itcmCode;
}

// Stage a CSS arguments-addresses section by copying it XRAM->LRAM->DTCM at the
// current rewriter insertion point. Returns the DTCM-side memref SSA value.
static Value stageCssArgsAddressesSection(
    RewriterBase &rewriter, Location loc, TypedValue<MemRefType> argsAddresses
) {

    auto lramArgsAddresses = memref::AllocOp::create(
        rewriter, loc,
        createMemRefTypeWithMemorySpace(argsAddresses.getType(), torq_hl::MemorySpace::Lram)
    );

    if (failed(createTorqCopy(rewriter, loc, argsAddresses, lramArgsAddresses))) {
        llvm::report_fatal_error("failed to create copy to LRAM");
    }

    auto dtcmArgsAddresses = memref::AllocOp::create(
        rewriter, loc,
        createMemRefTypeWithMemorySpace(argsAddresses.getType(), torq_hl::MemorySpace::Dtcm)
    );

    if (failed(createTorqCopy(rewriter, loc, lramArgsAddresses, dtcmArgsAddresses))) {
        llvm::report_fatal_error("failed to create copy to DTCM");
    }

    return dtcmArgsAddresses;
}

// Look up the CSS program binary referenced by `importOp` and compute the
// CreateInvocationOp result types (invocation + code section + args-addresses
// section) for a call site with `argsNumber` inputs+inits.
static FailureOr<SmallVector<Type>> resolveCssInvocationResultTypes(
    OpBuilder &builder, torq_hl::ImportProgramOp importOp, int64_t argsNumber
) {
    auto ctx = builder.getContext();
    auto invocationType = torq_hl::InvocationType::get(ctx, torq_hl::Executor::CSS);

    auto program = SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableOp>(
        importOp, importOp.getNameAttr()
    );
    if (!program) {
        return importOp.emitError() << "Symbol not found: " << importOp.getName();
    }

    auto binaryOp = SymbolTable::lookupNearestSymbolFrom<IREE::HAL::ExecutableBinaryOp>(
        program, builder.getStringAttr("code")
    );
    if (!binaryOp) {
        return importOp.emitError() << "Program not found in " << importOp.getName();
    }

    auto binaryDataAttr = cast<DenseIntElementsAttr>(binaryOp.getDataAttr());
    auto codeVectorType = dyn_cast<VectorType>(binaryDataAttr.getType());
    auto codeSectionType =
        MemRefType::get(codeVectorType.getShape(), codeVectorType.getElementType());
    auto argsAddressesSectionType = MemRefType::get({argsNumber + 1}, builder.getI32Type());
    return SmallVector<Type>{invocationType, codeSectionType, argsAddressesSectionType};
}

// Lower a single torq_hl.call_program op into create_invocation + start_program
// + wait_program. If `sharedItcmCode` is non-null the CSS code section staging
// is skipped and the call's start_program references that pre-staged ITCM
// allocation instead.
static LogicalResult
lowerCallProgramOp(RewriterBase &rewriter, torq_hl::CallProgramOp callOp, Value sharedItcmCode) {
    rewriter.setInsertionPoint(callOp);

    auto ctx = callOp.getContext();
    auto importProgramOp = callOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();
    if (!importProgramOp) {
        return callOp.emitError("call_program operand is not an import_program");
    }

    auto executor = callOp.getProgram().getType().getExecutor();
    auto invocationType = torq_hl::InvocationType::get(ctx, executor);

    SmallVector<Type> invocationResultsTypes = {invocationType};
    if (executor == torq_hl::Executor::CSS) {
        int64_t argsNumber = callOp.getInputs().size() + callOp.getInits().size();
        auto maybeTypes = resolveCssInvocationResultTypes(rewriter, importProgramOp, argsNumber);
        if (failed(maybeTypes)) {
            return failure();
        }
        invocationResultsTypes = std::move(*maybeTypes);
    }

    auto createInvocationOp = torq_hl::CreateInvocationOp::create(
        rewriter, callOp.getLoc(), invocationResultsTypes, importProgramOp.getName(),
        callOp.getProgram(), nullptr, nullptr, nullptr, nullptr
    );

    SmallVector<Value> startOpCodeSections;
    if (executor == torq_hl::Executor::CSS) {
        if (sharedItcmCode) {
            startOpCodeSections.push_back(sharedItcmCode);
        }
        else {
            auto code = cast<TypedValue<MemRefType>>(createInvocationOp.getCodeSections()[0]);
            startOpCodeSections.push_back(stageCssCodeSection(rewriter, callOp.getLoc(), code));
        }

        auto argsAddresses = cast<TypedValue<MemRefType>>(createInvocationOp.getCodeSections()[1]);
        startOpCodeSections.push_back(
            stageCssArgsAddressesSection(rewriter, callOp.getLoc(), argsAddresses)
        );
    }

    auto argValues = callOp.getOperands().slice(1, callOp.getOperands().size() - 1);

    auto startOp = torq_hl::StartProgramOp::create(
        rewriter, callOp.getLoc(),
        /* invocation = */ createInvocationOp.getInvocation(),
        /* code_sections = */ startOpCodeSections,
        /* args = */ argValues
    );

    torq_hl::WaitProgramOp::create(rewriter, callOp.getLoc(), TypeRange{}, startOp.getInvocation());

    rewriter.eraseOp(callOp);
    return success();
}

// Lowers each call_program at its own site. For CSS calls, the program's code
// staging (XRAM->LRAM->ITCM copy) is hoisted to the smallest enclosing point
// that still covers every clone produced by later UnrollForallLoops -- the
// outermost scf.forall ancestor if any, otherwise the call site itself -- and
// cached, so the cloned start_program ops all reference the same ITCM
// allocation. The CSS program then stays at ITCM offset 0 (its binary is
// linked assuming offset 0). The cache key is the ImportProgramOp; in
// practice AssignOperationsToCpuPrograms assigns a unique program id per
// outlined op, so each CSS import is referenced by exactly one call_program.
//
// Staying at the smallest necessary scope keeps the ITCM allocation's
// lifetime short, so when two different CSS programs run sequentially in the
// same dispatch they can share a single ITCM slot (each at offset 0) instead
// of being placed side-by-side. See synaptics-torq/torq-compiler-dev#1615.
class CallProgramPattern : public OpRewritePattern<torq_hl::CallProgramOp> {
  public:
    CallProgramPattern(
        MLIRContext *ctx, llvm::DenseMap<torq_hl::ImportProgramOp, Value> &cssCodeCache
    )
        : OpRewritePattern(ctx), cssCodeCache(cssCodeCache) {}

    LogicalResult
    matchAndRewrite(torq_hl::CallProgramOp callOp, PatternRewriter &rewriter) const override {

        auto importOp = callOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();
        if (!importOp) {
            return callOp.emitError("call_program operand is not an import_program");
        }

        Value sharedItcmCode;
        if (importOp.getType().getExecutor() == torq_hl::Executor::CSS) {
            auto it = cssCodeCache.find(importOp);
            if (it != cssCodeCache.end()) {
                sharedItcmCode = it->second;
            }
            else {
                // Pick the outermost scf.forall ancestor of the call (within
                // the parent function); if there isn't one, hoist no further
                // than the call site. This keeps the ITCM allocation alive
                // only across the iterations that share it.
                Operation *hoistAnchor = callOp;
                for (Operation *parent = callOp->getParentOp();
                     parent && !isa<FunctionOpInterface>(parent); parent = parent->getParentOp()) {
                    if (isa<scf::ForallOp>(parent)) {
                        hoistAnchor = parent;
                    }
                }
                rewriter.setInsertionPoint(hoistAnchor);

                // Materialize a CreateInvocationOp at the hoist point solely to
                // obtain an SSA value for the code section memref; the per-call
                // invocation and args section are emitted in lowerCallProgramOp.
                // The invocation and args-addresses results of this hoisted op
                // are unused.
                int64_t argsNumber = callOp.getInputs().size() + callOp.getInits().size();
                auto maybeTypes = resolveCssInvocationResultTypes(rewriter, importOp, argsNumber);
                if (failed(maybeTypes)) {
                    return failure();
                }

                auto hoistedInvocationOp = torq_hl::CreateInvocationOp::create(
                    rewriter, hoistAnchor->getLoc(), *maybeTypes, importOp.getName(),
                    importOp.getResult(), nullptr, nullptr, nullptr, nullptr
                );

                auto code = cast<TypedValue<MemRefType>>(hoistedInvocationOp.getCodeSections()[0]);
                sharedItcmCode = stageCssCodeSection(rewriter, hoistAnchor->getLoc(), code);

                cssCodeCache.try_emplace(importOp, sharedItcmCode);
            }
        }

        return lowerCallProgramOp(rewriter, callOp, sharedItcmCode);
    }

  private:
    llvm::DenseMap<torq_hl::ImportProgramOp, Value> &cssCodeCache;
};

class LowerCallProgramToStartWaitPass
    : public impl::LowerCallProgramToStartWaitBase<LowerCallProgramToStartWaitPass> {
  public:
    LowerCallProgramToStartWaitPass() = default;
    LowerCallProgramToStartWaitPass(const LowerCallProgramToStartWaitPass &pass) {}

    void runOnOperation() override;
};

void LowerCallProgramToStartWaitPass::runOnOperation() {
    MLIRContext *ctx = &getContext();
    llvm::DenseMap<torq_hl::ImportProgramOp, Value> cssCodeCache;

    RewritePatternSet patterns(ctx);
    patterns.add<CallProgramPattern>(ctx, cssCodeCache);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerCallProgramToStartWaitPass() {
    return std::make_unique<LowerCallProgramToStartWaitPass>();
}

} // namespace mlir::syna::torq
