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

// Look up the CSS program binary referenced by `importOp` and return the memref
// type of its code section.
static FailureOr<MemRefType>
getCssCodeSectionType(OpBuilder &builder, torq_hl::ImportProgramOp importOp) {
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
    return MemRefType::get(codeVectorType.getShape(), codeVectorType.getElementType());
}

// Create a torq_hl.program_code op projecting the shared XRAM code memref of the
// CSS program. It is placed right after the import_program so it is
// loop-invariant. This is called once per program (from the ITCM-staging
// cache-miss path), so every invocation of the program reuses this single value
// and the binary is placed in XRAM exactly once. See
// synaptics-torq/torq-compiler-dev#1615.
static FailureOr<Value>
createProgramCode(RewriterBase &rewriter, torq_hl::ImportProgramOp importOp) {
    auto maybeType = getCssCodeSectionType(rewriter, importOp);
    if (failed(maybeType)) {
        return failure();
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(importOp);
    auto programCodeOp = torq_hl::ProgramCodeOp::create(
        rewriter, importOp.getLoc(), *maybeType, importOp.getResult()
    );

    return programCodeOp.getCode();
}

// Lower a single torq_hl.call_program op into create_invocation + start_program
// + wait_program. For CSS calls `sharedItcmCode` is the ITCM-staged code memref
// (shared across all invocations of the program); the call's create_invocation
// only produces the per-invocation args-addresses section.
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
        // The code section is shared via torq_hl.program_code; create_invocation
        // only carries the per-invocation args-addresses section.
        int64_t argsNumber = callOp.getInputs().size() + callOp.getInits().size();
        invocationResultsTypes.push_back(MemRefType::get({argsNumber + 1}, rewriter.getI32Type()));
    }

    auto createInvocationOp = torq_hl::CreateInvocationOp::create(
        rewriter, callOp.getLoc(), invocationResultsTypes, importProgramOp.getName(),
        callOp.getProgram(), nullptr, nullptr, nullptr, nullptr
    );

    SmallVector<Value> startOpCodeSections;
    if (executor == torq_hl::Executor::CSS) {
        assert(sharedItcmCode && "CSS call requires staged ITCM code");
        startOpCodeSections.push_back(sharedItcmCode);

        auto argsAddresses = cast<TypedValue<MemRefType>>(createInvocationOp.getCodeSections()[0]);
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
// memref is the single torq_hl.program_code value at the function entry, and its
// staging (XRAM->LRAM->ITCM copy) is hoisted to the smallest enclosing point
// that still covers every clone produced by later UnrollForallLoops -- the
// outermost scf.forall ancestor if any, otherwise the call site itself -- and
// cached, so the cloned start_program ops all reference the same ITCM
// allocation. The CSS program then stays at ITCM offset 0 (its binary is
// linked assuming offset 0). The cache is keyed on the program symbol, so even
// if the same program is imported by more than one import_program op the code
// is projected (program_code) and staged into ITCM exactly once.
//
// Staying at the smallest necessary scope keeps the ITCM allocation's
// lifetime short, so when two different CSS programs run sequentially in the
// same dispatch they can share a single ITCM slot (each at offset 0) instead
// of being placed side-by-side. See synaptics-torq/torq-compiler-dev#1615.
class CallProgramPattern : public OpRewritePattern<torq_hl::CallProgramOp> {
  public:
    CallProgramPattern(MLIRContext *ctx, llvm::DenseMap<StringAttr, Value> &itcmCodeCache)
        : OpRewritePattern(ctx), itcmCodeCache(itcmCodeCache) {}

    LogicalResult
    matchAndRewrite(torq_hl::CallProgramOp callOp, PatternRewriter &rewriter) const override {

        auto importOp = callOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();
        if (!importOp) {
            return callOp.emitError("call_program operand is not an import_program");
        }

        Value sharedItcmCode;
        if (importOp.getType().getExecutor() == torq_hl::Executor::CSS) {

            // Reusing the cached ITCM staging across call sites assumes the
            // cached value dominates this call. That holds today because the
            // function body is a single flat block (the staging is hoisted to a
            // function-entry import or an enclosing scf.forall). If multi-block
            // function bodies are ever introduced here, this needs a dominance
            // check before reuse to avoid producing use-before-def IR.
            //
            // The cache miss path runs exactly once per program symbol, so the
            // program_code projection is created there too: one program_code op
            // per program, hence a single XRAM placement, without a second cache.
            StringAttr programSymbol = importOp.getNameAttr().getAttr();
            auto it = itcmCodeCache.find(programSymbol);
            if (it != itcmCodeCache.end()) {
                sharedItcmCode = it->second;
            }
            else {
                auto maybeProgramCode = createProgramCode(rewriter, importOp);
                if (failed(maybeProgramCode)) {
                    return failure();
                }
                Value programCode = *maybeProgramCode;

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

                sharedItcmCode = stageCssCodeSection(
                    rewriter, hoistAnchor->getLoc(), cast<TypedValue<MemRefType>>(programCode)
                );

                itcmCodeCache.try_emplace(programSymbol, sharedItcmCode);
            }
        }

        return lowerCallProgramOp(rewriter, callOp, sharedItcmCode);
    }

  private:
    llvm::DenseMap<StringAttr, Value> &itcmCodeCache;
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
    llvm::DenseMap<StringAttr, Value> itcmCodeCache;

    RewritePatternSet patterns(ctx);
    patterns.add<CallProgramPattern>(ctx, itcmCodeCache);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLowerCallProgramToStartWaitPass() {
    return std::make_unique<LowerCallProgramToStartWaitPass>();
}

} // namespace mlir::syna::torq
