// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::syna::torq {
namespace {

struct WrapperInliner : InlinerInterface {
    using InlinerInterface::InlinerInterface;

    bool isLegalToInline(Operation *, Operation *, bool) const override { return true; }
    bool isLegalToInline(Region *, Region *, bool, IRMapping &) const override { return true; }
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const override { return true; }
};

struct ApplyTorqWrapperPass : impl::ApplyTorqWrapperBase<ApplyTorqWrapperPass> {
    ApplyTorqWrapperPass() = default;
    explicit ApplyTorqWrapperPass(StringRef file) { wrapperFile = file.str(); }

    llvm::SmallPtrSet<Operation *, 8> active;
    llvm::SmallPtrSet<Operation *, 8> completed;

    LogicalResult inlineCalls(func::FuncOp function, WrapperInliner &inliner) {
        if (completed.contains(function))
            return success();
        if (!active.insert(function).second)
            return function.emitError("recursive calls are not supported by torq wrappers");

        SmallVector<func::CallOp> calls;
        function.walk([&](func::CallOp call) { calls.push_back(call); });
        for (func::CallOp call : calls) {
            auto callee =
                SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            if (!callee)
                return call.emitError("unresolved wrapper callee: ") << call.getCallee();
            if (callee.isExternal())
                continue;
            if (failed(inlineCalls(callee, inliner)))
                return failure();
            auto cloneRegion = [](OpBuilder &, Region *source, Block *, Block *after,
                                  IRMapping &mapping, bool) {
                source->cloneInto(after->getParent(), after->getIterator(), mapping);
            };
            if (failed(inlineCall(inliner, cloneRegion, call, callee, &callee.getBody())))
                return call.emitError("could not inline wrapper callee: ") << call.getCallee();
            call.erase();
        }
        active.erase(function);
        completed.insert(function);
        return success();
    }

    void runOnOperation() override {
        if (wrapperFile.empty()) {
            getOperation().emitError("torq-apply-wrapper requires wrapper-file");
            return signalPassFailure();
        }
        auto wrapper = parseSourceFile<ModuleOp>(wrapperFile, &getContext());
        if (!wrapper) {
            getOperation().emitError("could not load torq wrapper: ") << wrapperFile;
            return signalPassFailure();
        }
        ModuleOp module = getOperation();
        SmallVector<func::FuncOp> definitions;
        SmallVector<func::FuncOp> externals;
        bool hasEntryPoint = false;
        bool hasModel = false;
        for (Operation &operation : wrapper->getBody()->getOperations()) {
            auto function = dyn_cast<func::FuncOp>(operation);
            if (!function) {
                operation.emitError("torq wrapper may only contain func.func operations");
                return signalPassFailure();
            }
            Operation *existing = SymbolTable::lookupSymbolIn(module, function.getSymName());
            if (existing) {
                auto model = dyn_cast<func::FuncOp>(existing);
                if (!function.isExternal() || !model || model.isExternal()) {
                    function.emitError("wrapper symbol conflicts with model symbol: ")
                        << function.getSymName();
                    return signalPassFailure();
                }
                if (function.getFunctionType() != model.getFunctionType()) {
                    function.emitError(
                        "wrapper declaration does not match model signature; expected "
                    ) << model.getFunctionType();
                    return signalPassFailure();
                }
                hasModel = true;
            }
            else if (function.isExternal()) {
                externals.push_back(function);
            }
            else {
                definitions.push_back(function);
                hasEntryPoint |= function.isPublic();
            }
        }
        if (!hasEntryPoint || !hasModel) {
            module.emitError(
                "torq wrapper requires a public definition and a declaration of a model function"
            );
            return signalPassFailure();
        }

        wrapper->walk([&](func::CallOp call) {
            auto callee = wrapper->lookupSymbol<func::FuncOp>(call.getCallee());
            if (!llvm::is_contained(externals, callee))
                return;
            OpBuilder builder(call);
            auto external = torq_hl::ExternalCallOp::create(
                builder, call.getLoc(), call.getResultTypes(), call.getCallee(), call.getOperands()
            );
            call.replaceAllUsesWith(external.getResults());
            call.erase();
        });
        for (auto function : module.getOps<func::FuncOp>()) {
            if (!function.isExternal())
                function.setPrivate();
        }
        for (auto function : definitions)
            function->moveBefore(module.getBody(), module.getBody()->end());

        active.clear();
        completed.clear();
        WrapperInliner inliner(&getContext());
        for (auto function : definitions) {
            if (failed(inlineCalls(function, inliner)))
                return signalPassFailure();
        }
        if (failed(verify(module)))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createApplyTorqWrapperPass(StringRef wrapperFile) {
    return std::make_unique<ApplyTorqWrapperPass>(wrapperFile);
}

} // namespace mlir::syna::torq
