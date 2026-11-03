// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/SymbolTable.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {
namespace {

class LowerCssExternalFunctionsPass
    : public impl::LowerCssExternalFunctionsBase<LowerCssExternalFunctionsPass> {
    void runOnOperation() override {
        ModuleOp module = getOperation();
        SmallVector<torq_hl::ExternalCallOp> calls;
        module.walk([&](torq_hl::ExternalCallOp call) { calls.push_back(call); });
        for (auto call : calls) {
            if (call.getNumResults()) {
                call.emitError("external CSS call must be bufferized before LLVM lowering");
                return signalPassFailure();
            }
            OpBuilder builder(call);
            auto pointerType = LLVM::LLVMPointerType::get(&getContext());
            SmallVector<Value> pointers;
            for (Value input : call.getInputs()) {
                auto type = dyn_cast<MemRefType>(input.getType());
                if (!type || !type.getLayout().isIdentity() ||
                    !type.getElementType().isIntOrFloat() || type.getElementTypeBitWidth() < 8 ||
                    type.getElementTypeBitWidth() % 8) {
                    call.emitError(
                        "external CSS calls require dense memrefs with byte-addressable elements"
                    );
                    return signalPassFailure();
                }
                auto address =
                    memref::ExtractAlignedPointerAsIndexOp::create(builder, call.getLoc(), input);
                // CSS uses 32-bit pointers.
                auto integer = arith::IndexCastOp::create(
                    builder, call.getLoc(), builder.getI32Type(), address
                );
                pointers.push_back(LLVM::IntToPtrOp::create(
                    builder, call.getLoc(), pointerType, integer.getResult(), nullptr
                ));
            }
            auto functionType = LLVM::LLVMFunctionType::get(
                LLVM::LLVMVoidType::get(&getContext()),
                SmallVector<Type>(pointers.size(), pointerType)
            );
            Operation *symbol = SymbolTable::lookupSymbolIn(module, call.getFunctionName());
            auto function = dyn_cast_or_null<LLVM::LLVMFuncOp>(symbol);
            if (symbol && (!function || function.getFunctionType() != functionType)) {
                call.emitError("conflicting external CSS function signature: ")
                    << call.getFunctionName();
                return signalPassFailure();
            }
            if (!function) {
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(module.getBody());
                function = LLVM::LLVMFuncOp::create(
                    builder, call.getLoc(), call.getFunctionName(), functionType
                );
            }
            function->setAttr("hal.import.static", builder.getUnitAttr());
            LLVM::CallOp::create(builder, call.getLoc(), function, pointers);
            call.erase();
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerCssExternalFunctionsPass() {
    return std::make_unique<LowerCssExternalFunctionsPass>();
}

} // namespace mlir::syna::torq
