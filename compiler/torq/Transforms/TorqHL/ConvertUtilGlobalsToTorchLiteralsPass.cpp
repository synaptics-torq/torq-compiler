// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#define DEBUG_TYPE "torq-convert-util-globals-to-torch-literals"

// Convert util.global constants consumed by torch_c.from_builtin_tensor into
// torch.vtensor.literal ops.
//
// Motivation: IREE-Turbine exports constant weights as util.global initial
// values, e.g.
//
//   util.global private @__auto.fc.weight = dense<...> : tensor<...xbf16>
//
// Those globals hold dense constant values, but the surrounding IR still uses
// util.global.load + torch_c.from_builtin_tensor.
// The torch backend pipeline was written for torch.vtensor.literal constants and
// does not reliably lower util.global / torch_c.from_builtin_tensor patterns.
// This pass folds such globals into torch.vtensor.literal so the rest of the
// torch pipeline sees the same IR shape as a non-turbine export.
//
// TODO: This is a workaround. Once Torq's compiler and runtime support
// util.global constants directly, this pass should be removed.

using namespace mlir::iree_compiler;
using namespace mlir::torch;

namespace mlir::syna::torq_hl {

namespace {

static DenseElementsAttr convertInitialValueToDenseElements(Attribute attr) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr))
        return denseAttr;
    if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(attr)) {
        auto shapedType = cast<ShapedType>(resourceAttr.getType());
        auto blob = resourceAttr.getRawHandle().getBlob();
        if (blob)
            return DenseElementsAttr::getFromRawBuffer(shapedType, blob->getData());
    }
    return {};
}

class ConvertUtilGlobalsToTorchLiteralsPass
    : public impl::ConvertUtilGlobalsToTorchLiteralsBase<ConvertUtilGlobalsToTorchLiteralsPass> {
  public:
    using ConvertUtilGlobalsToTorchLiteralsBase<
        ConvertUtilGlobalsToTorchLiteralsPass>::ConvertUtilGlobalsToTorchLiteralsBase;

    void runOnOperation() override {
        ModuleOp module = getOperation();

        // Build a map from global name to its dense initial value.
        DenseMap<StringRef, DenseElementsAttr> globalInitialValues;
        for (auto globalOp : module.getOps<IREE::Util::GlobalOpInterface>()) {
            if (!globalOp.isGlobalPrivate())
                continue;
            auto initialValueAttr = globalOp.getGlobalInitialValue();
            if (!initialValueAttr)
                continue;
            auto denseAttr = convertInitialValueToDenseElements(initialValueAttr);
            if (!denseAttr)
                continue;
            globalInitialValues[globalOp.getGlobalName().getValue()] = denseAttr;
        }
        if (globalInitialValues.empty())
            return;

        // Collect util.global.load ops that feed a single
        // torch_c.from_builtin_tensor. We must not erase operations while walking,
        // so gather everything first and mutate in a second pass.
        struct LoadUse {
            IREE::Util::GlobalLoadOpInterface loadOp;
            TorchConversion::FromBuiltinTensorOp fromBuiltinOp;
            Torch::ValueTensorType vtensorType;
            DenseElementsAttr value;
        };
        SmallVector<LoadUse> loadUses;
        module.walk([&](IREE::Util::GlobalLoadOpInterface loadOp) {
            StringRef globalName = loadOp.getGlobalName();
            auto it = globalInitialValues.find(globalName);
            if (it == globalInitialValues.end())
                return;
            if (!loadOp->hasOneUse())
                return;

            auto fromBuiltinOp =
                dyn_cast<TorchConversion::FromBuiltinTensorOp>(*loadOp->user_begin());
            if (!fromBuiltinOp)
                return;

            auto resultType = fromBuiltinOp.getResult().getType();
            auto vtensorType = dyn_cast<Torch::ValueTensorType>(resultType);
            if (!vtensorType)
                return;

            loadUses.push_back(LoadUse{loadOp, fromBuiltinOp, vtensorType, it->second});
        });

        for (auto &loadUse : loadUses) {
            OpBuilder builder(loadUse.fromBuiltinOp);
            auto literalOp = Torch::ValueTensorLiteralOp::create(
                builder, loadUse.fromBuiltinOp.getLoc(), loadUse.vtensorType, loadUse.value
            );
            loadUse.fromBuiltinOp.getResult().replaceAllUsesWith(literalOp.getResult());
            loadUse.fromBuiltinOp.erase();
            if (loadUse.loadOp->use_empty())
                loadUse.loadOp.erase();
        }

        // Find globals that still have loads referencing them.
        DenseSet<StringRef> globalsStillReferenced;
        module.walk([&](IREE::Util::GlobalLoadOpInterface loadOp) {
            globalsStillReferenced.insert(loadOp.getGlobalName());
        });

        // Erase globals that are no longer referenced by any load.
        SmallVector<IREE::Util::GlobalOpInterface> globalsToErase;
        for (auto globalOp : module.getOps<IREE::Util::GlobalOpInterface>()) {
            if (!globalsStillReferenced.contains(globalOp.getGlobalName().getValue()))
                globalsToErase.push_back(globalOp);
        }
        for (auto globalOp : globalsToErase) {
            globalOp.erase();
        }
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertUtilGlobalsToTorchLiteralsPass() {
    return std::make_unique<ConvertUtilGlobalsToTorchLiteralsPass>();
}

} // namespace mlir::syna::torq_hl
