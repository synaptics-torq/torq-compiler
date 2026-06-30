// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-inline-dense-resources"

// Inline DenseResourceElementsAttr constants as DenseElementsAttr.
//
// Motivation: the IREE FX importer materializes bf16 weights as
// DenseResourceElementsAttr, e.g.
//
//   "torch.vtensor.literal"() <{value = dense_resource<torch_tensor_96_3_7_7_torch.bfloat16>
//                              : tensor<96x3x7x7xbf16>}>
//     : () -> !torch.vtensor<[96,3,7,7],bf16>
//
// When these survive into the Torq backend, the serializer aborts while
// processing the corresponding memref::GlobalOp (Serializer::processGlobalOp):
//   cast<mlir::ShapedType>(mlir::BaseMemRefType) failed
// because the global's type does not match what the serializer expects.
// Inlining the data as DenseElementsAttr avoids that code path entirely.
//
// This is a workaround for torch-mlir issue #3840. Remove it once the
// upstream FX importer can emit inline DenseElementsAttr (or once the
// Torq backend correctly serializes DenseResourceElementsAttr constants).

namespace mlir::syna::torq_hl {

namespace {

class InlineDenseResourcesPass : public impl::InlineDenseResourcesBase<InlineDenseResourcesPass> {
  public:
    using InlineDenseResourcesBase::InlineDenseResourcesBase;

    void runOnOperation() override {
        llvm::DenseMap<Attribute, Attribute> replacements;

        getOperation()->walk([&](Operation *op) {
            bool updated = false;
            SmallVector<NamedAttribute> attrs(op->getAttrs());
            for (auto &attr : attrs) {
                if (auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(attr.getValue())) {
                    auto it = replacements.find(resourceAttr);
                    if (it != replacements.end()) {
                        attr.setValue(it->second);
                        updated = true;
                        continue;
                    }

                    auto replacement = convertResourceAttr(resourceAttr);
                    if (replacement) {
                        attr.setValue(replacement);
                        replacements[resourceAttr] = replacement;
                        updated = true;
                    }
                }
            }
            if (updated)
                op->setAttrs(attrs);
        });
    }

    static DenseElementsAttr convertResourceAttr(DenseResourceElementsAttr resourceAttr) {
        auto shapedType = cast<ShapedType>(resourceAttr.getType());
        auto elementType = shapedType.getElementType();

        auto blob = resourceAttr.getRawHandle().getBlob();
        if (!blob)
            return {};
        ArrayRef<char> data = blob->getData();

        // Handle i1 specially: DenseResourceElementsAttr stores booleans as bytes,
        // but DenseElementsAttr expects APInt(1) per element.
        if (elementType.isInteger(1)) {
            llvm::SmallVector<APInt> boolValues;
            boolValues.reserve(shapedType.getNumElements());
            for (char byte : data) {
                boolValues.push_back(APInt(1, byte & 1));
            }
            return DenseElementsAttr::get(shapedType, boolValues);
        }

        // For all other types, use getFromRawBuffer.
        return DenseElementsAttr::getFromRawBuffer(shapedType, data);
    }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createInlineDenseResourcesPass() {
    return std::make_unique<InlineDenseResourcesPass>();
}

} // namespace mlir::syna::torq_hl
