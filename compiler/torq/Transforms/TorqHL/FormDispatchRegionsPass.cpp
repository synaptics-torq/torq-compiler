// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

#define DEBUG_TYPE "torq-form-dispatch-regions"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq_hl {

class FormDispatchRegionsPass : public impl::FormDispatchRegionsBase<FormDispatchRegionsPass> {
  public:
    FormDispatchRegionsPass() = default;
    FormDispatchRegionsPass(const FormDispatchRegionsPass &pass) {}
    FormDispatchRegionsPass(const FormDispatchRegionsOptions &options)
        : FormDispatchRegionsBase(options) {}

    void runOnOperation() override;

  private:
    mlir::FunctionOpInterface funcOp;
    SmallVector<SmallVector<Operation *>> fusionGroups;

    LogicalResult formMultipleDispatches();
    LogicalResult formSingleDispatch();
};

bool willBeFusedWithConsumer(const Operation *op) {
    if (!op)
        return false;
    return TypeSwitch<const Operation *, bool>(op)
        .Case<linalg::TransposeOp, torq_hl::SegmentationOp>([&](auto op) { return true; })
        .Case<torq_hl::Conv2DOp, torq_hl::DepthwiseConv2DOp, torq_hl::AddOp>([&](auto op) {
            return op.getSegmentOutput();
        })
        .Default([&](const Operation *op) { return false; });
}

LogicalResult FormDispatchRegionsPass::formMultipleDispatches() {
    IRRewriter rewriter(funcOp->getContext());

    // assign operations to a fusion group
    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (op.getDialect()->getNamespace() != torq_hl::TorqHLDialect::getDialectNamespace()) {
            continue;
        }

        if (willBeFusedWithConsumer(&op)) {
            continue;
        }

        SmallVector<Operation *> fusionGroup;

        for (auto operand : op.getOperands()) {
            if (auto operandOp = operand.getDefiningOp<linalg::TransposeOp>()) {
                fusionGroup.push_back(operandOp);
            }
            else {
                auto newOp = operand.getDefiningOp();
                if (!willBeFusedWithConsumer(newOp))
                    continue;
                // Fuse this operation with its comsumer.
                if (auto destOp = dyn_cast<DestinationStyleOpInterface>(newOp)) {
                    LLVM_DEBUG({
                        llvm::dbgs() << "FormDispatchRegionsPass ";
                        destOp.dump();
                    });
                    // Fuse all transpose operation producers if any
                    for (auto inputValue : destOp.getDpsInputs()) {
                        if (auto opInput = inputValue.getDefiningOp<linalg::TransposeOp>()) {
                            fusionGroup.push_back(opInput);
                        }
                    }
                }
                fusionGroup.push_back(newOp);
            }
        }
        fusionGroup.push_back(&op);

        for (auto opUser : op.getUsers()) {
            if (auto transposeUser = mlir::dyn_cast<linalg::TransposeOp>(opUser)) {
                fusionGroup.push_back(opUser);
            }
        }
        fusionGroups.push_back(fusionGroup);
    }

    return success();
}

LogicalResult FormDispatchRegionsPass::formSingleDispatch() {
    IRRewriter rewriter(funcOp->getContext());
    SmallVector<Operation *> fusionGroup;

    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (isa<func::ReturnOp>(op)) {
            continue;
        }

        fusionGroup.push_back(&op);
    }

    fusionGroups.push_back(fusionGroup);

    return success();
}

void FormDispatchRegionsPass::runOnOperation() {

    funcOp = getOperation();

    if (disableDispatchFusion) {
        if (failed(formMultipleDispatches())) {
            funcOp.emitError("Failed to form multi dispatch regions");
            return signalPassFailure();
        }
    }
    else {
        if (failed(formSingleDispatch())) {
            funcOp.emitError("Failed to form single dispatch regions");
            return signalPassFailure();
        }
    }

    IRRewriter rewriter(funcOp->getContext());

    for (auto fusionGroup : fusionGroups) {
        auto regionResult = IREE::Flow::wrapOpInDispatchRegion(rewriter, fusionGroup.back());

        if (failed(regionResult)) {
            return signalPassFailure();
        }
        for (auto op : llvm::reverse(fusionGroup)) {
            if (op == fusionGroup.back()) {
                continue;
            }
            regionResult =
                IREE::Flow::movePrecedingOpsIntoDispatchRegion(rewriter, op, *regionResult);
            if (failed(regionResult)) {
                return signalPassFailure();
            }
        }
    }
}

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createFormDispatchRegionsPass(bool disableDispatchFusion) {
    return std::make_unique<FormDispatchRegionsPass>(
        FormDispatchRegionsOptions{disableDispatchFusion}
    );
}

} // namespace mlir::syna::torq_hl
