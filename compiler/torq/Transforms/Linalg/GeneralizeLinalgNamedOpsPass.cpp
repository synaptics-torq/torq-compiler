// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::syna::torq {

class GeneralizeLinalgNamedOpsPass
    : public GeneralizeLinalgNamedOpsBase<GeneralizeLinalgNamedOpsPass> {
  public:
    using GeneralizeLinalgNamedOpsBase::GeneralizeLinalgNamedOpsBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        SmallVector<linalg::LinalgOp> namedOpCandidates;
        funcOp.walk([&](linalg::LinalgOp linalgOp) {
            if (linalgOp.getOperation() && !isa<linalg::GenericOp>(linalgOp.getOperation())) {

                namedOpCandidates.push_back(linalgOp);
            }
        });

        IRRewriter rewriter(&getContext());
        for (auto linalgOp : namedOpCandidates) {
            rewriter.setInsertionPoint(linalgOp);
            FailureOr<linalg::GenericOp> generalizedOp =
                linalg::generalizeNamedOp(rewriter, linalgOp);
            if (failed(generalizedOp)) {
                linalgOp->emitOpError("failed to generalize operation");
                return signalPassFailure();
            }
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createGeneralizeLinalgNamedOpsPass() {
    return std::make_unique<GeneralizeLinalgNamedOpsPass>();
}

} // namespace mlir::syna::torq
