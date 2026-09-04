// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// FuseTransposeIntoGroupPass
// --------------------------
// Copies the fuse group of a fuse group principal onto a producer
// linalg.transpose, walking through a tensor.pad if one sits between them.
// LinalgSlicing then folds the transpose into the principal's scf.forall.

#include "PassesDetail.h"

#include "TilingUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::syna::torq {

namespace {

// Give a producer transpose the fuse group of the op it feeds.
//
// Only a transpose with a single user and a static result shape is taken.
//
// tensor.pad between the transpose and the consumer is walked through. The
// control function in LinalgSlicingPass fuses a pad producer when the pads on
// the sliced dimension are zero.
//
// The consumer must slice the operand the transpose feeds. A fused producer only
// gets the slice its consumer tile reads, so when the slicing dimension does not
// index that operand every tile reads the whole transpose result and it stays
// live for the whole loop. On a 1x1 conv, sliced on output channels, that is the
// full NCHW input and LRAM overflows.
void fuseProducerTransposes(FunctionOpInterface funcOp) {
    funcOp->walk([&](Operation *consumer) {
        IntegerAttr group = isFuseGroupPrincipalOp(consumer);
        if (!group)
            return;

        // The principal must be in exactly one group; peelAndSlice asserts it.
        auto groups = consumer->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (!groups || groups.size() != 1)
            return;

        // We need the dimension LinalgSlicing will slice on, and the indexing
        // maps to tell whether an operand is sliced along it.
        auto linalgOp = dyn_cast<linalg::LinalgOp>(consumer);
        std::optional<size_t> slicingIter = getSlicingIterationDomainIndex(consumer);
        if (!linalgOp || !slicingIter)
            return;

        for (OpOperand &operand : consumer->getOpOperands()) {
            Operation *def = operand.get().getDefiningOp();
            if (auto padOp = dyn_cast_or_null<tensor::PadOp>(def))
                def = padOp.getSource().getDefiningOp();

            auto transposeOp = dyn_cast_or_null<linalg::TransposeOp>(def);
            if (!transposeOp)
                continue;
            if (!transposeOp->getResult(0).hasOneUse())
                continue;
            if (!cast<ShapedType>(transposeOp->getResult(0).getType()).hasStaticShape())
                continue;

            AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
            if (!llvm::any_of(map.getResults(), [&](AffineExpr expr) {
                    return expr.isFunctionOfDim(*slicingIter);
                }))
                continue;

            transposeOp->setAttr(TORQ_FUSE_GROUP, groups);
        }
    });
}

class FuseTransposeIntoGroupPass
    : public impl::FuseTransposeIntoGroupBase<FuseTransposeIntoGroupPass> {
  public:
    void runOnOperation() override {
        // Same guard as LinalgSlicing: nothing to split with one slice.
        if (TorqHw::get().getSliceCount() < 2)
            return;
        fuseProducerTransposes(getOperation());
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createFuseTransposeIntoGroupPass() {
    return std::make_unique<FuseTransposeIntoGroupPass>();
}

} // namespace mlir::syna::torq
