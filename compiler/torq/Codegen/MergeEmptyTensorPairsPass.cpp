// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// MergeEmptyTensorPairsPass
// -------------------------
// EliminateEmptyTensorsPass (upstream, iree-eliminate-empty-tensors) walks
// backward from a tensor.insert_slice's source operand looking for a
// tensor.empty() to redirect into the insert_slice's destination, but its
// traversal explicitly stops at tensor.extract_slice (it does not support
// "slices or reshapes" on the path, see EmptyTensorElimination.cpp). When a
// destination-style op's init operand is itself an extract_slice of a
// tensor.empty() (as happens for the first tile of an accumulation loop,
// where there is no prior partial result to slice from), the elimination
// pass gives up, leaving TWO separate tensor.empty() ops: one feeding the
// extract_slice (the op's init) and one used only as the insert_slice's
// destination — instead of the single, reused tensor.empty() that the
// pattern morally represents.
//
// This pass detects that exact shape:
//   %initEmpty = tensor.empty() : T
//   %slice = tensor.extract_slice %initEmpty[offsets][sizes][strides] : T -> T'
//   %result = SomeDpsOp(..., init: %slice, ...) -> T'
//   %destEmpty = tensor.empty() : T
//   %inserted = tensor.insert_slice %result into %destEmpty[offsets][sizes][strides]
// and merges %destEmpty into %initEmpty (they are two freshly-allocated,
// single-use, identically-typed placeholders for the exact same region), so
// downstream passes see the ordinary self-consistent extract-compute-insert
// pattern again.

#include "PassesDetail.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-merge-empty-tensor-pairs"

namespace mlir::syna::torq {

namespace {

class MergeEmptyTensorPair : public OpRewritePattern<tensor::InsertSliceOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::InsertSliceOp insertOp, PatternRewriter &rewriter) const override {

        auto destEmpty = insertOp.getDest().getDefiningOp<tensor::EmptyOp>();
        if (!destEmpty || !destEmpty->hasOneUse())
            return failure();

        auto dpsOp = insertOp.getSource().getDefiningOp<DestinationStyleOpInterface>();
        if (!dpsOp)
            return failure();

        auto opResult = cast<OpResult>(insertOp.getSource());
        OpOperand *tiedInit = dpsOp.getTiedOpOperand(opResult);
        if (!tiedInit)
            return failure();

        auto extractOp = tiedInit->get().getDefiningOp<tensor::ExtractSliceOp>();
        if (!extractOp)
            return failure();

        auto initEmpty = extractOp.getSource().getDefiningOp<tensor::EmptyOp>();
        if (!initEmpty || !initEmpty->hasOneUse())
            return failure();

        if (initEmpty.getType() != destEmpty.getType())
            return failure();

        // The extract and insert must operate on the exact same region for the
        // merge to be a no-op semantically.
        if (!llvm::equal(extractOp.getMixedOffsets(), insertOp.getMixedOffsets()) ||
            !llvm::equal(extractOp.getMixedSizes(), insertOp.getMixedSizes()) ||
            !llvm::equal(extractOp.getMixedStrides(), insertOp.getMixedStrides()))
            return failure();

        LLVM_DEBUG(
            llvm::dbgs() << "MergeEmptyTensorPair: merging " << destEmpty << " into " << initEmpty
                         << "\n"
        );

        rewriter.modifyOpInPlace(insertOp, [&] {
            insertOp.getDestMutable().assign(initEmpty.getResult());
        });
        rewriter.eraseOp(destEmpty);
        return success();
    }
};

class MergeEmptyTensorPairsPass
    : public impl::MergeEmptyTensorPairsBase<MergeEmptyTensorPairsPass> {
  public:
    void runOnOperation() override {
        RewritePatternSet patterns(&getContext());
        patterns.add<MergeEmptyTensorPair>(&getContext());
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMergeEmptyTensorPairsPass() {
    return std::make_unique<MergeEmptyTensorPairsPass>();
}

} // namespace mlir::syna::torq
