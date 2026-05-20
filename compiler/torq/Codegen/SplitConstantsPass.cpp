// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-split-constants"

namespace mlir::syna::torq {

namespace {

// Slices a constant memref based on the subview parameters and returns a new flat ConstOp.
static syna::torq_hl::ConstOp sliceConstant(
    PatternRewriter &rewriter, Location loc, syna::torq_hl::ConstOp constOp,
    memref::SubViewOp subviewOp
) {
    auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());

    auto offsets = subviewOp.getStaticOffsets();
    auto strides = subviewOp.getStaticStrides();
    auto srcType = cast<MemRefType>(constOp.getType());
    auto resultType = cast<MemRefType>(subviewOp.getType());

    auto srcStrides = mlir::computeStrides(srcType.getShape());
    auto resultStrides = mlir::computeStrides(resultType.getShape());
    int64_t numElems = resultType.getNumElements();

    // TODO(perf): This lazy range avoids copying the entire source constant for every slice,
    // which is the current minimal fix for very large constants. A faster follow-up is to add a
    // guarded raw-buffer path for byte-aligned numeric DenseElementsAttr values: use getRawData(),
    // copy contiguous innermost-dimension slice rows with memcpy, then rebuild with
    // DenseElementsAttr::getFromRawBuffer(). That avoids materializing one MLIR Attribute per
    // sliced element. The same follow-up should use DenseElementsAttr::reshape() for folded
    // collapse_shape and erase dead original torq_hl.const ops after their slices replace them, so
    // serialization does not keep unused huge constants.
    auto allValues = denseAttr.getValues<Attribute>();
    SmallVector<Attribute> slicedValues;
    slicedValues.reserve(numElems);

    for (int64_t linear = 0; linear < numElems; ++linear) {
        auto dimIndices = delinearize(linear, resultStrides);
        for (auto [idx, off, stride] : llvm::zip(dimIndices, offsets, strides))
            idx = off + idx * stride;
        slicedValues.push_back(allValues[linearize(dimIndices, srcStrides)]);
    }

    auto flatResultType = MemRefType::get(resultType.getShape(), resultType.getElementType());

    auto slicedAttr = DenseElementsAttr::get(
        RankedTensorType::get(flatResultType.getShape(), flatResultType.getElementType()),
        slicedValues
    );
    return syna::torq_hl::ConstOp::create(rewriter, loc, flatResultType, slicedAttr);
}

// Replaces the uses of oldVal with newVal and propagates type changes.
static void replaceUsesAndPropagateType(Value oldVal, Value newVal, PatternRewriter &rewriter) {
    for (auto &use : llvm::make_early_inc_range(oldVal.getUses())) {
        Operation *owner = use.getOwner();

        if (auto subviewUser = dyn_cast<memref::SubViewOp>(owner)) {
            rewriter.setInsertionPoint(subviewUser);
            auto constOp = newVal.getDefiningOp<syna::torq_hl::ConstOp>();
            bool isStatic = subviewUser.getOffsets().empty() && subviewUser.getSizes().empty() &&
                            subviewUser.getStrides().empty();

            if (constOp && isStatic) {
                auto foldedConst =
                    sliceConstant(rewriter, subviewUser.getLoc(), constOp, subviewUser);
                replaceUsesAndPropagateType(
                    subviewUser.getResult(), foldedConst.getResult(), rewriter
                );
                rewriter.eraseOp(subviewUser);
                continue;
            }

            auto newSubview = memref::SubViewOp::create(
                rewriter, subviewUser.getLoc(), newVal, subviewUser.getMixedOffsets(),
                subviewUser.getMixedSizes(), subviewUser.getMixedStrides()
            );
            replaceUsesAndPropagateType(subviewUser.getResult(), newSubview.getResult(), rewriter);
            rewriter.eraseOp(subviewUser);
            continue;
        }

        if (auto collapseUser = dyn_cast<memref::CollapseShapeOp>(owner)) {
            rewriter.setInsertionPoint(collapseUser);
            auto constOp = newVal.getDefiningOp<syna::torq_hl::ConstOp>();
            if (constOp) {
                auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());
                auto collapsedType = MemRefType::get(
                    cast<MemRefType>(collapseUser.getType()).getShape(),
                    cast<MemRefType>(constOp.getType()).getElementType()
                );
                auto collapsedAttr = DenseElementsAttr::get(
                    RankedTensorType::get(collapsedType.getShape(), collapsedType.getElementType()),
                    llvm::to_vector(denseAttr.getValues<Attribute>())
                );
                auto foldedConst = syna::torq_hl::ConstOp::create(
                    rewriter, collapseUser.getLoc(), collapsedType, collapsedAttr
                );
                replaceUsesAndPropagateType(
                    collapseUser.getResult(), foldedConst.getResult(), rewriter
                );
                rewriter.eraseOp(collapseUser);
                continue;
            }

            auto oldResultType = cast<MemRefType>(collapseUser.getType());
            auto newResultType =
                MemRefType::get(oldResultType.getShape(), oldResultType.getElementType());
            auto newCollapse = memref::CollapseShapeOp::create(
                rewriter, collapseUser.getLoc(), newResultType, newVal,
                collapseUser.getReassociationIndices()
            );
            replaceUsesAndPropagateType(
                collapseUser.getResult(), newCollapse.getResult(), rewriter
            );
            rewriter.eraseOp(collapseUser);

            continue;
        }

        if (auto expandUser = dyn_cast<memref::ExpandShapeOp>(owner)) {
            rewriter.setInsertionPoint(expandUser);

            auto oldResultType = cast<MemRefType>(expandUser.getType());
            auto newResultType =
                MemRefType::get(oldResultType.getShape(), oldResultType.getElementType());
            auto newExpand = memref::ExpandShapeOp::create(
                rewriter, expandUser.getLoc(), newResultType, newVal,
                expandUser.getReassociationIndices()
            );
            replaceUsesAndPropagateType(expandUser.getResult(), newExpand.getResult(), rewriter);
            rewriter.eraseOp(expandUser);

            continue;
        }

        if (auto loadUser = dyn_cast<syna::torq_hl::LoadOp>(owner)) {
            if (newVal.getDefiningOp<syna::torq_hl::ConstOp>()) {
                rewriter.setInsertionPoint(loadUser);
                if (succeeded(
                        createLoadOp(rewriter, loadUser.getLoc(), newVal, loadUser.getOutput())
                    )) {
                    rewriter.eraseOp(loadUser);
                    continue;
                }
                // If we couldn't create the load, we fall back to just changing its operand
            }
        }

        // Fall back to in-place operand replacement for non-view-like ops
        rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
    }
}

struct SplitConstSubView : OpRewritePattern<memref::SubViewOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::SubViewOp subviewOp, PatternRewriter &rewriter) const override {
        auto constOp = subviewOp.getSource().getDefiningOp<syna::torq_hl::ConstOp>();
        if (!constOp)
            return failure();

        if (!subviewOp.getOffsets().empty() || !subviewOp.getSizes().empty() ||
            !subviewOp.getStrides().empty()) {
            return failure();
        }

        auto newConst = sliceConstant(rewriter, subviewOp.getLoc(), constOp, subviewOp);
        if (!newConst)
            return failure();

        replaceUsesAndPropagateType(subviewOp.getResult(), newConst.getResult(), rewriter);
        rewriter.eraseOp(subviewOp);
        return success();
    }
};

struct SplitConstantsPass : public impl::SplitConstantsBase<SplitConstantsPass> {
    void runOnOperation() override {
        auto funcOp = getOperation();
        RewritePatternSet patterns(funcOp->getContext());
        patterns.add<SplitConstSubView>(funcOp->getContext());
        // TODO: add a pattern for "%0 = torq_hl.const ...; %1 = memref.collapse_shape %0"
        (void)applyPatternsGreedily(funcOp, std::move(patterns));
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSplitConstantsPass() {
    return std::make_unique<SplitConstantsPass>();
}

} // namespace mlir::syna::torq
