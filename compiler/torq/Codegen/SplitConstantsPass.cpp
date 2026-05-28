// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

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

bool isStaticSubview(memref::SubViewOp subviewOp) {
    return subviewOp.getOffsets().empty() && subviewOp.getSizes().empty() &&
           subviewOp.getStrides().empty();
}

bool isIdentityReinterpretCast(memref::ReinterpretCastOp reinterpretOp) {
    auto resultType = reinterpretOp.getType();
    if (!resultType.hasStaticShape())
        return false;

    auto offsets = reinterpretOp.getStaticOffsets();
    auto sizes = reinterpretOp.getStaticSizes();
    auto strides = reinterpretOp.getStaticStrides();
    if (offsets.size() != 1 || offsets[0] != 0)
        return false;
    if (!llvm::equal(sizes, resultType.getShape()))
        return false;
    return llvm::equal(strides, mlir::computeStrides(resultType.getShape()));
}

bool isSplittableDerivedMemRefOperation(Operation *op) {
    if (!isDerivedMemRefOperation(op))
        return false;

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op))
        return isStaticSubview(subviewOp);

    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(op))
        return isIdentityReinterpretCast(reinterpretOp);

    return true;
}

bool hasOnlySplittableUses(syna::torq_hl::ConstOp constOp) {
    for (Operation *user : constOp->getUsers()) {
        if (isSplittableDerivedMemRefOperation(user) &&
            getDerivedMemRefBase(user).get() == constOp.getResult())
            continue;

        return false;
    }
    return true;
}

FailureOr<syna::torq_hl::ConstOp> reshapeConstant(
    PatternRewriter &rewriter, Location loc, syna::torq_hl::ConstOp constOp, MemRefType resultType
) {
    if (!resultType.hasStaticShape())
        return failure();

    auto denseAttr = cast<DenseElementsAttr>(constOp.getValue());
    if (denseAttr.getNumElements() != resultType.getNumElements())
        return failure();

    auto reshapedAttr = DenseElementsAttr::get(
        RankedTensorType::get(resultType.getShape(), resultType.getElementType()),
        llvm::to_vector(denseAttr.getValues<Attribute>())
    );
    return syna::torq_hl::ConstOp::create(rewriter, loc, resultType, reshapedAttr);
}

// Slices a constant memref based on the subview parameters and returns a new flat ConstOp.
FailureOr<syna::torq_hl::ConstOp> sliceConstant(
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

MemRefType getDenseMemRefType(Type type) {
    auto memRefType = cast<MemRefType>(type);
    return MemRefType::get(memRefType.getShape(), memRefType.getElementType());
}

FailureOr<syna::torq_hl::ConstOp> foldDerivedMemRefConstant(
    PatternRewriter &rewriter, Operation *op, syna::torq_hl::ConstOp constOp
) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        if (!isStaticSubview(subviewOp))
            return failure();
        return sliceConstant(rewriter, subviewOp.getLoc(), constOp, subviewOp);
    }

    if (auto reinterpretOp = dyn_cast<memref::ReinterpretCastOp>(op)) {
        if (!isIdentityReinterpretCast(reinterpretOp))
            return failure();
        return reshapeConstant(
            rewriter, op->getLoc(), constOp, getDenseMemRefType(reinterpretOp.getType())
        );
    }

    if (auto memorySpaceCastOp = dyn_cast<memref::MemorySpaceCastOp>(op))
        return reshapeConstant(
            rewriter, op->getLoc(), constOp, cast<MemRefType>(memorySpaceCastOp.getType())
        );

    if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::ReshapeOp>(op))
        return reshapeConstant(
            rewriter, op->getLoc(), constOp, getDenseMemRefType(op->getResult(0).getType())
        );

    return failure();
}

// Replaces the uses of oldVal with newVal and propagates type changes.
void replaceUsesAndPropagateType(Value oldVal, Value newVal, PatternRewriter &rewriter) {
    for (auto &use : llvm::make_early_inc_range(oldVal.getUses())) {
        Operation *owner = use.getOwner();

        if (isSplittableDerivedMemRefOperation(owner) && &use == &getDerivedMemRefBase(owner)) {
            rewriter.setInsertionPoint(owner);
            auto constOp = newVal.getDefiningOp<syna::torq_hl::ConstOp>();
            if (constOp) {
                auto foldedConst = foldDerivedMemRefConstant(rewriter, owner, constOp);
                if (succeeded(foldedConst)) {
                    replaceUsesAndPropagateType(
                        owner->getResult(0), foldedConst->getResult(), rewriter
                    );
                    rewriter.eraseOp(owner);
                    continue;
                }
            }
        }

        if (auto subviewUser = dyn_cast<memref::SubViewOp>(owner)) {
            if (&use != &subviewUser.getSourceMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

            rewriter.setInsertionPoint(subviewUser);
            auto constOp = newVal.getDefiningOp<syna::torq_hl::ConstOp>();

            if (constOp && isStaticSubview(subviewUser)) {
                auto foldedConst =
                    sliceConstant(rewriter, subviewUser.getLoc(), constOp, subviewUser);
                if (failed(foldedConst))
                    continue;
                replaceUsesAndPropagateType(
                    subviewUser.getResult(), foldedConst->getResult(), rewriter
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
            if (&use != &collapseUser.getSrcMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

            rewriter.setInsertionPoint(collapseUser);
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
            if (&use != &expandUser.getSrcMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

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

        if (auto reinterpretUser = dyn_cast<memref::ReinterpretCastOp>(owner)) {
            if (&use != &reinterpretUser.getSourceMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

            rewriter.setInsertionPoint(reinterpretUser);
            auto newReinterpret = memref::ReinterpretCastOp::create(
                rewriter, reinterpretUser.getLoc(), reinterpretUser.getType(), newVal,
                reinterpretUser.getOffsets(), reinterpretUser.getSizes(),
                reinterpretUser.getStrides(), reinterpretUser.getStaticOffsets(),
                reinterpretUser.getStaticSizes(), reinterpretUser.getStaticStrides()
            );
            replaceUsesAndPropagateType(
                reinterpretUser.getResult(), newReinterpret.getResult(), rewriter
            );
            rewriter.eraseOp(reinterpretUser);
            continue;
        }

        if (auto memorySpaceCastUser = dyn_cast<memref::MemorySpaceCastOp>(owner)) {
            if (&use != &memorySpaceCastUser.getSourceMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

            rewriter.setInsertionPoint(memorySpaceCastUser);
            auto newMemorySpaceCast = memref::MemorySpaceCastOp::create(
                rewriter, memorySpaceCastUser.getLoc(), memorySpaceCastUser.getType(), newVal
            );
            replaceUsesAndPropagateType(
                memorySpaceCastUser.getResult(), newMemorySpaceCast.getResult(), rewriter
            );
            rewriter.eraseOp(memorySpaceCastUser);
            continue;
        }

        if (auto reshapeUser = dyn_cast<memref::ReshapeOp>(owner)) {
            if (&use != &reshapeUser.getSourceMutable()) {
                rewriter.modifyOpInPlace(owner, [&]() { use.set(newVal); });
                continue;
            }

            rewriter.setInsertionPoint(reshapeUser);
            auto newReshape = memref::ReshapeOp::create(
                rewriter, reshapeUser.getLoc(), getDenseMemRefType(reshapeUser.getType()), newVal,
                reshapeUser.getShape()
            );
            replaceUsesAndPropagateType(reshapeUser.getResult(), newReshape.getResult(), rewriter);
            rewriter.eraseOp(reshapeUser);
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

        if (!isStaticSubview(subviewOp)) {
            return failure();
        }

        if (!hasOnlySplittableUses(constOp)) {
            return failure();
        }

        auto newConst = sliceConstant(rewriter, subviewOp.getLoc(), constOp, subviewOp);
        if (failed(newConst))
            return failure();

        replaceUsesAndPropagateType(subviewOp.getResult(), newConst->getResult(), rewriter);
        rewriter.eraseOp(subviewOp);
        return success();
    }
};

struct SplitConstCollapseShape : OpRewritePattern<memref::CollapseShapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {
        auto constOp = collapseOp.getSrc().getDefiningOp<syna::torq_hl::ConstOp>();
        if (!constOp)
            return failure();

        if (!hasOnlySplittableUses(constOp))
            return failure();

        auto newConst = reshapeConstant(
            rewriter, collapseOp.getLoc(), constOp, getDenseMemRefType(collapseOp.getType())
        );
        if (failed(newConst))
            return failure();

        replaceUsesAndPropagateType(collapseOp.getResult(), newConst->getResult(), rewriter);
        rewriter.eraseOp(collapseOp);
        return success();
    }
};

struct SplitConstDerivedMemRef : RewritePattern {
    SplitConstDerivedMemRef(MLIRContext *context)
        : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (isa<memref::SubViewOp, memref::CollapseShapeOp>(op) ||
            !isSplittableDerivedMemRefOperation(op) || op->getNumResults() != 1)
            return failure();

        auto constOp = getDerivedMemRefBase(op).get().getDefiningOp<syna::torq_hl::ConstOp>();
        if (!constOp)
            return failure();

        if (!hasOnlySplittableUses(constOp))
            return failure();

        auto newConst = foldDerivedMemRefConstant(rewriter, op, constOp);
        if (failed(newConst))
            return failure();

        replaceUsesAndPropagateType(op->getResult(0), newConst->getResult(), rewriter);
        rewriter.eraseOp(op);
        return success();
    }
};

void eraseDeadConstants(FunctionOpInterface funcOp) {
    SmallVector<syna::torq_hl::ConstOp> deadConstants;
    funcOp.walk([&](syna::torq_hl::ConstOp constOp) {
        if (constOp->use_empty())
            deadConstants.push_back(constOp);
    });
    for (auto constOp : deadConstants)
        constOp.erase();
}

struct SplitConstantsPass : public impl::SplitConstantsBase<SplitConstantsPass> {
    void runOnOperation() override {
        auto funcOp = getOperation();
        RewritePatternSet patterns(funcOp->getContext());
        patterns.add<SplitConstSubView, SplitConstCollapseShape, SplitConstDerivedMemRef>(
            funcOp->getContext()
        );
        (void)applyPatternsGreedily(funcOp, std::move(patterns));
        eraseDeadConstants(funcOp);
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSplitConstantsPass() {
    return std::make_unique<SplitConstantsPass>();
}

} // namespace mlir::syna::torq
