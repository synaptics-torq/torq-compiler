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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-unroll-loop"

namespace mlir::syna::torq {

namespace {

static void unrollLoops(scf::ForOp forOp) {

    auto lbCstOp = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ubCstOp = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto stepCstOp = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();

    if (!lbCstOp || !ubCstOp || !stepCstOp || lbCstOp.value() < 0 || ubCstOp.value() < 0 ||
        stepCstOp.value() < 0) {
        return;
    }

    int64_t tripCount = llvm::divideCeil(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());

    (void)loopUnrollByFactor(forOp, tripCount);
}

// propagateStaticOffsetToExpandShapeChain() looks for the following pattern of ops and propagates
// static offset from memref.expand_shape input chain to all dynamic-offset ops in the chain.
//
// This pattern will be transformed from:
//
// %subview_50 = memref.subview %16[0, 32, 0] [1, 32, 2100] [1, 1, 1] : memref<1x64x2100xi8> to
// memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: 67200>>
// %cast_51 = memref.cast %subview_50 : memref<1x32x2100xi8, strided<[134400, 2100, 1], offset:
// 67200>> to memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: ?>>
// %expand_shape_52 = memref.expand_shape %cast_51 [[0], [1, 2], [3]] output_shape [1, 2, 16, 2100]
// : memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: 67200>> into memref<1x2x16x2100xi8,
// strided<[134400, 33600, 2100, 1], offset: ?>>
//
// to:
//
// %subview_50 = memref.subview %16[0, 32, 0] [1, 32, 2100] [1, 1, 1] : memref<1x64x2100xi8> to
// memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: 67200>>
// %cast_51 = memref.cast %subview_50 : memref<1x32x2100xi8, strided<[134400, 2100, 1], offset:
// 67200>> to memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: 67200>>
// %expand_shape_52 = memref.expand_shape %cast_51 [[0], [1, 2], [3]] output_shape [1, 2, 16, 2100]
// : memref<1x32x2100xi8, strided<[134400, 2100, 1], offset: 67200>> into memref<1x2x16x2100xi8,
// strided<[134400, 33600, 2100, 1], offset: 67200>>
//
void propagateStaticOffsetToExpandShapeChain(mlir::FunctionOpInterface funcOp) {
    funcOp->walk([&](mlir::memref::ExpandShapeOp expandOp) {
        auto castOp =
            llvm::dyn_cast_or_null<mlir::memref::CastOp>(expandOp.getSrc().getDefiningOp());
        if (!castOp)
            return;
        auto subviewOp =
            llvm::dyn_cast_or_null<mlir::memref::SubViewOp>(castOp.getSource().getDefiningOp());
        if (!subviewOp)
            return;

        auto subviewType = mlir::dyn_cast<mlir::MemRefType>(subviewOp.getResult().getType());
        auto castType = mlir::dyn_cast<mlir::MemRefType>(castOp.getResult().getType());
        if (!subviewType || !castType)
            return;

        auto subviewLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(subviewType.getLayout());
        auto castLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(castType.getLayout());
        if (!subviewLayout || !castLayout)
            return;

        // Only update if subview has static offset and cast has dynamic offset
        if (subviewLayout.getOffset() != mlir::ShapedType::kDynamic &&
            castLayout.getOffset() == mlir::ShapedType::kDynamic) {
            auto newLayout = mlir::StridedLayoutAttr::get(
                castOp.getContext(), subviewLayout.getOffset(), castLayout.getStrides()
            );
            auto newType = mlir::MemRefType::get(
                castType.getShape(), castType.getElementType(), newLayout, castType.getMemorySpace()
            );
            castOp.getResult().setType(newType);
            // Update only the offset for expand_shape result type if layout is strided and offset
            // is dynamic
            auto expandType = mlir::cast<mlir::MemRefType>(expandOp.getType());
            auto expandLayout = mlir::dyn_cast<mlir::StridedLayoutAttr>(expandType.getLayout());
            if (expandLayout && expandLayout.getOffset() == mlir::ShapedType::kDynamic) {
                auto newExpandLayout = mlir::StridedLayoutAttr::get(
                    expandOp.getContext(), subviewLayout.getOffset(), expandLayout.getStrides()
                );
                auto expandNewType = mlir::MemRefType::get(
                    expandType.getShape(), expandType.getElementType(), newExpandLayout,
                    expandType.getMemorySpace()
                );
                expandOp.getResult().setType(expandNewType);
            }
        }
    });
}

// Matches memref.subview(torq_hl.const) with fully-static unit-stride offsets
// and replaces it with a new torq_hl.const whose data is the sliced sub-region.
// The subview's result type may carry an identity strided layout; the new const
// always uses a flat type so no cast is needed by downstream consumers.
struct SplitConstSubView : OpRewritePattern<memref::SubViewOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::SubViewOp subviewOp, PatternRewriter &rewriter) const override {
        auto constOp = subviewOp.getSource().getDefiningOp<syna::torq_hl::ConstOp>();
        if (!constOp)
            return failure();
        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!denseAttr)
            return failure();

        auto offsets = subviewOp.getStaticOffsets();
        auto strides = subviewOp.getStaticStrides();
        if (!subviewOp.getOffsets().empty() || !subviewOp.getSizes().empty() ||
            !subviewOp.getStrides().empty()) {
            LLVM_DEBUG(llvm::dbgs() << "[SplitConstSubView] skip " << subviewOp.getLoc()
                                    << ": dynamic offsets/sizes/strides\n";
                       subviewOp.dump());
            return failure();
        }
        auto srcType = cast<MemRefType>(constOp.getType());
        auto resultType = cast<MemRefType>(subviewOp.getType());

        LLVM_DEBUG(llvm::dbgs() << "[SplitConstSubView] processing " << subviewOp.getLoc()
                                << ": src=" << srcType << " result=" << resultType << " offsets=[";
                   llvm::interleaveComma(offsets, llvm::dbgs()); llvm::dbgs() << "]\n");

        auto srcStrides = mlir::computeStrides(srcType.getShape());
        auto resultStrides = mlir::computeStrides(resultType.getShape());
        int64_t numElems = resultType.getNumElements();

        auto allValues = llvm::to_vector(denseAttr.getValues<Attribute>());
        SmallVector<Attribute> slicedValues;
        slicedValues.reserve(numElems);

        for (int64_t linear = 0; linear < numElems; ++linear) {
            auto dimIndices = delinearize(linear, resultStrides);
            for (auto [idx, off, stride] : llvm::zip(dimIndices, offsets, strides))
                idx = off + idx * stride;
            slicedValues.push_back(allValues[linearize(dimIndices, srcStrides)]);
        }

        auto flatResultType = MemRefType::get(
            resultType.getShape(), resultType.getElementType()
        ); // Turn to flat memref type without layout for the new const

        auto slicedAttr = DenseElementsAttr::get(
            RankedTensorType::get(flatResultType.getShape(), flatResultType.getElementType()),
            slicedValues
        );
        auto newConst = syna::torq_hl::ConstOp::create(
            rewriter, subviewOp.getLoc(), flatResultType, slicedAttr
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[SplitConstSubView] created torq_hl.const " << flatResultType
                         << " with " << numElems << " elements\n"
        );

        rewriter.replaceOp(subviewOp, newConst);
        return success();
    }
};

// Matches memref.collapse_shape(torq_hl.const) and replaces the pair with a
// single torq_hl.const whose shape is the collapsed shape. collapse_shape is a
// pure reshape — the element data is identical.
struct FoldConstCollapseShape : OpRewritePattern<memref::CollapseShapeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::CollapseShapeOp collapseOp, PatternRewriter &rewriter) const override {
        auto constOp = collapseOp.getSrc().getDefiningOp<syna::torq_hl::ConstOp>();
        if (!constOp)
            return failure();
        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!denseAttr)
            return failure();

        auto collapsedType = MemRefType::get(
            cast<MemRefType>(collapseOp.getType()).getShape(),
            cast<MemRefType>(constOp.getType()).getElementType()
        );
        auto collapsedAttr = DenseElementsAttr::get(
            RankedTensorType::get(collapsedType.getShape(), collapsedType.getElementType()),
            llvm::to_vector(denseAttr.getValues<Attribute>())
        );
        auto foldedConst = syna::torq_hl::ConstOp::create(
            rewriter, collapseOp.getLoc(), collapsedType, collapsedAttr
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[FoldConstCollapseShape] folded " << collapseOp.getLoc()
                         << " into torq_hl.const " << collapsedType << "\n"
        );

        rewriter.replaceOp(collapseOp, foldedConst);
        return success();
    }
};

// When SplitConstSubView replaces a strided subview with a flat torq_hl.const,
// the existing torq_hl.load still carries input_strides_bytes/shape/element_size_bytes
// computed for the original (padded/strided) source. This pattern detects such a
// load (input defined by a torq_hl.const) and recreates it via createLoadOp so
// the attributes are derived from the actual (now-flat) input and destination types,
// using the same stride-collapsing algorithm as bufferisation.
// Only loads into dense lram outputs are handled; non-dense outputs would need a
// new alloc, which would leave the original output buffer unwritten.
struct FixConstLoadStrides : OpRewritePattern<syna::torq_hl::LoadOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(syna::torq_hl::LoadOp loadOp, PatternRewriter &rewriter) const override {
        if (!loadOp.getInput().getDefiningOp<syna::torq_hl::ConstOp>())
            return failure();

        auto inputType = cast<MemRefType>(loadOp.getInput().getType());
        auto outputType = cast<MemRefType>(loadOp.getOutput().getType());

        // Compute what the correct attributes should be for this input/output pair.
        SmallVector<int64_t> expectedFromStrides, expectedToStrides, expectedShape;
        int64_t expectedElemSizeBytes;
        if (failed(computeStrides(
                inputType, outputType, expectedFromStrides, expectedToStrides, expectedShape,
                expectedElemSizeBytes
            )))
            return failure();

        // Already correct — returning failure() here prevents an infinite loop since
        // createLoadOp below emits a new load with the same const input.
        if (expectedShape == loadOp.getShape() &&
            expectedFromStrides == loadOp.getInputStridesBytes() &&
            expectedElemSizeBytes == loadOp.getElementSizeBytes())
            return failure();

        Value input = loadOp.getInput();
        Value output = loadOp.getOutput();
        Location loc = loadOp.getLoc();

        // Set insertion point before creating new ops, then erase the stale load.
        rewriter.setInsertionPoint(loadOp);
        if (failed(createLoadOp(rewriter, loc, input, output)))
            return failure();

        LLVM_DEBUG(llvm::dbgs() << "[FixConstLoadStrides] recreated load at " << loc << "\n");
        rewriter.eraseOp(loadOp);
        return success();
    }
};

// For each torq_hl.const memref whose users include memref.subview ops with
// fully-static unit-stride offsets, materialise a new smaller torq_hl.const
// directly for each subview and replace the subview with it.
// This runs before canonicalization so that downstream constant-fold patterns
// see scalar/small constants rather than subviews of a large one.
static void splitConstantsAtExtractSlice(mlir::FunctionOpInterface funcOp) {
    RewritePatternSet patterns(funcOp->getContext());
    patterns.add<SplitConstSubView, FoldConstCollapseShape, FixConstLoadStrides>(funcOp->getContext(
    ));
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
}

class UnrollLoopPass : public impl::UnrollLoopBase<UnrollLoopPass> {
  public:
    using UnrollLoopBase<UnrollLoopPass>::UnrollLoopBase;

    void runOnOperation() override;
};

void UnrollLoopPass::runOnOperation() {
    auto funcOp = getOperation();

    scf::ForOp innermostForOp;

    while (1) {
        funcOp->walk([&](scf::ForOp forOp) {
            auto nestedForOps = forOp.getOps<scf::ForOp>();
            if (nestedForOps.empty()) {
                innermostForOp = forOp;
            }
        });

        if (!innermostForOp) {
            break;
        }

        unrollLoops(innermostForOp);
        innermostForOp = nullptr;
    }

    // Canonicalize the function after unrolling using the canonicalizer pass
    auto &ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    for (auto *dialect : ctx.getLoadedDialects())
        dialect->getCanonicalizationPatterns(patterns);
    for (mlir::RegisteredOperationName op : ctx.getRegisteredOperations())
        op.getCanonicalizationPatterns(patterns, &ctx);
    (void)mlir::applyPatternsGreedily(funcOp, std::move(patterns));

    // add post process
    propagateStaticOffsetToExpandShapeChain(funcOp);
    // Split large tensor constants at their extract_slice users
    splitConstantsAtExtractSlice(funcOp);

    return;
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollLoopPass() {
    return std::make_unique<UnrollLoopPass>();
}

} // namespace mlir::syna::torq
