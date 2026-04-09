// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

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

    return;
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollLoopPass() {
    return std::make_unique<UnrollLoopPass>();
}

} // namespace mlir::syna::torq
