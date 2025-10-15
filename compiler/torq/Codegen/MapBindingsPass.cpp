// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-map-bindings"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

static void updateSubViewUserTypes(PatternRewriter &rewriter, Operation *op) {
    SmallVector<memref::SubViewOp> subViewUsers;
    for (auto &use : op->getUses()) {
        memref::SubViewOp subViewOp = dyn_cast_if_present<memref::SubViewOp>(use.getOwner());
        if (subViewOp) {
            subViewUsers.push_back(subViewOp);
        }
    }

    for (auto &subViewOp : subViewUsers) {

        // Update the subview the correct out type
        auto newType = memref::SubViewOp::inferRankReducedResultType(
            subViewOp.getType().getShape(), subViewOp.getSource().getType(),
            subViewOp.getStaticOffsets(), subViewOp.getStaticSizes(), subViewOp.getStaticStrides()
        );

        rewriter.modifyOpInPlace(subViewOp, [&]() {
            subViewOp.getResult().setType(cast<MemRefType>(newType));
        });

        // Recurse to also update subviews of subviews
        updateSubViewUserTypes(rewriter, subViewOp);
    }
}

class HalInterfaceBindingSubspanPattern
    : public OpRewritePattern<IREE::HAL::InterfaceBindingSubspanOp> {

  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(
        IREE::HAL::InterfaceBindingSubspanOp subspanOp, PatternRewriter &rewriter
    ) const override {

        auto origType = mlir::cast<MemRefType>(subspanOp.getResult().getType());

        // remove unused information from the output type
        auto outputType = MemRefType::get(origType.getShape(), origType.getElementType());

        auto byteOffset = mlir::cast<IntegerAttr>(
            subspanOp.getByteOffset().getDefiningOp<arith::ConstantOp>().getValueAttr()
        );

        // check that the memref type is what we expect
        if (origType.getLayout()) {
            if (auto layout = llvm::dyn_cast<AffineMapAttr>(origType.getLayout())) {
                if (!layout.isIdentity()) {
                    return subspanOp.emitError("unsupported non-identity affine layout");
                }
            }
            else if (auto layout = llvm::dyn_cast<StridedLayoutAttr>(origType.getLayout())) {
                auto elementSizeBytes = origType.getElementType().getIntOrFloatBitWidth() / 8;
                if (layout.getOffset() * elementSizeBytes != byteOffset.getValue()) {
                    return subspanOp.emitError("binding byte offset does not match layout");
                }
                auto strides = layout.getStrides();
                int lowerStride = 1;
                for (int i = strides.size() - 1; i >= 0; i--) {
                    if (strides[i] != lowerStride) {
                        return subspanOp.emitError("non natural strides are not supported");
                    }
                    lowerStride = origType.getShape()[i] * strides[i];
                }
            }
            else {
                return subspanOp.emitError("unsupported layout");
            }
        }

        bool isReadOnly = false;
        bool isWriteOnly = false;
        if (bitEnumContainsAny(
                subspanOp.getDescriptorFlags().value_or(IREE::HAL::DescriptorFlags::None),
                IREE::HAL::DescriptorFlags::ReadOnly
            )) {
            isReadOnly = true;
        }

        auto mapOp = rewriter.replaceOpWithNewOp<syna::torq_hl::MapBindingOp>(
            subspanOp, outputType, mlir::cast<IntegerAttr>(byteOffset), subspanOp.getBindingAttr(),
            rewriter.getBoolAttr(isReadOnly), rewriter.getBoolAttr(isWriteOnly)
        );

        // update the memref layout offsets of all the subviews since we changed the parent one
        updateSubViewUserTypes(rewriter, mapOp);

        return success();
    }
};

template <typename OpTy> struct ElideNoOp final : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

class MapBindingsPass : public MapBindingsBase<MapBindingsPass> {
  public:
    MapBindingsPass() = default;
    MapBindingsPass(const MapBindingsPass &pass) {}

    void runOnOperation() override;
};

void MapBindingsPass::runOnOperation() {
    auto funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();

    RewritePatternSet patterns(ctx);

    patterns.add<HalInterfaceBindingSubspanPattern>(ctx);
    patterns.add<ElideNoOp<memref::AssumeAlignmentOp>>(ctx);

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMapBindingsPass() {
    return std::make_unique<MapBindingsPass>();
}

} // namespace mlir::syna::torq
