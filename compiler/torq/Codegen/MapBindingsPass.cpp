// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <deque>

#define DEBUG_TYPE "torq-map-bindings"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

// Strip optional arith.index_cast{,ui} from v.
static Value stripIndexCast(Value v) {
    if (auto c = v.getDefiningOp<arith::IndexCastUIOp>())
        return c.getIn();
    if (auto c = v.getDefiningOp<arith::IndexCastOp>())
        return c.getIn();
    return v;
}

static Value stripAssumeAlignment(Value m) {
    while (auto a = m.getDefiningOp<memref::AssumeAlignmentOp>()) {
        m = a.getMemref();
    }
    return m;
}

static FailureOr<int32_t> traceBindingLoadIndex(Value v) {
    auto stripped = stripIndexCast(v);
    auto loadOp = stripped.getDefiningOp<memref::LoadOp>();
    if (!loadOp)
        return failure();
    if (!loadOp.getIndices().empty())
        return failure();
    // Runtime memcpys 4 bytes here; wider element types would truncate.
    if (!loadOp.getMemRefType().getElementType().isInteger(32))
        return failure();
    auto memref = stripAssumeAlignment(loadOp.getMemref());
    auto subspanOp = memref.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!subspanOp)
        return failure();
    return static_cast<int32_t>(subspanOp.getBinding().getZExtValue());
}

// Recover the push-constant ordinal that feeds a dynamic byte offset.
// Peels any number of util.assume.int (by OpResult number) and one index cast.
static FailureOr<int32_t> tracePushConstantOrdinal(Value v) {
    while (auto a = v.getDefiningOp<IREE::Util::AssumeIntOp>()) {
        auto r = dyn_cast<OpResult>(v);
        if (!r)
            return failure();
        v = a.getOperand(r.getResultNumber());
    }
    auto load = stripIndexCast(v).getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>();
    if (!load)
        return failure();
    return static_cast<int32_t>(load.getOrdinal().getZExtValue());
}

static void updateSubViewUserTypes(PatternRewriter &rewriter, Operation *op) {
    std::deque<OpOperand> stack;
    llvm::append_range(stack, op->getUsers());

    while (!stack.empty()) {
        Operation *currentOp = stack.back().getOwner();
        stack.pop_back();

        if (!currentOp) {
            continue;
        }

        bool changedOutput =
            TypeSwitch<Operation *, bool>(currentOp)
                .Case<memref::SubViewOp>([&](auto subViewOp) {
                    // Update the subview the correct out type
                    auto newType = memref::SubViewOp::inferRankReducedResultType(
                        subViewOp.getType().getShape(), subViewOp.getSource().getType(),
                        subViewOp.getStaticOffsets(), subViewOp.getStaticSizes(),
                        subViewOp.getStaticStrides()
                    );

                    rewriter.modifyOpInPlace(subViewOp, [&]() {
                        subViewOp.getResult().setType(cast<MemRefType>(newType));
                    });

                    return true;
                })
                .Case<memref::ExpandShapeOp>([&](memref::ExpandShapeOp expandOp) {
                    FailureOr<MemRefType> newType = memref::ExpandShapeOp::computeExpandedType(
                        expandOp.getSrcType(), expandOp.getStaticOutputShape(),
                        expandOp.getReassociationIndices()
                    );
                    assert(!failed(newType));

                    rewriter.modifyOpInPlace(expandOp, [&]() {
                        expandOp.getResult().setType(*newType);
                    });

                    return true;
                })
                .Case<memref::CollapseShapeOp>([&](memref::CollapseShapeOp collapseOp) {
                    auto newType = memref::CollapseShapeOp::computeCollapsedType(
                        collapseOp.getSrcType(), collapseOp.getReassociationIndices()
                    );

                    rewriter.modifyOpInPlace(collapseOp, [&]() {
                        collapseOp.getResult().setType(newType);
                    });

                    return true;
                })
                .Default([](auto) { return false; });

        if (changedOutput) {
            // Recurse to also update users of currentOp
            llvm::append_range(stack, currentOp->getUsers());
        }
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

        SmallVector<int64_t> outputShape;

        if (origType.hasStaticShape()) {
            outputShape.append(origType.getShape().begin(), origType.getShape().end());
        }
        else if (!origType.hasStaticShape()) {

            for (auto [idx, dim] : llvm::enumerate(origType.getShape())) {
                if (dim == ShapedType::kDynamic) {
                    auto dynamicDim = subspanOp.getDynamicDims()[idx];
                    if (auto constOp = dynamicDim.getDefiningOp<arith::ConstantOp>()) {
                        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValueAttr())) {
                            outputShape.push_back(intAttr.getInt());
                            continue;
                        }
                        else {
                            subspanOp.emitError("dynamic dimension must be a constant integer");
                            return failure();
                        }
                    }
                    else {
                        subspanOp.emitError("dynamic dimension must be a constant");
                        return failure();
                    }
                }
                else {
                    outputShape.push_back(dim);
                }
            }
        }

        // remove unused information from the output type
        auto outputType = MemRefType::get(outputShape, origType.getElementType());

        LLVM_DEBUG({ subspanOp.dump(); });

        bool isReadOnly = false;
        bool isWriteOnly = false; // IREE has no WriteOnly DescriptorFlag yet.
        if (bitEnumContainsAny(
                subspanOp.getDescriptorFlags().value_or(IREE::HAL::DescriptorFlags::None),
                IREE::HAL::DescriptorFlags::ReadOnly
            )) {
            isReadOnly = true;
        }

        auto byteOffsetValue = subspanOp.getByteOffset();
        auto byteOffsetConstantOp = byteOffsetValue.getDefiningOp<arith::ConstantOp>();

        if (!byteOffsetConstantOp) {
            // Route the dynamic region through a static buffer with
            // pre/post host copies (skip pre for write-only, post for read-only).
            // Subspan offset only comes from push-constant in observed IR;
            // the subview pattern below handles the binding-load source.
            auto ordinal = tracePushConstantOrdinal(byteOffsetValue);
            if (failed(ordinal)) {
                return subspanOp.emitError(
                    "dynamic byte offset must trace to a single hal.interface.constant.load"
                );
            }

            auto loc = subspanOp.getLoc();
            auto baseMap = rewriter.create<syna::torq_hl::MapBindingOp>(
                loc, outputType, rewriter.getIndexAttr(0), subspanOp.getBindingAttr(),
                rewriter.getBoolAttr(isReadOnly), rewriter.getBoolAttr(isWriteOnly)
            );
            auto destAlloc = rewriter.create<memref::AllocOp>(loc, outputType, ValueRange{});

            SmallVector<int64_t> fromStrides, toStrides, shape;
            int64_t elemSizeBytes;
            if (failed(computeStrides(
                    outputType, outputType, fromStrides, toStrides, shape, elemSizeBytes
                ))) {
                return subspanOp.emitError("failed to compute strides for dynamic host copy");
            }

            auto emitCopy = [&](Value out, Value in, syna::torq_hl::DynamicHostCopyDirection dir) {
                rewriter.create<syna::torq_hl::DynamicHostCopyOp>(
                    loc, out, in, rewriter.getDenseI64ArrayAttr(fromStrides),
                    rewriter.getDenseI64ArrayAttr(toStrides), rewriter.getDenseI64ArrayAttr(shape),
                    rewriter.getI64IntegerAttr(elemSizeBytes), rewriter.getI32IntegerAttr(*ordinal),
                    /*input_byte_offset_binding_index=*/IntegerAttr{},
                    /*index_unit_bytes=*/rewriter.getI64IntegerAttr(1),
                    syna::torq_hl::DynamicHostCopyDirectionAttr::get(rewriter.getContext(), dir)
                );
            };

            if (!isWriteOnly) {
                emitCopy(
                    destAlloc.getResult(), baseMap.getResult(),
                    syna::torq_hl::DynamicHostCopyDirection::PreDispatchRead
                );
            }
            if (!isReadOnly) {
                auto funcOp = subspanOp->getParentOfType<FunctionOpInterface>();
                OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPoint(funcOp.getFunctionBody().back().getTerminator());
                emitCopy(
                    baseMap.getResult(), destAlloc.getResult(),
                    syna::torq_hl::DynamicHostCopyDirection::PostDispatchWrite
                );
            }

            rewriter.replaceOp(subspanOp, destAlloc.getResult());
            updateSubViewUserTypes(rewriter, destAlloc);

            return success();
        }

        auto byteOffset = mlir::cast<IntegerAttr>(byteOffsetConstantOp.getValueAttr());

        // check that the memref type is what we expect
        if (origType.getLayout()) {
            if (auto layout = llvm::dyn_cast<AffineMapAttr>(origType.getLayout())) {
                if (!layout.isIdentity()) {
                    return subspanOp.emitError("unsupported non-identity affine layout");
                }
            }
            else if (auto layout = llvm::dyn_cast<StridedLayoutAttr>(origType.getLayout())) {

                auto elementSizeBytes = origType.getElementType().getIntOrFloatBitWidth() / 8;

                // check if the offset is dynamic
                if (layout.getOffset() != ShapedType::kDynamic) {
                    if (layout.getOffset() * elementSizeBytes != byteOffset.getValue()) {
                        return subspanOp.emitError("binding byte offset does not match layout");
                    }
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

        auto mapOp = rewriter.replaceOpWithNewOp<syna::torq_hl::MapBindingOp>(
            subspanOp, outputType, mlir::cast<IntegerAttr>(byteOffset), subspanOp.getBindingAttr(),
            rewriter.getBoolAttr(isReadOnly), rewriter.getBoolAttr(isWriteOnly)
        );

        // update the memref layout offsets of all the subviews since we changed the parent one
        updateSubViewUserTypes(rewriter, mapOp);

        return success();
    }
};

struct ElideNoOp final : public OpRewritePattern<memref::AssumeAlignmentOp> {
    using OpRewritePattern<memref::AssumeAlignmentOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::AssumeAlignmentOp op, PatternRewriter &rewriter) const override {
        rewriter.replaceOp(op, op.getMemref());
        return success();
    }
};

// Rewrites a memref.subview whose first-dim offset comes from a
// dispatch.workload.ordinal (and whose other dims are static contiguous)
// into a static alloc + a dynamic_host_copy that snapshots the runtime
// region. Matches the IR shape default dispatch creation emits.
class WorkloadOrdinalSubviewPattern : public OpRewritePattern<memref::SubViewOp> {
  public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(memref::SubViewOp subviewOp, PatternRewriter &rewriter) const override {
        // Accept: rank-N, offset[0] dynamic, offset[1..] = 0, all sizes static, all strides 1.
        auto offsets = subviewOp.getMixedOffsets();
        auto sizes = subviewOp.getMixedSizes();
        auto strides = subviewOp.getMixedStrides();
        if (offsets.empty() || offsets.size() != sizes.size() || offsets.size() != strides.size()) {
            return failure();
        }
        Value offsetValue = dyn_cast<Value>(offsets[0]);
        if (!offsetValue)
            return failure();
        for (size_t i = 1; i < offsets.size(); ++i) {
            auto a = dyn_cast<Attribute>(offsets[i]);
            if (!a || cast<IntegerAttr>(a).getInt() != 0)
                return failure();
        }
        int64_t numElements = 1;
        for (auto s : sizes) {
            auto a = dyn_cast<Attribute>(s);
            if (!a)
                return failure();
            numElements *= cast<IntegerAttr>(a).getInt();
        }
        for (auto s : strides) {
            auto a = dyn_cast<Attribute>(s);
            if (!a || cast<IntegerAttr>(a).getInt() != 1)
                return failure();
        }

        IntegerAttr constantIdxAttr;
        IntegerAttr bindingIdxAttr;
        if (auto ordinalOp = stripIndexCast(offsetValue)
                                 .getDefiningOp<IREE::TensorExt::DispatchWorkloadOrdinalOp>()) {
            constantIdxAttr = rewriter.getI32IntegerAttr(ordinalOp.getOrdinal().getZExtValue());
        }
        else {
            auto bindingResult = traceBindingLoadIndex(offsetValue);
            if (failed(bindingResult))
                return failure();
            bindingIdxAttr = rewriter.getI32IntegerAttr(*bindingResult);
        }

        auto resultType = subviewOp.getType();
        auto elementType = resultType.getElementType();
        int64_t elementBytes = div_ceil(elementType.getIntOrFloatBitWidth(), 8);

        auto staticDstType = MemRefType::get(
            resultType.getShape(), elementType, MemRefLayoutAttrInterface{},
            resultType.getMemorySpace()
        );
        auto loc = subviewOp.getLoc();
        auto alloc = rewriter.create<memref::AllocOp>(loc, staticDstType, ValueRange{});

        rewriter.create<syna::torq_hl::DynamicHostCopyOp>(
            loc, alloc.getResult(), subviewOp.getSource(),
            rewriter.getDenseI64ArrayAttr({elementBytes}),
            rewriter.getDenseI64ArrayAttr({elementBytes}),
            rewriter.getDenseI64ArrayAttr({numElements}), rewriter.getI64IntegerAttr(elementBytes),
            constantIdxAttr, bindingIdxAttr,
            /*index_unit_bytes=*/rewriter.getI64IntegerAttr(elementBytes),
            syna::torq_hl::DynamicHostCopyDirectionAttr::get(
                rewriter.getContext(), syna::torq_hl::DynamicHostCopyDirection::PreDispatchRead
            )
        );

        // Cast back to the original strided-layout type so downstream users keep typing.
        auto casted = rewriter.create<memref::CastOp>(loc, resultType, alloc.getResult());
        rewriter.replaceOp(subviewOp, casted.getResult());
        return success();
    }
};

class MapBindingsPass : public impl::MapBindingsBase<MapBindingsPass> {
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
    patterns.add<WorkloadOrdinalSubviewPattern>(ctx);
    patterns.add<ElideNoOp>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMapBindingsPass() {
    return std::make_unique<MapBindingsPass>();
}

} // namespace mlir::syna::torq
