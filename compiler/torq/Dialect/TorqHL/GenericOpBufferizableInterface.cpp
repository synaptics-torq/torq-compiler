// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BufferizationInterfaces.h"

#include "GenericOp.h"
#include "TorqHLDialect.h"
#include "TorqHLOps.h"

#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::syna::torq_hl {

namespace {

MemRefType getMemRefType(RankedTensorType tensorType) {

    MemorySpaceAttr memorySpace;

    memorySpace =
        torq_hl::MemorySpaceAttr::get(tensorType.getContext(), torq_hl::MemorySpace::Lram);

    return mlir::MemRefType::get(
        tensorType.getShape(), tensorType.getElementType(), nullptr, memorySpace
    );
}

bool isMemRefTypeCompatible(Type genType1, MemRefType type2) {

    auto type1 = mlir::dyn_cast<MemRefType>(genType1);

    if (!type1) {
        return false;
    }

    if (type1.getElementType() != type2.getElementType()) {
        return false;
    }

    if (type1.getShape() != type2.getShape()) {
        return false;
    }

    if (type1.getMemorySpace() != type2.getMemorySpace()) {

        // hal memory space is compatible with no memory space at all
        bool type1Hal = mlir::isa_and_nonnull<iree_compiler::IREE::HAL::DescriptorTypeAttr>(
            type1.getMemorySpace()
        );
        bool type2Hal = mlir::isa_and_nonnull<iree_compiler::IREE::HAL::DescriptorTypeAttr>(
            type2.getMemorySpace()
        );

        if (!(type1Hal && type2.getMemorySpace() == nullptr) &&
            !(type1.getMemorySpace() == nullptr && type2Hal)) {
            return false;
        }
    }

    // FIXME: we should check the layouts to ensure they are compatible
    // for instance we could have a strided layout with natural strides
    // and a identity layout
    if (type1.getLayout() != type2.getLayout()) {
        return false;
    }

    return true;
}

struct TorqHLGenericOpBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          TorqHLGenericOpBufferizableOpInterface, GenericOp> {

    bool
    bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {

        // Only inputs bufferize to a memory read (inits are never read).
        auto dstOp = cast<DestinationStyleOpInterface>(op);
        return dstOp.isDpsInput(&opOperand);
    }

    Value bufferizeInput(
        Value value, Location loc, RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options
    ) const {
        FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);

        if (failed(maybeBuffer)) {
            return nullptr;
        }

        auto buffer = *maybeBuffer;

        auto tensorType = mlir::cast<RankedTensorType>(value.getType());

        MemRefType inputMemRefType = getMemRefType(tensorType);

        if (isMemRefTypeCompatible(buffer.getType(), inputMemRefType)) {
            // the input buffer has already the right type, we can use it
            // directly
            return buffer;
        }
        else {

            // the input buffer has the wrong right type, we must copy it
            SmallVector<Value> dynamicDims;

            auto maybeNewAlloc = options.createAlloc(rewriter, loc, inputMemRefType, dynamicDims);

            if (failed(maybeNewAlloc)) {
                return nullptr;
            }

            auto ret = options.createMemCpy(rewriter, loc, buffer, *maybeNewAlloc);

            if (failed(ret)) {
                return nullptr;
            }

            return *maybeNewAlloc;
        }
    }

    Value bufferizeInit(
        Value value, Location loc, RewriterBase &rewriter,
        const bufferization::BufferizationOptions &options
    ) const {

        auto tensorType = mlir::cast<RankedTensorType>(value.getType());

        FailureOr<Value> maybeBuffer = getBuffer(rewriter, value, options);

        if (failed(maybeBuffer)) {
            return nullptr;
        }

        auto buffer = *maybeBuffer;

        MemRefType outputMemRefType = getMemRefType(tensorType);

        if (isMemRefTypeCompatible(buffer.getType(), outputMemRefType)) {
            // if the buffer is already in the right type, we can use it
            // directly
            return buffer;
        }
        else {

            // otherwise we need to allocate a new buffer
            SmallVector<Value> dynamicDims;
            auto maybeNewAlloc = options.createAlloc(rewriter, loc, outputMemRefType, dynamicDims);
            if (failed(maybeNewAlloc)) {
                return nullptr;
            }

            return *maybeNewAlloc;
        }

        return nullptr;
    }

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options
    ) const {

        OpBuilder::InsertionGuard g(rewriter);

        if (!bufferization::hasTensorSemantics(op)) {
            return op->emitError("op has not tensor semantics");
        }

        auto genericOp = cast<GenericOp>(op);

        SmallVector<Value> newOutputBuffers;

        rewriter.setInsertionPoint(op);

        GenericOpConfig config = GenericOpConfig::fromOperation(genericOp);

        // FIXME: for the moment we bufferize P in XRAM, but it would be better represented as a
        // vector
        FailureOr<Value> maybePBuffer = getBuffer(rewriter, genericOp.getP(), options);

        if (failed(maybePBuffer)) {
            return op->emitError("unable to bufferize P operand");
        }

        config.p = GenericOpParam(*maybePBuffer, genericOp.getPMap());
        newOutputBuffers.push_back(*maybePBuffer);

        if (genericOp.getD()) {
            auto dBuffer = bufferizeInput(genericOp.getD(), genericOp.getLoc(), rewriter, options);

            if (!dBuffer) {
                return op->emitError("unable to bufferize D operand");
            }

            config.d = GenericOpParam(dBuffer, genericOp.getDMap().value());
        }

        if (genericOp.getW()) {
            auto wBuffer = bufferizeInput(genericOp.getW(), genericOp.getLoc(), rewriter, options);

            if (!wBuffer) {
                return op->emitError("unable to bufferize W operand");
            }

            config.w = GenericOpParam(wBuffer, genericOp.getWMap().value());
        }

        if (genericOp.getScale()) {
            auto scaleBuffer =
                bufferizeInput(genericOp.getScale(), genericOp.getLoc(), rewriter, options);

            if (!scaleBuffer) {
                return op->emitError("unable to bufferize W operand");
            }

            config.scale = GenericOpParam(scaleBuffer, genericOp.getScaleMap().value());
        }

        if (genericOp.getBias()) {

            if (genericOp.getScale()) {
                if (genericOp.getScale() != genericOp.getBias()) {
                    return op->emitError("bias and scale must be the same tensor");
                }

                config.bias = GenericOpParam(config.scale.value(), genericOp.getBiasMap().value());
            }
            else {
                auto biasBuffer =
                    bufferizeInput(genericOp.getBias(), genericOp.getLoc(), rewriter, options);

                if (!biasBuffer) {
                    return op->emitError("unable to bufferize Bias operand");
                }

                config.bias = GenericOpParam(biasBuffer, genericOp.getBiasMap().value());
            }
        }

        if (genericOp.getQ()) {
            auto qBuffer = bufferizeInit(genericOp.getQ(), genericOp.getLoc(), rewriter, options);

            if (!qBuffer) {
                return op->emitError("unable to bufferize Q operand");
            }

            config.q = GenericOpParam(qBuffer, genericOp.getQMap().value());

            newOutputBuffers.push_back(qBuffer);
        }

        rewriter.create<GenericOp>(op->getLoc(), config);

        bufferization::replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

        return success();
    }
};

} // namespace

void registerGenericOpBufferizableOpInterfaceExternalModel(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *context, syna::torq_hl::TorqHLDialect *dialect) {
        // this matches what is promised in TorqHLDialect::initialize
        syna::torq_hl::GenericOp::attachInterface<TorqHLGenericOpBufferizableOpInterface>(*context);
    });
}

} // namespace mlir::syna::torq_hl
