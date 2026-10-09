// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BufferizationInterfaces.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/MemoryUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "torq-bufferization"

using mlir::bufferization::AnalysisState;
using mlir::bufferization::BufferizableOpInterface;
using mlir::bufferization::BufferizationOptions;
using mlir::bufferization::BufferizationState;
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::syna::torq_hl {

/// Build a torq_hl.reinterpret_cast that views `source` as a memref of
/// `resultElementType`: same shape, layout and memory space as the source.
/// Static entries of the source's layout are spelled as static op entries;
/// dynamic ones are backed by a `memref.extract_strided_metadata`. Returns
/// the new memref value in `result`.
static LogicalResult createReinterpretView(
    OpBuilder &rewriter, Location loc, Value source, Type resultElementType, Value &result
) {
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    if (!sourceType)
        return emitError(loc, "reinterpret view requires a ranked source memref");

    int64_t staticOffset = 0;
    SmallVector<int64_t> staticStrides;
    if (failed(sourceType.getStridesAndOffset(staticStrides, staticOffset)))
        return emitError(loc, "reinterpret view requires an identity or strided source layout");

    MemRefType resultType = MemRefType::get(
        sourceType.getShape(), resultElementType, sourceType.getLayout(),
        sourceType.getMemorySpace()
    );

    bool needMetadata = ShapedType::isDynamic(staticOffset) ||
                        llvm::any_of(sourceType.getShape(), ShapedType::isDynamic) ||
                        llvm::any_of(staticStrides, ShapedType::isDynamic);

    Value dynamicOffset;
    ValueRange dynamicSizes, dynamicStrides;
    if (needMetadata) {
        auto metadata = memref::ExtractStridedMetadataOp::create(rewriter, loc, source);
        dynamicOffset = metadata.getOffset();
        dynamicSizes = metadata.getSizes();
        dynamicStrides = metadata.getStrides();
    }

    OpFoldResult offset = ShapedType::isDynamic(staticOffset)
                              ? OpFoldResult(dynamicOffset)
                              : OpFoldResult(rewriter.getIndexAttr(staticOffset));
    SmallVector<OpFoldResult> sizes, strides;
    for (auto [i, dim] : llvm::enumerate(sourceType.getShape()))
        sizes.push_back(
            ShapedType::isDynamic(dim) ? OpFoldResult(dynamicSizes[i])
                                       : OpFoldResult(rewriter.getIndexAttr(dim))
        );
    for (auto [i, stride] : llvm::enumerate(staticStrides))
        strides.push_back(
            ShapedType::isDynamic(stride) ? OpFoldResult(dynamicStrides[i])
                                          : OpFoldResult(rewriter.getIndexAttr(stride))
        );

    result = torq_hl::ReinterpretCastOp::create(
                 rewriter, loc, resultType, source, offset, sizes, strides
    )
                 .getResult();
    return success();
}

namespace {

// bufferizes an dps operation by substituting all the operands with buffers
template <typename OpT>
static LogicalResult bufferizeOp(
    Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
    BufferizationState &state, TypeRange resultTypes = {}
) {

    if (!bufferization::hasTensorSemantics(op)) {
        return op->emitError("op has not tensor semantics");
    }

    DestinationStyleOpInterface dstOp = cast<DestinationStyleOpInterface>(op);

    SmallVector<Value> newOperands;
    SmallVector<Value> newValues;

    for (auto &operand : op->getOpOperands()) {

        if (isa<TensorType>(operand.get().getType())) {
            FailureOr<Value> maybeBuffer = getBuffer(rewriter, operand.get(), options, state);

            if (failed(maybeBuffer)) {
                return op->emitError("unable to bufferize input operand");
            }

            if (dstOp.isDpsInit(&operand)) {
                newValues.push_back(*maybeBuffer);
            }

            newOperands.push_back(*maybeBuffer);
        }
        else {

            // this is used for example in the case of the program attribute of the
            // torq_hl::CallProgramOp operator
            newOperands.push_back(operand.get());
        }
    }

    OpT::create(rewriter, op->getLoc(), resultTypes, newOperands, op->getAttrs());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, newValues);

    return success();
}

template <typename OpT>
struct TorqHLBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          TorqHLBufferizableOpInterface<OpT>, OpT> {

    bool
    bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {

        // Only inputs bufferize to a memory read (inits are never read).
        auto dstOp = cast<DestinationStyleOpInterface>(op);

        // Read the original buffer to be updated and copy to the init buffer in XRAM
        if (op->hasTrait<mlir::OpTrait::UpdateInPlaceTrait>() && dstOp.isDpsInit(&opOperand)) {
            return true;
        }

        return dstOp.isDpsInput(&opOperand);
    }

    bool
    bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
        // Only outputs bufferize to a memory write.
        auto dstOp = cast<DestinationStyleOpInterface>(op);
        return dstOp.isDpsInit(&opOperand);
    }

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
        bufferization::BufferizationState &state
    ) const {

        return bufferizeOp<OpT>(op, rewriter, options, state);
    }
};

struct ImportProgramOpBufferizableOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ImportProgramOpBufferizableOpInterface, torq_hl::ImportProgramOp> {

    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const BufferizationOptions &options,
        BufferizationState &state
    ) const {
        auto programOp = cast<torq_hl::ImportProgramOp>(op);

        auto tensorType = cast<RankedTensorType>(programOp.getType());
        auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        auto newProgramOp = torq_hl::ImportProgramOp::create(
            rewriter, op->getLoc(), memrefType, programOp.getName(), programOp.getArgAccessesAttr()
        );

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(
            op, newProgramOp.getType(), newProgramOp.getResult()
        );

        return success();
    }
};

struct CallProgramOpBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CallProgramOpBufferizableOpInterface, torq_hl::CallProgramOp> {

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const BufferizationOptions &options,
        BufferizationState &state
    ) const {
        return bufferizeOp<torq_hl::CallProgramOp>(op, rewriter, options, state);
    }
};

struct ConvertOpBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          ConvertOpBufferizableOpInterface, torq_hl::ConvertOp> {

    bool
    bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
        // only the input is read, the init is only written
        return opOperand.getOperandNumber() ==
               cast<torq_hl::ConvertOp>(op).getInputMutable().getOperandNumber();
    }

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const BufferizationOptions &options,
        BufferizationState &state
    ) const {

        OpBuilder::InsertionGuard g(rewriter);

        torq_hl::ConvertOp convertOp = cast<torq_hl::ConvertOp>(op);

        FailureOr<Value> maybeInputBuffer =
            getBuffer(rewriter, convertOp.getInput(), options, state);

        if (failed(maybeInputBuffer)) {
            return op->emitError("unable to bufferize result operand");
        }

        FailureOr<Value> maybeOutputBuffer =
            getBuffer(rewriter, convertOp.getInit(), options, state);

        if (failed(maybeOutputBuffer)) {
            return op->emitError("unable to bufferize result operand");
        }

        auto ret =
            options.createMemCpy(rewriter, op->getLoc(), *maybeInputBuffer, *maybeOutputBuffer);

        if (failed(ret)) {
            return op->emitError("unable to create memcpy from ")
                   << maybeInputBuffer->getType() << " to " << maybeOutputBuffer->getType();
        }

        bufferization::replaceOpWithBufferizedValues(rewriter, op, {*maybeOutputBuffer});

        return success();
    }
};

/// Bufferization of torq_hl.bitcast. It is a pure view of the source buffer
/// with a different element type of the same width, so it bufferizes to a
/// free torq_hl.reinterpret_cast (or a no-op when the element type does not
/// change) and never causes a copy.
struct BitcastOpBufferizableOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          BitcastOpBufferizableOpInterface, torq_hl::BitcastOp> {

    // A bitcast never reads or writes memory contents: it only re-describes
    // an existing buffer (like memref.reinterpret_cast / tensor.reshape).
    bool
    bufferizesToMemoryRead(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
        return false;
    }

    bool
    bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
        return false;
    }

    // The result is a 1:1 alias of the source buffer.
    bufferization::AliasingValueList
    getAliasingValues(Operation *op, OpOperand &opOperand, const AnalysisState &state) const {
        return {{op->getResult(0), bufferization::BufferRelation::Equivalent}};
    }

    FailureOr<bufferization::BufferLikeType> getBufferType(
        Operation *op, Value value, const BufferizationOptions &options,
        const BufferizationState &state, SmallVector<Value> &invocationStack
    ) const {
        auto bitcastOp = cast<torq_hl::BitcastOp>(op);
        auto maybeSrcBufferType = bufferization::detail::asMemRefType(
            bufferization::getBufferType(bitcastOp.getSource(), options, state, invocationStack)
        );
        if (failed(maybeSrcBufferType))
            return failure();
        auto sourceType = cast<MemRefType>(*maybeSrcBufferType);
        auto resultTensorType = cast<ShapedType>(bitcastOp.getResult().getType());
        return cast<bufferization::BufferLikeType>(MemRefType::get(
            resultTensorType.getShape(), resultTensorType.getElementType(), sourceType.getLayout(),
            sourceType.getMemorySpace()
        ));
    }

    LogicalResult bufferize(
        Operation *op, RewriterBase &rewriter, const BufferizationOptions &options,
        BufferizationState &state
    ) const {
        auto bitcastOp = cast<torq_hl::BitcastOp>(op);

        FailureOr<Value> maybeSourceBuffer =
            bufferization::getBuffer(rewriter, bitcastOp.getSource(), options, state);
        if (failed(maybeSourceBuffer))
            return failure();
        Value sourceBuffer = *maybeSourceBuffer;

        auto sourceType = dyn_cast<MemRefType>(sourceBuffer.getType());
        auto resultTensorType = dyn_cast<ShapedType>(bitcastOp.getResult().getType());
        if (!sourceType || !resultTensorType)
            return op->emitError("bitcast bufferization requires ranked source and result");

        if (sourceType.getElementType() == resultTensorType.getElementType()) {
            // Same element type: the bitcast is a no-op.
            replaceOpWithBufferizedValues(rewriter, op, sourceBuffer);
            return success();
        }

        Value resultBuffer;
        if (failed(createReinterpretView(
                rewriter, op->getLoc(), sourceBuffer, resultTensorType.getElementType(),
                resultBuffer
            )))
            return failure();
        replaceOpWithBufferizedValues(rewriter, op, resultBuffer);
        return success();
    }
};

} // namespace

void registerBufferizationInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *context, TorqHLDialect *dialect) {
        // this matches what is promised in TorqHLDialect::initialize
        syna::torq_hl::AddOp::attachInterface<TorqHLBufferizableOpInterface<syna::torq_hl::AddOp>>(
            *context
        );
        syna::torq_hl::Conv2DOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::Conv2DOp>>(*context);
        syna::torq_hl::DepthwiseConv2DOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::DepthwiseConv2DOp>>(*context);
        syna::torq_hl::FullyConnectedOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::FullyConnectedOp>>(*context);
        syna::torq_hl::AvgPool2DOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::AvgPool2DOp>>(*context);
        syna::torq_hl::TransposeOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::TransposeOp>>(*context);
        syna::torq_hl::SegmentationOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::SegmentationOp>>(*context);
        syna::torq_hl::MaxPool2dOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::MaxPool2dOp>>(*context);
        syna::torq_hl::MatMulOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::MatMulOp>>(*context);
        syna::torq_hl::ZeroPointBiasTableOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ZeroPointBiasTableOp>>(*context);
        syna::torq_hl::ZeroPointCorrectOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ZeroPointCorrectOp>>(*context);
        syna::torq_hl::GatherOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::GatherOp>>(*context);
        syna::torq_hl::Im2ColOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::Im2ColOp>>(*context);
        syna::torq_hl::FMAOp::attachInterface<TorqHLBufferizableOpInterface<syna::torq_hl::FMAOp>>(
            *context
        );
        syna::torq_hl::ExpandWeightsOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ExpandWeightsOp>>(*context);
        syna::torq_hl::MixedQuantMatMulOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::MixedQuantMatMulOp>>(*context);
        syna::torq_hl::MulOp::attachInterface<TorqHLBufferizableOpInterface<syna::torq_hl::MulOp>>(
            *context
        );
        syna::torq_hl::TableOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::TableOp>>(*context);
        syna::torq_hl::IdentityOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::IdentityOp>>(*context);
        syna::torq_hl::ArgMaxOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ArgMaxOp>>(*context);
        syna::torq_hl::FillOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::FillOp>>(*context);
        syna::torq_hl::ReduceOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ReduceOp>>(*context);
        syna::torq_hl::DeriveQuantParamsOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::DeriveQuantParamsOp>>(*context);
        syna::torq_hl::ScatterOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ScatterOp>>(*context);
        syna::torq_hl::Conv1DOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::Conv1DOp>>(*context);
        syna::torq_hl::ActOp::attachInterface<TorqHLBufferizableOpInterface<syna::torq_hl::ActOp>>(
            *context
        );
        syna::torq_hl::ElementWiseBinaryOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ElementWiseBinaryOp>>(*context);
        syna::torq_hl::ElementWiseUnaryOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ElementWiseUnaryOp>>(*context);
        syna::torq_hl::ElementWiseShiftOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ElementWiseShiftOp>>(*context);
        syna::torq_hl::ResizeNearestNeighborOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ResizeNearestNeighborOp>>(*context);
        syna::torq_hl::InterleavedInsertOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::InterleavedInsertOp>>(*context);
        syna::torq_hl::DepthToSpaceOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::DepthToSpaceOp>>(*context);
        syna::torq_hl::ReduceMeanOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ReduceMeanOp>>(*context);

        ImportProgramOp::attachInterface<ImportProgramOpBufferizableOpInterface>(*context);
        CallProgramOp::attachInterface<CallProgramOpBufferizableOpInterface>(*context);

        ConvertOp::attachInterface<ConvertOpBufferizableOpInterface>(*context);

        syna::torq_hl::BroadcastOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::BroadcastOp>>(*context);

        syna::torq_hl::SelectOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::SelectOp>>(*context);

        BitcastOp::attachInterface<BitcastOpBufferizableOpInterface>(*context);
    });
}

} // namespace mlir::syna::torq_hl
