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
using mlir::bufferization::BufferRelation;
using mlir::bufferization::replaceOpWithBufferizedValues;

namespace mlir::syna::torq_hl {

namespace {

// bufferizes an dps operation by substituting all the operands with buffers
template <typename OpT>
static LogicalResult bufferizeOp(
    Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options,
    TypeRange resultTypes = {}
) {

    if (!bufferization::hasTensorSemantics(op)) {
        return op->emitError("op has not tensor semantics");
    }

    DestinationStyleOpInterface dstOp = cast<DestinationStyleOpInterface>(op);

    SmallVector<Value> newOperands;
    SmallVector<Value> newValues;

    for (auto &operand : op->getOpOperands()) {

        if (isa<TensorType>(operand.get().getType())) {
            FailureOr<Value> maybeBuffer = getBuffer(rewriter, operand.get(), options);

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

    rewriter.create<OpT>(op->getLoc(), resultTypes, newOperands, op->getAttrs());
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
        Operation *op, RewriterBase &rewriter, const bufferization::BufferizationOptions &options
    ) const {

        return bufferizeOp<OpT>(op, rewriter, options);
    }
};

struct ImportProgramOpBufferizableOpInterface
    : public bufferization::BufferizableOpInterface::ExternalModel<
          ImportProgramOpBufferizableOpInterface, torq_hl::ImportProgramOp> {

    bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

    LogicalResult
    bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
        auto programOp = cast<torq_hl::ImportProgramOp>(op);

        auto tensorType = cast<RankedTensorType>(programOp.getType());
        auto memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

        auto newProgramOp = rewriter.create<torq_hl::ImportProgramOp>(
            op->getLoc(), memrefType, programOp.getName()
        );

        rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, newProgramOp.getResult());

        return success();
    }
};

struct CallProgramOpBufferizableOpInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          CallProgramOpBufferizableOpInterface, torq_hl::CallProgramOp> {

    LogicalResult
    bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
        return bufferizeOp<torq_hl::CallProgramOp>(op, rewriter, options);
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

    LogicalResult
    bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {

        OpBuilder::InsertionGuard g(rewriter);

        torq_hl::ConvertOp convertOp = cast<torq_hl::ConvertOp>(op);

        FailureOr<Value> maybeInputBuffer = getBuffer(rewriter, convertOp.getInput(), options);

        if (failed(maybeInputBuffer)) {
            return op->emitError("unable to bufferize result operand");
        }

        FailureOr<Value> maybeOutputBuffer = getBuffer(rewriter, convertOp.getInit(), options);

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
        syna::torq_hl::GatherOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::GatherOp>>(*context);
        syna::torq_hl::FMAOp::attachInterface<TorqHLBufferizableOpInterface<syna::torq_hl::FMAOp>>(
            *context
        );
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
        syna::torq_hl::ScatterOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ScatterOp>>(*context);
        syna::torq_hl::TransposeReshapeOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::TransposeReshapeOp>>(*context);
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
        syna::torq_hl::DepthToSpaceOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::DepthToSpaceOp>>(*context);
        syna::torq_hl::ReduceMeanOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::ReduceMeanOp>>(*context);

        ImportProgramOp::attachInterface<ImportProgramOpBufferizableOpInterface>(*context);
        CallProgramOp::attachInterface<CallProgramOpBufferizableOpInterface>(*context);

        ConvertOp::attachInterface<ConvertOpBufferizableOpInterface>(*context);

        syna::torq_hl::BroadcastOp::attachInterface<
            TorqHLBufferizableOpInterface<syna::torq_hl::BroadcastOp>>(*context);
    });
}

} // namespace mlir::syna::torq_hl
