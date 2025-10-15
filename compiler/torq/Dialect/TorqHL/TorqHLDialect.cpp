// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#ifdef ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/GenericOp.h"
#endif // ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/KernelInterface.cpp.inc"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.cpp.inc"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Transforms/InliningUtils.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

namespace mlir::syna::torq_hl {

namespace {
/// This class defines the interface for handling inlining for TorqHL
/// dialect operations.
struct TorqHLInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// All TorqHL dialect ops can be inlined.
    bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
};

} // namespace

void TorqHLDialect::initialize() {
    initializeTorqHLAttrs();

    addInterfaces<TorqHLInlinerInterface>();

    addOperations<
#define GET_OP_LIST
#include "torq/Dialect/TorqHL/TorqHLOps.cpp.inc"
        >();

#ifdef ENABLE_TORQ_GENERIC
    declarePromisedInterface<TilingInterface, torq_hl::GenericOp>();
    declarePromisedInterface<bufferization::BufferizableOpInterface, torq_hl::GenericOp>();

    declarePromisedInterface<PartialReductionOpInterface, torq_hl::GenericOp>();
#endif // ENABLE_TORQ_GENERIC

    declarePromisedInterfaces<
        bufferization::BufferizableOpInterface, syna::torq_hl::AddOp, syna::torq_hl::Conv2DOp,
        syna::torq_hl::DepthwiseConv2DOp, syna::torq_hl::FullyConnectedOp,
        syna::torq_hl::AvgPool2DOp, syna::torq_hl::TransposeOp, syna::torq_hl::SegmentationOp,
        syna::torq_hl::MatMulOp, syna::torq_hl::FMAOp, syna::torq_hl::ReduceOp,
        syna::torq_hl::MulOp, syna::torq_hl::IdentityOp, syna::torq_hl::ArgMaxOp,
        syna::torq_hl::TableOp, syna::torq_hl::ActOp, syna::torq_hl::BroadcastOp,
        syna::torq_hl::ReduceMeanOp>();
}

} // namespace mlir::syna::torq_hl
