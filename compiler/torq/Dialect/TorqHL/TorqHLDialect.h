// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/Dialect.h"

#include "torq/Dialect/TorqHL/TorqHLDialect.h.inc"

#include "mlir/IR/OpDefinition.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"

namespace mlir::syna::torq_hl {

struct KernelTensorEncoding {
    SmallVector<int64_t> stridesAlign{};
    int64_t paddingAlign = 0;
    bool denseOnly = false;
};

struct KernelInputEncoding {
    unsigned opIndex;
    KernelTensorEncoding encoding;
};

struct KernelEncoding {
    SmallVector<KernelInputEncoding> inputEncodings;
    KernelTensorEncoding outputEncoding;
    SmallVector<std::pair<unsigned, unsigned>> equalEncodingOperands{};
};

// One buffer access performed by an op's asynchronous execution window.
// Modeled on IREE Stream's AsyncAccessRange.
struct AsyncAccess {
    ArgAccessBitfield access = ArgAccessBitfield::None;
    TypedValue<MemRefType> buffer;
};

} // namespace mlir::syna::torq_hl

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "torq/Dialect/TorqHL/KernelInterface.h.inc"
