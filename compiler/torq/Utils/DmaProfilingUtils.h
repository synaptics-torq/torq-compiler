// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::syna::torq {

// Attribute names carrying async DMA cost estimates. Set on torq_hl ops by the
// annotation pass and, independently, on the lowered torq_hw start ops by
// ProfilingPass. The two are not connected: TorqHL->TorqHW lowering rebuilds DMA
// transfers as cfg/start/wait without copying attributes, so the torq_hl tags exist
// only for inspecting HL-level IR and ProfilingPass recomputes from the NDLs.
inline constexpr llvm::StringLiteral kAsyncDurationAttr = "torq-async-duration-cycles";
inline constexpr llvm::StringLiteral kAsyncBytesAttr = "torq-async-bytes";

/// Compute MTU (Maximum Transfer Unit) efficiency for AXI burst transfers.
///
/// Each AXI burst transfers (1 << mtu) beats of 16 bytes each. Every burst
/// incurs a fixed setup overhead on the AXI bus:
///
///   Cycle 1:   ARVALID  – master sends the address
///   Cycle 2:   ARREADY  – interconnect / DDR controller accepts the request
///   Cycle 3-6: DDR latency – row open / column access / data fetch
///   Cycle 7:   first 16 B data beat arrives
///   Cycle 8+:  continuous 16 B per cycle
///
/// As MTU increases, the setup overhead is amortised over more data beats:
///
///   MTU 2  (4 beats)  => [setup] 16B 16B 16B 16B              [setup] ...
///   MTU 3  (8 beats)  => [setup] 16B 16B 16B 16B 16B 16B 16B 16B  [setup] ...
///   MTU 4  (16 beats) => [setup] 16B x16                          [setup] ...
///
/// MTU efficiency captures this overhead:
///
///   mtu_efficiency = 2^mtu / (2^mtu + delta)
///
/// where delta models the per-burst setup penalty in beat-equivalent cycles
/// (address phase + DDR latency). delta is typically in the range [1, 3];
/// we default to 1.5.
double mtuEfficiency(unsigned mtu, double delta = 1.5);

// TODO: Currently we model DMA as unidirectional (in or out, not both
// simultaneously). Once bidirectional DMA support is available, the
// estimation formula must account for shared bandwidth between
// concurrent DMA in and DMA out transfers.
// Effective DMA throughput (bytes/cycle) for a given MTU.
double dmaThroughputBytesPerCycle(unsigned mtu);

// Total transfer size in bytes for a contiguous shape.
uint64_t transferBytes(llvm::ArrayRef<int64_t> shape, int64_t elementSizeBytes);

// Estimated DMA-in/out cycles = ceil(bytes / throughput).
uint64_t estimateDmaCycles(uint64_t bytes, double throughputBytesPerCycle);

// Returns the total DMA transfer size in BYTES.
//
// DMA NDL dims are constructed in bytes, not elements: the innermost contiguous
// dim is a byte count (contiguousElementsSizeBytes) and any outer dims are plain
// repeat counts (see createNdl and the StoreOp/LoadOp lowering in
// TorqHLToTorqHW/Patterns.cpp). The product of the dim counts is therefore
// already the total transfer size in bytes.
uint64_t dmaNdlTransferBytes(mlir::syna::torq_hw::DmaNdlAttr ndl);

// CDMA (LRAM<->DTCM/ITCM) transfer size in bytes for the source value.
uint64_t cdmaTransferBytes(mlir::Value src);

// Estimated CDMA cycles using the config-dependent throughput factor.
uint64_t estimateCdmaCycles(uint64_t bytes);

// Copy the async cost tags preserved on srcOp onto dstOp, or compute them as a
// DMA-in/out fallback when srcOp carries no tag (e.g. DMA ops created after the
// annotation pass ran, such as NSS program-loader loads created during outlining).
void setDmaAsyncAttrs(
    mlir::Operation *srcOp, mlir::Operation *dstOp, uint64_t bytes, double throughputBytesPerCycle
);

// Same as setDmaAsyncAttrs but for CDMA (LRAM<->DTCM/ITCM) transfers.
void setCdmaAsyncAttrs(mlir::Operation *srcOp, mlir::Operation *dstOp, mlir::Value src);

} // namespace mlir::syna::torq
