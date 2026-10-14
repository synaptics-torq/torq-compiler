// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/ProfilingUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::syna::torq {

/// One compute step: a start_program and the wait_program that collects it.
struct Program {
    torq_hl::StartProgramOp startOp;
    torq_hl::WaitProgramOp waitOp;
};

/// Returns true when `allocOp` is allocated in LRAM.
bool isLramAlloc(memref::AllocOp allocOp);

/// Returns the byte footprint of a statically shaped LRAM allocation, or zero.
int64_t lramAllocBytes(memref::AllocOp allocOp);

/// Follows derived memref operations to the value that owns the storage.
Value resolveMemRefBase(Value memref);

/// Returns the allocation backing `memref`, or null if it has no alloc base.
memref::AllocOp getBaseAlloc(Value memref);

/// Collects matched start/wait program pairs in the supplied operation order.
SmallVector<Program> collectPrograms(ArrayRef<Operation *> order);

/// Returns true when `op` is a side-effect-free producer that can be hoisted.
bool isPureProducer(Operation *op);

/// What the real LRAM allocator makes of an op order.
struct AllocOutcome {
    /// The allocator produced a compilable function.
    bool ok = false;
    /// Makespan of the allocated stream (inserted swaps included) under the in-order
    /// NSS model, or -1 when the allocation failed.
    int64_t simulatedMakespan = -1;
};

/// Makespan of `funcOp`'s top-level NSS stream: loads and stores block it, a slice
/// program runs from its start to its wait. Transfers without a duration attribute
/// (the allocator's swaps) are priced from their size.
int64_t simulateStream(FunctionOpInterface funcOp);

/// Runs the real allocation tail of the NSS pipeline on a clone of `funcOp`.
AllocOutcome measureRealAllocation(FunctionOpInterface funcOp, int64_t poolSize);

/// Returns true when the global planner places the reordered LRAM buffers without swaps.
bool orderPlacesInLram(FunctionOpInterface funcOp, int64_t poolSize);

/// Reorders the containing block so operations appear in exactly `order`.
void applyOrderToBlock(ArrayRef<Operation *> order);

/// Returns the program launched by `startOp`, or null when it cannot be resolved.
torq_hl::ProgramOp getProgramOp(torq_hl::StartProgramOp startOp);

/// Accumulates profiled compute cycles per program and returns whether profiling was found.
bool computeProgramCycles(
    FunctionOpInterface funcOp, llvm::DenseMap<Operation *, int64_t> &programCycles
);

/// Returns the annotated DMA transfer duration of `loadOp`, or zero when absent.
int64_t dmaCyclesForLoad(torq_hl::LoadOp loadOp);

} // namespace mlir::syna::torq
