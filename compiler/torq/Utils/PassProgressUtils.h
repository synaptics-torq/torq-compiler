// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace mlir::syna {
namespace torq {

/// PassInstrumentation that prints pass names, timings and memory usage.
/// Useful for debugging long compilations.
/// At destruction, prints a sorted summary of timing and top memory peaks.
///
/// NOTE: Memory tracking reads process-wide RSS via /proc/self/status.
/// When MLIR runs passes on multiple threads, memory deltas include
/// allocations from all concurrent threads and cannot be attributed to
/// a single pass. For precise per-pass memory, run with
/// --mlir-disable-threading to force single-threaded execution.
struct ProgressLogger : public mlir::PassInstrumentation {
    using Clock = std::chrono::steady_clock;

    struct Timing {
        std::string name;
        int64_t ms;
        int64_t memDeltaKB = 0;
    };

    std::string prefix;
    std::optional<Clock::time_point> startTime;
    size_t startMemKB = 0;
    std::vector<Timing> timings;
    llvm::raw_ostream *os;

    explicit ProgressLogger(std::string prefix = "", llvm::raw_ostream *os = nullptr);

    void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
    void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;

    void printSummary();

    ~ProgressLogger() override { printSummary(); }
};

} // namespace torq
} // namespace mlir::syna
