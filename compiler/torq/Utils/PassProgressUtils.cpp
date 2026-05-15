// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassProgressUtils.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <fstream>

namespace mlir::syna {
namespace torq {

/// Read current resident set size (RSS) in kilobytes from /proc/self/status.
/// Returns 0 if unable to read.
static size_t getCurrentRSSKB() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open())
        return 0;
    std::string line;
    while (std::getline(status, line)) {
        if (line.find("VmRSS:") == 0) {
            // Format: "VmRSS:    123456 kB"
            size_t start = line.find_first_of("0123456789");
            size_t end = line.find_last_of("0123456789");
            if (start != std::string::npos && end != std::string::npos && end >= start) {
                return std::stoull(line.substr(start, end - start + 1));
            }
        }
    }
    return 0;
}

static std::string formatMemoryDelta(int64_t deltaKB) {
    if (deltaKB == 0)
        return "0 MB";
    if (deltaKB > 1024 * 1024) {
        std::string s;
        llvm::raw_string_ostream(s) << llvm::format("+%.2f GB", deltaKB / (1024.0 * 1024.0));
        return s;
    }
    if (deltaKB > 1024) {
        std::string s;
        llvm::raw_string_ostream(s) << llvm::format("+%.1f MB", deltaKB / 1024.0);
        return s;
    }
    std::string s;
    llvm::raw_string_ostream(s) << llvm::format("+%lld kB", deltaKB);
    return s;
}

ProgressLogger::ProgressLogger(std::string p, llvm::raw_ostream *out)
    : prefix(std::move(p)), os(out ? out : &llvm::errs()) {}

void ProgressLogger::runBeforePass(mlir::Pass *pass, mlir::Operation *op) {
    startTime = Clock::now();
    startMemKB = getCurrentRSSKB();
    if (prefix.empty()) {
        *os << "[TORQ] + " << pass->getName() << "\n";
    }
    else {
        *os << "[TORQ " << prefix << "] + " << pass->getName() << "\n";
    }
}

void ProgressLogger::runAfterPass(mlir::Pass *pass, mlir::Operation *op) {
    assert(startTime);
    auto ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - *startTime).count();
    int64_t memDeltaKB = static_cast<int64_t>(getCurrentRSSKB()) - static_cast<int64_t>(startMemKB);
    std::string memStr = formatMemoryDelta(memDeltaKB);
    if (prefix.empty()) {
        *os << "[TORQ] - " << pass->getName() << " (" << ms << " ms, " << memStr << ")\n";
    }
    else {
        *os << "[TORQ " << prefix << "] - " << pass->getName() << " (" << ms << " ms, " << memStr
            << ")\n";
    }
    timings.push_back({std::string(pass->getName()), ms, memDeltaKB});
}

void ProgressLogger::printSummary() {
    if (timings.empty())
        return;

    int64_t totalMs = 0;
    int64_t totalMemKB = 0;
    size_t zeroTimeCount = 0;
    std::vector<Timing> nonZeroTime;
    for (const auto &t : timings) {
        totalMs += t.ms;
        totalMemKB += t.memDeltaKB;
        if (t.ms > 10)
            nonZeroTime.push_back(t);
        else
            ++zeroTimeCount;
    }

    // --- Timing summary (only passes with > 0 ms) ---
    llvm::stable_sort(nonZeroTime, [](const Timing &a, const Timing &b) {
        if (a.ms != b.ms)
            return a.ms > b.ms;
        return a.memDeltaKB > b.memDeltaKB;
    });

    if (!prefix.empty()) {
        *os << "\n[TORQ " << prefix << "] === Pass timing summary ===\n";
    }
    else {
        *os << "\n[TORQ] === Pass timing summary ===\n";
    }

    for (size_t i = 0; i < nonZeroTime.size(); ++i) {
        double pct = totalMs > 0 ? (100.0 * nonZeroTime[i].ms / totalMs) : 0.0;
        std::string memStr = formatMemoryDelta(nonZeroTime[i].memDeltaKB);
        *os << "  " << (i + 1) << ". " << nonZeroTime[i].name << " : " << nonZeroTime[i].ms
            << " ms (" << llvm::format("%.1f", pct) << "%), " << memStr << "\n";
    }

    if (zeroTimeCount > 0) {
        *os << "  ... " << zeroTimeCount << " other pass" << (zeroTimeCount > 1 ? "es" : "")
            << " : <= 10 ms\n";
    }

    *os << "  Total: " << timings.size() << " passes, " << totalMs << " ms, "
        << formatMemoryDelta(totalMemKB) << "\n";

    // --- Top 10 memory peak summary ---
    std::vector<Timing> byMem = timings;
    llvm::stable_sort(byMem, [](const Timing &a, const Timing &b) {
        return a.memDeltaKB > b.memDeltaKB;
    });

    size_t memSummaryCount = std::min(size_t(10), byMem.size());
    if (memSummaryCount > 0 && byMem[0].memDeltaKB > 0) {
        if (!prefix.empty()) {
            *os << "\n[TORQ " << prefix << "] === Top 10 memory peak summary ===\n";
        }
        else {
            *os << "[TORQ] === Top 10 memory peak summary ===\n";
        }
        for (size_t i = 0; i < memSummaryCount; ++i) {
            if (byMem[i].memDeltaKB <= 0)
                break;
            *os << "  " << (i + 1) << ". " << byMem[i].name << " : "
                << formatMemoryDelta(byMem[i].memDeltaKB) << "\n";
        }
    }

    if (!prefix.empty()) {
        *os << "[TORQ " << prefix << "] ==============================\n\n";
    }
    else {
        *os << "[TORQ] ==============================\n\n";
    }
}

} // namespace torq
} // namespace mlir::syna
