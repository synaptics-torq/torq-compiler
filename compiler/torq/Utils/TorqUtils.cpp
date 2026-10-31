// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqUtils.h"
#include "ExecutorAssignment.h"
#include "llvm/Support/Debug.h"

llvm::cl::list<std::string> clExecuteOnHost(
    "torq-execute-on-host", llvm::cl::desc("Force operation execution on host"),
    llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated, llvm::cl::callback([](const std::string &Str) {
        llvm::dbgs() << "*** Forced Host Execution: " << Str << "\n";
    })
);

llvm::cl::list<std::string> clExecuteOnCSS(
    "torq-execute-on-css", llvm::cl::desc("Force operation execution on CSS"), llvm::cl::ZeroOrMore,
    llvm::cl::CommaSeparated, llvm::cl::callback([](const std::string &Str) {
        if (llvm::is_contained(clExecuteOnHost, Str)) {
            llvm::errs() << "*** Conflict Host/CSS Execution: " << Str << "\n";
            exit(1);
        }
        llvm::dbgs() << "*** Forced CSS Execution: " << Str << "\n";
    })
);

namespace mlir::syna {

namespace torq {

std::pair<bool, LogicalResult>
setTargetExecutorIfForced(Operation *op, PatternRewriter &rewriter, std::string opName) {
    if (llvm::is_contained(clExecuteOnHost, opName)) {
        if (getTargetExecutor(op) != torq_hl::Executor::Host) {
            setTargetExecutorAttr(op, torq_hl::Executor::Host);
            return {true, success()};
        }
        return {true, rewriter.notifyMatchFailure(op, "target executor is Host")};
    }
    if (llvm::is_contained(clExecuteOnCSS, opName)) {
        if (getTargetExecutor(op) != torq_hl::Executor::CSS) {
            setTargetExecutorAttr(op, torq_hl::Executor::CSS);
            return {true, success()};
        }
        return {true, rewriter.notifyMatchFailure(op, "target executor is CSS")};
    }
    return {false, failure()};
}

} // namespace torq
} // namespace mlir::syna
