// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include "torq/Utils/TorqUtils.h"

#include <fstream>

#define DEBUG_TYPE "torq-analyze-tensor-sizes"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

static llvm::cl::opt<std::string> clEnableAnalyzeTensorSizes(
    "torq-export-tensor-sizes", llvm::cl::desc("Save tensor sizes analysis to a file"),
    llvm::cl::init("")
);

namespace {

// This pass allows to dump the size of all tensors used by linalg to roughly estimate the LRAM size
// required to perform the full network without ever writing back to intermediate results to XRAM.
//
// To dump an anlysis of the linalg IR use the parameter --torq-export-tensor-sizes and to analyze
// the exported file use the "scripts/analyze-tensor-sizes.py".
//
// The code doesn't try to detect operations that can be done at compile time so some of the
// operations in list may not be really required to run at runtime.
class AnalyzeTensorSizesPass : public impl::AnalyzeTensorSizesBase<AnalyzeTensorSizesPass> {
  public:
    AnalyzeTensorSizesPass() = default;
    AnalyzeTensorSizesPass(const AnalyzeTensorSizesPass &pass) {}

    void runOnOperation() override;
};

static FailureOr<int64_t> getValueSize(Value val) {
    if (auto rankedTensorType = dyn_cast<RankedTensorType>(val.getType())) {

        if (!rankedTensorType.hasStaticShape()) {
            return failure();
        }

        auto type = rankedTensorType.getElementType();

        int64_t elementSize = div_ceil(type.getIntOrFloatBitWidth(), 8);

        return rankedTensorType.getNumElements() * elementSize;
    }
    else if (val.getType().isIntOrFloat()) {

        return div_ceil(val.getType().getIntOrFloatBitWidth(), 8);
    }
    else {

        return failure();
    }
}

void AnalyzeTensorSizesPass::runOnOperation() {

    if (clEnableAnalyzeTensorSizes.empty()) {
        return;
    }

    // open the export file and write the header
    std::error_code ec;
    llvm::raw_fd_ostream exportFile(clEnableAnalyzeTensorSizes, ec, llvm::sys::fs::OF_Text);

    if (ec) {
        llvm::errs() << "Failed to open export file: " << clEnableAnalyzeTensorSizes << " ("
                     << ec.message() << ")\n";
        signalPassFailure();
        return;
    }

    getOperation().walk([&](Operation *op) {
        int resultSizes = 0;
        bool failedEstimationForInputs = false;
        bool failedEstimationForResults = false;

        for (auto result : op->getResults()) {
            auto sizeOrFailure = getValueSize(result);
            if (failed(sizeOrFailure)) {
                failedEstimationForResults = true;
                continue;
            }
            resultSizes += sizeOrFailure.value();
        }

        int inputSizes = 0;
        for (auto operand : op->getOperands()) {
            auto sizeOrFailure = getValueSize(operand);
            if (failed(sizeOrFailure)) {
                failedEstimationForInputs = true;
                continue;
            }
            inputSizes += sizeOrFailure.value();
        }

        if (failedEstimationForInputs || failedEstimationForResults) {
            exportFile << op->getName() << ",unknown, unknown" << "," << op->getLoc() << "\n";
        }
        else {
            exportFile << op->getName() << "," << inputSizes << "," << resultSizes << ","
                       << op->getLoc() << "\n";
        }
    });

    exportFile.flush();
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeTensorSizesPass() {
    return std::make_unique<AnalyzeTensorSizesPass>();
}

} // namespace mlir::syna::torq
