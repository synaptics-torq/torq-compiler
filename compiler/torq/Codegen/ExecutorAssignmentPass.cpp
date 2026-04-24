// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "torq/Codegen/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <optional>
#include <unordered_map>

#define DEBUG_TYPE "torq-executor-assignment"

namespace mlir::syna::torq {

namespace {

class ExecutorAssignmentPass : public impl::ExecutorAssignmentBase<ExecutorAssignmentPass> {
  public:
    ExecutorAssignmentPass() = default;
    explicit ExecutorAssignmentPass(std::string executorMap) {
        executorMapFile = std::move(executorMap);
    }
    ExecutorAssignmentPass(const ExecutorAssignmentPass &pass) {}

    void runOnOperation() override;

  private:
    FailureOr<std::unordered_map<std::string, torq_hl::Executor>> loadExecutorMap();
    std::string generateOpIdentifier(Operation *op);
    std::optional<torq_hl::Executor> parseExecutorString(StringRef str);

    // Cache the loaded executor map across function invocations
    std::optional<FailureOr<std::unordered_map<std::string, torq_hl::Executor>>> cachedMap;
};

FailureOr<std::unordered_map<std::string, torq_hl::Executor>>
ExecutorAssignmentPass::loadExecutorMap() {
    std::unordered_map<std::string, torq_hl::Executor> map;

    if (executorMapFile.empty()) {
        return map; // Empty map means use default behavior
    }

    LLVM_DEBUG(llvm::dbgs() << "Loading executor map from: " << executorMapFile << "\n";);

    auto fileOrErr = llvm::MemoryBuffer::getFile(executorMapFile);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Error reading executor map file '" << executorMapFile
                     << "': " << ec.message() << "\n";
        return failure();
    }

    auto jsonOrErr = llvm::json::parse((*fileOrErr)->getBuffer());
    if (!jsonOrErr) {
        llvm::errs() << "Error parsing executor map JSON from " << executorMapFile << "\n";
        return failure();
    }

    auto *root = jsonOrErr->getAsObject();
    if (!root) {
        llvm::errs() << "Executor map must be a JSON object\n";
        return failure();
    }

    auto *assignments = root->getObject("op_assignments");
    auto *ops = root->getObject("ops");

    if (!assignments && !ops) {
        llvm::errs() << "Executor map missing 'op_assignments' or 'ops' key\n";
        return failure();
    }

    // Parse 'op_assignments' format (compiler input format)
    // Format: {"271:12": {"executor": "nss"}}
    if (assignments) {
        LLVM_DEBUG(llvm::dbgs() << "Parsing op_assignments format (compiler format)\n";);
        for (const auto &[key, value] : *assignments) {
            auto executorObj = value.getAsObject();
            if (!executorObj) {
                continue;
            }

            auto executorStr = executorObj->getString("executor");
            if (!executorStr) {
                continue;
            }

            if (auto executor = parseExecutorString(*executorStr)) {
                map[key.str()] = *executor;
            }
        }
    }

    // Parse 'ops' format (discovery output format)
    // Format: {"Conv_...": {"recommended_executor": "nss", "mlir_location": "271:12"}}
    // If recommended_executor is null or missing, skip assignment for that operation
    if (ops) {
        LLVM_DEBUG(llvm::dbgs() << "Parsing ops format (discovery format)\n";);
        for (const auto &[opName, value] : *ops) {
            auto opObj = value.getAsObject();
            if (!opObj) {
                continue;
            }

            // Get the mlir_location for the key
            auto locationStr = opObj->getString("mlir_location");
            if (!locationStr) {
                continue;
            }

            // Get the recommended_executor - check if it exists and is not null
            auto executorValue = opObj->get("recommended_executor");
            if (!executorValue) {
                // No recommended_executor field - skip this operation
                continue;
            }

            // Check if it's null
            if (executorValue->getAsNull()) {
                LLVM_DEBUG(llvm::dbgs() << "Skipping operation " << opName.str()
                                        << " - recommended_executor is null\n";);
                continue;
            }

            // Try to get as string
            auto executorStr = executorValue->getAsString();
            if (!executorStr || executorStr->empty()) {
                LLVM_DEBUG(llvm::dbgs() << "Skipping operation " << opName.str()
                                        << " - recommended_executor is empty\n";);
                continue;
            }

            if (auto executor = parseExecutorString(*executorStr)) {
                map[locationStr->str()] = *executor;
                LLVM_DEBUG(llvm::dbgs() << "Added assignment: " << locationStr->str() << " -> "
                                        << *executorStr << "\n";);
            }
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "Loaded " << map.size() << " executor assignments\n";);
    return map;
}

std::optional<torq_hl::Executor> ExecutorAssignmentPass::parseExecutorString(StringRef str) {
    if (str == "nss") {
        return torq_hl::Executor::Slice;
    }
    else if (str == "css") {
        return torq_hl::Executor::CSS;
    }
    else if (str == "host") {
        return torq_hl::Executor::Host;
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "Unknown executor: " << str << "\n";);
        return std::nullopt;
    }
}

static std::string extractLocationKey(Location loc) {
    // Extract line:column from location for matching
    // We use line:column only (without filename) since filenames can vary
    // between different compilation contexts (layer vs full model)

    // Try direct FileLineColLoc
    if (auto fileLoc = mlir::dyn_cast<FileLineColLoc>(loc)) {
        return std::to_string(fileLoc.getLine()) + ":" + std::to_string(fileLoc.getColumn());
    }

    // Unwrap CallSiteLoc and check callee
    if (auto callSite = mlir::dyn_cast<CallSiteLoc>(loc)) {
        auto calleeKey = extractLocationKey(callSite.getCallee());
        if (!calleeKey.empty()) {
            return calleeKey;
        }
        // If callee has no file:line, try caller
        return extractLocationKey(callSite.getCaller());
    }

    // Check FusedLoc
    if (auto fusedLoc = mlir::dyn_cast<FusedLoc>(loc)) {
        for (auto innerLoc : fusedLoc.getLocations()) {
            auto key = extractLocationKey(innerLoc);
            if (!key.empty()) {
                return key;
            }
        }
    }

    // Try NameLoc as fallback (for backward compatibility)
    if (auto nameLoc = mlir::dyn_cast<NameLoc>(loc)) {
        return nameLoc.getName().str();
    }

    return "";
}

std::string ExecutorAssignmentPass::generateOpIdentifier(Operation *op) {
    // Generate unique identifier for operation
    // Use file:line:column location as key
    auto locKey = extractLocationKey(op->getLoc());
    if (!locKey.empty()) {
        return locKey;
    }

    // Fallback: use operation type + pointer (should not reach here if MLIR has proper locations)
    std::string fallbackName;
    llvm::raw_string_ostream os(fallbackName);
    os << op->getName().stripDialect();
    os << "_" << reinterpret_cast<uintptr_t>(op);
    return fallbackName;
}

void ExecutorAssignmentPass::runOnOperation() {
    auto funcOp = getOperation();

    // Load and cache the executor map on first use
    if (!cachedMap) {
        cachedMap = loadExecutorMap();
    }
    auto mapOrErr = *cachedMap;
    if (failed(mapOrErr)) {
        return signalPassFailure();
    }

    auto &executorMap = *mapOrErr;
    if (executorMap.empty()) {
        LLVM_DEBUG(
            llvm::dbgs(
            ) << "[ExecutorAssignmentPass] No executor map provided, using default behavior\n";
        );
        return;
    }

    LLVM_DEBUG({
        llvm::dbgs() << "[ExecutorAssignmentPass] Loaded " << executorMap.size()
                     << " executor assignments\n";
        for (const auto &[name, executor] : executorMap) {
            llvm::dbgs() << "[ExecutorAssignmentPass]   Map entry: " << name << " -> "
                         << stringifyExecutor(executor) << "\n";
        }
    });

    int assignedCount = 0;
    int skippedCount = 0;
    int linalgCount = 0;
    int tensorCount = 0;

    // Assign executors based on the map
    funcOp.walk([&](Operation *op) {
        std::string dialect = op->getDialect()->getNamespace().str();

        // Skip operations that don't support executor assignment
        if (dialect != linalg::LinalgDialect::getDialectNamespace() &&
            dialect != tensor::TensorDialect::getDialectNamespace()) {
            return;
        }

        if (dialect == linalg::LinalgDialect::getDialectNamespace()) {
            linalgCount++;
        }
        else {
            tensorCount++;
        }

        // Get operation identifier
        std::string opName = generateOpIdentifier(op);
        LLVM_DEBUG(llvm::dbgs() << "[ExecutorAssignmentPass] Checking op: " << opName
                                << " (dialect: " << dialect << ")\n";);

        auto it = executorMap.find(opName);
        if (it != executorMap.end()) {
            auto executor = it->second;
            setTargetExecutorAttr(op, executor);
            assignedCount++;
            LLVM_DEBUG(llvm::dbgs() << "[ExecutorAssignmentPass] Assigned " << opName << " to "
                                    << stringifyExecutor(executor) << "\n";);
        }
        else {
            skippedCount++;
            LLVM_DEBUG(llvm::dbgs()
                           << "[ExecutorAssignmentPass] Op " << opName << " NOT found in map\n";);
        }
    });

    LLVM_DEBUG(llvm::dbgs() << "[ExecutorAssignmentPass] Summary: linalg=" << linalgCount
                            << " tensor=" << tensorCount << " assigned=" << assignedCount
                            << " skipped=" << skippedCount << "\n";);
    LLVM_DEBUG(llvm::dbgs() << "ExecutorAssignmentPass: assigned=" << assignedCount
                            << " skipped=" << skippedCount << "\n";);
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createExecutorAssignmentPass(StringRef executorMapPath
) {
    return std::make_unique<ExecutorAssignmentPass>(executorMapPath.str());
}

} // namespace mlir::syna::torq
