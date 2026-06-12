// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"

#define DEBUG_TYPE "torq-annotate-linalg-affinities"

namespace mlir::syna::torq {

using namespace mlir::iree_compiler;

namespace {

static llvm::cl::opt<std::string> clDumpOperationAffinities(
    "torq-dump-operation-affinities-path",
    llvm::cl::desc("Path to dump the affinities of operations"), llvm::cl::init("")
);
static llvm::cl::opt<std::string> clLoadOperationAffinities(
    "torq-load-operation-affinities-path",
    llvm::cl::desc("Path to load the affinities of operations"), llvm::cl::init("")
);

class AnnotateLinalgAffinitiesPass
    : public impl::AnnotateLinalgAffinitiesBase<AnnotateLinalgAffinitiesPass> {
  public:
    AnnotateLinalgAffinitiesPass() = default;
    AnnotateLinalgAffinitiesPass(const AnnotateLinalgAffinitiesPass &pass) {}

    void runOnOperation() override;
};

static LogicalResult dumpOperationAffinities(
    const llvm::MapVector<Operation *, std::optional<IREE::HAL::DevicePromiseAttr>>
        &operationAffinities,
    const std::string &dumpPath
) {
    // Dump the affinities as currently annotated in a JSON file with format
    // [
    // { "index": 0, "op": "linalg.matmul", "affinity": "torq" },
    // { "index": 1, "op": "linalg.add", "affinity": "local" },
    // { "index": 2, "op": "linalg.add", "affinity": "default" },
    // ...
    // ]

    llvm::json::Array affinitiesArray;

    int index = 0;
    for (const auto &[op, affinity] : operationAffinities) {
        llvm::json::Object entry;
        entry["index"] = index++;
        entry["op"] = op->getName().getStringRef().str();

        // Get device name from affinity's device reference
        std::string affinityName = "default";

        if (affinity.has_value()) {
            if (auto deviceRef = affinity->getDevice()) {
                affinityName = deviceRef.getValue();
            }
        }

        entry["affinity"] = affinityName;

        affinitiesArray.push_back(std::move(entry));
    }

    std::error_code ec;
    llvm::raw_fd_ostream file(dumpPath, ec);
    if (ec) {
        llvm::errs() << "Failed to open file for writing: " << dumpPath << "\n";
        return failure();
    }

    file << llvm::formatv("{0:2}", llvm::json::Value(std::move(affinitiesArray)));
    return success();
}

static LogicalResult loadOperationAffinities(
    llvm::MapVector<Operation *, std::optional<IREE::HAL::DevicePromiseAttr>> &operationAffinities,
    llvm::StringMap<std::optional<IREE::HAL::DevicePromiseAttr>> &namedAffinities,
    const std::string &loadPath
) {
    // Load the affinities from a JSON file with format
    // [
    // { "index": 0, "op": "linalg.matmul", "affinity": "torq" },
    // { "index": 1, "op": "linalg.add", "affinity": "local" },
    // { "index": 2, "op": "linalg.add", "affinity": "default" },
    // ...
    // ]
    // and update the provided operationAffinities map. The index field is used to match the
    // affinity to the correct operation in the module.

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bufferOrErr =
        llvm::MemoryBuffer::getFile(loadPath);
    if (!bufferOrErr) {
        llvm::errs() << "Failed to read file: " << loadPath << "\n";
        return failure();
    }

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(bufferOrErr.get()->getBuffer());
    if (!parsed) {
        llvm::errs() << "Failed to parse JSON file: " << loadPath << "\n";
        return failure();
    }

    llvm::json::Array *array = parsed->getAsArray();
    if (!array) {
        llvm::errs() << "JSON file does not contain an array\n";
        return failure();
    }

    for (const auto &entry : *array) {
        const llvm::json::Object *obj = entry.getAsObject();
        if (!obj)
            continue;

        auto indexOpt = obj->getInteger("index");
        auto affinityNameOpt = obj->getString("affinity");

        if (indexOpt && affinityNameOpt) {
            size_t index = *indexOpt;
            if (index < operationAffinities.size()) {
                auto affinityIt = namedAffinities.find(*affinityNameOpt);
                if (affinityIt == namedAffinities.end()) {
                    llvm::errs() << "Unknown affinity name: " << *affinityNameOpt << "\n";
                    return failure();
                }

                auto it = operationAffinities.begin();
                std::advance(it, index);
                operationAffinities[it->first] = affinityIt->second;
            }
        }
    }

    return success();
}

// This pass annotates Linalg operations with device affinities based on command-line
// options or defaults. The pass also annotates the module with a default device (torq)
// and a stream topology between torq and local devices.
//
// The affinities are represented by introducing flow.tensor_transfer operations
// for the inputs of Linalg operations, with the device affinity attached to the
// tensor_transfer operation. By default the affinities are not set, which means
// the IREE runtime will determine the best affinity for each operation. The user
// can specify affinities for specific operations by providing a JSON file with the
// --torq-load-operation-affinities-path option, which will be loaded and applied by
// this pass. The user can also dump the affinities of operations and the MLIR file to which
// they refer using the --torq-dump-operation-affinities-path option. This can be used
// to create the affinities JSON file.
//

void AnnotateLinalgAffinitiesPass::runOnOperation() {
    auto *ctx = &getContext();

    llvm::StringMap<std::optional<IREE::HAL::DevicePromiseAttr>> namedAffinities;

    // Build a device promises pointing to the local and torq devices.
    // This is resolved to a concrete device affinity by MaterializeTargetDevices.
    for (auto &namedAffinity : {"torq", "local"}) {
        namedAffinities.insert(
            {namedAffinity,
             IREE::HAL::DevicePromiseAttr::get(ctx, StringAttr::get(ctx, namedAffinity), -1ll)}
        );
    }

    // we also add a default affinity which means "let IREE figure out the best affinity"
    namedAffinities.insert({"default", std::nullopt});

    // the default device is torq
    getOperation()->setAttr("hal.device.default", StringAttr::get(ctx, "torq"));

    // use an ordered map so that we can reliably dump and load affinities in a deterministic order
    llvm::MapVector<Operation *, std::optional<IREE::HAL::DevicePromiseAttr>> operationAffinities;

    // by default
    for (auto funcOp : getOperation().getBodyRegion().getOps<func::FuncOp>()) {
        for (auto op : funcOp.getBody().getOps<linalg::LinalgOp>()) {
            op->setAttr(
                "torq.affinity.operation.index",
                IntegerAttr::get(IntegerType::get(ctx, 64), operationAffinities.size())
            );
            operationAffinities.insert({op, std::nullopt});
        }
    }

    if (!clLoadOperationAffinities.empty()) {
        if (failed(loadOperationAffinities(
                operationAffinities, namedAffinities, clLoadOperationAffinities
            ))) {
            return signalPassFailure();
        }
    }

    if (!clDumpOperationAffinities.empty()) {
        if (failed(dumpOperationAffinities(operationAffinities, clDumpOperationAffinities))) {
            return signalPassFailure();
        }

        // dump the mlir to clDumpOperationAffinities + ".mlir" for debugging purposes
        std::string mlirDumpPath = clDumpOperationAffinities + ".mlir";
        std::error_code ec;
        llvm::raw_fd_ostream file(mlirDumpPath, ec);
        if (ec) {
            llvm::errs() << "Failed to open file for writing: " << mlirDumpPath << "\n";
            return signalPassFailure();
        }

        getOperation().print(file);
        file.close();

        llvm::dbgs() << "Dumped operation affinities to " << clDumpOperationAffinities
                     << " and MLIR to " << mlirDumpPath << "\n";
    }

    for (auto [op, affinity] : operationAffinities) {

        auto linalgOp = dyn_cast<linalg::LinalgOp>(op);

        if (!linalgOp)
            continue;

        OpBuilder builder(op);

        for (auto operand : linalgOp.getDpsInputOperands()) {

            if (!isa<TensorType>(operand->get().getType())) {
                continue;
            }

            if (!affinity.has_value()) {
                continue;
            }

            auto newInput = IREE::Flow::TensorTransferOp::create(
                builder, linalgOp.getLoc(), operand->get(), affinity.value()
            );
            operand->set(newInput.getResult());
        }
    }

    // Create a transparent stream topology between torq and local devices
    SmallVector<IREE::HAL::DeviceLinkAttr> links;
    auto torqSymbol = SymbolRefAttr::get(ctx, "torq");
    auto localSymbol = SymbolRefAttr::get(ctx, "local");
    links.push_back(IREE::HAL::DeviceLinkAttr::get(
        ctx, torqSymbol, localSymbol, /*unified_memory=*/true, /*transparent_access=*/true, nullptr
    ));
    links.push_back(IREE::HAL::DeviceLinkAttr::get(
        ctx, localSymbol, torqSymbol, /*unified_memory=*/true, /*transparent_access=*/true, nullptr
    ));
    auto topologyAttr = IREE::HAL::DeviceTopologyAttr::get(ctx, links);
    getOperation()->setAttr("stream.topology", topologyAttr);
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createAnnotateLinalgAffinitiesPass() {
    return std::make_unique<AnnotateLinalgAffinitiesPass>();
}

} // namespace mlir::syna::torq
