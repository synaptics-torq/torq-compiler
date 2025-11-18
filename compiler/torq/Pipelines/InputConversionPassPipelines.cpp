// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "InputConversionPassPipelines.h"

#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"

#include "compiler/plugins/input/TOSA/InputConversion/Passes.h"
#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "torch-mlir/Conversion/TorchToLinalg/TorchToLinalg.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-torq-pass-pipeline"

using namespace mlir::syna::torq_hl;

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clDisableSlices;

//===----------------------------------------------------------------------===//
// -torq-torqhl-transformation-pipeline
//===----------------------------------------------------------------------===//

static llvm::cl::opt<bool> clDisableDispatchFusion(
    "torq-disable-dispatch-fusion", llvm::cl::desc("Disable fusion of dispatches into one"),
    llvm::cl::init(false)
);

void buildTosaTransformPassPipeline(OpPassManager &passManager) {

    if (!clDisableSlices) {
        // lower TOSA to a mix of linalg and torq_hl operators
        passManager.addNestedPass<func::FuncOp>(createTosaToTorqHLConversionPass());

        // try to simplify the IR (notably getting rid of useless linalg operations)
        passManager.addPass(mlir::createCanonicalizerPass());
    }

    // use the standard tosa pipeline
    iree_compiler::buildTOSAInputConversionPassPipeline(passManager);

    // divide operations into  dispatch into Flow::RegionOps
    if (!clDisableDispatchFusion) {
        passManager.addNestedPass<func::FuncOp>(
            iree_compiler::Preprocessing::createMakeSingleDispatchForFunctionPass()
        );
    }
}

void registerTosaTransformPassPipeline() {
    PassPipelineRegistration<> transformPassPipeline(
        "torq-tosa-transformation-pipeline", "Convert from tosa to torq_hl operations",
        [](OpPassManager &passManager) { buildTosaTransformPassPipeline(passManager); }
    );
}

void buildTorchTransformPassPipeline(OpPassManager &passManager) {

    passManager.addPass(mlir::torch::Torch::createLowerToBackendContractPass(10, true, true, {}, "")
    );
    mlir::torch::TorchConversion::createTorchBackendToLinalgOnTensorsBackendPipeline(passManager);
    if (!clDisableDispatchFusion) {
        passManager.addNestedPass<func::FuncOp>(
            iree_compiler::Preprocessing::createMakeSingleDispatchForFunctionPass()
        );
    }
}

void buildLinalgTransformPassPipeline(OpPassManager &passManager) {
    if (!clDisableDispatchFusion) {
        passManager.addNestedPass<func::FuncOp>(
            createFormDispatchRegionsPass(clDisableDispatchFusion)
        );
    }
}

void registerTorchTransformPassPipeline() {
    PassPipelineRegistration<> transformPassPipeline(
        "torq-torch-transformation-pipeline", "Convert from torch to torq_hl operations",
        [](OpPassManager &passManager) { buildTorchTransformPassPipeline(passManager); }
    );
}

//===----------------------------------------------------------------------===//
// Other Pass Pipelines
//===----------------------------------------------------------------------===//

void buildTosaToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager) {
    buildTosaTransformPassPipeline(passManager);
}

void buildTorchToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager) {
    buildTorchTransformPassPipeline(passManager);
}

void buildLinalgToTorqHLOpsInputConversionPassPipeline(OpPassManager &passManager) {
    buildLinalgTransformPassPipeline(passManager);
}

} // namespace mlir::syna::torq
