// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Codegen/Passes.h"
#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TorqHLToTorqHW/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "torq/Dialect/TorqHL/BufferizationInterfaces.h"
#include "torq/Dialect/TorqHL/GenericOpBufferizableInterface.h"
#include "torq/Dialect/TorqHL/TilingInterfaces.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Pipelines/InputConversionPassPipelines.h"
#include "torq/Target/TorqTarget.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"

#include <stdio.h>

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {
namespace {

struct TORQSession : public PluginSession<
                         TORQSession, TorqTargetOptions, PluginActivationPolicy::DefaultActivated> {

    static void registerPasses() {
        registerCodegenTORQPasses();
        torq_hl::registerTorqHLPasses();
        registerOptimizeLinalgForTorqPasses();

        registerLinalgToTorqHLPasses();
        registerTosaToTorqHLPasses();
        registerTorqHLToTorqHWPasses();

        registerTosaTransformPassPipeline();
        registerTorchTransformPassPipeline();
        // torq_hw::registerTorqHWTransformPassPipeline();
    }

    void onRegisterDialects(DialectRegistry &registry) override {
        registry.insert<mlir::syna::torq_hl::TorqHLDialect>();
        registry.insert<mlir::syna::torq_hw::TorqHWDialect>();
        torq_hl::registerBufferizationInterfaceExternalModels(registry);
#ifdef ENABLE_TORQ_GENERIC
        torq_hl::registerGenericOpBufferizableOpInterfaceExternalModel(registry);
        torq_hl::registerTilingInterfaceExternalModels(registry);
#endif // ENABLE_TORQ_GENERIC

        // These are used for JIT compilation of constant manipulation functions
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);

        torq::tensor::registerTilingInterfaceExternalModels(registry);
    }

    bool extendCustomInputConversionPassPipeline(
        OpPassManager &passManager, std::string_view typeMnemonic
    ) override {

        if (typeMnemonic == "tosa-torq") {
            buildTosaToTorqHLOpsInputConversionPassPipeline(passManager);
            return true;
        }

        if (typeMnemonic == "torch-torq" || typeMnemonic == "onnx-torq") {
            if (typeMnemonic == "onnx-torq") {
                passManager.addNestedPass<func::FuncOp>(
                    mlir::torch::onnx_c::createTorchOnnxToTorchPass()
                );
            }
            buildTorchToTorqHLOpsInputConversionPassPipeline(passManager);
            return true;
        }

        if (typeMnemonic == "linalg-torq") {
            buildLinalgToTorqHLOpsInputConversionPassPipeline(passManager);
            return true;
        }

        return false;
    }

    void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
        typeMnemonics.insert("tosa-torq");
        typeMnemonics.insert("torch-torq");
        typeMnemonics.insert("onnx-torq");
        typeMnemonics.insert("linalg-torq");
    }

    void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) override {
        targets.add("torq", [&]() { return createTarget(options); });
    }

    void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) override {
        targets.add("torq", [=]() { return createBackend(options); });
    }
};

} // namespace
} // namespace mlir::syna::torq

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::syna::torq::TorqTargetOptions);

extern "C" bool iree_register_compiler_plugin_torq(mlir::iree_compiler::PluginRegistrar *registrar
) {
    registrar->registerPlugin<::mlir::syna::torq::TORQSession>("torq");
    return true;
}
