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

#include "iree/compiler/Dialect/HAL/Target/TargetOptions.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"

#include <algorithm>
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
        registerTorqTypeConversionPasses();

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

        if (typeMnemonic == "stablehlo-torq") {
            buildStableHLOToTorqHLOpsInputConversionPassPipeline(passManager);
            return true;
        }

        if (typeMnemonic == "stablehlo_xla-torq") {
            buildStableHLOXLAToTorqHLOpsInputConversionPassPipeline(passManager);
            return true;
        }

        return false;
    }

    void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
        typeMnemonics.insert("tosa-torq");
        typeMnemonics.insert("torch-torq");
        typeMnemonics.insert("onnx-torq");
        typeMnemonics.insert("linalg-torq");
        typeMnemonics.insert("stablehlo-torq");
        typeMnemonics.insert("stablehlo_xla-torq");
    }

    void populateDetectedCustomInputConversionTypes(ModuleOp &module, StringSet<> &typeMnemonics)
        override {

        // Only run when target backend is Torq.
        // Otherwise, we would override the base input types (e.g. "onnx")
        // and run the torq pipeline even for non-torq backends like llvm-cpu, which can't handle
        // the resulting IR.

        // FIXME (upgrade): this is a quick workaround
        const auto &halTargetOptions = IREE::HAL::TargetOptions::FromFlags::get();
        bool hasTorqTarget = std::find(
                                 halTargetOptions.legacyTargetBackends.begin(),
                                 halTargetOptions.legacyTargetBackends.end(), "torq"
                             ) != halTargetOptions.legacyTargetBackends.end();

        if (!hasTorqTarget)
            return;

        auto *ctx = module.getContext();
        const Dialect *tosaDialect = ctx->getLoadedDialect("tosa");
        const Dialect *torchDialect = ctx->getLoadedDialect("torch");
        const Dialect *torchConversionDialect = ctx->getLoadedDialect("torch_c");
        const Dialect *linalgDialect = ctx->getLoadedDialect("linalg");

        const Dialect *chloDialect = ctx->getLoadedDialect("chlo");
        const Dialect *stablehloDialect = ctx->getLoadedDialect("stablehlo");
        const Dialect *vhloDialect = ctx->getLoadedDialect("vhlo");

        bool hasTosa = false;
        bool hasTorch = false;
        bool hasLinalg = false;
        bool hasStableHLO = false;
        bool hasTuples = false;

        module.walk([&](Operation *op) {
            Dialect *d = op->getDialect();
            if (d == tosaDialect) {
                hasTosa = true;
                return WalkResult::interrupt();
            }
            if (d == torchDialect || d == torchConversionDialect) {
                hasTorch = true;
                return WalkResult::interrupt();
            }
            if (d == linalgDialect) {
                hasLinalg = true;
                return WalkResult::interrupt();
            }
            if (d == chloDialect || d == stablehloDialect || d == vhloDialect) {
                hasStableHLO = true;
                // Check for tuple types (distinguishes stablehlo from stablehlo_xla).
                // Once tuples are found, we can stop immediately.
                if (auto funcOp = dyn_cast<mlir::FunctionOpInterface>(op)) {
                    for (auto t : funcOp.getArgumentTypes()) {
                        if (isa<TupleType>(t)) {
                            hasTuples = true;
                            return WalkResult::interrupt();
                        }
                    }
                    for (auto t : funcOp.getResultTypes()) {
                        if (isa<TupleType>(t)) {
                            hasTuples = true;
                            return WalkResult::interrupt();
                        }
                    }
                }
                for (auto t : op->getOperandTypes()) {
                    if (isa<TupleType>(t)) {
                        hasTuples = true;
                        return WalkResult::interrupt();
                    }
                }
                for (auto t : op->getResultTypes()) {
                    if (isa<TupleType>(t)) {
                        hasTuples = true;
                        return WalkResult::interrupt();
                    }
                }
                // Keep scanning — a later op may have tuples.
            }
            return WalkResult::advance();
        });

        bool hasOnnx = false;
        for (auto funcOp : module.getOps<func::FuncOp>()) {
            if (funcOp->getAttrOfType<mlir::IntegerAttr>("torch.onnx_meta.opset_version")) {
                hasOnnx = true;
                break;
            }
        }

        if (hasTosa) {
            typeMnemonics.insert("tosa-torq");
        }
        else if (hasOnnx) {
            typeMnemonics.insert("onnx-torq");
        }
        else if (hasTorch) {
            typeMnemonics.insert("torch-torq");
        }
        else if (hasLinalg) {
            typeMnemonics.insert("linalg-torq");
        }
        else if (hasStableHLO) {
            if (hasTuples) {
                typeMnemonics.insert("stablehlo_xla-torq");
            }
            else {
                typeMnemonics.insert("stablehlo-torq");
            }
        }
    }

    void resolveDetectedCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
        // Remove corresponding base types for Torq specific input types.
        // This avoids "mixture of input types" errors.
        static const std::pair<StringRef, StringRef> overrides[] = {
            {"tosa-torq", "tosa"},
            {"torch-torq", "torch"},
            {"onnx-torq", "onnx"},
            {"stablehlo-torq", "stablehlo"},
            {"stablehlo_xla-torq", "stablehlo_xla"},
        };
        for (auto &[torqType, baseType] : overrides) {
            if (typeMnemonics.contains(torqType)) {
                typeMnemonics.erase(baseType);
            }
        }
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
