// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TorqTarget.h"

#include "torq/Codegen/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Pipelines/TranslationPassPipelines.h"
#include "torq/Serialization/Serialization.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq_executable_def_builder.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <iostream>

#define DEBUG_TYPE "iree-torq-target"

namespace mlir::syna::torq {

class TorqTargetDevice : public iree_compiler::IREE::HAL::TargetDevice {
  public:
    TorqTargetDevice(const TorqTargetOptions &options) : options_(options) {}

    IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
        MLIRContext *context, const IREE::HAL::TargetRegistry &targetRegistry
    ) const override {
        Builder b(context);
        SmallVector<NamedAttribute> configItems;

        auto configAttr = b.getDictionaryAttr(configItems);

        SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
        targetRegistry.getTargetBackend("torq")->getDefaultExecutableTargets(
            context, "torq", configAttr, executableTargetAttrs
        );

        return IREE::HAL::DeviceTargetAttr::get(
            context, b.getStringAttr("torq"), configAttr, executableTargetAttrs
        );
    }

  private:
    const TorqTargetOptions &options_;
};

class TorqTargetBackend : public IREE::HAL::TargetBackend {
  public:
    TorqTargetBackend(const TorqTargetOptions &options) : options_(options) {}

    std::string getLegacyDefaultDeviceID() const override { return "torq"; }

    void getDefaultExecutableTargets(
        MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
        SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
    ) const override {

        executableTargetAttrs.push_back(getExecutableTarget(context));
    }

    IREE::HAL::ExecutableTargetAttr getExecutableTarget(MLIRContext *context) const {
        Builder b(context);

        SmallVector<NamedAttribute> configItems;

        return IREE::HAL::ExecutableTargetAttr::get(
            context, b.getStringAttr("torq"), b.getStringAttr("torq-fb"),
            b.getDictionaryAttr(configItems)
        );
    }

    void getDependentDialects(DialectRegistry &registry) const override {

        auto loweringPass = createTORQLowerExecutableTargetPass();
        loweringPass->getDependentDialects(registry);

        registry.insert<IREE::Codegen::IREECodegenDialect>();
    }

    void buildConfigurationPassPipeline(
        IREE::HAL::ExecutableTargetAttr targetAttr, OpPassManager &passManager
    ) override {}

    void buildTranslationPassPipeline(
        IREE::HAL::ExecutableTargetAttr targetAttr, OpPassManager &passManager
    ) override {
        buildTORQCodegenPassPipeline(passManager);
    }

    void buildLinkingPassPipeline(OpPassManager &passManager) override {}

    LogicalResult serializeExecutable(
        const SerializationOptions &options, IREE::HAL::ExecutableVariantOp variantOp,
        OpBuilder &executableBuilder
    ) override {

        SmallVector<IREE::HAL::ExecutableExportOp> exportOps =
            llvm::to_vector(variantOp.getOps<IREE::HAL::ExecutableExportOp>());

        if (exportOps.empty()) {
            return variantOp.emitError() << "at least one hal.executable.export op is required";
        }

        if (exportOps.size() > 1) {
            return variantOp.emitError() << "multiple hal.executable.export ops are not supported";
        }

        ModuleOp innerModuleOp = variantOp.getInnerModule();
        if (!innerModuleOp) {
            return innerModuleOp.emitError("expected a non-empty inner module");
        }

        DenseIntElementsAttr binaryAttr;

        if (failed(syna::torq::serializeTorqHW(innerModuleOp, binaryAttr))) {
            return innerModuleOp.emitError("failed to serialize to torq binary");
        }

        auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
            variantOp.getLoc(), variantOp.getSymName(), variantOp.getTarget().getFormat(),
            binaryAttr
        );

        binaryOp.setMimeTypeAttr(executableBuilder.getStringAttr("application/x-flatbuffers"));

        return success();
    }

  private:
    const TorqTargetOptions &options_;
};

std::shared_ptr<IREE::HAL::TargetDevice> createTarget(const TorqTargetOptions &options) {
    return std::make_shared<TorqTargetDevice>(options);
}

std::shared_ptr<IREE::HAL::TargetBackend> createBackend(const TorqTargetOptions &options) {
    return std::make_shared<TorqTargetBackend>(options);
}

} // namespace mlir::syna::torq
