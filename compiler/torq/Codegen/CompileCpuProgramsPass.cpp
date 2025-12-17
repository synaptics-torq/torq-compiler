// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "torq/Codegen/css_bootstrap/css_kernel_riscv.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "compiler/plugins/target/LLVMCPU/LLVMIRPasses.h"
#include "compiler/plugins/target/LLVMCPU/LLVMTargetOptions.h"
#include "compiler/plugins/target/LLVMCPU/LinkerTool.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "torq-compile-css-tasks"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

static llvm::cl::opt<bool> clEnableCSSForHost(
    "torq-css-host", llvm::cl::desc("Create CSS binaries suitable for running on the host"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clKeepCssLinkingArtifacts(
    "torq-keep-css-linking-artifacts",
    llvm::cl::desc("Keep the artifacts generated during linking of CSS kernels"),
    llvm::cl::init(false)
);

static llvm::cl::opt<std::string> clCreateCssSymbols(
    "torq-create-css-symbols",
    llvm::cl::desc("Store the symbols for all css programs in the specified directory"),
    llvm::cl::init("")
);

static llvm::cl::opt<std::string> clTargetHostTriple(
    "torq-target-host-triple",
    llvm::cl::desc("Specify the target host triple for the host CPU of the target"),
    llvm::cl::init("")
);

static llvm::cl::opt<std::string> clTargetHostCpu(
    "torq-target-host-cpu",
    llvm::cl::desc("Specify the target host cpu for the host CPU of the target"), llvm::cl::init("")
);

static llvm::cl::opt<std::string> clTargetHostCpuFeatures(
    "torq-target-host-cpu-features",
    llvm::cl::desc("Specify the target host cpu features for the host CPU of the target"),
    llvm::cl::init("")
);

namespace {

class CssLinker {

  public:
    CssLinker(std::string libraryName) : libraryName(libraryName) {

        // create a temporary file with the linker script
        loadLinkerScript();

        // create temporary files with the boostrap code for the kernel (asm and c code)
        auto cssConfig = TorqHw::get().getCssConfig();

        addObjectFile(cssConfig.kernel.as_string(), "c.o");
        addObjectFile(cssConfig.bootstrap.as_string(), "s.o");

        // link libc/libm/compiler_rt for soft float support
        if (cssConfig.mabi == "ilp32") {
            addObjectFile(cssConfig.libc.as_string(), "libc.a");
            addObjectFile(cssConfig.libm.as_string(), "libm.a");
            addObjectFile(cssConfig.compiler_rt.as_string(), "compiler_rt.a");
        }
    }

    void addObjectFile(std::string objectData, std::string objectName) {

        auto artifact = IREE::HAL::Artifact::createTemporary(libraryName, objectName);

        auto &os = artifact.outputFile->os();

        os.write(objectData.data(), objectData.size());
        os.flush();
        os.close();

        objectFiles.push_back(std::move(artifact));
    }

    LogicalResult link(
        IREE::HAL::ExecutableVariantOp variantOp, std::vector<uint8_t> &codeVec, bool asElf = false,
        bool largeItcm = false
    ) {

        auto libraryFile = IREE::HAL::Artifact::createTemporary(libraryName, asElf ? "elf" : "bin");

        const SmallVector<std::string> &toolNames{"iree-lld", "lld", "ld.lld", "lld-link"};

        std::string toolPath = iree_compiler::findTool(toolNames);

        if (toolPath.empty()) {
            variantOp.emitError() << "failed to find lld tool";
            return failure();
        }

        SmallVector<std::string> command;

        command.push_back(toolPath);

        /* Specify the linker mode (required for iree-lld )*/
        command.push_back("-flavor");
        command.push_back("gnu");

        /* Hide build info that makes files unreproducible. */
        command.push_back("--build-id=none");

        /* Use the specified linker script when generating the library*/
        command.push_back("-T");
        command.push_back(linkerScript.path);

        /* Set the output path where we want to generate the library */
        command.push_back("-o");
        command.push_back(libraryFile.path);

        /* Do not use dynamic linker (Disable output of .interp section) */
        command.push_back("--no-dynamic-linker");

        /* Do not link against shared libraries */
        command.push_back("--static");

        /* Only search directories specified on the command line. */
        command.push_back("--nostdlib");

        /* Strip all symbols */
        if (!asElf) {
            command.push_back("-s");
        }

        /* Output a bare metal binary instead of an ELF file */
        if (!asElf) {
            command.push_back("--oformat=binary");
        }

        command.push_back("--defsym __stack_size=" + std::to_string(HwInfo::css_stack_size));

        /* setup the right memory addresses depending on the target */

        auto config = TorqHw::get().getCssConfig();

        command.push_back("--defsym __itcm_start=0x" + llvm::utohexstr(config.itcmStart));
        command.push_back("--defsym __dtcm_start=0x" + llvm::utohexstr(config.dtcmStart));
        command.push_back("--defsym __css_regs_start=0x" + llvm::utohexstr(config.regsStart));

        if (largeItcm) {
            command.push_back("--defsym __itcm_size=" + std::to_string(HwInfo::itcm_size * 100));
        }
        else {
            command.push_back("--defsym __itcm_size=" + std::to_string(HwInfo::itcm_size));
        }

        /* Link the object files */
        for (auto &objectFile : objectFiles) {
            command.push_back(objectFile.path);
        }

        std::string fullCommand = llvm::join(command, " ");

        int ret = system(fullCommand.c_str());

        auto bareMetalFileData = libraryFile.read();

        if (ret != 0 || !bareMetalFileData || bareMetalFileData->empty()) {

            llvm::errs() << "linking command failed: " << fullCommand << "\n";

            keepIntermediates();

            if (!asElf) {
                llvm::dbgs() << "\ntrying to re-link as elf for debug purposes with large itcmSize "
                                "and debug information\n";
                std::vector<uint8_t> elfCodeVec;
                if (failed(link(variantOp, elfCodeVec, true, true))) {
                    variantOp.emitError()
                        << "failed to re-link as elf with large itcm size for debug purposes";
                    return failure();
                }

                llvm::dbgs() << "Relinked with large ITCM at " << libraryFile.path << "\n";
            }

            return failure();
        }

        if (clKeepCssLinkingArtifacts) {
            llvm::dbgs() << "linking command: " << fullCommand << "\n\n";
            keepIntermediates();
            libraryFile.outputFile->keep();

            if (!asElf) {
                llvm::dbgs() << "\nre-linking as elf for debug purposes\n";
                std::vector<uint8_t> elfCodeVec;
                if (failed(link(variantOp, elfCodeVec, true))) {
                    variantOp.emitError() << "failed to re-link as elf for debug purposes";
                    return failure();
                }
            }
        }

        codeVec.insert(codeVec.end(), bareMetalFileData->begin(), bareMetalFileData->end());

        return success();
    }

    void keepIntermediates() {
        for (auto &objectFile : objectFiles) {
            objectFile.keep();
        }
        linkerScript.keep();
    }

    void loadLinkerScript() {
        linkerScript = IREE::HAL::Artifact::createTemporary(libraryName, "ld");

        auto &os = linkerScript.outputFile->os();

        auto linkerScript = TorqHw::get().getCssConfig().linkerScript.as_string();
        os.write(linkerScript.c_str(), linkerScript.size());
        os.flush();
        os.close();
    }

  private:
    std::string libraryName;
    SmallVector<IREE::HAL::Artifact> objectFiles;
    IREE::HAL::Artifact linkerScript;
};

static FailureOr<std::vector<uint8_t>> linkSharedHostLibrary(
    const llvm::Triple targetTriple, IREE::HAL::LLVMTarget &target, std::string libraryName,
    SmallVector<std::string> objects
) {

    SmallVector<IREE::HAL::Artifact> objectFiles;

    for (auto object : objects) {
        auto objectFile = IREE::HAL::Artifact::createTemporary(libraryName, "o");
        auto &os = objectFile.outputFile->os();
        os << object;
        os.flush();
        os.close();
        objectFiles.push_back(std::move(objectFile));
    }

    std::unique_ptr<iree_compiler::IREE::HAL::LinkerTool> linkerTool;
    iree_compiler::IREE::HAL::LLVMTargetOptions options;
    options.target = target;
    linkerTool = iree_compiler::IREE::HAL::LinkerTool::getForTarget(targetTriple, options);
    if (!linkerTool) {
        llvm::errs() << "failed to create linker tool for target triple: " << targetTriple.str()
                     << "\n";
        return failure();
    }

    auto linkArtifactsOr = linkerTool->linkDynamicLibrary(libraryName, objectFiles);
    if (!linkArtifactsOr.has_value()) {
        llvm::errs() << "failed to link executable and generate target dylib (check "
                        "above for more specific error messages)";
        return failure();
    }
    auto &linkArtifacts = linkArtifactsOr.value();

    // Load the linked ELF file and pack into an attr.
    auto elfFile = linkArtifacts.libraryFile.read();
    if (!elfFile.has_value()) {
        llvm::errs() << "failed to read back dylib temp file at " << linkArtifacts.libraryFile.path;
        return failure();
    }

    // Create a constant with the code
    return std::vector<uint8_t>{elfFile->begin(), elfFile->end()};
}

class CompileCpuProgramsPass : public CompileCpuProgramsBase<CompileCpuProgramsPass> {
  public:
    using CompileCpuProgramsBase<CompileCpuProgramsPass>::CompileCpuProgramsBase;

    void runOnOperation() override;

  private:
    FailureOr<std::string> compile(
        IREE::HAL::ExecutableVariantOp variantOp, const IREE::HAL::LLVMTarget &target,
        llvm::TargetMachine &targetMachine
    );
    LogicalResult compileAndLink(IREE::HAL::ExecutableVariantOp variantOp);
};

static void addCssLoweringPasses(OpPassManager &pipeline) {

    OpPassManager &modulePassManager = pipeline.nest<ModuleOp>();

    {
        FunctionLikeNest functionPassManager(modulePassManager);
        addCommonTargetExecutablePreprocessingPasses(functionPassManager);
    }

    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager).addPass(createLLVMCPULowerExecutableTargetPass);

    FunctionLikeNest(modulePassManager).addPass(createEraseHALDescriptorTypeFromMemRefPass);

    FunctionLikeNest(modulePassManager)
        // Linalg -> SCF
        .addPass(createMemrefCopyToLinalgPass)
        .addPass(createConvertLinalgToLoopsPass)
        .addPass(createConvertBf16ArithToF32Pass)
        .addPass(createConvertBf16ToUInt16BuffersPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    // Handled tensor-type constants.
    addConstantBufferizePasses(modulePassManager);

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        // Handle complex operation conversion.
        .addPass(createConvertComplexToStandardPass)
        // math dialect elementry functions -> polynomial form.
        .addPass(createPolynomialApproximationPass)
        .addPass(createHoistStaticallyBoundAllocationsPass);

    FunctionLikeNest(modulePassManager)
        // Resolve get_buffer_descriptor ops. All structural buffer manipulations
        // must conclude before this point.
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass)
        .addPass(createCheckCssStackSizePass)
        // SCF -> CF
        .addPass(createConvertSCFToCFPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        // (HAL, IREE, Linalg, CF) -> LLVM
        .addPass(arith::createArithExpandOpsPass)
        .addPass(memref::createExpandOpsPass)
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createEmulateNarrowTypePass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    modulePassManager.addPass(iree_compiler::createConvertToLLVMPass(true));
    modulePassManager.addPass(createReconcileUnrealizedCastsPass());
    modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(createAddFastMathFlagsPass());

    // FIXME: here we unfuse the FMA ops to keep the code size small but
    // there may be better ways to do this
    pipeline.nest<ModuleOp>().nest<LLVM::LLVMFuncOp>().addPass(createLLVMCPUUnfuseFMAOpsPass());
}

static void addHostLoweringPasses(OpPassManager &pipeline) {

    OpPassManager &modulePassManager = pipeline.nest<ModuleOp>();

    {
        FunctionLikeNest functionPassManager(modulePassManager);
        addCommonTargetExecutablePreprocessingPasses(functionPassManager);
    }

    buildLLVMCPUCodegenPassPipeline(pipeline, false);

    // FIXME: here we unfuse the FMA ops to keep the code size small but
    // there may be better ways to do this
    pipeline.nest<ModuleOp>().nest<LLVM::LLVMFuncOp>().addPass(createLLVMCPUUnfuseFMAOpsPass());
}

FailureOr<std::string> CompileCpuProgramsPass::compile(
    IREE::HAL::ExecutableVariantOp variantOp, const IREE::HAL::LLVMTarget &target,
    llvm::TargetMachine &targetMachine
) {

    auto llvmCompilationPipeline = OpPassManager(variantOp.getOperationName());

    if (variantOp.getTarget().getBackend() == "llvm-css") {
        addCssLoweringPasses(llvmCompilationPipeline);
    }
    else {
        addHostLoweringPasses(llvmCompilationPipeline);
    }

    /* 1. Lower input MLIR to llvm_ir dialect */

    if (failed(runPipeline(llvmCompilationPipeline, variantOp))) {
        return failure();
    }

    /* 2. Translate the llvm_ir dialect to an actual LLVM IR object */

    llvm::LLVMContext context;

    auto llvmModule = mlir::translateModuleToLLVMIR(variantOp.getInnerModule(), context, "css");
    if (!llvmModule) {
        return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                        "dialect to the native llvm::Module";
    }

    /* 4. Compile the LLVM IR of the kernel into a object file using LLVM */

    if (variantOp.getTarget().getBackend() == "llvm-css") {

        // set -ffreestanding-like behavior.
        for (auto &func : *llvmModule) {
            func.addFnAttr("no-builtins");
        }
    }

    llvmModule->setDataLayout(targetMachine.createDataLayout());
    llvmModule->setTargetTriple(targetMachine.getTargetTriple().str());

    if (failed(IREE::HAL::runLLVMIRPasses(target, &targetMachine, llvmModule.get()))) {
        return variantOp.emitError()
               << "failed to run LLVM-IR opt passes for IREE::HAL::ExecutableOp "
                  "targeting '"
               << targetMachine.getTargetTriple().str() << "'";
    }

    std::string objectData;

    if (failed(IREE::HAL::runEmitObjFilePasses(
            &targetMachine, llvmModule.get(), llvm::CodeGenFileType::ObjectFile, &objectData
        ))) {
        return variantOp.emitError() << "failed to compile LLVM-IR module to an object file";
    }

    return objectData;
}

LogicalResult CompileCpuProgramsPass::compileAndLink(IREE::HAL::ExecutableVariantOp variantOp) {

    /* 0. Create an LLVM target suitable for the target */

    bool hostBuild = variantOp.getTarget().getBackend() == "llvm-host" || clEnableCSSForHost;

    std::optional<IREE::HAL::LLVMTarget> maybeTarget;

    if (hostBuild) {

        auto hostTriple =
            clTargetHostTriple.empty() ? llvm::sys::getProcessTriple() : clTargetHostTriple;
        auto hostCpu = clTargetHostCpu.empty() ? std::string("host") : clTargetHostCpu;
        auto hostFeatures =
            clTargetHostCpuFeatures.empty() ? std::string("host") : clTargetHostCpuFeatures;

        maybeTarget = IREE::HAL::LLVMTarget::create(hostTriple, hostCpu, hostFeatures, false);
    }
    else {

        maybeTarget = IREE::HAL::LLVMTarget::create(
            "riscv32-pc-linux-elf", "generic-rv32", TorqHw::get().getCssConfig().mattrs, true
        );
    }

    if (!maybeTarget) {
        return variantOp.emitError() << "failed to create target";
    }

    auto target = *maybeTarget;

    auto targetMachine = createTargetMachine(target);
    if (!targetMachine) {
        return variantOp.emitError() << "failed to create target machine for target triple '"
                                     << target.getTriple() << "'";
    }

    /* 1. Compile the variant to an object file */

    auto maybeObject = compile(variantOp, target, *targetMachine);

    if (failed(maybeObject)) {
        return failure();
    }

    /* 2. Link the target program */

    auto libraryName = variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    std::vector<uint8_t> codeVec;

    if (hostBuild) {

        auto maybeCodeVec = linkSharedHostLibrary(
            targetMachine->getTargetTriple(), target, libraryName, {*maybeObject}
        );

        if (failed(maybeCodeVec)) {
            return failure();
        }

        codeVec = *maybeCodeVec;
    }
    else {

        CssLinker linker(libraryName);

        linker.addObjectFile(*maybeObject, "o");

        if (failed(linker.link(variantOp, codeVec))) {
            return failure();
        }

        if (clCreateCssSymbols != "") {

            std::vector<uint8_t> elfVec;

            if (failed(linker.link(variantOp, elfVec, true))) {
                return failure();
            }

            auto cssProgramExecutable = variantOp.getParentOp<IREE::HAL::ExecutableOp>();

            auto dispatchExecutable =
                cssProgramExecutable->getParentOfType<IREE::HAL::ExecutableOp>();

            std::string dumpPath = clCreateCssSymbols + "/" + dispatchExecutable.getName().str();

            llvm::sys::fs::create_directories(dumpPath);

            std::string fileName = dumpPath + "/" + cssProgramExecutable.getName().str() + ".elf";

            std::error_code ec;
            llvm::raw_fd_ostream outFile(fileName, ec, llvm::sys::fs::OF_None);
            if (ec) {
                llvm::errs() << "Error opening file " << fileName << ": " << ec.message() << "\n";
                return failure();
            }
            else {
                outFile.write(reinterpret_cast<const char *>(elfVec.data()), elfVec.size());
                outFile.close();
            }
        }
    }

    /* 3. Substitute the Variant IR with the compiled program  */

    OpBuilder builder(variantOp);

    builder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), "code", hostBuild ? "host" : "css", codeVec
    );

    variantOp.erase();

    return success();
}

void CompileCpuProgramsPass::runOnOperation() {

    auto moduleOp = getOperation();

    SmallVector<IREE::HAL::ExecutableVariantOp> variantOps;
    for (auto executableOp : moduleOp.getOps<IREE::HAL::ExecutableOp>()) {
        for (auto variant : executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>()) {
            variantOps.push_back(variant);
        }
    }

    for (auto variantOp : variantOps) {
        if (failed(compileAndLink(variantOp))) {
            return signalPassFailure();
        }
    }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCompileCpuProgramsPass() {
    return std::make_unique<CompileCpuProgramsPass>();
}

} // namespace mlir::syna::torq
