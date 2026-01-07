// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TorqHLToTorqHW/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "iree-torq-lower-executable-target-pass"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

using CodeGenPipeline = IREE::Codegen::DispatchLoweringPassPipeline;

static llvm::cl::opt<bool> clEnableTorqProfiling(
    "torq-enable-profiling", llvm::cl::desc("enable torq profiling"), llvm::cl::init(false)
);

llvm::cl::opt<bool> clDisableSlices(
    "torq-disable-slices", llvm::cl::desc("Disable execution of programs on slices"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clDisableCSS(
    "torq-disable-css", llvm::cl::desc("Disable execution of programs on CSS"),
    llvm::cl::init(false)
);

llvm::cl::opt<bool> clDisableHost(
    "torq-disable-host", llvm::cl::desc("Disable execution of programs on host"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clFromPreBufferizedIR(
    "torq-from-prebufferized-ir",
    llvm::cl::desc("Compile pre-bufferized MLIR inputs (used for testing)"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clDisableSeg(
    "torq-disable-segmentation-fusion",
    llvm::cl::desc("Disable fusion of segmentation operations with producer"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clEnableTorqTileAndFuse(
    "torq-enable-tile-and-fuse", llvm::cl::desc("enable torq tiling"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clForceTorqHLTiling(
    "torq-force-torq-hl-tiling", llvm::cl::desc("force TorqHL tiling"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clDisableSlicing(
    "torq-disable-slicing", llvm::cl::desc("disable slicing"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clUnrollLoopAfterBufferization(
    "torq-unroll-loop-after-bufferization", llvm::cl::desc("unroll loops after bufferization"),
    llvm::cl::init(false)
);

namespace {
/// Lowers a hal.executable.variant inner module to TORQ scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to TORQ dialect.
class TORQLowerExecutableTargetPass
    : public TORQLowerExecutableTargetBase<TORQLowerExecutableTargetPass> {
  public:
    TORQLowerExecutableTargetPass() = default;
    TORQLowerExecutableTargetPass(const TORQLowerExecutableTargetPass &pass) {}

    void runOnOperation() override;

  private:
    void addSlicePasses(OpPassManager &pm);
    void addCpuPasses(OpPassManager &pm);
    void addNssPasses(OpPassManager &pm);
};
} // namespace

void TORQLowerExecutableTargetPass::addSlicePasses(OpPassManager &pm) {

    auto &funcPm = pm.nest<func::FuncOp>();

    funcPm.addPass(createDecomposeSoftmaxPass());

    // optimize linalg ops for torq
    // this pass use some tags from tile-and-fuse mark pass
    funcPm.addPass(createOptimizeLinalgForTorqPass());
    funcPm.addPass(createCanonicalizerPass());

    // Convert tensor-level elementwise arith ops (e.g. addi, subi, andi) into
    // explicit linalg.generic form. This makes all elementwise math uniform
    // in Linalg, so later passes like createLinalgTilePass and pattern matching can
    // handle them consistently.
    funcPm.addPass(mlir::createConvertElementwiseToLinalgPass());

    if (clEnableTorqTileAndFuse) {
        funcPm.addPass(createMarkPatternsForTileAndFusePass());
        funcPm.addPass(createTileAndFusePass());
        funcPm.addPass(createCanonicalizerPass());
        funcPm.addPass(createUnrollLoopPass());
        funcPm.addPass(createCanonicalizerPass());
    }

    // lower the linalg operators to torq_hl before tiling
#if !TORQ_EXPERIMENTAL_LINALG_CONV_TILING
    funcPm.addPass(createLinalgToTorqHLPreConversionPass());
    funcPm.addPass(createCanonicalizerPass());
#endif

    // Handles valid pad operations
    funcPm.addPass(createValidToSamePadPass());

    funcPm.addPass(createMarkHostExecutorPass());

    // tile the linalg ops or tilingInterface ops
    funcPm.addPass(createLramTilePass());
    funcPm.addPass(createCanonicalizerPass());

#ifdef ENABLE_TORQ_GENERIC
    // specialize linalg.generic ops to linalg named ops
    funcPm.addPass(createSpecializeLinalgGenericOpPass());
#endif // ENABLE_TORQ_GENERIC

    if (!clEnableTorqTileAndFuse) {
        // unroll loops
        // better call this pass later as possible
        // just after it all are torqhl related passes that need static rank or shape, etc.
        // some dynamic attr because of tile could be populated by unroll loop pass
        if (!clUnrollLoopAfterBufferization) {
            funcPm.addPass(createUnrollLoopPass());
            funcPm.addPass(createCanonicalizerPass());
        }
    }

    // lower arith ops to torq_hl
    funcPm.addPass(createArithToTorqHLConversionPass());
    funcPm.addPass(createCanonicalizerPass());

    // lower the linalg operators to torq_hl
#if TORQ_EXPERIMENTAL_LINALG_CONV_TILING
    funcPm.addPass(createLinalgToTorqHLPreConversionPass());
#endif
    funcPm.addPass(createLinalgToTorqHLConversionPass());
    funcPm.addPass(createCanonicalizerPass());

#ifdef ENABLE_TORQ_GENERIC
    // we fold all linalg.fill used to initialize pvalues to 0 to constants
    // so that they don't get converted to torq_hl.generic operations in the
    // next pass (we will simplify these p values when converting to torq_hw)
    funcPm.addPass(createFoldPValueInitsPass());
#endif // ENABLE_TORQ_GENERIC

    // FIXME: not working currently
    // convert all the linalg operations that we received to torq_hl.generic
    // so that we will be able to run them on the NPU, this is a conversion
    // that will fail if anything remains that cannot be converted to torq_hl.
    // generic (we don't know how to run those anyways)
    // funcPm.addPass(createLinalgToTorqHLGenericPass(true));

    if (!clEnableTorqTileAndFuse || clForceTorqHLTiling) {
        funcPm.addPass(createTorqHlTilePass());
    }
    funcPm.addPass(createKernelSelectionPass());
    // op segment output feature enabled by default, diabled it for cross-check
    if (!clDisableSeg) {
        funcPm.addPass(torq_hl::createTorqHLOptimizeSegmentationPass());
    }
    funcPm.addPass(createEncodeTensorsPass());

    // Note: slicing should be done *before* kernel selection, but kernel selection is doing weight
    // reorganinzation and this can't operate on a subview of the weights.
    const auto sliceCount = TorqHw::get().getSliceCount();
    LLVM_DEBUG({
        llvm::dbgs() << "Slicing " << (clDisableSlicing ? "disabled" : "enabled")
                     << " slice count: " << sliceCount << "\n";
    });
    if (!clDisableSlicing && sliceCount > 1) {
        funcPm.addPass(createSlicingPass());
    }

    funcPm.addPass(createFoldConvertPass());
    funcPm.addPass(createCompileTimeConstComputePass());
}

void TORQLowerExecutableTargetPass::addCpuPasses(OpPassManager &pm) {

    // tile the operations for CSS (at the moment all the operations are sent to CSS if it's not
    // disabled )
    if (!clDisableCSS) {

        auto &funcPm = pm.nest<func::FuncOp>();

        // mark any new operations that were created after the last time this pass was run
        funcPm.addPass(createMarkHostExecutorPass());

        // tile the linalg ops or tilingInterface op before lowering to css tasks
        funcPm.addPass(createDtcmTilePass());
        funcPm.addPass(createCanonicalizerPass());
    }

    // outline all the operation that were not transformed into torq_hl kernels
    // to CPU programs and compile them

    pm.addPass(createAssignOperationsToCpuProgramsPass(clDisableCSS, clDisableHost));
    pm.addPass(createOutlineCpuProgramsPass());
    pm.addPass(createCompileCpuProgramsPass());

    auto &funcPm = pm.nest<func::FuncOp>();

    // make sure any remaining scalar in the IR is wrapped in a tensor
    // this happens when one input of a program was a scalar and was produced
    // by another program
    funcPm.addPass(createScalarsToTensorsPass());

    if (!clUnrollLoopAfterBufferization) {
        funcPm.addPass(createUnrollLoopPass());
    }

    // since to run on CSS we inserted a bunch of copies to LRAM we can
    // now simplify them
    funcPm.addPass(createFoldConvertPass());

    funcPm.addPass(createCanonicalizerPass());
}

void TORQLowerExecutableTargetPass::addNssPasses(OpPassManager &pm) {

    auto &funcPm = pm.nest<func::FuncOp>();

    if (!clFromPreBufferizedIR) {

        // TODO: Do we really need torq_hl::ConstOp? It is used only to get XRAM address, can't
        // we find another way to do that with arith::ConstantOp?
        funcPm.addPass(createLowerArithConstantsPass());

        addTorqComprehensiveBufferizePasses(
            funcPm, createTorqAllocation, createTorqCopy, getTorqMemSpaceAttr
        );

        funcPm.addPass(createEraseHALDescriptorTypeFromMemRefPass());

        funcPm.addPass(createMapBindingsPass());
    }

    funcPm.addPass(createLowerCallProgramToStartWaitPass());

    if (clUnrollLoopAfterBufferization) {
        // unroll all loops since NSS cannot deal with them
        funcPm.addPass(createUnrollLoopPass());
        funcPm.addPass(createCanonicalizerPass());
    }

    funcPm.addPass(createOutlineSliceProgramsPass());

    // Canonicalize to fold constants and types after the loop unrolling in the previous pass
    funcPm.addPass(createCanonicalizerPass());

    if (!clFromPreBufferizedIR) {
        funcPm.addPass(createAddDeallocationPass());
    }

    if (!clUnrollLoopAfterBufferization) {
        // unroll all loops since NSS cannot deal with them
        funcPm.addPass(createUnrollLoopPass());
        funcPm.addPass(createCanonicalizerPass());
    }

    // assign addresses to all allocations
    funcPm.addPass(createAssignAddressesPass());

    // lower torq_hl operation inside torq_hl::ProgramOp to torq_hw operations
    funcPm.addNestedPass<torq_hl::ProgramOp>(createConvertSliceProgramToTorqHwPass());

    // create NSS programs with all the NSS instructions
    funcPm.addPass(createOutlineNSSProgramsPass());

    // annotate all create_invocation/wait_program operations with addresses/values based
    // on the execution flow
    funcPm.addPass(createResolveInvocationArgumentsPass());

    // lower torq_hl operation inside torq_hl::ProgramOp to torq_hw operations
    funcPm.addNestedPass<torq_hl::ProgramOp>(createConvertNssProgramToTorqHwPass());

    // find the numeric addresses of all the torq_hw operations by looking up the corresponding
    // values
    funcPm.addPass(createResolveAddressesPass());

    // add all object identifiers used in serialization to the IR
    funcPm.addPass(createAssignObjectsIdentifiersPass());

    if (clEnableTorqProfiling) {
        funcPm.addPass(createProfilingPass());
    }
}

static FailureOr<func::FuncOp> getDispatchFunction(ModuleOp moduleOp) {
    auto funcOps = moduleOp.getOps<func::FuncOp>();

    if (std::distance(funcOps.begin(), funcOps.end()) != 1) {
        return moduleOp.emitOpError("expected a single function in the module");
    }

    return *funcOps.begin();
}

void TORQLowerExecutableTargetPass::runOnOperation() {

    LLVM_DEBUG({ llvm::dbgs() << "Lowering dispatch " << getOperation().getSymName() << "\n"; });

    // find the dispatch function we are processing
    auto maybeDispatchFuncOp = getDispatchFunction(getOperation());

    if (failed(maybeDispatchFuncOp)) {
        return signalPassFailure();
    }

    // distribute the work to the workgroups (we have only one at the moment)
    auto distributeToWorkgroupsPipeline = OpPassManager(maybeDispatchFuncOp->getOperationName());
    distributeToWorkgroupsPipeline.addPass(createTileAndDistributeToWorkgroupsPass());

    if (failed(runPipeline(distributeToWorkgroupsPipeline, *maybeDispatchFuncOp))) {
        return signalPassFailure();
    }

    auto pipeline = OpPassManager(getOperation().getOperationName());

    if (!clFromPreBufferizedIR && !clDisableSlices) {
        addSlicePasses(pipeline);
    }

    if (!clDisableCSS || !clDisableHost) {
        addCpuPasses(pipeline);
    }

    addNssPasses(pipeline);

    if (failed(runPipeline(pipeline, getOperation()))) {
        return signalPassFailure();
    }
}

std::unique_ptr<OperationPass<ModuleOp>> createTORQLowerExecutableTargetPass() {
    return std::make_unique<TORQLowerExecutableTargetPass>();
}

} // namespace mlir::syna::torq
