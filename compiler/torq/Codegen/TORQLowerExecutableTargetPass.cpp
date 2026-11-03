// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "TilingUtils.h"

#include "torq/Codegen/BufferizationUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Passes.h"
#include "torq/Conversions/TorqHLToTorqHW/Passes.h"
#include "torq/Conversions/TosaToTorqHL/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Transforms/Linalg/Passes.h"
#include "torq/Transforms/TorqHL/Passes.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/PassProgressUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>

#define DEBUG_TYPE "iree-torq-lower-executable-target-pass"

static llvm::cl::opt<bool> clShowPassProgress(
    "torq-show-pass-progress",
    llvm::cl::desc("Print progress messages for each pass in the Torq pipeline"),
    llvm::cl::init(false)
);

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

static llvm::cl::opt<bool> clEnableTorqProfiling(
    "torq-enable-profiling", llvm::cl::desc("enable torq profiling"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clEnableNssCodeLoadOverlap(
    "torq-enable-nss-code-load-overlap",
    llvm::cl::desc("Load NSS program code while the program before it runs"), llvm::cl::init(true)
);

// Off by default until the hardware gain is confirmed: the compile-time profile counts the overlap
// as free (it does not model the shared bandwidth), and the prefetch list scheduler plans for one
// DMA engine that runs loads and stores in turn.
static llvm::cl::opt<bool> clEnableDmaChannelPairing(
    "torq-enable-dma-channel-pairing",
    llvm::cl::desc("Run the input and output DMA channels at the same time where the addresses "
                   "allow it (experimental)"),
    llvm::cl::init(false)
);

static llvm::cl::opt<std::string> clDebugInfo(
    "torq-debug-info", llvm::cl::desc("save debug information for a model"), llvm::cl::init("")
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

// Kept as a flag for debugging/testing: compiles inputs that are already bufferized (memrefs
// instead of tensors), skipping the bufferization, allocation and map-bindings passes. Off by
// default; the normal pipeline is unaffected.
static llvm::cl::opt<bool> clFromPreBufferizedIR(
    "torq-from-prebufferized-ir",
    llvm::cl::desc("Compile pre-bufferized MLIR inputs (used for testing)"), llvm::cl::init(false)
);

static llvm::cl::opt<bool> clDisableSeg(
    "torq-disable-segmentation-fusion",
    llvm::cl::desc("Disable fusion of segmentation operations with producer"), llvm::cl::init(false)
);

llvm::cl::opt<bool> clDisableSlicing(
    "torq-disable-slicing", llvm::cl::desc("disable slicing"), llvm::cl::init(false)
);

static llvm::cl::opt<std::string> clExecutorMap(
    "torq-executor-map",
    llvm::cl::desc("Path to executor assignment JSON from capability profiler"), llvm::cl::init("")
);

static llvm::cl::opt<bool> clEnableTransposeOptimization(
    "torq-enable-transpose-optimization",
    llvm::cl::desc("enable transpose layout optimization pass"), llvm::cl::init(true)
);

enum class IsNanMode { Lower, Remove };

static llvm::cl::opt<IsNanMode> clTorqIsNanMode(
    "torq-isnan-mode", llvm::cl::desc("Choose how to handle IsNaN operations"),
    llvm::cl::values(
        clEnumValN(IsNanMode::Lower, "lower", "Lower IsNaN operations to comparisons"),
        clEnumValN(
            IsNanMode::Remove, "remove",
            "Completely remove IsNaN checks: improves performance but risky; assumes graph never "
            "contains IsNaN"
        )
    ),
    llvm::cl::init(IsNanMode::Lower)
);

static llvm::cl::opt<PrefetchAlgorithm> clPrefetchAlgorithm(
    "torq-prefetch-algorithm",
    llvm::cl::desc("how DMA transfers are moved to overlap slice compute"),
    llvm::cl::values(
        clEnumValN(PrefetchAlgorithm::Auto, "auto", "constant-loads"),
        clEnumValN(PrefetchAlgorithm::None, "none", "no prefetching"),
        clEnumValN(
            PrefetchAlgorithm::ConstantLoads, "constant-loads",
            "load each constant in front of the wait of the program before its consumer "
            "(torq-prefetch-constant-loads)"
        ),
        clEnumValN(
            PrefetchAlgorithm::StreamSchedule, "stream-schedule",
            "reschedule the whole NSS stream over both slices, the DMA engine and LRAM "
            "(torq-prefetch-inputs), and give matmul tile loops a smaller tile budget so the "
            "next weight tile can load meanwhile"
        )
    ),
    llvm::cl::init(PrefetchAlgorithm::Auto)
);

PrefetchAlgorithm prefetchAlgorithm() {
    if (clPrefetchAlgorithm == PrefetchAlgorithm::Auto) {
        return PrefetchAlgorithm::ConstantLoads;
    }
    return clPrefetchAlgorithm;
}

bool prefetchInputsEnabled() { return prefetchAlgorithm() == PrefetchAlgorithm::StreamSchedule; }

namespace {

// The usable LRAM size for a pipeline built with `budgetPercent` of LRAM. Only a tile-fit
// probe that TileAndFuse builds for a smaller tile budget gets less than 100.
int64_t getLramSizeBasedOnBudget(int budgetPercent) {
    int64_t lramSize = TorqHw::get().getLramSize() * budgetPercent / 100;
    int nssProgramPoolSize = HwInfo::nss_max_program_size * 2;
    lramSize -= nssProgramPoolSize;
    return lramSize;
}

// CompileTimeConst handling depends on whether the tile-and-fuse memory-fit
// simulation is driving the pipeline: the probe runs stubs (constants stay
// fake), the real pipeline outlines and computes them.
void addCompileTimeConstPasses(OpPassManager &funcPm, bool optimizeForTileAndFuse) {
    if (optimizeForTileAndFuse) {
        funcPm.addPass(createCompileTimeConstStubsPass());
    }
    else {
        funcPm.addPass(createCompileTimeConstOutlinePass());
        funcPm.addPass(createCompileTimeConstComputePass());
    }
    funcPm.addPass(createCanonicalizerPass());
}

void addPostTileAndFuseLoweringPasses(
    OpPassManager &funcPm, bool optimizeForTileAndFuse, SliceBufferSharing sliceBufferSharing,
    int budgetPercent
) {
    if (!clDisableSlicing && TorqHw::get().getSliceCount() > 1) {
        // Group each pure-elementwise chain (erf/tanh/mul) before slicing, so the
        // chain slices as one scf.forall and its middle values stay in LRAM.
        funcPm.addPass(createCoalesceElementwiseChainsPass());
        funcPm.addPass(createFuseTransposeIntoGroupPass());
        funcPm.addPass(createLinalgSlicingPass(TorqHw::get().getSliceCount()));
    }
    funcPm.addPass(createReductionTilingPass());

    // lower the linalg operators to torq_hl before tiling
    funcPm.addPass(createLinalgToTorqHLPreConversionPass());

    // Padding, then weight layout and input segmentation of the conv and pool kernels, in this
    // order and without a canonicalizer between them
    funcPm.addPass(createAlignStride2ConvPoolPaddingPass());
    funcPm.addPass(createLegalizeStride2ConvPaddingPass());
    funcPm.addPass(createEvenStride2ConvPoolInputsPass());
    funcPm.addPass(createConvPoolSamePaddingPass());
    funcPm.addPass(createConvPoolKernelLayoutPass());

    funcPm.addPass(createCanonicalizerPass());

    addCompileTimeConstPasses(funcPm, optimizeForTileAndFuse);

    funcPm.addPass(createMarkHostExecutorPass());

    // lower arith ops to torq_hl
    funcPm.addPass(createArithToTorqHLConversionPass());
    funcPm.addPass(createCanonicalizerPass());

    // lower the linalg operators to torq_hl
    funcPm.addPass(createLinalgToTorqHLConversionPass());
    funcPm.addPass(createCanonicalizerPass());

    if (!clDisableSeg) {
        funcPm.addPass(torq_hl::createTorqHLOptimizeSegmentationPass());
    }

    addCompileTimeConstPasses(funcPm, optimizeForTileAndFuse);

    funcPm.addPass(createEncodeTensorsPass());
    auto lramSize = getLramSizeBasedOnBudget(budgetPercent);
    funcPm.addPass(createFoldConvertPass(lramSize, sliceBufferSharing));
    funcPm.addPass(createCanonicalizerPass());

    // Tile TorqHL ops with large inputs before encoding so the tiled ops receive
    // the same encoding as all other ops.  The pass inline-unrolls its loops so
    // all tensor shapes are static by the time EncodeTensorsPass runs.
    if (!optimizeForTileAndFuse) {
        funcPm.addPass(createTorqHLTileLargeInputsPass());
        // Tiling leaves each tile slicing a whole-tensor torq_hl.convert; swap
        // that to only convert the slice each tile actually reads.
        funcPm.addPass(createSwapConvertExtractSlicePass());
    }
}

void addCpuPasses(OpPassManager &pm, int budgetPercent = 100) {

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

    // since to run on CSS we inserted a bunch of copies to LRAM we can
    // now simplify them
    auto lramSize = getLramSizeBasedOnBudget(budgetPercent);
    funcPm.addPass(createFoldConvertPass(lramSize));

    funcPm.addPass(createCanonicalizerPass());
}

void addNssUpToAssignLramAddresses(
    OpPassManager &pm, bool optimizeForTileAndFuse = false, bool isRealPipeline = false,
    int budgetPercent = 100
) {
    auto &funcPm = pm.nest<func::FuncOp>();

    if (!clFromPreBufferizedIR) {

        // TODO: Do we really need torq_hl::ConstOp? It is used only to get XRAM address, can't
        // we find another way to do that with arith::ConstantOp?
        funcPm.addPass(createLowerArithConstantsPass());

        addTorqComprehensiveBufferizePasses(
            funcPm, createTorqAllocation, createTorqCopy, getTorqMemSpaceAttr
        );

        funcPm.addPass(createEraseHALDescriptorTypeFromMemRefPass());

        int64_t lramSize = getLramSizeBasedOnBudget(budgetPercent);
        funcPm.addPass(createLowerHostCopiesToNpuPass(lramSize));

        funcPm.addPass(createMapBindingsPass());
    }

    funcPm.addPass(createLowerCallProgramToStartWaitPass());

    // unroll all loops since NSS cannot deal with them
    funcPm.addPass(createUnrollLoopPass());

    // Bindings are resolved after MapBindings, so element-reinterpreting
    // torq_hl.reinterpret_cast views (bufferized torq_hl.bitcast) now have
    // constant offsets; the canonicalizer constifies them (the op's
    // ReinterpretCastOpConstantFolder) as the slice argument address
    // resolution and the LRAM address assignment only accept views with
    // static geometry. (Entries that only become constant after the forall
    // unrolling are constified by the second canonicalizer below.)
    funcPm.addPass(createCanonicalizerPass());
    funcPm.addPass(createCSEPass());

    // always on: only splits flag- or marker-selected constants
    funcPm.addPass(createSplitConstantsPass());

    funcPm.addPass(torq_hl::createTorqHLFoldTableConstantPass());

    funcPm.addPass(createOutlineSliceProgramsPass());
    // Reuse LRAM buffers for repeated read-only loads (const/input) within each
    // slice-program region BEFORE the forall body is unrolled per slice, so the
    // saving replicates onto every slice. Must run pre-unroll: the slices share LRAM,
    // but this pass would merge loads made inside different slice bodies, which is
    // known to miscompile. Loads that FoldConvert moved in front of the forall are
    // already shared by all the slices.
    funcPm.addPass(createEliminateRedundantLramLoadsPass());
    funcPm.addPass(createUnrollForallLoopsPass());
    funcPm.addPass(createScheduleSliceProgramsPass());

    // Canonicalize to fold constants and types after the loop unrolling in the
    // previous pass; this also constifies element-reinterpreting
    // torq_hl.reinterpret_cast views whose offsets only became constants here
    // (the address assignment only accepts views with static geometry).
    funcPm.addPass(createCanonicalizerPass());

    // Lower slice ProgramOp bodies to TorqHW before deallocation/addressing.
    // Kept here so memory-management passes run after the op form is stabilized.
    //
    // Skipped for the tile-and-fuse memory-fit trial
    if (!optimizeForTileAndFuse) {
        funcPm.addNestedPass<torq_hl::ProgramOp>(createConvertSliceProgramToTorqHwPass());

        if (clEnableTorqProfiling || prefetchInputsEnabled()) {
            funcPm.addPass(createAnnotateDmaAndSliceCyclesPass());
        }
    }

    // The total LRAM pool for the allocator, NSS program slots included. It is the whole
    // LRAM, except in a tile-fit probe built for a smaller tile budget.
    int64_t lramPoolSize =
        getLramSizeBasedOnBudget(budgetPercent) + HwInfo::nss_max_program_size * 2;

    // Reorder the NSS stream so that loads overlap slice compute. PrefetchInputs places
    // with the same pool size AssignLramAddresses uses.
    if (!optimizeForTileAndFuse && prefetchInputsEnabled()) {
        funcPm.addPass(createPrefetchInputsPass(lramPoolSize));
    }

    addNssLramAllocationPasses(funcPm, isRealPipeline, lramPoolSize);
}

void addNssPostAssignLramAddressesPasses(OpPassManager &pm) {
    {
        auto &funcPm = pm.nest<func::FuncOp>();

        // size slice code sections to their compiled programs so the XRAM layout
        // does not reserve the outlining upper bound for every invocation
        funcPm.addPass(createSizeSliceCodeSectionsPass());

        // assign addresses to DTCM, ITCM, and XRAM
        funcPm.addPass(createAssignDtcmItcmXramAddressesPass());

        // compile slice programs
        funcPm.addPass(createCompileSliceInvocationsPass());
    }

    // create globals for all objects used in the function
    pm.addPass(createCreateGlobalsPass());

    {
        auto &funcPm = pm.nest<func::FuncOp>();

        // create NSS programs with all the NSS instructions
        funcPm.addPass(createOutlineNSSProgramsPass());

        funcPm.addPass(createCanonicalizerPass());

        // segment the NSS programs in small blocks that fit the NSS block size
        funcPm.addPass(createSegmentNSSProgramsPass());

        // load the next NSS program code while the program before it runs
        if (clEnableNssCodeLoadOverlap) {
            funcPm.addPass(createOverlapNSSCodeLoadsPass());
        }

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

        // run the two DMA channels at the same time where the addresses allow it
        if (clEnableDmaChannelPairing) {
            funcPm.addPass(createPairDmaChannelsPass());
        }

        // measure the actual NSS program sizes and assign their XRAM addresses
        funcPm.addPass(createAssignNSSProgramsAddressesPass());

        // after the NSS block sizes are measured, so that each code block load is profiled with
        // the size it really transfers
        if (clEnableTorqProfiling) {
            funcPm.addPass(createProfilingPass());
        }

        // compile NSS programs
        funcPm.addPass(createCompileNSSInvocationsPass());
    }
}

void addNssPasses(OpPassManager &pm) {
    addNssUpToAssignLramAddresses(pm);
    addNssPostAssignLramAddressesPasses(pm);
}

void addSlicePassesUpToTileAndFuse(OpPassManager &pm) {
    auto &funcPm = pm.nest<func::FuncOp>();

    // Decompose out-of-limits named convs that are already in NCHW/NCW form
    // (this catches every conv1d, which has no NHWC variant) before
    // OptimizeLinalgForTorq runs: the in-range conv1d sub-convs it produces
    // then flow through the conv1d->conv2d conversion, so no named conv1d
    // reaches tiling.
    //
    // DecomposeConvToSpaceToDepthPass runs first: for an out-of-limits conv2d
    // whose kernel and stride share a common divisor it shrinks the conv into
    // a single in-limits conv via SpaceToDepth, which is cheaper than the
    // slice-and-add decomposition; everything else bails and falls through to
    // DecomposeConvsForHardwarePass.
    funcPm.addPass(createDecomposeConvToSpaceToDepthPass());
    funcPm.addPass(createDecomposeConvsForHardwarePass());

    // optimize linalg ops for torq
    // this pass use some tags from tile-and-fuse mark pass
    funcPm.addPass(createOptimizeLinalgForTorqPass());
    // remove IsNaN checks: improves performance but risky; assumes graph never contains IsNaN
    if (clTorqIsNanMode.getValue() == IsNanMode::Remove) {
        funcPm.addPass(createRemoveIsNanPass());
    }
    // Fold IREE 3.10's rank-5 depthwise reshape wrapper to rank-4 so the
    // 4D-only NHWC->NCHW conversion below sees a uniform rank-4 chain (#1412).
    funcPm.addPass(createFoldUnitExtentWrapperPass());
    // Convert NHWC conv/pool/depthwise NHWC to NCHW
    funcPm.addPass(createConvertNhwcOpToNchwPass());
    funcPm.addPass(createCanonicalizerPass());

    // Second decomposition round: catches convs that only reached NCHW/NCW
    // named form after the NHWC->NCHW conversion (or standardization) above.
    // Same ordering as the first round: SpaceToDepth first for the divisible
    // cases, then the exhaustive decomposition for everything else.
    funcPm.addPass(createDecomposeConvToSpaceToDepthPass());
    funcPm.addPass(createDecomposeConvsForHardwarePass());
    funcPm.addPass(createCanonicalizerPass());

    if (clEnableTransposeOptimization) {
        funcPm.addPass(createOptimizeTransposeLayoutPass());
    }

    // Convert tensor-level elementwise arith ops (e.g. addi, subi, andi) into
    // explicit linalg.generic form. This makes all elementwise math uniform
    // in Linalg, so later passes like createLinalgTilePass and pattern matching can
    // handle them consistently.
    funcPm.addPass(mlir::createConvertElementwiseToLinalgPass());

    if (!clExecutorMap.empty()) {
        funcPm.addPass(createExecutorAssignmentPass(clExecutorMap));
        funcPm.addPass(createCanonicalizerPass());
    }

    // After the last canonicalizer before tile-and-fuse: it would fold the two transposes
    // back into one
    funcPm.addPass(createSplitSmallChannelTransposePass());
    funcPm.addPass(createMarkPatternsForTileAndFusePass());
    funcPm.addPass(createReductionTilingPass(/*longReductionsOnly=*/true));
    funcPm.addPass(createTileAndFusePass(TorqHw::get().getSliceCount()));
}

void addSlicePassesAfterTileAndFuse(OpPassManager &pm, SliceBufferSharing sliceBufferSharing) {
    // NB: do not add passes after Tile and Fuse here, place them in
    // addPassesPostTileAndFuseUpToAssignLramAddresses, so T&F can use them too.
    addPassesPostTileAndFuseUpToAssignLramAddresses(
        pm, false, /*isRealPipeline=*/true, sliceBufferSharing
    );
    addNssPostAssignLramAddressesPasses(pm);
}

FailureOr<func::FuncOp> getDispatchFunction(ModuleOp moduleOp) {
    auto funcOps = moduleOp.getOps<func::FuncOp>();

    if (std::distance(funcOps.begin(), funcOps.end()) != 1) {
        return moduleOp.emitOpError("expected a single function in the module");
    }

    return *funcOps.begin();
}

LogicalResult saveDebugInfo(FunctionOpInterface op, const std::string &outputPath) {

    auto funcName = op.getName();

    llvm::sys::fs::create_directories(outputPath);

    std::string fileName = outputPath + "/" + funcName.str() + ".mlirb";

    std::error_code ec;

    llvm::raw_fd_ostream os(fileName, ec, llvm::sys::fs::OF_None);

    if (ec) {
        llvm::errs() << "Error opening file " << fileName << ": " << ec.message() << "\n";
        return failure();
    }

    if (failed(mlir::writeBytecodeToFile(op->getParentOp(), os))) {
        llvm::errs() << "Error writing bytecode to file " << fileName << "\n";
        return failure();
    }

    os.close();

    return success();
}

bool usesTileAndFuse() { return !clFromPreBufferizedIR && !clDisableSlices; }

// The whole pipeline, or with tile-and-fuse only up to and including TileAndFuse (see
// runAfterTileAndFuse).
void addAllPasses(OpPassManager &pipeline) {
    pipeline.addPass(createAnalyzeTensorSizesPass());

    if (usesTileAndFuse()) {
        addSlicePassesUpToTileAndFuse(pipeline);
        return;
    }

    if (!clDisableCSS || !clDisableHost)
        addCpuPasses(pipeline);

    addNssPasses(pipeline);
}

/// Lowers a hal.executable.variant inner module to TORQ scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code,
/// - then convert to TORQ dialect.
struct TORQLowerExecutableTargetPass
    : public impl::TORQLowerExecutableTargetBase<TORQLowerExecutableTargetPass> {
    TORQLowerExecutableTargetPass() = default;
    TORQLowerExecutableTargetPass(const TORQLowerExecutableTargetPass &pass) {}

    using RunPipelineFn = function_ref<LogicalResult(function_ref<void(OpPassManager &)>)>;

    // Run the pipeline after TileAndFuse with the slices of each scf.forall sharing their
    // buffers (see FoldConvert). Sharing keeps a buffer resident across all the slices, and
    // TileAndFuse picked its tiles without it, so on rare occasions LRAM planning then fails.
    // Only the real planner can tell, so in that case run again from the same IR with less
    // sharing. The shared outputs are what add residency, so drop them first and keep the
    // shared inputs, which save LRAM.
    LogicalResult runAfterTileAndFuse(RunPipelineFn runPipeline) {
        ModuleOp moduleOp = getOperation();
        OwningOpRef<ModuleOp> original = moduleOp.clone();

        auto runWith = [&](SliceBufferSharing sharing) {
            return runPipeline([sharing](OpPassManager &pm) {
                addSlicePassesAfterTileAndFuse(pm, sharing);
            });
        };

        for (SliceBufferSharing sharing :
             {SliceBufferSharing::InputsAndOutputs, SliceBufferSharing::Inputs}) {
            SmallVector<Diagnostic> diagnostics;
            bool outOfMemory = false;
            LogicalResult result = failure();
            {
                // Hold back this attempt's diagnostics: they are dropped if we try again. The
                // handler is on the shared context, so leave other threads' diagnostics alone.
                const uint64_t ownerThreadId = llvm::get_threadid();
                ScopedDiagnosticHandler holdBack(moduleOp.getContext(), [&](Diagnostic &diag) {
                    if (llvm::get_threadid() != ownerThreadId)
                        return failure();
                    outOfMemory |= diag.str() == OUT_OF_MEMORY_MESSAGE;
                    diagnostics.push_back(std::move(diag));
                    return success();
                });
                result = runWith(sharing);
            }

            if (succeeded(result) || !outOfMemory) {
                for (Diagnostic &diag : diagnostics)
                    moduleOp.getContext()->getDiagEngine().emit(std::move(diag));
                return result;
            }

            LLVM_DEBUG(
                llvm::dbgs() << "LRAM planning failed with slice buffer sharing "
                             << static_cast<int>(sharing) << ", retrying with less\n"
            );
            OwningOpRef<ModuleOp> copy = original->clone();
            moduleOp.getBodyRegion().takeBody(copy->getBodyRegion());
        }

        return runWith(SliceBufferSharing::None);
    }

    // At 3.12 the export workgroup-count region keeps a workgroup_count_from_slice
    // placeholder unless a codegen pass resolves it. The pass that did this for us
    // (TileAndDistributeToWorkgroupsPass) was removed upstream. TORQ dispatches are
    // a single host-visible workgroup — the slice programs carry the real
    // parallelism — matching what the old pass produced. Resolve leftover
    // placeholders to (1, 1, 1) so the op never leaks into host code, where it has
    // no VM lowering.
    static void resolveWorkgroupCountRegion(FunctionOpInterface funcOp) {
        std::optional<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
        if (!exportOp)
            return;
        Block *body = exportOp->getWorkgroupCountBody();
        if (!body)
            return;
        SmallVector<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp> placeholderOps;
        body->walk([&](IREE::TensorExt::DispatchWorkgroupCountFromSliceOp op) {
            placeholderOps.push_back(op);
        });
        for (auto op : placeholderOps) {
            OpBuilder builder(op);
            SmallVector<Value> ones(
                op.getNumResults(), arith::ConstantIndexOp::create(builder, op.getLoc(), 1)
            );
            op->replaceAllUsesWith(ones);
            op->erase();
        }
    }

    void runOnOperation() override {

        LLVM_DEBUG({ llvm::dbgs() << "Lowering dispatch " << getOperation().getSymName() << "\n"; }
        );

        // find the dispatch function we are processing
        auto maybeDispatchFuncOp = getDispatchFunction(getOperation());
        if (failed(maybeDispatchFuncOp)) {
            return signalPassFailure();
        }

        auto dispatchName = maybeDispatchFuncOp->getName().str();

        // Create a shared DefaultTimingManager so that --mlir-timing instruments both
        // inner PassManagers in the same timing tree.
        //
        // Background: applyPassManagerCLOptions does NOT enable timing — it handles
        // crash reproducers, stats, and IR printing only. Timing is enabled separately
        // via applyDefaultTimingPassManagerCLOptions / enableTiming. Each call to
        // enableTiming() with no argument creates an independent DefaultTimingManager
        // that prints a separate per-dispatch report. Sharing one TimingScope ensures
        // all sub-pass timings are aggregated into a single report.
        DefaultTimingManager tm;
        applyDefaultTimingManagerCLOptions(tm);
        TimingScope timingScope = tm.getRootScope();

        // distribute the work to the workgroups (we have only one at the moment)
        PassManager distributeToWorkgroupsPipeline(
            getOperation().getContext(), maybeDispatchFuncOp->getOperationName()
        );
        if (clShowPassProgress) {
            distributeToWorkgroupsPipeline.addInstrumentation(
                std::make_unique<torq::ProgressLogger>(dispatchName + "::workgroups")
            );
        }
        distributeToWorkgroupsPipeline.addPass(createTileAndDistributeToWorkgroupsUsingForallOpPass(
        ));
        distributeToWorkgroupsPipeline.enableTiming(timingScope);
        if (failed(mlir::applyPassManagerCLOptions(distributeToWorkgroupsPipeline)))
            return signalPassFailure();
        if (failed(distributeToWorkgroupsPipeline.run(*maybeDispatchFuncOp))) {
            return signalPassFailure();
        }

        resolveWorkgroupCountRegion(*maybeDispatchFuncOp);

        auto runPipeline = [&](function_ref<void(OpPassManager &)> addPasses) {
            PassManager pipeline(getOperation().getContext(), getOperation().getOperationName());
            if (clShowPassProgress) {
                pipeline.addInstrumentation(std::make_unique<torq::ProgressLogger>(dispatchName));
            }
            addPasses(pipeline);
            pipeline.enableTiming(timingScope);
            if (failed(mlir::applyPassManagerCLOptions(pipeline)))
                return failure();
            return pipeline.run(getOperation());
        };

        if (failed(runPipeline(addAllPasses))) {
            return signalPassFailure();
        }

        if (usesTileAndFuse() && failed(runAfterTileAndFuse(runPipeline))) {
            return signalPassFailure();
        }

        if (!clDebugInfo.empty()) {
            // look the function up again: runAfterTileAndFuse may have replaced it
            if (failed(saveDebugInfo(*getDispatchFunction(getOperation()), clDebugInfo))) {
                return signalPassFailure();
            }
        }
    }
};

} // namespace

void addPassesPostTileAndFuseUpToAssignLramAddresses(
    OpPassManager &pipeline, bool optimizeForTileAndFuse, bool isRealPipeline,
    SliceBufferSharing sliceBufferSharing, int budgetPercent
) {
    // NB: This is the post-tiling pipeline used by TileAndFusePass for its memory-fit
    // simulation. It must match exactly what happens after TileAndFuse in the real
    // pipeline, up to and including LRAM address assignment.
    auto &funcPm = pipeline.nest<func::FuncOp>();

    // Keep the fuse group on reshapes that canonicalize rebuilds (dynamic -> static).
    funcPm.addPass(createFuseGroupPreservingCanonicalizerPass());
    if (optimizeForTileAndFuse) {
        funcPm.addPass(createReplaceForLoopsWithMidIterationPass());
    }
    funcPm.addPass(createPeelTileLoopsPass());

    addPostTileAndFuseLoweringPasses(
        funcPm, optimizeForTileAndFuse, sliceBufferSharing, budgetPercent
    );

    if (!clDisableCSS || !clDisableHost) {
        addCpuPasses(pipeline, budgetPercent);
    }

    addNssUpToAssignLramAddresses(pipeline, optimizeForTileAndFuse, isRealPipeline, budgetPercent);
}

void addNssLramAllocationPasses(OpPassManager &funcPm, bool isRealPipeline, int64_t lramSize) {
    if (!clFromPreBufferizedIR) {
        funcPm.addPass(createAddDeallocationPass());
    }

    // Only the real pipeline prefetches: a probe throws its result away, so the overlap
    // would buy nothing there and would hide its LRAM cost from the tile choice.
    if (isRealPipeline) {
        // only one prefetch algorithm runs in a compile
        if (prefetchAlgorithm() == PrefetchAlgorithm::ConstantLoads) {
            funcPm.addPass(createPrefetchConstantLoadsPass());
        }
    }

    // assign addresses to all allocations. The probe plans without re-planning, so the
    // tile it picks never depends on a forced compaction; the real pass re-plans for
    // what the passes after the probe change.
    funcPm.addPass(createAssignLramAddressesPass(/*replan=*/isRealPipeline, lramSize));
}

std::unique_ptr<OperationPass<ModuleOp>> createTORQLowerExecutableTargetPass() {
    return std::make_unique<TORQLowerExecutableTargetPass>();
}

} // namespace mlir::syna::torq
