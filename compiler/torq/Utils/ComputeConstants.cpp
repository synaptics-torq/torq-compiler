
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/hal/local/executable_library.h"

#include "torq/Conversions/TorqHLToLinalg/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "llvm/Support/TargetSelect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"

#include <numeric>

#define DEBUG_TYPE "torq-compute-constants"

using namespace mlir::linalg;

namespace mlir::syna::torq {

llvm::cl::opt<std::string> clDumpComputeConstantsIR(
    "torq-dump-compute-constants-ir",
    llvm::cl::desc("Dump IR used to compute constants to a directory"), llvm::cl::init("")
);

llvm::cl::opt<bool> clInsertDebugTrapInComputeConstants(
    "torq-insert-debug-trap-in-compute-constants",
    llvm::cl::desc("Insert a debug trap in the generated code for computing constants"),
    llvm::cl::init(false)
);

using namespace mlir::iree_compiler;

static LogicalResult
createZeroConstant(Value value, Location loc, OpBuilder &builder, IRMapping &map) {

    auto zeroAttr = builder.getZeroAttr(value.getType());

    if (!zeroAttr) {
        return failure();
    }

    auto zeroValue = arith::ConstantOp::create(builder, loc, value.getType(), zeroAttr);

    // map any reference to assumeZero to zeroValue
    map.map(value, zeroValue);

    LLVM_DEBUG({
        llvm::dbgs() << "Created zero constant: ";
        zeroValue.dump();
    });

    return success();
}

static FailureOr<IREE::HAL::ExecutableVariantOp> createModule(
    ModuleOp topModuleOp, MLIRContext *context, Location loc, ArrayRef<Value> values,
    ArrayRef<SmallVector<Operation *>> opsSets, llvm::ArrayRef<Value> assumeZero
) {
    if (values.empty() || values.size() != opsSets.size()) {
        return failure();
    }

    OpBuilder builder(topModuleOp);

    builder.setInsertionPointToStart(topModuleOp.getBody());

    auto executableOp = IREE::HAL::ExecutableOp::create(builder, loc, "test");

    builder.setInsertionPointToStart(&executableOp.getBody().front());
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::get(
        context, builder.getStringAttr("llvm-native"), builder.getStringAttr("native")
    );
    auto variantOp = IREE::HAL::ExecutableVariantOp::create(builder, loc, "native", targetAttr);
    builder.setInsertionPointToStart(&variantOp.getBody().front());

    auto moduleOp = ModuleOp::create(builder, loc);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto funcOp = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType(TypeRange{}, TypeRange{})
    );

    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(),
        mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault
    );

    succeeded(setTranslationInfo(funcOp, translationInfo));

    funcOp.addEntryBlock();

    builder.setInsertionPointToStart(&funcOp.getBody().front());

    SmallVector<IREE::HAL::InterfaceBindingSubspanOp, 4> subspanOps;
    subspanOps.reserve(values.size());

    auto bindingAttr = IREE::HAL::PipelineBindingAttr::get(
        context, IREE::HAL::DescriptorType::StorageBuffer, IREE::HAL::DescriptorFlags::Indirect
    );

    SmallVector<IREE::HAL::PipelineBindingAttr> bindingsAttrs(values.size(), bindingAttr);

    auto layoutAttr = IREE::HAL::PipelineLayoutAttr::get(
        context, bindingsAttrs, 0, IREE::HAL::PipelineLayoutFlags::Indirect
    );

    // Create one output binding per requested value.
    for (auto [binding, value] : llvm::enumerate(values)) {

        RankedTensorType resultType = cast<RankedTensorType>(value.getType());
        auto dispatchTensorType = IREE::TensorExt::DispatchTensorType::get(
            IREE::TensorExt::TensorAccess::WriteOnly, resultType
        );
        auto subspanOp = IREE::HAL::InterfaceBindingSubspanOp::create(
            builder, loc, dispatchTensorType, layoutAttr, APInt(64, binding), nullptr, ValueRange{},
            builder.getIndexAttr(4)
        );
        subspanOps.push_back(subspanOp);
    }

    IRMapping map;

    // substitute the value assumeZero with a zero constant with the same type
    for (auto zeroValue : assumeZero) {
        if (zeroValue) {
            if (failed(createZeroConstant(zeroValue, zeroValue.getLoc(), builder, map))) {
                return failure();
            }
        }
    }

    // Clone each operation once while preserving ordering from the op sets.
    DenseSet<Operation *> clonedOps;
    for (auto ops : opsSets) {
        for (auto op : ops) {
            if (clonedOps.insert(op).second) {
                builder.clone(*op, map);
            }
        }
    }

    // Store each requested result to its dedicated output binding.
    for (auto [binding, value] : llvm::enumerate(values)) {
        IREE::TensorExt::DispatchTensorStoreOp::create(
            builder, loc, map.lookup(value), subspanOps[binding], ValueRange{}
        );
    }

    // return nothing (this is the calling convention for IREE dispatches)
    func::ReturnOp::create(builder, loc, ValueRange{});

    return variantOp;
}

static void setupPipeline(PassManager &pm) {

    if (!clDumpComputeConstantsIR.empty()) {
        llvm::sys::fs::create_directories(clDumpComputeConstantsIR);
        pm.enableIRPrintingToFileTree(
            [](Pass *, Operation *) { return false; }, [](Pass *, Operation *) { return true; },
            false, false, false, clDumpComputeConstantsIR
        );
    }

    OpPassManager &executableOpPm = pm.nest<IREE::HAL::ExecutableOp>();

    OpPassManager &variantPassManager = executableOpPm.nest<IREE::HAL::ExecutableVariantOp>();

    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();

    {
        FunctionLikeNest functionPassManager(modulePassManager);
        addCommonTargetExecutablePreprocessingPasses(functionPassManager);
    }

    // these passes are taken from addLowerToLLVMPasses in
    // third_party/iree/compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp

    FunctionLikeNest(modulePassManager).addPass(createEliminateEmptyTensorsPass);
    FunctionLikeNest(modulePassManager).addPass(bufferization::createEmptyTensorToAllocTensorPass);

    FunctionLikeNest(modulePassManager).addPass(createLLVMCPULowerExecutableTargetPass);

    // Lower any remaining TorqHL ops (e.g. transpose) to Linalg so the JIT
    // pipeline can process them. This is needed because constant weight packs
    // may depend on TorqHL ops created by earlier conversion passes.
    FunctionLikeNest(modulePassManager).addPass(createTorqHLToLinalgConversionPass);

    FunctionLikeNest(modulePassManager).addPass(createEraseHALDescriptorTypeFromMemRefPass);
    // FIXME: we should limit the passes below to the minimum required to pre-compute
    // constants

    FunctionLikeNest(modulePassManager)
        // LinalgExt -> SCF
        .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)
        // Linalg -> SCF
        .addPass(createMemrefCopyToLinalgPass)
        .addPass(createConvertLinalgToLoopsPass)
        .addPass(createConvertBf16ArithToF32Pass)
        .addPass(createConvertBf16ToUInt16BuffersPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    // Handled tensor-type constants.
    modulePassManager.addPass(createIREEBufferizeConstantsPass());

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        .addPass(createMathTransformPass)
        .addPass(createHoistStaticallyBoundAllocationsPass);

    FunctionLikeNest(modulePassManager)
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass)
        // SCF -> CF
        .addPass(createSCFToControlFlowPass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass)
        // (HAL, IREE, Linalg, CF) -> LLVM
        .addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(affine::createAffineExpandIndexOpsPass)
        .addPass([&]() {
            arith::ArithExpandOpsPassOptions options;
            options.includeBf16 = true;
            options.includeF4E2M1 = true;
            options.includeF8E8M0 = true;
            return arith::createArithExpandOpsPass(options);
        })
        //.addPass(memref::createExpandOpsPass)
        //.addPass(memref::createFoldMemRefAliasOpsPass)
        .addPass(createEmulateNarrowTypePass)
        .addPass(createCanonicalizerPass)
        .addPass(createCSEPass);

    modulePassManager.addPass(createConvertToLLVMPass(false));
    modulePassManager.addPass(createReconcileUnrealizedCastsPass());

    // We rely on MLIR symbol visibility being correct after this point and need
    // to mirror the LLVM linkage that was assigned during conversion.
    modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(createAddFastMathFlagsPass());
}

static void insertDebugTrap(IREE::HAL::ExecutableVariantOp evOp) {

    // import llvm.debugtrap() in the module
    OpBuilder builder(evOp);
    builder.setInsertionPointToStart(evOp.getInnerModule().getBody());

    auto llvmVoidType = mlir::LLVM::LLVMVoidType::get(evOp.getContext());
    auto llvmFuncType = mlir::LLVM::LLVMFunctionType::get(llvmVoidType, {}, false);
    auto debugTrapFunc =
        mlir::LLVM::LLVMFuncOp::create(builder, evOp.getLoc(), "llvm.debugtrap", llvmFuncType);
    debugTrapFunc.setLinkage(LLVM::Linkage::External);

    // add a debug trap at the start of the function to make it easier to set a breakpoint on the
    // generated code
    evOp.walk([&](LLVM::LLVMFuncOp funcOp) {
        if (funcOp.getName() != "main") {
            return WalkResult::advance();
        }

        OpBuilder builder(funcOp);
        builder.setInsertionPointToStart(&funcOp.getBody().front());
        LLVM::CallOp::create(builder, funcOp.getLoc(), debugTrapFunc, ValueRange{});
        return WalkResult::interrupt();
    });
}

// By default llvm lowers allocations to llvm.alloca() which allocate memory on the stack.
// This can be an issue if we allocate large buffers because we can overflow the stack.
// To avoid this we replace alloca with calls to malloc() and free() so that the memory
// is allocated on the heap.
static void replaceStackAllocationsWithMalloc(IREE::HAL::ExecutableVariantOp evOp) {
    auto loc = evOp.getLoc();
    // import malloc() in the module
    OpBuilder builder(evOp);
    builder.setInsertionPointToStart(evOp.getInnerModule().getBody());
    auto llvmI64Type = builder.getIntegerType(64);
    auto llvmPtrType = LLVM::LLVMPointerType::get(evOp.getContext());
    auto funcType = LLVM::LLVMFunctionType::get(llvmPtrType, {llvmI64Type}, false);
    LLVM::LLVMFuncOp mallocFunc = LLVM::LLVMFuncOp::create(builder, loc, "malloc", funcType);
    mallocFunc.setLinkage(LLVM::Linkage::External);

    // import free() in the module
    auto llvmVoidType = mlir::LLVM::LLVMVoidType::get(evOp.getContext());
    funcType = LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);
    auto freeFunc = LLVM::LLVMFuncOp::create(builder, loc, "free", funcType);
    freeFunc.setLinkage(LLVM::Linkage::External);

    // Replace stack alloca with call to malloc
    SmallVector<Value> allocations;
    evOp.walk([&](LLVM::AllocaOp allocaOp) {
        OpBuilder builder(allocaOp);
        builder.setInsertionPointAfter(allocaOp);

        auto llvmI64Type = builder.getIntegerType(64);

        // compute the size of the allocation
        auto elementSize = allocaOp.getElemType().getIntOrFloatBitWidth() / 8;
        assert(elementSize > 0);

        Value sizeValue = LLVM::ConstantOp::create(
            builder, allocaOp.getLoc(), llvmI64Type,
            builder.getIntegerAttr(llvmI64Type, elementSize)
        );

        if (allocaOp.getArraySize()) {

            Value arraySize = allocaOp.getArraySize();

            // convert the array size to i64 if it's not already
            if (arraySize.getType().getIntOrFloatBitWidth() < 64) {
                arraySize =
                    LLVM::ZExtOp::create(builder, allocaOp.getLoc(), llvmI64Type, arraySize);
            }

            // multiply by the array size
            sizeValue = LLVM::MulOp::create(builder, allocaOp.getLoc(), sizeValue, arraySize);
        }

        // create the malloc call
        auto mallocCall =
            LLVM::CallOp::create(builder, allocaOp.getLoc(), mallocFunc, ValueRange{sizeValue});

        // replace all uses of the alloca with the result of the cast
        allocaOp.replaceAllUsesWith(mallocCall.getResult());
        // erase the alloca
        allocaOp.erase();
        // Keep track of all allocations so that we can free them later
        allocations.push_back(mallocCall.getResult());
    });

    // Add calls to free() at the end of the function for each allocation
    evOp.walk([&](LLVM::LLVMFuncOp funcOp) {
        funcOp.walk([&](LLVM::ReturnOp returnOp) {
            OpBuilder builder(returnOp);
            for (auto alloc : allocations) {
                LLVM::CallOp::create(builder, returnOp.getLoc(), freeFunc, ValueRange{alloc});
            }
        });
    });

    if (failed(verify(evOp.getInnerModule()))) {
        llvm::report_fatal_error(
            "Failed to verify module after replacing stack allocations with malloc"
        );
    }
}

static LogicalResult passManagerSetup(
    ModuleOp moduleOp, MLIRContext *context, IREE::HAL::ExecutableVariantOp variantOp
) {
    auto pm = PassManager(context);

    // create a pipeline that compiles from tensor to LLVM
    setupPipeline(pm);

    // try to run the pipeline on the IR we just created to lower the module to llvm IR
    if (failed(pm.run(moduleOp))) {
        LLVM_DEBUG({ llvm::dbgs() << "Failed to compile module\n"; });
        return failure();
    }

    // replace alloca with malloc/free to avoid stack overflow on large allocations
    replaceStackAllocationsWithMalloc(variantOp);

    if (clInsertDebugTrapInComputeConstants) {

        // insert a debug trap in the generated code that will interrupt the compiler when
        // we try to execute the generated code
        insertDebugTrap(variantOp);

        auto llvmModuleOp = variantOp.getInnerModule();

        // print out the IR and reset the location of operations to this IR dump
        OpPrintingFlags printFlags;
        if (failed(generateLocationsFromIR("/tmp/constants.mlir", llvmModuleOp, printFlags))) {
            return failure();
        }

        // add the IR necessary to emit debug infos
        PassManager debugPm(context);
        debugPm.addPass(LLVM::createDIScopeForLLVMFuncOpPass());
        if (failed(debugPm.run(llvmModuleOp))) {
            return failure();
        }
    }

    return success();
}

static FailureOr<std::unique_ptr<mlir::ExecutionEngine>>
executionEngineSetup(IREE::HAL::ExecutableVariantOp variantOp) {
    // FIXME:
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // create a optimization pipeline for llvm
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr
    );

    // compile the module using the JIT engine
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;

    if (clInsertDebugTrapInComputeConstants) {
        engineOptions.enableGDBNotificationListener = true;
    }

    auto maybeEngine = mlir::ExecutionEngine::create(variantOp.getInnerModule(), engineOptions);
    if (!maybeEngine) {
        return failure();
    }

    LLVM_DEBUG({ llvm::dbgs() << "Compilation succeeded\n"; });

    return std::move(maybeEngine.get());
}

static FailureOr<SmallVector<DenseElementsAttr>>
invokeExecution(mlir::ExecutionEngine &engine, ArrayRef<RankedTensorType> outputTypes) {
    if (outputTypes.empty()) {
        LLVM_DEBUG({ llvm::dbgs() << "Invalid output bindings passed to invokeExecution\n"; });
        return failure();
    }

    // create the parameters for the function call
    iree_hal_executable_environment_v0_t environment;
    iree_hal_executable_dispatch_state_v0_t dispatch_state;
    iree_hal_executable_workgroup_state_v0_t workgroup_state;

    memset(&environment, 0, sizeof(environment));
    memset(&dispatch_state, 0, sizeof(dispatch_state));
    memset(&workgroup_state, 0, sizeof(workgroup_state));

    SmallVector<SmallVector<char, 0>, 4> outputData;
    outputData.reserve(outputTypes.size());

    SmallVector<void *, 4> bindingsAddr;
    bindingsAddr.reserve(outputTypes.size());

    for (auto outputType : outputTypes) {
        auto bufferSize = outputType.getNumElements() * outputType.getElementTypeBitWidth() / 8;
        outputData.emplace_back(bufferSize, 0);
        bindingsAddr.push_back(outputData.back().data());
    }

    dispatch_state.binding_count = outputTypes.size();
    dispatch_state.binding_ptrs = bindingsAddr.data();

    // look for the main function in the module
    auto main = engine.lookup("main");

    if (!main) {
        llvm::errs() << "Failed to find 'main' function in the module\n";
        return failure();
    }

    // call the function we compiled
    using MainFnType =
        void (*)(iree_hal_executable_environment_v0_t *, iree_hal_executable_dispatch_state_v0_t *, iree_hal_executable_workgroup_state_v0_t *);

    auto mainFn = reinterpret_cast<MainFnType>(main.get());

    LLVM_DEBUG({ llvm::dbgs() << "Executing compiled function\n"; });

    mainFn(&environment, &dispatch_state, &workgroup_state);

    SmallVector<DenseElementsAttr> outputs;
    outputs.reserve(outputTypes.size());
    for (auto [idx, outputType] : llvm::enumerate(outputTypes)) {
        outputs.push_back(DenseElementsAttr::getFromRawBuffer(outputType, outputData[idx]));
    }

    return outputs;
}

FailureOr<SmallVector<DenseElementsAttr>> computeValueFromOps(
    Location loc, ArrayRef<Value> values, ArrayRef<SmallVector<Operation *>> opsSets,
    llvm::ArrayRef<Value> assumeZero
) {
    if (values.empty() || values.size() != opsSets.size()) {
        return failure();
    }

    SmallVector<RankedTensorType, 4> outputBindings;
    outputBindings.reserve(values.size());
    for (auto value : values) {
        outputBindings.push_back(cast<RankedTensorType>(value.getType()));
    }

    // create a new free-standing module containing the operations we were asked to compute
    ModuleOp moduleOp = ModuleOp::create(loc);

    auto maybeVariantOp =
        createModule(moduleOp, moduleOp.getContext(), loc, values, opsSets, assumeZero);

    if (failed(maybeVariantOp)) {
        return failure();
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Module that will be compiled:\n";
        moduleOp.dump();
    });

    if (failed(passManagerSetup(moduleOp, moduleOp.getContext(), *maybeVariantOp))) {
        return failure();
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Module that will be JIT-ed:\n";
        moduleOp.dump();
    });

    auto maybeEngine = executionEngineSetup(*maybeVariantOp);
    if (failed(maybeEngine)) {
        return failure();
    }

    return invokeExecution(**maybeEngine, outputBindings);
}

SmallVector<Operation *>
outlineAndReturnOps(Value value, bool recursive, llvm::ArrayRef<Value> assumeZero) {

    LLVM_DEBUG({
        llvm::dbgs() << "Trying to compute value of:\n";
        value.dump();
    });

    auto outputType = dyn_cast<RankedTensorType>(value.getType());

    // we don't support computing constants for non-ranked tensor types
    if (!outputType) {
        return {};
    }

    // we cannot compute the constant value of an argument
    if (!value.getDefiningOp()) {
        return {};
    }

    // set of all ops we already visited (including operations within operations)
    DenseSet<Operation *> visitedOps;

    // set of ops that we need to copy to perform the computation
    // (these are only sibilings of operation defining the value we want to compute)
    DenseSet<Operation *> opsSet;

    // ops that we still need to analyze to find dependencies
    SmallVector<Operation *> toAnalyze;

    // for each function we want to process we will walk it and find out all the
    // operations that are needed to compute it, we start the process using the root
    // operation we were asked to compute

    // TODO: this should be done with a backward slice analyis, but right now we have
    // special rules when recursive is set to false, and assumeZero is non empty,
    // we should probably get rid of that.

    toAnalyze.push_back(value.getDefiningOp());

    while (!toAnalyze.empty()) {
        auto currentOp = toAnalyze.front();
        toAnalyze.erase(toAnalyze.begin());

        // skip this operation if we already visited it
        if (visitedOps.contains(currentOp)) {
            continue;
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Processing operation: ";
            currentOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
            llvm::dbgs() << "\n";
        });

        opsSet.insert(currentOp);

        // walk the operation and all the nested operations to find all the operands
        // that we need to compute this value
        auto ret = currentOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
            LLVM_DEBUG({
                llvm::dbgs() << "Visiting operation: ";
                op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
                llvm::dbgs() << "\n";
            });

            // add the operation to the set of visited ops ( doing it here
            // will help to detect BlockArguments that we can allow)
            visitedOps.insert(op);

            for (auto operand : op->getOperands()) {

                LLVM_DEBUG({
                    llvm::dbgs() << "Processing operand: ";
                    operand.print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
                    llvm::dbgs() << "\n";
                });

                // if the operand is the value that we assume to be zero, we don't
                // need to process it
                if (llvm::is_contained(assumeZero, operand)) {
                    LLVM_DEBUG({ llvm::dbgs() << "Operand is assumeZero, skipping\n"; });
                    continue;
                }

                auto blockArgument = dyn_cast<BlockArgument>(operand);
                auto operandOp = operand.getDefiningOp();

                if (blockArgument) {

                    auto parentOp = blockArgument.getOwner()->getParentOp();

                    if (opsSet.contains(parentOp)) {
                        LLVM_DEBUG({
                            llvm::dbgs()
                                << "If in opset the parent operation is already outlined\n";
                        });
                        continue;
                    }
                    operandOp = parentOp;
                }

                // the value depends on the input, we cannot compute this
                if (isa<IREE::TensorExt::DispatchTensorLoadOp,
                        IREE::HAL::InterfaceBindingSubspanOp>(operandOp)) {
                    LLVM_DEBUG({
                        llvm::dbgs() << "Value depends on inputs, cannot compute statically\n";
                    });

                    return WalkResult::interrupt();
                }

                // the operation is both not an block argument and not an operation, we don't
                // support this
                if (!operandOp) {
                    return WalkResult::interrupt();
                }
                if (!recursive) {
                    auto constOp = dyn_cast<arith::ConstantOp>(operandOp);

                    if (!constOp) {
                        // We only support constant ops if we are not recursive
                        return WalkResult::interrupt();
                    }
                }

                if (operandOp->isProperAncestor(currentOp)) {
                    LLVM_DEBUG({ llvm::dbgs() << "Operand and Parent stuck in cycle\n"; });
                    return WalkResult::interrupt();
                }

                if (currentOp->isProperAncestor(operandOp)) {
                    LLVM_DEBUG({ llvm::dbgs() << "Operand is an Op inside the Current Op\n"; });
                    continue;
                }

                LLVM_DEBUG({ llvm::dbgs() << "Adding to list to analyze\n"; });

                // schedules the operation for analysis
                toAnalyze.push_back(operandOp);
            }

            return WalkResult::advance();
        });

        if (ret == WalkResult::interrupt()) {
            LLVM_DEBUG({ llvm::dbgs() << "Cannot build constant IR to compute value\n"; });
            return {};
        }
    }

    // create an ordered set of ops so that the use-def are ok
    SmallVector<Operation *> ops;

    Operation *topOp = value.getDefiningOp();
    while (topOp->getParentOp())
        topOp = topOp->getParentOp();

    topOp->walk([&](Operation *op) {
        if (opsSet.contains(op)) {
            ops.push_back(op);
        }
    });

    LLVM_DEBUG({
        llvm::dbgs() << "Ops to process:\n";
        for (auto op : ops) {
            op->dump();
        }
    });
    return ops;
}

FailureOr<SmallVector<Attribute>> computeAllConstAttr(
    SmallVectorImpl<Value> &values, bool recursive, llvm::ArrayRef<Value> assumeZero
) {
    if (values.empty()) {
        return SmallVector<Attribute>{};
    }

    SmallVector<SmallVector<Operation *>, 4> opsSets;
    opsSets.reserve(values.size());

    SmallVector<Value> constValues;
    constValues.reserve(values.size());
    for (auto value : values) {
        SmallVector<Operation *> ops = outlineAndReturnOps(value, recursive, assumeZero);
        if (ops.empty()) {
            LLVM_DEBUG({ llvm::errs() << "Failed to outline ops to compute value\n"; });
            continue;
        }
        constValues.push_back(value);
        opsSets.push_back(std::move(ops));
    }
    if (constValues.empty()) {
        return failure();
    }

    auto output = computeValueFromOps(
        values.front().getDefiningOp()->getLoc(), constValues, opsSets, assumeZero
    );

    if (failed(output) || output->size() != constValues.size()) {
        return failure();
    }

    SmallVector<Attribute> attrs;
    attrs.reserve(output->size());
    for (auto attr : *output) {
        attrs.push_back(attr);
    }
    values = constValues;
    return attrs;
}

FailureOr<Attribute>
computeConstAttr(Value value, bool recursive, llvm::ArrayRef<Value> assumeZero) {
    SmallVector<Value, 1> values = {value};
    auto output = computeAllConstAttr(values, recursive, assumeZero);
    if (failed(output) || output->empty()) {
        return failure();
    }

    auto denseAttr = dyn_cast<DenseElementsAttr>(output->front());
    if (!denseAttr) {
        return failure();
    }

    return denseAttr;
}

// Compute the constant value for the given LinalgOp.
FailureOr<SmallVector<Value>> computeAllArithConst(
    SmallVectorImpl<Value> &values, bool recursive, llvm::ArrayRef<Value> assumeZero
) {
    if (values.empty()) {
        return SmallVector<Value>{};
    }

    auto maybeAttrs = computeAllConstAttr(values, recursive, assumeZero);

    if (failed(maybeAttrs)) {
        return failure();
    }

    SmallVector<Value> constants;
    constants.reserve(values.size());
    for (auto [idx, value] : llvm::enumerate(values)) {
        auto denseAttr = dyn_cast<DenseElementsAttr>((*maybeAttrs)[idx]);
        if (!denseAttr) {
            return failure();
        }

        OpBuilder builder(value.getDefiningOp());
        constants.push_back(
            arith::ConstantOp::create(builder, value.getLoc(), denseAttr).getResult()
        );
    }

    LLVM_DEBUG({ llvm::dbgs() << "Constants successfully computed\n"; });

    return constants;
}

// Compute the constant value for the given LinalgOp.
FailureOr<Value> computeArithConst(Value value, bool recursive, llvm::ArrayRef<Value> assumeZero) {
    SmallVector<Value, 1> values = {value};
    auto maybeValues = computeAllArithConst(values, recursive, assumeZero);
    if (failed(maybeValues) || maybeValues->empty()) {
        return failure();
    }

    return maybeValues->front();
}

// Compute the constant value for the given LinalgOp.
FailureOr<SmallVector<Value>> computeAllArithConst(
    SmallVectorImpl<Operation *> &ops, bool recursive, llvm::ArrayRef<Value> assumeZero
) {
    if (ops.empty()) {
        return SmallVector<Value>{};
    }

    SmallVector<Value> values;
    values.reserve(ops.size());
    DenseMap<Value, Operation *> valueToOps;

    for (auto op : ops) {
        if (op->getNumResults() != 1) {
            // We only support ops with one output for now.
            return failure();
        }
        valueToOps[op->getResult(0)] = op;
        values.push_back(op->getResult(0));
    }

    auto allConsts = computeAllArithConst(values, recursive, assumeZero);
    SmallVector<Operation *> opsToProcess;
    opsToProcess.reserve(values.size());
    for (auto value : values) {
        if (valueToOps.contains(value)) {
            opsToProcess.push_back(valueToOps[value]);
        }
    }
    ops = opsToProcess;
    return allConsts;
}

// Compute the constant value for the given LinalgOp.
FailureOr<Value>
computeArithConst(LinalgOp linalgOp, bool recursive, llvm::ArrayRef<Value> assumeZero) {
    SmallVector<Operation *, 1> ops = {linalgOp.getOperation()};
    auto maybeValues = computeAllArithConst(ops, recursive, assumeZero);
    if (failed(maybeValues) || maybeValues->empty()) {
        return failure();
    }

    return maybeValues->front();
}

} // namespace mlir::syna::torq
