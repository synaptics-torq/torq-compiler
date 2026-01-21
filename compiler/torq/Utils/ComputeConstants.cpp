
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/TargetSelect.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"

#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
// #include "torq/Codegen/Passes.h"
#include "llvm/Support/Debug.h"

#include "iree/hal/local/executable_library.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <numeric>

#define DEBUG_TYPE "torq-compute-constants"

using namespace mlir::syna::torq_hl;
using namespace mlir::linalg;

namespace mlir::syna::torq {

llvm::cl::opt<std::string> clDumpComputeConstantsIR(
    "torq-dump-compute-constants-ir",
    llvm::cl::desc("Dump IR used to compute constants to a directory"), llvm::cl::init("")
);

using namespace mlir::iree_compiler;

static LogicalResult
createZeroConstant(Value value, Location loc, OpBuilder &builder, IRMapping &map) {

    auto zeroAttr = builder.getZeroAttr(value.getType());

    if (!zeroAttr) {
        return failure();
    }

    auto zeroValue = builder.create<arith::ConstantOp>(loc, value.getType(), zeroAttr);

    // map any reference to assumeZero to zeroValue
    map.map(value, zeroValue);

    LLVM_DEBUG({
        llvm::dbgs() << "Created zero constant: ";
        zeroValue.dump();
    });

    return success();
}

static FailureOr<IREE::HAL::ExecutableVariantOp> createModule(
    ModuleOp topModuleOp, MLIRContext *context, Location loc, Value value,
    ArrayRef<Operation *> ops, const std::vector<Value> &assumeZero
) {
    OpBuilder builder(topModuleOp);

    builder.setInsertionPointToStart(topModuleOp.getBody());

    auto executableOp = builder.create<IREE::HAL::ExecutableOp>(loc, "test");

    builder.setInsertionPointToStart(&executableOp.getBody().front());
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::get(
        context, builder.getStringAttr("llvm-native"), builder.getStringAttr("native")
    );
    auto variantOp = builder.create<IREE::HAL::ExecutableVariantOp>(loc, "native", targetAttr);
    builder.setInsertionPointToStart(&variantOp.getBody().front());

    auto moduleOp = builder.create<ModuleOp>(loc);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto funcOp = builder.create<func::FuncOp>(
        loc, "main", builder.getFunctionType(TypeRange{}, TypeRange{})
    );

    auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
        funcOp.getContext(),
        mlir::iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUDefault
    );

    succeeded(setTranslationInfo(funcOp, translationInfo));

    funcOp.addEntryBlock();

    builder.setInsertionPointToStart(&funcOp.getBody().front());

    // create a subspan that we will use to return the result
    RankedTensorType resultType = cast<RankedTensorType>(value.getType());
    auto dispatchTensorType =
        IREE::Flow::DispatchTensorType::get(IREE::Flow::TensorAccess::WriteOnly, resultType);
    auto subspanOp = builder.create<IREE::HAL::InterfaceBindingSubspanOp>(
        loc, dispatchTensorType, APInt(64, 0), APInt(64, 0),
        IREE::HAL::DescriptorType::StorageBuffer, nullptr, ValueRange{}, builder.getIndexAttr(4)
    );

    IRMapping map;

    // substitute the value assumeZero with a zero constant with the same type
    for (auto zeroValue : assumeZero) {
        if (zeroValue) {
            if (failed(createZeroConstant(zeroValue, zeroValue.getLoc(), builder, map))) {
                return failure();
            }
        }
    }

    // clone all the operations in the new function
    for (auto op : ops) {
        builder.clone(*op, map);
    }

    // store the value that corresponds to the original value in the binding subspan
    builder.create<IREE::Flow::DispatchTensorStoreOp>(
        loc, map.lookup(value), subspanOp, ValueRange{}
    );

    // return nothing (this is the calling convention for IREE dispatches)
    builder.create<func::ReturnOp>(loc, ValueRange{});

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
    addConstantBufferizePasses(modulePassManager);

    FunctionLikeNest(modulePassManager)
        .addPass(createFoldTensorExtractOpPass)
        // math dialect elementry functions -> polynomial form.
        .addPass(createPolynomialApproximationPass)
        .addPass(createHoistStaticallyBoundAllocationsPass);

    FunctionLikeNest(modulePassManager)
        // Resolve get_buffer_descriptor ops. All structural buffer manipulations
        // must conclude before this point.
        .addPass(createIREEExpandStridedMetadataPass)
        .addPass(createCleanupBufferAllocViewPass)
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

    modulePassManager.addPass(createConvertToLLVMPass(false));
    modulePassManager.addPass(createReconcileUnrealizedCastsPass());

    // We rely on MLIR symbol visibility being correct after this point and need
    // to mirror the LLVM linkage that was assigned during conversion.
    modulePassManager.addPass(createLLVMCPUSynchronizeSymbolVisibilityPass());

    modulePassManager.addPass(createCanonicalizerPass());
    modulePassManager.addPass(createCSEPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(createAddFastMathFlagsPass());
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
    LLVM::LLVMFuncOp mallocFunc = builder.create<LLVM::LLVMFuncOp>(loc, "malloc", funcType);
    mallocFunc.setLinkage(LLVM::Linkage::External);

    // import free() in the module
    auto llvmVoidType = mlir::LLVM::LLVMVoidType::get(evOp.getContext());
    funcType = LLVM::LLVMFunctionType::get(llvmVoidType, {llvmPtrType}, false);
    auto freeFunc = builder.create<LLVM::LLVMFuncOp>(loc, "free", funcType);
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

        Value sizeValue = builder.create<LLVM::ConstantOp>(
            allocaOp.getLoc(), llvmI64Type, builder.getIntegerAttr(llvmI64Type, elementSize)
        );

        if (allocaOp.getArraySize()) {
            // multiply by the array size
            sizeValue = builder.create<LLVM::MulOp>(
                allocaOp.getLoc(), sizeValue,
                builder.create<LLVM::ZExtOp>(
                    allocaOp.getLoc(), llvmI64Type, allocaOp.getArraySize()
                )
            );
        }

        // create the malloc call
        auto mallocCall =
            builder.create<LLVM::CallOp>(allocaOp.getLoc(), mallocFunc, ValueRange{sizeValue});

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
                builder.create<LLVM::CallOp>(returnOp.getLoc(), freeFunc, ValueRange{alloc});
            }
        });
    });
}

FailureOr<DenseIntOrFPElementsAttr> computeValueFromOps(
    Location loc, Value value, ArrayRef<Operation *> ops, const std::vector<Value> &assumeZero
) {

    RankedTensorType outputType = cast<RankedTensorType>(value.getType());

    // create a new free-standing module containing the operations we were asked to compute
    ModuleOp moduleOp = ModuleOp::create(loc);

    auto maybeVariantOp =
        createModule(moduleOp, moduleOp.getContext(), loc, value, ops, assumeZero);

    if (failed(maybeVariantOp)) {
        return failure();
    }

    LLVM_DEBUG({
        llvm::dbgs() << "Module that will be compiled:\n";
        moduleOp.dump();
    });

    auto pm = PassManager(value.getContext());

    // create a pipeline that compiles from tensor to LLVM
    setupPipeline(pm);

    // try to run the pipeline on the IR we just created to lower the module to llvm IR
    if (failed(pm.run(moduleOp))) {
        LLVM_DEBUG({ llvm::dbgs() << "Failed to compile module\n"; });

        return failure();
    }

    // replace alloca with malloc/free to avoid stack overflow on large allocations
    replaceStackAllocationsWithMalloc(*maybeVariantOp);

    LLVM_DEBUG({
        llvm::dbgs() << "Module that will be JIT-ed:\n";
        moduleOp.dump();
    });

    // FIXME:
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // create a optimization pipeline for llvm
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/3, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr
    );

    // compile the module using the JIT engine
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine =
        mlir::ExecutionEngine::create(maybeVariantOp->getInnerModule(), engineOptions);
    if (!maybeEngine) {
        return failure();
    }

    auto &engine = maybeEngine.get();

    LLVM_DEBUG({ llvm::dbgs() << "Compilation succeeded\n"; });

    // create the parameters for the function call
    iree_hal_executable_environment_v0_t environment;
    iree_hal_executable_dispatch_state_v0_t dispatch_state;
    iree_hal_executable_workgroup_state_v0_t workgroup_state;

    memset(&environment, 0, sizeof(environment));
    memset(&dispatch_state, 0, sizeof(dispatch_state));
    memset(&workgroup_state, 0, sizeof(workgroup_state));

    auto bufferSize = outputType.getNumElements() * outputType.getElementTypeBitWidth() / 8;

    SmallVector<char> outputData(bufferSize, 0);

    void *bindingsAddr[1];
    bindingsAddr[0] = outputData.data();

    dispatch_state.binding_ptrs = bindingsAddr;

    // look for the main function in the module
    auto main = engine->lookup("main");

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

    // create an attribute from the output data
    DenseIntOrFPElementsAttr output =
        cast<DenseIntOrFPElementsAttr>(DenseElementsAttr::getFromRawBuffer(outputType, outputData));

    LLVM_DEBUG({ llvm::dbgs() << "Constant successfully computed"; });

    return output;
}

FailureOr<DenseIntOrFPElementsAttr>
computeValue(Value value, bool recursive, const std::vector<mlir::Value> &assumeZero) {

    LLVM_DEBUG({
        llvm::dbgs() << "Trying to compute value of:\n";
        value.dump();
    });

    auto outputType = dyn_cast<RankedTensorType>(value.getType());

    // we don't support computing constants for non-ranked tensor types
    if (!outputType) {
        return failure();
    }

    // we cannot compute the constant value of an argument
    if (!value.getDefiningOp()) {
        return failure();
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
            currentOp->dump();
        });

        opsSet.insert(currentOp);

        // walk the operation and all the nested operations to find all the operands
        // that we need to compute this value
        auto ret = currentOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
            LLVM_DEBUG({
                llvm::dbgs() << "Visiting operation: ";
                op->dump();
            });

            // add the operation to the set of visited ops ( doing it here
            // will help to detect BlockArguments that we can allow)
            visitedOps.insert(op);

            for (auto operand : op->getOperands()) {

                LLVM_DEBUG({
                    llvm::dbgs() << "Processing operand: ";
                    operand.dump();
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
                if (isa<IREE::Flow::DispatchTensorLoadOp, IREE::HAL::InterfaceBindingSubspanOp>(
                        operandOp
                    )) {
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
            return failure();
        }
    }

    // create an ordered set of ops so that the use-def are ok
    SmallVector<Operation *> ops;

    value.getDefiningOp()->getParentOp()->walk([&](Operation *op) {
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

    auto output = computeValueFromOps(value.getDefiningOp()->getLoc(), value, ops, assumeZero);

    if (failed(output)) {
        LLVM_DEBUG({ llvm::errs() << "Failed to compute value\n"; });
        // Failed to compute the constant value
        assert(false);
        return failure();
    }

    LLVM_DEBUG({ llvm::dbgs() << "Value successfully computed\n"; });

    return *output;
}

} // namespace mlir::syna::torq
