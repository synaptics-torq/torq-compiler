// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <mlir/Analysis/SliceAnalysis.h>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "torq-tile-and-fuse"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

enum class TileAndFuseProducersFuseMode { MaxSize, MaxProducers, OnlyPatterns, NoFuse };

static llvm::cl::opt<TileAndFuseProducersFuseMode> clTorqTileAndFuseProducersFuseMode(
    "torq-tile-and-fuse-producers-fuse-mode",
    llvm::cl::desc("Selects one of three predefined values"),
    llvm::cl::values(
        clEnumValN(
            TileAndFuseProducersFuseMode::MaxSize, "max-size",
            "Prefer bigger tile over fusing more producers: tile size is set to make the consumer "
            "fit in memory; producers are fused only if they fit in the same tile size)"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::MaxProducers, "max-producers",
            "Prefer more producers over tile size: tile size is the biggest size that can still "
            "fit all the producers"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::OnlyPatterns, "only-patterns",
            "Fuse producers only whene required to preserve patterns"
        ),
        clEnumValN(TileAndFuseProducersFuseMode::NoFuse, "no-fuse", "Do not fuse producers")
    ),
    llvm::cl::init(TileAndFuseProducersFuseMode::MaxSize) // Default value
);

torq_hl::Executor getExecutor(Operation *op) {
    return isMarkedFuseGroup(op) ? torq_hl::Executor::Slice : torq_hl::Executor::CSS;
}

// Replace the untiled `op` with its tiled version from `tiledResults.
void replaceTiledOp(
    IRRewriter &rewriter, Operation *op, const scf::SCFTileAndFuseResult &tiledResults
) {
    for (OpResult res : op->getResults()) {
        if (Value replacement = tiledResults.replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, replacement);
        }
    }
}

// Replace the untiled consumer `op` with its tiled version from `tiledResults`,
// and do the same for any producer that was yielded as well.
void applyTiledResults(
    IRRewriter &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
) {
    // Replace the root with its tiled result
    replaceTiledOp(rewriter, op, tiledResults);

    // In general, fused producers can declare that they want to be yielded.
    // Here we replace the untiled producers with the yielded result.
    for (Operation *prodOp : tiledResults.fusedProducers) {
        replaceTiledOp(rewriter, prodOp, tiledResults);
    }
}

// Return the iteration domains of `tilingInterfaceOp` that can be tiled, in the order which we
// should tile them.
// NB: The order of insertion to SmallSetVector is important. That is the order
// in which we will tile the domains. First one we insert, is the first we tile.
llvm::SmallSetVector<int64_t, 4> getTilingIterDomainsOrder(TilingInterface tilingInterfaceOp) {
    llvm::SmallSetVector<int64_t, 4> tilingIterDomsOrder;

    auto loopIteratorTypes = tilingInterfaceOp.getLoopIteratorTypes();

    // Handler that uses all the parallel domains.
    auto allParallelDomains = [&]() {
        for (size_t domain = 0; domain < loopIteratorTypes.size(); ++domain) {
            if (loopIteratorTypes[domain] == utils::IteratorType::parallel)
                tilingIterDomsOrder.insert(domain);
        }
    };

    // Handler for nhwc operations: we tile C, followed by H.
    auto nhwcOpHandler = [&]() {
        assert(
            loopIteratorTypes[3] == utils::IteratorType::parallel &&
            "expected domain 3 to be parallel"
        );
        tilingIterDomsOrder.insert(3); // C

        assert(
            loopIteratorTypes[1] == utils::IteratorType::parallel &&
            "expected domain 1 to be parallel"
        );
        tilingIterDomsOrder.insert(1); // H
    };

    // Decide which domains to tile, and in what order, based on the executor and the operation.
    switch (getExecutor(tilingInterfaceOp)) {
    case torq_hl::Executor::Slice: {
        Operation *principalOp = getFuseGroupPrincipalOpBackward(tilingInterfaceOp);
        assert(principalOp != nullptr && "could not find the principal op of the fuse group");

        TypeSwitch<Operation *>(principalOp)
            .Case<
                linalg::Conv2DNhwcHwcfOp, // NB: 4th iteration domain is actually F
                                          // (filters/output-channels) in this case
                linalg::DepthwiseConv2DNhwcHwcOp, linalg::PoolingNhwcMaxOp,
                linalg::PoolingNhwcMaxUnsignedOp, linalg::PoolingNhwcMinOp,
                linalg::PoolingNhwcMinUnsignedOp, linalg::PoolingNhwcSumOp>([&](auto) {
                nhwcOpHandler();
            })
            .Default([&](auto) { allParallelDomains(); });
        break;
    }
    case torq_hl::Executor::CSS:
        allParallelDomains();
        break;

    default:
        llvm_unreachable("expected Slice or CSS executor");
    }

    return tilingIterDomsOrder;
}

// Compute the result of `op` fully composed with preceding AffineApplyOps, and
// everything else set to 0.
// `op` must be one of AffineMinOp/AffineMaxOp.
llvm::FailureOr<int64_t> computeMinMaxAtZero(Operation *op) {
    bool isMin;
    AffineMap map;
    SmallVector<Value> operands;
    if (auto minOp = dyn_cast<affine::AffineMinOp>(op)) {
        isMin = true;
        map = minOp.getMap();
        operands = minOp.getOperands();
    }
    else {
        auto maxOp = cast<affine::AffineMaxOp>(op);
        isMin = false;
        map = maxOp.getMap();
        operands = maxOp.getOperands();
    }

    affine::fullyComposeAffineMapAndOperands(&map, &operands);
    SmallVector<Attribute> zeros(
        map.getNumDims(), getAsIndexOpFoldResult(op->getContext(), 0).template get<Attribute>()
    );
    SmallVector<Attribute> results;
    if (failed(map.constantFold(zeros, results)))
        return llvm::failure();

    Attribute *result;
    if (isMin) {
        result = llvm::min_element(results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    else {
        result = llvm::max_element(results, [](mlir::Attribute a, mlir::Attribute b) {
            return cast<mlir::IntegerAttr>(a).getInt() < cast<mlir::IntegerAttr>(b).getInt();
        });
    }
    if (!result)
        return llvm::failure();

    return cast<mlir::IntegerAttr>(*result).getInt();
}

// Compute the result of `op` fully composed with preceding AffineApplyOps, and
// everything else set to 0.
// `op` must be one of AffineMinOp/AffineMaxOp.
llvm::FailureOr<int64_t> computeIntAtZero(OpFoldResult sizeFoldResult) {
    if (std::optional<int64_t> constSize = getConstantIntValue(sizeFoldResult))
        return *constSize;

    assert(sizeFoldResult.is<Value>() && "expected a Value");

    Value sizeValue = sizeFoldResult.get<Value>();

    return llvm::TypeSwitch<Operation *, FailureOr<int64_t>>(sizeValue.getDefiningOp())
        .Case<affine::AffineMinOp, affine::AffineMaxOp>([&](auto minMaxOp) {
            // Evaluate the size at 0,0,...
            // This is a bit of a hack, what we really want is a tight upper bound
            // of size.
            return computeMinMaxAtZero(minMaxOp);
        })
        .Case<affine::AffineApplyOp>([&](auto applyOp) -> FailureOr<int64_t> {
            // Evaluate the size at 0,0,...
            // This is a bit of a hack, what we really want is a tight upper bound
            // of size.
            AffineMap map = applyOp.getMap();
            SmallVector<Value> operands = applyOp.getOperands();

            affine::fullyComposeAffineMapAndOperands(&map, &operands);
            SmallVector<Attribute> zeros(
                map.getNumDims(),
                getAsIndexOpFoldResult(applyOp->getContext(), 0).template get<Attribute>()
            );
            SmallVector<Attribute> results;
            if (failed(map.constantFold(zeros, results)))
                return llvm::failure();

            assert(results.size() == 1 && "AffineApplyOp should have exactly one reault");

            return cast<mlir::IntegerAttr>(results[0]).getInt();
        })
        .Default([](Operation *op) {
            op->emitWarning("can't compute producers tile size (unexpected operation type; "
                            "expected affine.min/max/apply)");
            LLVM_DEBUG(assert(false && "unexpected operation"));
            return llvm::failure();
        });
}

// Add to `ops` all the ops that need to be cloned to support `op` and
// `producerOps` (including those ops); and add to `inputs` all the values that
// are needed to drive `ops`, and are not in `ops`. An op is added to `ops` if
// it is accessed by an op in `ops`, and is from the same pattern-fuse-group, or
// it's not a TilingInterface.
void collectOpsForMemoryCheck(
    Operation *op, const SetVector<Operation *> &producerOps, SmallVector<Operation *> &ops,
    SmallVector<Value> &inputs
) {
    SmallVector<Operation *> queue = {op};
    queue.reserve(queue.size() + producerOps.size());
    llvm::append_range(queue, producerOps);

    DenseSet<Operation *> visitedOps(producerOps.size());
    llvm::set_union(visitedOps, queue);

    SmallVector<Value> maybeInputs;

    while (!queue.empty()) {
        Operation *currentOp = queue.pop_back_val();

        // TODO(sflur): use visitUsedValuesDefinedAbove instead (from
        // mlir/include/mlir/Transforms/RegionUtils.h)?
        currentOp->walk([&](Operation *walkOp) {
            for (Value operand : walkOp->getOperands()) {
                if (operand.getParentBlock() != currentOp->getBlock())
                    continue;

                // BlockArguments are definitely inputs
                if (isa<BlockArgument>(operand)) {
                    inputs.push_back(operand);
                    continue;
                }

                // Value is either a BlockArgument or the result of an operation
                Operation *operandOp = operand.getDefiningOp();
                assert(operandOp);

                if (visitedOps.contains(operandOp))
                    continue;

                if (!llvm::isa<TilingInterface>(operandOp) ||
                    checkShareFuseGroup(currentOp, operandOp)) {
                    visitedOps.insert(operandOp);
                    queue.push_back(operandOp);
                    continue;
                }

                // the defining op of operand might still be in `ops`, so this
                // is a maybeInput. We will check again at the end.
                maybeInputs.push_back(operand);
            }
        });
    }

    // Now that we know all the visited ops, add the real inputs from
    // `maybeInputs` to `inputs`.
    for (Value input : maybeInputs) {
        if (!visitedOps.contains(input.getDefiningOp()))
            inputs.push_back(input);
    }

    ops.reserve(ops.size() + visitedOps.size());
    llvm::append_range(ops, visitedOps);
}

// Construct a module with a single function, that includes tilingInterfaceOp
// and producerOps, and any other producers needed to keep pattern-fuse-groups
// intact. In addition, other ops that are not TilingInterface, that drive the
// included ops are included. The function arguments are, in order, the
// sizes of the domains that can be tiled as in tilingIterDomsOrder, and values
// that drive the included ops.
// The tile size arguments are left unused here as we don't do the tiling yet.
// This allows the function to be used when we initially check if an untiled op
// requires tiling at all.
ModuleOp extractOpsForMemoryCheck(
    Operation *consumerOp, ArrayRef<int64_t> tilingIterDomsOrder = {},
    const SetVector<Operation *> &producerOps = {}
) {
    MLIRContext *context = consumerOp->getContext();
    Location loc = consumerOp->getLoc();

    OpBuilder builder(context);

    ModuleOp moduleOp = builder.create<ModuleOp>(loc, "extracted_for_memory_check");
    builder.setInsertionPointToStart(moduleOp.getBody());

    SmallVector<Operation *> ops;
    SmallVector<Value> inputs;
    collectOpsForMemoryCheck(consumerOp, producerOps, ops, inputs);

    SmallVector<Type> inputTypes;
    inputTypes.reserve(tilingIterDomsOrder.size() + inputs.size());

    for (size_t i = 0; i < tilingIterDomsOrder.size(); ++i)
        inputTypes.push_back(builder.getIndexType());

    for (Value input : inputs)
        inputTypes.push_back(input.getType());

    FunctionType functionType = builder.getFunctionType(inputTypes, consumerOp->getResultTypes());

    func::FuncOp funcOp = builder.create<func::FuncOp>(loc, "extracted_tile", functionType);
    funcOp.addEntryBlock();

    IRMapping extractionMap;

    // Map the inputs to the function args. This will result in the function
    // args driving the appropriate cloned ops, when we clone them.
    auto nonDomainArgs = llvm::make_range(
        funcOp.getFunctionBody().getArguments().begin() + tilingIterDomsOrder.size(),
        funcOp.getFunctionBody().getArguments().end()
    );
    for (auto [input, arg] : llvm::zip_equal(inputs, nonDomainArgs))
        extractionMap.map(input, arg);

    // We have to clone `ops` in the order they appear in their parent block.
    llvm::sort(ops, [](auto lhsOp, auto rhsOp) {
        assert(lhsOp->getBlock() == rhsOp->getBlock() && "ops are not from the same block");
        return lhsOp->isBeforeInBlock(rhsOp);
    });

    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());

    for (Operation *op : ops)
        builder.clone(*op, extractionMap);

    Operation *clonedConsumerOp = extractionMap.lookupOrNull(consumerOp);
    assert(clonedConsumerOp);

    // NB: tileModuleForMemoryCheck relies on this being the last op in the
    // block, and clonedConsumerOp being it's immediate source.
    builder.create<func::ReturnOp>(loc, clonedConsumerOp->getResults());

    return moduleOp;
}

// "symbolically" tile and fuse everything in the only function in `moduleOp`.
// Tile sizes are non-constant (hence "symbolically"), and are the first
// arguments of the only function in the module.
llvm::LogicalResult tileModuleForMemoryCheck(
    ModuleOp moduleOp, ArrayRef<int64_t> tilingIterDomsOrder, size_t tileSizesCount
) {
    OpBuilder builder(moduleOp->getContext());

    auto funcOp = cast<func::FuncOp>(&moduleOp.getBody()->front());
    auto clonedConsumerOp = cast<func::ReturnOp>(&funcOp.getFunctionBody().front().back())
                                .getOperand(0)
                                .getDefiningOp();

    // Initially set all tile sizes to 0 (don't tile).
    SmallVector<OpFoldResult> tileSizes(tileSizesCount, builder.getIndexAttr(0));
    // Drive the sizes that we can tile by the appropriate function args.
    auto tileSizeArgs = llvm::make_range(
        funcOp.getFunctionBody().args_begin(),
        funcOp.getFunctionBody().args_begin() + tilingIterDomsOrder.size()
    );
    for (auto [domain, arg] : llvm::zip_equal(tilingIterDomsOrder, tileSizeArgs))
        tileSizes[domain] = arg;

    scf::SCFTileAndFuseOptions options{};
    options.tilingOptions.setTileSizes(tileSizes);

    // Do the tile and fuse.
    IRRewriter rewriter(moduleOp->getContext());
    auto tiledResults = scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, cast<TilingInterface>(clonedConsumerOp), options
    );
    if (failed(tiledResults)) {
        LLVM_DEBUG(assert(false));
        return llvm::failure();
    }
    applyTiledResults(rewriter, clonedConsumerOp, *tiledResults);

    // Since we don't care about the computation here, only the memory
    // footprint, and assuming the first tile is a good approximation for that,
    // we fix the tiling loops upper bound to their step, effectively resulting
    // in a single itteration. Canonicalize should simplify this later.
    for (LoopLikeOpInterface loop : tiledResults->loops) {
        auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
        if (!forOp)
            return llvm::failure();

        forOp.setUpperBound(forOp.getStep());
    }

    return llvm::success();
}

class TileAndFusePass : public TileAndFuseBase<TileAndFusePass> {
  private:
    // Normally one should use OpPassManager, and run it with the pass'
    // runPipeline. This is not possible in this case as it requires that only
    // operations nested under the current operation can be scheduled, and we
    // want to run on a module. Hence, we have to use PassManager. PassManager
    // requires a context to be constructed (and is not copy constructable).
    // Hence, we delay the construction until its first use, and we don't copy
    // it (will be constructed again in the new pass).
    std::unique_ptr<PassManager> assignAddressesPipeline_;

    // Holds all the untiled ops we have already tiled, so we don't tile them
    // again.
    llvm::DenseSet<Operation *> untiledTiledOps_;

  public:
    TileAndFusePass() {}

    TileAndFusePass(const TileAndFusePass &pass) : TileAndFuseBase(pass) {}

    void runOnOperation() override;

  private:
    llvm::LogicalResult runAssignAddressesPipeline(ModuleOp moduleOp);

    llvm::FailureOr<bool> checkModuleFitsInMemory(ModuleOp moduleOp);

    llvm::FailureOr<bool> checkTileFitsInMemory(
        ModuleOp moduleOp, ArrayRef<int64_t> tilingIterDomsOrder, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    );

    void tileAndFuse(TilingInterface tiOp);

    LogicalResult searchTileSizeForDim(
        IRRewriter &rewriter, ModuleOp moduleOp, ArrayRef<int64_t> tilingIterDomsOrder,
        MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes, size_t dim,
        int64_t iterDomainSize, int64_t minFactor, int64_t maxFactor
    );

    llvm::FailureOr<bool> fitTileToMemory(
        Operation *consumerOp, const SetVector<Operation *> &producerOps,
        ArrayRef<int64_t> tilingIterDomsOrder, ArrayRef<int64_t> iterDomainSizes,
        MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes
    );

    std::tuple<bool, bool> fuseControlMaxSize(
        IRRewriter &rewriter, mlir::tensor::ExtractSliceOp candidateSliceOp,
        OpResult producerOpResult, bool isDestinationOperand
    );

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseToSize(
        IRRewriter &rewriter, TilingInterface tilingInterfaceOp,
        llvm::ArrayRef<OpFoldResult> iterDomainSizes, SmallVector<OpFoldResult> tileSizes,
        TileAndFuseProducersFuseMode fuseMode
    );
};

llvm::LogicalResult TileAndFusePass::runAssignAddressesPipeline(ModuleOp moduleOp) {
    if (assignAddressesPipeline_ == nullptr) {
        assignAddressesPipeline_ = std::make_unique<PassManager>(moduleOp->getContext());

        if (failed(applyPassManagerCLOptions(*assignAddressesPipeline_)))
            assert(false);

        addPassesPostTileAndFuseUpToAssignLramAddresses(*assignAddressesPipeline_);
    }

    return assignAddressesPipeline_->run(moduleOp);
}

// NB: moduleOp is mutated by this function, and can't be used again.
llvm::FailureOr<bool> TileAndFusePass::checkModuleFitsInMemory(ModuleOp moduleOp) {
    bool failure = false;
    bool memoryOverflow = false;
    mlir::ScopedDiagnosticHandler diagHandler(
        moduleOp->getContext(),
        [&](mlir::Diagnostic &diag) -> LogicalResult {
            // TODO(sflur): is there a better way to identify this? The pass
            // could write something to the IR and we will look for it here?
            // Maybe use an argument (see diag.getArguments()).
            if (diag.str() == OUT_OF_MEMORY_MESSAGE) {
                memoryOverflow = true;
                // Signal to the rest of the system that we are handling this issue.
                return llvm::success();
            }

            // Something else (other than the expected memory overflow) bad happened.
            failure = true;
            llvm::errs() << "Tiling-and-fuse: the following error was encountered while running "
                            "the pipeline to checking if a tile fits in memory (it might not be "
                            "a real error):\n";
            llvm::errs().flush();

            // Signal that we are not handling this issue.
            return llvm::failure();
        }
    );

    if (failed(runAssignAddressesPipeline(moduleOp))) {
        if (memoryOverflow)
            return false;

        // This assert is just to catch things early in debug
        LLVM_DEBUG(assert(!failure));

        if (failure)
            return llvm::failure();
    }

    assert(!memoryOverflow && "this should have been captured above");

    return true;
}

// Return true iff the tile fits in memory
llvm::FailureOr<bool> TileAndFusePass::checkTileFitsInMemory(
    ModuleOp moduleOp, ArrayRef<int64_t> tilingIterDomsOrder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    OpBuilder builder(moduleOp->getContext());

    // Clone the module, as the check is distructive
    IRMapping map;
    ModuleOp fixedModuleOp = cast<ModuleOp>(builder.clone(*moduleOp, map));
    OpEraseGuard fixedModuleEraseGuard(fixedModuleOp);

    auto funcOp = cast<func::FuncOp>(&fixedModuleOp.getBody()->front());
    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    auto tileSizeArgs = llvm::make_range(
        funcOp.getFunctionBody().args_begin(),
        funcOp.getFunctionBody().args_begin() + tilingIterDomsOrder.size()
    );
    for (auto [domain, arg] : llvm::zip_equal(tilingIterDomsOrder, tileSizeArgs)) {
        // NB1: sizes[domain] is a Value/Attribute in the original module, so we
        // need to copy it to the fixedModuleOp before we can use it there.
        // NB2: as far as I can tell, it is almost always an attribute, except
        // when called from fuseControlMaxSize/Producers, where it is an affine
        // op, in which case we will try to evaluate its value in the first tile (ignoring offsets).
        llvm::FailureOr<int64_t> maybeSize = computeIntAtZero(sizes[domain]);
        if (failed(maybeSize))
            return llvm::failure();

        Value size = builder.create<arith::ConstantOp>(
            funcOp->getLoc(), builder.getIntegerAttr(builder.getIndexType(), *maybeSize)
        );

        arg.replaceAllUsesWith(size);
    }

    return checkModuleFitsInMemory(fixedModuleOp);
}

// Return true if a length can be represented using the SDIM encoding in
// Utils/Kernel.cpp without tripping the decomposeIntoTwoFactors() assert.
// A "SDIM-friendly" length is either:
//   * <= 0x7fff  – fits in a single SDIM counter, or
//   * factorizable as a * b with a, b <= 0x7fff – can be encoded as two nested SDIM loops.
const int64_t kMaxFactor = 0x7fff;
bool isSdimFriendlyCount(int64_t number) {
    if (number <= kMaxFactor)
        return true;

    for (int64_t i = div_ceil(number, kMaxFactor); i <= (int64_t)std::sqrt((double)number) + 1;
         ++i) {
        if (number % i == 0) {
            int64_t j = number / i;
            if (j <= kMaxFactor) {
                return true;
            }
        }
    }
    return false;
}

// Return true if using `tileSize` to 1-D tile `originalSize` is compatible
// with SDIM handling in Kernel.cpp.
//
// We require that both:
//   * tileSize          – the common tile length, and
//   * remainder         – the final (possibly smaller) tile where
//
//         originalSize = k * tileSize + remainder
//
// are SDIM-friendly. This guarantees that addMemNdlDims() can always encode
// both the main tiles and the tail tile without decomposeIntoTwoFactors()
// ever returning {-1, -1}.
bool isValidTileSize(int64_t iterDomainSize, int64_t tileSize) {
    if (!isSdimFriendlyCount(tileSize))
        return false;

    int64_t remainder = iterDomainSize % tileSize;
    return (remainder == 0 || isSdimFriendlyCount(remainder));
}

// Binary-search for the largest tile size along a single iteration domain that fits in memory.
// Precondition: the tile fits when sizes[domain] = div_ceil(iterDomainSize, maxFactor),
//               and does not fit when sizes[domain] = div_ceil(iterDomainSize, minFactor).
// Postcondition: sizes[domain] is set to the largest fitting tile size.
LogicalResult TileAndFusePass::searchTileSizeForDim(
    IRRewriter &rewriter, ModuleOp moduleOp, ArrayRef<int64_t> tilingIterDomsOrder,
    MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes, size_t domain,
    int64_t iterDomainSize, int64_t minFactor, int64_t maxFactor
) {
    // TODO(sflur): rethinking the whole search. benchmark an example with
    // different tile sizes but the same factor and see if there's a difference.
    // I suspect the size does matter.
    // TODO(sflur): given the double div_ceil optimization below, the factor
    // space is not linear, hence the binary search is not splitting it in half
    // correctly in each step. The result is still valid, just could converge
    // faster. There are only about 2*sqrt(iterDomainSize) effective factors,
    // which optimally can reduce the search from log(iterDomainSize) to
    // log(sqrt(iterDomainSize)).
    // nth_factor(n) = if n <= sqrt(iterDomainSize) then n
    //                 else iterDomainSize / (2*sqrt(iterDomainSize) - n)
    // Now we can do a binary search over n between 1 and 2*sqrt(iterDomainSize).
    // For each n we calculate midFactor = nth_factor(n).
    while (maxFactor != minFactor + 1) {
        int64_t midFactor = midpoint(minFactor, maxFactor);
        sizes[domain] = rewriter.getIndexAttr(div_ceil(iterDomainSize, midFactor));
        llvm::FailureOr<bool> tileFits =
            checkTileFitsInMemory(moduleOp, tilingIterDomsOrder, offsets, sizes);
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits) {
            // At the very end of the file there's a proof that the assignment
            // below is better than `maxFactor = midFactor`.
            maxFactor = div_ceil(iterDomainSize, div_ceil(iterDomainSize, midFactor));
        }
        else {
            // At the very end of the file there's a proof that the assignment
            // below is better than `minFactor = midFacto`.
            minFactor = div_ceil(iterDomainSize, div_ceil(iterDomainSize, midFactor) - 1) - 1;
        }
    }

    int64_t tileSize = div_ceil(iterDomainSize, maxFactor);

    // Adjust tileSize downwards until both the common tile size and the
    // remainder tile size are SDIM-friendly. This preserves the
    // memory-fit invariant because we only shrink the tile.
    for (tileSize = std::min(int64_t(kMaxFactor * kMaxFactor), tileSize); tileSize > 0;
         --tileSize) {
        if (isValidTileSize(iterDomainSize, tileSize))
            break;
    }
    // isValidTileSize should always succeed for tileSize == 1
    assert(tileSize > 0 && "failed to find SDIM-friendly tile size for domain");

    sizes[domain] = rewriter.getIndexAttr(tileSize);

    return LogicalResult::success();
}

// Return true iff sizes was changed (made smaller). If the tile is not big
// enough, use binary search to find the biggest tile that fits in memory.
llvm::FailureOr<bool> TileAndFusePass::fitTileToMemory(
    Operation *consumerOp, const SetVector<Operation *> &producerOps,
    ArrayRef<int64_t> tilingIterDomsOrder, ArrayRef<int64_t> iterDomainSizes,
    MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes
) {
    ModuleOp moduleOp = extractOpsForMemoryCheck(
        cast<TilingInterface>(consumerOp), tilingIterDomsOrder, producerOps
    );
    OpEraseGuard moduleEraseGuard(moduleOp);

    if (failed(tileModuleForMemoryCheck(moduleOp, tilingIterDomsOrder, iterDomainSizes.size())))
        return llvm::failure();

    // First establish that the original tile is not big enough.
    llvm::FailureOr<bool> tileFits =
        checkTileFitsInMemory(moduleOp, tilingIterDomsOrder, offsets, sizes);
    if (failed(tileFits))
        return LogicalResult::failure();
    if (*tileFits)
        // fits in memory, no need to change the tile
        return false;

    IRRewriter rewriter(moduleOp->getContext());

    OpFoldResult one = getAsIndexOpFoldResult(rewriter.getContext(), 1);

    // Shrink pass: set domains to 1, one by one, until the tile fits.
    ArrayRef<int64_t>::iterator tilingDomainIter;
    for (tilingDomainIter = tilingIterDomsOrder.begin();
         tilingDomainIter != tilingIterDomsOrder.end(); ++tilingDomainIter) {
        int64_t domain = *tilingDomainIter;
        if (*getConstantIntValue(sizes[domain]) == 1)
            continue;

        sizes[domain] = one;
        tileFits = checkTileFitsInMemory(moduleOp, tilingIterDomsOrder, offsets, sizes);
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits)
            break;
    }

    if (tilingDomainIter == tilingIterDomsOrder.end()) {
        consumerOp->emitWarning("tile-and-fuse: operation can't be tiled: no more domains to tile");
        LLVM_DEBUG(assert(false));
        return LogicalResult::failure();
    }

    // Grow-back pass: inflate domains forced to 1 above back to larger tiles
    // using binary search. Iterate in reverse from where the shrink pass stopped.
    do {
        int64_t domain = *tilingDomainIter;
        if (iterDomainSizes[domain] == 1)
            continue;

        sizes[domain] = rewriter.getIndexAttr(iterDomainSizes[domain]);
        tileFits = checkTileFitsInMemory(moduleOp, tilingIterDomsOrder, offsets, sizes);
        if (failed(tileFits))
            return LogicalResult::failure();
        if (*tileFits)
            continue;

        // We know `iterDomainSizes[domain]` is too big (tested above), hence factor
        // 1 = iterDomainSizes[domain]/iterDomainSizes[domain] is a good minimum, and we know
        // sizes[domain] = 1 does fit, hence factor iterDomainSizes[domain] =
        // iterDomainSizes[domain]/1 is a good maximum.
        if (failed(searchTileSizeForDim(
                rewriter, moduleOp, tilingIterDomsOrder, offsets, sizes, domain,
                iterDomainSizes[domain],
                /*minFactor=*/1, /*maxFactor=*/iterDomainSizes[domain]
            )))
            return LogicalResult::failure();
    } while (tilingDomainIter-- != tilingIterDomsOrder.begin());
    // NB: the tilingDimIter-- above goes passed the .begin() at the very end,
    // which is not nice, and depending on the implementation of
    // ArrayRef::iterator, could fail. The implementation is just a pointer, so
    // this is ok (as long as we don't try to dereference it, which we don't).

    return true;
}

// an SCFTileAndFuseOptions::ControlFnTy for the max-size fuse mode.
std::tuple<bool, bool> TileAndFusePass::fuseControlMaxSize(
    IRRewriter &rewriter, mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    static const std::tuple<bool, bool> doNotFuse = std::make_tuple(false, false);

    Operation *producerOp = producerOpResult.getOwner();
    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi)
        return doNotFuse;

    // TODO: if producer has users outside of the tiling fuse group, maybe we should yield it?
    bool yieldProducerReplacement = false;

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    // Get the producer's operand tiles
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        producerOp->emitWarning(
            "tile-and-fuse: failed to compute producer's operand slices, skipping producer."
        );
        LLVM_DEBUG(assert(false));
        return doNotFuse;
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    if (!isMarkedFuseGroup(producerOp)) {
        // Not part of a pattern-fuse-group; there are no restrictions on the domains.

        llvm::SmallSetVector<int64_t, 4> tilingIterDomsOrder =
            getTilingIterDomainsOrder(producerTi);

        ModuleOp moduleOp = extractOpsForMemoryCheck(producerTi, tilingIterDomsOrder.getArrayRef());
        OpEraseGuard moduleEraseGuard(moduleOp);
        if (failed(tileModuleForMemoryCheck(
                moduleOp, tilingIterDomsOrder.getArrayRef(), iterSizes.size()
            )))
            assert(false && "failed to compute producer's tiled memory size");

        llvm::FailureOr<bool> producerFits = checkTileFitsInMemory(
            moduleOp, tilingIterDomsOrder.getArrayRef(), mappedOffsets, mappedSizes
        );
        if (failed(producerFits)) {
            producerOp->emitWarning(
                "tile-and-fuse: failed to check if tiled producer overflows, skipping producer."
            );
            LLVM_DEBUG(assert(false));
            return doNotFuse;
        }

        // Fuse iff producer fits in the tile
        return std::make_tuple(*producerFits, yieldProducerReplacement);
    }

    if (!isFuseGroupOutput(producerOp)) {
        // Is part of a pattern-fuse-group, but not the output operation (bottom most).
        // We only check the output operation of such group.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    llvm::SmallSetVector<int64_t, 4> tilingIterDomsOrder = getTilingIterDomainsOrder(producerTi);

    // The domains the producer can be tiled over
    // Don't fuse if we are tiling a domain the producer's fuse group can not be tiled over
    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (mappedSizes[index] != iterSizes[index] && !tilingIterDomsOrder.contains(index)) {
            return doNotFuse;
        }
    }

    ModuleOp moduleOp = extractOpsForMemoryCheck(producerTi, tilingIterDomsOrder.getArrayRef());
    OpEraseGuard moduleEraseGuard(moduleOp);
    if (failed(
            tileModuleForMemoryCheck(moduleOp, tilingIterDomsOrder.getArrayRef(), iterSizes.size())
        ))
        assert(false && "failed to compute producer's tiled memory size");

    llvm::FailureOr<bool> producerFuseGroupFits = checkTileFitsInMemory(
        moduleOp, tilingIterDomsOrder.getArrayRef(), mappedOffsets, mappedSizes
    );
    if (failed(producerFuseGroupFits)) {
        producerOp->emitWarning("tile-and-fuse: failed to check if tiled (pattern-fuse-group) "
                                "producers overflows, skipping producer.");
        LLVM_DEBUG(assert(false));
        return doNotFuse;
    }

    // Fuse iff producer fits in the tile
    return std::make_tuple(*producerFuseGroupFits, yieldProducerReplacement);
}

// an SCFTileAndFuseOptions::ControlFnTy for the max-producers fuse mode.
std::tuple<bool, bool> fuseControlMaxProducers(
    IRRewriter &rewriter, mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    static const std::tuple<bool, bool> doNotFuse = std::make_tuple(false, false);

    Operation *producerOp = producerOpResult.getOwner();
    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi)
        return doNotFuse;

    // TODO: if producer has users outside of the tiling fuse group, maybe we should yield it?
    bool yieldProducerReplacement = false;

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    // From here on we check if the producer can be tiled on the domains that
    // are being tiled. The first element in the return tuple indicates if the
    // producer should be fused (true) or not (false).

    if (!isMarkedFuseGroup(producerOp)) {
        // Not part of a pattern-fuse-group; there are no restrictions on the domains.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    if (!isFuseGroupOutput(producerOp)) {
        // Is part of a pattern-fuse-group, but not the output operation (bottom most).
        // We only check the output operation of such group.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    // The domains the producer can be tiled over
    llvm::SmallSetVector<int64_t, 4> tilingIterDomsOrder = getTilingIterDomainsOrder(producerTi);

    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        producerOp->emitWarning(
            "tile-and-fuse: failed to compute producer's operand slices, skipping producer."
        );
        LLVM_DEBUG(assert(false));
        return doNotFuse;
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    // Don't fuse if we are tiling a domain the producer can not be tiled over
    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (getConstantIntValue(mappedSizes[index]) != getConstantIntValue(iterSizes[index]) &&
            !tilingIterDomsOrder.contains(index)) {
            LLVM_DEBUG({
                llvm::dbgs() << "Do not fuse (can not be tiled over domain " << index << ", "
                             << mappedSizes[index] << " != " << iterSizes[index]
                             << "): " << producerOp->getName() << "\n";
            });
            return doNotFuse;
        }
    }

    // Producer can be fused
    return std::make_tuple(true, yieldProducerReplacement);
}

FailureOr<scf::SCFTileAndFuseResult> TileAndFusePass::tileAndFuseToSize(
    IRRewriter &rewriter, TilingInterface tilingInterfaceOp,
    llvm::ArrayRef<OpFoldResult> iterDomainSizes, SmallVector<OpFoldResult> tileSizes,
    TileAndFuseProducersFuseMode fuseMode
) {
    // set domains that are not being tiled to 0
    OpFoldResult zero = getAsIndexOpFoldResult(&getContext(), 0);
    for (auto &&[iterDomainSize, tileSize] : llvm::zip_equal(iterDomainSizes, tileSizes)) {
        if (iterDomainSize == tileSize)
            tileSize = zero;
    }

    LLVM_DEBUG({
        llvm::dbgs() << "  tile sizes: ";
        llvm::interleave(*getConstantIntValues(tileSizes), llvm::dbgs(), "x");
        llvm::dbgs() << "\n";
    });

    scf::SCFTileAndFuseOptions options{};
    options.tilingOptions.setTileSizes(tileSizes);

    std::optional<int64_t> maybePatternFuseGroup = isFuseGroupOutput(tilingInterfaceOp);

    scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn;
    switch (fuseMode) {
    case TileAndFuseProducersFuseMode::MaxSize:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxSize(
                rewriter, candidateSliceOp, producerOpResult, isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::MaxProducers:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxProducers(
                rewriter, candidateSliceOp, producerOpResult, isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::OnlyPatterns:
        fusionControlFn = [maybePatternFuseGroup](
                              mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand
                          ) {
            bool shouldFuse =
                maybePatternFuseGroup && isMarkedFuseGroup(producerOpResult.getOwner());
            return std::make_tuple(shouldFuse, false);
        };
        break;

    case TileAndFuseProducersFuseMode::NoFuse:
        fusionControlFn = [](mlir::tensor::ExtractSliceOp candidateSliceOp,
                             OpResult producerOpResult,
                             bool isDestinationOperand) { return std::make_tuple(false, false); };
    }

    options.setFusionControlFn(fusionControlFn);

    return scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp, options);
}

void TileAndFusePass::tileAndFuse(TilingInterface tiOp) {
    // Check if we already tiled tiOp (as producer)
    if (untiledTiledOps_.contains(tiOp))
        return;

    // For pattern-fuse-groups, we only tile from the bottom most op, to make
    // sure the whole group is tiled together.
    if (isMarkedFuseGroup(tiOp) && !isFuseGroupOutput(tiOp))
        return;

    // Check if tiOp already fits in memory.
    {
        ModuleOp moduleOp = extractOpsForMemoryCheck(tiOp);
        OpEraseGuard moduleEraseGuard(moduleOp);
        llvm::FailureOr<bool> opFitsInMemory = checkModuleFitsInMemory(moduleOp);
        if (failed(opFitsInMemory)) {
            tiOp->emitWarning(
                "tile-and-fuse: initial memory overflow check failed, skipping this operation."
            );
            LLVM_DEBUG(assert(false));
            return;
        }

        if (*opFitsInMemory)
            return;
    }

    LLVM_DEBUG({
        llvm::dbgs() << tiOp->getName() << " needs to be tiled\n";
        llvm::dbgs() << "  " << TorqHw::get().getAvailableMemoryForTiling()
                     << " bytes of available memory for tiling\n";

        if (Attribute groupId = tiOp->getAttr(TORQ_FUSE_GROUP_ID)) {
            llvm::dbgs() << "  " << TORQ_FUSE_GROUP_ID << ": " << getConstantIntValue(groupId)
                         << "\n";
            if (isMarkedFuseGroup(tiOp)) {
                Operation *principalOp = getFuseGroupPrincipalOpBackward(tiOp);
                llvm::dbgs() << "  principal op: " << principalOp->getName() << "\n";
                Attribute fuseGroup = principalOp->getAttr(TORQ_FUSE_GROUP_ID);
                llvm::dbgs() << "  " << TORQ_FUSE_GROUP << ": " << getConstantIntValue(fuseGroup)
                             << "\n";
            }
        }
    });

    IRRewriter rewriter(&getContext());

    // The untiled domain sizes
    auto [iterDomainOffsets, iterDomainSizes, iterDomainStrides] =
        getOffsetsSizesAndStrides(tiOp.getIterationDomain(rewriter));

    std::optional<SmallVector<int64_t>> iterDomainConstSizes =
        getConstantIntValues(iterDomainSizes);
    if (!iterDomainConstSizes) {
        tiOp->emitWarning(
            "tile-and-fuse: iteration domain sizes are not constants, skipping this operation."
        );
        LLVM_DEBUG(assert(false));
        return;
    }

    // Will hold the new tile size
    SmallVector<OpFoldResult> tileOffsets(iterDomainOffsets), tileSizes(iterDomainSizes);

    // Get the order in which we should tile the domains.
    llvm::SmallSetVector<int64_t, 4> tilingIterDomsOrder = getTilingIterDomainsOrder(tiOp);

    // Find a tile size that fits tiOp in memory (no producers, except for the
    // required pattern-fuse-group members)
    llvm::FailureOr<bool> tileChanged = fitTileToMemory(
        tiOp, {}, tilingIterDomsOrder.getArrayRef(), *iterDomainConstSizes, tileOffsets, tileSizes
    );
    if (failed(tileChanged)) {
        tiOp->emitWarning("tile-and-fuse: failed to tile an operation, skipping it");
        LLVM_DEBUG(assert(false));
        return;
    }

    assert(*tileChanged && "untiled op does not fit in memory but tiling to a single tile does");

    LLVM_DEBUG({
        llvm::dbgs() << "  iteration sizes: ";
        llvm::interleave(*getConstantIntValues(iterDomainSizes), llvm::dbgs(), "X");
        llvm::dbgs() << "\n";

        llvm::dbgs() << "  iteration types: ";
        llvm::interleave(tiOp.getLoopIteratorTypes(), llvm::dbgs(), ", ");
        llvm::dbgs() << "\n";
    });

    // Do the actual tiling (might need to do it again later)!
    FailureOr<scf::SCFTileAndFuseResult> tiledResults = tileAndFuseToSize(
        rewriter, tiOp, iterDomainSizes, tileSizes, clTorqTileAndFuseProducersFuseMode.getValue()
    );
    if (failed(tiledResults)) {
        tiOp->emitWarning("tile-and-fuse: failed to tile operation, skipping it.");
        LLVM_DEBUG(assert(false));
        return;
    }

    // In MaxProducers mode, we readjust the tile sizes to fit all the producers.
    if (clTorqTileAndFuseProducersFuseMode.getValue() ==
        TileAndFuseProducersFuseMode::MaxProducers) {

        // Will hold the new tile size
        SmallVector<OpFoldResult> producersTileOffsets(tileOffsets), producersTileSizes(tileSizes);

        // Now that we know which producers were fused, find a tile that fits them too.
        llvm::FailureOr<bool> tileChanged = fitTileToMemory(
            tiOp, tiledResults->fusedProducers, tilingIterDomsOrder.getArrayRef(),
            *iterDomainConstSizes, producersTileOffsets, producersTileSizes
        );
        if (failed(tileChanged)) {
            // Fall back to MaxSize
            // This can happen if one of the producers does not tile well over
            // the domains that the consumer dictates. A better solution will
            // be to find the problematic producers and not fuse them, if
            // possible, and only if that does not work fall back to MaxSize.
            // The problem is to identify the culprits. We could add to
            // fuseControlMaxProducers a check that the producer can be tiled to
            // a 1x1x.. tile (over all the domains in tilingIterDomsOrder) before
            // fusing it.
            tiledResults = tileAndFuseToSize(
                rewriter, tiOp, iterDomainSizes, tileSizes, TileAndFuseProducersFuseMode::MaxSize
            );
            if (failed(tiledResults)) {
                tiOp->emitWarning("tile-and-fuse: failed to tile operation, skipping it.");
                LLVM_DEBUG(assert(false));
                return;
            }
        }
        else if (*tileChanged) {
            // Producers required a smaller tile, do the actual tiling one more
            // time (the previous one will be discarded by canonicalize).
            tiledResults = tileAndFuseToSize(
                rewriter, tiOp, iterDomainSizes, producersTileSizes,
                clTorqTileAndFuseProducersFuseMode.getValue()
            );
            if (failed(tiledResults)) {
                tiOp->emitError("tile-and-fuse: failed to tile operation, skipping it.");
                LLVM_DEBUG(assert(false));
                return;
            }
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "  fused " << tiledResults->fusedProducers.size() << " producers\n";
        for (Operation *producerOp : tiledResults->fusedProducers) {
            llvm::dbgs() << "  | " << producerOp->getName() << "\n";
        }
    });

    // Replace the untiled tiOp with the tiled results.
    applyTiledResults(rewriter, tiOp, *tiledResults);

    LLVM_DEBUG({
        // Check if the tiled and fused loop fits in memory

        Operation *tiledOp = nullptr;
        for (OpResult res : tiOp->getResults()) {
            if (Value replacement = tiledResults->replacements.lookup(res)) {
                tiledOp = replacement.getDefiningOp();
                break;
            }
        }
        assert(tiledOp);

        ModuleOp moduleOp = extractOpsForMemoryCheck(tiledOp);
        OpEraseGuard moduleEraseGuard(moduleOp);

        llvm::FailureOr<bool> opFitsInMemory = checkModuleFitsInMemory(moduleOp);
        assert(succeeded(opFitsInMemory));
        assert(*opFitsInMemory);
    });

    untiledTiledOps_.insert(tiOp);
    llvm::set_union(untiledTiledOps_, tiledResults->fusedProducers);
}

void TileAndFusePass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << "Tile and Fuse - START\n");

    FunctionOpInterface funcOp = getOperation();

    // Walk on all the TilingInterface ops in the function, in reverse order of
    // appearance. This guarantees that when we tile an op, we have already
    // considered all its users.
    // NB: Since tileAndFuse mutates the IR, we can't call it directly from the
    // walk. We first construct a queue, and than iterate over it.
    SmallVector<TilingInterface> orderTi;
    funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface tiOp) {
        orderTi.push_back(tiOp);
    });

    untiledTiledOps_.reserve(orderTi.size());

    for (TilingInterface tiOp : orderTi) {
        tileAndFuse(tiOp);
    }

    LLVM_DEBUG(llvm::dbgs() << "Tile and Fuse - DONE\n");
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileAndFusePass() {
    return std::make_unique<TileAndFusePass>();
}

} // namespace mlir::syna::torq

// A quick explanation for the optimizations in TileAndFusePass::searchTileSizeForDim
//
// Explaining why `maxFactor = div_ceil(iterDomainSize, div_ceil(iterDomainSize, midFactor))` is
// safe.
//
// We need to show:
// 1. That the new maxFactor results in the same tile size as midFactor, which
//    we already checked and found to fit in memory.
//    That is, we need to show `div_ceil(iterDomainSize, maxFactor) == div_ceil(iterDomainSize,
//    midFactor)`, where maxFactor is the double div_ceil.
//
// 2. And, `maxFactor <= midFactor`, so the binary search will converge (and
//    because in most cases the inequality is actually strict, we get a faster
//    convergence, but we don't show this here).
//
// From here on everything is in math semantics. In particular "/" means
// division over real numbers (i.e. no rounding), and "=" means equality (not assignment).
//
// Lemma: for every natural numbers a,b,c,d, if ceil(a / b) = c and ceil(a / c) = d then d <= b
// Proof: a / b <= ceil(a / b) = c  ==>  a <= c*b
//        d = ceil(a / c) <= ceil(c*b / c) = b  ==>  d <= b
//
// To prove 1 and 2 above, we first rewrite things like this:
//   midSize = ceil(iterDomainSize / midFactor)
//   maxFactor = ceil(iterDomainSize / midSize)
//   maxSize = ceil(iterDomainSize / maxFactor)
//
// The proof of 2 is immediate from the lemma, where a=iterDomainSize, b=midFactor, c=midSize, and
// d=maxFactor.
//
// To prove 1 we need to show midSize == maxSize, which we will do by showing midSize <= maxSize,
// and midSize >= maxSize. The latter is just another application of the lemma where
// a=iterDomainSize, b=midSize, c=maxFactor, and d=maxSize. And the former:
//   maxFactor <= midFactor  ==>  ceil(iterDomainSize / maxFactor) >= ceil(iterDomainSize /
//   midFactor)  ==>  maxSize >= midSize

// Explaining why `minFactor = div_ceil(iterDomainSize, div_ceil(iterDomainSize, midFactor) - 1) -1`
// is safe.
//
// From here on everything is in math semantics.
//
// We rewrite things like this:
//   midSize = ceil(iterDomainSize / midFactor)
//   midSize' = ceil(iterDomainSize / midFactor) - 1
//   minFactor = ceil(iterDomainSize / midSize') - 1
//   minSize = ceil(iterDomainSize / minFactor)
//   Note that: minSize - 1 = ceil(iterDomainSize / minFactor) - 1
//
// We need to show:
// 1. midSize = minSize
// 2. midFactor <= minFactor
//
// Lemma: for every natural numbers a,b,c,d, if ceil(a / b) - 1 = c, and ceil(a / c) - 1 = d, then d
// >= b Proof: a / b + 1 - 1 > ceil(a / b) - 1 = c  ==>  a > b*c
//        d = ceil(a / c) - 1 > ceil(b*c / c) - 1 = b - 1  ==> d >= b
//
// The proof of 2 is immediate from the lemma.
//
// To prove 1 we need to show midSize == minSize, which we will do by showing minSize >= midSize,
// and minSize <= midSize. An application of the lemma gives us:
//   minSize - 1 >= midSize'  ==>  minSize - 1 >= midSize - 1  ==>  minSize >= midSize.
// And:
//   minFactor >= midFactor  ==>  ceil(iterDomainSize / minFactor) <= ceil(iterDomainSize /
//   midFactor)  ==>  minSize <= midSize
