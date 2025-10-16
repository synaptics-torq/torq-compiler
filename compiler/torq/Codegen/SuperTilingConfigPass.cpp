// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <mlir/Analysis/SliceAnalysis.h>
#include <optional>
#include <tuple>
#include <utility>

// TODO: all the TypeSwitchs should be extracted to SuperTilingInterface

#define DEBUG_TYPE "torq-super-tiling-config"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class SuperTilingConfigPass : public SuperTilingConfigBase<SuperTilingConfigPass> {
  public:
    SuperTilingConfigPass() = default;
    SuperTilingConfigPass(const SuperTilingConfigPass &pass) {}

    void runOnOperation() override;
};

static const std::string TORQ_TILING_DFS = "torq-tiling-dfs";
static const std::string TORQ_TILING_FUSED = "torq-tiling-fused";

torq_hl::Executor getExecutor(Operation *op) {
    return isMarkedFuseGroup(op) ? torq_hl::Executor::Slice : torq_hl::Executor::CSS;
}

void forwardDfs(Operation *firstOp, SmallVector<Operation *> &orderTi) {
    // auto boolAttrTrue = BoolAttr::get(firstOp->getContext(), true);
    auto boolAttrFalse = BoolAttr::get(firstOp->getContext(), false);

    std::deque<std::pair<Operation *, ResultRange::user_iterator>> stack;

    if (firstOp->getAttrOfType<BoolAttr>(TORQ_TILING_DFS) ||
        !isa_and_nonnull<TilingInterface>(firstOp))
        return;

    firstOp->setAttr(TORQ_TILING_DFS, boolAttrFalse);
    stack.push_back(std::make_pair(firstOp, firstOp->getResults().user_begin()));

    while (!stack.empty()) {
        auto &[op, user] = stack.back();

        if (user == op->getResults().user_end()) {
            // Close op
            stack.pop_back();
            orderTi.push_back(op);
            continue;
        }

        auto userOp = *user++;
        if (userOp->getAttrOfType<BoolAttr>(TORQ_TILING_DFS) ||
            !isa_and_nonnull<TilingInterface>(userOp)) {
            // userOp was already opened (and possibly closed).
            continue;
        }

        // Open userOp
        userOp->setAttr(TORQ_TILING_DFS, boolAttrFalse);
        stack.push_back(std::make_pair(userOp, userOp->getResults().user_begin()));
    }
}

void orderTiOps(FunctionOpInterface &funcOp, SmallVector<Operation *> &orderTi) {
    for (auto &block : funcOp.getBlocks()) {
        for (auto &op : block.getOperations()) {
            forwardDfs(&op, orderTi);
        }
    }
}

SmallVector<int64_t> getTilingDimOrder(TilingInterface tilingInterfaceOp) {
    auto loopIteratorTypes = tilingInterfaceOp.getLoopIteratorTypes();

    // Handler that use all dimensions for tiling (as needed), in that order.
    auto allParallelDims = [&]() {
        SmallVector<int64_t> tilingDimOrder;
        for (size_t dim = 0; dim < loopIteratorTypes.size(); ++dim) {
            if (loopIteratorTypes[dim] != utils::IteratorType::parallel)
                continue;

            tilingDimOrder.push_back(dim);
        }
        return tilingDimOrder;
    };

    // Handler for nhwc operations we tile h and c (as needed), in that order.
    // The HW currently does not support tiling w (can't do the padding right), so we skip it.
    // The HW currently does not support tiling h when stride is > 1.
    auto nhwcOpWithStires = [](mlir::DenseIntElementsAttr stridesAttr) {
        SmallVector<int64_t> tilingDimOrder;
        SmallVector<int64_t, 2> strides;
        for (int64_t stride : stridesAttr.getValues<int64_t>()) {
            strides.push_back(stride);
        }
        if (strides[0] > 1) {
            // tile only the channels
            tilingDimOrder.push_back(3); // C
        }
        else {
            // tile horizontally and then channels
            tilingDimOrder.push_back(1); // H
            tilingDimOrder.push_back(3); // C
        }
        return tilingDimOrder;
    };

    // Decide which dimensions to tile, and in what order, based on the executor and the operation.
    switch (getExecutor(tilingInterfaceOp)) {
    case torq_hl::Executor::Slice: {
        Operation *principalOp = getFuseGroupPrincipalOpBackward(tilingInterfaceOp);
        assert(principalOp != nullptr && "could not find the principal op of the fuse group");

        return TypeSwitch<Operation *, SmallVector<int64_t>>(principalOp)
            .Case<
                linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp,
                linalg::PoolingNhwcMaxOp, linalg::PoolingNhwcMaxUnsignedOp,
                linalg::PoolingNhwcMinOp, linalg::PoolingNhwcMinUnsignedOp,
                linalg::PoolingNhwcSumOp>([&](auto convOp) {
                return nhwcOpWithStires(convOp.getStrides());
            })
            .Default([&](auto) { return allParallelDims(); });
    }
    case torq_hl::Executor::CSS:
        return allParallelDims();

    default:
        llvm_unreachable("expected NSS or CSS executor");
    }
}

int64_t bytesOfSlice(Type type, ArrayRef<OpFoldResult> resultSizes) {
    if (auto rankedType = dyn_cast<RankedTensorType>(type)) {
        int64_t bytes = div_ceil(rankedType.getElementTypeBitWidth(), 8);
        for (auto sizeValue : resultSizes) {
            auto size = getConstantIntValue(sizeValue);
            assert(size && "slice size is not constant");
            bytes *= size.value();
        }
        return bytes;
    }

    return div_ceil(type.getIntOrFloatBitWidth(), 8);
}

// This iteration domain is for all the loops of the op.
typedef std::tuple<
    Operation *, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
    OpIterationDomain;
// This iteration domain is only for the result loops (the owner op might have more loops).
typedef std::tuple<
    OpResult, SmallVector<OpFoldResult> /* offsets */, SmallVector<OpFoldResult> /* sizes */>
    OpResultIterationDomain;

llvm::FailureOr<SmallVector<OpResultIterationDomain>> linalgOperandSlicesFromIterDomain(
    IRRewriter &rewriter, linalg::LinalgOp linalgOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    auto allSliceParameter = linalg::computeAllSliceParameters(
        rewriter, linalgOp->getLoc(), linalgOp, linalgOp->getOperands(), offsets, sizes, {}, true
    );

    SmallVector<OpResultIterationDomain> oprandIterDomains;
    oprandIterDomains.reserve(allSliceParameter.size());
    for (auto [operand, sliceParams] : llvm::zip(linalgOp->getOperands(), allSliceParameter)) {

        OpResult operandOpResult = dyn_cast<OpResult>(operand);

        if (sliceParams) {
            oprandIterDomains.push_back(
                std::make_tuple(operandOpResult, sliceParams->offsets, sliceParams->sizes)
            );
            continue;
        }

        if (RankedTensorType rankedType = dyn_cast<RankedTensorType>(operandOpResult.getType())) {
            SmallVector<OpFoldResult> sizes, offsets;
            sizes.reserve(rankedType.getShape().size());
            for (auto size : rankedType.getShape()) {
                sizes.push_back(rewriter.getIndexAttr(size));
                offsets.push_back(rewriter.getIndexAttr(0));
            }
            oprandIterDomains.push_back(std::make_tuple(operandOpResult, offsets, sizes));
            continue;
        }

        oprandIterDomains.push_back(std::make_tuple(
            operandOpResult, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)},
            SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)}
        ));
    }

    return oprandIterDomains;
}

llvm::FailureOr<SmallVector<OpResultIterationDomain>> tensorPadOperandSlicesFromIterDomain(
    IRRewriter &rewriter, tensor::PadOp padOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    auto operand = padOp.getSource();
    OpResult operandOpResult = dyn_cast<OpResult>(operand);

    // TODO: I think the following is a safe over approximation, maybe do
    // something more precise.
    return SmallVector<OpResultIterationDomain>{std::make_tuple(
        operandOpResult, SmallVector<OpFoldResult>(offsets), SmallVector<OpFoldResult>(sizes)
    )};
}

// Return true iff the tile (including consumerOp and producerOps) can fit in availableMemoryBytes.
// When consumerOp is a member of a fuse group, it must be the output operation of that group. In
// that case, the entire group will be checked, even if the other members are not in producerOps.
// This gives the correct result when the function is called to check if the group needs to be
// tiled, before it was tiled.
// TODO: cache all the iteration domains so they don't need to be reallocated every time.
llvm::FailureOr<bool> checkTileFitsInMemory(
    IRRewriter &rewriter, Operation *consumerOp, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, const SetVector<Operation *> &producerOps,
    int64_t availableMemoryBytes
) {
    // How it works: we start from consumerOp, we calculate for each operand the slice needed for
    // the tile, and sum up the bytes for all the slices. If the sum is greater than
    // availableMemoryBytes, we return false; otherwise, we compute the iteration domain for the
    // operand based on the its slice (it might have more loops), and add it to the queue. We
    // process elements in the queue in the same way until it is empty. For members of fuse groups,
    // we don't check the sum of their immediate operands, instead we check the sum of operands that
    // are not in the same group.

    // FIXME: we currently check that each operation/fuse-group can fit in availableMemoryBytes.
    // This does not take into account that while computing an operand, other operands already take
    // space.

    // Accumulate for each fuse-group the size of operands external to the group.
    llvm::DenseMap<IntegerAttr, int64_t> fusedGroupsBytes;
    // Keep track of which operands we already counted, so we don't count them multiple times (e.g.
    // usually the same tensor.empty is used multiple times by the same group; we want to count it
    // only once).
    llvm::DenseMap<IntegerAttr, SetVector<Operation *>> fusedGroupsOperands;

    std::deque<OpIterationDomain> queue;
    queue.push_back(std::make_tuple(
        consumerOp, SmallVector<OpFoldResult>(offsets), SmallVector<OpFoldResult>(sizes)
    ));
    while (!queue.empty()) {
        // Can't use structured bindings because we later capture some of them in a lambda (C++20
        // supports it):
        // auto [op, offsets, sizes] = queue.front();
        // so Instead we do this:
        auto op = std::get<0>(queue.front());
        auto offsets = std::get<1>(queue.front());
        auto sizes = std::get<2>(queue.front());
        queue.pop_front();

        // Compute operand slices
        llvm::FailureOr<SmallVector<OpResultIterationDomain>> operandSlices =
            TypeSwitch<Operation *, llvm::FailureOr<SmallVector<OpResultIterationDomain>>>(op)
                .Case<linalg::LinalgOp>([&](auto linalgOp) {
                    return linalgOperandSlicesFromIterDomain(rewriter, linalgOp, offsets, sizes);
                })
                .Case<tensor::PadOp>([&](tensor::PadOp padOp) {
                    return tensorPadOperandSlicesFromIterDomain(rewriter, padOp, offsets, sizes);
                })
                .Default([](auto) { return LogicalResult::failure(); });

        if (failed(operandSlices)) {
            op->emitError("failed to compute operand slices");
            return LogicalResult::failure();
        }

        // Accumulate memory used by operand slices
        int64_t totalOpBytes = 0;

        // For each operand slice, compute its memory usage, and add the owner to the queue as
        // needed.
        for (auto [opResult, resultOffsets, resultSizes] : *operandSlices) {
            Operation *resultOp = opResult.getOwner();

            // Compute memory usage
            int64_t operandBytes = bytesOfSlice(opResult.getType(), resultSizes);
            totalOpBytes += operandBytes;
            if (totalOpBytes > availableMemoryBytes && !isMarkedFuseGroup(op)) {
                return false;
            }

            bool shareFuseGroup = false;

            // If op is in a fuse group, check if this is an external operand, and add it to the
            // group's memory.
            if (auto fuseGroupAttr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
                auto operandFuseGroupAttr = resultOp->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);

                for (auto intAttr : fuseGroupAttr.getAsRange<IntegerAttr>()) {
                    if (operandFuseGroupAttr &&
                        std::find(
                            operandFuseGroupAttr.begin(), operandFuseGroupAttr.end(), intAttr
                        ) != operandFuseGroupAttr.end()) {

                        shareFuseGroup = true;
                    }
                    else {
                        if (fusedGroupsOperands[intAttr].insert(resultOp)) {
                            fusedGroupsBytes[intAttr] += operandBytes;
                            if (fusedGroupsBytes[intAttr] > availableMemoryBytes) {
                                return false;
                            }
                        }
                    }
                }
            }

            // Add operand to the queue as needed
            if (producerOps.contains(resultOp) || shareFuseGroup) {
                auto tiOp = cast<TilingInterface>(resultOp);

                // Get the iteration domain for all the loops of the operand owner
                SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
                if (failed(tiOp.getIterationDomainTileFromResultTile(
                        rewriter, opResult.getResultNumber(), resultOffsets, resultSizes,
                        mappedOffsets, mappedSizes
                    ))) {

                    tiOp->emitError("getIterationDomainTileFromResultTile failed");
                    return LogicalResult::failure();
                }

                queue.push_back(std::make_tuple(resultOp, mappedOffsets, mappedSizes));
            }
        }
    }

    return true;
}

// C++20 std::midpoint: computes average without overflow
inline int64_t midpoint(int64_t min, int64_t max) { return min + ((max - min) / 2); }

// Return true iff sizes was changed (made smaller).
// If the tile is not big enough, use binary search to find the biggest tile that fits in
// availableMemoryBytes.
llvm::FailureOr<bool> fitTileToMemory(
    IRRewriter &rewriter, Operation *consumerOp, const SetVector<Operation *> &producerOps,
    const SmallVector<int64_t> &tilingDimOrder, size_t availableMemoryBytes,
    MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes
) {
    // TODO: change the binary search to look for a factor

    LLVM_DEBUG({
        llvm::dbgs() << availableMemoryBytes << " bytes of available memory for tiling\n";
    });

    // First establish that the original tile is not big enough.
    auto opsFitsInMem = checkTileFitsInMemory(
        rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
    );
    if (failed(opsFitsInMem)) {
        return LogicalResult::failure();
    }
    if (*opsFitsInMem) {
        // ops fit in memory, no need to change the tile
        return false;
    }

    for (auto dim : tilingDimOrder) {
        LLVM_DEBUG({ llvm::dbgs() << "tiling dim " << dim << "\n"; });

        int64_t maxSize = *getConstantIntValue(sizes[dim]);
        if (maxSize == 1) {
            continue;
        }

        sizes[dim] = getAsIndexOpFoldResult(rewriter.getContext(), 1);
        auto checkResult = checkTileFitsInMemory(
            rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
        );
        if (failed(checkResult)) {
            return LogicalResult::failure();
        }
        if (!*checkResult) {
            continue;
        }

        int64_t minSize = 1;

        // Loop invariant: consumerOp tiled to sizes fits in availableMemoryBytes, when sizes[dim]
        // is set to minSize, and does not fit when it is set to maxSize.
        // NB: from the above, it is clear that `maxSize != minSize` is always true, hence the while
        // condition is as it is.
        while (maxSize != minSize + 1) {
            int64_t midSize = midpoint(minSize, maxSize);
            sizes[dim] = rewriter.getIndexAttr(midSize);

            auto checkResult = checkTileFitsInMemory(
                rewriter, consumerOp, offsets, sizes, producerOps, availableMemoryBytes
            );
            if (failed(checkResult)) {
                return LogicalResult::failure();
            }

            if (*checkResult) {
                minSize = midSize;
            }
            else {
                maxSize = midSize;
            }
        }
        sizes[dim] = rewriter.getIndexAttr(minSize);

        return true;
    }

    return LogicalResult::failure();
}

void replaceTiledOp(
    IRRewriter &rewriter, Operation *op, const scf::SCFTileAndFuseResult &tiledResults
) {
    for (OpResult res : op->getResults()) {
        if (auto replacement = tiledResults.replacements.lookup(res)) {
            rewriter.replaceAllUsesWith(res, replacement);
        }
    }
}

void applyTiledResults(
    IRRewriter &rewriter, Operation *op, scf::SCFTileAndFuseResult &tiledResults
) {
    // Replace the root with its tiled result
    replaceTiledOp(rewriter, op, tiledResults);

    // Mark root so we don't tile it's producers (as roots)
    op->setAttr(TORQ_TILING_FUSED, rewriter.getBoolAttr(true));

    for (auto prodOp : tiledResults.fusedProducers) {
        // In general, fused producers can declare that they want to be
        // yielded. Here we replace the original producers with the yielded
        // result. In practice, we use the default fusion options, so no
        // producers are yielded.
        replaceTiledOp(rewriter, prodOp, tiledResults);

        // Mark producers so we don't tile their producers (as roots)
        prodOp->setAttr(TORQ_TILING_FUSED, rewriter.getBoolAttr(true));
    }
}

std::tuple<bool, bool> fuseControl(
    IRRewriter &rewriter, tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    Operation *producerOp = producerOpResult.getOwner();

    // TODO: if producer has users outside of the tiling fuse group, maybe we should yield it?
    bool yieldProducerReplacement = false;

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     // Don't fuse
    //     return std::make_tuple(false, yieldProducerReplacement);
    // }

    // From here on we check if the producer can be tiled on the dimensions that
    // are being tiled. The first element in the return tuple indicates if the
    // producer should be fused (true) or not (false).

    if (!isMarkedFuseGroup(producerOp)) {
        // Not part of a pattern-fuse-group; there are no restrictions on the dimensions.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    if (!isFuseGroupOutput(producerOp)) {
        // Is part of a pattern-fuse-group, but not the output operation (bottom most).
        // We only check the output operation of such group.
        return std::make_tuple(true, yieldProducerReplacement);
    }

    auto producerTi = cast<TilingInterface>(producerOp);

    // The dimensions the producer can be tiled over
    auto dimOrder = getTilingDimOrder(producerTi);
    llvm::SmallSetVector<int64_t, 4> tilingDims{dimOrder.begin(), dimOrder.end()};

    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        // Don't fuse
        return std::make_tuple(false, yieldProducerReplacement);
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    // Don't fuse if we are tiling a dimension the producer can not be tiled over
    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (mappedSizes[index] != iterSizes[index] && !tilingDims.contains(index)) {
            // Don't fuse
            return std::make_tuple(false, yieldProducerReplacement);
        }
    }

    // Producer can be fused
    return std::make_tuple(true, yieldProducerReplacement);
}

FailureOr<scf::SCFTileAndFuseResult> tileAndFuseToSize(
    IRRewriter &rewriter, TilingInterface tilingInterfaceOp,
    llvm::ArrayRef<OpFoldResult> completeIterSizes, ArrayRef<OpFoldResult> tileIterSizes
) {
    // set dimensions that are not being tiled to 0 (and convert to int64_t).
    SmallVector<int64_t> tileSizes(completeIterSizes.size(), 0);
    for (size_t i = 0; i < completeIterSizes.size(); ++i) {
        auto iterSize = getConstantIntValue(completeIterSizes[i]);
        auto tileSize = getConstantIntValue(tileIterSizes[i]);
        if (*tileSize != *iterSize) {
            tileSizes[i] = *tileSize;
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "tile sizes: ";
        bool isFirst = true;
        for (auto d : tileSizes) {
            llvm::dbgs() << (isFirst ? "" : "x") << d;
            isFirst = false;
        }
        llvm::dbgs() << "\n";
    });

    scf::SCFTileAndFuseOptions options{};
    // Consider using setSCFTileSizes from iree/compiler/Codegen/Utils/Utils.h
    options.tilingOptions.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
    options.setFusionControlFn([&](tensor::ExtractSliceOp candidateSliceOp,
                                   OpResult producerOpResult, bool isDestinationOperand) {
        return fuseControl(rewriter, candidateSliceOp, producerOpResult, isDestinationOperand);
    });

    return scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp, options);
}

void tileAndFuse(MLIRContext *context, Operation *op) {
    // Check if it was already fused as a producer
    if (op->getAttrOfType<BoolAttr>(TORQ_TILING_FUSED)) {
        return;
    }

    // For fuse-groups, we only tile from the bottom most op
    if (isMarkedFuseGroup(op) && !isFuseGroupOutput(op)) {
        return;
    }

    IRRewriter rewriter(context);

    int64_t availableMemoryBytes = TorqHw::get().getAvailableMemoryForTiling();

    auto tilingInterfaceOp = cast<TilingInterface>(op);
    assert(tilingInterfaceOp && "operation not implementing TilingInterface");

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(tilingInterfaceOp.getIterationDomain(rewriter));

    SmallVector<OpFoldResult> tileOffsets(iterOffsets), tileSizes(iterSizes);

    // Get the order in which we should tile the dimensions.
    SmallVector<int64_t> tilingDimOrder = getTilingDimOrder(tilingInterfaceOp);

    // Find a tile that can fit op, starting from tileSizes.
    auto tileChanged = fitTileToMemory(
        rewriter, op, {}, tilingDimOrder, availableMemoryBytes, tileOffsets, tileSizes
    );
    if (failed(tileChanged)) {
        op->emitError("failed to find a tile size for op");
        return;
    }

    // If the original tileSizes fits, there's no need to tile.
    if (!*tileChanged) {
        return;
    }

    LLVM_DEBUG({
        llvm::dbgs() << op->getName() << " (" << TORQ_FUSE_GROUP_ID << " = "
                     << getConstantIntValue(op->getAttr(TORQ_FUSE_GROUP_ID))
                     << ") needs to be tiled\n";

        llvm::dbgs() << "iteration sizes: ";
        bool isFirst = true;
        for (auto size : iterSizes) {
            llvm::dbgs() << (isFirst ? "" : "x") << getConstantIntValue(size);
            isFirst = false;
        }
        llvm::dbgs() << "\n";

        llvm::dbgs() << "iteration types: ";
        isFirst = true;
        for (auto iterType : tilingInterfaceOp.getLoopIteratorTypes()) {
            llvm::dbgs() << (isFirst ? "" : ", ") << iterType;
            isFirst = false;
        }
        llvm::dbgs() << "\n";
    });

    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        tileAndFuseToSize(rewriter, tilingInterfaceOp, iterSizes, tileSizes);
    if (failed(tiledResults)) {
        op->emitError("super tiling failed");
        return;
    }

    // Now that we know which producers were fused, find a tile that fits them too.
    tileChanged = fitTileToMemory(
        rewriter, op, tiledResults->fusedProducers, tilingDimOrder, availableMemoryBytes,
        tileOffsets, tileSizes
    );
    if (failed(tileChanged)) {
        op->emitError("failed to find a tile size for producers");
        return;
    }

    if (*tileChanged) {
        // The second fitTileToMemory returned a smaller tile.

        tiledResults = tileAndFuseToSize(rewriter, tilingInterfaceOp, iterSizes, tileSizes);
        if (failed(tiledResults)) {
            op->emitError("super tiling failed");
            return;
        }
    }

    LLVM_DEBUG({
        llvm::dbgs() << "fused " << tiledResults->fusedProducers.size() << " producers\n";
    });

    applyTiledResults(rewriter, op, *tiledResults);
}

void SuperTilingConfigPass::runOnOperation() {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    SmallVector<Operation *> orderTi;
    orderTiOps(funcOp, orderTi);

    for (auto *op : orderTi) {
        tileAndFuse(context, op);
    }

    LLVM_DEBUG({ llvm::dbgs() << "Super Tiling - DONE\n"; });
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createSuperTilingConfigPass() {
    return std::make_unique<SuperTilingConfigPass>();
}

} // namespace mlir::syna::torq
