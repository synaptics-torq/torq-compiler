// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "torq-linalg-tile"

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clDisableHost;
extern llvm::cl::opt<bool> clFallbackF32ToHost;

namespace {

static bool isMonotonicPositive(AffineExpr expr) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        return true;
    }
    else if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
        return true;
    }
    else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        return constExpr.getValue() >= 0;
    }
    else if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
        switch (binExpr.getKind()) {
        case AffineExprKind::Add:
        case AffineExprKind::Mul:
            return isMonotonicPositive(binExpr.getLHS()) && isMonotonicPositive(binExpr.getRHS());
        case AffineExprKind::FloorDiv:
        case AffineExprKind::CeilDiv:
            return isMonotonicPositive(binExpr.getLHS()) && isMonotonicPositive(binExpr.getRHS());
        case AffineExprKind::Mod:
            // mod is not monotonic
            return false;
        default:
            return false;
        }
    }
    return false;
}

static FailureOr<int> getMemoryRequirements(Value value) {

    if (auto rankedType = dyn_cast<RankedTensorType>(value.getType())) {
        if (!rankedType.hasStaticShape()) {
            return failure();
        }
        return rankedType.getNumElements() * div_ceil(rankedType.getElementTypeBitWidth(), 8);
    }
    else if (isa<FloatType, IntegerType>(value.getType())) {
        return div_ceil(value.getType().getIntOrFloatBitWidth(), 8);
    }
    else {
        return failure();
    }
}

static FailureOr<int> getMemoryRequirements(linalg::LinalgOp op) {

    int totalSize = 0;

    for (auto operand : op.getOperation()->getOperands()) {
        auto memoryRequirement = getMemoryRequirements(operand);
        if (failed(memoryRequirement)) {
            return failure();
        }
        totalSize += *memoryRequirement;
    }
    return totalSize;
}

// Memory requirements for SoftmaxOp: 2*input + output
// When decomposing the softmax on NSS, most of the ops follow the same
// input->output pattern, but two of them are torq_hl::select and
// torq_hl::ElementwiseBinary which have 2 inputs and 1 output.
// Until we convert these torq_hl ops to linalg, we have to tile the
// whole softmax accordingly.
static FailureOr<int> getMemoryRequirements(linalg::SoftmaxOp op) {
    auto inputMemReq = getMemoryRequirements(op.getInput());
    if (failed(inputMemReq)) {
        return failure();
    }
    auto outputMemReq = getMemoryRequirements(op.getOutput());
    if (failed(outputMemReq)) {
        return failure();
    }
    // 2*input + output for softmax intermediate buffers
    return 2 * (*inputMemReq) + (*outputMemReq);
}

// Estimate tile memory requirements for SoftmaxOp
static FailureOr<int>
getSoftmaxTileMemoryRequirements(linalg::SoftmaxOp op, const SmallVector<int64_t> &tile) {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    int64_t elementSize = div_ceil(inputType.getElementTypeBitWidth(), 8);

    int64_t numElements = 1;
    for (size_t i = 0; i < tile.size(); i++) {
        if (tile[i] == 0) {
            numElements *= inputType.getDimSize(i);
        }
        else {
            numElements *= tile[i];
        }
    }

    // 2*input + output for softmax
    return 3 * numElements * elementSize;
}

static FailureOr<int>
getTileMemoryRequirements(linalg::LinalgOp op, const SmallVector<int64_t> &tile) {
    auto loopRanges = op.getStaticLoopRanges();

    SmallVector<int64_t> maxIndex;

    assert(loopRanges.size() == tile.size() && "tile size must match number of loops");

    for (int i = 0; i < loopRanges.size(); i++) {
        if (tile[i] == 0) {
            maxIndex.push_back(loopRanges[i] - 1);
        }
        else {
            maxIndex.push_back(tile[i] - 1);
        }
    }

    int totalSize = 0;

    for (auto &opOperand : op.getOperation()->getOpOperands()) {

        auto map = op.getMatchingIndexingMap(&opOperand);

        for (auto expr : map.getResults()) {
            if (!isMonotonicPositive(expr)) {
                return failure();
            }
        }

        auto operandMaxIndex = map.compose(maxIndex);

        SmallVector<int64_t> shape;
        for (auto s : operandMaxIndex) {
            shape.push_back(s + 1);
        }

        int64_t elementSize;

        if (auto rankedType = dyn_cast<RankedTensorType>(opOperand.get().getType())) {
            elementSize = div_ceil(rankedType.getElementTypeBitWidth(), 8);
        }
        else {
            elementSize = div_ceil(opOperand.get().getType().getIntOrFloatBitWidth(), 8);
        }

        int64_t numElements = 1;
        for (auto s : shape) {
            numElements *= s;
        }

        totalSize += numElements * elementSize;
    }

    return totalSize;
}

// find a tiling vector that used to tile srcOp ensure less than maxSize memory is required
// to store the data, this function will always tile at least once
// the tile found is not guaranteed to be the largest one (i.e. the optimal one)
static FailureOr<SmallVector<int64_t>> findValidParallelTile(linalg::LinalgOp srcOp, int maxSize) {

    // FIXME: here we should create a proper integer linear programming problem
    // and solve it for the largest tile that fits in maxSize, for the moment we
    // use this heuristic

    if (srcOp.getNumParallelLoops() < 1) {
        return failure();
    }

    SmallVector<int64_t> tileVector(srcOp.getNumLoops(), 0);
    SmallVector<unsigned> parallelDims;
    srcOp.getParallelDims(parallelDims);

    auto dimSize = srcOp.getStaticLoopRanges();

    // do not try to tile the operation if any of the dimensions is dynamic
    for (auto s : dimSize) {
        if (s == ShapedType::kDynamic) {
            return failure();
        }
    }

    // start tiling by assuming we keep the whole computation in a single tile
    for (int i = 0; i < tileVector.size(); i++) {

        // do not tile dimension that are :
        // - shape 1
        // - not parallel
        if (dimSize[i] == 1 || srcOp.getIteratorTypesArray()[i] != utils::IteratorType::parallel) {
            tileVector[i] = 0;
        }
        // if the dimension is parallel, try to keep the dimension in the tile
        else {
            tileVector[i] = dimSize[i];
        }
    }

    // try to tile more until we fit in memory
    while (true) {

        // try to find a new tilingVector that computes
        // smaller tiles of data, the hope is that we can
        // tile the whole computation in few tiles and the
        // required memory will be smaller than the maxSize
        bool found = false;
        for (int idx = 0; idx < tileVector.size(); idx++) {
            // check if the dimension is parallel
            if (tileVector[idx] == 0) {
                continue;
            }

            // check if the dimension can be tiled
            if (tileVector[idx] > 1) {

                // try to half the tile size
                tileVector[idx] = std::max(tileVector[idx] / 2, int64_t(1));
                found = true;
                break;
            }
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Tiling vector: [";
            for (auto size : tileVector) {
                llvm::dbgs() << size << ", ";
            }
            llvm::dbgs() << "]\n";
        });

        if (!found) {
            LLVM_DEBUG({
                llvm::dbgs(
                ) << "Failed to find a valid tile for op (cannot make a tile small enough): "
                  << srcOp << "\n";
            });
            return failure();
        }

        auto maybeMemReq = getTileMemoryRequirements(srcOp, tileVector);

        if (failed(maybeMemReq)) {
            LLVM_DEBUG({ llvm::dbgs() << "Failed to compute memory requirements\n"; });
            return failure();
        }

        LLVM_DEBUG({ llvm::dbgs() << "Estimated memory requirements: " << *maybeMemReq << "\n"; });

        if (*maybeMemReq <= maxSize) {
            return tileVector;
        }
    }

    return failure();
}

// Find a valid tile for SoftmaxOp that fits in maxSize memory
// Does not tile the reduction dimension (softmax dimension)
static FailureOr<SmallVector<int64_t>> findValidSoftmaxTile(linalg::SoftmaxOp op, int maxSize) {
    auto inputType = cast<RankedTensorType>(op.getInput().getType());
    int64_t rank = inputType.getRank();
    int64_t reductionDim = op.getDimension();

    // Check for static shape
    if (!inputType.hasStaticShape()) {
        return failure();
    }

    SmallVector<int64_t> tileVector(rank, 0);

    // Initialize tile vector - don't tile reduction dimension
    for (int64_t i = 0; i < rank; i++) {
        if (i == reductionDim || inputType.getDimSize(i) == 1) {
            tileVector[i] = 0; // Don't tile this dimension
        }
        else {
            tileVector[i] = inputType.getDimSize(i);
        }
    }

    // Check if we have any parallel dimensions to tile
    bool hasParallelDim = false;
    for (int64_t i = 0; i < rank; i++) {
        if (tileVector[i] > 0) {
            hasParallelDim = true;
            break;
        }
    }
    if (!hasParallelDim) {
        return failure();
    }

    // Try to find a tile that fits in memory
    while (true) {
        bool found = false;
        for (int64_t idx = 0; idx < rank; idx++) {
            if (tileVector[idx] <= 1) {
                continue;
            }
            // Halve the tile size
            tileVector[idx] = std::max(tileVector[idx] / 2, int64_t(1));
            found = true;
            break;
        }

        LLVM_DEBUG({
            llvm::dbgs() << "Softmax tiling vector: [";
            for (auto size : tileVector) {
                llvm::dbgs() << size << ", ";
            }
            llvm::dbgs() << "]\n";
        });

        if (!found) {
            LLVM_DEBUG({
                llvm::dbgs(
                ) << "Failed to find valid tile for softmax (cannot make tile small enough)\n";
            });
            return failure();
        }

        auto maybeMemReq = getSoftmaxTileMemoryRequirements(op, tileVector);
        if (failed(maybeMemReq)) {
            return failure();
        }

        LLVM_DEBUG({ llvm::dbgs() << "Softmax estimated memory: " << *maybeMemReq << "\n"; });

        if (*maybeMemReq <= maxSize) {
            return tileVector;
        }
    }

    return failure();
}

static LogicalResult tileMatMulForSlices(
    linalg::LinalgOp srcOp, PatternRewriter &rewriter, SmallVector<OpFoldResult> tileSizes
) {

    if (isa<linalg::BatchMatmulOp>(srcOp)) {
        // TODO: before we change kernel to support batch matmul, we just
        // tile batch matmul with batch size 1 to support batch > 1 case
        tileSizes[0] = rewriter.getIndexAttr(1);
    }

    // matmul op will automatically inserts linalg.fill op before it for output init
    // we need to tile this linalg.fill together with matmul op in case unmatched
    // tile sizes for linalg.fill and matmul op
    auto options =
        scf::SCFTileAndFuseOptions().setTilingOptions(scf::SCFTilingOptions().setTileSizes(tileSizes
        ));

    // Avoid fusing operations that do not run on the slice
    auto fusionControlFn = [](tensor::ExtractSliceOp, OpResult opResult,
                              bool) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
        // If producer memory requirement is less than maxAllowedMemory, do not fuse
        // use 10kB as threshold for now but we should probably calculate it based on MatMul
        const uint32_t maxAllowedMemory = 10000;
        auto maybeMemoryRequirement = getMemoryRequirements(opResult);
        if (succeeded(maybeMemoryRequirement) && (*maybeMemoryRequirement < maxAllowedMemory)) {
            return std::nullopt;
        }
        auto srcOp = opResult.getDefiningOp<linalg::LinalgOp>();
        if (!srcOp) {
            return std::nullopt;
        }
        if (getTargetExecutor(srcOp, torq_hl::Executor::Slice) != torq_hl::Executor::Slice)
            return std::nullopt;
        return scf::SCFTileAndFuseOptions::ControlFnResult{false};
    };
    options.setFusionControlFn(fusionControlFn);

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(srcOp.getOperation()), options
        );
    assert(succeeded(tileAndFuseResult));
    rewriter.replaceOp(srcOp, tileAndFuseResult->replacements[srcOp->getResult(0)]);

    // peel the loops so that all operations are fixed size
    SmallVector<scf::ForOp> forOps;
    for (auto loop : tileAndFuseResult->loops) {
        forOps.push_back(cast<scf::ForOp>(loop));
    }

    linalg::peelLoops(rewriter, forOps);

    return success();
}

static LogicalResult parallelTileAndPeel(
    linalg::LinalgOp srcOp, PatternRewriter &rewriter, int maxSize,
    std::optional<torq_hl::Executor> targetExecutor
) {

    auto maybeTileVector = findValidParallelTile(srcOp, maxSize);

    if (failed(maybeTileVector)) {
        return failure();
    }

    // create an effective tiling vector where only necessary dimensions are tiled
    SmallVector<int64_t> effectiveTileVector(maybeTileVector->size());
    auto dimSize = srcOp.getStaticLoopRanges();

    for (auto i = 0; i < effectiveTileVector.size(); i++) {

        // mark the tiles that match the dimension with 0 to indicate we don't want to tile them
        // this avoids generating useless for loops
        if (dimSize[i] == (*maybeTileVector)[i]) {
            effectiveTileVector[i] = 0;
        }
        else {
            effectiveTileVector[i] = (*maybeTileVector)[i];
        }
    }

    // create the tile sizes vector used for tiling
    SmallVector<OpFoldResult> tileSizes =
        getAsIndexOpFoldResult(rewriter.getContext(), effectiveTileVector);
    auto options = scf::SCFTilingOptions().setTileSizes(tileSizes);

    // FIXME: hack to make matmul and batch work with the current kernels
    // this can be removed once we start using super tiling
    if (targetExecutor.has_value() && targetExecutor.value() == torq_hl::Executor::Slice &&
        (isa<linalg::BatchMatmulOp>(srcOp) || isa<linalg::MatmulOp>(srcOp))) {
        return tileMatMulForSlices(srcOp, rewriter, tileSizes);
    }

    // tile the operation
    auto tilingInterfaceOp = cast<TilingInterface>(srcOp.getOperation());

    FailureOr<scf::SCFTilingResult> maybeTileResult =
        scf::tileUsingSCF(rewriter, tilingInterfaceOp, options);

    if (failed(maybeTileResult)) {
        LLVM_DEBUG({ llvm::dbgs() << "Failed to tile op: " << srcOp << "\n"; });
        return failure();
    }

    rewriter.replaceOp(srcOp, maybeTileResult->replacements);

    // peel the loops so that all operations are fixed size
    SmallVector<scf::ForOp> forOps;
    for (auto loop : maybeTileResult->loops) {
        forOps.push_back(cast<scf::ForOp>(loop));
    }

    linalg::peelLoops(rewriter, forOps);

    LLVM_DEBUG({ llvm::dbgs() << "Finished\n"; });

    return success();
}

// mlir::iree_compiler::IREE::LinalgExt::SortOp implements TilingInterface,
// but tiling the sort dimension produces independently sorted sub-arrays, not a
// globally sorted result. There is no merge step to combine tiles. So we
// cannot use tiling to split a large sort into smaller CSS pieces.
// Instead, we check if the op fits in CSS memory (accounting for unused outputs
// that create memref.allocas) and fall back to Host execution when it doesn't.
class CheckSortOpMemory : public OpRewritePattern<mlir::iree_compiler::IREE::LinalgExt::SortOp> {
  public:
    CheckSortOpMemory(
        MLIRContext *context, int maxSize,
        std::optional<torq_hl::Executor> targetExecutor = std::nullopt
    )
        : OpRewritePattern<mlir::iree_compiler::IREE::LinalgExt::SortOp>(context), maxSize(maxSize),
          targetExecutor(targetExecutor) {}

    LogicalResult matchAndRewrite(
        mlir::iree_compiler::IREE::LinalgExt::SortOp sortOp, PatternRewriter &rewriter
    ) const override {

        if (targetExecutor.has_value() &&
            getTargetExecutor(sortOp, *targetExecutor) != *targetExecutor) {
            LLVM_DEBUG({
                llvm::dbgs() << "CheckSortOpMemory: skipping sort op with executor "
                             << stringifyExecutor(getTargetExecutor(sortOp)) << "\n";
            });
            return failure();
        }

        int totalSize = 0;
        int unusedAllocaSize = 0;
        for (auto result : sortOp.getResults()) {
            auto rankedType = dyn_cast<RankedTensorType>(result.getType());
            if (!rankedType || !rankedType.hasStaticShape()) {
                LLVM_DEBUG({
                    llvm::dbgs() << "CheckSortOpMemory: dynamic shape, falling back to Host\n";
                });
                // Cannot estimate memory: fall back to Host
                rewriter.modifyOpInPlace(sortOp, [&]() {
                    setTargetExecutorAttr(sortOp, torq_hl::Executor::Host);
                });
                return success();
            }
            int64_t resultSize =
                rankedType.getNumElements() * div_ceil(rankedType.getElementTypeBitWidth(), 8);
            totalSize += resultSize;
            if (result.use_empty()) {
                unusedAllocaSize += resultSize;
            }
        }

        LLVM_DEBUG({
            llvm::dbgs() << "CheckSortOpMemory: sort op total size=" << totalSize
                         << " unusedAllocaSize=" << unusedAllocaSize << " maxSize=" << maxSize
                         << " cssStackSize=" << HwInfo::css_stack_size << "\n";
        });

        // Unused outputs of DPS sort ops create memref.allocas on the CSS stack.
        // If any unused output exceeds the CSS stack size, the op cannot run on CSS.
        if (unusedAllocaSize > HwInfo::css_stack_size) {
            LLVM_DEBUG({
                llvm::dbgs(
                ) << "CheckSortOpMemory: unused outputs exceed CSS stack, falling back to Host\n";
            });
            rewriter.modifyOpInPlace(sortOp, [&]() {
                setTargetExecutorAttr(sortOp, torq_hl::Executor::Host);
            });
            return success();
        }

        // Also check total DTCM budget
        if (totalSize <= maxSize) {
            return rewriter.notifyMatchFailure(sortOp, "Sort op fits in CSS memory");
        }

        // Exceeds CSS DTCM budget: fall back to Host
        LLVM_DEBUG({
            llvm::dbgs() << "CheckSortOpMemory: exceeds CSS DTCM memory, falling back to Host\n";
        });
        rewriter.modifyOpInPlace(sortOp, [&]() {
            setTargetExecutorAttr(sortOp, torq_hl::Executor::Host);
        });
        return success();
    }

  private:
    int maxSize;
    std::optional<torq_hl::Executor> targetExecutor;
};

class TileLinalgOpOperation : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  public:
    TileLinalgOpOperation(
        MLIRContext *context, int maxTileSize,
        std::optional<torq_hl::Executor> targetExecutor = std::nullopt
    )
        : OpInterfaceRewritePattern<linalg::LinalgOp>(context), maxTileSize(maxTileSize),
          targetExecutor(targetExecutor) {}

    LogicalResult
    matchAndRewrite(linalg::LinalgOp srcOp, PatternRewriter &rewriter) const override {

        if (targetExecutor.has_value() &&
            getTargetExecutor(srcOp, *targetExecutor) != *targetExecutor) {
            return failure();
        }

        if (clFallbackF32ToHost && !isa<linalg::TransposeOp>(srcOp) &&
            !isa<linalg::FillOp>(srcOp)) {

            if (getTargetExecutor(srcOp) == torq_hl::Executor::Host) {
                return failure();
            }
        }

        // If this is an LRAM run we shouldn't need to tile anything again here.
        if (targetExecutor.has_value() && *targetExecutor == torq_hl::Executor::Slice)
            return rewriter.notifyMatchFailure(srcOp, "No need to tile this operation");

        auto maybeDataSize = getMemoryRequirements(srcOp);

        if (failed(maybeDataSize)) {

            if (clDisableHost) {
                return srcOp->emitError()
                       << "unable to find memory requirements for op and cannot fall back to host";
            }

            // since we cannot know the memory requirements we can't run this on the NPU
            rewriter.modifyOpInPlace(srcOp, [&]() {
                setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
            });

            return success();
        }

        if (*maybeDataSize <= maxTileSize) {
            return rewriter.notifyMatchFailure(srcOp, "No need to tile this operation");
        }

        // tile the operation to fit it in maxTileSize
        auto status = parallelTileAndPeel(srcOp, rewriter, maxTileSize, targetExecutor);

        if (failed(status)) {

            if (clDisableHost) {
                return srcOp->emitError()
                       << "unable to tile operation and cannot fall back to host";
            }

            // since we cannot tile this we can't run it on the NPU
            rewriter.modifyOpInPlace(srcOp, [&]() {
                setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
            });
        }

        return success();
    }

  private:
    int maxTileSize;
    std::optional<torq_hl::Executor> targetExecutor;
};

template <typename OpTy> struct TensorOpConversion : public OpRewritePattern<OpTy> {
    using OpRewritePattern<OpTy>::OpRewritePattern;

    LogicalResult matchAndRewrite(OpTy tensorOp, PatternRewriter &rewriter) const override {

        if (getTargetExecutor(tensorOp, torq_hl::Executor::CSS) != torq_hl::Executor::CSS) {
            return failure();
        }

        return static_cast<LogicalResult>(
            linalg::rewriteInDestinationPassingStyle(rewriter, tensorOp)
        );
    }
};

/// Tile reduction dimensions of linalg.generic ops whose reduction input
/// exceeds LRAM.
class TileReductionForLramPass : public impl::TileReductionForLramBase<TileReductionForLramPass> {
  public:
    using TileReductionForLramBase<TileReductionForLramPass>::TileReductionForLramBase;

    void runOnOperation() {
        auto funcOp = getOperation();
        const uint32_t lramSize = TorqHw::get().getAvailableMemoryForTiling();

        SmallVector<linalg::GenericOp> opsToTile;
        funcOp.walk([&](linalg::GenericOp genericOp) {
            if (genericOp.getNumReductionLoops() == 0)
                return;

            // TODO: Handle ops with multiple reduction dims
            SmallVector<unsigned> reductionDims;
            genericOp.getReductionDims(reductionDims);
            if (reductionDims.size() != 1) {
                LLVM_DEBUG({
                    llvm::dbgs() << "TileReductionForLram: only one reduction dimension is "
                                    "supported, but found "
                                 << reductionDims.size() << "; aborting.\n";
                });
                return;
            }

            FailureOr<int64_t> totalBytes = estimateOperandBytes(genericOp);
            if (failed(totalBytes)) {
                LLVM_DEBUG({
                    llvm::dbgs(
                    ) << "TileReductionForLram: skipping op because operand memory could "
                         "not be estimated\n";
                });
                return;
            }

            if (*totalBytes > lramSize) {
                LLVM_DEBUG({
                    llvm::dbgs(
                    ) << "TileReductionForLram: selected op for reduction tiling; totalBytes="
                      << *totalBytes << " lramSize=" << lramSize << "\n";
                });
                opsToTile.push_back(genericOp);
            }
        });

        IRRewriter rewriter(&getContext());
        for (linalg::GenericOp genericOp : opsToTile) {
            if (failed(tileReductionToFitLram(rewriter, genericOp, lramSize))) {
                genericOp.emitError() << "failed to tile reduction dimension to fit LRAM size "
                                      << lramSize << " bytes";
                signalPassFailure();
                return;
            }
        }
    }

  private:
    FailureOr<int64_t> estimateOperandBytes(linalg::GenericOp genericOp) {
        int64_t totalBytes = 0;
        for (OpOperand &operand : genericOp->getOpOperands()) {
            auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
            if (!tensorType || !tensorType.hasStaticShape())
                return failure();

            int64_t elementBytes = llvm::divideCeil(tensorType.getElementTypeBitWidth(), 8);
            totalBytes += tensorType.getNumElements() * elementBytes;
        }
        return totalBytes;
    }

    FailureOr<int64_t> estimateTiledOperandBytes(
        linalg::GenericOp genericOp, unsigned reductionDim, int64_t tileSize
    ) {
        int64_t estimatedBytes = 0;

        for (OpOperand &operand : genericOp->getOpOperands()) {
            auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
            if (!tensorType || !tensorType.hasStaticShape()) {
                LLVM_DEBUG({
                    llvm::dbgs(
                    ) << "TileReductionForLram: cannot estimate tiled bytes for non-static "
                         "ranked tensor operand\n";
                });
                return failure();
            }

            AffineMap map = genericOp.getMatchingIndexingMap(&operand);
            int64_t numElements = 1;
            for (auto [i, expr] : llvm::enumerate(map.getResults())) {
                int64_t dimSize = tensorType.getDimSize(i);
                if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
                    if (dimExpr.getPosition() == reductionDim) {
                        numElements *= tileSize;
                        continue;
                    }
                }
                numElements *= dimSize;
            }
            int64_t elementBytes = llvm::divideCeil(tensorType.getElementTypeBitWidth(), 8);
            estimatedBytes += numElements * elementBytes;
        }
        return estimatedBytes;
    }

    LogicalResult
    tileReductionToFitLram(IRRewriter &rewriter, linalg::GenericOp genericOp, uint32_t lramSize) {
        SmallVector<unsigned> reductionDims;
        genericOp.getReductionDims(reductionDims);
        assert(reductionDims.size() == 1);
        unsigned reductionDim = reductionDims[0];

        auto loopRanges = genericOp.getStaticLoopRanges();
        int64_t reductionSize = loopRanges[reductionDim];
        if (reductionSize == ShapedType::kDynamic) {
            LLVM_DEBUG({
                llvm::dbgs() << "TileReductionForLram: cannot tile op because reduction dimension "
                             << reductionDim << " is dynamic\n";
            });
            return failure();
        }
        if (reductionSize <= 1)
            return failure();

        // The reduce write descriptor (CEPW n_size) can encode at most 1<<16 elements per
        // reduction, so a single tile's reduction length must not exceed this regardless of how
        // much LRAM is available. Reference:
        // third_party/torq-hw/ct/torq_api.c:L1444: _reg_ndl_desc_gen: Assertion `n>=1 &&
        // n<=(1<<16)' failed
        constexpr int64_t kMaxReduceTileElements = 1 << 16;

        // Repeatedly halve the reduction tile size until the estimated memory requirement fits in
        // LRAM and the tile is within the hardware reduce descriptor limit.
        int64_t tileSize = reductionSize;
        bool fitsInLram = false;

        while (tileSize > 1) {
            tileSize = std::max(tileSize / 2, int64_t(1));

            FailureOr<int64_t> estimatedBytes =
                estimateTiledOperandBytes(genericOp, reductionDim, tileSize);
            if (failed(estimatedBytes))
                return failure();

            LLVM_DEBUG({
                llvm::dbgs() << "TileReductionForLram: candidate tileSize=" << tileSize
                             << " estimatedBytes=" << *estimatedBytes << " lramSize=" << lramSize
                             << "\n";
            });

            if (*estimatedBytes <= lramSize && tileSize <= kMaxReduceTileElements) {
                fitsInLram = true;
                break;
            }
        }

        if (!fitsInLram) {
            LLVM_DEBUG({
                llvm::dbgs() << "TileReductionForLram: failed to find reduction tile that fits "
                                "in LRAM\n";
            });
            return failure();
        }

        unsigned numLoops = genericOp.getNumLoops();
        SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));
        tileSizes[reductionDim] = rewriter.getIndexAttr(tileSize);

        LLVM_DEBUG({
            llvm::dbgs() << "TileReductionForLram: tiling reductionDim=" << reductionDim
                         << " reductionSize=" << reductionSize << " tileSize=" << tileSize << "\n";
        });

        // Create a loop to accumulate partial results into the same output buffer.
        scf::SCFTilingOptions options;
        options.setTileSizes(tileSizes);
        options.setReductionTilingStrategy(ReductionTilingStrategy::FullReduction);
        options.setReductionDims({reductionDim});

        rewriter.setInsertionPoint(genericOp);
        FailureOr<scf::SCFTilingResult> tilingResult =
            scf::tileUsingSCF(rewriter, cast<TilingInterface>(genericOp.getOperation()), options);
        if (failed(tilingResult)) {
            LLVM_DEBUG({ llvm::dbgs() << "TileReductionForLram: scf::tileUsingSCF failed\n"; });
            return failure();
        }

        SmallVector<scf::ForOp> forOps;
        for (auto loop : tilingResult->loops) {
            if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
                forOps.push_back(forOp);
        }

        linalg::peelLoops(rewriter, forOps);
        rewriter.replaceOp(genericOp, tilingResult->replacements);
        return success();
    }
};

class LramTilePass : public impl::LramTileBase<LramTilePass> {
  public:
    using LramTileBase<LramTilePass>::LramTileBase;

    void runOnOperation() {
        auto funcOp = getOperation();

        MLIRContext *ctx = funcOp.getContext();

        RewritePatternSet tilePatterns(ctx);

        const uint32_t lramSize = TorqHw::get().getAvailableMemoryForTiling();

        // Add tiling patterns for the Slice executor
        tilePatterns.add<TileLinalgOpOperation>(ctx, lramSize, torq_hl::Executor::Slice);

        tensor::ControlConstantExtractSliceFusionFn controlFn = [](tensor::ExtractSliceOp op) {
            return true;
        };
        tensor::populateFoldConstantExtractSlicePatterns(tilePatterns, controlFn);

        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps);
        if (failed(applyPatternsGreedily(getOperation(), std::move(tilePatterns), config))) {
            return signalPassFailure();
        }

        return;
    }
};

class DtcmTilePass : public impl::DtcmTileBase<DtcmTilePass> {
  public:
    using DtcmTileBase<DtcmTilePass>::DtcmTileBase;

    void runOnOperation() {
        MLIRContext *ctx = &getContext();

        RewritePatternSet convertToLinalgPatterns(ctx);

        // to make tiling of tensor.pad/generate possible we convert them to linalg ops first
        convertToLinalgPatterns.add<TensorOpConversion<tensor::PadOp>>(ctx);
        convertToLinalgPatterns.add<TensorOpConversion<tensor::GenerateOp>>(ctx);

        if (failed(applyPatternsGreedily(getOperation(), std::move(convertToLinalgPatterns)))) {
            return signalPassFailure();
        }

        RewritePatternSet tilePatterns(ctx);

        const uint32_t cssMem = HwInfo::dtcm_size - HwInfo::css_stack_size - 32;
        tilePatterns.add<TileLinalgOpOperation>(ctx, cssMem, torq_hl::Executor::CSS);
        tilePatterns.add<CheckSortOpMemory>(ctx, cssMem, torq_hl::Executor::CSS);

        GreedyRewriteConfig config;
        config.setStrictness(GreedyRewriteStrictness::ExistingOps);
        if (failed(applyPatternsGreedily(getOperation(), std::move(tilePatterns), config))) {
            return signalPassFailure();
        }

        return;
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLramTilePass() {
    return std::make_unique<LramTilePass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileReductionForLramPass() {
    return std::make_unique<TileReductionForLramPass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDtcmTilePass() {
    return std::make_unique<DtcmTilePass>();
}

} // namespace mlir::syna::torq
