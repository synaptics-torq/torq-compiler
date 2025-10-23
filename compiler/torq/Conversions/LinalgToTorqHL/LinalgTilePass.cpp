// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-linalg-tile"

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clDisableHost;

llvm::cl::opt<bool> clFallbackF32ToHost(
    "torq-fallback-f32-to-host",
    llvm::cl::desc("Fallback to host execution of any operation that uses f32"),
    llvm::cl::init(true)
);

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

static FailureOr<int> getMemoryRequirements(linalg::LinalgOp op) {

    int totalSize = 0;

    for (auto operand : op.getOperation()->getOperands()) {

        if (auto rankedType = dyn_cast<RankedTensorType>(operand.getType())) {
            if (!rankedType.hasStaticShape()) {
                return failure();
            }
            totalSize +=
                rankedType.getNumElements() * div_ceil(rankedType.getElementTypeBitWidth(), 8);
        }
        else if (isa<FloatType, IntegerType>(operand.getType())) {
            totalSize += div_ceil(operand.getType().getIntOrFloatBitWidth(), 8);
        }
        else {
            return failure();
        }
    }

    return totalSize;
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

        if (*maybeMemReq < maxSize) {
            return tileVector;
        }
    }

    return failure();
}

static bool isIm2ColOp(linalg::LinalgOp op) {
    // check if op has attribute marking it as im2col
    return op && op->hasAttr("torq.im2col") &&
           cast<BoolAttr>(op->getAttr("torq.im2col")).getValue();
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

    // Avoid fusing img2col operations as we want to run them on the host in one go
    auto fusionControlFn = [](tensor::ExtractSliceOp, OpResult op, bool) {
        bool fuse = !isIm2ColOp(op.getDefiningOp<linalg::LinalgOp>());
        return std::make_tuple(fuse, false);
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

#if 0
class TileReduceOperation : public OpRewritePattern<linalg::LinalgOp> {
  public:
    TileReduceOperation(MLIRContext *context, torq_hl::SubSystem subSystem)
        : OpRewritePattern<linalg::LinalgOp>(context), subSystem(subSystem) {}
    
    LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp, PatternRewriter &rewriter) const override {
            if (linalgOp.getNumReductionLoops() < 1) {
        return failure();
    }

    SmallVector<int64_t> tileSizes(linalgOp.getNumLoops(), 0);

    SmallVector<unsigned> reductionDims;
    linalgOp.getReductionDims(reductionDims);

    for (int i = 0; i < reductionDims.size(); ++i) {
        if (reductionDims[reductionDims.size() - 1 - i] != linalgOp.getNumLoops() - 1 - i) {
            return rewriter.notifyMatchFailure(linalgOp, "reductionDims not the innermost ones");
        }
    }

    if (reductionDims.size() == linalgOp.getNumReductionLoops()) {
        for (unsigned i = 0; i < linalgOp.getNumReductionLoops(); i++) {
            tileSizes[reductionDims[i]] = 64;
        }
    }

    if (!tileSizes.empty()) {
        SmallVector<OpFoldResult> sizes;
        for (int64_t size : tileSizes) {
            sizes.push_back(rewriter.getIndexAttr(size));
        }
        FailureOr<scf::SCFReductionTilingResult> results = scf::tileReductionUsingScf(
            rewriter, cast<PartialReductionOpInterface>(linalgOp.getOperation()), sizes
        );
        if (failed(results)) {
            return failure();
        }
    }

    return success();
}
#endif

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

        // fallback any sqrt operation to host because we don't know how to compile this yet for CSS
        auto ret = srcOp.walk([](Operation *op) {
            if (isa<math::SqrtOp, math::RsqrtOp, math::PowFOp, math::ErfOp, arith::DivFOp>(op)) {
                return WalkResult::interrupt();
            }

            return WalkResult::advance();
        });

        if (ret.wasInterrupted()) {
            rewriter.modifyOpInPlace(srcOp, [&]() {
                setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
            });

            return success();
        }

        if (clFallbackF32ToHost && !isa<linalg::TransposeOp>(srcOp) &&
            !isa<linalg::FillOp>(srcOp)) {

            // FIXME: this is a quick workaround to avoid tiling operations that we want on the host
            auto isFp32Op = false;

            if (isIm2ColOp(srcOp)) {
                setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
                return success();
            }

            for (auto operand : srcOp->getOperands()) {
                if (auto rankedType = dyn_cast<RankedTensorType>(operand.getType())) {
                    if (rankedType.getElementType().isF32()) {
                        isFp32Op = true;
                        break;
                    }
                }
                else if (operand.getType().isF32()) {
                    isFp32Op = true;
                    break;
                }
            }

            // force fp32 ops to run on host and do not tile them
            if (isFp32Op && !clDisableHost) {

                std::string failReason;
                std::string opName;
                int32_t minIntValue = 0;
                int32_t maxIntValue = 0;
                float minFloatValue = 0.0f;
                float maxFloatValue = 0.0f;

                bool canExecuteOnTorq = false;

                // check if the operation can be lowered to a torq kernel
                if (isTorqCastOp(srcOp, opName, failReason)) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqNegateOp(srcOp, failReason)) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqAbsOp(srcOp, failReason)) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqCeilOp(srcOp, failReason)) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqClampOp(
                             srcOp, minIntValue, maxIntValue, minFloatValue, maxFloatValue,
                             failReason
                         )) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqFloorOp(srcOp, failReason)) {
                    canExecuteOnTorq = true;
                }
                else if (isTorqMatMul(srcOp, failReason)) {
                    canExecuteOnTorq = true;
                }

                // if not possible we want to run it on the host
                if (!canExecuteOnTorq) {
                    rewriter.modifyOpInPlace(srcOp, [&]() {
                        setTargetExecutorAttr(srcOp, torq_hl::Executor::Host);
                    });

                    return success();
                }
            }
        }

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

        if (*maybeDataSize < maxTileSize) {
            return rewriter.notifyMatchFailure(srcOp, "No need to tile this operation");
        }

        // tile the operation to fit it in DTCM
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

struct TensorPadOpConversion : public OpRewritePattern<tensor::PadOp> {
    using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(tensor::PadOp padTensorOp, PatternRewriter &rewriter) const override {

        if (getTargetExecutor(padTensorOp, torq_hl::Executor::CSS) != torq_hl::Executor::CSS) {
            return failure();
        }

        return static_cast<LogicalResult>(
            linalg::rewriteInDestinationPassingStyle(rewriter, padTensorOp)
        );
    }
};

class LramTilePass : public LramTileBase<LramTilePass> {
  public:
    using LramTileBase<LramTilePass>::LramTileBase;

    void runOnOperation() {
        auto funcOp = getOperation();

        MLIRContext *ctx = funcOp.getContext();

        RewritePatternSet tilePatterns(ctx);

        const uint32_t lramSize = TorqHw::get().getAvailableMemoryForTiling();

        tilePatterns.add<TileLinalgOpOperation>(ctx, lramSize, torq_hl::Executor::Slice);

        tensor::ControlConstantExtractSliceFusionFn controlFn = [](tensor::ExtractSliceOp op) {
            return true;
        };
        tensor::populateFoldConstantExtractSlicePatterns(tilePatterns, controlFn);

        GreedyRewriteConfig config;
        config.strictMode = GreedyRewriteStrictness::ExistingOps;
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(tilePatterns), config))) {
            return signalPassFailure();
        }

        return;
    }
};

class DtcmTilePass : public DtcmTileBase<DtcmTilePass> {
  public:
    using DtcmTileBase<DtcmTilePass>::DtcmTileBase;

    void runOnOperation() {
        auto funcOp = getOperation();

        MLIRContext *ctx = funcOp.getContext();

        RewritePatternSet tilePatterns(ctx);

        // to make tiling of tensor.pad possible we convert them to linalg ops first
        tilePatterns.add<TensorPadOpConversion>(ctx);

        const uint32_t cssMem = HwInfo::dtcm_size - HwInfo::css_stack_size;
        tilePatterns.add<TileLinalgOpOperation>(ctx, cssMem, torq_hl::Executor::CSS);

        tensor::ControlConstantExtractSliceFusionFn controlFn = [](tensor::ExtractSliceOp op) {
            return true;
        };
        tensor::populateFoldConstantExtractSlicePatterns(tilePatterns, controlFn);

        GreedyRewriteConfig config;
        config.strictMode = GreedyRewriteStrictness::ExistingOps;
        if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(tilePatterns), config))) {
            return signalPassFailure();
        }

        return;
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLramTilePass() {
    return std::make_unique<LramTilePass>();
}

std::unique_ptr<InterfacePass<FunctionOpInterface>> createDtcmTilePass() {
    return std::make_unique<DtcmTilePass>();
}

} // namespace mlir::syna::torq
