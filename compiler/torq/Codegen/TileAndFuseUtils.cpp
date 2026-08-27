// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TileAndFuseUtils.h"

#include "TilingUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#define DEBUG_TYPE "torq-tile-and-fuse-utils"

namespace mlir::syna::torq {

int64_t getSmallestTileSize(const TilingInfo &tilingInfo, int64_t domain, int64_t domainSize) {
    for (int64_t size = 1; size < domainSize; ++size) {
        if (tilingInfo.adjustSize[domain](size) == size)
            return size;
    }
    return domainSize;
}

llvm::FailureOr<int64_t> computeSizeAtFirstIteration(OpFoldResult sizeFoldResult) {
    if (std::optional<int64_t> constSize = getConstantIntValue(sizeFoldResult))
        return *constSize;

    assert(isa<Value>(sizeFoldResult) && "expected a Value");
    Value sizeValue = cast<Value>(sizeFoldResult);

    llvm::DenseMap<Value, Attribute> computedValuse;
    auto result = computeValueAtFirstIteration(sizeValue, computedValuse);
    if (failed(result)) {
        sizeValue.getDefiningOp()->emitWarning(
            "can't compute producers tile size (unexpected operation type; "
            "expected affine.min/max/apply)"
        );
        LLVM_DEBUG(assert(false && "unexpected operation"));
        return llvm::failure();
    }

    return cast<IntegerAttr>(*result).getInt();
}

// Compute the order in which fitTileToMemory's shrink pass should cut domains:
// greedily by the number of operand bytes each shrink removes across the whole
// fuse group (consumer + producers), ties broken by the legacy tilingOrder.
//
// Rationale: shrinking a domain that does not index the dominant operand buys
// almost nothing but gets committed anyway, forcing the remaining domains into
// much smaller tiles during grow-back. For a matmul the M dim does not index
// the KxN weight operand, so shrinking M first is nearly free of benefit and
// caps the N regrowth — observed on bert's FFN batch_matmul [1,128,768]x[768,
// 3072]: legacy order produced a [M=7, N=308] tile (190 tiles, ~86MB of weight
// DMA for a 4.5MiB matrix, ~18x re-streaming) instead of [M=128, N~160]
// (~20 tiles, weights read ~once).
//
// Correspondence between a group op's iteration domains and the consumer's
// tileable domains is established by dimension-extent equality. If anything is
// ambiguous or non-static (duplicate extents among tileable domains, an op
// domain matching more than one candidate, non-dim map results), we return
// nullopt and the caller keeps the legacy order. Convs bail out here naturally
// (their reduction dims share extents with parallel dims), so their tile
// shapes are unchanged.
// Named matmuls plus the generics replaceBatchMatmulWithBroadcastGeneric marks:
// matmuls to every tiling heuristic, but invisible to isa<>.
static bool isMatmulLikeOp(Operation *op) {
    return isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ContractOp>(op) ||
           op->hasAttr(TORQ_BROADCAST_MATMUL);
}

std::optional<SmallVector<int64_t>> computeShrinkOrderByReduction(
    Operation *consumerOp, const SetVector<Operation *> &producerOps, const TilingInfo &tilingInfo,
    ArrayRef<int64_t> iterDomainSizes, ArrayRef<OpFoldResult> sizes, bool fallback
) {
    // Scope the reorder strictly to matmul-family groups — the case it was
    // designed for. For all other ops the extent-matching heuristic is not
    // worth the risk; keep the legacy order.
    bool hasMatmul = isMatmulLikeOp(consumerOp);
    if (!hasMatmul) {
        for (Operation *producer : producerOps) {
            if (isMatmulLikeOp(producer)) {
                hasMatmul = true;
                break;
            }
        }
    }
    LLVM_DEBUG(
        llvm::dbgs() << "shrink-order: hasMatmul=" << hasMatmul << " for " << consumerOp->getName()
                     << "\n"
    );
    if (!hasMatmul)
        return std::nullopt;

    // Map full extent -> tileable consumer domain; bail on duplicate extents.
    DenseMap<int64_t, int64_t> extentToDomain;
    for (int64_t domain : tilingInfo.tilingOrder) {
        auto [it, inserted] = extentToDomain.try_emplace(iterDomainSizes[domain], domain);
        if (!inserted) {
            LLVM_DEBUG(
                llvm::dbgs() << "shrink-order: bail duplicate extent " << iterDomainSizes[domain]
                             << "\n"
            );
            return std::nullopt;
        }
    }

    // Current candidate tile size per consumer domain (0 means untiled: full).
    SmallVector<int64_t> curSizes;
    for (const OpFoldResult &size : sizes) {
        llvm::FailureOr<int64_t> maybeSize = computeSizeAtFirstIteration(size);
        if (failed(maybeSize)) {
            LLVM_DEBUG(llvm::dbgs() << "shrink-order: bail curSizes\n");
            return std::nullopt;
        }
        int64_t value = *maybeSize;
        curSizes.push_back(value > 0 ? value : std::numeric_limits<int64_t>::max());
    }

    auto minSizeOf = [&](int64_t domain) {
        return fallback ? getSmallestTileSize(tilingInfo, domain, iterDomainSizes[domain])
                        : tilingInfo.minSize[domain];
    };

    SmallVector<Operation *> groupOps(producerOps.begin(), producerOps.end());
    groupOps.push_back(consumerOp);
    SmallPtrSet<Operation *, 8> groupOpSet(groupOps.begin(), groupOps.end());

    // Estimated group operand bytes when `shrinkDomain` is forced to its
    // minimum (or the current candidate sizes when shrinkDomain == -1).
    auto estimateBytes = [&](int64_t shrinkDomain) -> std::optional<int64_t> {
        SmallPtrSet<Value, 8> seen;
        int64_t total = 0;
        for (Operation *op : groupOps) {
            auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
            if (!linalgOp)
                return std::nullopt;
            auto ranges = linalgOp.getStaticLoopRanges();
            if (llvm::is_contained(ranges, ShapedType::kDynamic))
                return std::nullopt;

            // Tile size per op domain, via extent correspondence.
            SmallVector<int64_t> opTile(ranges.size());
            for (size_t i = 0; i < ranges.size(); ++i) {
                auto it = extentToDomain.find((int64_t)ranges[i]);
                if (it == extentToDomain.end()) {
                    // Reduction/broadcast domain of the op: not tiled by the
                    // consumer tile, keeps its full extent.
                    opTile[i] = ranges[i];
                    continue;
                }
                int64_t consumerDomain = it->second;
                int64_t tile = consumerDomain == shrinkDomain ? minSizeOf(consumerDomain)
                                                              : curSizes[consumerDomain];
                opTile[i] = std::min(tile, (int64_t)ranges[i]);
            }

            for (OpOperand &operand : op->getOpOperands()) {
                Value value = operand.get();
                if (!seen.insert(value).second)
                    continue;
                // An in-group operand aliases its producer's init (e.g. a fill
                // feeding the accumulator); counting it double-weights output dims.
                if (Operation *def = value.getDefiningOp(); def && groupOpSet.contains(def))
                    continue;
                auto type = dyn_cast<RankedTensorType>(value.getType());
                if (!type) {
                    // Scalar operand (e.g. a fill value): fixed bytes; don't
                    // void the whole estimate over it.
                    if (!isa<ShapedType>(value.getType()))
                        continue;
                    return std::nullopt;
                }
                if (!type.hasStaticShape())
                    return std::nullopt;
                AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
                int64_t elements = 1;
                for (auto [resultIndex, expr] : llvm::enumerate(map.getResults())) {
                    // Constant-0 = broadcast unit dim: one element regardless of
                    // tile. Other constants are unvalidated; keep the legacy bail.
                    if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr)) {
                        if (cstExpr.getValue() != 0)
                            return std::nullopt;
                        continue;
                    }
                    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
                    if (!dimExpr)
                        return std::nullopt;
                    int64_t extent = opTile[dimExpr.getPosition()];
                    elements *= std::min(extent, type.getDimSize(resultIndex));
                }
                total += elements * llvm::divideCeil(type.getElementTypeBitWidth(), 8);
            }
        }
        return total;
    };

    std::optional<int64_t> baseEstimate = estimateBytes(-1);
    if (!baseEstimate) {
        LLVM_DEBUG(llvm::dbgs() << "shrink-order: bail baseEstimate\n");
        return std::nullopt;
    }

    SmallVector<std::pair<int64_t, int64_t>> removedByDomain; // (removed bytes, domain)
    for (int64_t domain : tilingInfo.tilingOrder) {
        if (curSizes[domain] == minSizeOf(domain))
            continue; // already at minimum, shrink pass would skip it
        std::optional<int64_t> shrunk = estimateBytes(domain);
        if (!shrunk || *shrunk > *baseEstimate) {
            LLVM_DEBUG(
                llvm::dbgs() << "shrink-order: bail estimate d" << domain
                             << (shrunk ? " non-monotone" : " failed") << "\n"
            );
            return std::nullopt; // estimate must be monotone shrink-only; bail
        }
        removedByDomain.emplace_back(*baseEstimate - *shrunk, domain);
    }

    // Most bytes removed first; ties keep legacy tilingOrder (stable sort).
    llvm::stable_sort(removedByDomain, [](const auto &a, const auto &b) {
        return a.first > b.first;
    });

    LLVM_DEBUG({
        llvm::dbgs() << "shrink-order: base=" << *baseEstimate << " order:";
        for (auto &[removed, domain] : removedByDomain)
            llvm::dbgs() << " d" << domain << "(-" << removed << ")";
        llvm::dbgs() << "\n";
    });

    SmallVector<int64_t> order;
    for (auto &[removed, domain] : removedByDomain)
        order.push_back(domain);
    // Domains already at their minimum are not in removedByDomain; the legacy
    // loop skips them anyway, so no need to append them.
    return order;
}

namespace {

constexpr int64_t kMaxReduceTileElements = 1 << 16;

llvm::FailureOr<int64_t>
estimateTiledOperandBytes(linalg::LinalgOp linalgOp, unsigned reductionDim, int64_t tileSize) {
    int64_t estimatedBytes = 0;

    for (OpOperand &operand : linalgOp->getOpOperands()) {
        // The output/init accumulator is not reduced by the K-split; T&F tiles
        // its parallel (M/N) dims separately. Count only the inputs (A/B) here
        // so a matmul whose output alone overflows LRAM is still K-split for its
        // inputs rather than being (wrongly) sent to Host.
        if (linalgOp.isDpsInit(&operand))
            continue;

        auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!tensorType || !tensorType.hasStaticShape())
            return llvm::failure();

        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
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

// Replace `op` (linalg.matmul [M,K]x[K,N] or linalg.batch_matmul [B,M,K]x[B,K,N])
// with K/chunkSize accumulating chunk matmuls. chunkSize must divide K evenly.
//
// The split is expressed as an scf.for reduction loop (FullReduction), which
// carries the accumulator as a loop iter_arg, and then immediately fully
// unrolled. T&F's fit-to-memory walk expects loop-free IR, so the unroll must
// happen before control returns to TileAndFusePass.
llvm::LogicalResult
splitMatmulAlongK(IRRewriter &rewriter, linalg::LinalgOp op, int64_t chunkSize) {
    SmallVector<unsigned> reductionDims;
    op.getReductionDims(reductionDims);
    if (reductionDims.size() != 1)
        return llvm::failure();
    unsigned kDim = reductionDims[0];

    unsigned numLoops = op.getNumLoops();
    SmallVector<OpFoldResult> tileSizes(numLoops, rewriter.getIndexAttr(0));
    tileSizes[kDim] = rewriter.getIndexAttr(chunkSize);

    scf::SCFTilingOptions options;
    options.setTileSizes(tileSizes);
    options.setReductionTilingStrategy(ReductionTilingStrategy::FullReduction);
    options.setReductionDims({kDim});

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op.getOperation()), options);
    if (failed(tilingResult))
        return llvm::failure();

    // chunkSize divides K, so the generated reduction loop has a static trip
    // count. Verify that before erasing `op`; otherwise the caller needs to
    // fall back to Host and the original op must stay intact.
    for (auto loop : tilingResult->loops) {
        if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
            if (!forOp.getStaticTripCount().has_value())
                return llvm::failure();
    }

    // Replace the original matmul with the loop result(s) first. The loop
    // iter_arg is the running accumulator; after the unroll below the loop
    // result is RAUW'd to the final accumulated value, so the original op's
    // users end up consuming that final value.
    rewriter.replaceOp(op, tilingResult->replacements);

    for (auto loop : tilingResult->loops) {
        if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
            (void)loopUnrollFull(forOp);
    }

    return llvm::success();
}

} // namespace

// LRAM budget for a single non-tileable op: total LRAM minus 14k headroom for
// descriptors (used to be TorqHw::get().getAvailableLramSize()). Borrowed from
// OptimizeConv1DPattern.cpp.
static int64_t getLramTilingBudget() {
    return static_cast<int64_t>(TorqHw::get().getLramSize() - 14 * 1024);
}

void splitOversizedMatmulsAlongK(func::FuncOp funcOp) {
    const int64_t lramSize = getLramTilingBudget();

    SmallVector<std::pair<linalg::LinalgOp, int64_t>> opsToSplit;
    funcOp.walk([&](linalg::LinalgOp linalgOp) {
        if (!isa<linalg::MatmulOp, linalg::BatchMatmulOp>(linalgOp))
            return;
        // Host-executed matmuls don't need LRAM tiling.
        if (getTargetExecutor(linalgOp) == torq_hl::Executor::Host)
            return;

        // K-splitting replaces the matmul with new tiled matmuls / scf.for loops
        // that do not carry TORQ_FUSE_GROUP markings. T&F's fuse-group invariant
        // requires every marked op to be reachable from its principal op, so
        // leave fuse-group matmuls to T&F's parallel-dim tiling rather than
        // breaking the group here.
        if (isMarkedFuseGroup(linalgOp))
            return;

        // K-splitting rewrites the matmul into a chain of accumulating chunk
        // matmuls, so every chunk after the first has a non-zero init.
        //  - bf16: still K-split. Each chunk's NSS matmul accumulates in fp32 and
        //    rounds its result to bf16 once, so the chain adds one extra bf16
        //    rounding of the running sum at every chunk boundary (up to 0.5 ULP
        //    of bf16, ~2^-9 relative, per boundary, so ~chunkCount * 2^-9 total).
        //    That is the accuracy cost of K-splitting bf16; it is gated by the
        //    tests' bf16 abs-gate tolerance.
        //  - f16/f32/f64: skip. The TORQ matmul is non-accumulating and the
        //    explicit elementwise add of the non-zero init is not supported for
        //    these types on the NSS path (f32/f64 lowering asserts instead of
        //    falling back; f16 accumulation is not supported either), so leave
        //    them to T&F's parallel (M/N) tiling.
        //  - integer accumulators: still K-split (exact accumulation).
        Value init = linalgOp.getDpsInits()[0];
        if (auto initType = dyn_cast<RankedTensorType>(init.getType()))
            if (initType.getElementType().isFloat() && !initType.getElementType().isBF16())
                return;

        SmallVector<unsigned> reductionDims;
        for (auto [i, iteratorType] : llvm::enumerate(linalgOp.getIteratorTypesArray())) {
            if (iteratorType == utils::IteratorType::reduction)
                reductionDims.push_back(i);
        }
        if (reductionDims.size() != 1)
            return;
        unsigned kDim = reductionDims[0];

        auto loopRanges = linalgOp.getStaticLoopRanges();
        int64_t reductionSize = loopRanges[kDim];
        if (reductionSize == ShapedType::kDynamic || reductionSize <= 1)
            return;

        bool operandAloneOversized = false;
        for (OpOperand &operand : linalgOp->getOpOperands()) {
            auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
            if (!tensorType || !tensorType.hasStaticShape())
                return;
            int64_t bytes = tensorType.getNumElements() *
                            llvm::divideCeil(tensorType.getElementTypeBitWidth(), 8);
            if (bytes > lramSize)
                operandAloneOversized = true;
        }
        if (!operandAloneOversized)
            return;

        // Split only when NO single parallel-dim shrink can fit the op.
        bool parallelShrinkCanFit = false;
        for (auto [i, iteratorType] : llvm::enumerate(linalgOp.getIteratorTypesArray())) {
            if (iteratorType != utils::IteratorType::parallel || loopRanges[i] <= 1)
                continue;
            llvm::FailureOr<int64_t> estimatedBytes = estimateTiledOperandBytes(linalgOp, i, 1);
            if (failed(estimatedBytes))
                return;
            if (*estimatedBytes <= lramSize) {
                parallelShrinkCanFit = true;
                break;
            }
        }
        if (parallelShrinkCanFit)
            return;

        // Halve K until the per-chunk operands fit LRAM.
        int64_t tileSize = reductionSize;
        bool fitsInLram = false;
        while (tileSize > 1) {
            tileSize = std::max(tileSize / 2, int64_t(1));
            llvm::FailureOr<int64_t> estimatedBytes =
                estimateTiledOperandBytes(linalgOp, kDim, tileSize);
            if (failed(estimatedBytes))
                return;
            if (*estimatedBytes <= lramSize && tileSize <= kMaxReduceTileElements) {
                fitsInLram = true;
                break;
            }
        }
        if (!fitsInLram) {
            // Can't K-split this matmul to fit LRAM: run it on Host instead.
            setTargetExecutorAttr(linalgOp.getOperation(), torq_hl::Executor::Host);
            return;
        }

        // Snap down to the largest divisor of K <= tileSize so the rewrite
        // produces an exact chunk count (K/2^i does not divide non-power-of-2 K).
        while (tileSize > 1 && reductionSize % tileSize != 0)
            tileSize--;

        opsToSplit.emplace_back(linalgOp, tileSize);
    });

    IRRewriter rewriter(funcOp->getContext());
    for (auto [linalgOp, chunkSize] : opsToSplit) {
        if (failed(splitMatmulAlongK(rewriter, linalgOp, chunkSize))) {
            // Splitting failed; fall back to Host rather than failing the pass.
            setTargetExecutorAttr(linalgOp.getOperation(), torq_hl::Executor::Host);
        }
    }
}

} // namespace mlir::syna::torq
