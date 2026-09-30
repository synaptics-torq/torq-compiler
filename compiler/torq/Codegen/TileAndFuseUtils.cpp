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
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#define DEBUG_TYPE "torq-tile-and-fuse-utils"

namespace mlir::syna::torq {

llvm::FailureOr<int64_t>
estimateTiledOperandBytes(linalg::LinalgOp linalgOp, ArrayRef<int64_t> tileSizes) {
    if (tileSizes.size() != linalgOp.getNumLoops())
        return llvm::failure();

    int64_t estimatedBytes = 0;
    for (OpOperand &operand : linalgOp->getOpOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
        if (!tensorType || !tensorType.hasStaticShape())
            return llvm::failure();

        AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
        int64_t numElements = 1;
        for (auto [resultIndex, expr] : llvm::enumerate(map.getResults())) {
            int64_t dimSize = tensorType.getDimSize(resultIndex);
            if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
                dimSize = std::min(dimSize, tileSizes[dimExpr.getPosition()]);
            numElements *= dimSize;
        }
        estimatedBytes +=
            numElements * llvm::divideCeil(tensorType.getElementTypeBitWidth(), int64_t(8));
    }
    return estimatedBytes;
}

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

// Compute the order in which fitTileToMemory's shrink pass should cut domains.
// For a weight-bearing principal, compare only its activation (DPS input 0)
// and weight (DPS input 1) memory. Prioritize domains indexing the larger
// operand, which naturally selects channel domains when weights dominate and
// frame domains when activations dominate. Indexing maps make this independent
// of NCHW/NHWC and applicable to convolutions, contractions, and equivalent
// generic operations.
//
// Otherwise, for a matmul group, order domains greedily by the number of
// operand bytes each shrink removes across the whole fuse group (consumer +
// producers), ties broken by the legacy tilingOrder.
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
// For the matmul fallback, correspondence between a group op's iteration
// domains and the consumer's tileable domains is established by
// dimension-extent equality. If anything is ambiguous or non-static (duplicate
// extents among tileable domains, an op domain matching more than one
// candidate, non-dim map results), we return nullopt and the caller keeps the
// legacy order.
// Named matmuls plus the generics replaceBatchMatmulWithBroadcastGeneric marks:
// matmuls to every tiling heuristic, but invisible to isa<>.
static bool isMatmulLikeOp(Operation *op) {
    return isa<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ContractOp>(op) ||
           op->hasAttr(TORQ_BROADCAST_MATMUL);
}

static bool
hasBroadcastBatchMatmul(Operation *consumerOp, const SetVector<Operation *> &producerOps) {
    if (consumerOp->hasAttr(TORQ_BROADCAST_MATMUL))
        return true;
    return llvm::any_of(producerOps, [](Operation *producer) {
        return producer->hasAttr(TORQ_BROADCAST_MATMUL);
    });
}

struct WeightedPrincipal {
    linalg::LinalgOp op;
    int64_t inputBytes;
    int64_t weightBytes;
};

static std::optional<WeightedPrincipal>
findWeightedPrincipal(Operation *consumerOp, const SetVector<Operation *> &producerOps) {
    SmallVector<Operation *> candidates;
    if (isMarkedFuseGroup(consumerOp))
        candidates.push_back(getFuseGroupPrincipalOpBackward(consumerOp));
    else
        candidates.push_back(consumerOp);
    for (Operation *producer : producerOps)
        if (isFuseGroupPrincipalOp(producer))
            candidates.push_back(producer);

    std::optional<WeightedPrincipal> principal;
    int64_t largestOperandBytes = -1;
    for (Operation *candidate : candidates) {
        auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(candidate);
        if (!linalgOp || linalgOp.getNumReductionLoops() == 0 || linalgOp.getNumDpsInputs() < 2)
            continue;

        auto inputType = dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[0].getType());
        auto weightType = dyn_cast<RankedTensorType>(linalgOp.getDpsInputs()[1].getType());
        if (!inputType || !weightType || !inputType.hasStaticShape() ||
            !weightType.hasStaticShape())
            continue;

        auto storageBytes = [](RankedTensorType type) {
            return type.getNumElements() * llvm::divideCeil(type.getElementTypeBitWidth(), 8);
        };
        int64_t inputBytes = storageBytes(inputType);
        int64_t weightBytes = storageBytes(weightType);
        if (inputBytes + weightBytes > largestOperandBytes) {
            principal = WeightedPrincipal{linalgOp, inputBytes, weightBytes};
            largestOperandBytes = inputBytes + weightBytes;
        }
    }
    return principal;
}

std::optional<SmallVector<int64_t>> computeChannelFirstShrinkOrder(
    Operation *consumerOp, const SetVector<Operation *> &producerOps, const TilingInfo &tilingInfo,
    ArrayRef<int64_t> iterDomainSizes
) {
    // A broadcast batch matmul's RHS batch dimension is weight-only but is not
    // an output-channel dimension. Keep it on the batch-first path to avoid
    // re-streaming the full RHS for every M tile.
    if (hasBroadcastBatchMatmul(consumerOp, producerOps))
        return std::nullopt;

    std::optional<WeightedPrincipal> principal = findWeightedPrincipal(consumerOp, producerOps);
    if (!principal)
        return std::nullopt;

    SmallVector<int64_t> ranges = principal->op.getStaticLoopRanges();
    SmallVector<utils::IteratorType> iteratorTypes = principal->op.getIteratorTypesArray();
    AffineMap weightMap = principal->op.getMatchingIndexingMap(principal->op.getDpsInputOperand(1));

    std::optional<int64_t> principalChannel;
    for (auto [domain, iteratorType] : llvm::enumerate(iteratorTypes)) {
        if (iteratorType != utils::IteratorType::parallel)
            continue;
        int64_t domainIndex = domain;
        bool indexesWeight = llvm::any_of(weightMap.getResults(), [&](AffineExpr expr) {
            return expr.isFunctionOfDim(domainIndex);
        });
        if (!indexesWeight)
            continue;
        if (principalChannel)
            return std::nullopt;
        principalChannel = domain;
    }
    if (!principalChannel)
        return std::nullopt;

    std::optional<int64_t> rootChannel;
    int64_t channelExtent = ranges[*principalChannel];
    for (int64_t domain : tilingInfo.tilingOrder) {
        if (iterDomainSizes[domain] != channelExtent)
            continue;
        if (rootChannel)
            return std::nullopt;
        rootChannel = domain;
    }
    if (!rootChannel)
        return std::nullopt;

    SmallVector<int64_t> order{*rootChannel};
    for (int64_t domain : tilingInfo.tilingOrder)
        if (domain != *rootChannel)
            order.push_back(domain);
    return order;
}

static std::optional<SmallVector<int64_t>> computePrincipalInputWeightShrinkOrder(
    Operation *consumerOp, const SetVector<Operation *> &producerOps, const TilingInfo &tilingInfo,
    ArrayRef<int64_t> iterDomainSizes
) {
    // The RHS of a broadcast batch matmul remains batch-dependent. Its batch
    // dimension is therefore weight-only in the generic indexing map, but is
    // not an output-channel dimension. Do not apply the dominant-weight
    // ordering: keeping batch whole forces M tiling and re-streams the full
    // RHS for every M tile.
    if (hasBroadcastBatchMatmul(consumerOp, producerOps))
        return std::nullopt;

    std::optional<WeightedPrincipal> principal = findWeightedPrincipal(consumerOp, producerOps);
    if (!principal)
        return std::nullopt;

    linalg::LinalgOp linalgOp = principal->op;
    int64_t inputBytes = principal->inputBytes;
    int64_t weightBytes = principal->weightBytes;
    if (inputBytes == weightBytes)
        return std::nullopt;

    AffineMap inputMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
    AffineMap weightMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));
    auto indexesDomain = [](AffineMap map, int64_t domain) {
        return llvm::any_of(map.getResults(), [&](AffineExpr expr) {
            return expr.isFunctionOfDim(domain);
        });
    };

    SmallVector<int64_t> principalRanges = linalgOp.getStaticLoopRanges();
    if (llvm::is_contained(principalRanges, ShapedType::kDynamic))
        return std::nullopt;

    // Map the principal's preferred domains onto the tiling root. For an
    // NCHW-conv -> NHWC-transpose chain this maps conv d1 (C=1536) to transpose
    // d3 (C=1536), without relying on the two operations having the same layout.
    const bool prioritizeWeights = weightBytes > inputBytes;
    SmallVector<int64_t> preferred;
    SmallVector<int64_t> shared;
    SmallVector<int64_t> remaining;
    bool hasDistinctDependency = false;
    for (int64_t domain : tilingInfo.tilingOrder) {
        bool inputDepends = false;
        bool weightDepends = false;
        for (auto [principalDomain, extent] : llvm::enumerate(principalRanges)) {
            if (extent != iterDomainSizes[domain])
                continue;
            inputDepends |= indexesDomain(inputMap, principalDomain);
            weightDepends |= indexesDomain(weightMap, principalDomain);
        }
        hasDistinctDependency |= inputDepends != weightDepends;
        bool preferredOnly =
            prioritizeWeights ? weightDepends && !inputDepends : inputDepends && !weightDepends;
        if (preferredOnly)
            preferred.push_back(domain);
        else if (inputDepends && weightDepends)
            shared.push_back(domain);
        else
            remaining.push_back(domain);
    }
    if (!hasDistinctDependency || preferred.empty())
        return std::nullopt;

    SmallVector<int64_t> order(std::move(preferred));
    order.append(shared);
    order.append(remaining);

    LLVM_DEBUG({
        llvm::dbgs() << "shrink-order: weighted principal=" << linalgOp->getName()
                     << " input-bytes=" << inputBytes << " weight-bytes=" << weightBytes
                     << " prioritize=" << (prioritizeWeights ? "weights" : "input") << " order:";
        for (int64_t domain : order)
            llvm::dbgs() << " d" << domain;
        llvm::dbgs() << "\n";
    });
    return order;
}

std::optional<SmallVector<int64_t>> computeShrinkOrderByReduction(
    Operation *consumerOp, const SetVector<Operation *> &producerOps, const TilingInfo &tilingInfo,
    ArrayRef<int64_t> iterDomainSizes, ArrayRef<OpFoldResult> sizes, bool fallback
) {
    if (std::optional<SmallVector<int64_t>> order = computePrincipalInputWeightShrinkOrder(
            consumerOp, producerOps, tilingInfo, iterDomainSizes
        ))
        return order;

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

// Return true if a length can be represented using the SDIM encoding in
// Utils/Kernel.cpp without tripping the decomposeIntoTwoFactors() assert.
// A "SDIM-friendly" length is either:
//   * <= 0x7fff  – fits in a single SDIM counter, or
//   * factorizable as a * b with a, b <= 0x7fff – can be encoded as two nested SDIM loops.
constexpr int64_t kMaxFactor = 0x7fff;
bool isSdimFriendlyCount(int64_t number) {
    if (number <= kMaxFactor)
        return true;

    for (int64_t i = llvm::divideCeil(number, kMaxFactor);
         i <= (int64_t)std::sqrt((double)number) + 1; ++i) {
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
bool isSdimFriendlyTileSize(int64_t iterDomainSize, int64_t tileSize) {
    if (!isSdimFriendlyCount(tileSize))
        return false;

    int64_t remainder = iterDomainSize % tileSize;
    return (remainder == 0 || isSdimFriendlyCount(remainder));
}

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

int64_t makeSdimFriendlyTileSize(int64_t iterDomainSize, int64_t tileSize) {
    // A non-positive size is already invalid (e.g. the "too small" sentinel that
    // makeByteAlignedTileSize returns when composed via getFixpointF in
    // TileAndFusePass.cpp); pass it through instead of searching and tripping
    // the assert.
    if (tileSize <= 0)
        return tileSize;

    // Adjust tileSize downwards until both the common tile size and the
    // remainder tile size are SDIM-friendly. This preserves the
    // memory-fit invariant because we only shrink the tile.
    for (tileSize = std::min(int64_t(kMaxFactor * kMaxFactor), tileSize); tileSize > 0;
         --tileSize) {
        if (isSdimFriendlyTileSize(iterDomainSize, tileSize))
            break;
    }
    // isValidTileSize should always succeed for tileSize == 1
    assert(tileSize > 0 && "failed to find SDIM-friendly tile size for domain");

    return tileSize;
}

int64_t largestReductionTileAtMost(int64_t reductionSize, int64_t requestedTileSize) {
    return makeSdimFriendlyTileSize(reductionSize, std::min(reductionSize, requestedTileSize));
}

// LRAM budget for a single non-tileable op: total LRAM minus 14k headroom for
// descriptors (used to be TorqHw::get().getAvailableLramSize()). Borrowed from
// OptimizeConv1DPattern.cpp.
int64_t getLramTilingBudget() {
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
