// Copyright 2025 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MatmulTilingHeuristics.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "torq-tile-and-fuse"

using namespace mlir;

namespace mlir::syna::torq {

// Escape hatches for A/B-ing each heuristic.
static llvm::cl::opt<bool> clDisableMatmulSizeOrder(
    "torq-disable-matmul-size-order",
    llvm::cl::desc("Consider a matmul's parallel domains for tiling in index order rather than "
                   "largest-first."),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clDisableMatmulVectorAlign(
    "torq-disable-matmul-vector-align",
    llvm::cl::desc("Do not keep the matmul N tile a whole number of ALU vector columns."),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clDisableMatmulLoopReorder(
    "torq-disable-matmul-loop-reorder",
    llvm::cl::desc("Always visit matmul tiles for(m)for(n); never reverse to for(n)for(m)."),
    llvm::cl::init(false)
);

namespace {

// The kernels the N-tile heuristics are validated on: the named contractions.
// The broadcast-matmul generics (torq-broadcast-matmul) are deliberately not
// included: their N tile also sets the K-split chunking, and aligning it was
// measured as a net loss on a transformer model.
bool isMatmulKernel(Operation *op) {
    return isa_and_nonnull<linalg::MatmulOp, linalg::BatchMatmulOp, linalg::ContractOp>(op);
}

// Elements the ALU consumes per cycle along the vectorized (N) axis of
// `matmulOp`, taken from the kernel model so the two cannot drift; 0 when the
// operand types have no ALU mapping.
int64_t getMatmulAluVectorWidth(Operation *matmulOp) {
    auto inputs = cast<linalg::LinalgOp>(matmulOp).getDpsInputs();
    if (inputs.size() < 2)
        return 0;
    // The kernel (MatMulPattern) vectorizes the [K,N] operand as the ALU input
    // and streams the [M,K] operand as the weight; mirror that operand order.
    DType iType = getDType(getElementTypeOrSelf(inputs[1].getType()));
    DType wType = getDType(getElementTypeOrSelf(inputs[0].getType()));
    if (iType == DType::none || wType == DType::none || isCompressed(iType))
        return 0;
    // Alu::iWidth asserts on pairings no matmul kernel produces (a float weight
    // on an int input, int products wider than 32 bits); skip those here.
    DType wDeduced = isFloat(iType) ? DType::bf16 : toUncompressed(wType);
    if (isFloat(iType) != isFloat(wDeduced))
        return 0;
    if (isInt(iType) && sizeofType(iType) * sizeofType(wDeduced) > 4)
        return 0;
    Slice slice;
    return slice.alu.iWidth(iType, wType);
}

// The iteration-domain index driving the innermost (contiguous) output
// dimension -- the vectorized axis. Read off the output's indexing map because
// it differs per op: for linalg.matmul (M,N,K) it is 1 (N), not the last index.
std::optional<int64_t> getInnermostOutputDomain(Operation *op) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp || linalgOp.getNumDpsInits() != 1)
        return std::nullopt;
    AffineMap map = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
    if (map.getNumResults() == 0)
        return std::nullopt;
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(map.getNumResults() - 1));
    if (!dimExpr)
        return std::nullopt;
    return static_cast<int64_t>(dimExpr.getPosition());
}

// Total ALU vector columns needed to cover `domainSize` in tiles of `size`:
// every full tile costs ceil(size/V) columns and the tail tile ceil(tail/V).
int64_t totalVectorColumns(int64_t vectorWidth, int64_t domainSize, int64_t size) {
    int64_t fullTiles = domainSize / size;
    int64_t tail = domainSize % size;
    return fullTiles * llvm::divideCeil(size, vectorWidth) +
           (tail ? llvm::divideCeil(tail, vectorWidth) : 0);
}

// Round a tile size DOWN to a whole number of ALU vector columns, but only when
// that saves vector columns over the whole dimension (63 over 126 stays: the
// snap would only add tiles and DMA). Shrink-only, so "fits in memory" holds.
int64_t makeVectorAlignedTileSize(int64_t vectorWidth, int64_t domainSize, int64_t size) {
    if (vectorWidth <= 1 || size <= 0 || domainSize <= vectorWidth || size >= domainSize)
        return size;
    int64_t remainder = size % vectorWidth;
    if (remainder == 0 || size - remainder <= 0)
        return size;
    int64_t aligned = size - remainder;
    if (totalVectorColumns(vectorWidth, domainSize, aligned) <
        totalVectorColumns(vectorWidth, domainSize, size))
        return aligned;
    return size;
}

// The element bit width `input` is streamed at from external memory: with a
// fused dequant the DMA moves the int8/int4 source, not the bf16 the matmul
// consumes, so follow the fused chain of shape-preserving all-parallel generics
// back to the streamed tensor. Returns 0 when no width can be determined.
int64_t getStreamedBitWidth(Value input, const llvm::SetVector<Operation *> &fusedProducers) {
    Value current = input;
    for (int depth = 0; depth < 8; ++depth) {
        Operation *producerOp = current.getDefiningOp();
        if (!producerOp || !fusedProducers.contains(producerOp))
            break;
        auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
        if (!genericOp || genericOp.getNumDpsInits() != 1 || genericOp.getNumReductionLoops() != 0)
            break;
        auto resultType = dyn_cast<RankedTensorType>(current.getType());
        if (!resultType || !resultType.hasStaticShape())
            break;
        // The streamed operand is the one with the result's full shape (scales and
        // zero points are smaller); two same-shape inputs leave no single source.
        Value next = nullptr;
        for (Value operand : genericOp.getDpsInputs()) {
            auto operandType = dyn_cast<RankedTensorType>(operand.getType());
            if (!operandType || !operandType.hasStaticShape() ||
                operandType.getShape() != resultType.getShape())
                continue;
            if (next) {
                next = nullptr;
                break;
            }
            next = operand;
        }
        if (!next)
            break;
        current = next;
    }
    auto shapedType = dyn_cast<ShapedType>(current.getType());
    if (!shapedType || !shapedType.getElementType().isIntOrFloat())
        return 0;
    return shapedType.getElementType().getIntOrFloatBitWidth();
}

} // namespace

Operation *findTiledMatmul(Operation *op, const llvm::SetVector<Operation *> &fusedProducers) {
    if (isMatmulKernel(op))
        return op;
    if (isMarkedFuseGroup(op)) {
        // e.g. the collapsed-batch-matmul form `matmul -> truncf generic`, whose
        // group is anchored on the generic.
        Operation *principalOp = getFuseGroupPrincipalOpBackward(op);
        if (isMatmulKernel(principalOp))
            return principalOp;
    }
    // e.g. an FFN up-projection fused into its activation's tile.
    for (Operation *producerOp : fusedProducers)
        if (isMatmulKernel(producerOp))
            return producerOp;
    return nullptr;
}

void applyMatmulTilingHeuristics(
    TilingInfo &tilingInfo, TilingInterface op, ArrayRef<int64_t> iterDomainSizes,
    const llvm::SetVector<Operation *> &fusedProducers
) {
    Operation *matmulOp = findTiledMatmul(op, fusedProducers);
    if (!matmulOp)
        return;

    // A tensor.extract reads a hidden input that is not sliced with the output
    // tile, so a changed tile shape can drive it to an unsupported lowering.
    auto containsExtract = [](Operation *candidate) {
        return candidate->walk([](tensor::ExtractOp) { return WalkResult::interrupt(); }
        ).wasInterrupted();
    };
    bool hasExtract = containsExtract(op) || llvm::any_of(fusedProducers, containsExtract);
    // Convolution and pooling tiles carry a halo the matmul cost model ignores.
    Operation *principalOp = isMarkedFuseGroup(op) ? getFuseGroupPrincipalOpBackward(op) : op;
    bool isConvOrPool =
        isa<linalg::ConvolutionOpInterface>(principalOp) ||
        isa<linalg::PoolingNhwcMaxOp, linalg::PoolingNchwMaxOp, linalg::PoolingNcwMaxOp,
            linalg::PoolingNhwcSumOp, linalg::PoolingNchwSumOp>(principalOp);
    if (hasExtract || isConvOrPool)
        return;

    // Consider the largest parallel domain first, so the shrink pass cuts it
    // first and the grow-back re-inflates it last: the search settles near the
    // maximum-area (minimum tile count) tile instead of a thin slab.
    if (!clDisableMatmulSizeOrder) {
        SmallVector<int64_t> domains(tilingInfo.tilingOrder.begin(), tilingInfo.tilingOrder.end());
        llvm::stable_sort(domains, [&](int64_t a, int64_t b) {
            return iterDomainSizes[a] > iterDomainSizes[b];
        });
        tilingInfo.tilingOrder.clear();
        tilingInfo.tilingOrder.insert(domains.begin(), domains.end());
    }

    // Compute is Tm * ceil(Tn/V) * Tk, so an N tile off the vector width pays a
    // partly-used vector per column step: keep it aligned when that saves
    // columns, and prefer at least one whole vector. When even one vector cannot
    // fit (deep-K shapes), fitTileToMemory's fallback shrink pass ignores minSize
    // and lands on the same sub-vector tile a floor-less search would.
    if (clDisableMatmulVectorAlign)
        return;
    int64_t vectorWidth = getMatmulAluVectorWidth(matmulOp);
    std::optional<int64_t> nDomain = getInnermostOutputDomain(op);
    if (vectorWidth <= 1 || !nDomain || !tilingInfo.tilingOrder.contains(*nDomain))
        return;
    int64_t domainSize = iterDomainSizes[*nDomain];
    tilingInfo.adjustSize[*nDomain] = composeSizeAdjustments(
        [=](int64_t size) { return makeVectorAlignedTileSize(vectorWidth, domainSize, size); },
        tilingInfo.adjustSize[*nDomain]
    );
    if (domainSize > vectorWidth)
        tilingInfo.minSize[*nDomain] = std::max(tilingInfo.minSize[*nDomain], vectorWidth);
}

SmallVector<int64_t> chooseMatmulLoopInterchange(
    Operation *op, ArrayRef<OpFoldResult> tileSizes,
    const llvm::SetVector<Operation *> &fusedProducers
) {
    if (clDisableMatmulLoopReorder)
        return {};
    auto matmulOp = dyn_cast_if_present<linalg::MatmulOp>(findTiledMatmul(op, fusedProducers));
    if (!matmulOp)
        return {};

    auto lhsType = dyn_cast<ShapedType>(matmulOp.getInputs()[0].getType());
    auto rhsType = dyn_cast<ShapedType>(matmulOp.getInputs()[1].getType());
    if (!lhsType || !rhsType || !lhsType.hasStaticShape() || !rhsType.hasStaticShape())
        return {};

    // The byte model assumes the canonical [M,K]x[K,N] orientation; keep the
    // default order for anything else (e.g. a transposed [N,K] RHS).
    if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
        return {};

    const int64_t M = lhsType.getDimSize(0);
    const int64_t K = lhsType.getDimSize(1);
    const int64_t N = rhsType.getDimSize(1);

    // Priced per operand at the streamed (pre-dequant) width, in bits so
    // sub-byte types weigh in at their true cost.
    const int64_t bitsA = getStreamedBitWidth(matmulOp.getInputs()[0], fusedProducers);
    const int64_t bitsB = getStreamedBitWidth(matmulOp.getInputs()[1], fusedProducers);
    if (bitsA == 0 || bitsB == 0)
        return {};
    int64_t bitsOut = bitsA;
    if (auto outType = dyn_cast<ShapedType>(op->getResult(0).getType());
        outType && outType.getElementType().isIntOrFloat())
        bitsOut = outType.getElementType().getIntOrFloatBitWidth();

    // The anchor domains carrying the matmul's M and N: its own for a bare
    // matmul, else the last two output dimensions, guarded by extent equality so
    // a reshaping or transposing chain keeps the default order.
    int64_t mDomain, nDomain;
    if (op == matmulOp.getOperation()) {
        if (tileSizes.size() != 3)
            return {};
        mDomain = 0;
        nDomain = 1;
    }
    else {
        auto anchor = dyn_cast<linalg::LinalgOp>(op);
        if (!anchor || anchor.getNumDpsInits() != 1 || anchor.getNumLoops() != tileSizes.size())
            return {};
        AffineMap outMap = anchor.getMatchingIndexingMap(anchor.getDpsInitOperand(0));
        if (outMap.getNumResults() < 2)
            return {};
        auto mExpr = dyn_cast<AffineDimExpr>(outMap.getResult(outMap.getNumResults() - 2));
        auto nExpr = dyn_cast<AffineDimExpr>(outMap.getResult(outMap.getNumResults() - 1));
        if (!mExpr || !nExpr)
            return {};
        mDomain = mExpr.getPosition();
        nDomain = nExpr.getPosition();
        SmallVector<int64_t> ranges = anchor.getStaticLoopRanges();
        if (ranges[mDomain] != M || ranges[nDomain] != N)
            return {};
    }

    // A tile size of 0 means "not tiled", i.e. the whole dimension.
    auto extent = [](OpFoldResult ofr, int64_t whole) {
        std::optional<int64_t> v = getConstantIntValue(ofr);
        if (!v || *v <= 0)
            return whole;
        return std::min(*v, whole);
    };
    const int64_t tm = extent(tileSizes[mDomain], M);
    const int64_t tn = extent(tileSizes[nDomain], N);

    // The inner loop's operand is re-streamed once per outer tile unless it fits
    // LRAM whole; compute and tile-fit are visit-order-invariant, so this trades
    // only DMA.
    const int64_t sM = llvm::divideCeil(M, tm);
    const int64_t sN = llvm::divideCeil(N, tn);
    const int64_t lramBits = getLramTilingBudget() * 8;
    const bool bResident = bitsB * K * N <= lramBits;
    const bool aResident = bitsA * M * K <= lramBits;
    const int64_t trafficMN =
        (bitsA * M * K + bitsB * (bResident ? K * N : K * N * sM) + bitsOut * M * N) / 8;
    const int64_t trafficNM =
        (bitsA * (aResident ? M * K : M * K * sN) + bitsB * K * N + bitsOut * M * N) / 8;

    LLVM_DEBUG({
        llvm::dbgs() << "  loop-order: for(m)for(n)=" << trafficMN
                     << "B  for(n)for(m)=" << trafficNM << "B  (s_m=" << sM << " s_n=" << sN
                     << " bits A/B=" << bitsA << "/" << bitsB << ")\n";
    });
    if (trafficNM >= trafficMN)
        return {};

    // Swap the two parallel loops; every other loop (reduction, batch) stays put.
    SmallVector<int64_t> interchange = llvm::to_vector(llvm::seq<int64_t>(0, tileSizes.size()));
    std::swap(interchange[mDomain], interchange[nDomain]);
    return interchange;
}

} // namespace mlir::syna::torq
