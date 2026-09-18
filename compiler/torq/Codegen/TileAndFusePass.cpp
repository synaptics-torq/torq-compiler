// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MatmulTilingHeuristics.h"
#include "PassesDetail.h"
#include "TileAndFuseUtils.h"
#include "TilingUtils.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OwningOpRef.h"
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
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <iomanip>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "torq-tile-and-fuse"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

/// Return true if two ops can be fused — they must have the same executor
/// assignment, or at least one must be unassigned (e.g. constants, casts).
static bool canFuse(Operation *consumer, Operation *producer) {
    // Ops without an executor assignment can fuse with anything.
    if (!consumer->hasAttr("torq-executor") || !producer->hasAttr("torq-executor"))
        return true;
    return getTargetExecutor(consumer) == getTargetExecutor(producer);
}

extern llvm::cl::opt<bool> clDisableSlicing;

namespace {

enum class TileAndFuseProducersFuseMode {
    MaxSize,
    MaxSizeAllDoms,
    MaxProducers,
    OnlyPatterns,
};

// Result of fitTileToMemory.
//
// The probe is stricter than the real pipeline. It skips the LRAM-residency
// recovery on purpose, because the recovery costs DMA and that DMA should not
// change the tile choice. It also models memory conservatively. So the probe
// can place less than the real pipeline can.
//
// That makes the answer one-way. A tile that fits the probe also fits the real
// pipeline. The reverse does not hold: a tile the probe rejects may still be
// fine. NoFit is that case, so callers must not read it as an error.
enum class TileFit {
    FitsUnchanged, // the first tile already fits
    FitsShrunk,    // a smaller tile fits
    NoFit,         // nothing fits, sizes holds the minimum
    ProbeFail,     // the probe itself broke, do not use sizes
};

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
            TileAndFuseProducersFuseMode::MaxSizeAllDoms, "internal-max-size-all-domains",
            "Do not use this option! Calculate an intermidate value used by the max-producers "
            "option internally"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::MaxProducers, "max-producers",
            "Prefer more producers over tile size: tile size is the biggest size that can still "
            "fit all the producers"
        ),
        clEnumValN(
            TileAndFuseProducersFuseMode::OnlyPatterns, "only-patterns",
            "Fuse producers only whene required to preserve patterns"
        )
    ),
    llvm::cl::init(TileAndFuseProducersFuseMode::MaxProducers) // Default value
);

static llvm::cl::opt<unsigned> clTorqTileAndFuseDistanceLimit(
    "torq-tile-and-fuse-distance-limit",
    llvm::cl::desc("Limit the (linalg) distance of producers that are fused (use 0 for no limit)."),
    llvm::cl::init(0) // Default value
);

static llvm::cl::opt<bool> clDisableFitShrinkReorder(
    "torq-disable-fit-shrink-reorder",
    llvm::cl::desc("Disable reduction-aware domain ordering in fitTileToMemory's shrink pass "
                   "(fall back to legacy tilingOrder shrinking)"),
    llvm::cl::init(false) // Default value
);

const std::string TORQ_TNF_DISTANCE = "torq-tnf-distance";

void setSourcesDistance(Operation *op, std::optional<int64_t> distance) {
    if (!distance)
        return;

    for (auto operand : op->getOperands()) {
        if (auto srcOp = operand.getDefiningOp()) {
            srcOp->setAttr(
                TORQ_TNF_DISTANCE,
                IntegerAttr::get(
                    IntegerType::get(op->getContext(), 64, IntegerType::Signed), *distance - 1
                )
            );
        }
    }
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
bool isSdimFriendlyTileSize(int64_t iterDomainSize, int64_t tileSize) {
    if (!isSdimFriendlyCount(tileSize))
        return false;

    int64_t remainder = iterDomainSize % tileSize;
    return (remainder == 0 || isSdimFriendlyCount(remainder));
}

int64_t makeSdimFriendlyTileSize(int64_t iterDomainSize, int64_t tileSize) {
    // A non-positive size is already invalid (e.g. the "too small" sentinel that
    // makeByteAlignedTileSize returns when composed via getFixpointF below); pass
    // it through instead of searching (and tripping the assert).
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

// The number of sub-byte values that together occupy a whole number of bytes
// for `type`, i.e. the tile-size granularity that keeps every tile boundary
// byte-aligned. For widths that divide a byte this is just values-per-byte
// (i1 -> 8, i2 -> 4, i4 -> 2). For widths that don't (i6) it is the smallest
// group that lands on a byte boundary: 4 i6 values = 24 bits = 3 bytes -> 4.
// Returns 1 for anything that is not a sub-byte integer (>= 8 bits), as those
// already fill whole bytes and need no sub-byte alignment.
int64_t typeValuesPerByte(Type type) {
    auto shapedType = dyn_cast<ShapedType>(type);
    if (!shapedType)
        return 1;
    auto intType = dyn_cast<IntegerType>(shapedType.getElementType());
    if (!intType)
        return 1;
    unsigned width = intType.getWidth();
    if (width == 0 || width >= 8)
        return 1;
    // Smallest number of values whose total bits are a multiple of 8, so a tile
    // that is a multiple of it always starts and ends on a byte boundary.
    return 8 / std::gcd(8u, width);
}

// How many sub-byte values are packed into a single byte for `op` and the
// producers that get tiled together with it.
//
// Some quantized weights use sub-byte integer types: e.g. i4 packs 2 values per
// byte, i2 packs 4, i1 packs 8. The matmul itself works on the *dequantized*
// (bf16) weight, so the sub-byte types live on its producers (the dequant
// generic and the reshape ops upstream). We therefore look back through the
// producers feeding `op`, not just `op` itself.
//
// The walk is bounded to `op`'s fuse group: we only follow a producer that
// shares the fuse group, so we stay inside this tile and never wander into the
// activation input or another matmul's tile. The leaf operand types are still
// inspected before that check, so the packed weight is counted even though its
// binding/load is outside the group.
//
// Returns 1 when nothing sub-byte is found (meaning "every value already fills a
// whole byte, no alignment needed"). When several sub-byte widths are present we
// keep the largest pack factor, because that's the strictest alignment.
int64_t getSubByteValuesPerByte(Operation *op) {
    int64_t valuesPerByte = 1;

    SmallVector<Operation *> queue = {op};
    DenseSet<Operation *> visitedOps = {op};

    while (!queue.empty()) {
        Operation *currentOp = queue.pop_back_val();

        for (Value operand : currentOp->getOperands()) {
            valuesPerByte = std::max(valuesPerByte, typeValuesPerByte(operand.getType()));

            Operation *producerOp = operand.getDefiningOp();
            if (!producerOp || visitedOps.contains(producerOp))
                continue;

            // Stay inside this tile: only follow producers that share `op`'s
            // fuse group. Anything else (the activation input, another tile)
            // hands `op` a finished value and does not constrain it.
            if (!checkShareFuseGroup(op, producerOp))
                continue;

            visitedOps.insert(producerOp);
            queue.push_back(producerOp);
        }
    }

    return valuesPerByte;
}

// Round a tile size DOWN so it covers a whole number of bytes for sub-byte
// types.
//
// Sub-byte values are packed two-or-more to a byte (i4 -> 2 per byte). The DMA
// can only address whole bytes, so a tile that started on an odd value would
// begin "half a byte in", which the descriptor can't express (this is what
// broke bufferization). We avoid that by keeping every tile size a multiple of
// `valuesPerByte`; tile starts are multiples of the tile size, so byte-aligned
// tile sizes give byte-aligned tile boundaries.
//
// We only ever round *down* (shrinking keeps the "fits in memory" guarantee).
// When `size` is smaller than one whole byte's worth of values (0 < size <
// valuesPerByte) there is no byte-aligned tile that small, so we return 0 to
// signal "too small".
int64_t makeByteAlignedTileSize(int64_t valuesPerByte, int64_t domainSize, int64_t size) {
    if (valuesPerByte <= 1 || domainSize < valuesPerByte || size <= 0)
        return size;
    int64_t remainder = size % valuesPerByte;
    if (remainder == 0)
        return size;
    // size - remainder is 0 exactly when size < valuesPerByte, i.e. too small
    // to hold one whole byte's worth of values. We return that 0 as-is.
    return size - remainder;
}

TilingInfo getTilingInfo(TilingInterface tilingInterfaceOp, int64_t sliceCount) {
    TilingInfo tilingInfo;

    auto loopIteratorTypes = tilingInterfaceOp.getLoopIteratorTypes();

    for (size_t domain = 0; domain < loopIteratorTypes.size(); ++domain) {
        if (loopIteratorTypes[domain] == utils::IteratorType::parallel)
            tilingInfo.tilingOrder.insert(domain);
    }

    OpBuilder builder(tilingInterfaceOp->getContext());
    auto [iterDomainOffsets, iterDomainSizes, iterDomainStrides] =
        getOffsetsSizesAndStrides(tilingInterfaceOp.getIterationDomain(builder));

    std::optional<SmallVector<int64_t>> iterDomainConstSizes =
        getConstantIntValues(iterDomainSizes);
    assert(iterDomainConstSizes && "iteration domain sizes are not constants");

    Operation *principalOp = isMarkedFuseGroup(tilingInterfaceOp)
                                 ? getFuseGroupPrincipalOpBackward(tilingInterfaceOp)
                                 : tilingInterfaceOp;
    assert(principalOp != nullptr && "could not find the principal op of the fuse group");

    tilingInfo.minSize.append(loopIteratorTypes.size(), 1);
    if (!clDisableSlicing && sliceCount > 1) {
        if (isa<linalg::Conv2DNhwcHwcfOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::Conv2DNhwcHwcfOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::Conv2DNchwFchwOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::Conv2DNchwFchwOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::DepthwiseConv2DNhwcHwcOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::DepthwiseConv2DNchwChwOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::DepthwiseConv2DNchwChwOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::PoolingNhwcMaxOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::PoolingNhwcMaxOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::PoolingNchwMaxOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::PoolingNchwMaxOp] =
                sliceCount * kGrouping;
        }
        else if (isa<linalg::PoolingNcwMaxOp>(principalOp)) {
            tilingInfo.minSize[SlicingIterationDomainIndex::PoolingNcwMaxOp] =
                sliceCount * kGrouping;
        }
    }

    tilingInfo.adjustSize.reserve(loopIteratorTypes.size());
    for (size_t index = 0; index < loopIteratorTypes.size(); ++index) {
        auto domainSize = (*iterDomainConstSizes)[index];
        tilingInfo.adjustSize.push_back([=](int64_t size) {
            return makeSdimFriendlyTileSize(domainSize, size);
        });
    }

    // If any sub-byte (e.g. i4) weight feeds this op, the innermost tiled
    // dimension must be byte-aligned (a multiple of values-per-byte) so the DMA
    // never sees a half-byte offset. Only the contiguous (innermost) axis is
    // packed; the outer dimensions step over whole rows, so they are already
    // byte-aligned for any tile size.
    int64_t valuesPerByte = getSubByteValuesPerByte(tilingInterfaceOp);
    if (valuesPerByte > 1 && !tilingInfo.tilingOrder.empty()) {
        int64_t innerDim = tilingInfo.tilingOrder.back();
        auto domainSize = (*iterDomainConstSizes)[innerDim];
        auto makeByteAligned = [=](int64_t size) {
            return makeByteAlignedTileSize(valuesPerByte, domainSize, size);
        };
        tilingInfo.adjustSize[innerDim] =
            composeSizeAdjustments(makeByteAligned, tilingInfo.adjustSize[innerDim]);

        // Also raise the *minimum* tile size for this dim, mirroring the conv/pool
        // minSize handling above. The shrink pass in fitTileToMemory seeds each
        // domain directly from minSize (bypassing adjustSize), so without this it
        // would try a width-1 sub-byte tile (e.g. 640x1xi4) that the DMA can't
        // address on a byte boundary. getSmallestTileSize returns the smallest
        // byte-aligned, SDIM-friendly tile for this dim (2 for i4).
        tilingInfo.minSize[innerDim] = std::max(
            tilingInfo.minSize[innerDim], getSmallestTileSize(tilingInfo, innerDim, domainSize)
        );
    }

    /**********************************************************************************************
     * The commented code below forces convolutions with stride 2 to have even tile sizes for
     * the H channel
     **********************************************************************************************
     *
     * Operation *principalOp = isMarkedFuseGroup(tilingInterfaceOp)
     *                              ? getFuseGroupPrincipalOpBackward(tilingInterfaceOp)
     *                              : tilingInterfaceOp;
     *
     * assert(principalOp != nullptr && "could not find the principal op of the fuse group");
     *
     * auto getFixpointF = [](std::function<int64_t(int64_t)> f1, std::function<int64_t(int64_t)>
     *f2) { return [=](int64_t size) { int64_t newSize = f2(f1(size)); while (newSize != size) {
     *             size = newSize;
     *             newSize = f2(f1(size));
     *         }
     *         return newSize;
     *     };
     * };
     *
     * auto makeEven = [](int64_t size) {
     *     return size - (size % 2);
     * };
     *
     * if (auto conv = dyn_cast<linalg::Conv2DNchwFchwOp>(principalOp)) {
     *     if (llvm::all_of(conv.getStrides(), [](APInt value) { return value == 2; })) {
     *         tilingInfo.adjustSize[1] = getFixpointF(makeEven, tilingInfo.adjustSize[1]);
     *     }
     * }
     * else if (auto dw = dyn_cast<linalg::DepthwiseConv2DNchwChwOp>(principalOp)) {
     *     if (llvm::all_of(dw.getStrides(), [](APInt value) { return value == 2; }))
     *         tilingInfo.adjustSize[1] = getFixpointF(makeEven, tilingInfo.adjustSize[1]);
     * }
     * else if (auto pooling = dyn_cast<linalg::PoolingNchwSumOp>(principalOp)) {
     *     if (llvm::all_of(pooling.getStrides(), [](APInt value) { return value == 2; }))
     *         tilingInfo.adjustSize[1] = getFixpointF(makeEven, tilingInfo.adjustSize[1]);
     * }
     * else if (auto pooling = dyn_cast<linalg::PoolingNchwMaxOp>(principalOp)) {
     *     if (llvm::all_of(pooling.getStrides(), [](APInt value) { return value == 2; }))
     *         tilingInfo.adjustSize[1] = getFixpointF(makeEven, tilingInfo.adjustSize[1]);
     * }
     */
    return tilingInfo;
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

                // Don't collect tiled operations
                if (llvm::isa<scf::ForOp>(operandOp)) {
                    maybeInputs.push_back(operand);
                    continue;
                }

                if (llvm::isa<TilingInterface>(operandOp) &&
                    !checkShareFuseGroup(currentOp, operandOp)) {
                    maybeInputs.push_back(operand);
                    continue;
                }

                visitedOps.insert(operandOp);
                queue.push_back(operandOp);
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
    const std::string &moduleName, Operation *consumerOp, const TilingInfo &tilingInfo = {},
    const SetVector<Operation *> &producerOps = {}
) {
    MLIRContext *context = consumerOp->getContext();
    Location loc = consumerOp->getLoc();

    OpBuilder builder(context);

    ModuleOp moduleOp = ModuleOp::create(builder, loc, moduleName);
    builder.setInsertionPointToStart(moduleOp.getBody());

    SmallVector<Operation *> ops;
    SmallVector<Value> inputs;
    collectOpsForMemoryCheck(consumerOp, producerOps, ops, inputs);

    SmallVector<Type> inputTypes;
    inputTypes.reserve(tilingInfo.tilingOrder.size() + inputs.size());

    for (size_t i = 0; i < tilingInfo.tilingOrder.size(); ++i)
        inputTypes.push_back(builder.getIndexType());

    for (Value input : inputs)
        inputTypes.push_back(input.getType());

    FunctionType functionType = builder.getFunctionType(inputTypes, consumerOp->getResultTypes());

    // This counter is to give the dumps different names.
    static unsigned instanceCount = 0;
    std::ostringstream funcName;
    funcName << "extracted_ops_" << std::setw(3) << std::setfill('0') << instanceCount++;

    func::FuncOp funcOp = func::FuncOp::create(builder, loc, funcName.str(), functionType);
    funcOp.addEntryBlock();

    IRMapping extractionMap;

    // Map the inputs to the function args. This will result in the function
    // args driving the appropriate cloned ops, when we clone them.
    auto nonDomainArgs = llvm::make_range(
        funcOp.getFunctionBody().getArguments().begin() + tilingInfo.tilingOrder.size(),
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
    func::ReturnOp::create(builder, loc, clonedConsumerOp->getResults());

    return moduleOp;
}

// "symbolically" tile and fuse everything in the only function in `moduleOp`.
// Tile sizes are non-constant (hence "symbolically"), and are the first
// arguments of the only function in the module.
llvm::LogicalResult
tileModuleForMemoryCheck(ModuleOp moduleOp, const TilingInfo &tilingInfo, size_t tileSizesCount) {
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
        funcOp.getFunctionBody().args_begin() + tilingInfo.tilingOrder.size()
    );
    for (auto [domain, arg] : llvm::zip_equal(tilingInfo.tilingOrder, tileSizeArgs))
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

    return llvm::success();
}

class TileAndFusePass : public impl::TileAndFuseBase<TileAndFusePass> {
  private:
    // Normally one should use OpPassManager, and run it with the pass'
    // runPipeline. This is not possible in this case as it requires that only
    // operations nested under the current operation can be scheduled, and we
    // want to run on a module. Hence, we have to use PassManager. PassManager
    // requires a context to be constructed (and is not copy constructable).
    // Hence, we delay the construction until its first use, and we don't copy
    // it (will be constructed again in the new pass).
    std::unique_ptr<PassManager> assignAddressesPipelineOptimized_;
    std::unique_ptr<PassManager> assignAddressesPipelineNotOptimized_;

    // Holds all the untiled ops we have already tiled, so we don't tile them
    // again.
    llvm::DenseSet<Operation *> untiledTiledOps_;

  public:
    TileAndFusePass() {}

    explicit TileAndFusePass(const TileAndFuseOptions &options) {
        this->sliceCount = options.sliceCount;
    }

    TileAndFusePass(const TileAndFusePass &pass) : TileAndFuseBase(pass) {
        this->sliceCount = pass.sliceCount;
    }

    void runOnOperation() override;

  private:
    void initPipelines(MLIRContext *context);

    llvm::LogicalResult runAssignAddressesPipeline(ModuleOp moduleOp, bool optimizeForTileAndFuse);

    llvm::FailureOr<bool> checkModuleFitsInMemory(ModuleOp moduleOp, bool optimizeForTileAndFuse);

    llvm::FailureOr<bool> checkTileFitsInMemory(
        ModuleOp moduleOp, const TilingInfo &tilingInfo, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    );

    void tileAndFuse(TilingInterface tiOp, const DenseSet<TilingInterface> &toTileOps);

    LogicalResult searchTileSizeForDim(
        IRRewriter &rewriter, ModuleOp moduleOp, const TilingInfo &tilingInfo,
        MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes, size_t dim,
        int64_t iterDomainSize, int64_t minFactor, int64_t maxFactor
    );

    TileFit fitTileToMemory(
        Operation *consumerOp, const SetVector<Operation *> &producerOps,
        const TilingInfo &tilingInfo, ArrayRef<int64_t> iterDomainSizes,
        MutableArrayRef<OpFoldResult> offsets, MutableArrayRef<OpFoldResult> sizes
    );

    std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> fuseControlMaxSize(
        IRRewriter &rewriter, bool allDomains, const DenseSet<TilingInterface> &toTileOps,
        mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
        bool isDestinationOperand
    );

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseToSize(
        IRRewriter &rewriter, TilingInterface tilingInterfaceOp,
        llvm::ArrayRef<OpFoldResult> iterDomainSizes, SmallVector<OpFoldResult> tileSizes,
        TileAndFuseProducersFuseMode fuseMode,
        std::optional<llvm::SetVector<Operation *> *> producerOps,
        const DenseSet<TilingInterface> &toTileOps
    );
};

void TileAndFusePass::initPipelines(MLIRContext *context) {
    if (assignAddressesPipelineOptimized_ == nullptr) {
        assignAddressesPipelineOptimized_ = std::make_unique<PassManager>(context);

        if (failed(applyPassManagerCLOptions(*assignAddressesPipelineOptimized_)))
            assert(false);

        addPassesPostTileAndFuseUpToAssignLramAddresses(*assignAddressesPipelineOptimized_, true);
    }

    if (assignAddressesPipelineNotOptimized_ == nullptr) {
        assignAddressesPipelineNotOptimized_ = std::make_unique<PassManager>(context);

        if (failed(applyPassManagerCLOptions(*assignAddressesPipelineNotOptimized_)))
            assert(false);

        addPassesPostTileAndFuseUpToAssignLramAddresses(
            *assignAddressesPipelineNotOptimized_, false
        );
    }
}

llvm::LogicalResult
TileAndFusePass::runAssignAddressesPipeline(ModuleOp moduleOp, bool optimizeForTileAndFuse) {
    initPipelines(moduleOp->getContext());

    LLVM_DEBUG(llvm::dbgs() << "*** Running the pipeline for: " << moduleOp.getName() << "\n");

    auto result = llvm::failure();
    if (optimizeForTileAndFuse)
        result = assignAddressesPipelineOptimized_->run(moduleOp);
    else
        result = assignAddressesPipelineNotOptimized_->run(moduleOp);

    LLVM_DEBUG(
        llvm::dbgs() << "*** pipeline finished (" << (succeeded(result) ? "succeeded" : "failed")
                     << ")\n"
    );

    return result;
}

// NB: moduleOp is mutated by this function, and can't be used again.
llvm::FailureOr<bool>
TileAndFusePass::checkModuleFitsInMemory(ModuleOp moduleOp, bool optimizeForTileAndFuse) {
    bool failure = false;
    bool memoryOverflow = false;

    // The handler goes on the MLIRContext, which all worker threads share. But
    // these two flags belong to this call only. Sibling functions probe at the
    // same time, so skip diagnostics from other threads. Returning failure() is
    // safe. The engine tries handlers newest first and stops at the first one
    // that returns success, so the owner still gets it. This is what
    // mlir::ParallelDiagnosticHandler does.
    const uint64_t ownerThreadId = llvm::get_threadid();

    auto diagHandler = [&](mlir::Diagnostic &diag) -> LogicalResult {
        if (llvm::get_threadid() != ownerThreadId)
            return llvm::failure();

        if (memoryOverflow) {
            // If we already saw the OUT_OF_MEMORY_MESSAGE, suppress all messages.
            return llvm::success();
        }

        if (diag.str() == OUT_OF_MEMORY_MESSAGE) {
            memoryOverflow = true;
            // Signal that we are handling this issue (don't print error message).
            return llvm::success();
        }

        // Something else (other than the expected memory overflow) bad happened.
        failure = true;
        diag.append(" (encountered while running the pipeline to checking if a tile fits in memory)"
        );
        // Signal that we are not handling this issue (error message will be printed).
        return llvm::failure();
    };

    using DiagHandlerFn = std::function<LogicalResult(mlir::Diagnostic &)>;

    mlir::ScopedDiagnosticHandler diagHandlerRAII(
        moduleOp->getContext(),
        (optimizeForTileAndFuse ? DiagHandlerFn(diagHandler)
                                : DiagHandlerFn([&](mlir::Diagnostic &) {
                                      // Not reached today. Both callers pass true.
                                      // Kept the same so it cannot break the same way later.
                                      if (llvm::get_threadid() != ownerThreadId)
                                          return llvm::failure();
                                      failure = true;
                                      return llvm::failure();
                                  }))
    );

    if (failed(runAssignAddressesPipeline(moduleOp, optimizeForTileAndFuse))) {
        if (memoryOverflow)
            return false;

        // This assert is just to catch things early in debug
        LLVM_DEBUG(assert(!failure));

        if (failure)
            return llvm::failure();

        // The pipeline failed, but neither flag is set. This is the path the bug
        // ran through. A stolen diagnostic left the owner here, and falling
        // through said the tile fits. Return failure instead. Logging showed
        // this branch never runs today.
        return llvm::failure();
    }

    assert(!memoryOverflow && "this should have been captured above");

    return true;
}

// Return true iff the tile fits in memory
llvm::FailureOr<bool> TileAndFusePass::checkTileFitsInMemory(
    ModuleOp moduleOp, const TilingInfo &tilingInfo, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    OpBuilder builder(moduleOp->getContext());

    // Clone the module, as the check is distructive
    IRMapping map;
    OwningOpRef<ModuleOp> fixedModuleOp = cast<ModuleOp>(builder.clone(*moduleOp, map));

    auto funcOp = cast<func::FuncOp>(&fixedModuleOp->getBody()->front());
    // This counter is to give the dumps different names.
    static unsigned instanceCount = 0;
    std::ostringstream funcName;
    funcName << funcOp.getName().str() << "_" << std::setw(3) << std::setfill('0')
             << instanceCount++;
    funcOp.setName(funcName.str());

    builder.setInsertionPointToStart(&funcOp.getFunctionBody().front());
    auto tileSizeArgs = llvm::make_range(
        funcOp.getFunctionBody().args_begin(),
        funcOp.getFunctionBody().args_begin() + tilingInfo.tilingOrder.size()
    );
    for (auto [domain, arg] : llvm::zip_equal(tilingInfo.tilingOrder, tileSizeArgs)) {
        // NB1: sizes[domain] is a Value/Attribute in the original module, so we
        // need to copy it to the fixedModuleOp before we can use it there.
        // NB2: as far as I can tell, it is almost always an attribute, except
        // when called from fuseControlMaxSize/Producers, where it is an affine
        // op, in which case we will try to evaluate its value in the first tile (ignoring offsets).
        llvm::FailureOr<int64_t> maybeSize = computeSizeAtFirstIteration(sizes[domain]);
        if (failed(maybeSize)) {
            LLVM_DEBUG(assert(false));
            return llvm::failure();
        }

        Value size = arith::ConstantOp::create(
            builder, funcOp->getLoc(), builder.getIntegerAttr(builder.getIndexType(), *maybeSize)
        );

        arg.replaceAllUsesWith(size);
    }

    return checkModuleFitsInMemory(*fixedModuleOp, true);
}

// Binary-search for the largest tile size along a single iteration domain that fits in memory.
// Precondition: the tile fits when sizes[domain] = div_ceil(iterDomainSize, maxFactor),
//               and does not fit when sizes[domain] = div_ceil(iterDomainSize, minFactor).
// Postcondition: sizes[domain] is set to the largest fitting tile size.
LogicalResult TileAndFusePass::searchTileSizeForDim(
    IRRewriter &rewriter, ModuleOp moduleOp, const TilingInfo &tilingInfo,
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
    while (maxFactor > minFactor + 1) {
        int64_t midFactor = midpoint(minFactor, maxFactor);

        int64_t tileSize = div_ceil(iterDomainSize, midFactor);
        tileSize = tilingInfo.adjustSize[domain](tileSize);
        assert(tileSize != 0); // assuming the initial maxFactor was valid

        sizes[domain] = rewriter.getIndexAttr(tileSize);

        llvm::FailureOr<bool> tileFits =
            checkTileFitsInMemory(moduleOp, tilingInfo, offsets, sizes);
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
    tileSize = tilingInfo.adjustSize[domain](tileSize);
    assert(tileSize != 0); // assuming the initial maxFactor was valid

    sizes[domain] = rewriter.getIndexAttr(tileSize);

    return LogicalResult::success();
}

// Find a tile for `consumerOp` that the memory probe accepts, and write it to
// `sizes`. If the first tile is too big, shrink every domain to its minimum,
// then grow them back with binary search, as far as they still fit. See TileFit
// for the return values. NoFit is a normal answer, not an error.
TileFit TileAndFusePass::fitTileToMemory(
    Operation *consumerOp, const SetVector<Operation *> &producerOps, const TilingInfo &tilingInfo,
    ArrayRef<int64_t> iterDomainSizes, MutableArrayRef<OpFoldResult> offsets,
    MutableArrayRef<OpFoldResult> sizes
) {
    // The grow-back pass below bounds every domain by iterDomainSizes, so the
    // caller's candidate must be the full domain (a pre-shrunk candidate would
    // have its bound lifted).
    assert(llvm::all_of(llvm::enumerate(iterDomainSizes), [&](auto it) {
        return getConstantIntValue(sizes[it.index()]) == it.value();
    }));

    OwningOpRef<ModuleOp> moduleOp = extractOpsForMemoryCheck(
        "fit_tile_to_memory", cast<TilingInterface>(consumerOp), tilingInfo, producerOps
    );

    if (failed(tileModuleForMemoryCheck(*moduleOp, tilingInfo, iterDomainSizes.size())))
        return TileFit::ProbeFail;

    // First establish that the original tile is not big enough.
    llvm::FailureOr<bool> tileFits = checkTileFitsInMemory(*moduleOp, tilingInfo, offsets, sizes);
    if (failed(tileFits))
        return TileFit::ProbeFail;
    if (*tileFits) {
        // fits in memory, no need to change the tile
        return TileFit::FitsUnchanged;
    }

    IRRewriter rewriter(moduleOp->getContext());

    size_t growBackEnd = 0;
    const bool isMatmulTile = findTiledMatmul(consumerOp, producerOps) != nullptr;

    // Set when a shrink pass bailed out because the fit check itself failed
    // (as opposed to sweeping every domain without finding a fit).
    bool fitCheckFailed = false;

    // Shrink pass: set domains to their smallest valid tile, one by one, until
    // the tile fits. The smallest valid tile is usually 1, but for sub-byte
    // (e.g. i4) dimensions it is one byte's worth of values (2 for i4), because
    // a width-1 sub-byte slice can't be addressed on a byte boundary.
    //
    // Domain order: by default computeShrinkOrderByReduction() picks the domain
    // whose shrink removes the most group operand bytes first (e.g. the N dim
    // of a matmul, which sizes the weight tile, rather than the M dim, which
    // does not index the weights); --torq-disable-fit-shrink-reorder restores
    // the legacy tilingOrder.
    //
    // The fallback pass continues from the first pass's all-minSize tile rather
    // than restarting from the full domain: that tile is already known not to
    // fit, and restarting would replay every first-pass fit check (each a full
    // LRAM-allocator run) for domains whose minSize equals the smallest size.
    auto shrinkPass = [&](bool fallback) {
        growBackEnd = 0;
        fitCheckFailed = false;

        SmallVector<int64_t> order(tilingInfo.tilingOrder.begin(), tilingInfo.tilingOrder.end());
        if (!clDisableFitShrinkReorder) {
            if (std::optional<SmallVector<int64_t>> reordered = computeShrinkOrderByReduction(
                    consumerOp, producerOps, tilingInfo, iterDomainSizes, sizes, fallback
                ))
                order = std::move(*reordered);
        }

        for (int64_t domain : order) {
            int64_t minTileSize =
                fallback ? getSmallestTileSize(tilingInfo, domain, iterDomainSizes[domain])
                         : tilingInfo.minSize[domain];
            if (getConstantIntValue(sizes[domain]) == minTileSize)
                continue;

            auto position = llvm::find(tilingInfo.tilingOrder, domain);
            growBackEnd = std::max(
                growBackEnd, size_t(std::distance(tilingInfo.tilingOrder.begin(), position)) + 1
            );
            sizes[domain] = rewriter.getIndexAttr(minTileSize);
            tileFits = checkTileFitsInMemory(*moduleOp, tilingInfo, offsets, sizes);
            if (failed(tileFits)) {
                fitCheckFailed = true;
                return LogicalResult::failure();
            }

            if (*tileFits)
                return LogicalResult::success();
        }
        return LogicalResult::failure();
    };

    // We try the shrinking pass twice, once with the preferred minSize, and if that is not small
    // enough, we try again with 1's.
    if (shrinkPass(false).failed()) {
        if (fitCheckFailed)
            return TileFit::ProbeFail;

        if (shrinkPass(true).failed()) {
            if (fitCheckFailed)
                return TileFit::ProbeFail;

            // Every domain is at its minimum and still nothing fits. That does
            // not make the op impossible. The probe is stricter than the real
            // pipeline (see TileFit), so a block it rejects at every tile size
            // can still compile.
            //
            // Keep the minimum tile in `sizes` and let the caller decide. Return
            // here: the grow-back loop below needs a tile that fits, and would
            // read a past-the-end iterator.
            LLVM_DEBUG(llvm::dbgs() << "  no probe tile fits; keeping the minimum tile\n");
            return TileFit::NoFit;
        }
    }

    // Grow-back pass: binary-search every shrunk domain up to the largest fitting
    // tile, walking tilingOrder in reverse. Under desc-size ordering that grows
    // the smallest domain first, landing near the maximum Tm*Tn (minimum tile
    // count); sizes only grow here, so a single reverse sweep is a fixpoint.
    // Preserve the legacy grow-back boundary for non-matmul tiles, particularly
    // conv/pool fallback: only revisit domains through the last one shrunk in
    // the successful pass. Matmuls also revisit the first pass's N floor.
    if (isMatmulTile)
        growBackEnd = tilingInfo.tilingOrder.size();
    auto growBackDomains = tilingInfo.tilingOrder.getArrayRef().take_front(growBackEnd);
    for (int64_t domain : llvm::reverse(growBackDomains)) {
        if (iterDomainSizes[domain] == 1)
            continue;
        if (getConstantIntValue(sizes[domain]) == iterDomainSizes[domain])
            continue;

        int64_t maxFactor = div_ceil(iterDomainSizes[domain], *getConstantIntValue(sizes[domain]));
        sizes[domain] = rewriter.getIndexAttr(iterDomainSizes[domain]);
        tileFits = checkTileFitsInMemory(*moduleOp, tilingInfo, offsets, sizes);
        if (failed(tileFits))
            return TileFit::ProbeFail;
        if (*tileFits)
            continue;

        // The full domain failed, while the pre-growth size fits. Bound the
        // search by that known fitting size instead of starting over at 1.
        if (failed(searchTileSizeForDim(
                rewriter, *moduleOp, tilingInfo, offsets, sizes, domain, iterDomainSizes[domain],
                /*minFactor=*/1, maxFactor
            )))
            return TileFit::ProbeFail;
    }

    return TileFit::FitsShrunk;
}

bool shouldFuseMultiUsersOp(Operation *op) {
    if (std::distance(op->getUsers().begin(), op->getUsers().end()) <= 2)
        return true;

    if (llvm::all_of(op->getOperands(), [](Value operand) {
            return llvm::isa_and_nonnull<
                arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp,
                arith::ConstantFloatOp>(operand.getDefiningOp());
        }))
        return true;

    return false;
}

// an SCFTileAndFuseOptions::ControlFnTy for the max-size fuse mode.
std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> TileAndFusePass::fuseControlMaxSize(
    IRRewriter &rewriter, bool allDomains, const DenseSet<TilingInterface> &toTileOps,
    mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand
) {
    Operation *producerOp = producerOpResult.getOwner();

    // Refuse to fuse across executor boundaries (NSS vs Host etc).
    Operation *consumerOp = *candidateSliceOp->getUsers().begin();
    if (!canFuse(consumerOp, producerOp))
        return std::nullopt;

    auto doNotFuse = [&]() {
        assert(
            !isMarkedFuseGroup(producerOp) ||
            isFuseGroupOutput(producerOp) && "Unable to fuse a pattern fuse group member"
        );
        return std::nullopt;
    };

    const scf::SCFTileAndFuseOptions::ControlFnResult fuseAndDoNotYieldProducer{false};

    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi)
        return doNotFuse();

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    std::optional<int64_t> distance;
    if (auto distanceAttr = producerOp->getAttrOfType<mlir::IntegerAttr>(TORQ_TNF_DISTANCE))
        distance = distanceAttr.getSInt();

    // Must come before the operand-slice computation below: slicing fails for some producers
    // (e.g. a reduction), and doNotFuse() on a group member the driver cannot skip loops forever.
    if (isMarkedFuseGroup(producerOp) && !isFuseGroupOutput(producerOp)) {
        setSourcesDistance(producerOp, distance);
        return fuseAndDoNotYieldProducer;
    }

    // Get the producer's operand tiles
    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        producerOp->emitWarning(
            "tile-and-fuse: failed to compute producer's operand slices, skipping producer."
        );
        // LLVM_DEBUG(assert(false));
        return doNotFuse();
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    TilingInfo tilingInfo = getTilingInfo(producerTi, this->sliceCount);

    if (!toTileOps.contains(producerTi) && !isa<linalg::FillOp>(producerOp)) {
        return doNotFuse();
    }

    if (!shouldFuseMultiUsersOp(producerOp)) {
        return doNotFuse();
    }

    if (distance && *distance <= 0)
        return doNotFuse();

    if (!allDomains) {
        for (size_t index = 0; index < iterSizes.size(); ++index) {
            if (mappedSizes[index] != iterSizes[index]) {
                // Don't fuse if we are tiling a domain the producer can not be tiled over
                if (!tilingInfo.tilingOrder.contains(index))
                    return doNotFuse();

                // If the required size needs adjusting, we can't fuse this producer
                llvm::FailureOr<int64_t> maybeSize =
                    computeSizeAtFirstIteration(mappedSizes[index]);
                if (failed(maybeSize)) {
                    LLVM_DEBUG(assert(false));
                    return doNotFuse();
                }
                if (tilingInfo.adjustSize[index](*maybeSize) != *maybeSize)
                    return doNotFuse();
            }
        }
    }

    OwningOpRef<ModuleOp> moduleOp =
        extractOpsForMemoryCheck("fuse_control_max_size_output", producerTi, tilingInfo);
    if (failed(tileModuleForMemoryCheck(*moduleOp, tilingInfo, iterSizes.size())))
        assert(false && "failed to compute producer's tiled memory size");

    llvm::FailureOr<bool> producerFits =
        checkTileFitsInMemory(*moduleOp, tilingInfo, mappedOffsets, mappedSizes);
    if (failed(producerFits)) {
        producerOp->emitWarning("tile-and-fuse: failed to check if tiled "
                                "producer overflows, skipping producer.");
        LLVM_DEBUG(assert(false));
        return doNotFuse();
    }
    if (!*producerFits)
        return doNotFuse();

    // Fuse iff producer fits in the tile
    return fuseAndDoNotYieldProducer;
}

// an SCFTileAndFuseOptions::ControlFnTy for the max-producers fuse mode.
std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> fuseControlMaxProducers(
    IRRewriter &rewriter, std::optional<llvm::SetVector<Operation *> *> producerOps,
    mlir::tensor::ExtractSliceOp candidateSliceOp, OpResult producerOpResult,
    bool isDestinationOperand, int64_t sliceCount
) {
    Operation *producerOp = producerOpResult.getOwner();

    // Refuse to fuse across executor boundaries (NSS vs Host etc).
    Operation *consumerOp = *candidateSliceOp->getUsers().begin();
    if (!canFuse(consumerOp, producerOp))
        return std::nullopt;

    auto doNotFuse = [&]() {
        assert(
            !isMarkedFuseGroup(producerOp) ||
            isFuseGroupOutput(producerOp) && "Unable to fuse a pattern fuse group member"
        );
        return std::nullopt;
    };

    const scf::SCFTileAndFuseOptions::ControlFnResult fuseAndDoNotYieldProducer{false};

    auto producerTi = dyn_cast<TilingInterface>(producerOp);
    if (!producerTi)
        return doNotFuse();

    if (producerOps && !(*producerOps)->contains(producerOp))
        return doNotFuse();

    // FIXME: should we not fuse destination operands?
    // if (isDestinationOperand) {
    //     return doNotFuse;
    // }

    std::optional<int64_t> distance;
    if (auto distanceAttr = producerOp->getAttrOfType<mlir::IntegerAttr>(TORQ_TNF_DISTANCE))
        distance = distanceAttr.getSInt();

    // Before the operand-slice computation below, as in fuseControlMaxSize.
    if (isMarkedFuseGroup(producerOp) && !isFuseGroupOutput(producerOp)) {
        setSourcesDistance(producerOp, distance);
        return fuseAndDoNotYieldProducer;
    }

    SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
    if (failed(producerTi.getIterationDomainTileFromResultTile(
            rewriter, producerOpResult.getResultNumber(), candidateSliceOp.getMixedOffsets(),
            candidateSliceOp.getMixedSizes(), mappedOffsets, mappedSizes
        ))) {
        producerOp->emitWarning(
            "tile-and-fuse: failed to compute producer's operand slices, skipping producer."
        );
        // LLVM_DEBUG(assert(false));
        return doNotFuse();
    }

    auto [iterOffsets, iterSizes, iterStrides] =
        getOffsetsSizesAndStrides(producerTi.getIterationDomain(rewriter));

    assert(
        mappedSizes.size() == iterSizes.size() && "expected mapped and iter sizes to be the same"
    );

    TilingInfo tilingInfo = getTilingInfo(producerTi, sliceCount);

    for (size_t index = 0; index < iterSizes.size(); ++index) {
        if (mappedSizes[index] != iterSizes[index]) {
            // Don't fuse if we are tiling a domain the producer can not be tiled over
            if (!tilingInfo.tilingOrder.contains(index))
                return doNotFuse();

            // If the required size needs adjusting, we can't fuse this producer
            llvm::FailureOr<int64_t> maybeSize = computeSizeAtFirstIteration(mappedSizes[index]);
            if (failed(maybeSize)) {
                LLVM_DEBUG(assert(false));
                return doNotFuse();
            }
            if (tilingInfo.adjustSize[index](*maybeSize) != *maybeSize)
                return doNotFuse();
        }
    }

    if (distance && *distance <= 0)
        return doNotFuse();
    setSourcesDistance(producerOp, distance);

    // Producer can be fused
    return fuseAndDoNotYieldProducer;
}

FailureOr<scf::SCFTileAndFuseResult> TileAndFusePass::tileAndFuseToSize(
    IRRewriter &rewriter, TilingInterface tilingInterfaceOp,
    llvm::ArrayRef<OpFoldResult> iterDomainSizes, SmallVector<OpFoldResult> tileSizes,
    TileAndFuseProducersFuseMode fuseMode,
    std::optional<llvm::SetVector<Operation *> *> producerOps,
    const DenseSet<TilingInterface> &toTileOps
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

    // Visit matmul tiles for(n)for(m) when that streams fewer bytes. Not for the
    // producer-discovery probe (MaxSizeAllDoms), whose nest is discarded.
    if (fuseMode != TileAndFuseProducersFuseMode::MaxSizeAllDoms) {
        static const llvm::SetVector<Operation *> noProducers;
        SmallVector<int64_t> interchange = chooseMatmulLoopInterchange(
            tilingInterfaceOp.getOperation(), tileSizes, producerOps ? **producerOps : noProducers
        );
        if (!interchange.empty()) {
            options.tilingOptions.setInterchange(interchange);
            LLVM_DEBUG(llvm::dbgs() << "  loop order: for(n)for(m) [reversed]\n");
        }
    }

    scf::SCFTileAndFuseOptions::ControlFnTy fusionControlFn;
    switch (fuseMode) {
    case TileAndFuseProducersFuseMode::MaxSize:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxSize(
                rewriter, false, toTileOps, candidateSliceOp, producerOpResult, isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::MaxSizeAllDoms:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxSize(
                rewriter, true, toTileOps, candidateSliceOp, producerOpResult, isDestinationOperand
            );
        };
        break;

    case TileAndFuseProducersFuseMode::MaxProducers:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand) {
            return fuseControlMaxProducers(
                rewriter, producerOps, candidateSliceOp, producerOpResult, isDestinationOperand,
                this->sliceCount
            );
        };
        break;

    case TileAndFuseProducersFuseMode::OnlyPatterns:
        fusionControlFn = [&](mlir::tensor::ExtractSliceOp candidateSliceOp,
                              OpResult producerOpResult, bool isDestinationOperand
                          ) -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
            bool shouldFuse = checkShareFuseGroup(tilingInterfaceOp, producerOpResult.getOwner());

            if (!shouldFuse) {
                return std::nullopt;
            }

            return scf::SCFTileAndFuseOptions::ControlFnResult{false};
        };
        break;
    }

    options.setFusionControlFn(fusionControlFn);

    if (clTorqTileAndFuseDistanceLimit.getValue() != 0) {
        setSourcesDistance(tilingInterfaceOp, clTorqTileAndFuseDistanceLimit.getValue());
    }

    return scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp, options);
}

void TileAndFusePass::tileAndFuse(
    TilingInterface tiOp, const DenseSet<TilingInterface> &toTileOps
) {
    LLVM_DEBUG({
        llvm::dbgs() << tiOp->getName() << " needs to be tiled\n";
        llvm::dbgs() << "  " << TorqHw::get().getLramSize() << " bytes of LRAM\n";

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

    TilingInfo tilingInfo = getTilingInfo(tiOp, this->sliceCount);

    llvm::SetVector<Operation *> producerOps;
    std::optional<llvm::SetVector<Operation *> *> restrictToProducerOps = std::nullopt;

    // The fuse mode used for the actual tiling below. Starts as the
    // command-line selection, but may fall back to OnlyPatterns if
    // MaxProducers cannot fit the candidate producers in memory.
    TileAndFuseProducersFuseMode fuseMode = clTorqTileAndFuseProducersFuseMode.getValue();

    // In MaxProducers mode, first find all the potential producers that fit in
    // the smallest tile. As we don't know which domains will actually be tiled
    // yet, we use the MaxSizeAllDoms option below, that ignores domain order
    // constraints.
    if (fuseMode == TileAndFuseProducersFuseMode::MaxProducers) {

        SmallVector<OpFoldResult> smallestTileSizes(iterDomainSizes);
        for (auto dim : tilingInfo.tilingOrder) {
            // Smallest valid tile per domain: 1 normally, but one byte's worth of
            // values for sub-byte (e.g. i4) dimensions, so we never form a
            // width-1 sub-byte slice that can't sit on a byte boundary.
            smallestTileSizes[dim] = rewriter.getIndexAttr(
                getSmallestTileSize(tilingInfo, dim, (*iterDomainConstSizes)[dim])
            );
        }

        FailureOr<scf::SCFTileAndFuseResult> tiledResults = tileAndFuseToSize(
            rewriter, tiOp, iterDomainSizes, smallestTileSizes,
            TileAndFuseProducersFuseMode::MaxSizeAllDoms, std::nullopt, toTileOps
        );
        if (failed(tiledResults)) {
            tiOp->emitWarning("tile-and-fuse: failed to tile operation, skipping it.");
            LLVM_DEBUG(assert(false));
            return;
        }

        producerOps = tiledResults->fusedProducers;
        restrictToProducerOps = &producerOps;

        // The probe nest survives only as dead code, but its slice ops keep phantom
        // uses on the producers' results; erase it so use counts consulted during
        // the real tiling see the real graph.
        if (!tiledResults->loops.empty() && tiledResults->loops.front()->use_empty())
            rewriter.eraseOp(tiledResults->loops.front());
    }

    auto includePatternProducers = [&]() {
        SmallVector<Operation *> worklist{tiOp.getOperation()};
        while (!worklist.empty()) {
            Operation *current = worklist.pop_back_val();
            for (Value operand : current->getOperands()) {
                Operation *producer = operand.getDefiningOp();
                if (producer && checkShareFuseGroup(tiOp, producer) && producerOps.insert(producer))
                    worklist.push_back(producer);
            }
        }
        // MaxSize and OnlyPatterns ignore the restriction for fusion, but loop
        // selection still needs the required producers.
        restrictToProducerOps = &producerOps;
    };
    includePatternProducers();

    // The matmul heuristics need to know which producers the tile will fuse,
    // hence after the probe rather than in getTilingInfo.
    applyMatmulTilingHeuristics(tilingInfo, tiOp, *iterDomainConstSizes, producerOps);

    // Find a tile size that fits tiOp in memory, together with all the
    // candidate producers in MaxProducers mode (pattern-fuse-group members are
    // always included by extractOpsForMemoryCheck).
    TileFit tileFit = fitTileToMemory(
        tiOp, producerOps, tilingInfo, *iterDomainConstSizes, tileOffsets, tileSizes
    );

    // MaxProducers fallback: if the consumer tile plus all candidate producers
    // can't fit in LRAM even at the smallest tile size, give up on fusing
    // optional producers and retry with OnlyPatterns, which only fuses what
    // pattern-fuse-groups strictly require.
    //
    // NoFit comes here too, not only ProbeFail. It is the same case: even
    // the smallest tile does not fit. Fusing fewer producers may find a tile
    // that really fits, and OnlyPatterns needs less memory either way. So this
    // retry either finds a real fit or leaves the recovery more room.
    if ((tileFit == TileFit::ProbeFail || tileFit == TileFit::NoFit) &&
        fuseMode == TileAndFuseProducersFuseMode::MaxProducers) {
        LLVM_DEBUG(llvm::dbgs() << "  max-producers does not fit, falling back to only-patterns\n");
        fuseMode = TileAndFuseProducersFuseMode::OnlyPatterns;
        producerOps.clear();
        includePatternProducers();
        tileOffsets.assign(iterDomainOffsets.begin(), iterDomainOffsets.end());
        tileSizes.assign(iterDomainSizes.begin(), iterDomainSizes.end());
        tilingInfo = getTilingInfo(tiOp, this->sliceCount);
        applyMatmulTilingHeuristics(tilingInfo, tiOp, *iterDomainConstSizes, producerOps);
        tileFit = fitTileToMemory(
            tiOp, producerOps, tilingInfo, *iterDomainConstSizes, tileOffsets, tileSizes
        );
    }

    switch (tileFit) {
    case TileFit::ProbeFail:
        tiOp->emitWarning("tile-and-fuse: failed to tile an operation, skipping it");
        LLVM_DEBUG(assert(false));
        return;

    // FIXME: this should be an assert. An untiled op that does not fit in memory
    // cannot fit as a single tile either. It still fires because Conv2dConvert
    // calls foldBackwardPadding when convertToInterleaved fails, after marking
    // has already returned. Fix that, then drop this case.
    // $ pytest
    // tests/test_keras_app.py::test_keras_app_tflite_torq[layer_inceptionv3_conv2d_13-sim-xxx-v4]
    // --torq-chips=next.group
    case TileFit::FitsUnchanged:
        return;

    case TileFit::NoFit:
        // Tile to the minimum and let the real pipeline decide (see TileFit).
        // Leaving the op untiled would be worse: an untiled op is never smaller
        // than the minimum tile. If the real pipeline also fails, the user gets
        // the real allocation error, which says more than a failure here.
        tiOp->emitRemark("tile-and-fuse: no tile fits the memory probe; using the minimum tile "
                         "and leaving the decision to the real pipeline");
        break;

    case TileFit::FitsShrunk:
        break;
    }

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
        rewriter, tiOp, iterDomainSizes, tileSizes, fuseMode, restrictToProducerOps, toTileOps
    );
    if (failed(tiledResults)) {
        tiOp->emitWarning("tile-and-fuse: failed to tile operation, skipping it.");
        LLVM_DEBUG(assert(false));
        return;
    }

    LLVM_DEBUG({
        llvm::dbgs() << "  fused " << tiledResults->fusedProducers.size() << " producers\n";
        for (Operation *producerOp : tiledResults->fusedProducers) {
            llvm::dbgs() << "  | " << producerOp->getName() << "\n";
        }
    });

    // Replace the untiled tiOp with the tiled results.
    applyTiledResults(rewriter, tiOp, *tiledResults);

    // Erase the untiled op, so its producers will have use_empty (and not tiled
    // again) if this is their only user.
    assert(tiOp->use_empty() && "tiled operation still has users");
}

// Match the reduction body shape every ONNX-imported reduction has: a single two-operand combine
// op (addf, maximumf, ...) reading the block arguments, plus the yield. Returns the combine op, or
// null for anything more exotic. This contract is what lets rewriteKeepdimsFuseGroupReduction
// rebuild the body at a wider accumulator type instead of retyping arbitrary cloned ops.
static Operation *matchSingleCombineReduction(linalg::GenericOp op) {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
        return nullptr;
    if (!llvm::is_contained(op.getIteratorTypesArray(), utils::IteratorType::reduction))
        return nullptr;

    Block &body = op.getRegion().front();
    Operation *combine = &body.front();
    auto yield = cast<linalg::YieldOp>(body.getTerminator());
    if (body.getOperations().size() != 2 || combine->getNumOperands() != 2 ||
        combine->getNumResults() != 1 || yield.getOperand(0) != combine->getResult(0) ||
        !llvm::all_of(combine->getOperands(), llvm::IsaPred<BlockArgument>))
        return nullptr;
    return combine;
}

// Split a keepdims output map into kept positions (AffineDimExpr) and dropped ones
// (AffineConstantExpr(0) at a unit dim). Fails on any other result form, and when nothing is
// dropped (the map is already a projected permutation) or nothing is kept.
static LogicalResult splitKeepdimsOutputMap(
    AffineMap outMap, RankedTensorType outType, SmallVector<int64_t> &keptPos,
    SmallVector<AffineExpr> &keptExprs
) {
    for (int64_t i = 0; i < (int64_t)outMap.getNumResults(); ++i) {
        AffineExpr e = outMap.getResult(i);
        if (auto c = dyn_cast<AffineConstantExpr>(e)) {
            if (c.getValue() != 0 || outType.getDimSize(i) != 1)
                return failure();
        }
        else if (isa<AffineDimExpr>(e)) {
            keptPos.push_back(i);
            keptExprs.push_back(e);
        }
        else {
            return failure();
        }
    }
    if (keptExprs.empty() || (int64_t)keptExprs.size() == outType.getRank())
        return failure();
    return success();
}

// Reassociation between the keepdims shape and the reduced shape: each kept dim starts a group;
// dropped unit dims join the current group (or the first, if they precede any kept dim), so every
// group holds exactly one kept dim.
static SmallVector<ReassociationIndices>
keepdimsReassociation(ArrayRef<int64_t> keptPos, int64_t rank) {
    SmallVector<ReassociationIndices> reassoc(keptPos.size());
    int64_t g = -1;
    for (int64_t i = 0; i < rank; ++i) {
        if (llvm::is_contained(keptPos, i))
            ++g;
        reassoc[g < 0 ? 0 : g].push_back(i);
    }
    return reassoc;
}

// Elementwise extf/truncf cast built as a linalg.generic so it tiles with the fuse group; the
// direction is inferred from the element widths.
static linalg::GenericOp
createTileableCast(IRRewriter &rewriter, Location loc, Value src, RankedTensorType dstType) {
    Type srcElem = cast<RankedTensorType>(src.getType()).getElementType();
    Type dstElem = dstType.getElementType();
    bool extend = dstElem.getIntOrFloatBitWidth() > srcElem.getIntOrFloatBitWidth();
    AffineMap idMap = rewriter.getMultiDimIdentityMap(dstType.getRank());
    SmallVector<utils::IteratorType> parIters(dstType.getRank(), utils::IteratorType::parallel);
    Value empty = tensor::EmptyOp::create(rewriter, loc, dstType.getShape(), dstElem);
    return linalg::GenericOp::create(
        rewriter, loc, TypeRange{dstType}, ValueRange{src}, ValueRange{empty},
        ArrayRef<AffineMap>{idMap, idMap}, parIters,
        [&](OpBuilder &b, Location l, ValueRange args) {
            Value v = extend ? arith::ExtFOp::create(b, l, dstElem, args[0]).getResult()
                             : arith::TruncFOp::create(b, l, dstElem, args[0]).getResult();
            linalg::YieldOp::create(b, l, v);
        }
    );
}

// Recreate op's reduction with the reduced (non-keepdims) output map, rebuilding the combine op at
// the accumulator type and widening the incoming element when the accumulator is wider.
static linalg::GenericOp createReducedReduction(
    IRRewriter &rewriter, Location loc, linalg::GenericOp op, Operation *combine, Value init,
    RankedTensorType accType, AffineMap reducedOutMap
) {
    Type accTy = accType.getElementType();
    Block &body = op.getRegion().front();
    AffineMap inMap = op.getIndexingMapsArray().front();
    auto buildBody = [&](OpBuilder &b, Location l, ValueRange args) {
        Value in = args[0].getType() == accTy
                       ? args[0]
                       : arith::ExtFOp::create(b, l, accTy, args[0]).getResult();
        IRMapping map;
        map.map(body.getArgument(0), in);
        map.map(body.getArgument(1), args[1]);
        SmallVector<Value> operands;
        for (Value v : combine->getOperands())
            operands.push_back(map.lookup(v));
        OperationState st(
            combine->getLoc(), combine->getName().getStringRef(), operands, {accTy},
            combine->getAttrs()
        );
        linalg::YieldOp::create(b, l, b.create(st)->getResult(0));
    };
    return linalg::GenericOp::create(
        rewriter, loc, TypeRange{accType}, op.getInputs(), ValueRange{init},
        ArrayRef<AffineMap>{inMap, reducedOutMap}, op.getIteratorTypesArray(), buildBody
    );
}

// A keepdims reduction (e.g. an SE-block global-average-pool) has constant result positions in its
// output map, e.g. (d0,d1,d2,d3) -> (d0,d1,0,0). That is not a projected permutation, so upstream's
// linalg TilingInterface refuses to tile it, and tile-and-fuse then aborts at fuseControlMax*. The
// interface is registered by upstream IREE before the plugin loads and cannot be overridden, so
// rewrite the op into a non-keepdims reduction (a projected permutation) plus a tensor.expand_shape
// back to the keepdims shape. Scoped to fuse-group members: a standalone keepdims reduction is
// never tiled, so leave its accumulation order untouched.
static LogicalResult rewriteKeepdimsFuseGroupReduction(IRRewriter &rewriter, linalg::GenericOp op) {
    if (!isMarkedFuseGroup(op))
        return failure();
    Operation *combine = matchSingleCombineReduction(op);
    if (!combine)
        return failure();

    auto outType = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
    if (!outType)
        return failure();
    AffineMap outMap = op.getIndexingMapsArray().back();
    SmallVector<int64_t> keptPos;
    SmallVector<AffineExpr> keptExprs;
    if (failed(splitKeepdimsOutputMap(outMap, outType, keptPos, keptExprs)))
        return failure();

    Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);

    SmallVector<int64_t> reducedShape;
    for (int64_t p : keptPos)
        reducedShape.push_back(outType.getDimSize(p));
    Type elemTy = outType.getElementType();
    auto reducedType = RankedTensorType::get(reducedShape, elemTy);
    AffineMap reducedOutMap =
        AffineMap::get(outMap.getNumDims(), 0, keptExprs, rewriter.getContext());
    SmallVector<ReassociationIndices> reassoc = keepdimsReassociation(keptPos, outType.getRank());

    // Tiling changes the order and grouping of the sum. bf16 cannot accumulate thousands of values
    // (summing 6400 values at magnitude ~6400 has an ulp of ~32 -> ~70% error), so accumulate
    // narrow floats in f32 and truncate back; f32/f64/int already accumulate at full width.
    Type accTy = isa<FloatType>(elemTy) && elemTy.getIntOrFloatBitWidth() < 32
                     ? rewriter.getF32Type()
                     : elemTy;
    auto accType = RankedTensorType::get(reducedShape, accTy);

    SmallVector<Operation *> newOps;
    // Widening the init (rather than seeding zero) preserves its value: 0 for a sum, -inf for a
    // max.
    Value init =
        tensor::CollapseShapeOp::create(rewriter, loc, reducedType, op.getOutputs()[0], reassoc);
    if (accTy != elemTy) {
        auto initW = createTileableCast(rewriter, loc, init, accType);
        newOps.push_back(initW);
        init = initW.getResult(0);
    }

    linalg::GenericOp reduceOp =
        createReducedReduction(rewriter, loc, op, combine, init, accType, reducedOutMap);
    newOps.push_back(reduceOp);
    Value reduceResult = reduceOp.getResult(0);

    if (accTy != elemTy) {
        auto truncOp = createTileableCast(rewriter, loc, reduceResult, reducedType);
        newOps.push_back(truncOp);
        reduceResult = truncOp.getResult(0);
    }

    auto expanded = tensor::ExpandShapeOp::create(rewriter, loc, outType, reduceResult, reassoc);
    newOps.push_back(expanded);

    // Every new op keeps the group markers so the group still tiles as one unit
    // (isFuseGroupOutput walks def-use chains looking for members).
    if (Attribute fg = op->getAttr(TORQ_FUSE_GROUP))
        for (Operation *nop : newOps)
            nop->setAttr(TORQ_FUSE_GROUP, fg);
    if (Attribute fgId = op->getAttr(TORQ_FUSE_GROUP_ID))
        reduceOp->setAttr(TORQ_FUSE_GROUP_ID, fgId);

    rewriter.replaceOp(op, expanded.getResult());
    return success();
}

void TileAndFusePass::runOnOperation() {
    LLVM_DEBUG(llvm::dbgs() << "Tile and Fuse - START\n");

    FunctionOpInterface funcOp = getOperation();

    // Reduction split phase: K-chunk LRAM-oversized matmuls before the parallel
    // fit. The split is unrolled (no scf.for), so the fit-check below stays
    // loop-free. Matmuls that cannot be K-split are marked for Host execution.
    if (auto func = dyn_cast<func::FuncOp>(funcOp.getOperation()))
        splitOversizedMatmulsAlongK(func);

    // Walk on all the TilingInterface ops in the function, in reverse order of
    // appearance. This guarantees that when we tile an op, we have already
    // considered all its users.
    // NB: Since tileAndFuse mutates the IR, we can't call it directly from the
    // walk. We first construct a queue, and than iterate over it.
    SmallVector<TilingInterface> orderTi;
    funcOp.walk<WalkOrder::PostOrder, ReverseIterator>([&](TilingInterface tiOp) {
        // Ops assigned to a non-NPU executor (e.g. Host) don't run on the NPU,
        // so LRAM fit is meaningless for them: skip both the fit check and
        // tiling. Unassigned ops (no attr) default to the NPU path.
        if (getTargetExecutor(tiOp) == torq_hl::Executor::Host)
            return;

        // For pattern-fuse-groups, we only tile from the bottom most op, to make
        // sure the whole group is tiled together.
        if (isMarkedFuseGroup(tiOp) && !isFuseGroupOutput(tiOp))
            return;

        // Check if tiOp already fits in memory.
        OwningOpRef<ModuleOp> moduleOp = extractOpsForMemoryCheck("check_needs_tiling", tiOp);
        llvm::FailureOr<bool> opFitsInMemory = checkModuleFitsInMemory(*moduleOp, true);
        if (failed(opFitsInMemory)) {
            tiOp->emitWarning(
                "tile-and-fuse: initial memory overflow check failed, skipping this operation."
            );
            LLVM_DEBUG(assert(false));
            return;
        }

        if (*opFitsInMemory)
            return;

        orderTi.push_back(tiOp);
    });

    DenseSet<TilingInterface> toTileOps(orderTi.begin(), orderTi.end());

    // Only groups that don't fit (their output op landed in orderTi) are about to be tiled, so only
    // those need their keepdims reductions rewritten; groups that fit keep their accumulation
    // order.
    {
        IRRewriter rewriter(funcOp.getContext());
        for (TilingInterface tiOp : orderTi) {
            Operation *outputOp = tiOp.getOperation();
            if (!isMarkedFuseGroup(outputOp))
                continue;
            SmallVector<linalg::GenericOp> members;
            funcOp.walk([&](linalg::GenericOp genericOp) {
                if (genericOp.getOperation() != outputOp &&
                    checkShareFuseGroup(genericOp, outputOp))
                    members.push_back(genericOp);
            });
            for (linalg::GenericOp genericOp : members)
                (void)rewriteKeepdimsFuseGroupReduction(rewriter, genericOp);
        }
    }

    untiledTiledOps_.reserve(orderTi.size());

    for (auto [count, tiOp] : llvm::enumerate(orderTi)) {
        LLVM_DEBUG({
            if (orderTi.size() > 50)
                llvm::dbgs() << "Processing TI op: " << count << "/" << orderTi.size() << "\n";
        });

        if (!tiOp->use_empty())
            tileAndFuse(tiOp, toTileOps);

        if (tiOp->use_empty()) {
            if (auto fuseGroup = isFuseGroupOutput(tiOp)) {
                removeFuseGroupMarkingBackwards(tiOp, *fuseGroup);
            }

            tiOp->erase();
        }
    }

    LLVM_DEBUG(llvm::dbgs() << "Tile and Fuse - DONE\n");
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createTileAndFusePass(int64_t sliceCount) {
    return std::make_unique<TileAndFusePass>(TileAndFuseOptions{sliceCount});
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
