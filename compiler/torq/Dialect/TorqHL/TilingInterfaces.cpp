// This is a copy of
// iree/third_party/llvm-project/mlir/lib/Dialect/Linalg/Transforms/TilingInterfaceImpl.cpp because
// we need to use the private implementation

//===- TilingInterfaceImpl.cpp - Implementation of TilingInterface -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TilingInterfaces.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include <optional>

#ifdef ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/GenericOp.h"
#endif // ENABLE_TORQ_GENERIC
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::syna;

//===----------------------------------------------------------------------===//
// Shared helpers used by every TorqHL TilingInterface model
//===----------------------------------------------------------------------===//

/// Extract a ranked-tensor slice.  Stride is always 1 on every dimension.
static Value makeSlice(
    OpBuilder &b, Location loc, Value tensor, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes
) {
    SmallVector<OpFoldResult> strides(offsets.size(), b.getIndexAttr(1));
    return tensor::ExtractSliceOp::create(b, loc, tensor, offsets, sizes, strides);
}

/// Size of dimension `d` of a tensor as an OpFoldResult (folds when static).
static OpFoldResult dimOf(OpBuilder &b, Location loc, Value tensor, unsigned d) {
    return b.createOrFold<tensor::DimOp>(loc, tensor, d);
}

/// Full iteration range for dimension `d` of `tensor`: [0, size, 1].
static Range fullRange(OpBuilder &b, Location loc, Value tensor, unsigned d) {
    return {b.getIndexAttr(0), dimOf(b, loc, tensor, d), b.getIndexAttr(1)};
}

/// Extract a constant from an OpFoldResult, falling back to `fallback`.
/// Used for computing Conv2D padding adjustments that must remain static.
static int64_t getConstOrFallback(OpFoldResult ofr, int64_t fallback) {
    if (std::optional<int64_t> v = getConstantIntValue(ofr))
        return *v;
    return fallback;
}

/// Compute the tile of `scaleBias` that corresponds to tiling `channelCount`
/// output channels at [chanOff, chanOff+chanSz).
///
/// `scaleBias.dim(0)` may equal `channelCount` (float: one value per channel)
/// or `channelCount * factor` (quantized: `factor` values per channel, e.g.
/// factor=2 for interleaved int8 scale+bias).  In both cases the tile starts
/// at `chanOff * factor` and has size `chanSz * factor`.  If `dim(0)` is not
/// an exact multiple of `channelCount` the tensor is returned unchanged.
static Value sliceScaleBias(
    OpBuilder &b, Location loc, Value scaleBias, int64_t channelCount, OpFoldResult chanOff,
    OpFoldResult chanSz
) {
    auto sbTy = cast<RankedTensorType>(scaleBias.getType());
    int64_t sbDim0 = sbTy.getDimSize(0);
    if (sbDim0 % channelCount != 0)
        return scaleBias;

    int64_t factor = sbDim0 / channelCount;
    AffineExpr d0;
    bindDims(b.getContext(), d0);
    // Scale the channel offset/size by the interleaving factor.
    OpFoldResult sbOff0 = affine::makeComposedFoldedAffineApply(b, loc, d0 * factor, {chanOff});
    OpFoldResult sbSz0 = affine::makeComposedFoldedAffineApply(b, loc, d0 * factor, {chanSz});

    SmallVector<OpFoldResult> sbOffs(sbTy.getRank(), b.getIndexAttr(0));
    SmallVector<OpFoldResult> sbSzs;
    sbOffs[0] = sbOff0;
    sbSzs.push_back(sbSz0);
    for (unsigned d = 1; d < (unsigned)sbTy.getRank(); ++d)
        sbSzs.push_back(dimOf(b, loc, scaleBias, d));
    return makeSlice(b, loc, scaleBias, sbOffs, sbSzs);
}

//===----------------------------------------------------------------------===//
// Conv2D spatial input window helper
//
// Given an output-space tile [outOff, outOff+outSz) along one spatial dim,
// compute:
//   - The corresponding input slice start and size (after clamping to [0, inDimSz)).
//   - The new padding values for the tiled op.
//
// The math (all values are in element coordinates):
//
//   in_start_unc = outOff * stride - padLow              (may be negative)
//   in_end_unc   = (outOff + outSz - 1) * stride
//                + (kernelSz - 1) * dilation + 1 - padLow
//
//   in_start = max(0, in_start_unc)                      (clamp to valid input)
//   in_end   = min(inDimSz, in_end_unc)
//   in_size  = in_end - in_start
//
//   new_padLow  = max(0, -in_start_unc)                  (= max(0, padLow - outOff*stride))
//   new_padHigh = max(0, in_end_unc - inDimSz)
//
// When outOff == 0 and outSz == H_out (i.e., no spatial tiling), these simplify
// to the original pad values — so the formula is always safe to apply.
//
// NOTE: new_padLow and new_padHigh are returned as static constants when
// outOff / outSz are constants (the common case after PeelTileLoopsPass).
// When they are dynamic (SCF loop IVs), getConstOrFallback preserves the
// original padding, which is correct for N / C_out tiling where spatial
// offsets stay at 0 / full-range.
//===----------------------------------------------------------------------===//
struct SpatialSlice {
    OpFoldResult start; // clamped input start
    OpFoldResult size;  // clamped input size
    int64_t newPadLow;  // updated padding on the low side
    int64_t newPadHigh; // updated padding on the high side
};

static SpatialSlice computeSpatialSlice(
    OpBuilder &b, Location loc, OpFoldResult outOff, OpFoldResult outSz, int64_t stride,
    int64_t dilation, int64_t padLow, int64_t padHigh, int64_t kernelSz, int64_t inDimSz
) {
    MLIRContext *ctx = b.getContext();
    AffineExpr d0, d1;
    bindDims(ctx, d0, d1);

    // in_start_unc = d0 * stride - padLow
    OpFoldResult inStartUnc =
        affine::makeComposedFoldedAffineApply(b, loc, d0 * stride - padLow, {outOff});

    // in_end_unc = (d0 + d1 - 1) * stride + (kernelSz - 1) * dilation + 1 - padLow
    int64_t kernelSpan = (kernelSz - 1) * dilation + 1;
    OpFoldResult inEndUnc = affine::makeComposedFoldedAffineApply(
        b, loc, (d0 + d1 - 1) * stride + kernelSpan - padLow, {outOff, outSz}
    );

    // in_start = max(0, in_start_unc)
    AffineMap maxZeroMap = AffineMap::get(1, 0, {getAffineConstantExpr(0, ctx), d0}, ctx);
    Value inStartUncVal = getValueOrCreateConstantIndexOp(b, loc, inStartUnc);
    Value inStart = affine::AffineMaxOp::create(b, loc, maxZeroMap, inStartUncVal);

    // in_end = min(inDimSz, in_end_unc)
    AffineMap minMap = AffineMap::get(1, 0, {getAffineConstantExpr(inDimSz, ctx), d0}, ctx);
    Value inEndUncVal = getValueOrCreateConstantIndexOp(b, loc, inEndUnc);
    Value inEnd = affine::AffineMinOp::create(b, loc, minMap, inEndUncVal);

    Value inSize = arith::SubIOp::create(b, loc, inEnd, inStart);

    // new_padLow  = max(0, padLow - outOff * stride)
    //             = max(0, -in_start_unc)
    int64_t newPadLow =
        getConstOrFallback(inStartUnc, 0) < 0 ? -getConstOrFallback(inStartUnc, 0) : 0;
    // Tighten: keep original pad when outOff is not statically 0.
    if (std::optional<int64_t> offConst = getConstantIntValue(outOff))
        newPadLow = std::max(int64_t(0), padLow - *offConst * stride);
    else
        newPadLow = padLow; // dynamic offset: fall back to original

    int64_t newPadHigh = 0;
    if (std::optional<int64_t> offConst = getConstantIntValue(outOff)) {
        if (std::optional<int64_t> szConst = getConstantIntValue(outSz)) {
            int64_t endUnc = (*offConst + *szConst - 1) * stride + kernelSpan - padLow;
            newPadHigh = std::max(int64_t(0), endUnc - inDimSz);
        }
    }

    return {getAsOpFoldResult(inStart), getAsOpFoldResult(inSize), newPadLow, newPadHigh};
}

//===----------------------------------------------------------------------===//
// External Model for implementing `TilingInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

/// Return the SSA values that represent the data point accessed using a given
/// `indexingMap` for a given point in the iteration space represented by `ivs`.
static SmallVector<Value>
getIndicesForAccess(OpBuilder &b, Location loc, AffineMap indexingMap, ValueRange ivs) {
    SmallVector<Value> indices;
    indices.reserve(indexingMap.getNumResults());
    for (auto result : indexingMap.getResults()) {
        AffineMap m = AffineMap::get(indexingMap.getNumDims(), indexingMap.getNumSymbols(), result);
        Value v = affine::AffineApplyOp::create(b, loc, m, ivs);
        indices.push_back(v);
    }
    return indices;
}

/// Method to inline the payload of a `linalgOp` given the iteration space
/// point and values for the arguments of the payload.
static LogicalResult
inlinePayload(OpBuilder &b, LinalgOp linalgOp, ValueRange ivs, ValueRange argValues) {
    Block *body = linalgOp.getBlock();
    IRMapping map;
    map.map(body->getArguments(), argValues);
    for (auto &op : body->without_terminator()) {
        if (auto indexOp = dyn_cast<IndexOp>(&op)) {
            map.map(indexOp.getResult(), ivs[indexOp.getDim()]);
            continue;
        }
        b.clone(op, map);
    }

    Operation *terminator = body->getTerminator();
    Location loc = terminator->getLoc();
    for (const auto &operand : llvm::enumerate(terminator->getOperands())) {
        Value toStore = map.lookupOrDefault(operand.value());
        OpOperand *storeInto = linalgOp.getDpsInitOperand(operand.index());
        auto indices = getIndicesForAccess(b, loc, linalgOp.getMatchingIndexingMap(storeInto), ivs);
        memref::StoreOp::create(
            b, loc, toStore, linalgOp.getDpsInitOperand(operand.index())->get(), indices
        );
    }
    return success();
}

//===----------------------------------------------------------------------===//
// External Model for implementing `TilingInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

namespace {
/// External model implementation of TilingInterface for LinalgOps. An external
/// model implementation is used for now till the use of `TilingInterface` is
/// on-par with the current Linalg tiling + fusion patterns. Once it is
/// maybe possible to move this into the op-definition (though there are
/// advantages to leaving it as an external model)
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>, LinalgOpTy> {
    /// Return the loop iterator type.
    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        LinalgOpTy concreteOp = cast<LinalgOpTy>(op);
        return concreteOp.getIteratorTypesArray();
    }

    /// Return the iteration domain range.
    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPoint(op);
        Location loc = op->getLoc();
        LinalgOp linalgOp = cast<LinalgOp>(op);
        SmallVector<OpFoldResult> allShapesSizes = linalgOp.createFlatListOfOperandDims(b, loc);
        AffineMap map = linalgOp.getShapesToLoopsMap();

        return llvm::to_vector(llvm::map_range(map.getResults(), [&](AffineExpr loopExpr) {
            OpFoldResult ofr =
                affine::makeComposedFoldedAffineApply(b, loc, loopExpr, allShapesSizes);
            return Range{b.getIndexAttr(0), ofr, b.getIndexAttr(1)};
        }));
    }

    /// Instantiate the tiled implementation of the operation.
    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        // Leave the `sizeBounds` value empty. That is only needed when the `sizes`
        // specified could lead to out of bounds accesses.
        Location loc = op->getLoc();
        LinalgOp linalgOp = cast<LinalgOp>(op);
        SmallVector<Value> valuesToTile = linalgOp->getOperands();
        SmallVector<Value, 4> tiledOperands =
            makeTiledShapes(b, loc, linalgOp, valuesToTile, offsets, sizes, {}, true);

        SmallVector<Type> resultTensorTypes = getTensorOutputTypes(linalgOp, tiledOperands);

        Operation *tiledOp = clone(b, linalgOp, resultTensorTypes, tiledOperands);
        offsetIndices(b, cast<LinalgOp>(tiledOp), offsets);

        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    /// Utility to fetch the offsets and sizes when applied as per the indexing
    /// map of the linalg op. This helps in fusing the linalg op as a consumer of
    /// a given slice op.
    void getMappedOffsetAndSize(
        LinalgOp linalgOp, OpBuilder &b, AffineMap indexingMap, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVectorImpl<OpFoldResult> &mappedOffsets,
        SmallVectorImpl<OpFoldResult> &mappedSizes
    ) const {
        unsigned numLoops = linalgOp.getNumLoops();
        auto tilingInterfaceOp = cast<TilingInterface>(linalgOp.getOperation());
        mappedOffsets.resize(numLoops);
        mappedSizes.resize(numLoops);
        if (!indexingMap.isPermutation()) {
            SmallVector<Range> iterationDomain = tilingInterfaceOp.getIterationDomain(b);
            for (const auto &&[index, value] : llvm::enumerate(iterationDomain)) {
                mappedOffsets[index] = value.offset;
                mappedSizes[index] = value.size;
            }
        }
        for (const auto &&[index, value] : llvm::enumerate(indexingMap.getResults())) {
            unsigned dimPosition = cast<AffineDimExpr>(value).getPosition();
            mappedOffsets[dimPosition] = offsets[index];
            mappedSizes[dimPosition] = sizes[index];
        }
    }

    /// Method to return the position of the result tile computed by the tiled
    /// operation.
    LogicalResult getIterationDomainTileFromOperandTile(
        Operation *op, OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
        SmallVectorImpl<OpFoldResult> &iterDomainSizes
    ) const {
        auto linalgOp = cast<LinalgOp>(op);

        // Check that the indexing map used for the operand is a projected
        // permutation. This could be relaxed with a more general approach that can
        // map the offsets and sizes from the operand to iteration space tiles
        // (filling in full extent for dimensions not used to access the result).
        AffineMap indexingMap = linalgOp.getMatchingIndexingMap(&op->getOpOperand(operandNumber));
        if (!indexingMap.isProjectedPermutation()) {
            return op->emitError() << "unhandled get iter domain position when operand is not "
                                      "accessed using a permuted projection";
        }

        getMappedOffsetAndSize(
            linalgOp, b, indexingMap, offsets, sizes, iterDomainOffsets, iterDomainSizes
        );
        return success();
    }

    /// Return the details of the output tile generated by the tiled
    /// implementation.
    LogicalResult getResultTilePosition(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        Location loc = op->getLoc();
        LinalgOp linalgOp = cast<LinalgOp>(op);

        AffineExpr d0;
        bindDims(b.getContext(), d0);
        SmallVector<OpFoldResult> subShapeSizes =
            llvm::to_vector(llvm::map_range(sizes, [&](OpFoldResult ofr) {
                return affine::makeComposedFoldedAffineApply(b, loc, d0 - 1, ofr);
            }));

        OpOperand *outOperand = linalgOp.getDpsInitOperand(resultNumber);
        SliceParameters sliceParams = computeSliceParameters(
            b, loc, outOperand->get(), sizes, linalgOp.getMatchingIndexingMap(outOperand), offsets,
            /*ubs*/ {}, subShapeSizes, true
        );
        resultOffsets = sliceParams.offsets;
        resultSizes = sliceParams.sizes;
        return success();
    }

    LogicalResult getIterationDomainTileFromResultTile(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVectorImpl<OpFoldResult> &iterDomainOffsets,
        SmallVectorImpl<OpFoldResult> &iterDomainSizes
    ) const {
        auto linalgOp = cast<LinalgOp>(op);

        // Check that the indexing map used for the output is a projected
        // permutation. This could be relaxed with a more general approach that can
        // map the offsets and sizes from the result to iteration space tiles
        // (filling in full extent for dimensions not used to access the result).
        AffineMap indexingMap = linalgOp.getIndexingMapMatchingResult(op->getResult(resultNumber));
        if (!indexingMap.isProjectedPermutation()) {
            return op->emitOpError("unhandled tiled implementation generation when result is not "
                                   "accessed using a permuted projection");
        }

        getMappedOffsetAndSize(
            linalgOp, b, indexingMap, offsets, sizes, iterDomainOffsets, iterDomainSizes
        );
        return success();
    }

    FailureOr<TilingResult> generateResultTileValue(
        Operation *op, OpBuilder &b, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    ) const {
        SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
        if (failed(getIterationDomainTileFromResultTile(
                op, b, resultNumber, offsets, sizes, mappedOffsets, mappedSizes
            ))) {
            return failure();
        }
        auto tilingInterfaceOp = cast<TilingInterface>(op);
        FailureOr<TilingResult> tilingResult =
            tilingInterfaceOp.getTiledImplementation(b, mappedOffsets, mappedSizes);

        if (failed(tilingResult))
            return failure();

        if (tilingResult->tiledOps.size() != 1)
            return op->emitOpError("failed to generate tiled implementation");

        return TilingResult{
            tilingResult->tiledOps, SmallVector<Value>{tilingResult->tiledValues[resultNumber]}
        };
    }

    /// Method to generate the tiled implementation of an operation from the tile
    /// of the operand.
    FailureOr<TilingResult> getTiledImplementationFromOperandTile(
        Operation *op, OpBuilder &b, unsigned operandNumber, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes
    ) const {
        SmallVector<OpFoldResult> mappedOffsets, mappedSizes;
        if (failed(getIterationDomainTileFromOperandTile(
                op, b, operandNumber, offsets, sizes, mappedOffsets, mappedSizes
            ))) {
            return failure();
        }
        return getTiledImplementation(op, b, mappedOffsets, mappedSizes);
    }

    LogicalResult generateScalarImplementation(
        Operation *op, OpBuilder &builder, Location loc, ValueRange ivs
    ) const {
        auto linalgOp = cast<LinalgOp>(op);
        if (!linalgOp.hasPureBufferSemantics())
            return op->emitOpError("expected operation to have buffer semantics");

        SmallVector<Value> indexedValues;
        indexedValues.reserve(linalgOp->getNumOperands());
        Location linalgOpLoc = op->getLoc();
        /// Load the data corresponding to the block arguments that
        /// represent input operands.
        for (OpOperand &operand : linalgOp->getOpOperands()) {
            if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
                indexedValues.push_back(nullptr);
                continue;
            }
            if (linalgOp.isScalar(&operand)) {
                indexedValues.push_back(operand.get());
                continue;
            }
            SmallVector<Value> indices = getIndicesForAccess(
                builder, linalgOpLoc, linalgOp.getMatchingIndexingMap(&operand), ivs
            );
            Value load = memref::LoadOp::create(builder, linalgOpLoc, operand.get(), indices);
            indexedValues.push_back(load);
        }

        /// Inline the op payload and store the result.
        return inlinePayload(builder, linalgOp, ivs, indexedValues);
    }
};

//===----------------------------------------------------------------------===//
// External Model for implementing `PartialReductionInterface` for `LinalgOp`s.
//===----------------------------------------------------------------------===//

/// External model implementation of PartialReductionInterface for LinalgOps.
template <typename LinalgOpTy>
struct LinalgOpPartialReductionInterface
    : public PartialReductionOpInterface::ExternalModel<
          LinalgOpPartialReductionInterface<LinalgOpTy>, LinalgOpTy> {
    FailureOr<SmallVector<Value>> generateInitialTensorForPartialReduction(
        Operation *op, OpBuilder &b, Location loc, ArrayRef<OpFoldResult> sizes,
        ArrayRef<int> reductionDims
    ) const {
        auto linalgOp = cast<LinalgOp>(op);
        OpBuilder::InsertionGuard guard(b);

        if (linalgOp.hasPureBufferSemantics())
            return op->emitOpError("expected operation to have tensor semantics");

        SmallVector<Value> inits;
        for (int initIdx = 0, e = linalgOp.getNumDpsInits(); initIdx < e; ++initIdx) {
            // Insert the new parallel dimension based on the index of the reduction
            // loops. This could be controlled by user for more flexibility.
            SmallVector<Operation *, 4> combinerOps;
            if (!matchReduction(linalgOp.getRegionOutputArgs(), initIdx, combinerOps) ||
                combinerOps.size() != 1)
                return op->emitOpError("Failed to anaysis the reduction operation.");

            Operation *reductionOp = combinerOps[0];
            std::optional<TypedAttr> identity = arith::getNeutralElement(reductionOp);
            if (!identity.has_value())
                return op->emitOpError(
                    "Failed to get an identity value for the reduction operation."
                );

            ArrayRef<int64_t> oldShape = linalgOp.getShape(linalgOp.getDpsInitOperand(initIdx));

            // Calculate the new shape, we insert the new dimensions based on the
            // index of the reduction dimensions.
            SmallVector<int64_t> newOutputShape;
            SmallVector<Value> dynamicDims;
            int64_t currReductionDims = 0;
            DenseSet<int> reductionDimsSet(reductionDims.begin(), reductionDims.end());
            for (int64_t idx : llvm::seq<int64_t>(0, oldShape.size() + reductionDims.size())) {
                if (reductionDimsSet.contains(idx)) {
                    dispatchIndexOpFoldResults(sizes[idx], dynamicDims, newOutputShape);
                    currReductionDims++;
                    continue;
                }
                int64_t oldIdx = idx - currReductionDims;
                int64_t dim = oldShape[oldIdx];
                newOutputShape.push_back(dim);
                if (ShapedType::isDynamic(dim))
                    dynamicDims.push_back(tensor::DimOp::create(
                        b, loc, linalgOp.getDpsInitOperand(initIdx)->get(), oldIdx
                    ));
            }
            Value emptyTensor = tensor::EmptyOp::create(
                b, loc, newOutputShape, linalgOp.getRegionOutputArgs()[initIdx].getType(),
                dynamicDims
            );
            Value constantOp = arith::ConstantOp::create(b, loc, *identity);
            auto identityTensor = linalg::FillOp::create(b, loc, constantOp, emptyTensor);
            inits.push_back(identityTensor.getResult(0));
        }

        return inits;
    }

    FailureOr<TilingResult> tileToPartialReduction(
        Operation *op, OpBuilder &b, Location loc, ValueRange init, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, ArrayRef<int> reductionDims
    ) const {
        OpBuilder::InsertionGuard guard(b);
        auto linalgOp = cast<LinalgOp>(op);

        // Step 1. Extend init maps to have reduction dimension dims, since we
        // are converting them to parallel dimensions.
        SmallVector<AffineMap> newInitMaps;
        newInitMaps.reserve(linalgOp.getNumDpsInits());
        for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
            // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
            // this with a for range loop when we have it.
            AffineMap newMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(idx));
            for (int redPos : reductionDims) {
                newMap = newMap.insertResult(b.getAffineDimExpr(redPos), newMap.getNumResults());
            }
            newInitMaps.push_back(newMap);
        }

        // Step 2a: Extract a slice of the input operands.
        SmallVector<Value, 4> tiledInputs =
            makeTiledShapes(b, loc, linalgOp, linalgOp.getDpsInputs(), offsets, sizes, {}, true);

        // Step 2b: Extract a slice of the init operands.
        SmallVector<Value, 1> tiledInits;
        for (auto [valueMap, valueToTile] : llvm::zip_equal(newInitMaps, init)) {
            int64_t initRank = valueMap.getNumResults();
            SmallVector<OpFoldResult> initOffset(initRank, b.getIndexAttr(0));
            SmallVector<OpFoldResult> initStride(initRank, b.getIndexAttr(1));
            SmallVector<OpFoldResult> initSizes;
            for (AffineExpr dimExpr : valueMap.getResults()) {
                auto dim = cast<AffineDimExpr>(dimExpr);
                initSizes.push_back(sizes[dim.getPosition()]);
            }
            // TODO: Use SubsetExtractOpInterface here once available.
            auto extractSlice = tensor::ExtractSliceOp::create(
                b, loc, valueToTile, initOffset, initSizes, initStride
            );
            tiledInits.push_back(extractSlice);
        }

        // Update the indexing maps.
        SmallVector<AffineMap> newMaps = linalgOp.getIndexingMapsArray();
        // Change the init maps.
        for (int idx : llvm::seq<int>(0, linalgOp.getNumDpsInits())) {
            // TODO: linalg::Generic doesn't have getDpsInitOperands. Can replace
            // this with a for range loop when we have it.
            OpOperand *initOperand = linalgOp.getDpsInitOperand(idx);
            int64_t mapIdx = linalgOp.getIndexingMapIndex(initOperand);
            newMaps[mapIdx] = newInitMaps[idx];
        }

        // Step 3. Change the reduction dim iterator types.
        SmallVector<utils::IteratorType> newIteratorTypes = linalgOp.getIteratorTypesArray();
        for (int dim : reductionDims)
            newIteratorTypes[dim] = utils::IteratorType::parallel;

        // Step 4. Create the new generic op.
        auto genericOp = GenericOp::create(
            b, loc, ValueRange(tiledInits).getTypes(), tiledInputs, tiledInits, newMaps,
            newIteratorTypes
        );
        IRMapping mapping;
        op->getRegion(0).cloneInto(&genericOp.getRegion(), genericOp.getRegion().begin(), mapping);
        return TilingResult{
            {genericOp.getOperation()},
            llvm::map_to_vector(genericOp->getResults(), [](OpResult r) -> Value { return r; })
        };
    }

    FailureOr<MergeResult> mergeReductions(
        Operation *op, OpBuilder &b, Location loc, ValueRange partialReduce,
        ArrayRef<int> reductionDims
    ) const {
        auto linalgOp = cast<LinalgOp>(op);
        SmallVector<int64_t> reductionDimsInt64(reductionDims.begin(), reductionDims.end());
        auto reduction = linalg::ReduceOp::create(
            b, loc, partialReduce, linalgOp.getDpsInits(), reductionDimsInt64,
            [&linalgOp](OpBuilder &b, Location loc, ValueRange inputs) {
                int64_t numInits = linalgOp.getNumDpsInits();
                SmallVector<Value> yieldedValues;
                for (int idx : llvm::seq<int>(0, numInits)) {
                    // Get the combiner op.
                    SmallVector<Operation *, 4> combinerOps;
                    matchReduction(linalgOp.getRegionOutputArgs(), idx, combinerOps);
                    Operation *clonedReductionOp = b.clone(*combinerOps[0]);
                    // Combine the input at idx and output at numInits + idx.
                    clonedReductionOp->setOperand(0, inputs[idx]);
                    clonedReductionOp->setOperand(1, inputs[numInits + idx]);
                    // Yield.
                    yieldedValues.push_back(clonedReductionOp->getResult(0));
                }
                linalg::YieldOp::create(b, loc, yieldedValues);
            }
        );
        return MergeResult{
            {reduction.getOperation()},
            llvm::map_to_vector(reduction->getResults(), [](OpResult r) -> Value { return r; })
        };
    }
};

} // namespace

//===----------------------------------------------------------------------===//
// TilingInterface for torq_hl::MatMulOp
//
// input1/input2/init are 2D [M,K]/[K,N]/[M,N], or 3D with a leading batch dim
// [B,M,K]/[B,K,N]/[B,M,N] (B is usually 1). numBatchDims = init.rank() - 2.
//
// Iteration domain: numBatchDims + 2 dims, all parallel
//   dim [0, numBatchDims)  — batch (parallel), full range every tile
//   dim numBatchDims       — M (parallel) : init.dim(numBatchDims) = input1.dim(numBatchDims)
//   dim numBatchDims + 1   — N (parallel) : init.dim(numBatchDims+1) = input2.dim(numBatchDims+1)
//
// K (the shared contraction dimension, the last dim of input1/input2) is a
// reduction that is never tiled; the full K range is included in every tile.
//
// Operand slicing for tile [b0:b0+bSz, m0:m0+mSz, n0:n0+nSz] (batch elided if numBatchDims==0):
//   input1 [.., M, K]  → [.., m0:m0+mSz, 0:K]   (slice batch/M, keep full K)
//   input2 [.., K, N]  → [.., 0:K, n0:n0+nSz]   (slice batch, keep full K, slice N)
//   scale_bias         → [n0:n0+nSz] if dim0==N,  (per-output-channel)
//                        unchanged if global scalar
//   init   [.., M, N]  → [.., m0:m0+mSz, n0:n0+nSz]
//===----------------------------------------------------------------------===//
struct MatMulOpTilingInterface
    : public TilingInterface::ExternalModel<MatMulOpTilingInterface, torq_hl::MatMulOp> {

    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        auto mm = cast<torq_hl::MatMulOp>(op);
        int64_t rank = cast<RankedTensorType>(mm.getInit().getType()).getRank();
        return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
    }

    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        auto mm = cast<torq_hl::MatMulOp>(op);
        Location loc = op->getLoc();
        int64_t rank = cast<RankedTensorType>(mm.getInit().getType()).getRank();
        SmallVector<Range> ranges;
        for (int64_t d = 0; d < rank; ++d)
            ranges.push_back(fullRange(b, loc, mm.getInit(), d));
        return ranges;
    }

    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        auto mm = cast<torq_hl::MatMulOp>(op);
        Location loc = op->getLoc();
        OpFoldResult zero = b.getIndexAttr(0);

        auto initTy = cast<RankedTensorType>(mm.getInit().getType());
        unsigned numBatchDims = initTy.getRank() - 2;

        OpFoldResult mOff = offsets[numBatchDims], nOff = offsets[numBatchDims + 1];
        OpFoldResult mSz = sizes[numBatchDims], nSz = sizes[numBatchDims + 1];

        // K is the reduction dimension — always take the full slice.
        OpFoldResult kSz = dimOf(b, loc, mm.getInput1(), numBatchDims + 1);

        // A batch dim may broadcast (operand size 1) even where the iteration
        // domain (from init) is larger, so clamp to [0, 1) there instead of
        // slicing past the operand's actual size.
        auto clampBatchDims = [&](Value operand) {
            auto ty = cast<RankedTensorType>(operand.getType());
            SmallVector<OpFoldResult> batchOffsets, batchSizes;
            for (unsigned d = 0; d < numBatchDims; ++d) {
                if (ty.getDimSize(d) == 1) {
                    batchOffsets.push_back(zero);
                    batchSizes.push_back(b.getIndexAttr(1));
                }
                else {
                    batchOffsets.push_back(offsets[d]);
                    batchSizes.push_back(sizes[d]);
                }
            }
            return std::make_pair(batchOffsets, batchSizes);
        };

        auto [batchOffsets1, batchSizes1] = clampBatchDims(mm.getInput1());
        SmallVector<OpFoldResult> input1Offsets = batchOffsets1;
        input1Offsets.append({mOff, zero});
        SmallVector<OpFoldResult> input1Sizes = batchSizes1;
        input1Sizes.append({mSz, kSz});
        Value tiledInput1 = makeSlice(b, loc, mm.getInput1(), input1Offsets, input1Sizes);

        auto [batchOffsets2, batchSizes2] = clampBatchDims(mm.getInput2());
        SmallVector<OpFoldResult> input2Offsets = batchOffsets2;
        input2Offsets.append({zero, nOff});
        SmallVector<OpFoldResult> input2Sizes = batchSizes2;
        input2Sizes.append({kSz, nSz});
        Value tiledInput2 = makeSlice(b, loc, mm.getInput2(), input2Offsets, input2Sizes);

        int64_t outN = initTy.getDimSize(numBatchDims + 1);
        Value tiledSB = sliceScaleBias(b, loc, mm.getScaleBias(), outN, nOff, nSz);

        Value tiledInit = makeSlice(b, loc, mm.getInit(), offsets, sizes);

        auto tiledOp = torq_hl::MatMulOp::create(
            b, loc, tiledInit.getType(), tiledInit, mm.getOutputZpAttr(), mm.getOutputMinAttr(),
            mm.getOutputMaxAttr(), mm.getShiftAttr(), tiledSB, tiledInput1, tiledInput2
        );
        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    LogicalResult getResultTilePosition(
        Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        // Output tile position equals the full (batch + [M, N]) iteration domain tile.
        resultOffsets = llvm::to_vector(offsets);
        resultSizes = llvm::to_vector(sizes);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// TilingInterface for torq_hl::FullyConnectedOp
//
// Semantically the same as MatMul:
//   output[M, N] = input[M, K] * weights[K, N]
//
// Iteration domain: 2D — [M (parallel), N (parallel)]
// The K reduction is never tiled.
//
// Operand slicing for tile [m0:m0+mSz, n0:n0+nSz]:
//   input   [M, K]  → [m0:m0+mSz, 0:K]
//   weights [K, N]  → [0:K, n0:n0+nSz]   (FC weights are already [K, N], reduction-dim major)
//   scale_bias      → [n0:n0+nSz] if per-channel, unchanged otherwise
//   init    [M, N]  → [m0:m0+mSz, n0:n0+nSz]
//===----------------------------------------------------------------------===//
struct FullyConnectedOpTilingInterface
    : public TilingInterface::ExternalModel<
          FullyConnectedOpTilingInterface, torq_hl::FullyConnectedOp> {

    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        return {utils::IteratorType::parallel, utils::IteratorType::parallel};
    }

    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        auto fc = cast<torq_hl::FullyConnectedOp>(op);
        Location loc = op->getLoc();
        return {fullRange(b, loc, fc.getInit(), 0), fullRange(b, loc, fc.getInit(), 1)};
    }

    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        auto fc = cast<torq_hl::FullyConnectedOp>(op);
        Location loc = op->getLoc();
        OpFoldResult zero = b.getIndexAttr(0);

        OpFoldResult mOff = offsets[0], nOff = offsets[1];
        OpFoldResult mSz = sizes[0], nSz = sizes[1];
        OpFoldResult kSz = dimOf(b, loc, fc.getInput(), 1);

        Value tiledInput = makeSlice(b, loc, fc.getInput(), {mOff, zero}, {mSz, kSz});
        Value tiledWeights = makeSlice(b, loc, fc.getWeights(), {zero, nOff}, {kSz, nSz});

        int64_t outN = cast<RankedTensorType>(fc.getInit().getType()).getDimSize(1);
        Value tiledSB = sliceScaleBias(b, loc, fc.getScaleBias(), outN, nOff, nSz);

        Value tiledInit = makeSlice(b, loc, fc.getInit(), {mOff, nOff}, {mSz, nSz});

        auto tiledOp = torq_hl::FullyConnectedOp::create(
            b, loc, tiledInit.getType(), tiledInit, fc.getInputZpAttr(), fc.getWeightZpAttr(),
            fc.getOutputZpAttr(), fc.getOutputMinAttr(), fc.getOutputMaxAttr(),
            fc.getShiftFactorAttr(), fc.getVectorizationModeAttr(), tiledWeights, tiledSB,
            tiledInput, fc.getIsBatchScaledAttr()
        );
        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    LogicalResult getResultTilePosition(
        Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        resultOffsets = {offsets[0], offsets[1]};
        resultSizes = {sizes[0], sizes[1]};
        return success();
    }
};

//===----------------------------------------------------------------------===//
// TilingInterface for torq_hl::Conv2DOp  (NCHW layout)
//
// Iteration domain: 7D
//   dim 0 — N     (parallel)  : batch        — init.dim(0)
//   dim 1 — C_out (parallel)  : out channels — init.dim(1)
//   dim 2 — H_out (parallel)  : output rows  — init.dim(2)
//   dim 3 — W_out (parallel)  : output cols  — init.dim(3)
//   dim 4 — C_in  (reduction) : in channels  — weights.dim(1)
//   dim 5 — KH    (reduction) : kernel rows  — weights.dim(2)
//   dim 6 — KW    (reduction) : kernel cols  — weights.dim(3)
//
// Tiling parallel dims only (scf::tileUsingSCF sets reduction tile sizes to 0,
// meaning the full reduction range is retained in each parallel tile).
//
// Spatial tiling (H_out, W_out) — see computeSpatialSlice above for the
// input window formula.  The Conv2DOp `pad` attribute is adjusted per tile
// to reflect what portion of the virtual padded input each tile actually sees.
//===----------------------------------------------------------------------===//
struct Conv2DOpTilingInterface
    : public TilingInterface::ExternalModel<Conv2DOpTilingInterface, torq_hl::Conv2DOp> {

    SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
        return {
            utils::IteratorType::parallel,  // N
            utils::IteratorType::parallel,  // C_out
            utils::IteratorType::parallel,  // H_out
            utils::IteratorType::parallel,  // W_out
            utils::IteratorType::reduction, // C_in
            utils::IteratorType::reduction, // KH
            utils::IteratorType::reduction, // KW
        };
    }

    SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
        auto conv = cast<torq_hl::Conv2DOp>(op);
        Location loc = op->getLoc();
        Value init = conv.getInit(), weights = conv.getWeights();
        return {
            fullRange(b, loc, init, 0),    // N
            fullRange(b, loc, init, 1),    // C_out
            fullRange(b, loc, init, 2),    // H_out
            fullRange(b, loc, init, 3),    // W_out
            fullRange(b, loc, weights, 1), // C_in
            fullRange(b, loc, weights, 2), // KH
            fullRange(b, loc, weights, 3), // KW
        };
    }

    FailureOr<TilingResult> getTiledImplementation(
        Operation *op, OpBuilder &b, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes
    ) const {
        auto conv = cast<torq_hl::Conv2DOp>(op);
        Location loc = op->getLoc();
        OpFoldResult zero = b.getIndexAttr(0);

        // offsets/sizes index the 7D iteration domain.
        OpFoldResult nOff = offsets[0], nSz = sizes[0];
        OpFoldResult coOff = offsets[1], coSz = sizes[1];
        OpFoldResult hOff = offsets[2], hSz = sizes[2];
        OpFoldResult wOff = offsets[3], wSz = sizes[3];

        auto weightTy = cast<RankedTensorType>(conv.getWeights().getType());

        // weights [C_out, C_in, KH, KW, ...]: dim 0 may store C_out directly, or
        // C_out/packingFactor with `packingFactor` channels interleaved into a
        // trailing dim (e.g. vectorization_mode packs 4 channels together, so
        // dim0 holds C_out/4 groups).  Divide the C_out offset/size by that
        // factor before slicing dim 0, so we never slice past dim0's actual size.
        Value weights = conv.getWeights();
        int64_t wRank = weightTy.getRank();
        int64_t cOutFull = cast<RankedTensorType>(conv.getInit().getType()).getDimSize(1);
        int64_t weightDim0 = weightTy.getDimSize(0);
        if (cOutFull % weightDim0 != 0) {
            return op->emitError() << "C_out (" << cOutFull
                                   << ") must be an exact multiple of the weight's dim0 size ("
                                   << weightDim0 << ")";
        }
        int64_t packingFactor = cOutFull / weightDim0;

        OpFoldResult wgtOff0 = coOff, wgtSz0 = coSz;
        if (packingFactor > 1) {
            AffineExpr d0;
            bindDims(b.getContext(), d0);
            wgtOff0 =
                affine::makeComposedFoldedAffineApply(b, loc, d0.floorDiv(packingFactor), {coOff});
            wgtSz0 =
                affine::makeComposedFoldedAffineApply(b, loc, d0.floorDiv(packingFactor), {coSz});
        }

        SmallVector<OpFoldResult> wgtOff(wRank, zero), wgtSz;
        wgtOff[0] = wgtOff0;
        wgtSz.push_back(wgtSz0);
        for (int64_t d = 1; d < wRank; ++d)
            wgtSz.push_back(dimOf(b, loc, weights, d));
        Value tiledWeights = makeSlice(b, loc, weights, wgtOff, wgtSz);

        // scale_bias: slice proportionally to the C_out tile — handles both
        // float (dim0 == C_out) and quantized int8 (dim0 == C_out * 2, i.e.
        // interleaved scale+bias per channel).
        Value tiledSB = sliceScaleBias(b, loc, conv.getScaleBias(), cOutFull, coOff, coSz);

        // init (output) [N, C_out, H_out, W_out]
        Value tiledInit =
            makeSlice(b, loc, conv.getInit(), {nOff, coOff, hOff, wOff}, {nSz, coSz, hSz, wSz});

        auto tiledOp = torq_hl::Conv2DOp::create(
            b, loc, tiledInit.getType(), tiledInit, conv.getInputZpAttr(), conv.getWeightZpAttr(),
            conv.getOutputZpAttr(), conv.getOutputMinAttr(), conv.getOutputMaxAttr(),
            conv.getShiftFactorAttr(), conv.getGroupsAttr(), conv.getPadAttr(),
            conv.getStrideAttr(), conv.getDilationAttr(), conv.getVectorizationModeAttr(),
            tiledWeights, tiledSB, conv.getInput(), conv.getNhwcInputAttr(),
            conv.getSegmentOutputAttr()
        );
        return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
    }

    LogicalResult getResultTilePosition(
        Operation *, OpBuilder &, unsigned, ArrayRef<OpFoldResult> offsets,
        ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
        SmallVector<OpFoldResult> &resultSizes
    ) const {
        // Result tile = [N, C_out, H_out, W_out] parallel dims.
        resultOffsets = {offsets[0], offsets[1], offsets[2], offsets[3]};
        resultSizes = {sizes[0], sizes[1], sizes[2], sizes[3]};
        return success();
    }
};

void mlir::syna::torq_hl::registerTilingInterfaceExternalModels(DialectRegistry &registry) {
    registry.addExtension(+[](MLIRContext *ctx, mlir::syna::torq_hl::TorqHLDialect *dialect) {
        torq_hl::MatMulOp::attachInterface<MatMulOpTilingInterface>(*ctx);
        torq_hl::FullyConnectedOp::attachInterface<FullyConnectedOpTilingInterface>(*ctx);
        torq_hl::Conv2DOp::attachInterface<Conv2DOpTilingInterface>(*ctx);

#ifdef ENABLE_TORQ_GENERIC
        // this matches what is promised in TorqHLDialect::initialize
        torq_hl::GenericOp::attachInterface<LinalgOpTilingInterface<torq_hl::GenericOp>>(*ctx);
        torq_hl::GenericOp::attachInterface<LinalgOpPartialReductionInterface<torq_hl::GenericOp>>(
            *ctx
        );
#endif // ENABLE_TORQ_GENERIC
    });
}
