// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// OptimizeTransposeLayoutPass
//===----------------------------------------------------------------------===//
//
// **PURPOSE**: Eliminate redundant NCHW↔NHWC transposes between NCHW regions
// by propagating NCHW→NHWC transposes through layout-agnostic elementwise ops.
//
// **KEY STRATEGY**: Only propagate NCHW→NHWC [0,2,3,1] transposes.
//                   Block NHWC→NCHW [0,3,1,2] transposes (structural boundaries).
//
// **APPROACH**:
//   1. Dataflow analysis marks which ops can propagate NCHW→NHWC transposes
//   2. Wrap propagatable ops with inverse/forward transpose pairs
//   3. Canonicalization cancels adjacent transpose pairs
//
// **RESULT**: Transposes between consecutive NCHW regions are eliminated,
//            preserving the NCHW regions created by ConvertNhwcOpToNchwPass.
//
// **EXAMPLE**: Eliminating transposes between NCHW regions
//
//   BEFORE:
//     %t1 = transpose [0,3,1,2] %input   // NHWC→NCHW (blocked)
//     %conv1 = conv_nchw %t1              // NCHW region
//     %t2 = transpose [0,2,3,1] %conv1    // NCHW→NHWC (can propagate)
//     %add = elementwise %t2              // elementwise in NHWC
//     %t3 = transpose [0,3,1,2] %add      // NHWC→NCHW (blocked)
//     %conv2 = conv_nchw %t3              // NCHW region
//
//   AFTER TRANSFORMATION (wrap %add with inverse/forward pairs):
//     %t1 = transpose [0,3,1,2] %input
//     %conv1 = conv_nchw %t1
//     %t2 = transpose [0,2,3,1] %conv1
//     %inv = transpose [0,3,1,2] %t2      // new inverse for %add
//     %add' = elementwise %inv             // now in NCHW
//     %fwd = transpose [0,2,3,1] %add'    // new forward
//     %t3 = transpose [0,3,1,2] %fwd
//     %conv2 = conv_nchw %t3
//
//   AFTER CANONICALIZATION (%t2+%inv, %fwd+%t3 cancel):
//     %t1 = transpose [0,3,1,2] %input
//     %conv1 = conv_nchw %t1
//     %add' = elementwise %conv1           // directly on NCHW
//     %conv2 = conv_nchw %add'             // directly on NCHW
//
//   Result: NCHW regions merged, intermediate transposes eliminated.
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/LayoutTransformUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-transpose-layout"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Simple Helper Functions
//===----------------------------------------------------------------------===//

// Check if affine map is elementwise (no dimension permutation).
static bool isElementwiseMap(AffineMap map) {

    if (map.isIdentity() || map.isMinorIdentity()) {
        return true;
    }

    // Check: must be constant (broadcast) or dimension in order (no permutation)
    int lastDim = -1;
    for (auto expr : map.getResults()) {
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
            continue;
        }
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
            int pos = dimExpr.getPosition();
            if (pos < lastDim) {
                return false; // out of order - permutation!
            }
            lastDim = pos;
        }
        else {
            return false; // complex expression
        }
    }
    return true;
}

// Check if linalg.generic is layout-agnostic elementwise (all parallel iterators, no permutations).
static bool isLayoutAgnosticElementwise(linalg::GenericOp genericOp) {
    // Must be all parallel iterators
    if (!llvm::all_of(genericOp.getIteratorTypesArray(), [](utils::IteratorType iter) {
            return iter == utils::IteratorType::parallel;
        }))
        return false;
    // All indexing maps must be elementwise-compatible
    int mapIdx = 0;
    for (auto map : genericOp.getIndexingMapsArray()) {
        bool isOk = isElementwiseMap(map);
        LLVM_DEBUG({
            llvm::dbgs() << "      Map " << mapIdx << ": " << map << (isOk ? " [OK]" : " [FAIL]")
                         << "\n";
        });
        if (!isOk) {
            LLVM_DEBUG(llvm::dbgs() << "      FAIL: Not an elementwise-compatible map\n");
            return false;
        }
        mapIdx++;
    }

    LLVM_DEBUG(llvm::dbgs() << "      PASS: All checks passed\n");
    return true;
}

//===----------------------------------------------------------------------===//
// Dataflow Lattice for Transpose Propagation
//===----------------------------------------------------------------------===//

// Lattice tracking whether an SSA value can propagate NCHW→NHWC transposes.
class TransposePropagationLattice : public dataflow::AbstractSparseLattice {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposePropagationLattice)
    using AbstractSparseLattice::AbstractSparseLattice;

    enum class PropagationState {
        Unknown,       // Not yet analyzed
        NoPropagation, // No transpose to propagate
        CanPropagate,  // Can propagate NCHW→NHWC transpose
        Blocking       // Blocks propagation (structural boundary or incompatible op)
    };

    void print(raw_ostream &os) const override {
        os << "TransposePropagation(";
        switch (state) {
        case PropagationState::Unknown:
            os << "Unknown";
            break;
        case PropagationState::NoPropagation:
            os << "NoPropagation";
            break;
        case PropagationState::CanPropagate:
            os << "CanPropagate[";
            for (auto p : permutation)
                os << p << ",";
            os << "]";
            break;
        case PropagationState::Blocking:
            os << "Blocking";
            break;
        }
        os << ")";
    }

    ChangeResult join(const AbstractSparseLattice &other) override {
        const auto &rhs = static_cast<const TransposePropagationLattice &>(other);
        if (state == rhs.state && permutation == rhs.permutation)
            return ChangeResult::NoChange;

        if (state != rhs.state) {
            if (state == PropagationState::Unknown) {
                state = rhs.state;
                permutation = rhs.permutation;
                return ChangeResult::Change;
            }
            if (rhs.state == PropagationState::Unknown)
                return ChangeResult::NoChange;
            state = PropagationState::Blocking;
            permutation.clear();
            return ChangeResult::Change;
        }

        if (state == PropagationState::CanPropagate && permutation != rhs.permutation) {
            state = PropagationState::Blocking;
            permutation.clear();
            return ChangeResult::Change;
        }
        return ChangeResult::NoChange;
    }

    void setPropagation(ArrayRef<int64_t> perm) {
        state = PropagationState::CanPropagate;
        permutation.assign(perm.begin(), perm.end());
    }

    void setNoPropagation() {
        state = PropagationState::NoPropagation;
        permutation.clear();
    }

    void setBlocking() {
        state = PropagationState::Blocking;
        permutation.clear();
    }

    bool canPropagate() const { return state == PropagationState::CanPropagate; }
    bool isBlocking() const { return state == PropagationState::Blocking; }
    ArrayRef<int64_t> getPermutation() const { return permutation; }

  private:
    PropagationState state = PropagationState::Unknown;
    SmallVector<int64_t, 4> permutation;
};

//===----------------------------------------------------------------------===//
// Dataflow Analysis
//===----------------------------------------------------------------------===//

// Forward dataflow analysis identifying ops that can propagate NCHW→NHWC transposes.
class TransposePropagationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TransposePropagationLattice> {
  public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    TransposePropagationAnalysis(DataFlowSolver &solver) : SparseForwardDataFlowAnalysis(solver) {}

    void setToEntryState(TransposePropagationLattice *lattice) override {
        lattice->setNoPropagation();
        propagateIfChanged(lattice, ChangeResult::Change);
    }

    LogicalResult visitOperation(
        Operation *op, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) override {

        LLVM_DEBUG({
            for (auto [idx, lattice] : llvm::enumerate(operands)) {
                llvm::dbgs() << "  Operand " << idx << ": ";
                lattice->print(llvm::dbgs());
                llvm::dbgs() << "\n";
            }
        });

        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
            visitTranspose(transposeOp, operands, results);
            return success();
        }

        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            visitGeneric(genericOp, operands, results);
            return success();
        }

        if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
            visitExtractSlice(sliceOp, operands, results);
            return success();
        }

        if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
            visitInsertSlice(insertOp, operands, results);
            return success();
        }

        if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
            visitExpandShape(expandOp, operands, results);
            return success();
        }

        if (isa<tensor::CollapseShapeOp>(op)) {
            for (auto *result : results) {
                result->setBlocking();
                propagateIfChanged(result, ChangeResult::Change);
            }
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (collapse_shape)\n");
            return success();
        }

        if (isa<tensor::EmptyOp>(op)) {
            // Keep Unknown - tensor.empty adapts to whatever layout is needed
            LLVM_DEBUG(llvm::dbgs() << "  -> Keep Unknown (tensor.empty)\n");
            return success();
        }

        // Default: inherit propagation from first operand
        for (auto *result : results) {
            if (operands.empty()) {
                result->setNoPropagation();
                propagateIfChanged(result, ChangeResult::Change);
            }
            else {
                propagateIfChanged(result, result->join(*operands[0]));
            }
        }

        return success();
    }

  private:
    void visitTranspose(
        linalg::TransposeOp transposeOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {

        // Only track layout conversion transposes
        if (!isLayoutConversionTranspose(transposeOp)) {
            results[0]->setBlocking();
            propagateIfChanged(results[0], ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (not layout conversion)\n");
            return;
        }

        auto perm = transposeOp.getPermutation();
        // Only propagate NCHW→NHWC [0,2,3,1] transposes.
        // Block NHWC→NCHW [0,3,1,2] transposes (structural NCHW region boundaries).
        bool isNchwToNhwc = isNchwToNhwcTranspose(perm);

        if (isNchwToNhwc) {
            results[0]->setPropagation(perm);
            propagateIfChanged(results[0], ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> CanPropagate (NCHW→NHWC)\n");
        }
        else {
            results[0]->setBlocking();
            propagateIfChanged(results[0], ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (NHWC→NCHW boundary)\n");
        }
    }

    void visitGeneric(
        linalg::GenericOp genericOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {
        if (!isLayoutAgnosticElementwise(genericOp)) {
            results[0]->setBlocking();
            propagateIfChanged(results[0], ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (not elementwise)\n");
            return;
        }

        // Elementwise: inherit propagation from any input
        TransposePropagationLattice *result = results[0];
        bool foundPropagation = false;

        for (auto *operand : operands) {
            if (operand->canPropagate()) {
                auto change = result->join(*operand);
                propagateIfChanged(result, change);
                foundPropagation = true;
                LLVM_DEBUG({
                    llvm::dbgs() << "  -> Inherited propagation: ";
                    result->print(llvm::dbgs());
                    llvm::dbgs() << "\n";
                });
                break;
            }
        }

        if (!foundPropagation) {
            result->setNoPropagation();
            propagateIfChanged(result, ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> NoPropagation\n");
        }
    }

    void visitExtractSlice(
        tensor::ExtractSliceOp sliceOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {
        const auto *sourceState = operands[0];
        auto *result = results[0];

        if (sourceState->canPropagate()) {
            result->setPropagation(sourceState->getPermutation());
            propagateIfChanged(result, ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> CanPropagate (extract_slice)\n");
        }
        else {
            auto change = result->join(*sourceState);
            propagateIfChanged(result, change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Inherited state\n");
        }
    }

    void visitInsertSlice(
        tensor::InsertSliceOp insertOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {
        const auto *sourceState = operands[0];
        auto *result = results[0];

        if (sourceState->canPropagate()) {
            result->setPropagation(sourceState->getPermutation());
            propagateIfChanged(result, ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> CanPropagate (insert_slice)\n");
        }
        else {
            // Inherit from dest (operand 1) if source doesn't propagate
            auto change = result->join(*operands[1]);
            propagateIfChanged(result, change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Inherited from dest\n");
        }
    }

    void visitExpandShape(
        tensor::ExpandShapeOp expandOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {

        const auto *sourceState = operands[0];
        auto *result = results[0];

        if (sourceState->canPropagate()) {
            result->setPropagation(sourceState->getPermutation());
            propagateIfChanged(result, ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> CanPropagate (expand_shape)\n");
        }
        else {
            auto change = result->join(*sourceState);
            propagateIfChanged(result, change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Inherited state\n");
        }
    }
};

//===----------------------------------------------------------------------===//
// Transformation Helpers
//===----------------------------------------------------------------------===//

/// Permute an affine map for NHWC→NCHW layout transformation.
/// For full-rank maps (4 results), transforms both result positions AND dimension references.
/// For broadcast maps (fewer results), only replaces dimension references.
///
/// Example 1 - Full rank: NHWC map (d0, d1, d2, d3) -> (d0, d1, d2, d3) with inversePerm [0,3,1,2]:
/// 1. Permute result positions: [d0, d1, d2, d3] → [d0, d3, d1, d2]
/// 2. Replace dims: d1→d2, d2→d3, d3→d1
/// 3. Final NCHW map: (d0, d1, d2, d3) -> (d0, d1, d2, d3) (identity)
///
/// Example 2 - Broadcast: NHWC map (d0, d1, d2, d3) -> (d0, 0, 0, d3) with inversePerm [0,3,1,2]:
/// 1. Permute result positions: [d0, 0, 0, d3] → [d0, d3, 0, 0]
/// 2. Replace dims: d3 (old C at pos 3) → d1 (new C at pos 1)
/// 3. Final NCHW map: (d0, d1, d2, d3) -> (d0, d1, 0, 0)
///
/// Example 3 - 2D broadcast: NHWC map (d0, d1, d2, d3) -> (d0, d3) with perm [0,2,3,1]:
/// 1. NOT permuted (only 2 results, not full rank)
/// 2. Replace dims: d3 → d1 (channel moves from pos 3 to pos 1)
/// 3. Final NCHW map: (d0, d1, d2, d3) -> (d0, d1)
static AffineMap
permuteAffineMap(AffineMap map, ArrayRef<int64_t> inversePerm, ArrayRef<int64_t> perm) {
    auto ctx = map.getContext();
    auto results = map.getResults();

    // Build dimension replacement map: d_old[i] → d_new[perm[i]]
    // perm tells us where each old dimension moves to in the new layout.
    SmallVector<AffineExpr> dimReplacements;
    dimReplacements.reserve(perm.size());
    for (auto newPos : perm) {
        dimReplacements.push_back(getAffineDimExpr(newPos, ctx));
    }

    // Check if this is a full-rank map (same number of results as permutation size)
    bool isFullRank = results.size() == inversePerm.size();

    SmallVector<AffineExpr> newResults;

    if (isFullRank) {
        // Full-rank map: permute result positions AND replace dimension references
        newResults.reserve(inversePerm.size());
        for (auto idx : inversePerm) {
            auto expr = results[idx];
            auto permutedExpr = expr.replaceDims(dimReplacements);
            newResults.push_back(permutedExpr);
        }
    }
    else {
        // Broadcast/reduced map: only replace dimension references, keep result structure
        newResults.reserve(results.size());
        for (auto expr : results) {
            auto permutedExpr = expr.replaceDims(dimReplacements);
            newResults.push_back(permutedExpr);
        }
    }

    return AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults, ctx);
}

// Transform linalg.generic to work in permuted (NCHW) layout.
static Value transformGenericOp(
    linalg::GenericOp genericOp, ArrayRef<int64_t> inversePerm, ArrayRef<int64_t> perm,
    IRRewriter &rewriter
) {
    Location loc = genericOp.getLoc();

    // Transform inputs - create fresh inverse transposes
    SmallVector<Value> newInputs;
    for (auto input : genericOp.getInputs()) {
        if (auto emptyOp = input.getDefiningOp<tensor::EmptyOp>()) {
            // Create tensor.empty with permuted shape
            auto origType = cast<RankedTensorType>(input.getType());
            SmallVector<int64_t> newShape;
            for (auto dim : inversePerm) {
                newShape.push_back(origType.getDimSize(dim));
            }
            auto newEmpty =
                tensor::EmptyOp::create(rewriter, loc, newShape, origType.getElementType());
            newInputs.push_back(newEmpty.getResult());

            LLVM_DEBUG({
                llvm::dbgs() << "║     → Created permuted tensor.empty\n";
                llvm::dbgs() << "║        Shape: ";
                for (auto dim : newShape)
                    llvm::dbgs() << dim << "x";
                llvm::dbgs() << origType.getElementType() << "\n";
            });
        }
        else {
            // Create inverse transpose to convert input to NCHW
            auto inputType = cast<RankedTensorType>(input.getType());

            // Skip inputs with rank mismatch (e.g., 1D/2D constants broadcast into 4D generic).
            // These don't need transposition - the indexing map already handles the broadcast.
            // Examples:
            // - tensor<1xbf16> with map (d0, d1, d2, d3) -> (d0)
            // - tensor<1x240xi8> with map (d0, d1, d2, d3) -> (d0, d3)
            if (inputType.getRank() != static_cast<int64_t>(inversePerm.size())) {
                newInputs.push_back(input);
                LLVM_DEBUG({
                    llvm::dbgs() << "║     → Skipping transpose for rank-" << inputType.getRank()
                                 << " broadcast input (perm is rank-" << inversePerm.size()
                                 << ")\n";
                });
                continue;
            }

            auto inverseTranspose =
                transposeValue(input, SmallVector<int64_t>(inversePerm), loc, rewriter);
            newInputs.push_back(inverseTranspose);

            LLVM_DEBUG({
                auto transposedType = cast<RankedTensorType>(inverseTranspose.getType());
                llvm::dbgs() << "║     → Created fresh inverse transpose on input\n";
                llvm::dbgs() << "║        From: ";
                for (auto dim : inputType.getShape())
                    llvm::dbgs() << dim << "x";
                llvm::dbgs() << inputType.getElementType();
                llvm::dbgs() << " To: ";
                for (auto dim : transposedType.getShape())
                    llvm::dbgs() << dim << "x";
                llvm::dbgs() << inputType.getElementType() << "\n";
            });
        }
    }

    // Create permuted output init
    auto origOutputType = cast<RankedTensorType>(genericOp.getOutputs()[0].getType());
    SmallVector<int64_t> newOutputShape;
    for (auto idx : inversePerm) {
        newOutputShape.push_back(origOutputType.getDimSize(idx));
    }

    auto newOutputType = RankedTensorType::get(newOutputShape, origOutputType.getElementType());
    auto newInit = tensor::EmptyOp::create(
        rewriter, loc, newOutputType.getShape(), newOutputType.getElementType()
    );

    // Permute indexing maps to match the NCHW layout.
    // Maps like (d0, 0, 0, d3) in NHWC become (d0, d1, 0, 0) in NCHW.
    // Maps like (d0, d1, d2, d3) in NHWC become (d0, d1, d2, d3) in NCHW.
    SmallVector<AffineMap> newMaps;
    for (auto map : genericOp.getIndexingMapsArray()) {
        auto newMap = permuteAffineMap(map, inversePerm, perm);
        newMaps.push_back(newMap);
        LLVM_DEBUG({ llvm::dbgs() << "║     → Permuted map: " << map << " → " << newMap << "\n"; });
    }

    // Rebuild generic with permuted shapes and permuted maps
    return rebuildGenericWithNewLayout(
        rewriter, genericOp, newInputs, newInit.getResult(), newMaps
    );
}

/// Transform extract_slice to work in permuted layout.
static Value transformExtractSliceOp(
    tensor::ExtractSliceOp sliceOp, ArrayRef<int64_t> perm4D, IRRewriter &rewriter
) {
    Location loc = sliceOp.getLoc();

    auto sourceType = cast<RankedTensorType>(sliceOp.getSource().getType());
    auto resultType = cast<RankedTensorType>(sliceOp.getType());
    auto inversePerm4D = Permutation(perm4D.begin(), perm4D.end()).reverse();

    // Adapt inverse perm to the source rank (handles 3D/4D/5D sources)
    auto srcInversePerm = adaptPermToRank(inversePerm4D, sourceType.getRank());
    // Adapt inverse perm to the result rank for the new result type
    auto resInversePerm = adaptPermToRank(inversePerm4D, resultType.getRank());

    Value source = sliceOp.getSource();
    if (auto emptyOp = source.getDefiningOp<tensor::EmptyOp>()) {
        SmallVector<int64_t> newShape;
        for (auto dim : srcInversePerm)
            newShape.push_back(sourceType.getDimSize(dim));
        source = tensor::EmptyOp::create(rewriter, loc, newShape, sourceType.getElementType());
        LLVM_DEBUG({
            llvm::dbgs() << "║     → Created permuted tensor.empty for source\n";
            llvm::dbgs() << "║        Shape: ";
            for (auto dim : newShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << sourceType.getElementType() << "\n";
        });
    }
    else {
        source = transposeValue(source, srcInversePerm, loc, rewriter);
        LLVM_DEBUG({
            auto transposedType = cast<RankedTensorType>(source.getType());
            llvm::dbgs() << "║     → Created fresh inverse transpose on extract_slice source\n";
            llvm::dbgs() << "║        From: ";
            for (auto dim : sourceType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << sourceType.getElementType();
            llvm::dbgs() << " To: ";
            for (auto dim : transposedType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << sourceType.getElementType() << "\n";
        });
    }

    // Permute slice parameters using source-rank inverse perm
    auto offsets = sliceOp.getMixedOffsets();
    auto sizes = sliceOp.getMixedSizes();
    auto strides = sliceOp.getMixedStrides();
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    for (auto dim : srcInversePerm) {
        newOffsets.push_back(offsets[dim]);
        newSizes.push_back(sizes[dim]);
        newStrides.push_back(strides[dim]);
    }

    // Compute new result type using result-rank inverse perm
    SmallVector<int64_t> newResultShape;
    for (auto dim : resInversePerm)
        newResultShape.push_back(resultType.getDimSize(dim));
    auto newResultType = RankedTensorType::get(newResultShape, resultType.getElementType());

    auto newSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, newResultType, source, newOffsets, newSizes, newStrides
    );
    return newSlice.getResult();
}

/// Transform insert_slice to work in permuted layout.
static Value transformInsertSliceOp(
    tensor::InsertSliceOp insertOp, ArrayRef<int64_t> perm4D, IRRewriter &rewriter
) {
    Location loc = insertOp.getLoc();
    auto inversePerm4D = Permutation(perm4D.begin(), perm4D.end()).reverse();

    // Transpose source using source-rank adapted perm
    Value source = insertOp.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto srcInversePerm = adaptPermToRank(inversePerm4D, sourceType.getRank());

    source = transposeValue(source, srcInversePerm, loc, rewriter);
    LLVM_DEBUG({
        auto transposedType = cast<RankedTensorType>(source.getType());
        llvm::dbgs() << "║     → Created fresh inverse transpose on insert_slice source\n";
        llvm::dbgs() << "║        From: ";
        for (auto dim : sourceType.getShape())
            llvm::dbgs() << dim << "x";
        llvm::dbgs() << sourceType.getElementType();
        llvm::dbgs() << " To: ";
        for (auto dim : transposedType.getShape())
            llvm::dbgs() << dim << "x";
        llvm::dbgs() << sourceType.getElementType() << "\n";
    });

    // Transpose dest using dest-rank adapted perm; dest rank == result rank
    Value dest = insertOp.getDest();
    auto destType = cast<RankedTensorType>(dest.getType());
    auto destInversePerm = adaptPermToRank(inversePerm4D, destType.getRank());

    if (auto emptyOp = dest.getDefiningOp<tensor::EmptyOp>()) {
        SmallVector<int64_t> newDestShape;
        for (auto dim : destInversePerm)
            newDestShape.push_back(destType.getDimSize(dim));
        dest = tensor::EmptyOp::create(rewriter, loc, newDestShape, destType.getElementType());
        LLVM_DEBUG({
            llvm::dbgs() << "║     → Created permuted tensor.empty for dest\n";
            llvm::dbgs() << "║        Shape: ";
            for (auto dim : newDestShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << destType.getElementType() << "\n";
        });
    }
    else {
        dest = transposeValue(dest, destInversePerm, loc, rewriter);
        LLVM_DEBUG({
            auto transposedType = cast<RankedTensorType>(dest.getType());
            llvm::dbgs(
            ) << "║     → Created fresh inverse transpose on dest (chain continuation)\n";
            llvm::dbgs() << "║        From: ";
            for (auto dim : destType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << destType.getElementType();
            llvm::dbgs() << " To: ";
            for (auto dim : transposedType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << destType.getElementType() << "\n";
        });
    }

    // Get insert parameters
    auto offsets = insertOp.getMixedOffsets();
    auto sizes = insertOp.getMixedSizes();
    auto strides = insertOp.getMixedStrides();
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    for (auto dim : destInversePerm) {
        newOffsets.push_back(offsets[dim]);
        newSizes.push_back(sizes[dim]);
        newStrides.push_back(strides[dim]);
    }

    auto newInsert = tensor::InsertSliceOp::create(
        rewriter, loc, source, dest, newOffsets, newSizes, newStrides
    );
    return newInsert.getResult();
}

/// Transform expand_shape to work in permuted layout.
static Value transformExpandShapeOp(
    tensor::ExpandShapeOp expandOp, ArrayRef<int64_t> perm4D, IRRewriter &rewriter
) {
    Location loc = expandOp.getLoc();
    Value source = expandOp.getSrc();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto resultType = cast<RankedTensorType>(expandOp.getResult().getType());
    auto inversePerm4D = Permutation(perm4D.begin(), perm4D.end()).reverse();

    // Adapt inverse perm to source rank for the transpose on the source
    auto srcInversePerm = adaptPermToRank(inversePerm4D, sourceType.getRank());
    source = transposeValue(source, srcInversePerm, loc, rewriter);

    // Adapt inverse perm to result rank for the new expand_shape output type
    auto resInversePerm = adaptPermToRank(inversePerm4D, resultType.getRank());
    SmallVector<int64_t> newOutputShape;
    for (auto dim : resInversePerm)
        newOutputShape.push_back(resultType.getDimSize(dim));

    auto newResultType = RankedTensorType::get(newOutputShape, resultType.getElementType());

    auto newSourceType = cast<RankedTensorType>(source.getType());
    auto newReassociation = mlir::getReassociationIndicesForReshape(newSourceType, newResultType);
    assert(
        newReassociation && "transformExpandShapeOp: failed to recompute reassociation "
                            "after permutation — incompatible shapes"
    );

    auto newExpand =
        tensor::ExpandShapeOp::create(rewriter, loc, newResultType, source, *newReassociation);
    return newExpand.getResult();
}

/// Post-analysis backward pass: promotes NoPropagation elementwise generics
/// whose results feed into CanPropagate ops.
///
/// Algorithm (reverse walk):
///   For each CanPropagate op (iterated last→first):
///     For each input operand:
///       If the defining op is a NoPropagation elementwise generic →
///         promote it to CanPropagate with the same NCHW→NHWC perm.
///
/// This reverse walk ensures that chains of elementwise generics are promoted
/// in a single pass. The outer fixpoint loop handles cases where the block is
/// not in strict topological order.
///
/// Only NCHW→NHWC [0,2,3,1] CanPropagate ops seed the backward walk —
/// these are the only perms produced by the forward analysis.
/// Mutates solver lattice states in-place (safe after initializeAndRun).
static void backwardPropagate(FunctionOpInterface funcOp, DataFlowSolver &solver) {
    SmallVector<Operation *> allOps;
    for (Operation &op : funcOp.getFunctionBody().front())
        allOps.push_back(&op);

    bool changed = true;
    while (changed) {
        changed = false;

        // Reverse order: process consumers before their producers so that
        for (auto it = allOps.rbegin(); it != allOps.rend(); ++it) {
            Operation *op = *it;
            if (op->getNumResults() == 0)
                continue;

            // Skip Transpose ops.
            if (isa<linalg::TransposeOp>(op))
                continue;

            const auto *lattice = solver.lookupState<TransposePropagationLattice>(op->getResult(0));
            if (!lattice || !lattice->canPropagate())
                continue;

            // Only seed backward walk from CanPropagate ops whose stored perm
            // is NCHW→NHWC [0,2,3,1] — the only perm the forward analysis ever
            // writes into a CanPropagate lattice.
            auto demandPerm = lattice->getPermutation();

            if (!isNchwToNhwcTranspose(demandPerm))
                continue;

            // Walk every input operand of this CanPropagate op
            // Use the operand Value directly for the lattice lookup — this is
            // correct for multi-result ops and avoids getResult(0) confusion.
            for (Value operand : op->getOperands()) {
                // Skip block arguments (no defining op).
                if (!operand.getDefiningOp())
                    continue;

                const auto *defLattice = solver.lookupState<TransposePropagationLattice>(operand);
                if (!defLattice)
                    continue;

                // Skip already-promoted or Blocking ops.
                if (defLattice->canPropagate() || defLattice->isBlocking())
                    continue;

                // Only elementwise generics are safe to promote.
                auto genericOp = dyn_cast<linalg::GenericOp>(operand.getDefiningOp());
                if (!genericOp || !isLayoutAgnosticElementwise(genericOp))
                    continue;

                // Promote in-place.
                const_cast<TransposePropagationLattice *>(defLattice)->setPropagation(demandPerm);
                changed = true;
                LLVM_DEBUG({
                    llvm::dbgs() << "  [BackwardPromote] ";
                    operand.print(llvm::dbgs());
                    llvm::dbgs() << " → CanPropagate[";
                    for (auto p : demandPerm)
                        llvm::dbgs() << p << ",";
                    llvm::dbgs() << "]\n";
                });
            }
        }
    }
}

// Wrap propagatable ops with inverse/forward transpose pairs.
// For ops marked CanPropagate, insert inverse transpose before (NHWC→NCHW)
// and forward transpose after (NCHW→NHWC). Canonicalization will cancel
// adjacent pairs, eliminating redundant transposes between NCHW regions.
static void insertTransposePairsAroundPropagateOps(
    FunctionOpInterface funcOp, DataFlowSolver &solver, IRRewriter &rewriter
) {

    SmallVector<Operation *> allOps;
    for (Operation &op : funcOp.getFunctionBody().front())
        allOps.push_back(&op);

    // Print the state of all ops (lattice already updated by backwardPropagate).
    LLVM_DEBUG({
        llvm::dbgs() << "=== Print the state of ops ===\n";
        for (Operation *op : allOps) {
            llvm::dbgs() << "Op: " << op->getName();
            if (op->getNumResults() > 0) {
                llvm::dbgs() << ", Result: " << op->getResult(0);
                const auto *lattice =
                    solver.lookupState<TransposePropagationLattice>(op->getResult(0));
                if (lattice) {
                    llvm::dbgs() << "\n  Lattice state: ";
                    lattice->print(llvm::dbgs());
                }
            }
            llvm::dbgs() << "\n";
        }
    });

    for (Operation *op : allOps) {
        if (op->getNumResults() == 0)
            continue;

        if (isa<linalg::TransposeOp>(op))
            continue;

        const auto *lattice = solver.lookupState<TransposePropagationLattice>(op->getResult(0));
        if (!lattice || !lattice->canPropagate())
            continue;

        auto perm4D = lattice->getPermutation(); // always 4D from lattice

        int64_t opRank = cast<RankedTensorType>(op->getResult(0).getType()).getRank();
        auto perm = adaptPermToRank(perm4D, opRank);
        auto inversePerm = Permutation(perm.begin(), perm.end()).reverse();

        // Transform op to work in NCHW layout
        rewriter.setInsertionPoint(op);
        Value transformedResult = nullptr;

        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            transformedResult = transformGenericOp(genericOp, inversePerm, perm, rewriter);
        }
        else if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
            transformedResult = transformExtractSliceOp(sliceOp, perm4D, rewriter);
        }
        else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
            transformedResult = transformInsertSliceOp(insertOp, perm4D, rewriter);
        }
        else if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
            transformedResult = transformExpandShapeOp(expandOp, perm4D, rewriter);
        }
        else {
            continue;
        }

        if (!transformedResult)
            continue;

        // Insert forward transpose after transformed op
        auto transformedType = cast<RankedTensorType>(transformedResult.getType());
        auto forwardPerm = adaptPermToRank(perm4D, transformedType.getRank());
        Value finalResult = transposeValue(transformedResult, forwardPerm, op->getLoc(), rewriter);
        rewriter.replaceOp(op, finalResult);
    }
}

//===----------------------------------------------------------------------===//
// Transpose Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Fold transpose of tensor.empty by using the output empty directly.
/// Replaces: transpose(input, tensor.empty<transposed_shape>, perm) →
/// tensor.empty<transposed_shape> This eliminates no-op transposes where the init is already the
/// correct transposed shape.
struct FoldTransposeOfEmpty : OpRewritePattern<linalg::TransposeOp> {
    using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::TransposeOp transposeOp, PatternRewriter &rewriter) const override {
        // Check if init (output) is tensor.empty
        auto emptyOp = transposeOp.getInit().getDefiningOp<tensor::EmptyOp>();
        if (!emptyOp)
            return failure();

        // Check if input is also tensor.empty (transpose of empty)
        auto inputEmptyOp = transposeOp.getInput().getDefiningOp<tensor::EmptyOp>();
        if (!inputEmptyOp)
            return failure();

        // Both input and output are empty tensors - the init already has the correct shape
        rewriter.replaceOp(transposeOp, transposeOp.getInit());
        return success();
    }
};

/// Compose back-to-back transposes: transpose(transpose(x, p1), p2) →
/// transpose(x, compose(p1, p2)).
///
/// Fixes upstream FoldTransposeWithTranspose which reuses the outer
/// transpose's init tensor.  After composition the init shape may be wrong
/// (e.g. when p1∘p2 = identity), causing fold() to crash with
/// "Assertion `false && "incorrect fold result type"' failed" at
/// mlir/lib/IR/Operation.cpp:624.
///
/// Our fix: build a fresh tensor.empty with the correct composed shape.
/// When the composed permutation is identity, eliminate both transposes.
struct ComposeTransposeOps : OpRewritePattern<linalg::TransposeOp> {
    using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::TransposeOp transposeOp, PatternRewriter &rewriter) const override {
        auto defTransposeOp = transposeOp.getInput().getDefiningOp<linalg::TransposeOp>();
        if (!defTransposeOp)
            return failure();

        ArrayRef<int64_t> defPerms = defTransposeOp.getPermutation();
        ArrayRef<int64_t> perms = transposeOp.getPermutation();

        SmallVector<int64_t> foldedPerms;
        foldedPerms.reserve(perms.size());
        for (int64_t p : perms)
            foldedPerms.push_back(defPerms[p]);

        Value origInput = defTransposeOp.getInput();

        // If the composed permutation is identity, both transposes cancel out.
        bool isIdentity = true;
        for (int64_t i = 0; i < static_cast<int64_t>(foldedPerms.size()); ++i) {
            if (foldedPerms[i] != i) {
                isIdentity = false;
                break;
            }
        }
        if (isIdentity) {
            rewriter.replaceOp(transposeOp, origInput);
            return success();
        }

        // Build init tensor with the correct shape for the composed perm.
        auto origType = cast<RankedTensorType>(origInput.getType());
        SmallVector<int64_t> newInitShape;
        newInitShape.reserve(foldedPerms.size());
        for (int64_t p : foldedPerms)
            newInitShape.push_back(origType.getDimSize(p));

        auto newInit = tensor::EmptyOp::create(
            rewriter, transposeOp.getLoc(), newInitShape, origType.getElementType()
        );

        rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
            transposeOp, origInput, newInit, foldedPerms
        );
        return success();
    }
};

//===----------------------------------------------------------------------===//
// OptimizeTransposeLayoutPass
//===----------------------------------------------------------------------===//

/// Pass implementation that applies transpose optimization patterns:
/// Also includes transpose canonicalization for back-to-back cancellation.
class OptimizeTransposeLayoutPass
    : public impl::OptimizeTransposeLayoutBase<OptimizeTransposeLayoutPass> {
  public:
    using OptimizeTransposeLayoutBase::OptimizeTransposeLayoutBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        // Step 1: Run dataflow analysis
        DataFlowSolver solver;
        solver.load<dataflow::DeadCodeAnalysis>();   // Standard dead code analysis
        solver.load<TransposePropagationAnalysis>(); // Our custom analysis

        if (failed(solver.initializeAndRun(funcOp))) {
            return signalPassFailure();
        }

        // Step 1b: Backward promotion — promote NoPropagation elementwise ops
        // that feed into CanPropagate consumers (e.g. residual branches).
        // Mutates the solver lattice in-place before transformation begins.
        backwardPropagate(funcOp, solver);

        // Step 2: Create transpose pairs around propagate ops using dataflow results
        IRRewriter rewriter(ctx);
        insertTransposePairsAroundPropagateOps(funcOp, solver, rewriter);

        // Step 3: Apply transpose canonicalization for back-to-back cancellation
        // Register the upstream canonicalization patterns (identity removal, etc.)
        // plus our custom patterns at higher benefit:
        // - FoldTransposeOfEmpty: transpose(empty) → empty
        // - ComposeTransposeOps: fixes buggy upstream FoldTransposeWithTranspose
        RewritePatternSet patterns(ctx);
        linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
        patterns.add<FoldTransposeOfEmpty, ComposeTransposeOps>(ctx, /*benefit=*/2);

        // Add MLIR's built-in constant folding for linalg ops (includes transpose)
        // transpose(constant) → constant
        linalg::ControlFusionFn alwaysFold = [](OpOperand *) { return true; };
        linalg::populateConstantFoldLinalgOperations(patterns, alwaysFold);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        SmallVector<Operation *> transposeOps;
        funcOp.walk([&](linalg::TransposeOp op) { transposeOps.push_back(op); });

        if (failed(applyOpPatternsGreedily(transposeOps, frozenPatterns))) {
            LLVM_DEBUG(llvm::dbgs() << "WARNING: Canonicalization failed\n");
        }
    }
};

/// Factory function to create the pass.
std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeTransposeLayoutPass() {
    return std::make_unique<OptimizeTransposeLayoutPass>();
}

} // namespace mlir::syna::torq
