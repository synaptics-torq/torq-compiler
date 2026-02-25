// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// OptimizeTransposeLayoutPass
//===----------------------------------------------------------------------===//
//
// **APPROACH**: Create transpose pairs around layout-agnostic operations,
// then rely on canonicalization to eliminate redundant transposes.
//
// **STRATEGY**: we create MORE transposes
// initially around operations that can work in different layouts, then let
// canonicalization clean up the redundant pairs.
//
// **STEPS**:
//   1. Run dataflow analysis ONCE on entire function
//      → TransposePropagationAnalysis identifies which ops can work in different layouts
//   2. For each operation that can propagate a transpose:
//      - Create inverse transposes on inputs (convert to needed layout)
//      - Transform the operation to work in inverse-permuted layout
//      - Create forward transpose on output (convert back to expected layout)
//   3. Transpose canonicalization eliminates back-to-back transpose pairs
//
// **KEY INSIGHT**: We temporarily increase transposes, but canonicalization
// removes redundant pairs, resulting in transposes moving closer to ops that
// actually need specific layouts.
//
// **EXAMPLE**:
//   BEFORE:
//     %t2 = elementwise %input             // works in NHWC
//     %t1 = transpose [0,2,3,1] %t2        // NCHW→NHWC
//     %out = conv %t2                      // requires NHWC
//     %t1 = transpose [0,2,3,1] %out       // NCHW→NHWC
//
//   AFTER TRANSFORMATION (more transposes initially):
//     %inv2 = transpose [0,3,1,2] %t1             // NHWC→NCHW (inverse)
//     %elem_new = elementwise %inv2               // elementwise in NCHW
//     %t1 =  transpose [0,2,3,1] %elem_new        // NCHW→NHWC (forward)
//     %fwd = transpose [0,2,3,1] %t1              // NCHW→NHWC
//     %out = conv %fwd                            // requires NHWC
//     %t1 = transpose [0,2,3,1] %out              // NCHW→NHWC
//
//   AFTER CANONICALIZATION (redundant pairs cancelled):
//     %inv2 = transpose [0,3,1,2] %t1             // NHWC→NCHW (inverse)
//     %elem_new = elementwise %inv2               // elementwise in original NCHW
//     %out = conv %elem_new                       // requires NHWC
//     %t1 = transpose [0,2,3,1] %out              // NCHW→NHWC only when needed
//
//
//   for smaller models we can't expect much improvement but for
//   larger models with many elementwise ops we can get significant reduction in transposes.
//===----------------------------------------------------------------------===//

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-transpose-layout"

namespace mlir::syna::torq {

//===----------------------------------------------------------------------===//
// Simple Helper Functions
//===----------------------------------------------------------------------===//

/// Check if affine map is elementwise (no dimension reordering)
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

/// Check if a linalg.generic is layout-agnostic elementwise (used by dataflow analysis)
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

/// Check if op can propagate layout changes (layout-agnostic)
static bool canPropagate(Operation *op) {
    // linalg.generic elementwise ops can propagate
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // Must be all parallel iterators
        return isLayoutAgnosticElementwise(genericOp);
    }

    // Slice ops can propagate
    if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op))
        return true;

    // Everything else (conv, matmul, etc.) cannot propagate
    return false;
}

/// Check if transpose is NCHW↔NHWC layout conversion (4D only)
static bool isLayoutConversionTranspose(linalg::TransposeOp transposeOp) {
    auto perm = transposeOp.getPermutation();
    if (perm.size() != 4)
        return false;

    // NCHW→NHWC: [0, 2, 3, 1] or NHWC→NCHW: [0, 3, 1, 2]
    return (perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1) ||
           (perm[0] == 0 && perm[1] == 3 && perm[2] == 1 && perm[3] == 2);
}

/// Get inverse permutation: [0,2,3,1] → [0,3,1,2]
static SmallVector<int64_t> getInversePermutation(ArrayRef<int64_t> perm) {
    SmallVector<int64_t> inverse(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
        inverse[perm[i]] = i;
    return inverse;
}

//===----------------------------------------------------------------------===//
// Topological Sorting for Entry Block
//===----------------------------------------------------------------------===//

/// Collect all computational ops from entry block in topological order
static SmallVector<Operation *> collectTopologicalOrder(FunctionOpInterface funcOp) {
    SmallVector<Operation *> allOps;
    Block &entryBlock = funcOp.getFunctionBody().front();
    // Collect computational ops (skip constants, allocations, etc.)
    for (Operation &op : entryBlock) {
        StringRef opName = op.getName().getStringRef();
        if (opName.starts_with("arith") || opName.starts_with("hal.") ||
            opName.starts_with("flow.") || opName == "func.return")
            continue;
        allOps.push_back(&op);
    }

    // Build in-degree map
    DenseMap<Operation *, int> inDegree;
    for (Operation *op : allOps)
        inDegree[op] = 0;

    for (Operation *op : allOps) {
        for (Value result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
                if (inDegree.count(user))
                    inDegree[user]++;
            }
        }
    }

    // Topological sort using queue
    SmallVector<Operation *> sorted;
    SmallVector<Operation *> queue;

    for (Operation *op : allOps) {
        if (inDegree[op] == 0)
            queue.push_back(op);
    }

    while (!queue.empty()) {
        Operation *op = queue.pop_back_val();
        sorted.push_back(op);

        for (Value result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
                if (inDegree.count(user) && --inDegree[user] == 0)
                    queue.push_back(user);
            }
        }
    }

    return sorted;
}

//===----------------------------------------------------------------------===//
// Dataflow Lattice for Transpose Propagation
//===----------------------------------------------------------------------===//

/// Lattice representing transpose propagation state of an SSA value
class TransposePropagationLattice : public dataflow::AbstractSparseLattice {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransposePropagationLattice)
    using AbstractSparseLattice::AbstractSparseLattice;

    enum class PropagationState {
        Unknown,       // Not yet analyzed
        NoPropagation, // No transpose to propagate
        CanPropagate,  // Can propagate transpose through this value
        Blocking       // Cannot propagate
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
    ArrayRef<int64_t> getPermutation() const { return permutation; }

  private:
    PropagationState state = PropagationState::Unknown;
    SmallVector<int64_t, 4> permutation;
};

//===----------------------------------------------------------------------===//
// Dataflow Analysis
//===----------------------------------------------------------------------===//

/// Dataflow analysis to determine which operations can propagate transposes
class TransposePropagationAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<TransposePropagationLattice> {
  public:
    using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

    TransposePropagationAnalysis(DataFlowSolver &solver) : SparseForwardDataFlowAnalysis(solver) {}

    void setToEntryState(TransposePropagationLattice *lattice) override {
        // Entry state: no transpose propagation (normal flow)
        lattice->setNoPropagation();
        propagateIfChanged(lattice, ChangeResult::Change);
    }

    void visitOperation(
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

        // Handle linalg.transpose - source of propagation
        if (auto transposeOp = dyn_cast<linalg::TransposeOp>(op)) {
            visitTranspose(transposeOp, operands, results);
            return;
        }

        // Handle linalg.generic - check if layout-agnostic elementwise
        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            visitGeneric(genericOp, operands, results);
            return;
        }

        // Handle tensor.extract_slice - can propagate with parameter adjustment
        if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
            visitExtractSlice(sliceOp, operands, results);
            return;
        }

        // Handle tensor.insert_slice - can propagate with parameter adjustment
        if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
            visitInsertSlice(insertOp, operands, results);
            return;
        }

        // Handle torq_hl.* operations - these block transpose propagation
        if (op->getDialect() && op->getDialect()->getNamespace() == "torq_hl") {
            for (auto *result : results) {
                result->setBlocking();
                propagateIfChanged(result, ChangeResult::Change);
            }
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (torq_hl operation)\n");
            return;
        }

        // Handle tensor.collapse_shape - these block transpose propagation
        if (isa<tensor::CollapseShapeOp>(op)) {
            for (auto *result : results) {
                result->setBlocking();
                propagateIfChanged(result, ChangeResult::Change);
            }
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (tensor.collapse_shape)\n");
            return;
        }

        // Handle tensor.empty - keep as Unknown (flexible, can adapt to any layout)
        if (isa<tensor::EmptyOp>(op)) {
            // Keep in Unknown state - tensor.empty can adapt to whatever layout is needed
            // If used in a transposed context, it can be created in that layout
            // If used in a normal context, it remains normal
            LLVM_DEBUG(llvm::dbgs() << "  -> tensor.empty: Keep Unknown (flexible)\n");
            return;
        }

        // Default: All other operations can propagate transpose
        // Operations with no operands get NoPropagation (e.g., constants)
        // Operations with operands inherit propagation state from first operand
        for (auto *result : results) {
            if (operands.empty()) {
                result->setNoPropagation();
                propagateIfChanged(result, ChangeResult::Change);
            }
            else {
                propagateIfChanged(result, result->join(*operands[0]));
            }
        }
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
        results[0]->setPropagation(perm);
        propagateIfChanged(results[0], ChangeResult::Change);

        LLVM_DEBUG({
            llvm::dbgs() << "  -> CanPropagate [";
            for (auto p : perm)
                llvm::dbgs() << p << ",";
            llvm::dbgs() << "]\n";
        });
    }

    void visitGeneric(
        linalg::GenericOp genericOp, ArrayRef<const TransposePropagationLattice *> operands,
        ArrayRef<TransposePropagationLattice *> results
    ) {

        // Check if this is a layout-agnostic elementwise operation
        if (!isLayoutAgnosticElementwise(genericOp)) {
            results[0]->setBlocking();
            propagateIfChanged(results[0], ChangeResult::Change);
            LLVM_DEBUG(llvm::dbgs() << "  -> Blocking (not layout-agnostic)\n");
            return;
        }

        // Elementwise can propagate - inherit from any input that can propagate
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

        // Extract slice can propagate transpose with adjusted parameters
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

        // Insert slice can propagate if source (operand 0) has transpose
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
};

//===----------------------------------------------------------------------===//
// Direct Transpose Sinking Using Dataflow Results
//===----------------------------------------------------------------------===//

/// Helper: Transform linalg.generic to work in permuted layout
static Value transformGenericOp(
    linalg::GenericOp genericOp, ArrayRef<int64_t> inversePerm, ArrayRef<int64_t> perm,
    IRRewriter &rewriter
) {
    Location loc = genericOp.getLoc();

    // Collect transformed inputs - always create fresh transformations locally
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
                rewriter.create<tensor::EmptyOp>(loc, newShape, origType.getElementType());
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
            // Always create fresh inverse transpose for each use
            auto inputType = cast<RankedTensorType>(input.getType());
            SmallVector<int64_t> transposedShape;
            for (auto dim : inversePerm) {
                transposedShape.push_back(inputType.getDimSize(dim));
            }
            auto inverseInit =
                rewriter.create<tensor::EmptyOp>(loc, transposedShape, inputType.getElementType());
            auto inverseTranspose =
                rewriter.create<linalg::TransposeOp>(loc, input, inverseInit, inversePerm);
            newInputs.push_back(inverseTranspose.getResult()[0]);

            LLVM_DEBUG({
                llvm::dbgs() << "║     → Created fresh inverse transpose on input\n";
                llvm::dbgs() << "║        From: ";
                for (auto dim : inputType.getShape())
                    llvm::dbgs() << dim << "x";
                llvm::dbgs() << inputType.getElementType();
                llvm::dbgs() << " To: ";
                for (auto dim : transposedShape)
                    llvm::dbgs() << dim << "x";
                llvm::dbgs() << inputType.getElementType() << "\n";
            });
        }
    }

    // Compute permuted output shape
    auto origOutputType = cast<RankedTensorType>(genericOp.getOutputs()[0].getType());
    SmallVector<int64_t> newOutputShape;
    for (auto idx : inversePerm) {
        newOutputShape.push_back(origOutputType.getDimSize(idx));
    }

    auto newOutputType = RankedTensorType::get(newOutputShape, origOutputType.getElementType());
    auto newInit = rewriter.create<tensor::EmptyOp>(
        loc, newOutputType.getShape(), newOutputType.getElementType()
    );

    // Create transformed generic op (same indexing maps - they're layout-agnostic)
    auto newGenericOp = rewriter.create<linalg::GenericOp>(
        loc, newOutputType, newInputs, ValueRange{newInit}, genericOp.getIndexingMapsArray(),
        genericOp.getIteratorTypesArray()
    );
    // Clone computation body
    IRMapping mapper;
    genericOp.getRegion().cloneInto(&newGenericOp.getRegion(), mapper);

    return newGenericOp.getResult(0);
}

/// Helper: Transform extract_slice to work in permuted layout
static Value transformExtractSliceOp(
    tensor::ExtractSliceOp sliceOp, ArrayRef<int64_t> inversePerm, ArrayRef<int64_t> perm,
    IRRewriter &rewriter
) {
    Location loc = sliceOp.getLoc();

    // Handle source similar to transformGenericOp approach
    Value source = sliceOp.getSource();
    if (auto emptyOp = source.getDefiningOp<tensor::EmptyOp>()) {
        // Create tensor.empty with permuted shape
        auto origType = cast<RankedTensorType>(source.getType());
        SmallVector<int64_t> newShape;
        for (auto dim : inversePerm) {
            newShape.push_back(origType.getDimSize(dim));
        }
        source = rewriter.create<tensor::EmptyOp>(loc, newShape, origType.getElementType());
        LLVM_DEBUG({
            llvm::dbgs() << "║     → Created permuted tensor.empty for source\n";
            llvm::dbgs() << "║        Shape: ";
            for (auto dim : newShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << origType.getElementType() << "\n";
        });
    }
    else {
        // Always create fresh inverse transpose for source
        auto sourceType = cast<RankedTensorType>(source.getType());

        // Handle rank expansion if needed (3D→4D)
        SmallVector<int64_t> sourceShape;
        if (sourceType.getRank() < inversePerm.size()) {
            // Expand shape by adding leading 1s
            for (size_t i = 0; i < inversePerm.size() - sourceType.getRank(); ++i) {
                sourceShape.push_back(1);
            }
        }
        for (auto dim : sourceType.getShape()) {
            sourceShape.push_back(dim);
        }

        SmallVector<int64_t> transposedShape;
        for (auto dim : inversePerm) {
            transposedShape.push_back(sourceShape[dim]);
        }
        auto inverseInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedShape, sourceType.getElementType());
        auto inverseTranspose =
            rewriter.create<linalg::TransposeOp>(loc, source, inverseInit, inversePerm);
        source = inverseTranspose.getResult()[0];

        LLVM_DEBUG({
            llvm::dbgs() << "║     → Created fresh inverse transpose on extract_slice source\n";
            llvm::dbgs() << "║        From: ";
            for (auto dim : sourceType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << sourceType.getElementType();
            llvm::dbgs() << " To: ";
            for (auto dim : transposedShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << sourceType.getElementType() << "\n";
        });
    }

    // Get slice parameters
    auto offsets = sliceOp.getMixedOffsets();
    auto sizes = sliceOp.getMixedSizes();
    auto strides = sliceOp.getMixedStrides();

    // Permute parameters according to inverse permutation
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    for (auto dim : inversePerm) {
        newOffsets.push_back(offsets[dim]);
        newSizes.push_back(sizes[dim]);
        newStrides.push_back(strides[dim]);
    }

    // Create transformed extract_slice
    auto newSlice =
        rewriter.create<tensor::ExtractSliceOp>(loc, source, newOffsets, newSizes, newStrides);

    return newSlice.getResult();
}

/// Helper: Transform insert_slice to work in permuted layout
static Value transformInsertSliceOp(
    tensor::InsertSliceOp insertOp, ArrayRef<int64_t> inversePerm, ArrayRef<int64_t> perm,
    IRRewriter &rewriter
) {
    Location loc = insertOp.getLoc();

    // Always create fresh inverse transpose for source
    Value source = insertOp.getSource();
    auto sourceType = cast<RankedTensorType>(source.getType());

    // Handle rank expansion if needed (3D→4D)
    SmallVector<int64_t> sourceShape;
    if (sourceType.getRank() < inversePerm.size()) {
        // Expand shape by adding leading 1s
        for (size_t i = 0; i < inversePerm.size() - sourceType.getRank(); ++i) {
            sourceShape.push_back(1);
        }
    }
    for (auto dim : sourceType.getShape()) {
        sourceShape.push_back(dim);
    }

    SmallVector<int64_t> transposedShape;
    for (auto dim : inversePerm) {
        transposedShape.push_back(sourceShape[dim]);
    }
    auto inverseInit =
        rewriter.create<tensor::EmptyOp>(loc, transposedShape, sourceType.getElementType());
    auto inverseTranspose =
        rewriter.create<linalg::TransposeOp>(loc, source, inverseInit, inversePerm);
    source = inverseTranspose.getResult()[0];

    LLVM_DEBUG({
        llvm::dbgs() << "║     → Created fresh inverse transpose on insert_slice source\n";
        llvm::dbgs() << "║        From: ";
        for (auto dim : sourceType.getShape())
            llvm::dbgs() << dim << "x";
        llvm::dbgs() << sourceType.getElementType();
        llvm::dbgs() << " To: ";
        for (auto dim : transposedShape)
            llvm::dbgs() << dim << "x";
        llvm::dbgs() << sourceType.getElementType() << "\n";
    });

    // Handle destination similar to transformGenericOp approach
    Value dest = insertOp.getDest();
    if (auto emptyOp = dest.getDefiningOp<tensor::EmptyOp>()) {
        // Create tensor.empty with permuted shape
        auto origDestType = cast<RankedTensorType>(dest.getType());
        SmallVector<int64_t> newDestShape;
        for (auto dim : inversePerm) {
            newDestShape.push_back(origDestType.getDimSize(dim));
        }
        dest = rewriter.create<tensor::EmptyOp>(loc, newDestShape, origDestType.getElementType());
        LLVM_DEBUG({
            llvm::dbgs() << "║     → Created permuted tensor.empty for dest\n";
            llvm::dbgs() << "║        Shape: ";
            for (auto dim : newDestShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << origDestType.getElementType() << "\n";
        });
    }
    else {
        // Always create fresh inverse transpose for destination (maintains chain)
        auto destType = cast<RankedTensorType>(dest.getType());

        // Handle rank expansion if needed (3D→4D)
        SmallVector<int64_t> destShape;
        if (destType.getRank() < inversePerm.size()) {
            // Expand shape by adding leading 1s
            for (size_t i = 0; i < inversePerm.size() - destType.getRank(); ++i) {
                destShape.push_back(1);
            }
        }
        for (auto dim : destType.getShape()) {
            destShape.push_back(dim);
        }

        SmallVector<int64_t> transposedDestShape;
        for (auto dim : inversePerm) {
            transposedDestShape.push_back(destShape[dim]);
        }
        auto destInverseInit =
            rewriter.create<tensor::EmptyOp>(loc, transposedDestShape, destType.getElementType());
        auto destInverseTranspose =
            rewriter.create<linalg::TransposeOp>(loc, dest, destInverseInit, inversePerm);
        dest = destInverseTranspose.getResult()[0];

        LLVM_DEBUG({
            llvm::dbgs(
            ) << "║     → Created fresh inverse transpose on dest (chain continuation)\n";
            llvm::dbgs() << "║        From: ";
            for (auto dim : destType.getShape())
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << destType.getElementType();
            llvm::dbgs() << " To: ";
            for (auto dim : transposedDestShape)
                llvm::dbgs() << dim << "x";
            llvm::dbgs() << destType.getElementType() << "\n";
        });
    }

    // Get insert parameters
    auto offsets = insertOp.getMixedOffsets();
    auto sizes = insertOp.getMixedSizes();
    auto strides = insertOp.getMixedStrides();

    // Permute parameters according to inverse permutation
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    for (auto dim : inversePerm) {
        newOffsets.push_back(offsets[dim]);
        newSizes.push_back(sizes[dim]);
        newStrides.push_back(strides[dim]);
    }

    // Create transformed insert_slice
    auto newInsert =
        rewriter.create<tensor::InsertSliceOp>(loc, source, dest, newOffsets, newSizes, newStrides);

    return newInsert.getResult();
}

/// Create transpose pairs around CanPropagate ops for eventual cancellation
/// Strategy: For ops that CanPropagate[perm], create inverse transpose before
/// and forward transpose after, then rely on canonicalization to cancel adjacent pairs.
/// This transforms: transpose → canPropagate_op → blocking_op
/// Into: canPropagate_op' → transpose → blocking_op (where ' means transformed layout)
static void insertTransposePairsAroundPropagateOps(
    FunctionOpInterface funcOp, DataFlowSolver &solver, IRRewriter &rewriter
) {
    auto allOps = collectTopologicalOrder(funcOp);

    for (Operation *op : allOps) {
        // Skip ops with no results
        if (op->getNumResults() == 0) {
            continue;
        }

        const auto *lattice = solver.lookupState<TransposePropagationLattice>(op->getResult(0));
        if (!lattice || !lattice->canPropagate())
            continue;

        auto perm = lattice->getPermutation();
        auto inversePerm = getInversePermutation(perm);

        // Skip if this is already a transpose - don't create redundant pairs
        if (isa<linalg::TransposeOp>(op)) {
            continue;
        }

        // Step 1: Create transformed version of the operation with permuted shapes
        rewriter.setInsertionPoint(op);
        Value transformedResult = nullptr;

        if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
            transformedResult = transformGenericOp(genericOp, inversePerm, perm, rewriter);
        }
        else if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
            transformedResult = transformExtractSliceOp(sliceOp, inversePerm, perm, rewriter);
        }
        else if (auto insertOp = dyn_cast<tensor::InsertSliceOp>(op)) {
            transformedResult = transformInsertSliceOp(insertOp, inversePerm, perm, rewriter);
        }
        else {
            continue;
        }

        // Step 2: Insert forward transpose after transformed op to restore original layout
        Value originalResult = op->getResult(0);
        auto originalType = cast<RankedTensorType>(originalResult.getType());
        auto transformedType = cast<RankedTensorType>(transformedResult.getType());

        // Handle rank mismatch by expanding original shape with leading 1s
        // Example: 3D tensor<80x80x32> becomes 4D tensor<1x80x80x32>
        SmallVector<int64_t> expandedOriginalShape;
        if (originalType.getRank() < transformedType.getRank()) {
            // Add leading 1s to match transformed rank
            int64_t rankDiff = transformedType.getRank() - originalType.getRank();
            for (int64_t i = 0; i < rankDiff; ++i) {
                expandedOriginalShape.push_back(1);
            }
        }
        // Add all original dimensions
        for (auto dim : originalType.getShape()) {
            expandedOriginalShape.push_back(dim);
        }

        // The forward transpose should convert from transformed layout back to original layout
        // Use perm (transformedType is in inverse-permuted layout, apply perm to get back)
        // and the output shape should match the expanded original shape
        auto forwardInit = rewriter.create<tensor::EmptyOp>(
            op->getLoc(), expandedOriginalShape, originalType.getElementType()
        );

        auto forwardTranspose = rewriter.create<linalg::TransposeOp>(
            op->getLoc(), transformedResult, forwardInit, perm
        );

        Value finalResult = forwardTranspose.getResult()[0];
        // Step 3: Replace original op's result with final result (after transpose and any rank
        // adjustment)
        rewriter.replaceOp(op, finalResult);
    }
}

//===----------------------------------------------------------------------===//
// OptimizeTransposeLayoutPass
//===----------------------------------------------------------------------===//

/// Pass implementation that applies transpose optimization patterns:
/// Also includes transpose canonicalization for back-to-back cancellation.
class OptimizeTransposeLayoutPass
    : public OptimizeTransposeLayoutBase<OptimizeTransposeLayoutPass> {
  public:
    using OptimizeTransposeLayoutBase::OptimizeTransposeLayoutBase;

    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        // Collect topological order for stats
        auto allOps = collectTopologicalOrder(funcOp);

        // Step 1: Run dataflow analysis
        DataFlowSolver solver;
        solver.load<dataflow::DeadCodeAnalysis>();          // Standard dead code analysis
        solver.load<dataflow::SparseConstantPropagation>(); // Required for sparse analysis
        solver.load<TransposePropagationAnalysis>();        // Our custom analysis

        if (failed(solver.initializeAndRun(funcOp))) {
            return signalPassFailure();
        }

        // Step 2: Create transpose pairs around propagate ops using dataflow results
        IRRewriter rewriter(ctx);
        insertTransposePairsAroundPropagateOps(funcOp, solver, rewriter);

        // Step 3: Apply transpose canonicalization for back-to-back cancellation
        RewritePatternSet patterns(ctx);
        linalg::TransposeOp::getCanonicalizationPatterns(patterns, ctx);

        auto frozenPatterns =
            FrozenRewritePatternSet(std::move(patterns), disabledPatterns, enabledPatterns);

        if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns))) {
            LLVM_DEBUG(llvm::dbgs() << "WARNING: Canonicalization failed\n");
        }
    }
};

/// Factory function to create the pass.
std::unique_ptr<InterfacePass<FunctionOpInterface>> createOptimizeTransposeLayoutPass() {
    return std::make_unique<OptimizeTransposeLayoutPass>();
}

} // namespace mlir::syna::torq