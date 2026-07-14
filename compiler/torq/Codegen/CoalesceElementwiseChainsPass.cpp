// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CoalesceElementwiseChainsPass
// -----------------------------
// erf / tanh (and int8 mul) become a long chain of small elementwise
// linalg.generic ops. With slicing on, each op is sliced into its own scf.forall
// and writes its full result to DRAM. These ops are memory-bound, so the extra
// DRAM traffic makes the chain slower.
//
// This pass runs just before LinalgSlicing. It groups a connected chain of
// pure-elementwise ops into one fuse group. LinalgSlicing then slices the whole
// chain as one scf.forall, so the in-between values stay in LRAM.
//
// It groups a chain only if every op is elementwise; a conv/matmul group is
// never touched. Same behavior as before: it does nothing when sliceCount < 2,
// and runs only when slicing runs.

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::syna::torq {

namespace {

// True if `op` is a pure-elementwise linalg.generic that can join a chain:
// all-parallel, projected-permutation maps (linalg::isElementwise), static shape.
//
// We do NOT reject ops that contain a trunc. tanh's chain has an integer block
// (a trunci from the exponent math) in the middle, so excluding trunc would split
// the chain. A lone trunc op is safe: it forms a size-1 region and is skipped.
// The separate "don't slice a standalone trunc" guard lives in ElementwisePattern
// and is unchanged.
bool isCoalescableElementwise(Operation *op) {
    auto genericOp = dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
        return false;
    if (!linalg::isElementwise(genericOp))
        return false;
    if (!cast<ShapedType>(genericOp->getResult(0).getType()).hasStaticShape())
        return false;
    return true;
}

// For each fuse-group id, record whether ALL of its ops are coalescable
// elementwise. Done in one walk. A singleton or an all-elementwise pattern group
// (e.g. mul+clamp) maps to true; a group with any conv/matmul op maps to false
// and becomes a hard boundary. Built once so the check below is an O(1) lookup.
llvm::DenseMap<int64_t, bool> computeGroupCoalescableMap(FunctionOpInterface funcOp) {
    llvm::DenseMap<int64_t, bool> groupAllCoalescable;
    funcOp->walk([&](Operation *op) {
        auto arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
        if (!arr)
            return;
        bool coalescable = isCoalescableElementwise(op);
        for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>()) {
            auto [it, inserted] =
                groupAllCoalescable.try_emplace(fuseGroupAttr.getInt(), coalescable);
            if (!inserted)
                it->second = it->second && coalescable;
        }
    });
    return groupAllCoalescable;
}

// A chain member: a coalescable elementwise op that is either ungrouped, or in a
// fuse group whose ops are ALL coalescable elementwise. This lets both a singleton
// (each polynomial step) and an all-elementwise pattern group (e.g. mul+clamp)
// join. A group with any non-elementwise op is rejected, so its size()==1
// invariant is kept.
bool isDissolvableElementwise(
    Operation *op, const llvm::DenseMap<int64_t, bool> &groupAllCoalescable
) {
    if (!isCoalescableElementwise(op))
        return false;
    if (!isMarkedFuseGroup(op))
        return true; // ungrouped
    ArrayAttr arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP);
    if (!arr)
        return true;
    for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>())
        if (!groupAllCoalescable.lookup(fuseGroupAttr.getInt()))
            return false;
    return true;
}

// Group a maximal connected chain of pure-elementwise generics (the erf/tanh/mul
// polynomial chain) under one fuse group, so LinalgSlicing slices it as one
// scf.forall and the middle values stay in LRAM (like slicing-OFF). Without this
// each step gets its own forall and a full DRAM round-trip.
//
// The chain grows over def-use edges (isDissolvableElementwise). Reused values
// are fine inside the chain (erf's x^2 feeds several muls); only a use that leaves
// the chain matters. The chain stops at any non-elementwise op, and never touches
// a conv/matmul fuse group (that would break its size()==1 invariant).
//
// The chain's leaf (its only op used outside the chain) is the principal. We give
// every member the leaf's id, so the leaf becomes the fuse-group principal and the
// other ops are skipped by ElementwisePattern. Each member's old fuse group is
// stripped first, so nothing outside the chain changes.
void coalesceElementwiseChains(FunctionOpInterface funcOp) {
    MLIRContext *context = funcOp.getContext();
    // Precompute per-group coalescability once, so the check below is O(1).
    llvm::DenseMap<int64_t, bool> groupAllCoalescable = computeGroupCoalescableMap(funcOp);

    // Collect all candidate seeds in deterministic order.
    SmallVector<linalg::GenericOp> seeds;
    funcOp->walk([&](linalg::GenericOp g) {
        if (isDissolvableElementwise(g, groupAllCoalescable))
            seeds.push_back(g);
    });

    llvm::DenseSet<Operation *> assigned;
    for (linalg::GenericOp seed : seeds) {
        if (assigned.contains(seed.getOperation()))
            continue;

        // BFS the maximal connected coalescable region around `seed`.
        llvm::SetVector<Operation *> region;
        SmallVector<Operation *> work{seed.getOperation()};
        region.insert(seed.getOperation());
        while (!work.empty()) {
            Operation *cur = work.pop_back_val();

            // Walk to producers (defining ops of operands).
            for (Value operand : cur->getOperands()) {
                Operation *def = operand.getDefiningOp();
                if (!def || region.contains(def))
                    continue;
                if (!isDissolvableElementwise(def, groupAllCoalescable))
                    continue;
                region.insert(def);
                work.push_back(def);
            }
            // Walk to consumers (users of results).
            for (Value result : cur->getResults()) {
                for (Operation *user : result.getUsers()) {
                    if (region.contains(user))
                        continue;
                    if (!isDissolvableElementwise(user, groupAllCoalescable))
                        continue;
                    region.insert(user);
                    work.push_back(user);
                }
            }
        }

        // A single op is no different from the standalone path; skip.
        if (region.size() < 2)
            continue;

        // Find the chain's leaf: the one op whose result is used outside the chain.
        // A well-formed chain has exactly one. If there are several (the chain forks
        // to many outside users) we bail out, to be safe.
        Operation *leaf = nullptr;
        bool multipleLeaves = false;
        for (Operation *op : region) {
            bool escapes = false;
            for (Value result : op->getResults()) {
                for (Operation *user : result.getUsers()) {
                    if (!region.contains(user)) {
                        escapes = true;
                        break;
                    }
                }
                if (escapes)
                    break;
            }
            if (escapes) {
                if (leaf) {
                    multipleLeaves = true;
                    break;
                }
                leaf = op;
            }
        }
        if (multipleLeaves || !leaf)
            continue;

        // Use the leaf's own per-op UID as the shared group id so the leaf is the
        // principal (isFuseGroupPrincipalOp(leaf) == leafFuseGroupAttr).
        auto leafFuseGroupAttr = leaf->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
        if (!leafFuseGroupAttr)
            continue;

        // Members may still carry an old fuse group (a singleton id, or a pattern
        // group like mul+clamp). Strip each one first, then set the single shared
        // leaf id. Those old groups live only inside this chain, so nothing outside
        // is affected, and peelAndSlice's size()==1 invariant is kept.
        for (Operation *op : region) {
            if (auto arr = op->getAttrOfType<ArrayAttr>(TORQ_FUSE_GROUP)) {
                SmallVector<int64_t> fuseGroups;
                for (IntegerAttr fuseGroupAttr : arr.getAsRange<IntegerAttr>())
                    fuseGroups.push_back(fuseGroupAttr.getInt());
                for (int64_t fuseGroup : fuseGroups)
                    removeFuseGroupMarkingBackwards(op, fuseGroup);
            }
        }
        ArrayAttr groupAttr = ArrayAttr::get(context, {leafFuseGroupAttr});
        for (Operation *op : region) {
            op->setAttr(TORQ_FUSE_GROUP, groupAttr);
            assigned.insert(op);
        }
    }
}

class CoalesceElementwiseChainsPass
    : public impl::CoalesceElementwiseChainsBase<CoalesceElementwiseChainsPass> {
  public:
    void runOnOperation() override {
        // Same guard as LinalgSlicing, so behavior is identical: coalesce only
        // ran when slicing ran (sliceCount >= 2).
        if (TorqHw::get().getSliceCount() < 2)
            return;
        coalesceElementwiseChains(getOperation());
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCoalesceElementwiseChainsPass() {
    return std::make_unique<CoalesceElementwiseChainsPass>();
}

} // namespace mlir::syna::torq
