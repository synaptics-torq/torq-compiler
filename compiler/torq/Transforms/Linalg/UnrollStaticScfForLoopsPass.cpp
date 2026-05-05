// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

namespace mlir::syna::torq {

// Fully unrolls scf.for loops with statically known bounds that appear at the
// linalg-on-tensors level. We need to run this before dispatch creation,
// because IREE otherwise turns a loop whose body contains a single dispatch
// into a host-side loop that re-dispatches the same executable with different
// workload ordinal values. The torq backend does not support dispatches that
// are parameterised on a runtime workload ordinal (it has no way to forward a
// dynamic slice offset into a DMA read address), so unrolling here converts
// each iteration into an independent dispatch with all offsets baked in as
// constants.
//
// A typical example is the per-timestep loop produced by the torch-mlir
// lowering of onnx.GRU (`scf.for %t = 0 to T step 1 iter_args(...)`), where T
// is static.
class UnrollStaticScfForLoopsPass
    : public impl::UnrollStaticScfForLoopsBase<UnrollStaticScfForLoopsPass> {
  public:
    using UnrollStaticScfForLoopsBase::UnrollStaticScfForLoopsBase;

    void runOnOperation() override {
        auto funcOp = getOperation();

        SmallVector<scf::ForOp, 4> loops;
        funcOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
        for (scf::ForOp forOp : loops) {
            std::optional<APInt> tripCount = forOp.getStaticTripCount();
            if (!tripCount || tripCount->isZero())
                continue;
            (void)loopUnrollByFactor(forOp, tripCount->getSExtValue());
        }
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createUnrollStaticScfForLoopsPass() {
    return std::make_unique<UnrollStaticScfForLoopsPass>();
}

} // namespace mlir::syna::torq
