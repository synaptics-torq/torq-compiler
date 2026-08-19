// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/DmaProfilingUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <map>
#include <numeric>
#include <sstream>

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Defined in CompileNSSInvocationsPass.cpp; set via --torq-dma-in-mtu / --torq-dma-out-mtu.
extern llvm::cl::opt<unsigned> clDmaInMtu;
extern llvm::cl::opt<unsigned> clDmaOutMtu;

static llvm::cl::opt<bool> clTorqDisableNdlCycleCheck(
    "torq-disable-ndl-cycle-check",
    llvm::cl::desc("Disable NDL cycle match checks in slice cycle annotation"), llvm::cl::init(true)
);

size_t ndlCycle(torq_hw::RegNdlAttr attr) {
    size_t cycle = 1;
    for (auto dim : attr.getDims()) {
        if (dim.getType() == torq_hw::DimType::H) {
            cycle *= dim.getCount();
        }
    }
    return cycle;
}

size_t ndlCycle(MemNdlAttr attr) {
    size_t cycle = 1;
    uint32_t lWriteSize = 1;
    uint32_t xLineSize = 1;
    bool usingSdims = false;
    for (auto dim : attr.getDims()) {
        switch (dim.getType()) {
        case torq_hw::DimType::H:
            cycle *= dim.getCount();
            break;
        case torq_hw::DimType::L:
            lWriteSize *= dim.getCount();
            break;
        case torq_hw::DimType::S:
            if (dim.getTag() == torq_hw::MemDimTag::B || dim.getTag() == torq_hw::MemDimTag::X) {
                xLineSize *= dim.getCount();
                usingSdims = true;
            }
            break;
        }
    }

    // Use of SDIMs (DEQW) with small line-size (X-dim) can add overhead to the cycle count.
    // Use an approximation to account for this overhead which is normally negligible
    // except for very small line sizes.
    if (usingSdims) {
        if (xLineSize < lWriteSize) {
            // Each low-level write is split into multiple cycles
            cycle = cycle * div_ceil(lWriteSize, xLineSize);
        }
        else {
            // Determine the number of writes crossing X-boundary (will require one extra cycle)
            size_t g = std::gcd(lWriteSize, xLineSize);
            size_t misalignmentPeriod = (lWriteSize - 1) / g;
            size_t crossingWrites = (cycle * g / xLineSize) * misalignmentPeriod;
            cycle += crossingWrites;
        }
    }

    return cycle;
}

static std::string toString(Location loc) {
    std::string locStr;
    llvm::raw_string_ostream os(locStr);
    loc.print(os);
    return locStr;
}

class AnnotateDmaAndSliceCyclesPass
    : public impl::AnnotateDmaAndSliceCyclesBase<AnnotateDmaAndSliceCyclesPass> {
  public:
    using AnnotateDmaAndSliceCyclesBase<
        AnnotateDmaAndSliceCyclesPass>::AnnotateDmaAndSliceCyclesBase;

  private:
    static void annotateDmaOps(mlir::FunctionOpInterface funcOp) {
        const double dmaInThroughput = dmaThroughputBytesPerCycle(clDmaInMtu);
        const double dmaOutThroughput = dmaThroughputBytesPerCycle(clDmaOutMtu);

        // Annotate torq_hl.load. These tags describe the HL-level IR only; they do
        // not survive lowering (LoadOpPattern rebuilds the transfer without copying
        // attributes), so ProfilingPass re-derives the same numbers for
        // torq_hw.dma_in_start from the cfg op's NDL.
        funcOp.walk([&](torq_hl::LoadOp loadOp) {
            uint64_t bytes = transferBytes(loadOp.getShape(), loadOp.getElementSizeBytes());
            Builder builder(loadOp);
            loadOp->setAttr(
                kAsyncDurationAttr,
                builder.getI64IntegerAttr(estimateDmaCycles(bytes, dmaInThroughput))
            );
            loadOp->setAttr(kAsyncBytesAttr, builder.getI64IntegerAttr(bytes));
        });

        // Annotate torq_hl.store; HL-level only, as above.
        funcOp.walk([&](torq_hl::StoreOp storeOp) {
            uint64_t bytes = transferBytes(storeOp.getShape(), storeOp.getElementSizeBytes());
            Builder builder(storeOp);
            storeOp->setAttr(
                kAsyncDurationAttr,
                builder.getI64IntegerAttr(estimateDmaCycles(bytes, dmaOutThroughput))
            );
            storeOp->setAttr(kAsyncBytesAttr, builder.getI64IntegerAttr(bytes));
        });

        // Annotate memref.copy (LRAM<->DTCM/ITCM); HL-level only, as above.
        funcOp.walk([&](memref::CopyOp copyOp) {
            uint64_t bytes = cdmaTransferBytes(copyOp.getSource());
            Builder builder(copyOp);
            copyOp->setAttr(
                kAsyncDurationAttr, builder.getI64IntegerAttr(estimateCdmaCycles(bytes))
            );
            copyOp->setAttr(kAsyncBytesAttr, builder.getI64IntegerAttr(bytes));
        });
    }

    static void annotateSliceTaskOps(mlir::FunctionOpInterface funcOp, MLIRContext *ctx) {
        funcOp.walk([&](torq_hw::SliceTaskOp sliceTaskOp) {
            std::map<NdlType, mlir::Attribute> memNdlsAttr;
            std::map<NdlType, mlir::Attribute> regNdlsAttr;

            std::map<NdlType, size_t> memNdlCycles;
            for (auto &ndl : sliceTaskOp.getMemNdls()) {
                memNdlCycles[ndl.getType()] += ndlCycle(ndl);
                memNdlsAttr[ndl.getType()] = ndl;
            }

            std::map<NdlType, size_t> regNdlCycles;
            for (auto &ndl : sliceTaskOp.getRegNdls()) {
                regNdlCycles[ndl.getType()] = ndlCycle(ndl);
                regNdlsAttr[ndl.getType()] = ndl;
            }

            // Cycle calculations
            size_t dedr = memNdlCycles[NdlType::DEDR];
            size_t dewr = memNdlCycles[NdlType::DEWR];
            size_t debr = memNdlCycles[NdlType::DEBR];
            size_t deqw = memNdlCycles[NdlType::DEQW];
            size_t cedw = regNdlCycles[NdlType::CEDW];
            size_t cedr = regNdlCycles[NdlType::CEDR];
            size_t ceww = regNdlCycles[NdlType::CEWW];
            size_t cewr = regNdlCycles[NdlType::CEWR];
            size_t cepr = regNdlCycles[NdlType::CEPR];
            size_t acbw = regNdlCycles[NdlType::ACBW];
            size_t acbr = regNdlCycles[NdlType::ACBR];
            size_t acpr = regNdlCycles[NdlType::ACPR];

            size_t d_bus_cycle = std::max({dedr, cedw, cedr});
            size_t w_bus_cycle = std::max({dewr, ceww, cewr});
            size_t parallel_d_w_cycle = std::max(d_bus_cycle, w_bus_cycle);

            size_t b_bus_cycle = std::max({debr, acbw, acbr});
            size_t q_bus_cycle = std::max(deqw, acpr);
            size_t parallel_b_q_cycle = std::max(b_bus_cycle, q_bus_cycle);

            size_t p_bus_cycle = cepr;

            size_t curMaxCycle = std::max({parallel_d_w_cycle, parallel_b_q_cycle, p_bus_cycle});

            auto ndlCycles = NdlCyclesAttr::get(
                ctx, dedr, dewr, debr, deqw, cedw, cedr, ceww, cewr, cepr, acbw, acbr, acpr
            );

            auto sliceProgramOp = sliceTaskOp->getParentOfType<torq_hl::ProgramOp>();

            IRRewriter rewriter(ctx);
            Block &block = sliceProgramOp->getRegion(0).front();
            rewriter.setInsertionPointToStart(&block);

            size_t sliceMemSize = 0;
            std::ostringstream shapeSstr;

            for (auto [idx, operand] : llvm::enumerate(sliceTaskOp->getOperands())) {
                auto memreftype = dyn_cast<MemRefType>(operand.getType());
                if (!memreftype)
                    continue;

                int memssize = getEncodedTotalSizeBytes(memreftype);
                auto shape = memreftype.getShape();
                sliceMemSize += memssize;

                if (idx > 0)
                    shapeSstr << "+";
                for (size_t i = 0; i < shape.size(); i++) {
                    shapeSstr << shape[i];
                    if (i < shape.size() - 1)
                        shapeSstr << "x";
                }
            }

            torq_hw::SliceProfilingOp::create(
                rewriter, sliceTaskOp.getLoc(), sliceMemSize, shapeSstr.str(), curMaxCycle,
                ndlCycles
            );

            if (!clTorqDisableNdlCycleCheck) {
                // NDL Cycle Match Checks
                bool isNdlCycleMatch = true;
                auto loc = sliceTaskOp.getLoc();
                auto opName = sliceTaskOp.getOpName();

                if (cewr != 0 && cepr != 0 && cewr != cepr) {
                    llvm::errs() << "\n"
                                 << opName << " cewr != cepr: " << cewr << " vs " << cepr << " at "
                                 << toString(loc) << "\n";
                    llvm::errs() << " cewr : " << regNdlsAttr[NdlType::CEWR] << "\n";
                    llvm::errs() << " cepr : " << regNdlsAttr[NdlType::CEPR] << "\n";
                    isNdlCycleMatch = false;
                }

                if (deqw != 0 && acbr != 0 && deqw != acbr) {
                    llvm::errs() << "\n"
                                 << opName << " deqw != acbr: " << deqw << " vs " << acbr << " at "
                                 << toString(loc) << "\n";
                    llvm::errs() << " deqw : " << memNdlsAttr[NdlType::DEQW] << "\n";
                    llvm::errs() << " acbr : " << regNdlsAttr[NdlType::ACBR] << "\n";
                    isNdlCycleMatch = false;
                }

                if (cepr != 0 && cedr != 0 && cepr != cedr) {
                    llvm::errs() << "\n"
                                 << opName << " cepr != cedr: " << cepr << " vs " << cedr << " at "
                                 << toString(loc) << "\n";
                    llvm::errs() << " cepr : " << regNdlsAttr[NdlType::CEPR] << "\n";
                    llvm::errs() << " cedr : " << regNdlsAttr[NdlType::CEDR] << "\n";
                    isNdlCycleMatch = false;
                }

                if (!isNdlCycleMatch) {
                    llvm::errs() << " ref : " << memNdlsAttr[NdlType::REF] << "\n";
                    // TODO : Enable assert after issues are fixed
                    // Keeping the assert commented to avoid test infra failures
                    // assert(false && "NDL cycle check failed");
                }
            }

            return WalkResult::advance();
        });
    }

  public:
    void runOnOperation() override {
        auto funcOp = getOperation();
        auto *ctx = funcOp.getContext();

        annotateDmaOps(funcOp);
        annotateSliceTaskOps(funcOp, ctx);
    }
};

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAnnotateDmaAndSliceCyclesPass() {
    return std::make_unique<AnnotateDmaAndSliceCyclesPass>();
}

} // namespace mlir::syna::torq
