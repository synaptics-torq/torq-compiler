// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Codegen/Pool.h"
#include "torq/Codegen/VirtualMemory.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-assign-addresses"

static llvm::cl::opt<int> clXramStartAddress(
    "torq-xram-start-address", llvm::cl::desc("XRAM Start Address"),
    llvm::cl::init(0x100000) // 1MB
);

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

namespace {

class AssignAddressesPass : public AssignAddressesBase<AssignAddressesPass> {
  public:
    AssignAddressesPass() = default;
    AssignAddressesPass(const AssignAddressesPass &pass) {}

    void runOnOperation() override;
};

// check if we peak memory usage exceeds the pool size before ever trying to find addresses
// (we may later fail allocation even if peak usage is below the pool size)
static LogicalResult
checkPeakMemoryUsage(Operation *parentOp, torq_hl::MemorySpace memSpace, int memSize) {

    int currentMemUsage = 0;
    int peakMemUsage = 0;
    int overflowMemUsage = 0;
    Operation *overflowAlloc{};
    Operation *peakAlloc{};
    DenseSet<Value> overflowAllocatedValues;
    DenseSet<Value> peakAllocatedValues;

    DenseSet<Value> allocatedValues;
    for (auto &op : parentOp->getRegion(0).getOps()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {

            if (getEncodingMemorySpace(allocOp.getType()) != memSpace) {
                continue;
            }
            allocatedValues.insert(allocOp);
            auto reserveSize = getEncodedTotalSizeBytes(allocOp.getType());
            currentMemUsage += reserveSize;
            if (currentMemUsage > memSize && !overflowAlloc) {
                overflowMemUsage = currentMemUsage;
                overflowAlloc = &op;
                overflowAllocatedValues = allocatedValues;
            }
            if (currentMemUsage > peakMemUsage) {
                peakMemUsage = currentMemUsage;
                peakAlloc = &op;
                peakAllocatedValues = allocatedValues;
            }
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
            if (getEncodingMemorySpace(deallocOp.getMemref().getType()) != memSpace) {
                continue;
            }
            allocatedValues.erase(deallocOp.getMemref());
            auto memRefType = mlir::cast<MemRefType>(deallocOp.getMemref().getType());
            auto releaseSize = getEncodedTotalSizeBytes(memRefType);
            currentMemUsage -= releaseSize;
        }
    }

    if (peakMemUsage <= memSize) {
        return success();
    }

    // Memory overflow
    llvm::errs() << "First overflow at operation : ";
    overflowAlloc->dump();
    llvm::errs() << "Memory allocated at first overflow:\n";
    for (auto value : overflowAllocatedValues) {
        auto type = cast<MemRefType>(value.getType());
        llvm::dbgs() << "   size: " << getEncodedTotalSizeBytes(type) << " value: ";
        value.dump();
    }
    llvm::errs() << "\nPeak operation at operation : ";
    peakAlloc->dump();
    llvm::errs() << "\nMemory allocated at peak:\n";
    for (auto value : peakAllocatedValues) {
        auto type = cast<MemRefType>(value.getType());
        llvm::dbgs() << "   size: " << getEncodedTotalSizeBytes(type) << " value: ";
        value.dump();
    }

    llvm::errs() << "Peak memory usage of " << peakMemUsage << " bytes at: ";
    peakAlloc->dump();
    return overflowAlloc->emitError() << "Memory usage: " << overflowMemUsage << " exceeds "
                                      << stringifyEnum(memSpace) << " size " << memSize << " bytes";
}

static LogicalResult setDerivedMemrefAddress(Operation *op) {

    int64_t baseAddress;

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
        auto maybeBaseAddress = getAddress(subviewOp.getSource());

        if (!maybeBaseAddress) {
            return subviewOp.emitError("source does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {
        auto maybeBaseAddress = getAddress(expandShapeOp.getSrc());

        if (!maybeBaseAddress) {
            return expandShapeOp.emitError("source does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(op)) {

        for (int i = 0; i < reinterpretCast.getResultRank(); i++) {
            if (reinterpretCast.isDynamicOffset(i) || reinterpretCast.isDynamicSize(i)) {
                return subviewOp.emitError("dynamic offsets or sizes not supported");
            }

            if (reinterpretCast.getStaticOffset(i) != 0) {
                return subviewOp.emitError("non-zero static offsets not supported");
            }
        }

        auto maybeBaseAddress = getAddress(reinterpretCast.getSource());

        if (!maybeBaseAddress) {
            return reinterpretCast.emitError("source does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else if (auto memorySpaceCast = dyn_cast<memref::MemorySpaceCastOp>(op)) {
        auto srcSpace = getEncodingMemorySpace(memorySpaceCast.getSource().getType());
        auto dstSpace = getEncodingMemorySpace(memorySpaceCast.getDest().getType());

        // casts with different spaces are not supported for the moment
        if (srcSpace != dstSpace) {
            return memorySpaceCast.emitError(
                "memory space cast with different source and destination spaces not supported"
            );
        }

        auto maybeBaseAddress = getAddress(memorySpaceCast.getSource());

        if (!maybeBaseAddress) {
            return memorySpaceCast.emitError("source does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(op)) {

        auto maybeBaseAddress = getAddress(collapseShapeOp.getSrc());

        if (!maybeBaseAddress) {
            return memorySpaceCast.emitError("src does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(op)) {

        auto maybeBaseAddress = getAddress(expandShapeOp.getSrc());

        if (!maybeBaseAddress) {
            return memorySpaceCast.emitError("src does not have an address assigned");
        }

        baseAddress = *maybeBaseAddress;
    }
    else {
        return op->emitError() << "not a derived memref operation";
    }

    // set the base address on the operation since the result memref type already contains the
    // offset
    if (failed(setAddress(op, baseAddress))) {
        return op->emitError() << "failed to set address";
    }

    return success();
}

static LogicalResult
allocateAddresses(Operation *parentOp, Pool &pool, torq_hl::MemorySpace memSpace) {

    auto status = checkPeakMemoryUsage(parentOp, memSpace, pool.usableSize());

    if (failed(status)) {
        return status;
    }

    for (auto &op : parentOp->getRegion(0).getOps()) {
        if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {

            if (getEncodingMemorySpace(allocOp.getType()) != memSpace) {
                continue;
            }

            FailureOr<int64_t> maybeAddr = pool.allocate(op.getResult(0));

            if (failed(maybeAddr)) {
                return op.emitError(
                    "Failed to allocate " + stringifyMemorySpace(memSpace) + " buffer of size " +
                    std::to_string(getEncodedTotalSizeBytes(allocOp.getType()))
                );
            }

            if (failed(setAddress(&op, *maybeAddr))) {
                return op.emitError("Failed to set " + stringifyMemorySpace(memSpace) + " address");
            }
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
            if (getEncodingMemorySpace(deallocOp.getMemref().getType()) != memSpace) {
                continue;
            }
            pool.free(deallocOp.getMemref());
        }
        else if (isDerivedMemRefOperation(&op)) {

            auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

            if (getEncodingMemorySpace(type) != memSpace) {
                continue;
            }

            if (failed(setDerivedMemrefAddress(&op))) {
                return failure();
            }
        }
    }

    return success();
}

static LogicalResult allocateLramAddresses(FunctionOpInterface funcOp) {

    // we reserve space for two NSS programs at the begininng of LRAM
    // (one block for the currently executing program and one for the
    // next program to be executed that will be loaded by the current one)
    int nssProgramPoolSize = HwInfo::nss_max_program_size * 2;

    // FIXME: Work-around to get bias with 8 bytes alignment, we need a better solution
    Pool pool(TorqHw::get().getLramSize(), nssProgramPoolSize, 8);

    // rewrite all the memrefs in the function to ensure we can
    // allocated addresses for all of them
    auto status = convertVirtualToPhysicalMemRefs(funcOp, pool, torq_hl::MemorySpace::Lram);

    if (failed(status)) {
        return status;
    }

    // clear the pool in case there are some buffers that are never deallocated
    pool.clear();

    // FIXME: this could be done in the convertVirtualToPhysicalMemRefs call
    status = allocateAddresses(funcOp, pool, syna::torq_hl::MemorySpace::Lram);

    if (failed(status)) {
        return status;
    }

    return success();
}

static LogicalResult allocateXramAddresses(FunctionOpInterface funcOp) {

    int64_t startAddress = clXramStartAddress;

    startAddress = align_ceil(startAddress, HwInfo::xram_page_size);

    // for the moment we just assign increasing addresses to operations (because the pool
    // will try to allocate large buffers at the end)

    for (auto &op : funcOp.getFunctionBody().getOps()) {

        if (auto createInvocationOp = dyn_cast<syna::torq_hl::CreateInvocationOp>(op)) {

            // for invocations we need to assigna an xram address for each code section they
            // produce
            SmallVector<int64_t> addresses;

            for (auto result : createInvocationOp.getCodeSections()) {

                auto type = mlir::cast<MemRefType>(result.getType());

                addresses.push_back(startAddress);

                startAddress = align_ceil(startAddress + getEncodedTotalSizeBytes(type), 4);

                if (startAddress > HwInfo::xram_size) {
                    return createInvocationOp.emitError(
                        "XRAM size exceeded while assigning addresses"
                    );
                }
            }

            createInvocationOp.setXramCodeAddresses(addresses);
        }
        else if (isa<memref::AllocOp, syna::torq_hl::MapBindingOp, syna::torq_hl::ConstOp>(op)) {

            assert(op.getNumResults() == 1 && "Expected a single result for alloc-like operations");

            auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

            if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
                continue;
            }

            setXramAddress(&op, startAddress);

            startAddress = align_ceil(startAddress + getEncodedTotalSizeBytes(type), 4);

            if (startAddress > HwInfo::xram_size) {
                return op.emitError("XRAM size exceeded while assigning addresses");
            }
        }
        else if (isDerivedMemRefOperation(&op)) {

            auto type = mlir::cast<MemRefType>(op.getResult(0).getType());

            if (getEncodingMemorySpace(type) != torq_hl::MemorySpace::Xram) {
                continue;
            }

            if (failed(setDerivedMemrefAddress(&op))) {
                return failure();
            }
        }
    }

    // mark the area used by the operations as reserved for later passes
    reserveXramArea(funcOp, startAddress);

    return success();
}

void AssignAddressesPass::runOnOperation() {
    auto funcOp = getOperation();

    if (failed(allocateLramAddresses(funcOp))) {
        llvm::errs() << "Failed to allocate LRAM addresses\n";
        return signalPassFailure();
    }

    Pool dtcmPool(HwInfo::dtcm_size - HwInfo::css_stack_size, 0, 4);

    if (failed(allocateAddresses(funcOp, dtcmPool, syna::torq_hl::MemorySpace::Dtcm))) {
        llvm::errs() << "Failed to allocate DTCM addresses\n";
        return signalPassFailure();
    }

    // FIXME: here we all allocations to be at the begining of the pool
    // so that we end up using only address 0 for the ITCM which is the
    // only addrees where the compiled dispatches can currently run
    Pool itcmPool(HwInfo::itcm_size, 0, 4, 8 * 1024);

    if (failed(allocateAddresses(funcOp, itcmPool, syna::torq_hl::MemorySpace::Itcm))) {
        llvm::errs() << "Failed to allocate ITCM addresses\n";
        return signalPassFailure();
    };

    if (failed(allocateXramAddresses(funcOp))) {
        llvm::errs() << "Failed to allocate XRAM addresses\n";
        return signalPassFailure();
    };
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createAssignAddressesPass() {
    return std::make_unique<AssignAddressesPass>();
}

} // namespace mlir::syna::torq
