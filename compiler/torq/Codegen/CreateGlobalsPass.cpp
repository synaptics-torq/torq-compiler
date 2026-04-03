// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"

#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "torq-create-globals"

namespace mlir::syna::torq {

namespace {

class CreateGlobalsPass : public impl::CreateGlobalsBase<CreateGlobalsPass> {
  public:
    using CreateGlobalsBase<CreateGlobalsPass>::CreateGlobalsBase;

    void runOnOperation() override;
};

static void replaceAllocations(IRRewriter &rewriter, SmallVector<memref::AllocOp> &allocOps) {

    // Replace each allocation with a memref.get_global and create a corresponding
    // memref.global in the module we are processing
    unsigned globalCounter = 0;

    for (memref::AllocOp allocOp : allocOps) {

        // Setup an insertion guard to ensure that the rewriter insertion point is reset
        // after we are done inserting the global and replacing the alloc with get_global.
        // This is important because we are inserting the global at the start of the module,
        // but we want to replace the alloc with get_global at the same location as the original
        // alloc.
        IRRewriter::InsertionGuard ig(rewriter);

        auto memrefType = allocOp.getMemref().getType();

        // Create a unique name for the global
        std::string globalName =
            (llvm::Twine("__intermediate_") + llvm::Twine(globalCounter++)).str();

        // Create memref.global operation
        auto globalOp = rewriter.create<memref::GlobalOp>(
            allocOp.getLoc(),
            /*sym_name=*/globalName,
            /*sym_visibility=*/rewriter.getStringAttr("private"),
            /*type=*/memrefType,
            /*initial_value=*/rewriter.getUnitAttr(),
            /*constant=*/false,
            /*alignment=*/nullptr
        );

        rewriter.modifyOpInPlace(globalOp, [&]() { copyAddressAttributes(allocOp, globalOp); });

        // Replace memref.alloc with memref.get_global
        rewriter.setInsertionPoint(allocOp);
        rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, memrefType, globalName);
    }
}

// Creates two globals in LRAM used to store NSS program being executed and next program
void addNssProgramSlots(IRRewriter &rewriter, Location loc) {

    auto programSectionType = MemRefType::get({HwInfo::nss_max_program_size}, rewriter.getI8Type());
    auto lramProgramSectionType =
        createMemRefTypeWithMemorySpace(programSectionType, torq_hl::MemorySpace::Lram);

    // allocate two memrefs that will be used to load the current and next program
    // the address has been reserved during the address allocation pass so we can just
    // set them here
    for (int i = 0; i < 2; i++) {

        std::string globalName = (llvm::Twine("__program_slot") + llvm::Twine(i)).str();

        auto globalOp = rewriter.create<memref::GlobalOp>(
            loc,
            /*sym_name=*/globalName,
            /*sym_visibility=*/rewriter.getStringAttr("private"),
            /*type=*/lramProgramSectionType,
            /*initial_value=*/rewriter.getUnitAttr(),
            /*constant=*/false,
            /*alignment=*/nullptr
        );

        setLramAddress(globalOp, i * HwInfo::nss_max_program_size);
    }
}

void CreateGlobalsPass::runOnOperation() {
    auto moduleOp = getOperation();

    // Collect all memref.alloc and memref.dealloc operations in dispatch
    // function defined by the module

    SmallVector<memref::AllocOp> allocOps;
    SmallVector<memref::DeallocOp> deallocOps;

    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
        for (auto &op : funcOp.getFunctionBody().getOps()) {
            if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
                allocOps.push_back(allocOp);
            }
            else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
                deallocOps.push_back(deallocOp);
            }
        }
    }

    IRRewriter rewriter(moduleOp.getContext());
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    // Insert two LRAM globals where we load and execute NSS programs in a ping pong fashion
    // these will be used when calling NSS programs in the NSS outline program pass
    addNssProgramSlots(rewriter, moduleOp.getLoc());

    // Replace memref.alloc with memref.get_global and create corresponding memref.global
    replaceAllocations(rewriter, allocOps);

    // Erase all memref.dealloc operations since the memory will now be globally allocated
    for (memref::DeallocOp deallocOp : deallocOps) {
        rewriter.eraseOp(deallocOp);
    }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCreateGlobalsPass() {
    return std::make_unique<CreateGlobalsPass>();
}

} // namespace mlir::syna::torq
