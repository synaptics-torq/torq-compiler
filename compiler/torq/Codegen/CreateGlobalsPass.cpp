// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
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
        auto globalOp = memref::GlobalOp::create(
            rewriter, allocOp.getLoc(),
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

static void
replaceConstOpsWithGlobals(IRRewriter &rewriter, SmallVector<torq_hl::ConstOp> &constOps) {

    // Find all torq_hl.const operations in the module and replace them with memref.get_global
    // and create corresponding memref.global for each of them
    unsigned globalCounter = 0;

    for (auto constOp : constOps) {

        // Setup an insertion guard to ensure that the rewriter insertion point is reset
        // after we are done inserting the global and replacing the const with get_global.
        IRRewriter::InsertionGuard ig(rewriter);

        auto value = constOp.getValue();

        auto memrefType = cast<MemRefType>(constOp.getOutput().getType());

        // Create a unique name for the global
        std::string globalName =
            (llvm::Twine("__const_global_") + llvm::Twine(globalCounter++)).str();

        // Create memref.global operation
        auto globalOp = memref::GlobalOp::create(
            rewriter, constOp.getLoc(),
            /*sym_name=*/globalName,
            /*sym_visibility=*/rewriter.getStringAttr("private"),
            /*type=*/memrefType,
            /*initial_value=*/value,
            /*constant=*/true,
            /*alignment=*/nullptr
        );

        rewriter.modifyOpInPlace(globalOp, [&]() { copyAddressAttributes(constOp, globalOp); });

        rewriter.setInsertionPoint(constOp);
        rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(constOp, memrefType, globalName);
    }
}

static DenseElementsAttr convertCodeDataToElementsAttr(
    IRRewriter &rewriter, DenseI8ArrayAttr codeData, MemRefType memrefType
) {
    SmallVector<Attribute> elements;
    elements.reserve(codeData.size());

    for (int8_t byte : codeData.getRawData()) {
        elements.push_back(rewriter.getI8IntegerAttr(byte));
    }

    auto tensorType = RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
    return DenseElementsAttr::get(tensorType, elements);
}

static void replaceDescriptorOpsWithGlobals(
    IRRewriter &rewriter, SmallVector<torq_hl::DescriptorOp> &descriptorOps
) {

    // Find all torq_hl.descriptor operations in the module and replace them with memref.get_global
    // and create corresponding memref.global for each of them
    unsigned globalCounter = 0;

    for (auto descriptorOp : descriptorOps) {

        // Setup an insertion guard to ensure that the rewriter insertion point is reset
        // after we are done inserting the global and replacing the descriptor with get_global.
        IRRewriter::InsertionGuard ig(rewriter);

        auto codeDataRange = descriptorOp.getCodeData().getAsRange<DenseI8ArrayAttr>();
        auto xramCodeAddressesAttr = descriptorOp.getXramCodeAddresses();
        auto descriptorName = descriptorOp.getName();

        for (auto [sectionData, codeSection] :
             llvm::zip(codeDataRange, descriptorOp.getCodeSections())) {

            auto memrefType = cast<MemRefType>(codeSection.getType());
            auto initialValue = convertCodeDataToElementsAttr(rewriter, sectionData, memrefType);
            auto codeSectionOpResult = cast<OpResult>(codeSection);

            // Create a unique name for the global
            std::string globalName =
                (llvm::Twine("__const_descriptor_") + llvm::Twine(descriptorName) +
                 llvm::Twine("_") + llvm::Twine(globalCounter) + llvm::Twine("_") +
                 llvm::Twine(codeSectionOpResult.getResultNumber()))
                    .str();

            // Create memref.global operation
            auto globalOp = memref::GlobalOp::create(
                rewriter, descriptorOp.getLoc(),
                /*sym_name=*/globalName,
                /*sym_visibility=*/rewriter.getStringAttr("private"),
                /*type=*/memrefType,
                /*initial_value=*/initialValue,
                /*constant=*/true,
                /*alignment=*/nullptr
            );

            rewriter.modifyOpInPlace(globalOp, [&]() {
                // the first result is the invocation, so we need to subtract 1 from the result
                // number to get the correct index into the xramCodeAddressesAttr
                setXramAddress(
                    globalOp, xramCodeAddressesAttr[codeSectionOpResult.getResultNumber() - 1]
                );
            });

            rewriter.setInsertionPoint(descriptorOp);

            auto getGlobalOp = memref::GetGlobalOp::create(
                rewriter, descriptorOp.getLoc(), memrefType, globalName
            );

            rewriter.replaceAllUsesWith(codeSectionOpResult, getGlobalOp.getResult());
        }

        globalCounter++;
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

        auto globalOp = memref::GlobalOp::create(
            rewriter, loc,
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

    // Collect all memref.alloc, memref.dealloc and torq_hl.const operations in dispatch
    // function defined by the module

    SmallVector<memref::AllocOp> allocOps;
    SmallVector<memref::DeallocOp> deallocOps;
    SmallVector<torq_hl::ConstOp> constOps;
    SmallVector<torq_hl::DescriptorOp> descriptorOps;

    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
        for (auto &op : funcOp.getFunctionBody().getOps()) {
            if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
                allocOps.push_back(allocOp);
            }
            else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
                deallocOps.push_back(deallocOp);
            }
            else if (auto constOp = dyn_cast<torq_hl::ConstOp>(op)) {
                constOps.push_back(constOp);
            }
            else if (auto descriptorOp = dyn_cast<torq_hl::DescriptorOp>(op)) {
                descriptorOps.push_back(descriptorOp);
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

    // Replace all the torq_hl.const operations with memref.get_global and create corresponding
    // memref.global
    replaceConstOpsWithGlobals(rewriter, constOps);

    // Replace all the torq_hl.descriptor operations with memref.get_global and create corresponding
    // memref.global for the descriptor
    replaceDescriptorOpsWithGlobals(rewriter, descriptorOps);
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCreateGlobalsPass() {
    return std::make_unique<CreateGlobalsPass>();
}

} // namespace mlir::syna::torq
