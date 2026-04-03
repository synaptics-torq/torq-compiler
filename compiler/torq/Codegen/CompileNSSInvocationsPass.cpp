// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "torq/Codegen/CompileInvocationUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "torq/Serialization/DescGen.h"
#include "torq/Utils/TorqUtils.h"

#define DEBUG_TYPE "torq-compile-nss-invocations"

using namespace mlir::iree_compiler;

namespace mlir::syna::torq {

static llvm::cl::opt<bool> clDisableAsyncSliceWait(
    "torq-disable-async-slice-wait1", llvm::cl::desc("Disable Asynchronous Slice Wait"),
    llvm::cl::init(false)
);

llvm::cl::opt<unsigned> clDmaInMtu(
    "torq-dma-in-mtu",
    llvm::cl::desc("DMA In (XRAM read) burst length as MTU value: burst = (1 << mtu) beats."),
    llvm::cl::init(4)
);

llvm::cl::opt<unsigned> clDmaOutMtu(
    "torq-dma-out-mtu",
    llvm::cl::desc("DMA Out (XRAM write) burst length as MTU value: burst = (1 << mtu) beats."),
    llvm::cl::init(4)
);

namespace {

class CompileNSSInvocationsPass
    : public impl::CompileNSSInvocationsBase<CompileNSSInvocationsPass> {
  public:
    CompileNSSInvocationsPass() = default;
    CompileNSSInvocationsPass(const CompileNSSInvocationsPass &pass) {}

    void runOnOperation() override;
};

static FailureOr<uint32_t> getCssAddress(Value value) {

    auto type = dyn_cast<MemRefType>(value.getType());

    if (!type) {
        return failure();
    }

    auto memSpace = getEncodingMemorySpace(type);

    std::optional<int32_t> maybeProgramAddress;

    switch (memSpace) {
    case torq_hl::MemorySpace::Dtcm:
        maybeProgramAddress =
            getDtcmAddress(value, HwInfo::css_dtcm_base_address + getMemRefTypeOffsetBytes(type));
        break;
    case torq_hl::MemorySpace::Itcm:
        maybeProgramAddress =
            getItcmAddress(value, HwInfo::css_itcm_base_address + getMemRefTypeOffsetBytes(type));
        break;
    default:
        return failure();
    }

    if (!maybeProgramAddress) {
        return failure();
    }

    return maybeProgramAddress.value();
}

static void convert(const torq_hw::DmaNdlAttr attr, DmaNdl &ndl) {
    for (auto dim : attr.getDims())
        ndl.addDim(dim.getCount(), dim.getStride());
}

template <typename T>
bool convert(T &cfgOp, mlir::syna::torq::DmaNdl &readNdl, mlir::syna::torq::DmaNdl &writeNdl) {

    if (!cfgOp.getRead() || !cfgOp.getReadAddress() || !cfgOp.getReadNdl()) {
        cfgOp.emitError("Missing read value, address or NDL");
        return false;
    }

    convert(cfgOp.getReadNdl(), readNdl);
    readNdl.setBaseAddress(*cfgOp.getReadAddress());

    if (!cfgOp.getWrite() || !cfgOp.getWriteAddress() || !cfgOp.getWriteNdl()) {
        cfgOp.emitError("Missing write value, address or NDL");
        return false;
    }

    convert(cfgOp.getWriteNdl(), writeNdl);
    writeNdl.setBaseAddress(*cfgOp.getWriteAddress());

    return true;
}

static LogicalResult toNssTask(syna::torq_hw::NssTaskOp taskOp, NssTask &task) {
    DmaParams dmaParams{};
    CssParams cssParams{};
    CDmaParams cdmaParams{};
    SliceParams sparams[HwInfo::slice_count]{};

    for (auto &op : taskOp.getBody().front().getOperations()) {
        if (auto cfgOp = dyn_cast<syna::torq_hw::DmaInCfgOp>(op)) {

            if (!convert(cfgOp, task.dixr, task.dilw)) {
                return cfgOp->emitError("Failed to convert DmaInCfgOp");
            }
        }
        else if (isa<syna::torq_hw::DmaInStartOp>(op)) {
            dmaParams.xrStart = true;
        }
        else if (isa<syna::torq_hw::DmaInWaitOp>(op)) {
            dmaParams.xrWait = true;
        }
        else if (auto cfgOp = dyn_cast<syna::torq_hw::DmaOutCfgOp>(op)) {

            if (!convert(cfgOp, task.dolr, task.doxw)) {
                return cfgOp->emitError("Failed to convert DmaOutCfgOp");
            }
        }
        else if (isa<syna::torq_hw::DmaOutStartOp>(op)) {
            dmaParams.xwStart = true;
        }
        else if (isa<syna::torq_hw::DmaOutWaitOp>(op)) {
            dmaParams.xwWait = true;
        }
        else if (auto sStartOp = dyn_cast<syna::torq_hw::SliceStartOp>(op)) {

            auto sid = sStartOp.getId().getSExtValue();
            if (sid >= HwInfo::slice_count) {
                return sStartOp->emitError("SliceId >= slice count");
            }

            sparams[sid].start = true;

            if (clDisableAsyncSliceWait) {
                sparams[sid].wait = true;
            }

            auto maybeCfgAddr = sStartOp.getProgramAddress();

            if (!maybeCfgAddr) {
                return sStartOp->emitError("Missing program address");
            }

            sparams[sid].cfgAddr = maybeCfgAddr.value();
        }
        else if (auto sWaitOp = dyn_cast<syna::torq_hw::SliceWaitOp>(op)) {

            auto sid = sWaitOp.getId().getSExtValue();
            if (sid >= HwInfo::slice_count) {
                return sWaitOp->emitError("SliceId >= slice count");
            }
            if (!clDisableAsyncSliceWait) {
                sparams[sid].wait = true;
            }
        }
        else if (auto cssStartOp = dyn_cast<syna::torq_hw::CSSStartOp>(op)) {
            cssParams.start = true;

            auto maybeProgramAddress = cssStartOp.getProgramAddress();

            if (!maybeProgramAddress) {
                return cssStartOp->emitError("Missing CSS program address");
            }

            // FIXME: currently we support only CSS programs at address 0x0
            assert(maybeProgramAddress.value() == 0 && "CSS program address must be zero");

            cssParams.startAddr = maybeProgramAddress.value();

            auto maybeArgsAddressAddress = cssStartOp.getArgAddressesAddress();

            if (!maybeArgsAddressAddress) {
                return cssStartOp->emitError("Missing CSS args addresses address");
            }

            cssParams.mbx[0] = maybeArgsAddressAddress.value();
        }
        else if (auto cssWaitOp = dyn_cast<syna::torq_hw::CSSWaitOp>(op)) {
            cssParams.wait = true;
        }
        else if (auto cdmaStartOp = dyn_cast<syna::torq_hw::CDMAStartOp>(op)) {
            cdmaParams.start = true;

            if (!cdmaStartOp.getSrcAddress()) {
                return cdmaStartOp.emitError("Missing source address");
            }

            cdmaParams.srcAddr = cdmaStartOp.getSrcAddress().value();

            if (!cdmaStartOp.getDestAddress()) {
                return cdmaStartOp.emitError("Missing destination address");
            }

            cdmaParams.dstAddr = cdmaStartOp.getDestAddress().value();

            auto srcType = cdmaStartOp.getSrc().getType();
            cdmaParams.len = getEncodedDataSizeBytes(srcType);
        }
        else if (auto cdmaWaitOp = dyn_cast<syna::torq_hw::CDMAWaitOp>(op)) {
            cdmaParams.wait = true;
        }
        else {
            return op.emitError("Unsupported operation in NSS task");
        }
    }
    task.setCssParams(cssParams);
    task.setCdmaParams(cdmaParams);
    dmaParams.mtuXr = clDmaInMtu;
    dmaParams.mtuXw = clDmaOutMtu;
    task.setDmaParams(dmaParams);
    for (size_t i = 0; i < HwInfo::slice_count; ++i)
        task.setSliceParams(i, sparams[i]);

    return success();
}

static LogicalResult processNssTask(
    DescGen &_npu, torq_hw::NssTaskOp nssTaskOp, int lramAddress, int xramAddress,
    size_t &remainingBlockSize
) {

    NssTask task;
    if (failed(toNssTask(nssTaskOp, task))) {
        return failure();
    }

    LLVM_DEBUG({
        llvm::dbgs() << llvm::formatv("*** NSS task ***\n");
        llvm::dbgs() << task.toLogStr() << "\n";
    });

    int size = task.compile(_npu.get(), lramAddress, xramAddress);

    if (size < 0) {
        // Task with negative length is an error
        return failure();
    }

    remainingBlockSize -= size;

    if (remainingBlockSize < 0) {
        return nssTaskOp->emitOpError("NSS task overflows block size");
    }

    return success();
}

static LogicalResult compileInvocation(
    torq_hl::CreateInvocationOp createInvocationOp, SmallVector<int32_t> &blockSizes,
    SmallVector<int8_t> *code = nullptr
) {

    auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    if (!programOp) {
        return createInvocationOp->emitOpError("program is not a torq_hl::ProgramOp");
    }

    auto maybeStartProgramOp = findStartProgramOp(createInvocationOp);

    if (failed(maybeStartProgramOp)) {
        return failure();
    }

    if (!createInvocationOp.getXramCodeAddresses() ||
        !createInvocationOp.getExecutorCodeAddresses()) {
        return createInvocationOp->emitOpError("NSS program must have XRAM and LRAM code addresses"
        );
    }

    auto xramCodeAddresses = createInvocationOp.getXramCodeAddresses().value();
    auto lramCodeAddresses = createInvocationOp.getExecutorCodeAddresses().value();

    auto blockCount = programOp.getBody().getBlocks().size();

    if (blockCount == 0) {
        return programOp->emitOpError("NSS program must have at least one block");
    }

    if (xramCodeAddresses.size() != blockCount) {
        return createInvocationOp->emitOpError(
            "Expected as many XRAM code address for NSS program as blocks in the program"
        );
    }

    if (lramCodeAddresses.size() != 1) {
        return createInvocationOp->emitOpError("Expected exactly one LRAM code address");
    }

    // compute the initial addresses and sizes for memory that will host cfgs
    auto lramStartAddress = lramCodeAddresses[0];

    // find the code size for each block that was kept aside to serialize it (this is the max size
    // we can use)
    blockSizes.clear();
    for (auto codeSection : createInvocationOp.getCodeSections()) {
        auto codeType = cast<MemRefType>(codeSection.getType());
        blockSizes.push_back(getEncodedTotalSizeBytes(codeType));
    }

    DescGen _npu;

    std::string dump_path = "";

    if (!_npu.open(dump_path.c_str())) {
        return createInvocationOp->emitError("Failed to open DescGen for NSS compilation");
    }

    if (!_npu.nssBegin(lramStartAddress, xramCodeAddresses[0])) {
        return failure();
    }

    int64_t nextLramAddress = AddressConstants::APPEND;
    int64_t nextXramAddress = AddressConstants::APPEND;

    for (auto [idx, block] : llvm::enumerate(programOp.getBody().getBlocks())) {

        size_t remainingBlockSize = blockSizes[idx];

        for (auto &nestedOp : block.getOperations()) {

            if (isa<torq_hl::ReturnOp, torq_hl::NextOp, torq_hl::GetBlockOp, memref::AllocOp,
                    memref::DeallocOp, memref::GetGlobalOp>(nestedOp)) {
                continue;
            }

            if (isDerivedMemRefOperation(&nestedOp)) {
                continue;
            }

            auto nssTaskOp = dyn_cast<torq_hw::NssTaskOp>(nestedOp);

            if (!nssTaskOp) {
                return nestedOp.emitOpError("not supported in a NSS program");
            }

            if (failed(processNssTask(
                    _npu, nssTaskOp, nextLramAddress, nextXramAddress, remainingBlockSize
                ))) {
                return failure();
            }

            // always append after the first iteration
            nextLramAddress = AddressConstants::APPEND;
            nextXramAddress = AddressConstants::APPEND;
        }

        // find the next LRAM address for the block from the next op
        if (auto nextOp = dyn_cast<torq_hl::NextOp>(block.getTerminator())) {

            auto maybeLramAddress = nextOp.getLramAddress();

            if (!maybeLramAddress) {
                return nextOp.emitError() << "Missing next LRAM address";
            }

            nextLramAddress = maybeLramAddress.value();

            // find the next XRAM address
            nextXramAddress = xramCodeAddresses[idx + 1];

            // add the 4 bytes of the NEXT instruction that are not counted above
            remainingBlockSize -= 4;
        }

        // update the block size to the effective used size
        blockSizes[idx] = blockSizes[idx] - remainingBlockSize;
    }

    if (_npu.nssEnd() < 0) {
        return failure();
    }

    if (code) {
        if (failed(updateCode(_npu, xramCodeAddresses[0], *code))) {
            return failure();
        }
    }

    _npu.close();

    return success();
}

static torq_hl::CreateInvocationOp rewriteInvocationWithNewSizes(
    torq_hl::CreateInvocationOp createInvocationOp, ArrayRef<int32_t> newBlockSizes
) {

    IRRewriter rewriter(createInvocationOp);

    SmallVector<Type> resultTypes;
    SmallVector<int64_t> updatedXramAddresses;

    int baseXramAddress = createInvocationOp.getXramCodeAddresses().value()[0];

    resultTypes.push_back(createInvocationOp.getInvocation().getType());

    for (auto size : newBlockSizes) {
        auto codeSectionType = MemRefType::get({static_cast<int64_t>(size)}, rewriter.getI8Type());
        resultTypes.push_back(codeSectionType);
        updatedXramAddresses.push_back(baseXramAddress);
        baseXramAddress += size;
    }

    auto newCreateInvocation = rewriter.replaceOpWithNewOp<torq_hl::CreateInvocationOp>(
        createInvocationOp, resultTypes, createInvocationOp->getOperands(),
        createInvocationOp->getAttrs()
    );

    newCreateInvocation.setXramCodeAddresses(updatedXramAddresses);

    return newCreateInvocation;
}

static SmallVector<OpOperand *> appendGetBlockOpsUses(torq_hl::GetBlockOp getBlockOp) {

    SmallVector<OpOperand *> getBlockOpsUses;
    SmallVector<Value> worklist;

    worklist.push_back(getBlockOp.getResult());

    while (!worklist.empty()) {

        auto currentValue = worklist.pop_back_val();

        for (auto &use : currentValue.getUses()) {
            if (auto nextOp = dyn_cast<torq_hl::NextOp>(use.getOwner())) {
                int argIndex =
                    use.getOperandNumber() - nextOp.getArguments().getBeginOperandIndex();
                auto nextBlockArg = nextOp.getSuccessor()->getArgument(argIndex);
                worklist.push_back(nextBlockArg);
            }
            else {
                getBlockOpsUses.push_back(&use);
            }
        }
    }

    return getBlockOpsUses;
}

static FailureOr<SmallVector<std::pair<torq_hl::GetBlockOp, SmallVector<OpOperand *>>>>
findGetBlockOpsUses(torq_hl::CreateInvocationOp createInvocationOp) {

    SmallVector<Value> worklist;
    SmallVector<std::pair<torq_hl::GetBlockOp, SmallVector<OpOperand *>>> getBlockOpsUses;

    worklist.push_back(createInvocationOp.getInvocation());

    while (!worklist.empty()) {

        auto currentValue = worklist.pop_back_val();

        for (auto &use : currentValue.getUses()) {

            // if it's a start program op, we need to track all the uses inside the program
            if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(use.getOwner())) {

                int argIndex = use.getOperandNumber() - startOp.getArgs().getBeginOperandIndex();

                // skip uses that are not arguments of the start op
                if (startOp.getArgs().getBeginOperandIndex() > use.getOperandNumber() ||
                    argIndex >= startOp.getArgs().size()) {
                    continue;
                }

                auto otherCreateInvocation =
                    startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

                if (!otherCreateInvocation) {
                    continue;
                }

                auto otherProgramOp =
                    otherCreateInvocation.getProgram().getDefiningOp<torq_hl::ProgramOp>();

                if (!otherProgramOp) {
                    continue;
                }

                worklist.push_back(otherProgramOp.getBody().getArgument(argIndex));
            }

            // if it is a next op, we need to track all the uses inside the next block
            else if (auto nextOp = dyn_cast<torq_hl::NextOp>(use.getOwner())) {

                int argIndex =
                    use.getOperandNumber() - nextOp.getArguments().getBeginOperandIndex();

                auto nextBlockArg = nextOp.getSuccessor()->getArgument(argIndex);

                worklist.push_back(nextBlockArg);
            }

            // if the use is a get_block op, we find all the uses
            else if (auto getBlockOp = dyn_cast<torq_hl::GetBlockOp>(use.getOwner())) {
                getBlockOpsUses.push_back(
                    std::make_pair(getBlockOp, appendGetBlockOpsUses(getBlockOp))
                );
            }
            else if (isa<torq_hl::WaitProgramOp>(use.getOwner())) {
                continue;
            }
            else {
                return use.getOwner()->emitOpError("Unsupported usage of NSS invocation");
            }
        }
    }

    return getBlockOpsUses;
}

static LogicalResult updateGetBlockOperations(torq_hl::CreateInvocationOp createInvocationOp) {

    // find all places where the invocation is used as input to a get_block op
    auto maybeGetBlockOps = findGetBlockOpsUses(createInvocationOp);

    if (failed(maybeGetBlockOps)) {
        return failure();
    }

    OpBuilder builder(createInvocationOp);

    auto blockSizes =
        createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>().getBlockSizes().value();

    for (auto [getBlockOp, uses] : *maybeGetBlockOps) {

        int blockIndex = getBlockOp.getBlockIndex().getSExtValue();

        if (blockIndex < 0 || static_cast<size_t>(blockIndex) >= blockSizes.size()) {
            return getBlockOp->emitOpError("Block index out of range");
        }

        auto codeSectionType =
            MemRefType::get({static_cast<int64_t>(blockSizes[blockIndex])}, builder.getI8Type());

        getBlockOp.getResult().setType(codeSectionType);

        for (auto use : uses) {
            if (auto hostCopyOp = dyn_cast<torq_hl::HostCopyOp>(use->getOwner())) {

                if (use->getOperandNumber() != hostCopyOp.getInputMutable().getOperandNumber()) {
                    return hostCopyOp->emitOpError("Unsupported usage of get block result");
                }

                builder.setInsertionPoint(hostCopyOp);

                // create a subview of the destination with the correct size where to copy the code
                auto subviewOp = builder.create<memref::SubViewOp>(
                    hostCopyOp.getLoc(), hostCopyOp.getOutput(), SmallVector<int64_t>{0},
                    SmallVector<int64_t>{blockSizes[blockIndex]}, SmallVector<int64_t>{1}
                );

                // we need to manually set the address on the subview because the pass that
                // sets it was already executed
                auto maybeLramAddress = getAddress(hostCopyOp.getOutput());
                if (!maybeLramAddress) {
                    return hostCopyOp->emitOpError("Missing LRAM address on host copy output");
                }
                setLramAddress(subviewOp, maybeLramAddress.value());

                // update the destination of the host copy to be the subview and set the element
                // size
                hostCopyOp.getOutputMutable().set(subviewOp);
                hostCopyOp.setElementSizeBytes(blockSizes[blockIndex]);
            }
            else if (auto dmaInCfg = dyn_cast<torq_hw::DmaInCfgOp>(use->getOwner())) {

                builder.setInsertionPoint(dmaInCfg->getParentOp());

                // create a subview of the destination with the correct size where to copy the code
                auto subviewOp = builder.create<memref::SubViewOp>(
                    dmaInCfg.getLoc(), dmaInCfg.getWrite(), SmallVector<int64_t>{0},
                    SmallVector<int64_t>{blockSizes[blockIndex]}, SmallVector<int64_t>{1}
                );

                // update the write buffer of the dma in cfg to be the subview and set the address
                // and size
                dmaInCfg.getWriteMutable().set(subviewOp);

                SmallVector<torq_hw::DmaDimAttr> dims = {torq_hw::DmaDimAttr::get(
                    dmaInCfg.getContext(), static_cast<int32_t>(blockSizes[blockIndex]), 1
                )};
                auto ndl = torq_hw::DmaNdlAttr::get(dmaInCfg.getContext(), dims);
                dmaInCfg.setReadNdlAttr(ndl);
                dmaInCfg.setWriteNdlAttr(ndl);

                dmaInCfg.setReadAddress(createInvocationOp.getXramCodeAddresses().value(
                )[blockIndex]);
            }
            else {
                return use->getOwner()->emitOpError("Unsupported usage of get block result");
            }
        }
    }

    return success();
}

LogicalResult updateCodeSectionSizes(torq_hl::CreateInvocationOp createInvocationOp) {

    // find the code size for each block that was kept aside to serialize it (this is the max size
    // we can use)
    SmallVector<int32_t> blockSizes;

    // compile the program to determine the actual used sizes (this updates blockSizes)
    if (failed(compileInvocation(createInvocationOp, blockSizes))) {
        return failure();
    }

    for (auto blockSize : blockSizes) {
        assert(blockSize % 4 == 0 && "Unexpected non-aligned block size");
    }

    // update the torq_block_size attribute on the program op
    createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>().setBlockSizes(blockSizes);

    // rewrite the createInvocationOp to have the correct code section sizes
    auto newCreateInvocation = rewriteInvocationWithNewSizes(createInvocationOp, blockSizes);

    // update all get_block operations and all their uses
    if (failed(updateGetBlockOperations(newCreateInvocation))) {
        return failure();
    }

    return success();
}

LogicalResult createDescriptors(torq_hl::CreateInvocationOp createInvocationOp) {

    auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    // since we call this function after packing the blocks we can use this shortcut to compute
    // the total code size
    int totalSize = 0;
    auto maybeBlockSizes = programOp.getBlockSizes();

    for (auto blockSize : *maybeBlockSizes) {
        totalSize += blockSize;
    }

    SmallVector<int32_t> blockSizes;
    SmallVector<int8_t> code(totalSize, 0);

    // compile the invocation to get the bitstream
    if (failed(compileInvocation(createInvocationOp, blockSizes, &code))) {
        return failure();
    }

    IRRewriter rewriter(createInvocationOp);

    auto jobId = createInvocationOp->getAttr("torq-job-id");

    auto descriptorOp = rewriter.replaceOpWithNewOp<torq_hl::DescriptorOp>(
        createInvocationOp, createInvocationOp->getResultTypes(), createInvocationOp.getNameAttr(),
        rewriter.getIndexAttr(0), createInvocationOp.getProgram(),
        createInvocationOp.getExecutorCodeAddressesAttr(),
        createInvocationOp.getXramCodeAddressesAttr(),
        rewriter.getArrayAttr({rewriter.getDenseI8ArrayAttr(code)})
    );

    descriptorOp->setAttr("torq-job-id", jobId);

    return success();
}

void CompileNSSInvocationsPass::runOnOperation() {
    auto funcOp = getOperation();

    // find all NSS invocation
    SmallVector<torq_hl::CreateInvocationOp> invocationOps;
    for (auto createInvocationOp : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {
        if (createInvocationOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS) {
            invocationOps.push_back(createInvocationOp);
        }
    }

    // fixup all the NSS invocation code blocks to have the smallest possible size
    for (auto createInvocationOp : invocationOps) {
        if (failed(updateCodeSectionSizes(createInvocationOp))) {
            signalPassFailure();
            return;
        }
    }

    // rescan to find all the new invocations
    invocationOps.clear();
    for (auto createInvocationOp : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {
        if (createInvocationOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS) {
            invocationOps.push_back(createInvocationOp);
        }
    }

    for (auto createInvocationOp : invocationOps) {
        if (failed(createDescriptors(createInvocationOp))) {
            signalPassFailure();
            return;
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileNSSInvocationsPass() {
    return std::make_unique<CompileNSSInvocationsPass>();
}

} // namespace mlir::syna::torq
