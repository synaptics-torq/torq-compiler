// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "torq/Serialization/DescGen.h"

#define DEBUG_TYPE "torq-compile-slice-invocation"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

namespace {

class CompileSliceInvocationPass : public CompileSliceInvocationBase<CompileSliceInvocationPass> {
  public:
    using CompileSliceInvocationBase::CompileSliceInvocationBase;
    void runOnOperation() override;
};

const MemNdlAttr *
getNdl(const ArrayRef<MemNdlAttr> &ndls, NdlType type, int64_t index = 0, int64_t setId = 0) {
    for (auto &ndl : ndls) {
        if (ndl.getType() == type && ndl.getIndex() == index && ndl.getSetId() == setId) {
            return &ndl;
        }
    }
    return nullptr;
}

const RegNdlAttr *getNdl(const ArrayRef<RegNdlAttr> &ndls, NdlType type, int64_t setId = 0) {
    for (auto &ndl : ndls) {
        if (ndl.getType() == type && ndl.getSetId() == setId) {
            return &ndl;
        }
    }
    return nullptr;
}

static void convert(const torq_hw::MemNdlAttr attr, MemNdl &ndl, ArrayAttr symbolValues) {
    for (auto dim : attr.getDims()) {

        auto maybeStride = dim.getStrideAsI64(symbolValues);

        if (!maybeStride.has_value()) {
            llvm::report_fatal_error("Unable to compute stride");
        }

        switch (dim.getType()) {
        case torq_hw::DimType::L:
            ndl.addLdim(dim.getTag(), dim.getCount(), *maybeStride);
            break;
        case torq_hw::DimType::H:
            ndl.addHdim(dim.getTag(), dim.getCount(), *maybeStride);
            break;
        case torq_hw::DimType::S:
            ndl.addSdim(dim.getTag(), dim.getCount(), *maybeStride);
            break;
        }
    }
    ndl.setSyncMode(attr.getSyncMode(), attr.getSyncNhd());

    // FIXME: How to handle attr.getNdlLaddr() and attr.getNdlXaddr() ?
}

static void convert(const torq_hw::RegNdlAttr attr, RegNdl &ndl) {
    for (auto dim : attr.getDims()) {
        switch (dim.getType()) {
        case torq_hw::DimType::L:
            ndl.addLdim(dim.getTag(), dim.getCount(), dim.getStride());
            break;
        case torq_hw::DimType::H:
            ndl.addHdim(dim.getTag(), dim.getCount(), dim.getStride());
            break;
        case torq_hw::DimType::S:
            llvm::report_fatal_error("DimType::S is not supported for RegNdl");
            break;
        }
    }
}

void convert(
    Operation *op, IRMapping &mapping, const ArrayRef<MemNdlAttr> &ndls, NdlType type,
    int64_t index, int64_t setId, MemNdl &memNdl, ArrayAttr symbolValuesAttr,
    Operation::operand_range buffers, int64_t byteAlignment = 0
) {

    if (buffers.size() == 0) {
        return;
    }

    auto buffer = mapping.lookup(*(buffers.begin()));

    auto ndl = getNdl(ndls, type, index, setId);

    if (!ndl)
        return;

    convert(*ndl, memNdl, symbolValuesAttr);

    auto bufferType = dyn_cast<MemRefType>(buffer.getType());

    if (!bufferType) {
        llvm::errs() << "FATAL: Buffer is not of MemRefType.\n";
        llvm::errs() << "Buffer: " << buffer << "\n";
        llvm::report_fatal_error("Buffer is not of MemRefType");
    }

    auto addr = getLramAddress(buffer, getMemRefTypeOffsetBytes(bufferType) + ndl->getOffset());

    if (!addr.has_value()) {
        llvm::errs() << "FATAL: Unable to get base address.\n";
        llvm::errs() << "Buffer: " << buffer << "\n";
        llvm::errs() << "NDL info:\n";
        llvm::errs() << "  Type: " << stringifyNdlType(type) << "\n";
        llvm::errs() << "  Index: " << index << ", SetId: " << setId << "\n";
        llvm::errs() << "  Offset: " << ndl->getOffset() << "\n";
        llvm::report_fatal_error("Unable to get base address");
    }

    memNdl.setBaseAddress(addr.value());

    if (byteAlignment && (addr.value() % byteAlignment) != 0) {
        llvm::report_fatal_error("buffer + offset has wrong alignment");
    }
}

static void convert(const ArrayRef<RegNdlAttr> &ndls, NdlType type, int64_t setId, RegNdl &regNdl) {
    if (auto ndl = getNdl(ndls, type, setId))
        convert(*ndl, regNdl);
}

static SliceTask toSliceTask(syna::torq_hw::SliceTaskOp op, IRMapping &mapping) {
    if (!op) {
        // Terminate with fatal error
        llvm::report_fatal_error("Op is null in toSliceTask");
    }
    LLVM_DEBUG({ llvm::dbgs() << "*** AI operation name: " << op.getOpName() << " ***\n"; });
    SliceTask task(AluMode::G64x4);
    auto sliceCfg = op.getSliceCfgAttr();
    auto aluOp0 = sliceCfg.getAluOp0Mode();
    if (!aluOp0.empty()) {
        task.setAluOp0(aluOp0[0], aluOp0[1], aluOp0[2], aluOp0[3]);
    }
    auto aluOp1 = sliceCfg.getAluOp1Mode();
    if (!aluOp1.empty()) {
        task.setAluOp1(aluOp1[0], aluOp1[1], aluOp1[2], aluOp1[3]);
    }

    // If all aluOp0[] is DBYP or if aluOp0 is empty, weights are not needed
    bool weightsBypasssed =
        aluOp0.empty() || std::all_of(aluOp0.begin(), aluOp0.end(), [](auto op) {
            return op == torq_hw::ALUOp0Mode::DBYP;
        });

    auto act_lsh = sliceCfg.getActLsh();
    task.setActivation(
        {{act_lsh[0], act_lsh[1], act_lsh[2], act_lsh[3]},
         sliceCfg.getActRsh(),
         sliceCfg.getActZeroPoint(),
         sliceCfg.getActClipMin(),
         sliceCfg.getActClipMax(),
         sliceCfg.getActMode()}
    );

    task.setKernel(
        {sliceCfg.getKernelLeft(), sliceCfg.getKernelRight(), sliceCfg.getKernelTop(),
         sliceCfg.getKernelBottom()}
    );

    task.setPad(
        {sliceCfg.getPadLeft(), sliceCfg.getPadRight(), sliceCfg.getPadTop(),
         sliceCfg.getPadBottom(), sliceCfg.getPadValue()}
    );

    task.setStride(sliceCfg.getStride(), sliceCfg.getStrideOffset());

    task.setTable(sliceCfg.getTable());

    // W: u8 for even cycle, s8 for odd cycle (all D's are treated as signed 8b)
    task.setAluDUnsigned(sliceCfg.getAluDUnsigned());
    task.setAluWUnsigned(sliceCfg.getAluWUnsigned());

    task.setActRoundingMode(sliceCfg.getActRoundMode());
    task.setAluActDisable(sliceCfg.getAluDisable(), sliceCfg.getActDisable());
    task.setWeightFormat(sliceCfg.getWeightFormat());
    task.setAluActNumberFormat(sliceCfg.getAluFormat(), sliceCfg.getActFormat());
    task.setActSumBits(sliceCfg.getActSumBits());

    // find all symbol values used in the ndls
    SmallVector<Attribute> symbolValues;

    IRRewriter rewriter(op.getContext());

    for (auto sym : op.getSymbols()) {
        auto addressOp = dyn_cast<torq_hw::GetAddressOp>(sym.getDefiningOp());

        if (!addressOp) {
            llvm::report_fatal_error("Symbol used by task is not of type GetAddressOp");
        }

        auto addr = getLramAddress(mapping.lookup(addressOp.getMemref()));

        if (!addr.has_value()) {
            llvm::report_fatal_error("Unable to get symbol base address");
        }

        symbolValues.push_back(rewriter.getIndexAttr(addr.value()));
    }

    auto symbolValuesAttr = ArrayAttr::get(op.getContext(), symbolValues);

    const auto &memNdls = op.getMemNdls();

    convert(*getNdl(memNdls, NdlType::REF), task.ref, symbolValuesAttr);

    convert(op, mapping, memNdls, NdlType::DEDR, 0, 0, task.dedr, symbolValuesAttr, op.getD());

    convert(op, mapping, memNdls, NdlType::DEDR, 0, 1, task.dedr1, symbolValuesAttr, op.getDx());

    convert(op, mapping, memNdls, NdlType::DEWR, 0, 0, task.dewr, symbolValuesAttr, op.getW());
    convert(op, mapping, memNdls, NdlType::DEBR, 0, 0, task.debr, symbolValuesAttr, op.getB());
    convert(op, mapping, memNdls, NdlType::DEBR, 0, 1, task.debr1, symbolValuesAttr, op.getBx());
    convert(op, mapping, memNdls, NdlType::DEQW, 0, 0, task.deqw, symbolValuesAttr, op.getQ());

    if (op.getW().empty() && !weightsBypasssed) {
        llvm::report_fatal_error("Weights are not bypassed but no weight buffer is provided");
    }

    const auto &regNdls = op.getRegNdls();
    convert(regNdls, NdlType::CEDW, 0, task.cedw);
    convert(regNdls, NdlType::CEDR, 0, task.cedr);
    convert(regNdls, NdlType::CEWW, 0, task.ceww);
    convert(regNdls, NdlType::CEWR, 0, task.cewr);
    convert(regNdls, NdlType::CEPR, 0, task.cepr);
    convert(regNdls, NdlType::ACBW, 0, task.acbw);
    convert(regNdls, NdlType::ACBR, 0, task.acbr);
    convert(regNdls, NdlType::ACPR, 0, task.acpr);
    convert(regNdls, NdlType::ACPW, 0, task.acpw);
    convert(regNdls, NdlType::ALDW, 0, task.aldw);

    // We don't want to dump these value here because they are already dumped
    // when we dump the corresponding constOp. In general we cannot use this
    // dump because there is no 1:1 mapping beween constants and arguments to
    // the NDLs
    /*
    if (!op.getB().empty()) {
        getConstData(op.getB()[0], task.bMem);
    }

    if (!op.getW().empty()) {
        getConstData(op.getW()[0], task.wMem);
    }

    if (!op.getD().empty()) {
        getConstData(op.getD()[0], task.dMem);
    }
    */

    // auto task = op->getParentOfType<torq_hl::SliceProgramOp>();
    // auto nextNdlXramBaseAddress = task.getXramAddress().value() + 0x200;
    // auto nextNdlLramBaseAddress = task.getLramAddress().value() + 0x200;

    return task;
}

static FailureOr<torq_hl::StartProgramOp>
findStartProgramOp(torq_hl::CreateInvocationOp createInvocationOp) {
    torq_hl::StartProgramOp startProgramOp;

    for (auto user : createInvocationOp.getInvocation().getUsers()) {
        startProgramOp = dyn_cast<torq_hl::StartProgramOp>(user);

        if (startProgramOp) {
            return startProgramOp;
        }
    }

    return createInvocationOp.emitError("No start_program operation found for invocation");
}

static LogicalResult processSliceTaskOp(
    DescGen &_npu, torq_hw::SliceTaskOp &taskOp, uint32_t &nextNdlLramBaseAddress,
    uint32_t &nextNdlXramBaseAddress, uint32_t &remainingNdlSize, uint32_t &remainingCfgSize,
    IRMapping &mapping
) {
    LLVM_DEBUG({ llvm::dbgs() << "*** layer type: " << taskOp->getName() << " ***\n"; });
    auto baseline_task = toSliceTask(taskOp, mapping);

    if (!baseline_task.isValid()) {
        LLVM_DEBUG({ llvm::dbgs() << " skip op " << taskOp->getName() << "\n"; });
        // nothing to generate, return quietly
        return success();
    }

    LLVM_DEBUG({ llvm::dbgs() << baseline_task.toLogStr() << "\n"; });

    uint32_t totalNdlSize;
    uint32_t totalCfgSize;

    auto ret = baseline_task.write(
        _npu.get(), _npu.getCurrentSliceId(), nextNdlLramBaseAddress, nextNdlXramBaseAddress,
        totalCfgSize, totalNdlSize
    );

    if (totalCfgSize > remainingCfgSize) {
        taskOp->emitOpError("CFG size exceeds maximum size");
        return failure();
    }

    if (totalNdlSize > remainingNdlSize) {
        taskOp->emitOpError("NDL size exceeds maximum size");
        return failure();
    }

    if (!ret) {
        taskOp->emitOpError("Failed to serialize task");
        return failure();
    }

    remainingCfgSize -= totalCfgSize;
    remainingNdlSize -= totalNdlSize;

    return success();
}

LogicalResult updateCode(DescGen &_npu, uint32_t xramAddress, SmallVector<int8_t> &code) {

    const torq_bitstream_segment_t *segment = _npu.getBitstream();

    while (segment) {

        if (segment->xram_addr == AddressConstants::NONE) {
            segment = segment->next;
            continue;
        }

        LLVM_DEBUG({
            std::string lramAddrString = llvm::formatv("{0:x+8}", segment->lram_addr);
            std::string xramAddrString = llvm::formatv("{0:x+8}", segment->xram_addr);

            llvm::dbgs() << " serializing code segment with "
                         << " xram address = " << xramAddrString << " (size = " << segment->size
                         << ")\n";

            std::stringstream ss;
            for (size_t i = 0; i < segment->size; i++) {
                ss << std::hex << "" << (int)segment->data[i];
            }
            llvm::dbgs() << "    data: " << ss.str() << "\n";
        });

        if (segment->xram_addr < xramAddress) {
            return failure();
        }

        auto offset = segment->xram_addr - xramAddress;

        if ((offset + segment->size) > code.size()) {
            return failure();
        }

        std::copy(segment->data, segment->data + segment->size, code.begin() + offset);

        segment = segment->next;
    }

    return success();
}

static LogicalResult compileProgram(
    uint32_t lramAddress, uint32_t xramAddress, uint32_t &cfgSize, uint32_t &ndlSize, int slc,
    torq_hl::ProgramOp programOp, IRMapping &mapping, SmallVector<int8_t> *code = nullptr
) {

    uint32_t nextNdlXramBaseAddress = xramAddress + cfgSize;
    uint32_t nextNdlLramBaseAddress = lramAddress + cfgSize;

    uint32_t remainingNdlSpace = ndlSize;
    uint32_t remainingCfgSpace = cfgSize;

    DescGen _npu;

    std::string dump_path = "";

    if (!_npu.open(dump_path.c_str())) {
        llvm::errs() << "Serializer failed to open " << dump_path << "\n";
        exit(1);
    }

    if (!_npu.beginCfg(slc, lramAddress, xramAddress)) {
        return failure();
    }

    for (auto &op : programOp.getOps()) {

        if (isa<torq_hw::SliceProfilingOp, torq_hw::GetAddressOp, torq_hl::ReturnOp>(op)) {
            continue;
        }

        auto taskOp = dyn_cast<torq_hw::SliceTaskOp>(op);

        if (!taskOp) {
            return op.emitOpError("Unsupported operation in program for slice");
        }

        if (failed(processSliceTaskOp(
                _npu, taskOp, nextNdlLramBaseAddress, nextNdlXramBaseAddress, remainingNdlSpace,
                remainingCfgSpace, mapping
            ))) {
            return failure();
        }
    }

    if (_npu.endCfg() < 0) {
        return failure();
    }

    if (code) {
        if (failed(updateCode(_npu, xramAddress, *code))) {
            return failure();
        }
    }

    cfgSize = cfgSize - remainingCfgSpace;
    ndlSize = ndlSize - remainingNdlSpace;

    _npu.close();

    return success();
}

static LogicalResult compileInvocation(torq_hl::CreateInvocationOp createInvocationOp) {

    // This function generates the bitstream for a task, this is stored only in
    // XRAM segments and will be loaded by the main NSS program

    auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    if (!programOp) {
        return createInvocationOp->emitOpError("program is not a torq_hl::ProgramOp");
    }

    auto maybeStartProgramOp = findStartProgramOp(createInvocationOp);

    if (failed(maybeStartProgramOp)) {
        return failure();
    }

    auto startOp = maybeStartProgramOp.value();

    // SliceID is needed only for logging purpose in order to dump the CFG descriptors in text
    // files, using the same id for all descriptors wouldn´t change the functionality.

    auto maybeId = createInvocationOp.getExecutorId();

    if (!maybeId) {
        return createInvocationOp->emitOpError("Missing executor id");
    }

    auto slc = maybeId.value().getZExtValue();

    auto maybeLramCodeAddress = getLramAddress(startOp.getCodeSections()[0]);

    if (!maybeLramCodeAddress) {
        return createInvocationOp->emitOpError("Missing lram code address");
    }

    auto maybeXramCodeAddresses = createInvocationOp.getXramCodeAddresses();

    if (!maybeXramCodeAddresses || maybeXramCodeAddresses->empty()) {
        return createInvocationOp->emitOpError("Missing xram code address");
    }

    // compute the initial addresses and sizes for memory that will host cfgs and ndls
    auto lramAddress = *maybeLramCodeAddress;
    auto xramAddress = (*maybeXramCodeAddresses)[0];

    // find the maximal code size that was kept aside to run this program
    auto codeType = cast<MemRefType>(createInvocationOp.getCodeSections()[0].getType());
    uint32_t programSize = getEncodedTotalSizeBytes(codeType);

    // we use 0x900 bytes for CFGs, the rest is for NDLs
    uint32_t cfgSize = 0x900;

    if (programSize <= cfgSize) {
        return createInvocationOp->emitOpError("Program size too small to fit CFGs instructions");
    }

    uint32_t ndlSize = programSize - cfgSize;

    auto startProgramOp = *maybeStartProgramOp;

    // compute the mapping from block arguments to invocation arguments

    IRMapping mapping;

    auto blockArgs = programOp.getBody().front().getArguments();

    auto argValues = startProgramOp.getArgs();

    for (auto [arg, value] : llvm::zip(blockArgs, argValues)) {
        mapping.map(arg, value);
    }

    // compile once the program to figure out the real cfg/ndl sizes
    if (failed(compileProgram(lramAddress, xramAddress, cfgSize, ndlSize, slc, programOp, mapping)
        )) {
        return createInvocationOp->emitOpError("Failed to compile slice program");
    }

    // re-compile again to get the actual bitstream, this time with the exact sizes
    SmallVector<int8_t> code(ndlSize + cfgSize, 0);
    if (failed(compileProgram(
            lramAddress, xramAddress, cfgSize, ndlSize, slc, programOp, mapping, &code
        ))) {
        return createInvocationOp->emitOpError("Failed to compile slice program");
    }

    // replace the createInvocation op code section a descriptor op

    IRRewriter rewriter(createInvocationOp);

    auto descriptorCode = rewriter.getDenseI8ArrayAttr(code);
    auto codeSections = rewriter.getArrayAttr({descriptorCode});
    auto codeSectionType =
        MemRefType::get({static_cast<int64_t>(code.size())}, rewriter.getI8Type());

    auto descriptorOp = rewriter.replaceOpWithNewOp<torq_hl::DescriptorOp>(
        createInvocationOp,
        TypeRange{createInvocationOp.getInvocation().getType(), codeSectionType},
        createInvocationOp.getNameAttr(), rewriter.getIndexAttr(slc),
        createInvocationOp.getProgram(), rewriter.getDenseI64ArrayAttr(lramAddress),
        rewriter.getDenseI64ArrayAttr(xramAddress), codeSections
    );

    // TODO: this is quite brittle, we probably should have an operation
    // to load the code sections instead of leveraging torq_hl.load ops
    // directly

    // update all the dma load of descriptors to use the correct size
    for (auto &use : descriptorOp.getCodeSections()[0].getUses()) {
        if (auto loadOp = dyn_cast<torq_hl::LoadOp>(use.getOwner())) {
            rewriter.setInsertionPoint(loadOp);

            auto destSubview = rewriter.create<memref::SubViewOp>(
                loadOp.getLoc(), loadOp.getOutput(), SmallVector<int64_t>{0},
                SmallVector<int64_t>{static_cast<int64_t>(code.size())}, SmallVector<int64_t>{1}
            );

            auto maybeLramAddress = getLramAddress(loadOp.getOutput());

            if (!maybeLramAddress.has_value()) {
                return loadOp.emitError("Unable to get LRAM address of load destination");
            }

            setLramAddress(destSubview, maybeLramAddress.value());

            loadOp.getOutputMutable().set(destSubview);

            // fix the amount of data to load
            loadOp.setElementSizeBytes(static_cast<int32_t>(code.size()));

            break;
        }
    }

    if (programOp.use_empty()) {
        rewriter.eraseOp(programOp);
    }

    return success();
}

void CompileSliceInvocationPass::runOnOperation() {
    auto funcOp = getOperation();

    SmallVector<torq_hl::CreateInvocationOp> invocationOps;
    for (auto createInvocationOp : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {
        if (createInvocationOp.getInvocation().getType().getExecutor() ==
            torq_hl::Executor::Slice) {
            invocationOps.push_back(createInvocationOp);
        }
    }

    for (auto createInvocationOp : invocationOps) {
        if (failed(compileInvocation(createInvocationOp))) {
            signalPassFailure();
            return;
        }
    }
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createCompileSliceInvocationPass() {
    return std::make_unique<CompileSliceInvocationPass>();
}

} // namespace mlir::syna::torq
