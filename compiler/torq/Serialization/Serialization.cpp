// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Serialization.h"

#include "DescGen.h"

#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq_executable_def_builder.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

using namespace std;
using namespace mlir::syna::torq_hw;

#define DEBUG_TYPE "torq-serialization"

namespace mlir::syna::torq {

static llvm::cl::opt<std::string> clTorqDescriptorDumpDir(
    "torq-dump-descriptors-dir",
    llvm::cl::desc("dump Torq API descriptors to the specified directory"), llvm::cl::init("")
);

static llvm::cl::opt<bool> clDisableAsyncSliceWait(
    "torq-disable-async-slice-wait", llvm::cl::desc("Disable Asynchronous Slice Wait"),
    llvm::cl::init(false)
);

static llvm::cl::opt<bool> clEnableBufferDebugInfo(
    "torq-enable-buffer-debug-info", llvm::cl::desc("Add buffer debug information to the model"),
    llvm::cl::init(false)
);

/// TorqHW serialization class
class Serializer {
  public:
    Serializer(iree_compiler::FlatbufferBuilder &builder, string dump_path);

    LogicalResult serializeFunction(mlir::FunctionOpInterface funcOp);

  private:
    LogicalResult processSliceTaskOp(
        torq_hw::SliceTaskOp &taskOp, uint32_t &nextNdlLramBaseAddress,
        uint32_t &nextNdlXramBaseAddress, uint32_t &remainingNdlSize, uint32_t &remainingCfgSize,
        torq_hl::CreateInvocationOp &createInvocationOp
    );

    LogicalResult
    addSegment(uint32_t xramAddress, mlir::DenseIntOrFPElementsAttr data, bool isCode);

    FailureOr<flatbuffers_vec_ref_t> serializeRuntimeProgram(FunctionOpInterface funcOp);

    LogicalResult serializeInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeSliceInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeCssInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeHostInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeNssInvocation(torq_hl::CreateInvocationOp &op);

    FailureOr<flatbuffers_uint8_vec_ref_t> serializeHostCode(mlir::FunctionOpInterface funcOp);

    LogicalResult processConstOp(torq_hl::ConstOp &op);

    LogicalResult processNssTask(
        torq_hw::NssTaskOp nssTask, torq_hl::CreateInvocationOp &invocation, int lramAddress,
        int xramAddress
    );

    flatbuffers_uint32_vec_ref_t createUI32Vector(std::optional<ArrayRef<int64_t>> vec);

    flatbuffers_string_vec_ref_t createLocationsVector(ArrayRef<Location> vec);

    iree_hal_torq_BufferDebugInfo_ref_t createBufferDebugInfo(
        int64_t bufferId, uint64_t allocationAction, uint64_t lastUseAction,
        uint64_t deallocationAction, TypedValue<MemRefType> buffer
    );

    flatbuffers_vec_ref_t createBuffersDebugInfoVector(mlir::FunctionOpInterface funcOp);

    flatbuffers_ref_t createLocationString(Location loc);

    // add the given segment to the segments loaded by jobX in the hardware test vector dump
    // this is required to ensure all buffers (including constants and code segments) required by
    // the program are correctly loaded (torq_api will only dump things in the current job)
    LogicalResult dumpSegmentDescriptor(
        int jobId, bool xram, int address, const void *data, int size, bool isCode
    );

    LogicalResult saveCodeSegments();

    iree_compiler::FlatbufferBuilder &_builder;
    SmallVector<iree_hal_torq_Segment_ref_t> _segments;
    DenseSet<void *> _savedSegments;
    DescGen _npu;

    int _nssBlockSize = 0;
    int _nssProgramCount = 0;

    std::string _dump_path;

    /// next id to use when dumping segment descriptors
    int _codeSegmentDumpId{0};
    int _constantSegmentDumpId{0};

    /// temporary location where we store stuff that goes into job0/tv.init.mem.lst while
    /// the file is still open by torq_api
    std::string _segmentDumpInfo{};

    template <typename T>
    LogicalResult createISegment(
        DenseIntOrFPElementsAttr &data, iree_compiler::FlatbufferBuilder &builder,
        flatbuffers_uint8_vec_ref_t &dataRef, int32_t xramAddress
    ) {

        auto bitWidth = data.getType().getElementTypeBitWidth();
        auto value = data.getSplatValue<IntegerAttr>();
        int byteWidth = bitWidth / 8;
        SmallVector<T> dataVec(data.getNumElements(), value.getInt());
        dataRef = flatbuffers_uint8_vec_create(
            builder, (const uint8_t *)dataVec.data(), data.size() * byteWidth
        );
        if (!succeeded(dumpSegmentDescriptor(
                0, true, xramAddress, dataVec.data(), dataVec.size_in_bytes(), false
            ))) {
            return failure();
        }
        return success();
    }

    template <typename T>
    LogicalResult createFSegment(
        DenseIntOrFPElementsAttr &data, iree_compiler::FlatbufferBuilder &builder,
        flatbuffers_uint8_vec_ref_t &dataRef, int32_t xramAddress
    ) {
        int dtypeSize = sizeof(T);
        auto totalSize = dtypeSize * data.size();
        T value = static_cast<T>(data.getSplatValue<APFloat>().bitcastToAPInt().getZExtValue());
        vector<T> dataVec(totalSize);
        std::fill(dataVec.begin(), dataVec.end(), value);
        dataRef = flatbuffers_uint8_vec_create(builder, (const uint8_t *)dataVec.data(), totalSize);
        if (!succeeded(dumpSegmentDescriptor(
                0, true, xramAddress, (const uint8_t *)dataVec.data(), totalSize, false
            ))) {
            return failure();
        }
        return success();
    }
};

static auto getDispatchName(Operation *moduleOp) {
    SmallVector<mlir::iree_compiler::IREE::HAL::ExecutableExportOp> exportOps =
        llvm::to_vector(moduleOp
                            ->getParentOfType<mlir::iree_compiler::IREE::HAL::ExecutableVariantOp>()
                            .getOps<mlir::iree_compiler::IREE::HAL::ExecutableExportOp>());
    return exportOps.front().getName().str();
}

LogicalResult serializeTorqHW(mlir::ModuleOp moduleOp, DenseIntElementsAttr &binaryAttr) {

    auto dispatchName = getDispatchName(moduleOp);

    std::string dump_path = "";

    if (clTorqDescriptorDumpDir != "") {
        std::filesystem::create_directory(clTorqDescriptorDumpDir.getValue());
        dump_path = clTorqDescriptorDumpDir + "/" + dispatchName;
    }

    if (clDisableAsyncSliceWait) {
        llvm::dbgs() << "\n***** Asynchronous Slice Wait is disabled *****\n\n";
    }

    LLVM_DEBUG({ llvm::dbgs() << "****** " << dispatchName << " ******\n"; });

    iree_compiler::FlatbufferBuilder builder;

    Serializer serializer(builder, dump_path);

    auto funcOps = llvm::to_vector(moduleOp.getOps<mlir::FunctionOpInterface>());

    if (funcOps.size() == 0) {
        return moduleOp.emitError("no function found");
    }

    if (funcOps.size() > 1) {
        return moduleOp.emitError("Multiple functions not supported");
    }

    if (failed(serializer.serializeFunction(funcOps[0])))
        return failure();

    binaryAttr = builder.getBufferAttr(moduleOp.getContext());

    return success();
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

static bool toNssTask(
    syna::torq_hw::NssTaskOp taskOp, NssTask &task, torq_hl::CreateInvocationOp &createInvocationOp
) {
    DmaParams dmaParams{};
    CssParams cssParams{};
    CDmaParams cdmaParams{};
    SliceParams sparams[HwInfo::slice_count]{};

    for (auto &op : taskOp.getBody().front().getOperations()) {
        if (auto cfgOp = dyn_cast<syna::torq_hw::DmaInCfgOp>(op)) {

            if (!convert(cfgOp, task.dixr, task.dilw)) {
                return false;
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
                return false;
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
                llvm::report_fatal_error("SliceId >= slice count");
            }
            sparams[sid].start = true;
            if (clDisableAsyncSliceWait) {
                sparams[sid].wait = true;
            }

            auto cfgAddr =
                getDataStartAddress(sStartOp.getProgram(), 0, createInvocationOp.getInvocation());

            if (!cfgAddr) {
                llvm::report_fatal_error("Cannot find LRAM address for slice program");
            }

            sparams[sid].cfgAddr = cfgAddr.value();
        }
        else if (auto sWaitOp = dyn_cast<syna::torq_hw::SliceWaitOp>(op)) {
            auto sid = sWaitOp.getId().getSExtValue();
            if (sid >= HwInfo::slice_count) {
                llvm::report_fatal_error("SliceId >= slice count");
            }
            if (!clDisableAsyncSliceWait) {
                sparams[sid].wait = true;
            }
        }
        else if (auto cssStartOp = dyn_cast<syna::torq_hw::CSSStartOp>(op)) {
            cssParams.start = true;

            if (!cssStartOp.getProgramAddress()) {
                cssStartOp.emitError("Missing program address");
                llvm::report_fatal_error("Missing program address");
            }

            // FIXME: currently we support only CSS programs at address 0x0
            assert(
                cssStartOp.getProgramAddress().value() == 0 && "CSS program address must be zero"
            );

            cssParams.startAddr = cssStartOp.getProgramAddress().value();

            if (!cssStartOp.getArgAddressesAddress()) {
                cssStartOp.emitError("Missing argument addresses address");
                llvm::report_fatal_error("Missing argument addresses address");
            }

            cssParams.mbx[0] = cssStartOp.getArgAddressesAddress().value();
        }
        else if (auto cssWaitOp = dyn_cast<syna::torq_hw::CSSWaitOp>(op)) {
            cssParams.wait = true;
        }
        else if (auto cdmaStartOp = dyn_cast<syna::torq_hw::CDMAStartOp>(op)) {
            cdmaParams.start = true;

            if (!cdmaStartOp.getSrcAddress()) {
                cdmaStartOp.emitError("Missing source address");
                llvm::report_fatal_error("Missing source address");
            }

            cdmaParams.srcAddr = cdmaStartOp.getSrcAddress().value();

            if (!cdmaStartOp.getDestAddress()) {
                cdmaStartOp.emitError("Missing destination address");
                llvm::report_fatal_error("Missing destination address");
            }

            cdmaParams.dstAddr = cdmaStartOp.getDestAddress().value();

            auto srcType = cdmaStartOp.getSrc().getType();
            cdmaParams.len = getEncodedDataSizeBytes(srcType);
        }
        else if (auto cdmaWaitOp = dyn_cast<syna::torq_hw::CDMAWaitOp>(op)) {
            cdmaParams.wait = true;
        }
        else {
            llvm::errs() << "Unable to serialize operation: " << op.getName() << "\n";
            llvm::report_fatal_error("Unable to serialize operation");
            return false;
        }
    }
    task.setCssParams(cssParams);
    task.setCdmaParams(cdmaParams);
    task.setDmaParams(dmaParams);
    for (size_t i = 0; i < HwInfo::slice_count; ++i)
        task.setSliceParams(i, sparams[i]);
    return true;
}

static void getConstData(Value val, DumpMemInfo &memInfo) {
    if (!val) {
        return;
    }
    auto allocOp = val.getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
        return;
    }
    for (auto user : allocOp->getUsers()) {
        if (auto dmaInOp = dyn_cast<torq_hw::DmaInCfgOp>(user)) {
            auto constOp = dyn_cast<torq_hl::ConstOp>(dmaInOp.getRead().getDefiningOp());
            if (!constOp) {
                continue;
            }
            auto valueAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue());
            if (!valueAttr)
                continue;
            const auto &rawData = valueAttr.getRawData();
            memInfo.data = static_cast<const void *>(rawData.data());
            memInfo.size = rawData.size();
            memInfo.addr = getXramAddress(constOp.getOperation()).value();

            return;
        }
    }
}

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

void convert(const ArrayRef<RegNdlAttr> &ndls, NdlType type, int64_t setId, RegNdl &regNdl) {
    if (auto ndl = getNdl(ndls, type, setId))
        convert(*ndl, regNdl);
}

void convert(
    torq_hl::CreateInvocationOp createInvocationOp, const ArrayRef<MemNdlAttr> &ndls, NdlType type,
    int64_t index, int64_t setId, MemNdl &memNdl, ArrayAttr symbolValuesAttr,
    ::mlir::Operation::operand_range buffer, int64_t byteAlignment = 0
) {
    auto ndl = getNdl(ndls, type, index, setId);
    if (!ndl)
        return;
    convert(*ndl, memNdl, symbolValuesAttr);
    if (!buffer.empty()) {
        auto memSpace = getEncodingMemorySpace(cast<MemRefType>(buffer[0].getType()));
        if (memSpace != torq_hl::MemorySpace::Lram) {
            createInvocationOp.emitError("Only LRAM memory space is supported for buffers in NDLs");
        }
        auto addr =
            getDataStartAddress(buffer[0], ndl->getOffset(), createInvocationOp.getInvocation());

        if (!addr.has_value()) {
            llvm::errs() << "FATAL: Unable to get base address.\n";
            llvm::errs() << "Buffer: " << buffer[0] << "\n";
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
}

static SliceTask
toSliceTask(syna::torq_hw::SliceTaskOp op, torq_hl::CreateInvocationOp createInvocationOp) {
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

        auto addr =
            getDataStartAddress(addressOp.getMemref(), 0, createInvocationOp.getInvocation());

        if (!addr.has_value()) {
            llvm::report_fatal_error("Unable to get symbol base address");
        }

        symbolValues.push_back(rewriter.getIndexAttr(addr.value()));
    }

    auto symbolValuesAttr = ArrayAttr::get(op.getContext(), symbolValues);

    const auto &memNdls = op.getMemNdls();

    convert(*getNdl(memNdls, NdlType::REF), task.ref, symbolValuesAttr);
    convert(
        createInvocationOp, memNdls, NdlType::DEDR, 0, 0, task.dedr, symbolValuesAttr, op.getD()
    );
    convert(
        createInvocationOp, memNdls, NdlType::DEDR, 0, 1, task.dedr1, symbolValuesAttr, op.getDx()
    );
    convert(
        createInvocationOp, memNdls, NdlType::DEWR, 0, 0, task.dewr, symbolValuesAttr, op.getW()
    );
    convert(
        createInvocationOp, memNdls, NdlType::DEBR, 0, 0, task.debr, symbolValuesAttr, op.getB()
    );
    convert(
        createInvocationOp, memNdls, NdlType::DEBR, 0, 1, task.debr1, symbolValuesAttr, op.getBx()
    );
    convert(
        createInvocationOp, memNdls, NdlType::DEQW, 0, 0, task.deqw, symbolValuesAttr, op.getQ()
    );

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

Serializer::Serializer(iree_compiler::FlatbufferBuilder &builder, std::string dump_path)
    : _builder(builder), _dump_path(dump_path) {
    if (!_npu.open(dump_path.c_str())) {
        llvm::errs() << "Serializer failed to open " << dump_path << "\n";
        llvm::report_fatal_error("Serializer failed to open dump_path");
    }
}

LogicalResult Serializer::processConstOp(torq_hl::ConstOp &constOp) {

    auto data = mlir::cast<DenseIntOrFPElementsAttr>(constOp.getValue());

    flatbuffers_uint8_vec_ref_t dataRef;

    auto xramAddress = getXramAddress(constOp.getOperation()).value();

    // FIXME: we probably don't want to use DenseIntElementsAttr in torq_hl::ConstOp, instead use
    // DenseIntArrayAttr that doesn't support splats
    if (data.isSplat()) {
        auto elementType = data.getElementType();
        auto bitWidth = data.getType().getElementTypeBitWidth();
        if (elementType.isInteger()) {
            if (bitWidth == 8) {
                if (createISegment<uint8_t>(data, _builder, dataRef, xramAddress).failed()) {
                    return failure();
                }
            }
            else if (bitWidth == 16) {
                if (createISegment<uint16_t>(data, _builder, dataRef, xramAddress).failed()) {
                    return failure();
                }
            }
            else if (bitWidth == 32) {
                if (createISegment<uint32_t>(data, _builder, dataRef, xramAddress).failed()) {
                    return failure();
                }
            }
            else {
                llvm::errs() << "Unsupported bit width: " << bitWidth << "\n";
                return failure();
            }
        }
        else if (elementType.isBF16()) {
            if (createFSegment<uint16_t>(data, _builder, dataRef, xramAddress)
                    .failed()) { // Create FSegment with 2 byte data
                return failure();
            }
        }
        else if (elementType.isF32()) {
            if (createFSegment<uint32_t>(data, _builder, dataRef, xramAddress)
                    .failed()) { // Create FSegment with 4 bytes data
                return failure();
            }
        }
        else {
            llvm::errs() << "Unsupported data type\n";
            return failure();
        }

        LLVM_DEBUG({
            llvm::dbgs() << "*** Constant from splat: size = " << data.getNumElements() << " ***\n";
        });

        _segments.push_back(iree_hal_torq_Segment_create(_builder, xramAddress, dataRef));
    }
    else {

        LLVM_DEBUG({
            llvm::dbgs() << "*** Constant: size = " << data.getRawData().size() << " ***\n";
        });

        return addSegment(xramAddress, data, false);
    }

    return success();
}

LogicalResult Serializer::processSliceTaskOp(
    torq_hw::SliceTaskOp &taskOp, uint32_t &nextNdlLramBaseAddress,
    uint32_t &nextNdlXramBaseAddress, uint32_t &remainingNdlSize, uint32_t &remainingCfgSize,
    torq_hl::CreateInvocationOp &createInvocationOp
) {
    LLVM_DEBUG({ llvm::dbgs() << "*** layer type: " << taskOp->getName() << " ***\n"; });
    auto baseline_task = toSliceTask(taskOp, createInvocationOp);

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

LogicalResult Serializer::serializeNssInvocation(torq_hl::CreateInvocationOp &op) {

    auto programOp = op.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    if (!programOp) {
        return op->emitOpError("program is not a torq_hl::ProgramOp");
    }

    auto blockCount = programOp.getBody().getBlocks().size();

    if (blockCount == 0) {
        return op->emitOpError("NSS program must have at least one block");
    }

    if (!op.getXramCodeAddresses() || !op.getExecutorCodeAddresses()) {
        return op->emitOpError("NSS program must have XRAM and LRAM code addresses");
    }

    auto xramCodeAddresses = op.getXramCodeAddresses().value();
    auto lramCodeAddresses = op.getExecutorCodeAddresses().value();

    if (xramCodeAddresses.size() != blockCount) {
        return op->emitOpError(
            "Expected as many XRAM code address for NSS program as blocks in the program"
        );
    }

    if (lramCodeAddresses.size() != 1) {
        return op->emitOpError(
            "Expected as many LRAM code address for NSS program as blocks in the program"
        );
    }

    if (!_npu.nssBegin(lramCodeAddresses[0], xramCodeAddresses[0])) {
        return failure();
    }

    int64_t nextLramAddress = AddressConstants::APPEND;
    int64_t nextXramAddress = AddressConstants::APPEND;

    for (auto [idx, block] : llvm::enumerate(programOp.getBody().getBlocks())) {

        // reset the nss block size counter that is used in processNssTask
        _nssBlockSize = 0;

        for (auto &nestedOp : block.getOperations()) {

            if (isa<torq_hl::ReturnOp, torq_hl::NextOp, torq_hl::GetBlockOp, memref::AllocOp,
                    memref::DeallocOp>(nestedOp)) {
                continue;
            }

            if (isDerivedMemRefOperation(&nestedOp)) {
                continue;
            }

            auto nssTaskOp = dyn_cast<torq_hw::NssTaskOp>(nestedOp);

            if (!nssTaskOp) {
                return nestedOp.emitOpError("not supported in a NSS program");
            }

            if (failed(processNssTask(nssTaskOp, op, nextLramAddress, nextXramAddress))) {
                return failure();
            }

            // always append after the first iteration
            nextLramAddress = AddressConstants::APPEND;
            nextXramAddress = AddressConstants::APPEND;
        }

        // find the next LRAM address for the block from the next op
        if (auto nextOp = dyn_cast<torq_hl::NextOp>(block.getTerminator())) {
            auto maybeAddress = getAddress(nextOp.getLramArea(), 0, op.getInvocation());

            if (!maybeAddress) {
                return nextOp.emitError() << "Missing next LRAM address";
            }

            nextLramAddress = maybeAddress.value();

            // find the next XRAM address
            nextXramAddress = xramCodeAddresses[idx + 1];
        }
    }

    if (_npu.nssEnd() < 0) {
        return failure();
    }

    // create a file tv.cdesc_addr.txt with the lram address where the NPU should start, this is
    // not dumped by default in torq_api
    if (clTorqDescriptorDumpDir != "") {
        std::ofstream file(
            _dump_path + "/job" + std::to_string(_nssProgramCount) + "/tv.cdesc_addr.txt"
        );
        file << "0x" << std::setw(8) << std::setfill('0') << std::hex << lramCodeAddresses[0]
             << "\n";
        file.close();
    }

    _nssProgramCount++;

    return success();
}

LogicalResult Serializer::serializeInvocation(torq_hl::CreateInvocationOp &createInvocationOp) {

    switch (createInvocationOp.getProgram().getType().getExecutor()) {
    case torq_hl::Executor::Slice:
        return serializeSliceInvocation(createInvocationOp);
    case torq_hl::Executor::CSS:
        return serializeCssInvocation(createInvocationOp);
    case torq_hl::Executor::NSS:
        return serializeNssInvocation(createInvocationOp);
    case torq_hl::Executor::Host:
        return serializeHostInvocation(createInvocationOp);
    default:
        return createInvocationOp->emitOpError("unsupported executor type for invocation");
    }
}

LogicalResult
Serializer::addSegment(uint32_t xramAddress, mlir::DenseIntOrFPElementsAttr data, bool isCode) {

    auto dataRef = flatbuffers_uint8_vec_create(
        _builder, (const uint8_t *)data.getRawData().data(), data.getRawData().size()
    );

    if (!succeeded(dumpSegmentDescriptor(
            0, true, xramAddress, data.getRawData().data(), data.getRawData().size(), isCode
        ))) {
        return failure();
    };

    _segments.push_back(iree_hal_torq_Segment_create(_builder, xramAddress, dataRef));

    return success();
}

LogicalResult Serializer::serializeCssInvocation(torq_hl::CreateInvocationOp &createInvocationOp) {

    if (createInvocationOp.getCodeSections().size() != 2) {
        return createInvocationOp->emitOpError("Expected exactly two code section for CSS program");
    }

    // add a section with the code for the CSS call
    auto cssProgram = createInvocationOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();

    if (!cssProgram) {
        return createInvocationOp->emitOpError("program is not a torq_hl::CSSProgramOp");
    }

    auto cssExecutable =
        SymbolTable::lookupNearestSymbolFrom<iree_compiler::IREE::HAL::ExecutableOp>(
            cssProgram, cssProgram.getNameAttr()
        );

    if (!cssExecutable) {
        return cssProgram.emitError() << "CSS executable not found:" << cssProgram.getName();
    }

    auto executableBinaryOp =
        SymbolTable::lookupNearestSymbolFrom<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
            cssExecutable, StringAttr::get(cssProgram.getContext(), "code")
        );

    if (!executableBinaryOp) {
        return cssExecutable.emitError()
               << "Executable binary css_task in " << cssExecutable.getName() << " not found";
    }

    auto xramAddress = getXramAddress(createInvocationOp.getCodeSections()[0], 0);

    if (!xramAddress) {
        return createInvocationOp.emitError() << "Missing xram address for code section #0";
    }

    auto text = executableBinaryOp.getData();

    LLVM_DEBUG({ llvm::dbgs() << "*** CSS text: size = " << text.getRawData().size() << " ***\n"; }
    );

    if (failed(addSegment(xramAddress.value(), text, true))) {
        return failure();
    }

    // add a segment with the arguments of the CSS call

    SmallVector<uint32_t> args;

    if (auto maybeArgsAttr = createInvocationOp.getInvocationArgs()) {

        args.push_back(maybeArgsAttr->size());

        for (auto arg : *maybeArgsAttr) {
            auto argValue = dyn_cast<torq_hl::AddressAttr>(arg);
            if (!argValue) {
                return createInvocationOp.emitError() << "Expected AddressAttr in invocation args";
            }

            switch (argValue.getMemSpace()) {
            case torq_hl::MemorySpace::Dtcm:
                args.push_back(argValue.getAddress() + HwInfo::css_dtcm_base_address);
                break;
            case torq_hl::MemorySpace::Itcm:
                args.push_back(argValue.getAddress() + HwInfo::css_itcm_base_address);
                break;
            default:
                return createInvocationOp.emitError() << "Unsupported memory space in CSS arg";
            }
        }
    }
    else {
        args.push_back(0); // No arguments, just the size
    }

    auto argsType = MemRefType::get(
        {(int64_t)args.size()}, IntegerType::get(createInvocationOp.getContext(), 32)
    );
    auto argsAttr =
        DenseIntElementsAttr::get(argsType, llvm::ArrayRef<uint32_t>(args.data(), args.size()));

    auto argsXramAddress = getXramAddress(createInvocationOp.getCodeSections()[1], 0);

    if (!argsXramAddress) {
        return createInvocationOp.emitError() << "Missing xram address for code section #1";
    }

    LLVM_DEBUG({
        llvm::dbgs() << "*** CSS args: size = " << argsAttr.getRawData().size() << " ***\n";

        llvm::dbgs() << "args:";
        for (auto arg : args) {
            llvm::dbgs() << " " << arg << "\n";
        }
    });

    if (failed(addSegment(argsXramAddress.value(), argsAttr, false))) {
        return failure();
    }

    return success();
}

LogicalResult Serializer::serializeHostInvocation(torq_hl::CreateInvocationOp &createInvocationOp) {

    // for the host invocations we don't need to serialize anything. The code is loaded once for
    // all dispatches and the arguments are specified in the host start action itself

    return success();
}

LogicalResult Serializer::serializeSliceInvocation(torq_hl::CreateInvocationOp &createInvocationOp
) {

    // This function generates the bitstream for a task, this is stored only in
    // XRAM segments and will be loaded by the main NSS program

    // SliceID is needed only for logging purpose in order to dump the CFG descriptors in text
    // files, using the same id for all descriptors wouldn´t change the functionality.

    auto maybeId = createInvocationOp.getExecutorId();

    if (!maybeId) {
        return createInvocationOp->emitOpError("Missing executor id");
    }

    auto slc = (*maybeId).getZExtValue();

    auto maybeLramCodeAddresses = createInvocationOp.getExecutorCodeAddresses();

    if (!maybeLramCodeAddresses || maybeLramCodeAddresses->empty()) {
        return createInvocationOp->emitOpError("Missing lram code address");
    }

    auto maybeXramCodeAddresses = createInvocationOp.getXramCodeAddresses();

    if (!maybeXramCodeAddresses || maybeLramCodeAddresses->empty()) {
        return createInvocationOp->emitOpError("Missing xram code address");
    }

    auto lramAddress = (*maybeLramCodeAddresses)[0];
    auto xramAddress = (*maybeXramCodeAddresses)[0];

    if (!_npu.beginCfg(slc, lramAddress, xramAddress)) {
        return failure();
    }

    auto codeType = cast<MemRefType>(createInvocationOp.getCodeSections()[0].getType());

    uint32_t programSize = getEncodedTotalSizeBytes(codeType);

    // reserve the first 0x200 bytes of the program to store CFGs, the rest for the NDLS
    // this value is chosen so that at least one CFG can fit there (and given the current
    // hardcoded choice of program size the corresponding NDLs can fit in the remaining space)
    // TODO: this program size should be computed based on the actual operations contained
    // in the program
    uint32_t remainingCfgSpace = 0x900;

    uint32_t nextNdlXramBaseAddress = xramAddress + remainingCfgSpace;
    uint32_t nextNdlLramBaseAddress = lramAddress + remainingCfgSpace;

    uint32_t remainingNdlSpace = programSize - remainingCfgSpace;

    if (programSize <= remainingCfgSpace) {
        return createInvocationOp->emitOpError("Program size too small to fit CFGs instructions");
    }

    auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

    if (!programOp) {
        return createInvocationOp->emitOpError("program is not a torq_hl::ProgramOp");
    }

    for (auto &op : programOp.getOps()) {

        if (isa<torq_hw::SliceProfilingOp, torq_hw::GetAddressOp, torq_hl::ReturnOp>(op)) {
            continue;
        }

        auto taskOp = dyn_cast<torq_hw::SliceTaskOp>(op);

        if (!taskOp) {
            op.emitOpError("Unsupported operation in program for slice");
            return failure();
        }

        if (failed(processSliceTaskOp(
                taskOp, nextNdlLramBaseAddress, nextNdlXramBaseAddress, remainingNdlSpace,
                remainingCfgSpace, createInvocationOp
            ))) {
            return failure();
        }
    }
    if (_npu.endCfg() < 0) {
        return failure();
    }
    return success();
}

LogicalResult Serializer::processNssTask(
    torq_hw::NssTaskOp nssTaskOp, torq_hl::CreateInvocationOp &createInvocationOp, int lramAddress,
    int xramAddress
) {
    NssTask task;
    if (!toNssTask(nssTaskOp, task, createInvocationOp)) {
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

    _nssBlockSize += size;

    if (_nssBlockSize > getCodeSize(nssTaskOp->getBlock())) {

        llvm::errs() << "Overflow code size of block ( block size is already " << _nssBlockSize
                     << " while maximum is " << getCodeSize(nssTaskOp->getBlock()) << ")\n";

        llvm::errs() << "NSS task:\n";
        nssTaskOp->dump();

        llvm::errs() << "Block:\n";
        nssTaskOp->getBlock()->dump();

        return failure();
    }
    return success();
}

LogicalResult Serializer::dumpSegmentDescriptor(
    int jobId, bool xram, int address, const void *data, int size, bool isCode
) {

    if (_dump_path.empty())
        return success();

    std::string dump_dir = _dump_path + "/job" + std::to_string(jobId) + "/";

    if (!std::filesystem::exists(dump_dir)) {
        if (!std::filesystem::create_directory(dump_dir)) {
            llvm::errs() << "Failed to create directory: " << dump_dir << "\n";
            return failure();
        }
    }

    std::string filename;

    if (isCode) {
        filename += "code." + std::to_string(_codeSegmentDumpId) + ".txt";
        _codeSegmentDumpId++;
    }
    else {
        filename += "constant." + std::to_string(_constantSegmentDumpId) + ".txt";
        _constantSegmentDumpId++;
    }

    // dump the data as hex
    std::ofstream wf(dump_dir + filename);
    if (!wf.good()) {
        return failure();
    }

    if (data) {
        for (size_t i = 0; i < size; i++) {
            wf << std::hex << std::setw(2) << std::setfill('0') << (int)((uint8_t *)data)[i]
               << '\n';
        }
    }

    wf.close();
    if (!wf.good()) {
        return failure();
    }

    // append the file name to the list of files in the corresponding mem.lst file
    // we store it in a temporary variable because the dump file is still open by
    // torq_api

    std::ostringstream memListFile;

    memListFile << (xram ? "xload" : "load");
    memListFile << "  0x" << std::hex << std::setw(8) << std::setfill('0') << address;
    memListFile << " " << std::dec << std::setw(10) << std::setfill(' ') << size;
    memListFile << "   1  hex  " << filename << "\n";

    _segmentDumpInfo += memListFile.str();

    return success();
}

LogicalResult Serializer::saveCodeSegments() {

    // add all segments that were not saved yet to the segments list
    const torq_bitstream_segment_t *segment = _npu.getBitstream();

    while (segment) {

        if (!_savedSegments.contains(segment->data)) {

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

                stringstream ss;
                for (size_t i = 0; i < segment->size; i++) {
                    ss << std::hex << "" << (int)segment->data[i];
                }
                llvm::dbgs() << "    data: " << ss.str() << "\n";
            });

            auto dataRef = flatbuffers_uint8_vec_create(_builder, segment->data, segment->size);

            _segments.push_back(iree_hal_torq_Segment_create(_builder, segment->xram_addr, dataRef)
            );

            _savedSegments.insert(segment->data);

            if (!_dump_path.empty() && segment->xram_addr != AddressConstants::NONE) {
                if (!succeeded(dumpSegmentDescriptor(
                        0, true, segment->xram_addr, segment->data, segment->size, true
                    ))) {
                    return failure();
                }
            }
        }

        segment = segment->next;
    }

    return success();
}

flatbuffers_uint32_vec_ref_t Serializer::createUI32Vector(std::optional<ArrayRef<int64_t>> vec) {

    flatbuffers_uint32_vec_start(_builder);

    if (vec) {
        for (uint32_t v : vec.value()) {
            flatbuffers_uint32_vec_push_create(_builder, v);
        }
    }

    return flatbuffers_uint32_vec_end(_builder);
}

flatbuffers_ref_t Serializer::createLocationString(Location loc) {

    std::string v;
    llvm::raw_string_ostream os(v);
    loc.print(os);
    os.flush();

    return flatbuffers_string_create(_builder, v.data(), v.size());
}

flatbuffers_string_vec_ref_t Serializer::createLocationsVector(ArrayRef<Location> locations) {

    SmallVector<flatbuffers_ref_t> refsVec;

    for (auto loc : locations) {
        refsVec.push_back(createLocationString(loc));
    }

    flatbuffers_string_vec_start(_builder);

    for (auto v : refsVec) {
        flatbuffers_string_vec_push(_builder, v);
    }

    return flatbuffers_string_vec_end(_builder);
}

static iree_hal_torq_BufferType_enum_t toBufferType(torq_hl::MemorySpace memorySpace) {
    switch (memorySpace) {
    case torq_hl::MemorySpace::Xram:
        return iree_hal_torq_BufferType_XRAM;
    case torq_hl::MemorySpace::Lram:
        return iree_hal_torq_BufferType_LRAM;
    case torq_hl::MemorySpace::Dtcm:
        return iree_hal_torq_BufferType_DTCM;
    case torq_hl::MemorySpace::Itcm:
        return iree_hal_torq_BufferType_ITCM;
    default:
        return iree_hal_torq_BufferType_OTHER;
    }
}

iree_hal_torq_BufferDebugInfo_ref_t Serializer::createBufferDebugInfo(
    int64_t bufferId, uint64_t allocationAction, uint64_t lastUseAction,
    uint64_t deallocationAction, TypedValue<MemRefType> buffer
) {

    auto memorySpace = getEncodingMemorySpace(buffer.getType());

    iree_hal_torq_BufferType_enum_t bufferType = toBufferType(memorySpace);
    auto maybeAddress = getDataStartAddress(buffer);

    uint32_t address = maybeAddress.value_or(-1);

    int32_t size = getEncodedTotalSizeBytes(buffer.getType());
    auto shape = createUI32Vector(buffer.getType().getShape());
    auto strides = createUI32Vector(getEncodedStridesElements(buffer.getType()));

    iree_hal_torq_ElementType_enum_t elementType;

    if (buffer.getType().getElementType().isInteger(8)) {
        elementType = iree_hal_torq_ElementType_I8;
    }
    else if (buffer.getType().getElementType().isInteger(16)) {
        elementType = iree_hal_torq_ElementType_I16;
    }
    else if (buffer.getType().getElementType().isInteger(32)) {
        elementType = iree_hal_torq_ElementType_I32;
    }
    else if (buffer.getType().getElementType().isBF16()) {
        elementType = iree_hal_torq_ElementType_BF16;
    }
    else if (buffer.getType().getElementType().isF32()) {
        elementType = iree_hal_torq_ElementType_F32;
    }
    else {
        elementType = iree_hal_torq_ElementType_OTHER;
    }

    return iree_hal_torq_BufferDebugInfo_create(
        _builder, bufferId, allocationAction, deallocationAction, lastUseAction, bufferType,
        address, size, shape, elementType, strides
    );
}

flatbuffers_vec_ref_t Serializer::createBuffersDebugInfoVector(mlir::FunctionOpInterface funcOp) {

    SmallVector<iree_hal_torq_BufferDebugInfo_ref_t> debugInfoVec;

    for (auto &op : funcOp.getFunctionBody().getOps()) {

        auto bufferIds = op.getAttrOfType<DenseI64ArrayAttr>("torq-buffer-ids");

        if (!bufferIds) {
            continue;
        }

        auto definingActionId = 0;
        Operation *definingOpAction = &op;

        do {
            if (auto maybeDefiningActionId =
                    definingOpAction->getAttrOfType<IntegerAttr>("torq-action-id")) {
                definingActionId = maybeDefiningActionId.getInt();
                break;
            }
            definingOpAction = definingOpAction->getPrevNode();
        } while (definingOpAction);

        int idx = 0;

        int64_t deallocationId = -1;

        for (auto result : op.getResults()) {

            auto memref = dyn_cast<TypedValue<MemRefType>>(result);

            if (!memref) {
                continue; // skip non-memref results
            }

            int64_t lastUseId = definingActionId;

            for (auto &use : memref.getUses()) {
                auto maybeActionId = use.getOwner()->getAttrOfType<IntegerAttr>("torq-action-id");

                if (!maybeActionId) {
                    continue;
                }

                if (auto deallocOp = dyn_cast<memref::DeallocOp>(use.getOwner())) {
                    deallocationId = maybeActionId.getInt();
                }
                else {
                    lastUseId = std::max(lastUseId, maybeActionId.getInt());
                }
            }

            if (memref.getType().getElementType().isIndex()) {
                // Indexes used for Reshape operations are not real buffers
                continue;
            }

            debugInfoVec.push_back(createBufferDebugInfo(
                bufferIds[idx], definingActionId, lastUseId, deallocationId, memref
            ));

            idx++;
        }
    }

    return flatcc_builder_create_offset_vector_direct(
        _builder, debugInfoVec.data(), debugInfoVec.size()
    );
}

FailureOr<flatbuffers_uint8_vec_ref_t>
Serializer::serializeHostCode(mlir::FunctionOpInterface funcOp) {

    auto hostCodeExecutableOp =
        SymbolTable::lookupNearestSymbolFrom<iree_compiler::IREE::HAL::ExecutableOp>(
            funcOp.getOperation(), StringAttr::get(funcOp.getContext(), "host_code")
        );

    // no host code in this dispatch
    if (!hostCodeExecutableOp) {
        return flatbuffers_uint8_vec_create(_builder, nullptr, 0);
    }

    if (clTorqDescriptorDumpDir != "") {
        llvm::errs(
        ) << "This model requires host programs that are not supported in descriptor dumps\n";
        return failure();
    }

    auto hostCodeExecutableBinaryOp =
        SymbolTable::lookupNearestSymbolFrom<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
            hostCodeExecutableOp, StringAttr::get(funcOp.getContext(), "code")
        );

    if (!hostCodeExecutableBinaryOp) {
        return hostCodeExecutableOp.emitError("Missing host_code binary in host_code executable");
    }

    auto data = hostCodeExecutableBinaryOp.getData().getRawData();

    return flatbuffers_uint8_vec_create(_builder, (const uint8_t *)data.data(), data.size());
}

FailureOr<flatbuffers_vec_ref_t>
Serializer::serializeRuntimeProgram(mlir::FunctionOpInterface funcOp) {

    DenseMap<Value, uint64_t> allocIdMap;

    SmallVector<iree_hal_torq_HostAction_ref_t> actions;
    for (auto &op : funcOp.getFunctionBody().getOps()) {

        // add the memref result ids to the map
        if (auto bufferIds = op.getAttrOfType<DenseI64ArrayAttr>("torq-buffer-ids")) {
            int idx = 0;
            for (auto result : op.getResults()) {
                if (auto memrefType = dyn_cast<MemRefType>(result.getType())) {
                    allocIdMap[result] = bufferIds[idx];
                    idx++;
                }
            }
        }

        if (isa<torq_hl::ProgramOp, torq_hl::CreateInvocationOp, torq_hl::ConstOp,
                torq_hl::MapBindingOp, func::ReturnOp, torq_hl::ImportProgramOp,
                torq_hl::GetBlockOp, torq_hw::DispatchProfilingOp>(op) ||
            isDerivedMemRefOperation(&op)) {
            continue; // skip these ops
        }

        iree_hal_torq_HostActionParams_union_ref_t params;

        if (auto hostActionOp = dyn_cast<torq_hl::HostCopyOp>(op)) {

            auto inputMemSpace = getEncodingMemorySpace(hostActionOp.getInput().getType());
            auto outputMemSpace = getEncodingMemorySpace(hostActionOp.getOutput().getType());

            auto inputAddress = getDataStartAddress(hostActionOp.getInput());
            if (!inputAddress) {
                op.emitOpError("Input buffer address is not set");
                return failure();
            }

            auto outputAddress = getDataStartAddress(hostActionOp.getOutput());
            if (!outputAddress) {
                op.emitOpError("Output buffer address is not set");
                return failure();
            }

            auto hostCopyParams = iree_hal_torq_HostCopyParams_create(
                _builder, toBufferType(inputMemSpace), toBufferType(outputMemSpace),
                inputAddress.value(), outputAddress.value(),
                createUI32Vector(hostActionOp.getInputStridesBytes()),
                createUI32Vector(hostActionOp.getOutputStridesBytes()),
                createUI32Vector(hostActionOp.getShape()), hostActionOp.getElementSizeBytes()

            );

            params = iree_hal_torq_HostActionParams_as_HostCopyParams(hostCopyParams);
        }
        else if (auto startOp = dyn_cast<torq_hl::StartProgramOp>(op)) {

            auto createInvocationOp =
                startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

            if (!createInvocationOp) {
                op.emitOpError("must start an invocation created with create_invocation");
                return failure();
            }

            if (startOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS) {

                auto programOp =
                    createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

                if (!programOp) {
                    op.emitOpError("program is not a torq_hl::ProgramOp");
                    return failure();
                }

                auto maybeCodeAddress = getLramAddress(startOp.getCodeSections()[0]);

                if (!maybeCodeAddress) {
                    op.emitOpError("Missing lram code address for start program");
                    return failure();
                }

                bool startsSlice1 = false;
                bool startsSlice2 = false;
                bool startsDmaOut = false;
                bool startsDmaIn = false;

                for (auto op : programOp.getBody().getOps<torq_hw::NssTaskOp>()) {
                    for (auto &innerOp : op.getBody().getOps()) {
                        if (auto dmaInOp = dyn_cast<torq_hw::DmaInStartOp>(innerOp)) {
                            startsDmaIn = true;
                        }
                        else if (auto dmaOutOp = dyn_cast<torq_hw::DmaOutStartOp>(innerOp)) {
                            startsDmaOut = true;
                        }
                        else if (auto sliceOp = dyn_cast<torq_hw::SliceStartOp>(innerOp)) {
                            if (sliceOp.getId() == 1) {
                                startsSlice1 = true;
                            }
                            else if (sliceOp.getId() == 2) {
                                startsSlice2 = true;
                            }
                        }
                    }
                }

                auto startNssParams = iree_hal_torq_StartNSSParams_create(
                    _builder, *maybeCodeAddress, startsSlice1, startsSlice2, startsDmaIn,
                    startsDmaOut
                );

                params = iree_hal_torq_HostActionParams_as_StartNSSParams(startNssParams);
            }
            else if (startOp.getInvocation().getType().getExecutor() == torq_hl::Executor::Host) {

                auto importProgramOp =
                    createInvocationOp.getProgram().getDefiningOp<torq_hl::ImportProgramOp>();

                if (!importProgramOp) {
                    return op.emitOpError("program is not a torq_hl::ImportProgramOp");
                }

                auto programName = _builder.createString(importProgramOp.getName());

                assert(createInvocationOp.getInvocationArgs().has_value());

                auto args = createInvocationOp.getInvocationArgs().value();

                flatbuffers_uint64_vec_start(_builder);
                for (auto arg : args) {

                    auto argValue = dyn_cast<torq_hl::AddressAttr>(arg);
                    if (!argValue) {
                        return startOp.emitError("expected AddressAttr arguments only");
                    }

                    if (argValue.getMemSpace() != torq_hl::MemorySpace::Xram) {
                        return startOp.emitError("expected XRAM memory space for host arguments");
                    }

                    flatbuffers_uint64_vec_push_create(_builder, argValue.getAddress());
                }
                auto argsRef = flatbuffers_uint64_vec_end(_builder);

                flatbuffers_uint64_vec_start(_builder);
                for (auto arg : startOp.getArgs()) {
                    auto memrefType = dyn_cast<MemRefType>(arg.getType());
                    if (!memrefType) {
                        return startOp.emitError("expected memref arguments only");
                    }
                    auto memrefSize = getEncodedTotalSizeBytes(memrefType);
                    flatbuffers_uint64_vec_push_create(_builder, memrefSize);
                }
                auto sizesRef = flatbuffers_uint64_vec_end(_builder);

                auto startHostParams =
                    iree_hal_torq_StartHostParams_create(_builder, programName, argsRef, sizesRef);

                params = iree_hal_torq_HostActionParams_as_StartHostParams(startHostParams);
            }
            else {
                return op.emitError("Unsupported executor");
            }
        }
        else if (auto waitOp = dyn_cast<torq_hl::WaitProgramOp>(op)) {

            auto maybeBufferIds = waitOp->getAttrOfType<DenseI64ArrayAttr>("torq-buffer-ids");

            auto bufferIds =
                createUI32Vector(maybeBufferIds ? maybeBufferIds.asArrayRef() : std::nullopt);

            if (waitOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS) {
                auto waitNssParams = iree_hal_torq_WaitNSSParams_create(_builder, bufferIds);
                params = iree_hal_torq_HostActionParams_as_WaitNSSParams(waitNssParams);
            }
            else if (waitOp.getInvocation().getType().getExecutor() == torq_hl::Executor::Host) {
                auto waitHostParams = iree_hal_torq_WaitHostParams_create(_builder, bufferIds);
                params = iree_hal_torq_HostActionParams_as_WaitHostParams(waitHostParams);
            }
            else {
                return op.emitError("Unsupported executor");
            }
        }
        else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
            auto maybeAddress = getAddress(allocOp.getResult());

            if (!maybeAddress) {
                return op.emitOpError("Missing xram address for alloc operation");
            }

            auto memSpace = getEncodingMemorySpace(allocOp.getType());
            auto size = getEncodedTotalSizeBytes(allocOp.getResult().getType());

            auto allocParams = iree_hal_torq_AllocParams_create(
                _builder, allocIdMap[allocOp.getMemref()], maybeAddress.value(), size,
                toBufferType(memSpace)
            );

            params = iree_hal_torq_HostActionParams_as_AllocParams(allocParams);
        }
        else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {

            auto maybeId = allocIdMap.find(deallocOp.getOperand());

            if (maybeId == allocIdMap.end()) {
                return op.emitOpError("Deallocating a buffer that was not allocated");
            }

            auto deallocParams = iree_hal_torq_DeallocParams_create(_builder, maybeId->second);

            params = iree_hal_torq_HostActionParams_as_DeallocParams(deallocParams);
        }
        else if (auto constOp = dyn_cast<arith::ConstantOp>(op);
                 constOp && isa<IndexType>(cast<ShapedType>(constOp.getType()).getElementType())) {
            // Ignore constants with element type of IndexType
            continue;
        }
        else if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op);
                 toMemrefOp && isa<arith::ConstantOp>(toMemrefOp.getTensor().getDefiningOp())) {
            // Ignore ToMemrefOp whose source is an arith.constant
            continue;
        }
        else {
            // unsupported operation
            return op.emitOpError("unsupported in function body");
        }

        auto action =
            iree_hal_torq_HostAction_create(_builder, params, createLocationString(op.getLoc()));

        actions.push_back(action);
    }

    return flatcc_builder_create_offset_vector_direct(_builder, actions.data(), actions.size());
}

LogicalResult Serializer::serializeFunction(mlir::FunctionOpInterface funcOp) {

    LLVM_DEBUG({ llvm::dbgs() << "---- serializing programs\n"; });
    for (auto op : funcOp.getFunctionBody().getOps<torq_hl::CreateInvocationOp>()) {
        if (failed(serializeInvocation(op))) {
            return failure();
        }
    }

    LLVM_DEBUG({ llvm::dbgs() << "---- saving xram segments\n"; });
    if (!succeeded(saveCodeSegments())) {
        return failure();
    };

    LLVM_DEBUG({ llvm::dbgs() << "---- serializing constants\n"; });
    for (auto constOp : funcOp.getFunctionBody().getOps<torq_hl::ConstOp>()) {
        if (failed(processConstOp(constOp))) {
            return failure();
        }
    }

    auto maybeRuntimeProgram = serializeRuntimeProgram(funcOp);

    if (failed(maybeRuntimeProgram)) {
        return failure();
    }

    LLVM_DEBUG({ llvm::dbgs() << "---- serializing binding data\n"; });
    SmallVector<iree_hal_torq_Binding_ref_t> bindingRefs;
    funcOp.walk([&](syna::torq_hl::MapBindingOp mapBindingOp) {
        auto sizeBytes = getEncodedTotalSizeBytes(mapBindingOp.getResult().getType());

        bindingRefs.push_back(iree_hal_torq_Binding_create(
            _builder, mapBindingOp.getBindingIndex().getZExtValue(),
            getXramAddress(mapBindingOp.getOperation()).value(),
            mapBindingOp.getOffset().getZExtValue(), sizeBytes, mapBindingOp.getIsReadOnly(),
            mapBindingOp.getIsWriteOnly()
        ));
    });

    // close the _torq object so that all the open file descriptors are released
    _npu.close();

    // write out the list of extra segments that need to be loaded in job0
    if (!_dump_path.empty()) {

        std::ofstream memListFile(_dump_path + "/job0/tv.init.mem.lst", std::ios_base::app);

        if (!memListFile.good()) {
            return failure();
        }

        memListFile << _segmentDumpInfo;

        memListFile.close();

        if (!memListFile.good()) {
            return failure();
        }
    }

    // creates the top-level object
    auto codeRef =
        flatcc_builder_create_offset_vector_direct(_builder, _segments.data(), _segments.size());
    auto bindingRef = flatcc_builder_create_offset_vector_direct(
        _builder, bindingRefs.data(), bindingRefs.size()
    );
    auto executableName = _builder.createString(getDispatchName(funcOp));

    flatbuffers_vec_ref_t bufferDebugInfo = 0;

    if (clEnableBufferDebugInfo) {
        bufferDebugInfo = createBuffersDebugInfoVector(funcOp);
    }
    else {
        bufferDebugInfo = iree_hal_torq_BufferDebugInfo_vec_create(_builder, nullptr, 0);
    }

    // find the xram used by this program
    int xramMin = std::numeric_limits<int>::max();
    int xramMax = 0;

    for (auto &op : funcOp.getFunctionBody().getOps()) {
        for (auto result : op.getResults()) {
            if (auto buffer = dyn_cast<TypedValue<MemRefType>>(result)) {
                auto maybeAddress = getXramAddress(buffer);
                if (maybeAddress) {
                    xramMin = std::min(xramMin, (int)maybeAddress.value());
                    xramMax = std::max(
                        xramMax,
                        (int)maybeAddress.value() + (int)getEncodedTotalSizeBytes(buffer.getType())
                    );
                }
            }
        }
    }

    if (xramMin == std::numeric_limits<int>::max()) {
        xramMin = 0; // no XRAM used, set to 0
    }

    auto maybeHostCode = serializeHostCode(funcOp);

    if (failed(maybeHostCode)) {
        return failure();
    }

    iree_hal_torq_ExecutableDef_create_as_root(
        _builder, executableName, codeRef, bindingRef, *maybeRuntimeProgram, bufferDebugInfo,
        *maybeHostCode
    );

    return success();
}

} // namespace mlir::syna::torq
