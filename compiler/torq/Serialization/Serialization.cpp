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
    LogicalResult
    addSegment(uint32_t xramAddress, mlir::DenseIntOrFPElementsAttr data, bool isCode);

    LogicalResult addUninitializedSegment(uint32_t xramAddress, int size);

    FailureOr<flatbuffers_vec_ref_t> serializeRuntimeProgram(FunctionOpInterface funcOp);

    LogicalResult serializeInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeDescriptor(torq_hl::DescriptorOp op);

    LogicalResult serializeCssInvocation(torq_hl::CreateInvocationOp &op);

    LogicalResult serializeHostInvocation(torq_hl::CreateInvocationOp &op);

    FailureOr<flatbuffers_uint8_vec_ref_t> serializeHostCode(mlir::FunctionOpInterface funcOp);

    LogicalResult processConstOp(torq_hl::ConstOp &op);

    LogicalResult processGlobalOp(memref::GlobalOp &globalOp);

    flatbuffers_uint32_vec_ref_t createUI32Vector(std::optional<ArrayRef<int64_t>> vec);

    iree_hal_torq_BufferDebugInfo_ref_t createBufferDebugInfo(
        int64_t bufferId, uint64_t allocationAction, uint64_t lastUseAction,
        uint64_t deallocationAction, TypedValue<MemRefType> buffer
    );

    flatbuffers_vec_ref_t createBuffersDebugInfoVector(mlir::FunctionOpInterface funcOp);

    // add the given segment to the segments loaded by jobX in the hardware test vector dump
    // this is required to ensure all buffers (including constants and code segments) required by
    // the program are correctly loaded (torq_api will only dump things in the current job)
    LogicalResult dumpSegmentDescriptor(
        int jobId, bool xram, int address, const void *data, int size, bool isCode
    );

    iree_compiler::FlatbufferBuilder &_builder;
    SmallVector<iree_hal_torq_Segment_ref_t> _segments;

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

Serializer::Serializer(iree_compiler::FlatbufferBuilder &builder, std::string dump_path)
    : _builder(builder), _dump_path(dump_path) {}

LogicalResult Serializer::processGlobalOp(memref::GlobalOp &globalOp) {

    auto maybeXramAddress = getXramAddress(globalOp.getOperation());

    // if there is no xram address we don't need to serialize this global,
    // for the moment LRAM/DTCM/ITCM buffers are not described in the model file
    // (except in the debug information)
    if (!maybeXramAddress) {
        return success();
    }

    auto xramAddress = maybeXramAddress.value();

    if (globalOp.isUninitialized()) {
        return addUninitializedSegment(xramAddress, getEncodedTotalSizeBytes(globalOp.getType()));
    }
    else {

        auto maybeData = globalOp.getInitialValue();

        if (!maybeData) {
            return globalOp.emitError("Global is initialized but has no initial value");
        }

        return addSegment(xramAddress, cast<DenseIntOrFPElementsAttr>(*maybeData), false);
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

        auto dataSize = bitWidth / 8 * data.getNumElements();

        _segments.push_back(iree_hal_torq_Segment_create(_builder, xramAddress, dataSize, dataRef));
    }
    else {

        LLVM_DEBUG({
            llvm::dbgs() << "*** Constant: size = " << data.getRawData().size() << " ***\n";
        });

        return addSegment(xramAddress, data, false);
    }

    return success();
}

LogicalResult Serializer::serializeInvocation(torq_hl::CreateInvocationOp &createInvocationOp) {

    switch (createInvocationOp.getProgram().getType().getExecutor()) {
    case torq_hl::Executor::CSS:
        return serializeCssInvocation(createInvocationOp);
    case torq_hl::Executor::Host:
        return serializeHostInvocation(createInvocationOp);
    default:
        return createInvocationOp->emitOpError("unsupported executor type for invocation");
    }
}

LogicalResult
Serializer::addSegment(uint32_t xramAddress, mlir::DenseIntOrFPElementsAttr data, bool isCode) {

    auto size = data.getRawData().size();

    auto dataRef =
        flatbuffers_uint8_vec_create(_builder, (const uint8_t *)data.getRawData().data(), size);

    if (!succeeded(
            dumpSegmentDescriptor(0, true, xramAddress, data.getRawData().data(), size, isCode)
        )) {
        return failure();
    };

    _segments.push_back(iree_hal_torq_Segment_create(_builder, xramAddress, size, dataRef));

    return success();
}

LogicalResult Serializer::addUninitializedSegment(uint32_t xramAddress, int size) {

    if (iree_hal_torq_Segment_start(_builder) ||
        iree_hal_torq_Segment_xram_address_add(_builder, xramAddress) ||
        iree_hal_torq_Segment_size_add(_builder, size)) {
        return failure();
    }

    auto segmentRef = iree_hal_torq_Segment_end(_builder);

    _segments.push_back(segmentRef);

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
            auto argValue = dyn_cast<torq_hl::BufferAttr>(arg);
            if (!argValue) {
                return createInvocationOp.emitError() << "Expected BufferAttr in invocation args";
            }

            auto memSpace = getEncodingMemorySpace(argValue.getMemrefType());

            switch (memSpace) {
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

LogicalResult Serializer::serializeDescriptor(torq_hl::DescriptorOp descriptorOp) {

    for (auto [codeAddr, codeAttr] :
         llvm::zip(descriptorOp.getXramCodeAddresses(), descriptorOp.getCodeData())) {
        auto codeData = cast<DenseI8ArrayAttr>(codeAttr);

        auto codeDataRef = codeData.getRawData();

        const uint8_t *codePtr = reinterpret_cast<const uint8_t *>(codeDataRef.data());

        auto dataRef = flatbuffers_uint8_vec_create(_builder, codePtr, codeDataRef.size());

        _segments.push_back(
            iree_hal_torq_Segment_create(_builder, codeAddr, codeDataRef.size(), dataRef)
        );

        if (!_dump_path.empty()) {
            if (!succeeded(dumpSegmentDescriptor(
                    0, true, codeAddr, codeDataRef.data(), codeDataRef.size(), true
                ))) {
                return failure();
            }
        }
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

flatbuffers_uint32_vec_ref_t Serializer::createUI32Vector(std::optional<ArrayRef<int64_t>> vec) {

    flatbuffers_uint32_vec_start(_builder);

    if (vec) {
        for (uint32_t v : vec.value()) {
            flatbuffers_uint32_vec_push_create(_builder, v);
        }
    }

    return flatbuffers_uint32_vec_end(_builder);
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

            // Use a worklist to follow uses transitively through
            // memref.subview and other view-like ops that don't carry
            // a torq-action-id themselves.
            SmallVector<Value> worklist;
            worklist.push_back(memref);

            while (!worklist.empty()) {
                Value current = worklist.pop_back_val();
                for (auto &use : current.getUses()) {
                    Operation *owner = use.getOwner();
                    auto maybeActionId = owner->getAttrOfType<IntegerAttr>("torq-action-id");

                    if (maybeActionId) {
                        if (isa<memref::DeallocOp>(owner)) {
                            deallocationId = maybeActionId.getInt();
                        }
                        else {
                            lastUseId = std::max(lastUseId, maybeActionId.getInt());
                        }
                    }
                    else if (isa<memref::SubViewOp>(owner)) {
                        // Follow through the subview to its users
                        for (auto result : owner->getResults()) {
                            worklist.push_back(result);
                        }
                    }
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
                torq_hl::DescriptorOp, torq_hl::MapBindingOp, func::ReturnOp,
                torq_hl::ImportProgramOp, torq_hl::GetBlockOp, torq_hw::DispatchProfilingOp,
                memref::GetGlobalOp>(op) ||
            isDerivedMemRefOperation(&op)) {
            continue; // skip these ops
        }

        if (auto constOp = dyn_cast<arith::ConstantOp>(op);
            constOp && isa<IndexType>(cast<ShapedType>(constOp.getType()).getElementType())) {
            // Ignore constants with element type of IndexType
            continue;
        }
        else if (auto toMemrefOp = dyn_cast<bufferization::ToMemrefOp>(op);
                 toMemrefOp && isa<arith::ConstantOp>(toMemrefOp.getTensor().getDefiningOp())) {
            // Ignore ToMemrefOp whose source is an arith.constant
            continue;
        }

        iree_hal_torq_HostActionParams_union_ref_t params;

        if (auto actionIdAttr = op.getAttrOfType<IntegerAttr>("torq-action-id")) {
            if (actionIdAttr.getInt() != actions.size()) {
                return op.emitOpError("Action IDs must be sequential starting from 0");
            }
        }
        else {
            return op.emitOpError("All actions must have a torq-action-id attribute");
        }

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

            if (startOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS) {

                auto descriptorOp = startOp.getInvocation().getDefiningOp<torq_hl::DescriptorOp>();

                if (!descriptorOp) {
                    op.emitOpError("must start an invocation created with descriptor");
                    return failure();
                }

                auto programOp = descriptorOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

                if (!programOp) {
                    op.emitOpError("program is not a torq_hl::ProgramOp");
                    return failure();
                }

                auto maybeCodeAddress = getAddress(startOp.getCodeSections()[0]);

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

                auto createInvocationOp =
                    startOp.getInvocation().getDefiningOp<torq_hl::CreateInvocationOp>();

                if (!createInvocationOp) {
                    op.emitOpError("must start an invocation created with create_invocation");
                    return failure();
                }

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

                    auto argValue = dyn_cast<torq_hl::BufferAttr>(arg);
                    if (!argValue) {
                        return startOp.emitError("expected BufferAttr arguments only");
                    }

                    auto memSpace = getEncodingMemorySpace(argValue.getMemrefType());

                    if (memSpace != torq_hl::MemorySpace::Xram) {
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

                // Serialize per-arg strides (bytes), shapes, and ndims from the
                // BufferAttr memref types which carry the actual XRAM layout
                // (including strided subview information).
                SmallVector<int64_t> allStrides, allShapes, allNdims;
                for (auto arg : args) {
                    auto argValue = cast<torq_hl::BufferAttr>(arg);
                    auto memrefType = argValue.getMemrefType();
                    auto stridesBytes = getEncodedStridesBytes(memrefType);
                    allStrides.append(stridesBytes);
                    allShapes.append(SmallVector<int64_t>(memrefType.getShape()));
                    allNdims.push_back(memrefType.getRank());
                }
                auto stridesRef = createUI32Vector(ArrayRef<int64_t>(allStrides));
                auto shapesRef = createUI32Vector(ArrayRef<int64_t>(allShapes));
                auto ndimsRef = createUI32Vector(ArrayRef<int64_t>(allNdims));

                auto startHostParams = iree_hal_torq_StartHostParams_create(
                    _builder, programName, argsRef, sizesRef, stridesRef, shapesRef, ndimsRef
                );

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
        else {
            // unsupported operation
            return op.emitOpError("unsupported in function body");
        }

        auto action = iree_hal_torq_HostAction_create(_builder, params);

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

    for (auto op : funcOp.getFunctionBody().getOps<torq_hl::DescriptorOp>()) {
        if (failed(serializeDescriptor(op))) {
            return failure();
        }
    }

    LLVM_DEBUG({ llvm::dbgs() << "---- serializing constants\n"; });
    for (auto constOp : funcOp.getFunctionBody().getOps<torq_hl::ConstOp>()) {
        if (failed(processConstOp(constOp))) {
            return failure();
        }
    }

    LLVM_DEBUG({ llvm::dbgs() << "---- serializing uninitalized sections\n"; });
    for (auto globalOp :
         funcOp.getOperation()->getParentOfType<ModuleOp>().getOps<memref::GlobalOp>()) {
        if (failed(processGlobalOp(globalOp))) {
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
