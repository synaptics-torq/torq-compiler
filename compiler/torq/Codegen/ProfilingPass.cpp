// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Transforms/TorqHL/Passes.h"
#include "torq/Transforms/TorqHL/PassesDetail.h"
#include "torq/Utils/DmaProfilingUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <fstream>
#include <map>
#include <tuple>

#define DEBUG_TYPE "torq-profiling"

static llvm::cl::opt<std::string> clTorqProfilingDump(
    "torq-dump-profiling", llvm::cl::desc("Dump profiling information to specified path"),
    llvm::cl::init("")
);

using namespace std;

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Defined in CompileNSSInvocationsPass.cpp; set via --torq-dma-in-mtu / --torq-dma-out-mtu.
extern llvm::cl::opt<unsigned> clDmaInMtu;
extern llvm::cl::opt<unsigned> clDmaOutMtu;

namespace {

enum class DmaType { In, Out, CdmaLramToDtcm, CdmaLramToItcm, CdmaDtcmToLram };

typedef uint64_t StartTime;
typedef uint64_t EndTime;

struct DmaEvent {
    int taskId;
    DmaType dmaType;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
    uint64_t bytes;
};

struct SliceEvent {
    int taskId;
    int id;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
};

struct CSSEvent {
    int taskId;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
};

typedef std::vector<DmaEvent> DmaTimeline;
typedef std::vector<SliceEvent> SliceTimeline;
typedef std::vector<CSSEvent> CSSTimeline;

struct ProfStruct {

    uint64_t timestamp{0};

    Attribute lastActionTimestampAttr;

    DmaTimeline dmaTimeline;
    SliceTimeline sliceTimeline;
    CSSTimeline cssTimeline;

    uint64_t currentDmaInBytes{0};
    uint64_t currentDmaOutBytes{0};

    int nextTaskIndex{0};

    void addToSliceTimeline(int taskId, int id, uint64_t start, uint64_t end, std::string loc) {
        sliceTimeline.push_back({taskId, id, start, end, loc}
        ); // loc+1 to match with the original line number in dump
    }

    int getLastSliceTime(int id) {
        for (auto it = sliceTimeline.rbegin(); it != sliceTimeline.rend(); ++it) {
            if (it->id == id) {
                return it->endTime;
            }
        }
        return 0;
    }

    void addToCSSTimeline(int taskId, uint64_t start, uint64_t end, std::string loc) {
        cssTimeline.push_back({taskId, start, end, loc});
    }

    int getLastCSSTime() {
        if (cssTimeline.empty()) {
            return 0;
        }
        return cssTimeline.back().endTime;
    }

    void addToDmaTimeline(
        int taskId, DmaType dmaType, uint64_t start, uint64_t end, std::string loc,
        uint64_t bytes = 0
    ) {
        dmaTimeline.push_back({taskId, dmaType, start, end, loc, bytes}
        ); // loc+1 to match with the original line number in dump
    }

    int getLastDmaTime(DmaType dmaType) {
        for (auto it = dmaTimeline.rbegin(); it != dmaTimeline.rend(); ++it) {
            if (it->dmaType == dmaType) {
                return it->endTime;
            }
        }
        return 0;
    }
};

static std::string toString(Location loc) {
    std::string locStr;
    llvm::raw_string_ostream os(locStr);
    loc.print(os);
    return locStr;
}

class ProfilingPass : public impl::ProfilingBase<ProfilingPass> {
  public:
    using ProfilingBase<ProfilingPass>::ProfilingBase;
    void runOnOperation() override;

    LogicalResult memProfiling(mlir::FunctionOpInterface funcOp);
    LogicalResult cycleProfiling(mlir::FunctionOpInterface funcOp);
    LogicalResult timeProfiling(mlir::Operation *funcOp);

  private:
    size_t totalMemSize{0};
    std::string memSummary{};
    size_t totalCycle{0};
};

LogicalResult ProfilingPass::memProfiling(mlir::FunctionOpInterface funcOp) {

    std::ostringstream shape_sstr;

    funcOp.walk([&](memref::AllocOp allocOp) {
        // Skip if not LRAM
        if (getEncodingMemorySpace(allocOp.getType()) != syna::torq_hl::MemorySpace::Lram) {
            return WalkResult::advance();
        }
        auto memreftype = allocOp.getType();
        int memssize = getEncodedTotalSizeBytes(memreftype);
        auto shape = memreftype.getShape();
        totalMemSize += memssize;
        for (size_t i = 0; i < shape.size(); i++) {
            shape_sstr << shape[i];
            if (i < shape.size() - 1)
                shape_sstr << "x";
        }
        shape_sstr << "+";

        return WalkResult::advance();
    });

    memSummary = shape_sstr.str();
    if (!memSummary.empty())
        memSummary.pop_back(); // remove the last '+'

    return success();
}

LogicalResult ProfilingPass::cycleProfiling(mlir::FunctionOpInterface funcOp) {
    // Slice-task cycle/memory annotation is now handled by
    // torq-annotate-dma-and-slice-cycles. This pass only consumes those ops.
    funcOp.walk([&](torq_hw::SliceProfilingOp profOp) {
        totalCycle = std::max<size_t>(totalCycle, profOp.getMaxCycle());
        return WalkResult::advance();
    });

    return success();
}

static LogicalResult
processOperationTime(Operation *op, const IRMapping &map, ProfStruct &prof, int taskId) {

    LLVM_DEBUG({
        llvm::dbgs() << "Processing operation: ";
        op->dump();

        llvm::dbgs() << "Current timestamp: " << prof.timestamp << "\n";
    });

    std::string asyncDurationAttrName = "torq-async-duration-cycles";
    std::string asyncBytesAttrName = "torq-async-bytes";

    Builder builder(op->getContext());

    // DMA cfg ops: store write/read NDL for use by corresponding start ops
    if (auto dmaInCfgOp = dyn_cast<torq_hw::DmaInCfgOp>(op)) {
        prof.currentDmaInBytes = dmaNdlTransferBytes(dmaInCfgOp.getWriteNdl());
    }
    else if (auto dmaOutCfgOp = dyn_cast<torq_hw::DmaOutCfgOp>(op)) {
        prof.currentDmaOutBytes = dmaNdlTransferBytes(dmaOutCfgOp.getReadNdl());
    }
    // Add the dma time to the timeline and annotate the start op
    else if (auto dmaInStartOp = dyn_cast<torq_hw::DmaInStartOp>(op)) {
        if (prof.currentDmaInBytes == 0) {
            return op->emitError() << "no preceding torq_hw.dma_in_cfg to size this transfer";
        }

        uint64_t bytes = prof.currentDmaInBytes;
        uint64_t cycles = estimateDmaCycles(bytes, dmaThroughputBytesPerCycle(clDmaInMtu));

        dmaInStartOp->setAttr(asyncBytesAttrName, builder.getI64IntegerAttr(bytes));
        dmaInStartOp->setAttr(asyncDurationAttrName, builder.getI64IntegerAttr(cycles));

        // TODO: check if there is a concurrent dma out and compute the bandwidth if shared
        prof.addToDmaTimeline(
            taskId, DmaType::In, prof.timestamp, prof.timestamp + cycles,
            toString(dmaInStartOp.getLoc()), bytes
        );
    }
    else if (auto dmaOutStartOp = dyn_cast<torq_hw::DmaOutStartOp>(op)) {
        if (prof.currentDmaOutBytes == 0) {
            return op->emitError() << "no preceding torq_hw.dma_out_cfg to size this transfer";
        }

        uint64_t bytes = prof.currentDmaOutBytes;
        uint64_t cycles = estimateDmaCycles(bytes, dmaThroughputBytesPerCycle(clDmaOutMtu));

        dmaOutStartOp->setAttr(asyncBytesAttrName, builder.getI64IntegerAttr(bytes));
        dmaOutStartOp->setAttr(asyncDurationAttrName, builder.getI64IntegerAttr(cycles));

        // TODO: check if there is a concurrent dma in and compute the bandwidth if shared
        prof.addToDmaTimeline(
            taskId, DmaType::Out, prof.timestamp, prof.timestamp + cycles,
            toString(dmaOutStartOp.getLoc()), bytes
        );
    }
    // Update the timestamp based on the end of dma operation.
    // If the timestamp is less than the end of dma, update the timestamp to the end of dma.
    else if (isa<torq_hw::DmaInWaitOp>(op)) {
        auto lastTimeStamp = prof.getLastDmaTime(DmaType::In);
        if (prof.timestamp < lastTimeStamp) {
            prof.timestamp = lastTimeStamp;
        }
    }
    else if (isa<torq_hw::DmaOutWaitOp>(op)) {
        auto lastTimeStamp = prof.getLastDmaTime(DmaType::Out);
        if (prof.timestamp < lastTimeStamp) {
            prof.timestamp = lastTimeStamp;
        }
    }
    else if (auto cdmaStartOp = dyn_cast<torq_hw::CDMAStartOp>(op)) {
        uint64_t bytes = cdmaTransferBytes(cdmaStartOp.getSrc());
        if (bytes == 0) {
            return op->emitError() << "cannot determine CDMA transfer size";
        }
        uint64_t cycles = estimateCdmaCycles(bytes);

        cdmaStartOp->setAttr(asyncBytesAttrName, builder.getI64IntegerAttr(bytes));
        cdmaStartOp->setAttr(asyncDurationAttrName, builder.getI64IntegerAttr(cycles));

        auto srcType = cdmaStartOp.getSrc().getType();
        auto dstType = cdmaStartOp.getDest().getType();

        DmaType type = DmaType::CdmaLramToDtcm;
        auto srcSpace = getEncodingMemorySpace(srcType);
        auto dstSpace = getEncodingMemorySpace(dstType);

        if (srcSpace == syna::torq_hl::MemorySpace::Lram &&
            dstSpace == syna::torq_hl::MemorySpace::Dtcm) {
            type = DmaType::CdmaLramToDtcm;
        }
        else if (srcSpace == syna::torq_hl::MemorySpace::Lram &&
                 dstSpace == syna::torq_hl::MemorySpace::Itcm) {
            type = DmaType::CdmaLramToItcm;
        }
        else if (srcSpace == syna::torq_hl::MemorySpace::Dtcm &&
                 dstSpace == syna::torq_hl::MemorySpace::Lram) {
            type = DmaType::CdmaDtcmToLram;
        }

        prof.addToDmaTimeline(
            taskId, type, prof.timestamp, prof.timestamp + cycles, toString(cdmaStartOp.getLoc()),
            bytes
        );
    }
    else if (isa<torq_hw::CDMAWaitOp>(op)) {
        uint64_t lastTimeStamp = 0;
        lastTimeStamp =
            std::max(lastTimeStamp, (uint64_t)prof.getLastDmaTime(DmaType::CdmaLramToDtcm));
        lastTimeStamp =
            std::max(lastTimeStamp, (uint64_t)prof.getLastDmaTime(DmaType::CdmaLramToItcm));
        lastTimeStamp =
            std::max(lastTimeStamp, (uint64_t)prof.getLastDmaTime(DmaType::CdmaDtcmToLram));

        if (prof.timestamp < lastTimeStamp) {
            prof.timestamp = lastTimeStamp;
        }
    }
    // Add the slice time based on referred slice_program op
    // This is added only when the slice_start in encountered
    else if (auto sliceStartOp = dyn_cast<torq_hw::SliceStartOp>(op)) {

        // the invocation is a blockargument, so we use the lookup
        // to find the actual value for the current invocation
        // that is executing when this function gets called
        auto invocation = map.lookup(sliceStartOp.getInvocation());

        uint64_t duration = 0;
        std::string opName = "";
        torq_hl::ProgramOp programOp;

        if (auto createInvocationOp = invocation.getDefiningOp<torq_hl::CreateInvocationOp>()) {

            programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

            if (!programOp) {
                llvm::report_fatal_error("Program is not a torq_hl::ProgramOp");
                return failure();
            }
        }
        else if (auto descriptorOp = invocation.getDefiningOp<torq_hl::DescriptorOp>()) {

            programOp = descriptorOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

            if (!programOp) {
                llvm::report_fatal_error("Program is not a torq_hl::ProgramOp");
                return failure();
            }
        }
        else {
            return invocation.getDefiningOp()->emitError()
                   << "Unsupported invocation operation for profiling";
        }

        programOp.getBody().walk([&](Operation *sop) {
            if (auto profOp = dyn_cast<torq_hw::SliceProfilingOp>(sop)) {
                duration += profOp.getMaxCycle();
            }
            return WalkResult::advance();
        });

        Location loc = programOp.getLoc();
        std::string locStr = toString(loc);

        prof.addToSliceTimeline(
            taskId, sliceStartOp.getId().getZExtValue(), prof.timestamp, prof.timestamp + duration,
            locStr
        );

        op->setAttr(asyncDurationAttrName, builder.getI64IntegerAttr(duration));

        prof.timestamp += 1; // Add 1 cycle for slice start
    }

    // Update the timestamp based on the end of slice operation.
    // If the timestamp is less than the end of slice, update the timestamp to the end of slice.
    else if (auto sliceWaitOp = dyn_cast<torq_hw::SliceWaitOp>(op)) {
        auto lastTimeStamp = prof.getLastSliceTime(sliceWaitOp.getId().getZExtValue());
        if (prof.timestamp < lastTimeStamp) {
            prof.timestamp = lastTimeStamp;
        }
    }
    // Handle CSS (Computation Slice System) operations - similar to Slice operations
    else if (auto cssStartOp = dyn_cast<torq_hw::CSSStartOp>(op)) {
        // CSS operations cycle count is not available so lets keep a fixed duration
        // Duration: 10000 cycles as requested
        const uint64_t cssFixedDuration = 8000; // 10us at 800MHz (10us * 800 cycles/us)
        prof.addToCSSTimeline(
            taskId, prof.timestamp, prof.timestamp + cssFixedDuration, toString(cssStartOp.getLoc())
        );
        op->setAttr(asyncDurationAttrName, builder.getI64IntegerAttr(cssFixedDuration));
        prof.timestamp += 1; // Add 1 cycle for CSS start overhead
    }
    else if (isa<torq_hw::CSSWaitOp>(op)) {
        // CSS wait - update timestamp to end of CSS operation
        auto lastTimeStamp = prof.getLastCSSTime();
        if (prof.timestamp < lastTimeStamp) {
            prof.timestamp = lastTimeStamp;
        }
    }
    else if (isa<memref::AllocOp, memref::DeallocOp, memref::CastOp, memref::MemorySpaceCastOp,
                 memref::ReinterpretCastOp, memref::ReshapeOp>(op)) {
        // These operations do not affect the profiling timestamp
    }
    else {
        return op->emitError() << "profiling not supported";
    }

    LLVM_DEBUG({ llvm::dbgs() << "Updated timestamp: " << prof.timestamp << "\n"; });

    return success();
}

void writeToCsv(const std::string &filename, const ProfStruct &prof) {

    // TODO: print the profiling information sorted by startTime

    std::ofstream file(filename);
    file << "taskId, elapsed_time(us), timestamp(us), event, location, bytes\n";
    for (int i = 0; i < prof.dmaTimeline.size(); i++) {
        const auto dmaT = prof.dmaTimeline[i];
        if (dmaT.dmaType == DmaType::In) {
            file << dmaT.taskId << "," << dmaT.endTime - dmaT.startTime << "," << dmaT.startTime
                 << "," << "DMA In" << "," << dmaT.loc << "," << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::Out) {
            file << dmaT.taskId << "," << dmaT.endTime - dmaT.startTime << "," << dmaT.startTime
                 << "," << "DMA Out" << "," << dmaT.loc << "," << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaLramToDtcm) {
            file << dmaT.taskId << "," << dmaT.endTime - dmaT.startTime << "," << dmaT.startTime
                 << "," << "CDMA LRAM to DTCM" << "," << dmaT.loc << "," << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaLramToItcm) {
            file << dmaT.taskId << "," << dmaT.endTime - dmaT.startTime << "," << dmaT.startTime
                 << "," << "CDMA LRAM to ITCM" << "," << dmaT.loc << "," << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaDtcmToLram) {
            file << dmaT.taskId << "," << dmaT.endTime - dmaT.startTime << "," << dmaT.startTime
                 << "," << "CDMA DTCM to LRAM" << "," << dmaT.loc << "," << dmaT.bytes << "\n";
        }
    }
    for (int i = 0; i < prof.sliceTimeline.size(); i++) {
        const auto sliceT = prof.sliceTimeline[i];
        file << sliceT.taskId << "," << sliceT.endTime - sliceT.startTime << "," << sliceT.startTime
             << "," << "Slice " << sliceT.id << "," << sliceT.loc << "," << "\n";
    }
    for (int i = 0; i < prof.cssTimeline.size(); i++) {
        const auto cssT = prof.cssTimeline[i];
        file << cssT.taskId << "," << cssT.endTime - cssT.startTime << "," << cssT.startTime << ","
             << "CSS" << "," << cssT.loc << "," << "\n";
    }
};

LogicalResult ProfilingPass::timeProfiling(mlir::Operation *topLevelOp) {
    ProfStruct prof;
    WalkExecutionOptions options;

    // for each operation in the execution walk update the profiling state
    // we use a pointer to prof to ensure when the lambda is copied the reference is valid
    auto onExecuteFun = [&prof](Operation *op, InvocationValue invocation, const IRMapping &map) {
        if (auto nssTask = dyn_cast<torq_hw::NssTaskOp>(op)) {

            OpBuilder builder(op);
            op->setAttr("torq-task-id", builder.getI64IntegerAttr(prof.nextTaskIndex));

            op->setAttr("torq-start-time-cycles", builder.getI64IntegerAttr(prof.timestamp));

            for (auto &nestedOp : op->getRegion(0).getOps()) {
                if (failed(processOperationTime(&nestedOp, map, prof, prof.nextTaskIndex))) {
                    return failure();
                }
            }

            op->setAttr("torq-end-time-cycles", builder.getI64IntegerAttr(prof.timestamp));

            prof.nextTaskIndex++;
        }
        else {
            // TODO: the host may be doing copies or spending time in other ways
            // we should keep that into account. For the moment we assume the host
            // takes 0 cyles

            return success();
        }

        return success();
    };

    auto onStartFun =
        [&prof](torq_hl::StartProgramOp op, InvocationValue invocation, const IRMapping &map) {
            Builder builder(op);
            prof.lastActionTimestampAttr = builder.getI64IntegerAttr(prof.timestamp);

            op->setAttr("torq-start-time-cycles", prof.lastActionTimestampAttr);

            // we assume the scheduling of the operation takes 0 cycles
            op->setAttr("torq-end-time-cycles", prof.lastActionTimestampAttr);

            return success();
        };

    auto onFinishFun =
        [&prof](torq_hl::WaitProgramOp op, InvocationValue invocation, InvocationReturns) {
            // we assume nothing happened since the last start
            op->setAttr("torq-start-time-cycles", prof.lastActionTimestampAttr);

            Builder builder(op);
            prof.lastActionTimestampAttr = builder.getI64IntegerAttr(prof.timestamp);
            op->setAttr("torq-end-time-cycles", prof.lastActionTimestampAttr);

            return success();
        };

    options.onExecute = onExecuteFun;
    options.onStart = onStartFun;
    options.onFinish = onFinishFun;

    // only walk into NSS tasks executions (do not walk into Host tasks)
    options.walkInto = [&](torq_hl::StartProgramOp startProgramOp, InvocationValue invocation,
                           const mlir::IRMapping &) {
        return startProgramOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS;
    };

    if (failed(walkExecution(topLevelOp, options))) {
        return failure();
    }

    if (!clTorqProfilingDump.empty()) {
        writeToCsv(clTorqProfilingDump, prof);
    }

    return success();
}

void ProfilingPass::runOnOperation() {
    auto funcOp = getOperation();

    if (failed(memProfiling(funcOp))) {
        funcOp.emitError() << "failed to profile memory";
    }

    if (failed(cycleProfiling(funcOp))) {
        funcOp.emitError() << "failed to profile cycle";
    }

    Operation *toAnalyze = funcOp.getOperation();

    if (failed(timeProfiling(toAnalyze))) {
        funcOp.emitError() << "failed to profile time";
    }

    IRRewriter rewriter(funcOp.getContext());

    Block &funcBlock = funcOp->getRegion(0).front();
    rewriter.setInsertionPointToStart(&funcBlock);

    torq_hw::DispatchProfilingOp::create(rewriter, funcOp.getLoc(), totalMemSize, totalCycle);
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createProfilingPass() {
    return std::make_unique<ProfilingPass>();
}

} // namespace mlir::syna::torq
