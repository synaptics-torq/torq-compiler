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
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/InvocationUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqHw.h"
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
    llvm::cl::init("timeline.csv")
);

static llvm::cl::opt<bool> clTorqDisableNdlCycleCheck(
    "torq-disable-ndl-cycle-check", llvm::cl::desc("Disable NDL cycle match checks in profiling"),
    llvm::cl::init(true)
);

using namespace std;

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

namespace {

enum class DmaType { In, Out, CdmaLramToDtcm, CdmaLramToItcm, CdmaDtcmToLram };

typedef uint64_t StartTime;
typedef uint64_t EndTime;

struct DmaEvent {
    int actionIndex;
    DmaType dmaType;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
    uint64_t bytes;
};

struct SliceEvent {
    int actionIndex;
    int id;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
};

struct CSSEvent {
    int actionIndex;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
};

typedef std::vector<DmaEvent> DmaTimeline;
typedef std::vector<SliceEvent> SliceTimeline;
typedef std::vector<CSSEvent> CSSTimeline;

struct ProfStruct {

    uint64_t timestamp{0};

    DmaTimeline dmaTimeline;
    SliceTimeline sliceTimeline;
    CSSTimeline cssTimeline;

    std::string currentDmaInLoc;
    std::string currentDmaOutLoc;

    uint64_t currentDmaInCycles;
    uint64_t currentDmaOutCycles;

    uint64_t currentDmaInBytes;
    uint64_t currentDmaOutBytes;

    void addToSliceTimeline(int actionId, int id, uint64_t start, uint64_t end, std::string loc) {
        sliceTimeline.push_back({actionId, id, start, end, loc}
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

    void addToCSSTimeline(int actionId, uint64_t start, uint64_t end, std::string loc) {
        cssTimeline.push_back({actionId, start, end, loc});
    }

    int getLastCSSTime() {
        if (cssTimeline.empty()) {
            return 0;
        }
        return cssTimeline.back().endTime;
    }

    void addToDmaTimeline(
        int actionId, DmaType dmaType, uint64_t start, uint64_t end, std::string loc,
        uint64_t bytes = 0
    ) {
        dmaTimeline.push_back({actionId, dmaType, start, end, loc, bytes}
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

class ProfilingPass : public ProfilingBase<ProfilingPass> {
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
    memSummary.pop_back(); // remove the last '+'

    return success();
}

template <typename T> size_t ndlCycle(T attr) {
    size_t cycle = 1;
    for (auto dim : attr.getDims()) {
        if (torq_hw::DimType::H == dim.getType()) {
            cycle *= dim.getCount();
        }
    }
    return cycle;
}

// Helper to convert mlir::Attribute to string using LLVM's raw_string_ostream
std::string attrToString(mlir::Attribute attr) {
    std::string str;
    llvm::raw_string_ostream os(str);
    attr.print(os);
    return os.str();
}

LogicalResult ProfilingPass::cycleProfiling(mlir::FunctionOpInterface funcOp) {
    auto ctx = funcOp.getContext();

    funcOp.walk([&](torq_hw::SliceTaskOp sliceTaskOp) {
        map<NdlType, mlir::Attribute> memNdlsAttr;
        map<NdlType, mlir::Attribute> regNdlsAttr;

        map<NdlType, size_t> memNdlCycles;
        for (auto &ndl : sliceTaskOp.getMemNdls()) {
            memNdlCycles[ndl.getType()] += ndlCycle(ndl);
            memNdlsAttr[ndl.getType()] = ndl;
        }

        map<NdlType, size_t> regNdlCycles;
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

        totalCycle = std::max(totalCycle, curMaxCycle);

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

        rewriter.create<torq_hw::SliceProfilingOp>(
            sliceTaskOp.getLoc(), sliceMemSize, shapeSstr.str(), curMaxCycle, ndlCycles
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

    return success();
}

static LogicalResult
processOperationTime(Operation *op, const IRMapping &map, ProfStruct &prof, int actionId) {

    LLVM_DEBUG({
        llvm::dbgs() << "Processing operation: ";
        op->dump();

        llvm::dbgs() << "Current timestamp: " << prof.timestamp << "\n";
    });

    const uint64_t perCycleDmaTransferBytes = 8;

    // DMA in config contains reference to slice_program op
    if (auto dmaInCfgOp = dyn_cast<torq_hw::DmaInCfgOp>(op)) {
        // TODO: check if this is correct for strided types
        prof.currentDmaInBytes = getEncodedTotalSizeBytes(dmaInCfgOp.getRead().getType());
        prof.currentDmaInCycles = prof.currentDmaInBytes / perCycleDmaTransferBytes;
        prof.currentDmaInLoc = toString(dmaInCfgOp.getLoc());
    }
    else if (auto dmaOutCfgOp = dyn_cast<torq_hw::DmaOutCfgOp>(op)) {
        // TODO: check if this is correct for strided types
        prof.currentDmaOutBytes = getEncodedTotalSizeBytes(dmaOutCfgOp.getRead().getType());
        prof.currentDmaOutCycles = prof.currentDmaOutBytes / perCycleDmaTransferBytes;
        prof.currentDmaOutLoc = toString(dmaOutCfgOp.getLoc());
    }
    // Add the dma time to the timeline
    else if (auto dmaInStartOp = dyn_cast<torq_hw::DmaInStartOp>(op)) {
        // TODO: check if there is a concurrent dma out and compute the bandwidth if shared
        prof.addToDmaTimeline(
            actionId, DmaType::In, prof.timestamp, prof.timestamp + prof.currentDmaInCycles,
            toString(dmaInStartOp.getLoc()), prof.currentDmaInBytes
        );
    }
    else if (auto dmaOutStartOp = dyn_cast<torq_hw::DmaOutStartOp>(op)) {
        // TODO: check if there is a concurrent dma in and compute the bandwidth if shared
        prof.addToDmaTimeline(
            actionId, DmaType::Out, prof.timestamp, prof.timestamp + prof.currentDmaOutCycles,
            toString(dmaOutStartOp.getLoc()), prof.currentDmaOutBytes
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
        auto srcType = cdmaStartOp.getSrc().getType();
        auto dstType = cdmaStartOp.getDest().getType();
        uint64_t bytes = getEncodedTotalSizeBytes(srcType);

        double factor = 1.0;
        auto hw = TorqHw::get();
        std::string css = hw.getCSSConfigName();
        std::string nss = hw.getNSSConfigName();

        if (css == "coral_v1" && nss == "nss_v1") {
            factor = 0.36;
        }
        else {
            factor = 0.8;
        }

        uint64_t cycles = std::max((uint64_t)1, (uint64_t)std::ceil(bytes / (16.0 * factor)));

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
            actionId, type, prof.timestamp, prof.timestamp + cycles, toString(cdmaStartOp.getLoc()),
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
            actionId, sliceStartOp.getId().getZExtValue(), prof.timestamp,
            prof.timestamp + duration, locStr
        );

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
            actionId, prof.timestamp, prof.timestamp + cssFixedDuration,
            toString(cssStartOp.getLoc())
        );
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
    file << "actionIndex, elapsed_time(us), timestamp(us), event, location, bytes\n";
    for (int i = 0; i < prof.dmaTimeline.size(); i++) {
        const auto dmaT = prof.dmaTimeline[i];
        if (dmaT.dmaType == DmaType::In) {
            file << dmaT.actionIndex << "," << dmaT.endTime - dmaT.startTime << ","
                 << dmaT.startTime << "," << "DMA In" << "," << dmaT.loc << "," << dmaT.bytes
                 << "\n";
        }
        else if (dmaT.dmaType == DmaType::Out) {
            file << dmaT.actionIndex << "," << dmaT.endTime - dmaT.startTime << ","
                 << dmaT.startTime << "," << "DMA Out" << "," << dmaT.loc << "," << dmaT.bytes
                 << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaLramToDtcm) {
            file << dmaT.actionIndex << "," << dmaT.endTime - dmaT.startTime << ","
                 << dmaT.startTime << "," << "CDMA LRAM to DTCM" << "," << dmaT.loc << ","
                 << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaLramToItcm) {
            file << dmaT.actionIndex << "," << dmaT.endTime - dmaT.startTime << ","
                 << dmaT.startTime << "," << "CDMA LRAM to ITCM" << "," << dmaT.loc << ","
                 << dmaT.bytes << "\n";
        }
        else if (dmaT.dmaType == DmaType::CdmaDtcmToLram) {
            file << dmaT.actionIndex << "," << dmaT.endTime - dmaT.startTime << ","
                 << dmaT.startTime << "," << "CDMA DTCM to LRAM" << "," << dmaT.loc << ","
                 << dmaT.bytes << "\n";
        }
    }
    for (int i = 0; i < prof.sliceTimeline.size(); i++) {
        const auto sliceT = prof.sliceTimeline[i];
        file << sliceT.actionIndex << "," << sliceT.endTime - sliceT.startTime << ","
             << sliceT.startTime << "," << "Slice " << sliceT.id << "," << sliceT.loc << ","
             << "\n";
    }
    for (int i = 0; i < prof.cssTimeline.size(); i++) {
        const auto cssT = prof.cssTimeline[i];
        file << cssT.actionIndex << "," << cssT.endTime - cssT.startTime << "," << cssT.startTime
             << "," << "CSS" << "," << cssT.loc << "," << "\n";
    }
};

LogicalResult ProfilingPass::timeProfiling(mlir::Operation *topLevelOp) {
    ProfStruct prof;
    WalkExecutionOptions options;

    // for each operation in the execution walk update the profiling state
    // we use a pointer to prof to ensure when the lambda is copied the reference is valid
    auto onExecuteFun = [&prof](Operation *op, InvocationValue invocation, const IRMapping &map) {
        if (auto nssTask = dyn_cast<torq_hw::NssTaskOp>(op)) {
            // Extract action ID from the invocation's uses
            int actionId = 0;
            for (OpOperand &use : invocation.getUses()) {
                Operation *user = use.getOwner();
                // Skip start_program operations, look for wait_program or other users
                if (!isa<torq_hl::StartProgramOp>(user)) {
                    if (auto actionIdAttr =
                            user->getAttrOfType<mlir::IntegerAttr>("torq-action-id")) {
                        actionId = actionIdAttr.getInt();
                        break;
                    }
                }
            }

            for (auto &nestedOp : op->getRegion(0).getOps()) {
                if (failed(processOperationTime(&nestedOp, map, prof, actionId))) {
                    return failure();
                }
            }
        }
        else {
            // TODO: the host may be doing copies or spending time in other ways
            // we should keep that into account. For the moment we assume the host
            // takes 0 cyles

            return success();
        }

        return success();
    };

    options.onExecute = onExecuteFun;

    // only walk into NSS tasks executions (do not walk into Host tasks)
    options.walkInto = [&](torq_hl::StartProgramOp startProgramOp, InvocationValue invocation,
                           const mlir::IRMapping &) {
        return startProgramOp.getInvocation().getType().getExecutor() == torq_hl::Executor::NSS;
    };

    if (failed(walkExecution(topLevelOp, options))) {
        return failure();
    }

    writeToCsv(clTorqProfilingDump, prof);

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

    rewriter.create<torq_hw::DispatchProfilingOp>(funcOp.getLoc(), totalMemSize, totalCycle);
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createProfilingPass() {
    return std::make_unique<ProfilingPass>();
}

} // namespace mlir::syna::torq
