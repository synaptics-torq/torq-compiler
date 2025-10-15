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
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <map>
#include <tuple>

#define DEBUG_TYPE "torq-profiling"

static llvm::cl::opt<std::string> clTorqProfilingDump(
    "torq-dump-profiling", llvm::cl::desc("Dump profiling information to specified path"),
    llvm::cl::init("timeline.csv")
);

using namespace std;

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

namespace {

enum class DmaType { In, Out };

typedef uint64_t StartTime;
typedef uint64_t EndTime;

struct DmaEvent {
    DmaType dmaType;
    StartTime startTime;
    EndTime endTime;
    std::string loc;
};

struct SliceEvent {
    int id;
    StartTime startTime;
    EndTime endTime;
    std::string opName;
    std::string loc;
};

typedef std::vector<DmaEvent> DmaTimeline;
typedef std::vector<SliceEvent> SliceTimeline;

struct ProfStruct {

    uint64_t timestamp{0};

    DmaTimeline dmaTimeline;
    SliceTimeline sliceTimeline;

    std::string currentDmaInLoc;
    std::string currentDmaOutLoc;

    uint64_t currentDmaInCycles;
    uint64_t currentDmaOutCycles;

    void
    addToSliceTimeline(int id, uint64_t start, uint64_t end, std::string opName, std::string loc) {
        sliceTimeline.push_back({id, start, end, opName, loc}
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

    void addToDmaTimeline(DmaType dmaType, uint64_t start, uint64_t end, std::string loc) {
        dmaTimeline.push_back({dmaType, start, end, loc}
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
            rewriter.getUnknownLoc(), sliceMemSize, shapeSstr.str(), curMaxCycle, ndlCycles
        );

        // NDL Cycle Match Checks
        bool isNdlCycleMatch = true;
        auto loc = sliceTaskOp.getLoc();
        auto opName = sliceTaskOp.getOpName();

        if (cewr != cepr) {
            llvm::errs() << "\n"
                         << opName << " cewr != cepr: " << cewr << " vs " << cepr << " at "
                         << toString(loc) << "\n";
            llvm::errs() << " cewr : " << regNdlsAttr[NdlType::CEWR] << "\n";
            llvm::errs() << " cepr : " << regNdlsAttr[NdlType::CEPR] << "\n";
            isNdlCycleMatch = false;
        }

        if (deqw != acbr) {
            llvm::errs() << "\n"
                         << opName << " deqw != acbr: " << deqw << " vs " << acbr << " at "
                         << toString(loc) << "\n";
            llvm::errs() << " deqw : " << memNdlsAttr[NdlType::DEQW] << "\n";
            llvm::errs() << " acbr : " << regNdlsAttr[NdlType::ACBR] << "\n";
            isNdlCycleMatch = false;
        }

        if (cepr != cedr) {
            llvm::errs() << "\n"
                         << opName << " cepr != cedr: " << cepr << " vs " << cedr << " at "
                         << toString(loc) << "\n";
            llvm::errs() << " cepr : " << regNdlsAttr[NdlType::CEPR] << "\n";
            llvm::errs() << " cedr : " << regNdlsAttr[NdlType::CEDR] << "\n";
            isNdlCycleMatch = false;
        }

        if (!isNdlCycleMatch) {
            llvm::errs() << " ref : " << memNdlsAttr[NdlType::REF] << "\n";
        }

        return WalkResult::advance();
    });

    return success();
}

static LogicalResult processOperationTime(Operation *op, const IRMapping &map, ProfStruct &prof) {

    LLVM_DEBUG({
        llvm::dbgs() << "Processing operation: ";
        op->dump();

        llvm::dbgs() << "Current timestamp: " << prof.timestamp << "\n";
    });

    const uint64_t perCycleDmaTransferBytes = 8;

    // DMA in config contains reference to slice_program op
    if (auto dmaInCfgOp = dyn_cast<torq_hw::DmaInCfgOp>(op)) {
        // TODO: check if this is correct for strided types
        prof.currentDmaInCycles =
            getEncodedTotalSizeBytes(dmaInCfgOp.getRead().getType()) / perCycleDmaTransferBytes;
        prof.currentDmaInLoc = toString(dmaInCfgOp.getLoc());
    }
    else if (auto dmaOutCfgOp = dyn_cast<torq_hw::DmaOutCfgOp>(op)) {
        // TODO: check if this is correct for strided types
        prof.currentDmaOutCycles =
            getEncodedTotalSizeBytes(dmaOutCfgOp.getRead().getType()) / perCycleDmaTransferBytes;
        prof.currentDmaOutLoc = toString(dmaOutCfgOp.getLoc());
    }
    // Add the dma time to the timeline
    else if (auto dmaInStartOp = dyn_cast<torq_hw::DmaInStartOp>(op)) {
        // TODO: check if there is a concurrent dma out and compute the bandwidth if shared
        prof.addToDmaTimeline(
            DmaType::In, prof.timestamp, prof.timestamp + prof.currentDmaInCycles,
            prof.currentDmaInLoc
        );
    }
    else if (auto dmaOutStartOp = dyn_cast<torq_hw::DmaOutStartOp>(op)) {
        // TODO: check if there is a concurrent dma in and compute the bandwidth if shared
        prof.addToDmaTimeline(
            DmaType::Out, prof.timestamp, prof.timestamp + prof.currentDmaOutCycles,
            prof.currentDmaOutLoc
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
    // Add the slice time based on referred slice_program op
    // This is added only when the slice_start in encountered
    else if (auto sliceStartOp = dyn_cast<torq_hw::SliceStartOp>(op)) {

        // the invocation is a blockargument, so we use the lookup
        // to find the actual value for the current invocation
        // that is executing when this function gets called
        auto invocationOp = map.lookup(sliceStartOp.getInvocation());

        auto createInvocationOp = invocationOp.getDefiningOp<torq_hl::CreateInvocationOp>();

        if (!createInvocationOp) {
            llvm::report_fatal_error("Invocation is not a torq_hl::CreateInvocationOp");
            return failure();
        }

        auto programOp = createInvocationOp.getProgram().getDefiningOp<torq_hl::ProgramOp>();

        if (!programOp) {
            llvm::report_fatal_error("Program is not a torq_hl::ProgramOp");
            return failure();
        }

        uint64_t duration = 0;
        std::string opName = "";

        programOp.getBody().walk([&](Operation *sop) {
            if (auto profOp = dyn_cast<torq_hw::SliceProfilingOp>(sop)) {
                duration += profOp.getMaxCycle();
            }
            if (auto taskOp = dyn_cast<torq_hw::SliceTaskOp>(sop)) {
                if (!opName.empty()) {
                    opName += "_";
                }
                opName += taskOp.getOpName();
            }
            return WalkResult::advance();
        });

        if (opName.empty()) {
            opName = "unknown";
        }

        prof.addToSliceTimeline(
            sliceStartOp.getId().getZExtValue(), prof.timestamp, prof.timestamp + duration, opName,
            toString(programOp.getLoc())
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
    for (int i = 0; i < prof.dmaTimeline.size(); i++) {
        const auto dmaT = prof.dmaTimeline[i];
        if (dmaT.dmaType == DmaType::In) {
            file << "DI" << i << "," << dmaT.startTime << "," << dmaT.endTime << "," << dmaT.loc
                 << ",DMA_IN\n";
        }
        else {
            file << "DO" << i << "," << dmaT.startTime << "," << dmaT.endTime << "," << dmaT.loc
                 << ",DMA_OUT\n";
        }
    }
    for (int i = 0; i < prof.sliceTimeline.size(); i++) {
        const auto sliceT = prof.sliceTimeline[i];
        file << "S" << i << "," << sliceT.startTime << "," << sliceT.endTime << "," << sliceT.loc
             << "," << sliceT.opName << "\n";
    }
};

LogicalResult ProfilingPass::timeProfiling(mlir::Operation *topLevelOp) {
    ProfStruct prof;
    WalkExecutionOptions options;

    // for each operation in the execution walk update the profiling state
    // we use a pointer to prof to ensure when the lambda is copied the reference is valid
    auto onExecuteFun = [&prof](Operation *op, InvocationValue invocation, const IRMapping &map) {
        if (auto nssTask = dyn_cast<torq_hw::NssTaskOp>(op)) {
            for (auto &nestedOp : op->getRegion(0).getOps()) {
                if (failed(processOperationTime(&nestedOp, map, prof))) {
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
    options.walkInto = [&](torq_hl::StartProgramOp startProgramOp, InvocationValue,
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

    rewriter.create<torq_hw::DispatchProfilingOp>(
        rewriter.getUnknownLoc(), totalMemSize, totalCycle
    );
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createProfilingPass() {
    return std::make_unique<ProfilingPass>();
}

} // namespace mlir::syna::torq
