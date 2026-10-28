// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWOps.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/ProfilingUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-pair-dma-channels"

namespace mlir::syna::torq {

namespace {

// The NPU has two independent DMA channels. torq_api.h describes them as hardware threads that the
// NSS CFG agent "simultaneously starts with start=1", each with its own start and wait bit:
//
//   NSS.DMA.XR   XRAM -> LRAM   (input,  AXI read channel)
//   NSS.DMA.XW   LRAM -> XRAM   (output, AXI write channel)
//
// The lowering in Conversions/TorqHLToTorqHW/Patterns.cpp gives every transfer its own closed
// task (cfg; start; wait), so the two channels are never busy at the same time. Every fixture we
// measured had zero overlap between them.
//
// CompileNSSInvocationsPass walks a task and sets xrStart and xwStart from the start ops it finds,
// so putting one input and one output in the same task is all it takes to run both at once. A task
// holds one descriptor per direction, so a pair is one of each.
//
// This runs after addresses are assigned, which is what makes it safe. We cannot change a tile or
// move a buffer, so the memory plan is whatever the allocator already decided. Pulling an input
// earlier is only allowed when its LRAM range does not touch the LRAM range of anything it moves
// over, and that is an interval test on real addresses, not a guess about aliasing.
struct AddrRange {
    int64_t begin = 0;
    int64_t end = 0; // exclusive

    bool valid() const { return end > begin; }

    bool overlaps(const AddrRange &other) const {
        return valid() && other.valid() && begin < other.end && other.begin < end;
    }
};

// Both ends of a transfer. LRAM and XRAM are separate address spaces, so the two ranges here are
// never compared against each other.
struct TransferRanges {
    AddrRange lram;
    AddrRange xram;
};

// How far an NDL walk reaches from its base address. A transfer can be strided, so the byte count
// is not the footprint: 1 + sum over dims of (count - 1) * stride.
static int64_t ndlSpan(torq_hw::DmaNdlAttr ndl) {
    int64_t span = 1;
    for (auto dim : ndl.getDims()) {
        span += (dim.getCount() - 1) * dim.getStride();
    }
    return span;
}

static AddrRange makeRange(std::optional<int32_t> addr, torq_hw::DmaNdlAttr ndl) {
    AddrRange r;
    if (!addr || !ndl) {
        return r; // No address, so leave it invalid and treat it as overlapping.
    }
    r.begin = *addr;
    r.end = *addr + ndlSpan(ndl);
    return r;
}

// Both ends of a transfer. An input reads XRAM and writes LRAM, an output does the reverse.
static TransferRanges transferRanges(Operation *cfg) {
    TransferRanges t;
    if (auto in = dyn_cast<torq_hw::DmaInCfgOp>(cfg)) {
        t.lram = makeRange(in.getWriteAddress(), in.getWriteNdl());
        t.xram = makeRange(in.getReadAddress(), in.getReadNdl());
    }
    else if (auto out = dyn_cast<torq_hw::DmaOutCfgOp>(cfg)) {
        t.lram = makeRange(out.getReadAddress(), out.getReadNdl());
        t.xram = makeRange(out.getWriteAddress(), out.getWriteNdl());
    }
    return t;
}

// What the input we move must not touch: the LRAM ranges read or written by the ops it moves over,
// and the XRAM ranges they write. Anything we cannot place sets unknown, which blocks the move.
struct Hazards {
    SmallVector<AddrRange> lram;
    SmallVector<AddrRange> xramWrites;
    bool unknown = false;

    bool clearOf(const TransferRanges &in) const {
        return !unknown &&
               llvm::none_of(lram, [&](const AddrRange &r) { return in.lram.overlaps(r); }) &&
               llvm::none_of(xramWrites, [&](const AddrRange &r) { return in.xram.overlaps(r); });
    }
};

// The bytes a static memref spans from addr, or nothing if either is unknown. The size is the one
// the allocator reserves, so padding and strides count.
static std::optional<AddrRange> memrefRange(Value value, std::optional<int64_t> addr) {
    auto mt = dyn_cast<MemRefType>(value.getType());
    if (!mt || !mt.hasStaticShape() || !addr) {
        return std::nullopt;
    }
    return AddrRange{*addr, *addr + getEncodedTotalSizeBytes(mt)};
}

// The range of a memref operand, on whichever side it lives. The input we move only touches LRAM
// (it writes there) and XRAM (it reads there), so an operand in DTCM or ITCM cannot clash with it
// and needs no range. An operand we cannot place at all blocks the move.
static void addOperandRange(Value value, Hazards &hazards) {
    auto mt = dyn_cast<MemRefType>(value.getType());
    if (!mt || !mt.hasStaticShape()) {
        hazards.unknown = true;
    }
    else if (std::optional<AddrRange> r = memrefRange(value, getLramAddress(value))) {
        hazards.lram.push_back(*r);
    }
    else if (std::optional<AddrRange> r = memrefRange(value, getXramAddress(value))) {
        // We do not know whether this operand is read or written, so treat it as a write. That is
        // the side the input we move cares about.
        hazards.xramWrites.push_back(*r);
    }
    else if (!getDtcmAddress(value) && !getItcmAddress(value)) {
        hazards.unknown = true;
    }
}

// torq_hl.get_block and the memref accessors only compute an address from their operands. They do
// not carry the Pure trait, so isMemoryEffectFree turns them down, and they are exactly what sits
// between an output and the input we want to pair it with.
static bool isAddressAccessor(Operation *op) {
    return isa<
        torq_hl::GetBlockOp, memref::SubViewOp, memref::GetGlobalOp, memref::ReinterpretCastOp>(op);
}

// Ops that reach neither LRAM nor XRAM, so an input may be moved over them for free. Everything
// else has to be understood before we move anything past it, which is what collectRanges enforces.
// Waits are not here: collectRanges handles them, because a wait ends a program that may still be
// using memory.
static bool cannotTouchMemory(Operation *op) {
    return isa<torq_hw::DmaInStartOp, torq_hw::DmaOutStartOp, torq_hw::NextOp,
               torq_hw::SliceProfilingOp, torq_hw::DispatchProfilingOp, torq_hw::GetAddressOp,
               torq_hw::GetFunctionBlockOp, torq_hw::GetCSSTaskBlockOp>(op) ||
           isMemoryEffectFree(op) || isAddressAccessor(op);
}

// The ranges a started program reaches: a slice program, a CSS program or a cDMA copy.
static void addStartRanges(Operation *start, Hazards &hazards) {
    auto addLram = [&](Value value) {
        if (std::optional<AddrRange> r = memrefRange(value, getLramAddress(value))) {
            hazards.lram.push_back(*r);
        }
        else {
            hazards.unknown = true;
        }
    };

    if (auto slice = dyn_cast<torq_hw::SliceStartOp>(start)) {
        // A slice program reaches LRAM through its arguments, and each of those resolves to a
        // real address. Reading them keeps the window open past a compute task, which is where
        // the large inputs live. The program itself also lives in LRAM.
        for (Value arg : slice.getArgs()) {
            addLram(arg);
        }
        addLram(slice.getProgram());
    }
    else if (auto cdma = dyn_cast<torq_hw::CDMAStartOp>(start)) {
        // A copy between LRAM and a core-local memory. It reads one end and writes the other,
        // and either way the LRAM end is a hazard for the input we move: reading it later than
        // planned gives the copy whatever the input just wrote.
        addOperandRange(cdma.getSrc(), hazards);
        addOperandRange(cdma.getDest(), hazards);
    }
    else if (auto css = dyn_cast<torq_hw::CSSStartOp>(start)) {
        // A program on the CSS, reading and writing the buffers it is handed.
        addOperandRange(css.getProgram(), hazards);
        addOperandRange(css.getArgsAddresses(), hazards);
        for (Value arg : css.getArgs()) {
            addOperandRange(arg, hazards);
        }
    }
    else {
        hazards.unknown = true;
    }
}

// Every range an op touches, on both sides. A task can hold slice programs and transfers, and we
// have to look at all of them before we move an input over it.
//
// On the LRAM side we keep every range, read or write: moving an input earlier makes it write LRAM
// sooner, and that is a hazard for a reader as much as for a writer. On the XRAM side we only keep
// the writes, because the input we move only reads there, and two readers cannot clash.
//
// A wait is a hazard too. A slice, CSS or cDMA program runs from its start to its wait, and the
// start is often in a task before the output, so it is not in the window at all. Moving the input
// over the wait makes it write LRAM while that program still runs, so the wait brings in the
// ranges of the program it waits for.
//
// This is closed by default. An op we do not recognise sets unknown and gives up the pair, because
// listing only the ops we know about is how a cDMA copy out of LRAM stayed invisible here and let
// an input overwrite a buffer that had not been copied to DTCM yet.
static void collectRanges(
    Operation *op, const DenseMap<Operation *, Operation *> &waitToStart, Hazards &hazards
) {
    op->walk([&](Operation *inner) {
        if (inner == op || isa<torq_hw::NssTaskOp, torq_hw::SliceTaskOp>(inner)) {
            return; // A container. What it holds is walked on its own.
        }
        if (isa<torq_hw::DmaInCfgOp, torq_hw::DmaOutCfgOp>(inner)) {
            TransferRanges t = transferRanges(inner);
            if (!t.lram.valid()) {
                hazards.unknown = true;
            }
            else {
                hazards.lram.push_back(t.lram);
            }
            if (isa<torq_hw::DmaOutCfgOp>(inner)) {
                if (!t.xram.valid()) {
                    hazards.unknown = true;
                }
                else {
                    hazards.xramWrites.push_back(t.xram);
                }
            }
        }
        else if (isa<torq_hw::SliceStartOp, torq_hw::CDMAStartOp, torq_hw::CSSStartOp>(inner)) {
            addStartRanges(inner, hazards);
        }
        else if (isa<torq_hw::SliceWaitOp, torq_hw::CDMAWaitOp, torq_hw::CSSWaitOp>(inner)) {
            Operation *start = waitToStart.lookup(inner);
            if (!start) {
                hazards.unknown = true;
            }
            else {
                addStartRanges(start, hazards);
            }
        }
        else if (isa<torq_hw::DmaInWaitOp, torq_hw::DmaOutWaitOp>(inner)) {
            // A transfer waited for here but started in another task is in flight across tasks,
            // and we do not track that. Its own task always holds the start in what we pair.
            bool in = isa<torq_hw::DmaInWaitOp>(inner);
            bool startHere = llvm::any_of(*inner->getBlock(), [&](Operation &o) {
                return in ? isa<torq_hw::DmaInStartOp>(o) : isa<torq_hw::DmaOutStartOp>(o);
            });
            if (!startHere) {
                hazards.unknown = true;
            }
        }
        else if (!cannotTouchMemory(inner)) {
            hazards.unknown = true;
        }
    });
}

// The start each slice, CSS and cDMA wait belongs to: the last start before it on the same slice,
// or on the CSS or cDMA engine. Built in program order over the whole function.
static DenseMap<Operation *, Operation *> matchWaitsToStarts(FunctionOpInterface funcOp) {
    DenseMap<Operation *, Operation *> waitToStart;
    DenseMap<int64_t, Operation *> lastSlice;
    Operation *lastCss = nullptr;
    Operation *lastCdma = nullptr;
    funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto start = dyn_cast<torq_hw::SliceStartOp>(op)) {
            lastSlice[start.getId().getSExtValue()] = op;
        }
        else if (auto wait = dyn_cast<torq_hw::SliceWaitOp>(op)) {
            if (Operation *start = lastSlice.lookup(wait.getId().getSExtValue())) {
                waitToStart[op] = start;
            }
        }
        else if (isa<torq_hw::CSSStartOp>(op)) {
            lastCss = op;
        }
        else if (isa<torq_hw::CSSWaitOp>(op) && lastCss) {
            waitToStart[op] = lastCss;
        }
        else if (isa<torq_hw::CDMAStartOp>(op)) {
            lastCdma = op;
        }
        else if (isa<torq_hw::CDMAWaitOp>(op) && lastCdma) {
            waitToStart[op] = lastCdma;
        }
    });
    return waitToStart;
}

// A task that holds exactly one transfer in one direction, and nothing else.
template <typename CfgOpT, typename StartOpT, typename WaitOpT>
static CfgOpT getSoleTransfer(torq_hw::NssTaskOp task) {
    CfgOpT cfg;
    for (Operation &op : task.getBody().front()) {
        if (auto c = dyn_cast<CfgOpT>(op)) {
            if (cfg) {
                return nullptr; // A task holds one descriptor per direction.
            }
            cfg = c;
        }
        else if (!isa<StartOpT, WaitOpT>(op)) {
            return nullptr;
        }
    }
    return cfg;
}

static torq_hw::DmaOutCfgOp getSoleOutput(torq_hw::NssTaskOp task) {
    return getSoleTransfer<torq_hw::DmaOutCfgOp, torq_hw::DmaOutStartOp, torq_hw::DmaOutWaitOp>(task
    );
}

static torq_hw::DmaInCfgOp getSoleInput(torq_hw::NssTaskOp task) {
    return getSoleTransfer<torq_hw::DmaInCfgOp, torq_hw::DmaInStartOp, torq_hw::DmaInWaitOp>(task);
}

// Everything the transfer reads must already be defined where we want to put it. Pulling a
// transfer earlier moves its config op, and the config takes the two memrefs as operands.
//
// Those operands are usually produced right before the transfer by an accessor such as
// torq_hl.get_block or memref.subview. Those only compute an address, so we can carry them along
// instead of giving up on the pair. Anything that touches memory stays where it is.
//
// Returns the ops to move, in their current order, or nothing if the pair is not possible.
static std::optional<SmallVector<Operation *>>
collectDefinitionsToHoist(Operation *cfg, Operation *target, DominanceInfo &dom) {
    SmallVector<Operation *> toMove;
    llvm::SmallPtrSet<Operation *, 8> seen;
    SmallVector<Value> worklist(cfg->getOperands().begin(), cfg->getOperands().end());

    while (!worklist.empty()) {
        Value v = worklist.pop_back_val();
        if (dom.properlyDominates(v, target)) {
            continue;
        }

        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != target->getBlock() ||
            !(isMemoryEffectFree(def) || isAddressAccessor(def))) {
            LLVM_DEBUG({
                llvm::dbgs() << "torq-pair-dma-channels: cannot hoist ";
                if (def) {
                    llvm::dbgs() << def->getName();
                }
                else {
                    llvm::dbgs() << "<block arg>";
                }
                llvm::dbgs() << "\n";
            });
            return std::nullopt;
        }

        if (seen.insert(def).second) {
            toMove.push_back(def);
            worklist.append(def->getOperands().begin(), def->getOperands().end());
        }
    }

    // Keep the original order so the ops still see each other in the right order.
    llvm::sort(toMove, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    return toMove;
}

// Put the ops in cfg, start, wait order. The hardware only looks at the start and wait bits, so the
// order does not reach it, but everything that reads this IR in order does care: with the waits
// left in the middle the two transfers still read as one after the other.
static void orderForConcurrentStart(Block &body) {
    SmallVector<Operation *> cfgs, starts, waits;
    for (Operation &op : body) {
        if (isa<torq_hw::DmaInCfgOp, torq_hw::DmaOutCfgOp>(op)) {
            cfgs.push_back(&op);
        }
        else if (isa<torq_hw::DmaInStartOp, torq_hw::DmaOutStartOp>(op)) {
            starts.push_back(&op);
        }
        else if (isa<torq_hw::DmaInWaitOp, torq_hw::DmaOutWaitOp>(op)) {
            waits.push_back(&op);
        }
    }
    for (Operation *op : cfgs) {
        op->moveBefore(&body, body.end());
    }
    for (Operation *op : starts) {
        op->moveBefore(&body, body.end());
    }
    for (Operation *op : waits) {
        op->moveBefore(&body, body.end());
    }
}

class PairDmaChannelsPass : public impl::PairDmaChannelsBase<PairDmaChannelsPass> {
  public:
    PairDmaChannelsPass() = default;
    PairDmaChannelsPass(const PairDmaChannelsPass &pass) {}

    void runOnOperation() override;

  private:
    unsigned paired_ = 0;
    unsigned blockedByAddress_ = 0;
    unsigned blockedByOperands_ = 0;
};

void PairDmaChannelsPass::runOnOperation() {
    auto funcOp = getOperation();
    DominanceInfo dom(funcOp);
    DenseMap<Operation *, Operation *> waitToStart = matchWaitsToStarts(funcOp);

    funcOp.walk([&](Block *block) {
        for (auto it = block->begin(); it != block->end(); ++it) {
            auto outTask = dyn_cast<torq_hw::NssTaskOp>(&*it);
            if (!outTask) {
                continue;
            }
            auto outCfg = getSoleOutput(outTask);
            if (!outCfg) {
                continue;
            }

            TransferRanges outRanges = transferRanges(outCfg);
            if (!outRanges.lram.valid() || !outRanges.xram.valid()) {
                continue;
            }

            // Look ahead for an input we can pull up next to this output. Prefer the biggest one:
            // a pair hides the shorter of the two transfers, so a tiny input next to a large
            // output buys almost nothing.
            torq_hw::NssTaskOp bestTask;
            torq_hw::DmaInCfgOp bestCfg;
            SmallVector<Operation *> bestHoist;
            Hazards passed;

            auto scan = std::next(it);
            for (unsigned k = 0; k < lookahead && scan != block->end(); ++k, ++scan) {
                if (auto candTask = dyn_cast<torq_hw::NssTaskOp>(&*scan)) {
                    if (auto candCfg = getSoleInput(candTask)) {
                        TransferRanges candRanges = transferRanges(candCfg);

                        // The input ends up writing LRAM at the same time the output reads it, and
                        // it ends up reading XRAM before everything it moves over. Both sides have
                        // to be clear.
                        bool safe = candRanges.lram.valid() && candRanges.xram.valid() &&
                                    !candRanges.lram.overlaps(outRanges.lram) &&
                                    !candRanges.xram.overlaps(outRanges.xram) &&
                                    passed.clearOf(candRanges);

                        std::optional<SmallVector<Operation *>> hoist;
                        if (safe) {
                            hoist = collectDefinitionsToHoist(candCfg, outTask, dom);
                            if (!hoist) {
                                safe = false;
                                ++blockedByOperands_;
                            }
                        }
                        else {
                            ++blockedByAddress_;
                        }

                        // A pair hides the shorter of the two transfers, so the biggest input
                        // hides the most time.
                        if (safe && (!bestCfg || dmaNdlTransferBytes(candCfg.getWriteNdl()) >
                                                     dmaNdlTransferBytes(bestCfg.getWriteNdl()))) {
                            bestTask = candTask;
                            bestCfg = candCfg;
                            bestHoist = *hoist;
                        }
                    }
                }
                collectRanges(&*scan, waitToStart, passed);
            }

            if (!bestTask) {
                LLVM_DEBUG(
                    llvm::dbgs() << "torq-pair-dma-channels: NO PARTNER for out "
                                 << dmaNdlTransferBytes(outCfg.getReadNdl()) << " B\n"
                );
                continue;
            }

            LLVM_DEBUG(
                llvm::dbgs() << "torq-pair-dma-channels: out "
                             << dmaNdlTransferBytes(outCfg.getReadNdl()) << " B with in "
                             << dmaNdlTransferBytes(bestCfg.getWriteNdl()) << " B\n"
            );

            for (Operation *op : bestHoist) {
                op->moveBefore(outTask);
            }

            Block &dst = outTask.getBody().front();
            Block &src = bestTask.getBody().front();
            dst.getOperations().splice(dst.end(), src.getOperations());
            bestTask.erase();
            orderForConcurrentStart(dst);
            ++paired_;
        }
    });

    LLVM_DEBUG(
        llvm::dbgs() << "torq-pair-dma-channels: paired " << paired_ << ", blocked by address "
                     << blockedByAddress_ << ", blocked by operands " << blockedByOperands_ << "\n"
    );
}

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createPairDmaChannelsPass() {
    return std::make_unique<PairDmaChannelsPass>();
}

} // namespace mlir::syna::torq
