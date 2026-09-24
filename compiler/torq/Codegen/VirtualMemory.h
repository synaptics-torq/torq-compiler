#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>
#include <torq/Dialect/TorqHL/TorqHLOps.h>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::syna::torq {

// A swap convertVirtualToPhysicalMemRefs performs: a spill swaps the buffers out to the
// swapMemSpace (XRAM for LRAM), each to be swapped back in at its next use.
// a relocation swaps them all out and immediately back in, so they land at fresh addresses.
struct Swap {
    enum class Kind { Spill, Relocate };
    Kind kind;
    llvm::SmallVector<Value> buffers;
};

// The swaps to perform just before the op at each index into the function's op list, in
// the order they were decided: a relocation followed by a spill of one of the buffers it
// moved must be applied in that order. Empty when the layout fits as is.
using SwapSchedule = llvm::DenseMap<int, llvm::SmallVector<Swap>>;

// Rewrite the virtual memrefs of `memorySpace` to physical ones, performing the swaps of
// `schedule`; they are the only source of swap-outs. Precondition: `schedule` was planned
// for this IR, so every swap applies to a resident, unpinned buffer and every op's operands
// and results fit in `capacityBytes` next to what is resident.
LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, int capacityBytes, torq_hl::MemorySpace memorySpace,
    const SwapSchedule &schedule
);

} // namespace mlir::syna::torq
