#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>
#include <torq/Dialect/TorqHL/TorqHLOps.h>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::syna::torq {

// One buffer convertVirtualToPhysicalMemRefs swaps out to the swapMemSpace (XRAM for
// LRAM), or back in at `offset`.
struct Swap {
    enum class Kind { Out, In };
    Kind kind;
    TypedValue<MemRefType> buffer;
    int64_t offset = -1;
    // a swap-out stores the contents if they are still read and no copy holds them; a
    // swap-in loads them if they are still read
    bool transfers = true;
    // the swap-in repeats the buffer's fill instead of loading the swap copy
    bool fill = false;
    // the swap-out evicts a filled buffer nothing has read since its fill ran: the fill
    // moves to the next swap-in
    bool cancelsFill = false;
};

// The swaps to perform just before the op at each index into the function's op list, in
// the order they were decided: a buffer swapped out and back in at one op (moved to make
// room) must be applied in that order. Empty when the layout fits as is.
using SwapSchedule = llvm::DenseMap<int, llvm::SmallVector<Swap>>;

// Rewrite the virtual memrefs of `memorySpace` to physical ones, performing the swaps of
// `schedule`; they are the only source of swap-outs and swap-ins, and the buffers they swap
// in get their planned address. Precondition: `schedule` was planned for this IR, so every
// swap applies to a resident, unpinned buffer and every op's operands and results fit in
// `capacityBytes` next to what is resident.
// This has no failure path and always returns success. lramSwapBytes
// (AssignAddressesPass.cpp) treats a successful plan as a successful allocation, so if a
// failure path is added here, update it or the estimate will disagree with the real
// allocation.
LogicalResult convertVirtualToPhysicalMemRefs(
    FunctionOpInterface funcOp, int capacityBytes, torq_hl::MemorySpace memorySpace,
    const SwapSchedule &schedule, int64_t addressBase
);

} // namespace mlir::syna::torq
