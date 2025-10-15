#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Support/LLVM.h>

namespace mlir::syna::torq {

// A Pool is an object that can be used to generate addresses for buffers that are stored in
// finite size memory. The addresses returned by the allocate function ensure that no two
// active buffers overlap.
//
// The pool allocates addresses from a linear address space [reserved, sizeBytes]. Allocations
// are aligned to word boundaries.
//
// Internally the pool allocates large and small objects separately to reduce fragmentation.
struct Pool {

    struct Allocation {
        int baseAddr; // in words
        int size;     // size in words
    };

    Pool(int sizeBytes, int reservedBytes, int wordSizeBytes, int largeAllocSizeBytes = 1024);

    // returns the total size of the currently allocated buffers
    int usedSize();

    // returns total number of bytes that can be allocated in the pool (excluding reserved bytes)
    int totalSize();

    // returns the total size of the buffer that is not reserved
    int usableSize();

    // allocates an address for the given memref (fails if no address can be found, this can
    // happen when the memory is fragmented or there is not enough space in memory to fit the
    // buffer)
    FailureOr<int> allocate(Value value);

    // free a buffer
    void free(Value value);

    // dump the current state of the pool to llvm::dbgs()
    void dump();

    // free all the allocations
    void clear();

  private:
    const int wordSizeBytes_; // number of bytes that compose a word
    SmallVector<bool> used_;  // one entry per word of the pool
    DenseMap<Value, Allocation> allocations_;
    const int largeAllocSizeBytes_; // minimal size in bytes of a large allocations
    int usedSizeWords_{0};          // number of words that are currently in use
    const int
        reservedBytes_; // number of bytes at the start of the address space that cannot be used
};

} // namespace mlir::syna::torq