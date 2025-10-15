#include "torq/Codegen/Pool.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-pool"

namespace mlir::syna::torq {

Pool::Pool(int sizeBytes, int reservedBytes, int wordSizeBytes, int largeAllocSizeBytes)
    : wordSizeBytes_(wordSizeBytes), used_(sizeBytes / wordSizeBytes, false),
      largeAllocSizeBytes_(largeAllocSizeBytes), reservedBytes_(reservedBytes) {
    clear();
}

int Pool::totalSize() { return used_.size() * wordSizeBytes_; }

int Pool::usedSize() { return usedSizeWords_ * wordSizeBytes_; }

int Pool::usableSize() { return totalSize() - reservedBytes_; }

void Pool::clear() {
    for (int i = 0; i < (reservedBytes_ + wordSizeBytes_ - 1) / wordSizeBytes_; i++) {
        used_[i] = true;
    }
    allocations_.clear();
    usedSizeWords_ = 0;
}

FailureOr<int> Pool::allocate(Value value) {

    int sizeBytes = getEncodedTotalSizeBytes(mlir::cast<MemRefType>(value.getType()));

    int sizeWords = (sizeBytes + wordSizeBytes_ - 1) / wordSizeBytes_;

    // allocate small objects at the beginning of the pool and large at the end
    // this prevents some fragmentation
    if (sizeBytes < largeAllocSizeBytes_) {
        for (int i = 0; i < used_.size() - sizeWords; i++) {
            if (!used_[i]) {
                bool found = true;
                for (int j = 0; j < sizeWords; j++) {
                    if (used_[i + j]) {
                        found = false;
                        i = i + j;
                        break;
                    }
                }

                if (found) {
                    LLVM_DEBUG({
                        llvm::dbgs() << "Allocating small object " << sizeBytes << " bytes at "
                                     << i * wordSizeBytes_ << "\n";
                    });
                    for (int j = 0; j < sizeWords; j++) {
                        used_[i + j] = true;
                    }
                    usedSizeWords_ += sizeWords;
                    allocations_[value] = {i, sizeWords};

                    return i * wordSizeBytes_;
                }
            }
        }
    }
    else {
        for (int i = used_.size(); i > sizeWords; i--) {

            int startIndex = i - sizeWords;

            if (!used_[startIndex]) {
                bool found = true;
                for (int j = 0; j < sizeWords; j++) {
                    if (used_[startIndex + j]) {
                        found = false;
                        i = startIndex + j;
                        break;
                    }
                }

                if (found) {
                    LLVM_DEBUG({
                        llvm::dbgs() << "Allocating large object " << sizeBytes << " bytes at "
                                     << startIndex * wordSizeBytes_ << "\n";
                    });
                    for (int j = 0; j < sizeWords; j++) {
                        used_[startIndex + j] = true;
                    }
                    usedSizeWords_ += sizeWords;
                    allocations_[value] = {startIndex, sizeWords};

                    return startIndex * wordSizeBytes_;
                }
            }
        }
    }

    LLVM_DEBUG({ llvm::dbgs() << "Failed to allocate object " << sizeBytes << " bytes\n"; });

    return failure();
}

void Pool::free(Value value) {
    auto allocation = allocations_[value];

    for (int i = 0; i < allocation.size; i++) {
        used_[allocation.baseAddr + i] = false;
    }
    usedSizeWords_ -= allocation.size;
    allocations_.erase(value);
}

void Pool::dump() {
    llvm::dbgs() << "Pool totalSize: " << totalSize() << " bytes, usable size: " << usableSize()
                 << " bytes, usedSize: " << usedSize() << " bytes, word size: " << wordSizeBytes_
                 << " bytes, current contents:\n";
    for (auto [value, range] : allocations_) {
        llvm::dbgs() << value << " at " << range.baseAddr * wordSizeBytes_
                     << " size: " << range.size * wordSizeBytes_ << " bytes\n";
    }
    llvm::dbgs() << "\n";
}

} // namespace mlir::syna::torq