#include "torq/Utils/CodeSizeUtils.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

// Currently we use a fixed size for each block in a function, later we should compute the
// actual size from the operations it contains.
constexpr int kBlockCodeSizeMax = 0x280;

int getCodeSize(ArrayRef<Operation *> operations) { return kBlockCodeSizeMax; }

int getCodeSize(Block *block) { return kBlockCodeSizeMax; }

int getCodeOffset(Block *block) {

    int offset = 0;

    for (auto &prevBlock : block->getParent()->getBlocks()) {
        if (&prevBlock == block) {
            break;
        }

        offset += getCodeSize(&prevBlock);
    }

    return offset;
}

} // namespace mlir::syna::torq
