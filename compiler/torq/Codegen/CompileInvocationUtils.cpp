#include "torq/Codegen/CompileInvocationUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "torq-compile-invocation-utils"

namespace mlir::syna::torq {

FailureOr<torq_hl::StartProgramOp> findStartProgramOp(torq_hl::CreateInvocationOp createInvocationOp
) {

    for (auto &use : createInvocationOp.getInvocation().getUses()) {
        auto startProgramOp = dyn_cast<torq_hl::StartProgramOp>(use.getOwner());

        if (!startProgramOp) {
            continue;
        }

        // the invocation may be used as argument of a start op, we don't want that
        if (use.getOperandNumber() != startProgramOp.getInvocationMutable().getOperandNumber()) {
            continue;
        }

        return startProgramOp;
    }

    return createInvocationOp.emitError("No start_program operation found for invocation");
}

LogicalResult updateCode(DescGen &_npu, uint32_t xramAddress, SmallVector<int8_t> &code) {

    const torq_bitstream_segment_t *segment = _npu.getBitstream();

    while (segment) {

        if (segment->xram_addr == AddressConstants::NONE) {
            segment = segment->next;
            continue;
        }

        LLVM_DEBUG({
            std::string lramAddrString = llvm::formatv("{0:x+8}", segment->lram_addr);
            std::string xramAddrString = llvm::formatv("{0:x+8}", segment->xram_addr);

            llvm::dbgs() << " serializing code segment with "
                         << " xram address = " << xramAddrString << " (size = " << segment->size
                         << ")\n";

            std::stringstream ss;
            for (size_t i = 0; i < segment->size; i++) {
                ss << std::hex << "" << (int)segment->data[i];
            }
            llvm::dbgs() << "    data: " << ss.str() << "\n";
        });

        if (segment->xram_addr < xramAddress) {
            return failure();
        }

        auto offset = segment->xram_addr - xramAddress;

        if ((offset + segment->size) > code.size()) {
            return failure();
        }

        std::copy(segment->data, segment->data + segment->size, code.begin() + offset);

        segment = segment->next;
    }

    return success();
}

} // namespace mlir::syna::torq