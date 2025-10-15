#pragma once

#include "mlir/IR/Builders.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

// Outline operations into a new torq_hl.program and insert a torq_hl.start_program to start the
// outlined program.
//
// The operations in targets must be in a valid order (i.e. if operation B depends on operation A,
// then A must be before B in the targets list). No values created by the operations being outlined
// must be used outside of the outlined program.

struct OutliningResults {
    torq_hl::ProgramOp program;
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    DenseMap<Value, Value> outputToInput;
};

FailureOr<OutliningResults> outlineProgram(
    OpBuilder &builder, std::string name, torq_hl::Executor executor,
    const SmallVector<Operation *> &targets, bool destinationStyle
);

} // namespace mlir::syna::torq