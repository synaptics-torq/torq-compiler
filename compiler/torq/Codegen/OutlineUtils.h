#pragma once

#include "mlir/IR/Builders.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"

namespace mlir::syna::torq {

// Returns all the operations that would be outlined if the list of targets are outlined (this
// includes all the nested operations of the target operations).
DenseSet<Operation *> getOutlinedOps(const SmallVector<Operation *> &targets);

// Returns all the values outside the outlined operations that are used by the outlined operations
SmallVector<Value> getOutlineOpsInputs(
    const SmallVector<Operation *> &targets, const DenseSet<Operation *> &targetsWithNested
);

// Returns all the values defined by the outlined operations that are used outside the outlined
// operations
SmallVector<Value> getOutlineOpsOutputs(
    const SmallVector<Operation *> &targets, const DenseSet<Operation *> &targetsWithNested
);

// Outline operations into a new torq_hl.program and insert a torq_hl.start_program to start the
// outlined program. This function doesn't remove the original operations.
//
// The code automatically computes the inputs of the program so that any operation inside the
// program that depends on an operation outside the program will have its input added to the
// program inputs.
//
// The code also automatically computes the outputs of the program so that any value defined inside
// the program that is used outside the program will have its output added to the program outputs.
//
// The operations in targets must be in a valid order (i.e. if operation B depends on operation A,
// then A must be before B in the targets list).
//
// The caller can specify with noOutputValues a set of values that should be put in the outputs
// even if they are used outside the program.
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

FailureOr<OutliningResults> outlineProgram(
    OpBuilder &builder, std::string name, torq_hl::Executor executor,
    const SmallVector<Operation *> &targets, bool destinationStyle,
    const SmallVector<Value> &inputs, const SmallVector<Value> &outputs
);

} // namespace mlir::syna::torq