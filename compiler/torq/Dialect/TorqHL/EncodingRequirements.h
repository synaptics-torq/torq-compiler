#pragma once

#include "mlir/IR/Dialect.h"
#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"
#include "torq/Utils/EncodingUtils.h"

namespace mlir::syna::torq_hl {

// inserts conversions before and after the op to ensure that all inputs and outputs
// have an encoding that matches the requirements of the operation, returns failure
// if the operation there was an error inserting the conversions, or true if any
// conversions were inserted and false if the op already matched the requirements
FailureOr<bool> encodeKernelInputOutputs(
    DestinationStyleOpInterface op, const KernelEncoding &encoding, RewriterBase &rewriter,
    Value initValue = Value(0)
);

// returns the encoding for a given operand of the specified op (this may be something
// explicitly set in encoding or an encoding requirement with no constraints)
KernelTensorEncoding getOperandEncoding(const KernelEncoding &encoding, OpOperand &opOperand);

EncodingRequirements toTensorEncodingRequirementsAttr(KernelTensorEncoding reqs);

} // namespace mlir::syna::torq_hl
