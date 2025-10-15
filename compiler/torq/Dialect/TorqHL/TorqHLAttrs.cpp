// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHL/TorqHLAttrs.h"
#include "torq/Dialect/TorqHL/TorqHLDialect.h"

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#define GET_ATTRDEF_CLASSES
#include "torq/Dialect/TorqHL/TorqHLAttrs.cpp.inc"
#include "torq/Dialect/TorqHL/TorqHLEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "torq/Dialect/TorqHL/TorqHLTypes.cpp.inc"

namespace mlir::syna::torq_hl {

LogicalResult TensorEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, MemorySpace memSpace,
    ::llvm::ArrayRef<int64_t> counts, ::llvm::ArrayRef<int64_t> strides, int64_t padding
) {

    if (counts.size() != strides.size()) {
        emitError() << "'counts' and 'strides' must have the same rank";
        return failure();
    }

    int minStride = 1;
    for (int i = counts.size() - 1; i >= 0; i--) {
        if (counts[i] <= 0) {
            emitError() << "'counts' must be positive at dimension " << i;
            return failure();
        }

        if (strides[i] < minStride) {
            emitError() << "'strides' at least " << minStride << " at dimension " << i;
            return failure();
        }
        minStride = std::max<int64_t>(counts[i] * minStride, strides[i]);
    }

    if (padding < 0) {
        emitError() << "'padding' must be non-negative";
        return failure();
    }

    return success();
}

void TorqHLDialect::initializeTorqHLAttrs() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "torq/Dialect/TorqHL/TorqHLAttrs.cpp.inc"
        >();

    addTypes<
#define GET_TYPEDEF_LIST
#include "torq/Dialect/TorqHL/TorqHLTypes.cpp.inc"
        >();
}

} // namespace mlir::syna::torq_hl
