// Copyright 2024 Synaptics Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/StorageUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/MathExtras.h"

namespace mlir::syna {

int64_t getStorageBitWidth(int64_t elementBitWidth) {
    if (elementBitWidth == 4) {
        return 4;
    }
    return torq::div_ceil(elementBitWidth, 8) * 8;
}

int64_t getStorageSizeBytes(int64_t elementCount, int64_t elementBitWidth) {
    return llvm::divideCeil(elementCount * getStorageBitWidth(elementBitWidth), int64_t{8});
}

} // namespace mlir::syna
