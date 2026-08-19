// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/DmaProfilingUtils.h"

#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cmath>

namespace mlir::syna::torq {

double mtuEfficiency(unsigned mtu, double delta) {
    double beats = static_cast<double>(1u << mtu);
    return beats / (beats + delta);
}

double dmaThroughputBytesPerCycle(unsigned mtu) {
    return TorqHw::get().getDmaThroughputBytesPerCycle() * mtuEfficiency(mtu);
}

uint64_t transferBytes(llvm::ArrayRef<int64_t> shape, int64_t elementSizeBytes) {
    uint64_t count = 1;
    for (auto dim : shape) {
        count *= static_cast<uint64_t>(dim);
    }
    return count * static_cast<uint64_t>(elementSizeBytes);
}

uint64_t estimateDmaCycles(uint64_t bytes, double throughputBytesPerCycle) {
    if (throughputBytesPerCycle <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::ceil(static_cast<double>(bytes) / throughputBytesPerCycle));
}

uint64_t dmaNdlTransferBytes(mlir::syna::torq_hw::DmaNdlAttr ndl) {
    if (!ndl) {
        return 0;
    }

    uint64_t bytes = 1;
    for (auto dim : ndl.getDims()) {
        bytes *= static_cast<uint64_t>(dim.getCount());
    }
    return bytes;
}

// CDMA throughput factor depends on the CSS/NSS hardware configuration.
static double cdmaThroughputFactor() {
    auto hw = TorqHw::get();
    if (hw.getCSSConfigName() == "coral_v1" && hw.getNSSConfigName() == "nss_v1") {
        return 0.36;
    }
    return 0.8;
}

uint64_t cdmaTransferBytes(mlir::Value src) {
    auto shapedSrcType = mlir::dyn_cast<mlir::ShapedType>(src.getType());
    return shapedSrcType ? static_cast<uint64_t>(getEncodedDataSizeElements(shapedSrcType)) : 0;
}

uint64_t estimateCdmaCycles(uint64_t bytes) {
    return std::max<uint64_t>(
        1, static_cast<uint64_t>(std::ceil(bytes / (16.0 * cdmaThroughputFactor())))
    );
}

void setDmaAsyncAttrs(
    mlir::Operation *srcOp, mlir::Operation *dstOp, uint64_t bytes, double throughputBytesPerCycle
) {
    mlir::Builder builder(dstOp->getContext());
    mlir::Attribute durAttr = srcOp->getAttr(kAsyncDurationAttr);
    mlir::Attribute bytesAttr = srcOp->getAttr(kAsyncBytesAttr);
    if (!durAttr) {
        durAttr = builder.getI64IntegerAttr(estimateDmaCycles(bytes, throughputBytesPerCycle));
    }
    if (!bytesAttr) {
        bytesAttr = builder.getI64IntegerAttr(bytes);
    }
    dstOp->setAttr(kAsyncDurationAttr, durAttr);
    dstOp->setAttr(kAsyncBytesAttr, bytesAttr);
}

void setCdmaAsyncAttrs(mlir::Operation *srcOp, mlir::Operation *dstOp, mlir::Value src) {
    mlir::Builder builder(dstOp->getContext());
    mlir::Attribute durAttr = srcOp->getAttr(kAsyncDurationAttr);
    mlir::Attribute bytesAttr = srcOp->getAttr(kAsyncBytesAttr);
    uint64_t bytes = cdmaTransferBytes(src);
    if (!durAttr) {
        durAttr = builder.getI64IntegerAttr(estimateCdmaCycles(bytes));
    }
    if (!bytesAttr) {
        bytesAttr = builder.getI64IntegerAttr(bytes);
    }
    dstOp->setAttr(kAsyncDurationAttr, durAttr);
    dstOp->setAttr(kAsyncBytesAttr, bytesAttr);
}

} // namespace mlir::syna::torq
