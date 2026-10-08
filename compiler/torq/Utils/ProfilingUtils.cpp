// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Utils/ProfilingUtils.h"

#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/TorqHw.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

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

size_t ndlCycle(torq_hw::RegNdlAttr attr) {
    size_t cycle = 1;
    for (auto dim : attr.getDims()) {
        if (dim.getType() == torq_hw::DimType::H) {
            cycle *= dim.getCount();
        }
    }
    return cycle;
}

size_t ndlCycle(torq_hw::MemNdlAttr attr) {
    size_t cycle = 1;
    uint32_t lWriteSize = 1;
    uint32_t xLineSize = 1;
    bool usingSdims = false;
    for (auto dim : attr.getDims()) {
        switch (dim.getType()) {
        case torq_hw::DimType::H:
            cycle *= dim.getCount();
            break;
        case torq_hw::DimType::L:
            lWriteSize *= dim.getCount();
            break;
        case torq_hw::DimType::S:
            if (dim.getTag() == torq_hw::MemDimTag::B || dim.getTag() == torq_hw::MemDimTag::X) {
                xLineSize *= dim.getCount();
                usingSdims = true;
            }
            break;
        }
    }

    // Use of SDIMs (DEQW) with small line-size (X-dim) can add overhead to the cycle count.
    // Use an approximation to account for this overhead which is normally negligible
    // except for very small line sizes.
    if (usingSdims) {
        if (xLineSize < lWriteSize) {
            // Each low-level write is split into multiple cycles
            cycle = cycle * div_ceil(lWriteSize, xLineSize);
        }
        else {
            // Determine the number of writes crossing X-boundary (will require one extra cycle)
            size_t g = std::gcd(lWriteSize, xLineSize);
            size_t misalignmentPeriod = (lWriteSize - 1) / g;
            size_t crossingWrites = (cycle * g / xLineSize) * misalignmentPeriod;
            cycle += crossingWrites;
        }
    }

    return cycle;
}

uint64_t estimateSliceTaskCycles(torq_hw::SliceTaskOp sliceTaskOp) {
    using torq_hw::NdlType;
    std::map<NdlType, size_t> memNdlCycles;
    for (auto &ndl : sliceTaskOp.getMemNdls())
        memNdlCycles[ndl.getType()] += ndlCycle(ndl);
    std::map<NdlType, size_t> regNdlCycles;
    for (auto &ndl : sliceTaskOp.getRegNdls())
        regNdlCycles[ndl.getType()] = ndlCycle(ndl);

    size_t dedr = memNdlCycles[NdlType::DEDR];
    size_t dewr = memNdlCycles[NdlType::DEWR];
    size_t debr = memNdlCycles[NdlType::DEBR];
    size_t deqw = memNdlCycles[NdlType::DEQW];
    size_t cedw = regNdlCycles[NdlType::CEDW];
    size_t cedr = regNdlCycles[NdlType::CEDR];
    size_t ceww = regNdlCycles[NdlType::CEWW];
    size_t cewr = regNdlCycles[NdlType::CEWR];
    size_t cepr = regNdlCycles[NdlType::CEPR];
    size_t acbw = regNdlCycles[NdlType::ACBW];
    size_t acbr = regNdlCycles[NdlType::ACBR];
    size_t acpr = regNdlCycles[NdlType::ACPR];

    size_t d_bus_cycle = std::max({dedr, cedw, cedr});
    size_t w_bus_cycle = std::max({dewr, ceww, cewr});
    size_t parallel_d_w_cycle = std::max(d_bus_cycle, w_bus_cycle);

    size_t b_bus_cycle = std::max({debr, acbw, acbr});
    size_t q_bus_cycle = std::max(deqw, acpr);
    size_t parallel_b_q_cycle = std::max(b_bus_cycle, q_bus_cycle);

    size_t p_bus_cycle = cepr;

    return std::max({parallel_d_w_cycle, parallel_b_q_cycle, p_bus_cycle});
}

} // namespace mlir::syna::torq
