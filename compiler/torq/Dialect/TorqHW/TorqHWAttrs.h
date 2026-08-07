// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include <optional>

#include "torq/Dialect/TorqHW/TorqHWEnums.h.inc"

namespace mlir::syna::torq_hw {

struct RegNdlDimData {
    DimType type;
    RegDimTag tag;
    int64_t count;
    int64_t stride;

    RegNdlDimData(DimType type, RegDimTag tag, int64_t count = 1, int64_t stride = 0)
        : type(type), tag(tag), count(count), stride(stride) {}
};

struct MemNdlDimData {
    DimType type;
    MemDimTag tag;
    int64_t count;

    MemNdlDimData(DimType type, MemDimTag tag, int64_t count, AffineExpr stride)
        : type(type), tag(tag), count(count), expr_(stride) {}

    MemNdlDimData(DimType type, MemDimTag tag, int64_t count = 1, int64_t stride = 0)
        : type(type), tag(tag), count(count), strideInt_(stride) {}

    AffineMapAttr getStrideAttr(int numSyms, MLIRContext *context) const;

    std::optional<int64_t> getIntStride() const { return strideInt_; }
    std::optional<AffineExpr> getExprStride() const { return expr_; }
    void setIntStride(int64_t stride) { strideInt_ = stride; }
    void setExprStride(AffineExpr stride) { expr_ = stride; }

  private:
    std::optional<int64_t> strideInt_{};
    std::optional<AffineExpr> expr_{};
};

using MemNdlDimsData = SmallVector<MemNdlDimData>;
using RegNdlDimsData = SmallVector<RegNdlDimData>;

struct MemNdlData {
    MemNdlData(
        NdlType type, MemNdlDimsData dims, int64_t offset = 0, int64_t set_id = 0,
        uint8_t sync_mode = 0, uint8_t sync_nhd = 0
    )
        : type(type), dims(dims), index(0), offset(offset), set_id(set_id), sync_mode(sync_mode),
          sync_nhd(sync_nhd) {}
    NdlType type;
    MemNdlDimsData dims;
    int64_t index;
    int64_t offset;
    int64_t set_id;
    uint8_t sync_mode;
    uint8_t sync_nhd;
};

struct RegNdlData {
    RegNdlData(NdlType type, RegNdlDimsData dims, int64_t set_id = 0)
        : type(type), dims(dims), set_id(set_id) {}
    NdlType type;
    RegNdlDimsData dims;
    int64_t set_id;
};

struct Ndls {
    const MemNdlData *getMemNdl(NdlType type, size_t index = 0, int64_t set_id = 0) const;
    MemNdlData *getMemNdl(NdlType type, size_t index = 0, int64_t set_id = 0);

    const RegNdlData *getRegNdl(NdlType type, size_t index = 0, int64_t set_id = 0) const;
    RegNdlData *getRegNdl(NdlType type, size_t index = 0, int64_t set_id = 0);

    void
    add(NdlType type, MemNdlDimsData dims, int64_t offset = 0, int64_t set_id = 0,
        uint8_t sync_mode = 0, uint8_t sync_nhd = 0) {
        add(MemNdlData(type, dims, offset, set_id, sync_mode, sync_nhd));
    }
    void add(NdlType type, RegNdlDimsData dims, int64_t set_id = 0) {
        add(RegNdlData(type, dims, set_id));
    }

    void add(const MemNdlData &ndl);
    void add(const RegNdlData &ndl);

    SmallVector<MemNdlData> memNdls;
    SmallVector<RegNdlData> regNdls;
};

class SliceCFGAttr;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SliceCFGAttr &cfg);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const torq_hw::Ndls &ndls);
llvm::raw_ostream &printNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::RegNdlData *ndl);
llvm::raw_ostream &printNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::MemNdlData *ndl);

} // namespace mlir::syna::torq_hw

#define GET_ATTRDEF_CLASSES
#include "torq/Dialect/TorqHW/TorqHWAttrs.h.inc"
