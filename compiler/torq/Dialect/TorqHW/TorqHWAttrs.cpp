// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWDialect.h"

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

#define GET_ATTRDEF_CLASSES
#include "torq/Dialect/TorqHW/TorqHWAttrs.cpp.inc"
#include "torq/Dialect/TorqHW/TorqHWEnums.cpp.inc"

namespace mlir::syna::torq_hw {

void TorqHWDialect::initializeTorqHWAttrs() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "torq/Dialect/TorqHW/TorqHWAttrs.cpp.inc"
        >();
}

SliceCFGAttr SliceCFGAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<ALUOp0Mode> alu_op0_mode,
    ArrayRef<ALUOp1Mode> alu_op1_mode, uint32_t alu_d_unsigned, uint32_t alu_w_unsigned,
    ACTMode act_mode, ArrayRef<uint32_t> act_lsh, int32_t act_rsh, int32_t act_clip_min,
    int32_t act_clip_max, int32_t act_zero_point, uint32_t no_p_clear, uint32_t no_p_output,
    uint32_t kernel_left, uint32_t kernel_right, uint32_t kernel_top, uint32_t kernel_bottom,
    int32_t pad_left, int32_t pad_right, int32_t pad_top, int32_t pad_bottom, int32_t pad_value,
    int32_t stride, int32_t stride_offset, uint32_t act_sum_bits, ArrayRef<int32_t> table
) {
    return get(
        context, alu_op0_mode, alu_op1_mode, alu_d_unsigned, alu_w_unsigned, act_mode, act_lsh,
        act_rsh, act_clip_min, act_clip_max, act_zero_point, no_p_clear, no_p_output, kernel_left,
        kernel_right, kernel_top, kernel_bottom, pad_left, pad_right, pad_top, pad_bottom,
        pad_value, stride, stride_offset, {} /*act_round_mode*/, {} /*weight_format*/,
        {} /*alu_disable*/, {} /*act_disable*/, {} /*alu_format*/, {} /*act_format*/, act_sum_bits,
        table
    );
}

SliceCFGAttr SliceCFGAttr::get(
    ::mlir::MLIRContext *context, ArrayRef<ALUOp0Mode> alu_op0_mode,
    ArrayRef<ALUOp1Mode> alu_op1_mode, uint32_t alu_d_unsigned, uint32_t alu_w_unsigned,
    ACTMode act_mode, ArrayRef<uint32_t> act_lsh, int32_t act_rsh, int32_t act_clip_min,
    int32_t act_clip_max, int32_t act_zero_point, uint32_t alu_disable, uint32_t act_disable,
    NumberFormat alu_format, NumberFormat act_format, uint32_t act_sum_bits
) {
    return get(
        context, alu_op0_mode, alu_op1_mode, alu_d_unsigned, alu_w_unsigned, act_mode, act_lsh,
        act_rsh, act_clip_min, act_clip_max, act_zero_point, {} /*no_p_clear*/, {} /*no_p_output*/,
        {} /*kernel_left*/, {} /*kernel_right*/, {} /*kernel_top*/, {} /*kernel_bottom*/,
        {} /*pad_left*/, {} /*pad_right*/, {} /*pad_top*/, {} /*pad_bottom*/, {} /*pad_value*/,
        {} /*stride*/, {} /*stride_offset*/, {} /*act_round_mode*/, {} /*weight_format*/,
        alu_disable, act_disable, alu_format, act_format, act_sum_bits, {} /*table*/
    );
}

SliceCFGAttr SliceCFGAttr::get(
    ::mlir::MLIRContext *context, uint32_t alu_disable, uint32_t act_disable, int32_t pad_value
) {
    return get(
        context, {} /* alu_op0_mode */, {} /* alu_op1_mode */, {} /* alu_d_unsigned */,
        {} /* alu_w_unsigned */, {} /* act_mode */, {0, 0, 0, 0} /* act_lsh */, {} /* act_rsh */,
        {} /* act_clip_min */, {} /* act_clip_max */, {} /* act_zero_point */, {} /* no_p_clear */,
        {} /* no_p_output */, {} /* kernel_left */, {} /* kernel_right */, {} /* kernel_top */,
        {} /* kernel_bottom */, {} /* pad_left */, {} /* pad_right */, {} /* pad_top */,
        {} /* pad_bottom */, pad_value, {} /* stride */, {} /* stride_offset */,
        {} /* act_round_mode */, {} /* weight_format */, alu_disable, act_disable,
        {} /* alu_format */, {} /* act_format */, {} /* act_sum_bits */, {} /*table*/
    );
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, SliceCFGAttr &cfg) {
    os << "SliceCfg(\n";
    os << "  alu_op0_mode: [";
    std::string sep;
    for (auto mode : cfg.getAluOp0Mode()) {
        os << sep << mode;
        sep = ", ";
    }
    sep = "], alu_op1_mode: [";
    for (auto mode : cfg.getAluOp1Mode()) {
        os << sep << mode;
        sep = ", ";
    }
    os << "]\n  alu_d_unsigned: " << cfg.getAluDUnsigned() << ",";
    os << "  alu_w_unsigned: " << cfg.getAluWUnsigned() << "\n";
    os << "  act_mode: " << cfg.getActMode() << ",";
    const auto alsh = cfg.getActLsh();
    os << "  act_lsh: [" << alsh[0] << "," << alsh[1] << "," << alsh[2] << "," << alsh[3] << "],";
    os << "  act_rsh: " << static_cast<int>(cfg.getActRsh()) << ",";
    os << "  act_clip_min: " << cfg.getActClipMin() << ",";
    os << "  act_clip_max: " << cfg.getActClipMax() << ",";
    os << "  act_zero_point: " << cfg.getActZeroPoint() << "\n";
    os << "  no_p_clear: " << static_cast<int>(cfg.getNoPClear()) << ",";
    os << "  no_p_output: " << static_cast<int>(cfg.getNoPOutput()) << "\n";
    os << "  kernel: " << "{l:" << cfg.getKernelLeft() << ", r:" << cfg.getKernelRight()
       << ", t:" << cfg.getKernelTop() << ", b:" << cfg.getKernelBottom() << "},";
    os << "  pad: " << "{l:" << cfg.getPadLeft() << ", r:" << cfg.getPadRight()
       << ", t:" << cfg.getPadTop() << ", b:" << cfg.getPadBottom() << "},";
    os << "  pad_value: " << cfg.getPadValue() << ",";
    os << "  stride: " << cfg.getStride() << ", stride_offset: " << cfg.getStrideOffset() << "\n";
    os << "  act_round_mode: " << cfg.getActRoundMode() << ",";
    os << "  weight_format: " << cfg.getWeightFormat() << ",";
    os << "  alu_disable: " << cfg.getAluDisable() << ",";
    os << "  act_disable: " << cfg.getActDisable() << ",";
    os << "  alu_format: " << cfg.getAluFormat() << ",";
    os << "  act_format: " << cfg.getActFormat() << ",";
    os << "  act_sum_bits: " << cfg.getActSumBits() << "\n";
    os << "  table_size: " << cfg.getTable().size() << "\n";
    os << ")\n";
    return os;
}

llvm::raw_ostream &printNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::MemNdlData *ndl) {
    const char *sep = "";
    os << type << (type == NdlType::REF ? " " : "");
    if (!ndl) {
        os << " : none\n";
        return os;
    }
    if (ndl->set_id) {
        os << ndl->set_id;
    }
    else {
        os << ' ';
    }
    os << ": ";
    auto printingType = DimType::H;
    for (auto &dim : ndl->dims) {
        if (dim.type != DimType::L && printingType == DimType::L) {
            os << "}";
        }
        os << (dim.type == DimType::L && printingType != DimType::L ? "{" : sep);
        sep = ",";
        if (dim.type == DimType::S && printingType != DimType::S) {
            os << "S(";
        }
        os << dim.tag << dim.count << ":";
        if (dim.getExprStride().has_value()) {
            os << "expr";
        }
        else {
            os << dim.getIntStride();
        }
        printingType = dim.type;
    }
    if (printingType == DimType::S) {
        os << ")";
    }
    if (ndl->offset) {
        os << " offset=" << ndl->offset;
    }
    if (ndl->sync_mode) {
        os << " sync=" << (char)ndl->sync_mode;
    }
    if (ndl->sync_nhd) {
        os << " nhdims=" << (int)ndl->sync_nhd;
    }
    os << "\n";
    return os;
}

llvm::raw_ostream &printNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::RegNdlData *ndl) {
    const char *sep = ",";
    if (!ndl) {
        os << type << " : none\n";
        return os;
    }
    os << type;
    if (ndl->set_id) {
        os << ndl->set_id;
    }
    else {
        os << ' ';
    }
    int count = 0;
    bool printingLDIms = true;
    for (auto &dim : ndl->dims) {
        if (dim.type != DimType::L && printingLDIms) {
            printingLDIms = false;
            os << "}";
        }
        os << (count++ == 0 ? ": {" : sep);
        os << dim.tag << dim.count << ":" << dim.stride;
    }
    os << "\n";
    return os;
}

static void printRegNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::Ndls &ndls) {
    if (auto ndl = ndls.getRegNdl(type)) {
        printNdl(os, type, ndl);
    }
}

static void printMemNdl(llvm::raw_ostream &os, NdlType type, const torq_hw::Ndls &ndls) {
    if (auto ndl = ndls.getMemNdl(type)) {
        printNdl(os, type, ndl);
    }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const torq_hw::Ndls &ndls) {
    printMemNdl(os, NdlType::REF, ndls);
    printMemNdl(os, NdlType::DEDR, ndls);
    printMemNdl(os, NdlType::DEWR, ndls);
    printMemNdl(os, NdlType::DEBR, ndls);
    printRegNdl(os, NdlType::CEDW, ndls);
    printRegNdl(os, NdlType::CEDR, ndls);
    printRegNdl(os, NdlType::CEWW, ndls);
    printRegNdl(os, NdlType::CEWR, ndls);
    printRegNdl(os, NdlType::ACBW, ndls);
    printRegNdl(os, NdlType::ACBR, ndls);
    printRegNdl(os, NdlType::CEPR, ndls);
    printRegNdl(os, NdlType::ACPR, ndls);
    printRegNdl(os, NdlType::ACPW, ndls);
    printMemNdl(os, NdlType::DEQW, ndls);
    return os;
}

std::optional<int64_t> MemDimAttr::getStrideAsI64(ArrayAttr symbolValues) const {

    // check if the affine map is a constant value, we can return it directly
    if (auto constStride =
            llvm::dyn_cast<AffineConstantExpr>(getStride().getAffineMap().getResult(0))) {
        return constStride.getValue();
    }

    SmallVector<Attribute, 1> result;
    bool hasPoison = false;
    auto foldResult =
        getStride().getAffineMap().constantFold(symbolValues.getValue(), result, &hasPoison);

    // the constant folding failed because there is some div by zero or other invalid operation
    if (hasPoison)
        return std::nullopt;

    // the constant folding didn't complete because there are missing infos
    if (failed(foldResult))
        return std::nullopt;

    auto integerAttr = llvm::dyn_cast<IntegerAttr>(result[0]);

    // folding resulted in some other types than integer (e.g. float)
    if (!integerAttr) {
        return std::nullopt;
    }

    return integerAttr.getInt();
}

MemNdlAttr MemNdlAttr::get(::mlir::MLIRContext *context, MemNdlData ndl, int64_t numSyms) {

    auto dimAttrs = llvm::map_to_vector(ndl.dims, [context, numSyms](auto data) {
        return MemDimAttr::get(context, data, numSyms);
    });

    return Base::get(
        context, ndl.type, dimAttrs, ndl.index, ndl.offset, ndl.set_id, ndl.sync_mode, ndl.sync_nhd
    );
}

MemNdlAttr MemNdlAttr::get(::mlir::MLIRContext *context, MemNdlData ndl) {
    auto dimAttrs = llvm::map_to_vector(ndl.dims, [context](auto data) {
        return MemDimAttr::get(context, data);
    });

    return Base::get(
        context, ndl.type, dimAttrs, ndl.index, ndl.offset, ndl.set_id, ndl.sync_mode, ndl.sync_nhd
    );
}

RegNdlAttr RegNdlAttr::get(::mlir::MLIRContext *context, RegNdlData ndl) {

    auto dimAttrs = llvm::map_to_vector(ndl.dims, [context](auto data) {
        return RegDimAttr::get(context, data);
    });

    return Base::get(context, ndl.type, dimAttrs, ndl.set_id);
}

} // namespace mlir::syna::torq_hw
