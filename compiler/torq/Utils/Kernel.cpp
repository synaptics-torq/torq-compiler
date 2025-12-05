// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Kernel.h"
#include "torq/Dialect/TorqHW/TorqHWAttrs.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/MemoryUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-easy-kernel"

using namespace mlir::syna::torq_hw;
using namespace std;

namespace mlir::syna::torq {

static int denseDims(const LData &);
static void fuse(LData &, int count = -1);
static void vectorize(LData &, int vectorSize, int vectorStride);
static void subviewDim(LData &, int dimIndex, int offset, int count);
static void reshapeDim(LData &, int dimIndex, const std::vector<int> &sizes, bool allowNonMultiple);
static void partitionByIndexParity1D(LData &);
static void partitionByIndexParity2D(LData &);

// Compute the stride of dimension i by multiplying the dimensions up to i (excluded)
// or the expicit stride if present
// if i < 0 compute the stride (size) of the entire shape
static Stride computeStride(const Shape &dims, int i) {
    int itemSize = 1;
    Stride stride = itemSize;
    for (int j = dims.size() - 1; j >= 0; j--) {
        if (dims[j].stride.hasVal()) {
            stride = dims[j].stride;
        }
        if (j == i) {
            break;
        }
        assert(!dims[j].stride.exprVal.has_value() && "Inner stride expression not supported");
        stride.intVal = stride.intVal.value() * dims[j].count;
    }
    return Stride(stride);
}

// Compute number of elements in the shape by multiplying the dimensions
// if numDims >= 0 only the final numDims dimensions are considered
// return -1 if the shape is not dense
static int denseElementCount(const Shape &dims, int numDims = -1) {
    int count = 1;
    int startDim = numDims < 0 ? 0 : dims.size() - numDims;
    for (int j = dims.size() - 1; j >= startDim; j--) {
        const auto &dim = dims[j];
        if (dim.stride.exprVal.has_value()) {
            return -1;
        }
        if (dim.stride.intVal.has_value()) {
            int stride = dim.stride.intVal.value();
            if (stride != count) {
                return -1;
            }
        }
        count *= dim.count;
    }
    return count;
}

// LRAM scatter-gather block info
struct SGBlockInfo {
    int size{1};           // Number of elements in each sgGroups
    int sgGroups{1};       // Number of scatter-gather groups
    Stride stride{};       // Stride between the groups
    bool isEvenOddSplit{}; // True if the sgGroups represent an even-odd split
    int outerGroups{1};    // Number of elements in outer dimensions
    Stride outerStride{};  // Stride between the outer elements
};

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SGBlockInfo &SGBlockInfo) {
    os << "{ " << "size: " << SGBlockInfo.size << ", sgGroups: " << SGBlockInfo.sgGroups
       << ", stride: " << SGBlockInfo.stride << ", outerGroups: " << SGBlockInfo.outerGroups
       << ", outerStride: " << SGBlockInfo.outerStride
       << ", isEvenOddSplit: " << SGBlockInfo.isEvenOddSplit << " }";
    return os;
    ;
}

static bool isEvenOddSplit(const Shape &dims, SGBlockInfo &block) {
    if (dims.size() < 2) {
        return false;
    }
    const auto &dim1 = dims[dims.size() - 1];
    const auto &dim0 = dims[dims.size() - 2];
    bool dim1NaturalStride = !dim1.stride.exprVal.has_value() &&
                             (!dim1.stride.intVal.has_value() ||
                              (dim1.stride.intVal.has_value() && dim1.stride.intVal.value() == 1));
    if (dim1.count != 2 || dim1NaturalStride) {
        return false;
    }
    if (!(dim0.stride.intVal.has_value() && dim0.stride.intVal.value() == 1)) {
        return false;
    }
    block.sgGroups = 2;
    block.size = dim0.count;
    block.stride = dim1.stride;
    block.isEvenOddSplit = true;
    return true;
}

// Compute number of elements in the shape by multiplying the dimensions
// dims can be:
// - 1D dense or not dense (if up to sgGroups fitting use scatter-gather)
// - 2D dense or with non-natural stride in dims[0] (if up to sgGroups fitting use scatter-gather)
// - 3D dense or with non-natural stride in dim[1] and/or dims[2]
//      (if up to sgGroups fitting use scatter-gather)
// If dense or 2D with inner size > busWidthItems then use outerGroups to handle the entire tranfer
//
// The shape must be dense, or have up to sgGroups non-contiguous groups.
// In this latter case, the non-contiguous groups can only happen in dims[0]
static SGBlockInfo sgElementCount(const Shape &dims, int busWidthItems, int sgMax = -1) {
    if (dims.empty()) {
        return {};
    }
    assert(dims.size() <= 3 && "Shape with more than 3 dimensions not supported");
    // Compute the total number of elements in the shape excluding the first dimension
    SGBlockInfo blockInfo;
    if (isEvenOddSplit(dims, blockInfo)) {
        return blockInfo;
    }
    bool isRank3 = dims.size() == 3;
    int count = denseElementCount(Shape{dims.begin() + isRank3 + 1, dims.end()});
    assert(count >= 0 && "Shape is  not dense");

    //  If the first dimension has a non-natural stride get the stride and the number of groups
    const auto &dim = dims[isRank3 ? 1 : 0];
    if (dim.stride.exprVal.has_value() ||
        (dim.stride.intVal.has_value() && dim.stride.intVal.value() != count)) {
        // Non-natural stride specified
        if (sgMax >= 0 && dim.count <= sgMax && count <= busWidthItems / dim.count) {
            // Use scatter-gather
            blockInfo.sgGroups = dim.count;
            blockInfo.size = count;
            blockInfo.stride = dim.stride;
        }
        else {
            // Cannot use scatter-gather, use outer dim
            assert(count <= busWidthItems && "Inner dimension larger than bus width");
            blockInfo.size = count;
            blockInfo.outerGroups = dim.count;
            blockInfo.outerStride = dim.stride;
        }
    }
    else {
        // The first dimension is also contiguous
        blockInfo.sgGroups = 1;
        blockInfo.size = count * dim.count;
    }

    if (isRank3) {
        assert(false && "Scatter-gather for 3D shapes not supported yet");
        // Handle the outer dimension
        assert(blockInfo.outerGroups == 1 && "Scatter-gather was not applied");
        blockInfo.outerGroups = dims[0].count;
        blockInfo.outerStride = dims[0].stride;
    }
    else if (blockInfo.size > busWidthItems) {
        if (blockInfo.size % busWidthItems != 0) {
            // To be verified if there are cases where we can accept this
            llvm::errs() << "Error block size " << blockInfo.size
                         << " must be a multiple of bus width " << busWidthItems << "\n";
            assert(false && "Block size must be a multiple of bus width");
        }
        // Divide by bus width to get number of bus transfers
        blockInfo.outerGroups = div_ceil(blockInfo.size, busWidthItems);
        blockInfo.outerStride = Stride(busWidthItems);
        blockInfo.size = busWidthItems;
    }

    return blockInfo;
}

int elementCount(const Shape &shape) {
    int count = 1;
    for (const auto &dim : shape) {
        count *= dim.count;
    }
    return count;
}

// Compute the total number of H iterations in an NDL
static int iterationCount(const MemNdlDimsData &dims) {
    int count = 1;
    for (auto &dim : dims) {
        count *= dim.type == DimType::H ? dim.count : 1;
    }
    return count;
}

// Compute the total number of H iterations in a reg-NDL
static int iterationCount(const RegNdlDimsData &dims) {
    int count = 1;
    for (auto &dim : dims) {
        count *= dim.type == DimType::H ? dim.count : 1;
    }
    return count;
}

// Compute the total number of H iterations in an NDL data struct
static int iterationCount(const MemNdlData *ndl) {
    assert(ndl && "NDL not defined");
    return iterationCount(ndl->dims);
}

// Extract a subshape from the original shape with the given indexes
static Shape getSubShape(const Shape &shape, const Indexes &ix) {
    assert(ix.size() <= shape.size());
    return Shape(&shape[ix.size()], &shape[shape.size()]);
}

// return bus width in bytes
static int getBusWidth(NdlType type, DType dataType) {
    switch (type) {
    case NdlType::DEDR:
        return HwInfo::iram_seg_width;
    case NdlType::DEWR:
        return HwInfo::wram_seg_width;
    case NdlType::DEBR:
        // In floating point mode act doesn't have the scale value
        return isFloat(dataType) ? HwInfo::breg_width / 2 : HwInfo::breg_width;
    case NdlType::DEQW:
        return HwInfo::act_width * sizeof(int32_t);
    default:
        break;
    }
    assert(false && "Unknown NDL type");
    return 0;
}

static int getBusScatterGather(NdlType type) {
    switch (type) {
    case NdlType::DEDR:
        return 4;
    case NdlType::DEQW:
        return 2;
    default:
        break;
    }
    return 0;
}

// SliceCfg contains the same fields as SliceCFGAttr
struct SliceCfg {
    torq_hw::ALUOp0Mode alu_op0_mode[4];
    torq_hw::ALUOp1Mode alu_op1_mode[4];
    uint32_t alu_d_unsigned;
    uint32_t alu_w_unsigned;
    torq_hw::ACTMode act_mode;
    SmallVector<uint32_t> act_lsh{0, 0, 0, 0};
    uint8_t act_rsh;
    int32_t act_clip_min;
    int32_t act_clip_max;
    int32_t act_zero_point;
    uint8_t no_p_clear;
    uint8_t no_p_output;
    LRTBDim kernel;
    LRTBDim pad;
    int32_t pad_value{-1};
    int32_t stride{1};
    int32_t stride_offset;
    torq_hw::RoundingMode act_round_mode;
    torq_hw::WeightFormat weight_format;

    // aluDisable: use 16bit to control 256 ALUs, each bit controls 16 ALUs
    // bit is 1 to disable the 16-ALU group
    // 0x0 to enable all ALU accumulation, 0xffff for no operations
    uint32_t alu_disable;

    // actDisable: use 4bit to control 16 ACTs, Each bit controls 4 ACTs;
    // bit is 1 to disable the 4-ACT group.
    // 0x0 to enable all 16 ACT, 0xf for no operations
    uint32_t act_disable;
    torq_hw::NumberFormat alu_format{torq_hw::NumberFormat::I};
    torq_hw::NumberFormat act_format{torq_hw::NumberFormat::I};
    uint32_t act_sum_bits;

    SliceCFGAttr toSliceCFGAttr(MLIRContext *ctx) const {
        return SliceCFGAttr::get(
            ctx, {alu_op0_mode[0], alu_op0_mode[1], alu_op0_mode[2], alu_op0_mode[3]},
            {alu_op1_mode[0], alu_op1_mode[1], alu_op1_mode[2], alu_op1_mode[3]}, alu_d_unsigned,
            alu_w_unsigned, act_mode, act_lsh,                   //
            act_rsh, act_clip_min, act_clip_max, act_zero_point, //
            no_p_clear,
            no_p_output,                                                       //
            kernel.left, kernel.right, kernel.top, kernel.bottom,              //
            pad.left, pad.right, pad.top, pad.bottom, pad_value,               //
            stride, stride_offset, act_round_mode, weight_format,              //
            alu_disable, act_disable, alu_format, act_format, act_sum_bits, {} /*table*/
        );
    }
};

static void ndlToStr(NdlType type, const torq_hw::MemNdlData *ndl) {
    if (!ndl) {
        LLVM_DEBUG(llvm::dbgs() << type << ": NULL\n");
        return;
    }

    LLVM_DEBUG(
        llvm::dbgs() << type << ": "; for (auto &dim
                                           : ndl->dims) {
            llvm::dbgs() << dim.tag << "(" << dim.type << ")" << dim.count << "[";
            if (dim.getExprStride().has_value()) {
                llvm::dbgs() << "expr] ";
            }
            else {
                llvm::dbgs() << dim.getIntStride() << "] ";
            }
        } if (ndl->offset) {
            llvm::dbgs() << "offset=" << ndl->offset;
        } llvm::dbgs() << "\n";
    );
}

static void ndlToStr(NdlType type, const torq_hw::RegNdlData *ndl) {

    if (!ndl) {
        LLVM_DEBUG(llvm::dbgs() << type << ": NULL\n");
        return;
    }

    LLVM_DEBUG(llvm::dbgs() << type << ": "; for (auto &dim
                                                  : ndl->dims) {
        llvm::dbgs() << dim.tag << "(" << dim.type << ")" << dim.count << "[" << dim.stride << "] ";
    } llvm::dbgs() << "\n";);
}

static void debugData(const Data &data) { LLVM_DEBUG(llvm::dbgs() << data << "\n"); }

// Return the number of elements in the last dimension of the shape or 1 if the shape is empty
static int backDimCount(const Shape &shape) {
    if (shape.empty()) {
        return 1;
    }
    // Note: the last dimension doesn't necessarily have to be dense as long as the stride
    // is small enough to be handled by corresponding NDL.
    return shape.back().count;
}

int sizeofType(DType type) {
    switch (type) {
    case DType::uint8:
    case DType::int8:
        return 1;
    case DType::uint16:
    case DType::int16:
        return 2;
    case DType::uint32:
    case DType::int32:
        return 4;
    case DType::bf16:
        return 2;
    case DType::fp32:
        return 4;
    case DType::none:
        return 0;
    }
    assert(false && "Unknown data type");
}

DType getDType(mlir::Type mlirType) {
    if (mlirType.isInteger(8) || mlirType.isInteger(1)) {
        return DType::int8;
    }
    else if (mlirType.isInteger(16)) {
        return DType::int16;
    }
    else if (mlirType.isInteger(32)) {
        return DType::int32;
    }
    else if (mlirType.isBF16()) {
        return DType::bf16;
    }
    else if (mlirType.isF32()) {
        return DType::fp32;
    }
    // TODO, should we assert here? assert(false && "Unsupported  data type");
    return DType::none;
}

bool isFloat(DType type) { return type == DType::bf16 || type == DType::fp32; }

bool isInt(DType type) { return type != DType::none && !isFloat(type); }

bool isUnsigned(DType type) {
    return type == DType::uint8 || type == DType::uint16 || type == DType::uint32;
}

bool hasScale(DType type) { return isInt(type); }

int scaleBiasWidth(DType type) { return hasScale(type) ? 2 : 1; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DType dtype) {
    switch (dtype) {
    case DType::uint8:
        os << "uint8";
        break;
    case DType::uint16:
        os << "uint16";
        break;
    case DType::uint32:
        os << "uint32";
        break;
    case DType::int8:
        os << "int8";
        break;
    case DType::int16:
        os << "int16";
        break;
    case DType::int32:
        os << "int32";
        break;
    case DType::bf16:
        os << "bf16";
        break;
    case DType::fp32:
        os << "fp32";
        break;
    case DType::none:
        os << "none";
        break;
    }
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IterVar &iv) {
    if (iv.iterId() < 0) {
        os << "Const(" << iv.constIndex() << ")";
    }
    else if (iv.isReverse())
        os << "R(" << static_cast<int>(iv.iterId()) << ")";
    else
        os << static_cast<int>(iv.iterId());
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Indexes &indexes) {
    os << "[";
    for (size_t i = 0; i < indexes.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << indexes[i];
    }
    os << "]";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Iterator &iterator) {
    os << static_cast<Indexes>(iterator);
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Stride &stride) {
    os
        << (stride.hasVal()
                ? (stride.intVal.has_value() ? std::to_string(stride.intVal.value()) : "expr")
                : "-");
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Shape &shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << shape[i].count;
        if (shape[i].stride.hasVal()) {
            os << "@" << shape[i].stride;
        }
    }
    os << "]";
    return os;
}

//
// LRTBDim class
//

LRTBDim::LRTBDim(::llvm::ArrayRef<int64_t> lrtb) {
    assert(lrtb.size() == 4 && "Expected 4 values for LRTB dimension");
    left = lrtb[0];
    right = lrtb[1];
    top = lrtb[2];
    bottom = lrtb[3];
}

LRTBDim LRTBDim::symmetric(const HWDim &dim) {
    LRTBDim b;
    b.left = (dim.w - 1) / 2;
    b.right = dim.w - b.left - 1;
    b.top = (dim.h - 1) / 2;
    b.bottom = dim.h - b.top - 1;
    return b;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HWDim &shape) {
    os << "{h:" << shape.h << ", w:" << shape.w << "}";
    return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const LRTBDim &shape) {
    os << "{l:" << shape.left << ", r:" << shape.right << ", t:" << shape.top
       << ", b:" << shape.bottom << "}";
    return os;
}

//
// Data class
//

Data::Data(const string &name, const Shape &shape, DType elementType, int offs)
    : _name(name), _shape(shape), _elementType(elementType), _offset(offs) {}

Data::Data(const std::string &name, const Shape &shape, const Indexes &ix, DType elType, int offs)
    : _name(name), _shape(shape), _ix(ix), _elementType(elType), _offset(offs) {
    assert(_ix.size() <= _shape.size());
}

Data::Data(
    const std::string &name, const Shape &shape, const Indexes &ix, DType elType, int offs,
    IterVar index
)
    : Data(name, shape, ix, elType, offs) {
    _ix.push_back(index);
    assert(_ix.size() <= _shape.size());
}

Data::Data(
    const std::string &name, const Shape &shape, const Indexes &ix, DType elType, int offs,
    const Indexes &indexes
)
    : Data(name, shape, ix, elType, offs) {
    _ix.insert(_ix.end(), indexes.begin(), indexes.end());
    assert(_ix.size() <= _shape.size());
}

const Indexes &Data::indexes() const { return _ix; }
const Shape &Data::shape() const { return _shape; }
Shape &Data::getShape() { return _shape; }
int Data::dim(int i) const {
    if (i < 0) {
        i = _shape.size() + i;
    }
    assert(i >= 0 && i < _shape.size() && "Index out of bounds");
    return _shape[i].count;
}
std::vector<int> Data::dims() const {
    std::vector<int> dims;
    for (const auto &dim : _shape) {
        dims.push_back(dim.count);
    }
    return dims;
}
std::vector<int> Data::dims(int begin, int end) const {
    if (begin < 0) {
        begin = _shape.size() + begin;
    }
    if (end < 0) {
        end = _shape.size() + end;
    }
    if (!(begin >= 0 && end <= _shape.size() && begin <= end)) {
        llvm::errs() << "Error: begin=" << begin << ", end=" << end << " invalid for shape "
                     << _shape << " of rank " << _shape.size() << "\n";
        assert(false && "Invalid begin end specified");
    }
    std::vector<int> dims;
    for (int i = begin; i < end; ++i) {
        dims.push_back(_shape[i].count);
    }
    return dims;
}

DType Data::elementType() const { return _elementType; }
void Data::setElementType(DType elType) { _elementType = elType; }

void Data::setShape(const Shape &shape) {
    _shape = shape;
    assert(_ix.size() <= _shape.size());
}

Shape Data::subShape() const { return Shape(&_shape[_ix.size()], &_shape[_shape.size()]); }

const std::string &Data::name() const { return _name; }

int Data::offset() const { return _offset; }

void Data::setOffset(int offset) { _offset = offset; }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Data &shape) {
    os << shape.name() << shape.subShape();
    if (shape.offset()) {
        os << "@" << shape.offset();
    }
    os << ":" << shape.elementType();
    return os;
}

LData::LData(Value value) : LData(cast<MemRefType>(value.getType())) {}

LData::LData(const MemRefType &type) : DataT(Shape{}, DType::none, 0) {
    const DType elementType = getDType(type.getElementType());
    assert(elementType != DType::none && "Invalid element type");

    Shape dataShape;
    auto shape = type.getShape();
    auto strides = getEncodedStridesElements(type);
    for (int i = 0; i < shape.size(); ++i) {
        dataShape.push_back({shape[i], strides[i]});
    }

    setElementType(elementType);
    setShape(dataShape);
}

LData &LData::insertDim(int pos, const ShapeItem &item) {
    auto &shape = getShape();
    shape.insert(shape.begin() + pos, item);
    return *this;
}

IData &IData::insertDim(int pos, const ShapeItem &item) {
    auto &shape = getShape();
    shape.insert(shape.begin() + pos, item);
    return *this;
}

LData &LData::eraseDim(int pos) {
    auto &shape = getShape();
    shape.erase(shape.begin() + pos);
    return *this;
}

LData &LData::moveDim(int fromDimIndex, int toDimIndex) {
    auto &shape = getShape();
    assert(fromDimIndex >= 0 && fromDimIndex < shape.size());
    assert(toDimIndex >= 0 && toDimIndex < shape.size());
    ShapeItem dim = shape[fromDimIndex];
    assert(dim.stride.intVal.has_value() && "Only dims with explicit int stride can be moved");
    if (fromDimIndex > 0 && !shape[fromDimIndex - 1].stride.hasVal()) {
        // Update the stride of the previous dim
        shape[fromDimIndex - 1].stride = dim.stride.intVal.value() * dim.count;
    }
    shape.erase(shape.begin() + fromDimIndex);
    shape.insert(shape.begin() + toDimIndex, dim);
    return *this;
}

int LData::denseDims() const { return torq::denseDims(*this); }

LData &LData::fuse(int count) {
    torq::fuse(*this, count);
    return *this;
}

LData &LData::fuse(const std::vector<int> &dims) {
    int fuseCount = 0;
    for (auto dim : llvm::reverse(dims)) {
        assert(dim == shape().size() - ++fuseCount && "Only trailing dimensions can be fused");
    }
    return fuse(dims.size());
}

LData &LData::vectorize(int vectorSize, int vectorStride) {
    torq::vectorize(*this, vectorSize, vectorStride);
    return *this;
}

LData &LData::subviewDim(int dimIndex, int offset, int count) {
    torq::subviewDim(*this, dimIndex, offset, count);
    return *this;
}

LData &LData::reshapeDim(int dimIndex, const std::vector<int> &newDims, bool allowNonMultiple) {
    torq::reshapeDim(*this, dimIndex, newDims, allowNonMultiple);
    return *this;
}

LData &LData::partitionByIndexParity1D() {
    torq::partitionByIndexParity1D(*this);
    return *this;
}

LData &LData::partitionByIndexParity2D() {
    torq::partitionByIndexParity2D(*this);
    return *this;
}

//
// SlicePrivate class
//

struct SlicePrivate {
    // Represents a for loop in the kernel
    struct Loop {
        int count;
        torq_hw::MemDimTag tag;
    };

    // Internal memory info
    struct Ram {
        // DType of each element
        DType elementType{DType::none};
        // Loop nesting level at the point the RAM is loaded
        int loadNesting = -1;
    };

    // Constructor
    SlicePrivate(const std::string &name);

    // Add dimensions to a mem-based NDL according to the data shape and iteration variables
    // return the NDL offset
    int addMemNdlDims(
        NdlType type, torq_hw::MemNdlDimsData &dims, const Data &data, int appendBlockSize = -1
    );

    // Add dimensions to a reg-based NDL according to the data shape and iteration variables
    void addDims(
        NdlType type, torq_hw::RegNdlDimsData &ndlDims, const Data &data, int loadNesting,
        bool bigM = false, bool fuseW = false
    );

    // Compute the total iteration count of all loops inside and including the iv loop
    int innerIterCount(const std::vector<Loop> &forStack, int iv);

    // Compute the total iteration count of all loops outside the iv loop
    int outerIterCount(const std::vector<Loop> &forStack, int iv);

    // ALU instructions
    void aluSetMode(torq_hw::ALUOp0Mode op0, torq_hw::ALUOp1Mode op1);
    void aluSetNumberFormat(DType dtype);
    PData aluAccumulate(const IData &idata, torq_hw::ALUOp1Mode acc);
    PData aluProductAccumulate(
        const IData &idata, const WData &wdata, ALUOp1Mode acc, bool outer, bool repeatWeight
    );

    // ACT instructions
    void actSetNumberFormat(DType dtype);
    QData actClamp(
        const PData &pdata, int actShift, int actZeroPoint, int actClipMin, int actClipMax,
        torq_hw::ACTMode actMode
    );
    QData actRescaleClamp(
        const PData &pdata, const BData &bdata, int actShift, int actZeroPoint, int actClipMin,
        int actClipMax, torq_hw::ACTMode actMode
    );

    void memNdl(NdlType ndlType, const LData &data, int appendBlockSize = -1);
    void dedr(const LData &data);
    void dewr(const LData &data);
    void debr(const LData &data);
    void deqw(const LData &output, int appendBlockSize = -1);

    void ref(const LData &data);

    void cedr(const IData &idata, const uint32_t weightSize);
    void cedw(const IData &idata);

    void cewr(const WData &wdata, bool outer = false, bool repeatWeight = false);
    void ceww(const WData &wdata);

    void acbr(const BData &bdata);
    void acbw(const BData &bdata);
    void acpr(const PData &pdata);
    void acpw(DType dataType, const uint32_t weightSize);

    void cepr(const PData &pdata);

    // Internal RAMs info
    Ram _iram;
    Ram _wram;
    Ram _bram;
    Ram _pram;

    // Current for-loop stack
    std::vector<Loop> _forStack;

    // Copy of the for-loop stack when the ALU operation was called
    std::vector<Loop> _stackCopy;

    // Slice configuration
    SliceCfg _cfg{};

    // NDLs conviguration
    torq_hw::Ndls _ndls{};

    // Height and width of an input channel (used for convolutions)
    int _inputChannelHeight{};
    int _inputChannelWidth{};

    // Name of the kernel
    std::string _name;
};

SlicePrivate::SlicePrivate(const string &name) : _name(name) {
    LLVM_DEBUG(llvm::dbgs() << "- Kernel " << _name << " ----------------------------\n");
}

// Decompose number into two factors, both <= 65535 possible
// If not possible returns a pair of -1s
static std::pair<int, int> decomposeIntoTwoFactors(int number) {
    const int kMaxFactor = 65535;
    for (int i = 1; i <= kMaxFactor && i <= number; ++i) {
        if (number % i == 0) {
            int j = number / i;
            if (j <= kMaxFactor) {
                return {i, j};
            }
        }
    }
    return {-1, -1};
}

static torq_hw::MemDimTag tagToMemDimTag(ShapeItem::Tag tag, torq_hw::MemDimTag defaultTag) {
    switch (tag) {
    case ShapeItem::Tag::None:
        return defaultTag;
    case ShapeItem::Tag::Main:
        return torq_hw::MemDimTag::A;
    case ShapeItem::Tag::KernelRows:
        return torq_hw::MemDimTag::J;
    case ShapeItem::Tag::KernelCols:
        return torq_hw::MemDimTag::I;
    }
    assert(false && "Unknown ShapeItem::Tag");
}

int SlicePrivate::addMemNdlDims(
    NdlType type, torq_hw::MemNdlDimsData &ndlDims, const Data &data, int appendBlockSize
) {
    bool useSDims = appendBlockSize >= 0;
    Shape dataDims = data.shape();
    const Indexes &ix = data.indexes();
    auto elementSize = sizeofType(data.elementType());
    int busWidth = getBusWidth(type, data.elementType());
    const int sgGroupsMax = getBusScatterGather(type);
    auto block = useSDims ? SGBlockInfo{appendBlockSize}
                          : sgElementCount(data.subShape(), busWidth / elementSize, sgGroupsMax);
    assert(block.size && "Block empty or not dense");
    assert(ix.size() <= dataDims.size());
    int offset = data.offset();

    if (type != NdlType::DEQW) {
        assert(!block.isEvenOddSplit && "Even-odd split only supported for DEQW");
        assert(!useSDims && "SDIMs only supported for DEQW");
    }

    // Let's start with LDIMs.
    // We currently handle with LDIMs only the element size and the block.size.
    // Any other dimension will be handled with HDIMs.
    ndlDims.push_back({DimType::L, MemDimTag::B, elementSize, 1});
    ndlDims.push_back({DimType::L, MemDimTag::D, block.size, elementSize});
    if (block.sgGroups > 1) {
        if (block.size > 1 && !block.isEvenOddSplit) {
            if (block.stride.exprVal.has_value()) {
                ndlDims.push_back(
                    {DimType::L, MemDimTag::G, block.sgGroups, block.stride.exprVal.value()}
                );
            }
            else {
                assert(block.stride.intVal.has_value());
                ndlDims.push_back(
                    {DimType::L, MemDimTag::G, block.sgGroups,
                     block.stride.intVal.value() * elementSize}
                );
            }
        }
        else {
            // Use DEQW even-odd split mode
            // Update D tag dim and count with special meaning
            assert(block.sgGroups == 2);
            assert(elementSize <= 2);
            ndlDims.back().count = block.sgGroups;
            if (block.stride.exprVal.has_value()) {
                ndlDims.back().setExprStride(block.stride.exprVal.value());
            }
            else {
                ndlDims.back().setIntStride(block.stride.intVal.value() * elementSize);
            }
            ndlDims.push_back({DimType::L, MemDimTag::G, block.size, elementSize});
        }
    }
    // Ensure we have at least 2 DIMs for _mem_ndl_desc_gen()
    ndlDims.push_back({DimType::H, MemDimTag::G, 1});

    // Generate an extra HDIM to load the entire size of the indexed data block
    if (block.outerGroups > 1) {
        if (block.outerStride.exprVal.has_value()) {
            ndlDims.push_back(
                {DimType::H, MemDimTag::V, block.outerGroups, block.outerStride.exprVal.value()}
            );
        }
        else {
            assert(block.outerStride.intVal.has_value());
            ndlDims.push_back(
                {DimType::H, MemDimTag::V, block.outerGroups,
                 block.outerStride.intVal.value() * elementSize}
            );
        }
    }

    // Generate additional HDIMs from loops
    // The NDL will have as many extra H-dimensions as the nesting of the loop we are in
    for (int i = _forStack.size() - 1; i >= 0; i--) {
        // Check if this loop appears in the list of indexed dimensions,
        // in this case compute the stride from dataDims.
        // If not, this loop doesn't affect this data load, so this is a repeat with stride 0
        Stride stride(0);
        ShapeItem::Tag tag{};
        int itemsCount = 0;
        const IterVar *iv{};
        int loopIterCount = _forStack[i].count;
        for (int dataDimensionIx = 0; dataDimensionIx < ix.size(); dataDimensionIx++) {
            const IterVar &iterVar{ix[dataDimensionIx]};
            if (iterVar.iterId() == i) {
                if (iterVar.divisor()) {
                    // Special case to split the same loop between mem and reg based NDLs
                    loopIterCount = div_ceil(loopIterCount, iterVar.divisor());
                }
                if (loopIterCount > dataDims[dataDimensionIx].count) {
                    llvm::errs() << "Out of bounds access to dimension " << dataDimensionIx
                                 << " of " << data.name() << data.shape()
                                 << " with iteration count " << loopIterCount
                                 << " while generating " << type << "\n";
                    assert(false && "Out of bounds access");
                }
                stride = computeStride(dataDims, dataDimensionIx);
                itemsCount = dataDims[dataDimensionIx].count;
                tag = dataDims[dataDimensionIx].tag;
                iv = &iterVar;
                break;
            }
        }

        assert(!iv || !iv->modulo() && "Iteration with modulo not allowed here");
        if (stride.exprVal.has_value()) {
            assert((!iv || !iv->isReverse()) && "Reverse iteration not allowed with expr");
            ndlDims.push_back({DimType::H, _forStack[i].tag, loopIterCount, stride.exprVal.value()}
            );
        }
        else {
            int strideVal = stride.intVal.value();
            if (iv && iv->isReverse()) {
                offset += (itemsCount - 1) * strideVal;
                strideVal = -strideVal;
            }
            // If useSDims and this iter variable is not used to explicitly index the output data,
            // this means that it will be covered by SDims, so use an A tag (with stride 0)
            // as required by the hardware.
            // Note: probably we don't really need to use tha A tag if strideVal == 0
            auto memDimTag = useSDims && !iv ? MemDimTag::A : tagToMemDimTag(tag, _forStack[i].tag);
            ndlDims.push_back({DimType::H, memDimTag, loopIterCount, strideVal * elementSize});
        }
    }

    // Handle constant indexes
    for (int dataDimensionIx = 0; dataDimensionIx < ix.size(); dataDimensionIx++) {
        const IterVar &iterVar{ix[dataDimensionIx]};
        if (iterVar.iterId() < 0 && iterVar.constIndex() != 0) {
            Stride stride = computeStride(dataDims, dataDimensionIx);
            assert(!stride.exprVal.has_value() && "Constant index not allowed for expr stride");
            int strideVal = stride.intVal.has_value() ? stride.intVal.value() : 0;
            offset += iterVar.constIndex() * strideVal;
        }
    }

    if (useSDims) {
        assert(block.outerGroups == 1 && "SDIMs only supported for single block transfers");

        // Check that we don't have more than 3 A dims (HW limitation)
        // A-dims come from loops whose iteration variable is not used in the indexing
        int aDimCnt = 0;
        for (auto &dim : ndlDims) {
            aDimCnt += dim.tag == MemDimTag::A;
        }
        if (aDimCnt > 3) {
            llvm::errs() << "Up to 3 unused loop var in append() indexing, got:" << aDimCnt << "\n";
            assert(false && "Too many A-dims");
        }

        // Add special SDIMs to cover the remaining size
        Shape sdimsShape = data.subShape();
        ndlDims.push_back({DimType::S, MemDimTag::B, elementSize, 1});
        int denseCnt = denseElementCount(sdimsShape);
        assert(denseCnt != 0 && "SDIM can't handle empty tensors");
        if (denseCnt > 0) {
            if (elementSize == 4) {
                // HW doesn't support elementSize == 4 but if the output is dense we can adjust
                // the data type and the count accordingly
                denseCnt *= 2;
                elementSize = 2;
                ndlDims.back().count = elementSize;
            }

            // Sequential write to a dense subtensor
            if (denseCnt > 65535) {
                auto cntPair = decomposeIntoTwoFactors(denseCnt);
                assert(cntPair.first != -1 && "SDIM dense count is a prime number too large");
                ndlDims.push_back({DimType::S, MemDimTag::X, cntPair.first, elementSize});
                ndlDims.push_back(
                    {DimType::S, MemDimTag::Y, cntPair.second, elementSize * cntPair.first}
                );
            }
            else {
                ndlDims.push_back({DimType::S, MemDimTag::X, denseCnt, elementSize});
            }
        }
        else if (sdimsShape.size() == 2 || sdimsShape.size() == 4) {
            // Should be even-odd split or 4-quadrants even-odd segmentation
            assert(elementSize <= 2 && "Non-dense SDIMs not supported for element size > 2");
            assert(sdimsShape[1].count == 2 && "Not even-odd split");
            assert((sdimsShape.size() < 2 || sdimsShape[3].count == 2) && "Not even-odd split");
            int i = 0;
            for (const auto &s : llvm::reverse(sdimsShape)) {
                assert(s.stride.intVal.has_value());
                ndlDims.push_back(
                    {DimType::S, i++ < 2 ? MemDimTag::X : MemDimTag::Y, s.count,
                     s.stride.intVal.value() * elementSize}
                );
            }
        }
        else {
            llvm::errs() << "Error: unsupported subshape for SDIM: " << sdimsShape << "\n";
            assert(false && "unsupported subshape for SDIM");
        }
    }
    return offset * elementSize;
}

void SlicePrivate::addDims(
    NdlType type, torq_hw::RegNdlDimsData &ndlDims, const Data &data, int loadNesting, bool bigM,
    bool fuseW
) {
    Shape dataDims = data.shape();
    const Indexes &ix = data.indexes();
    auto elementSize = sizeofType(data.elementType());
    if (loadNesting < 0) {
        llvm::errs() << "Processing NDL " << type << " for data that was never loaded\n";
        assert(false && "Data for NDL was never loaded");
    }

    // Check that the indexes are not referring to some loop before the load point
    // This is allowed only for modulo indexes which generate their own local loop
    for (const auto &iterVar : ix) {
        if (iterVar.iterId() < 0) {
            assert(iterVar.constIndex() == 0 && "Indexing with non-0 constant not allowed here");
        }
        else if (iterVar.iterId() < loadNesting && !iterVar.modulo()) {
            llvm::errs() << "Error, " << data << " uses index from loop " << iterVar
                         << " which is before load at level " << loadNesting << "\n";
            assert(false && "Invalid index in data");
        }
    }

    // We only add HDIMs here, LDIMs must have been added by the caller.
    // Generate HDIMs from loops
    // The NDL should have as many H-dims as the nesting of the loop we are in from loadNesting
    // Mem-based NDLs are not so flexible so we should group nearby repetitions in N or M dims,
    // and use W or S for loops that have a non-zero stride.
    torq_hw::RegDimTag repeatTags[] = {RegDimTag::M, RegDimTag::N};
    torq_hw::RegDimTag strideTags[] = {RegDimTag::S, RegDimTag::W};
    int repeatTagIx = 0;
    int strideTagIx = 0;
    bool prevDimIsRepeat = false;

    for (int i = _forStack.size() - 1; i >= loadNesting; i--) {
        // Check if this loop appears in the list of indexed dimensions,
        // in this case compute the stride from dataDims.
        // If not, this loop doesn't affect this data load, so this is a repeat with stride 0
        Stride stride(0);
        int loopIterCount = _forStack[i].count;
        for (int dataDimensionIx = 0; dataDimensionIx < ix.size(); dataDimensionIx++) {
            const IterVar &iterVar{ix[dataDimensionIx]};
            if (iterVar.iterId() == i) {
                assert(!iterVar.isReverse() && "Reverse iteration not allowed here");
                assert(
                    /*iterVar.modulo() ||*/ !iterVar.divisor() &&
                    "Iteration with divisor not allowed here"
                );
                if (iterVar.modulo()) {
                    // Special case to split the same loop between mem and reg based NDLs
                    loopIterCount = iterVar.modulo();
                }
                if (loopIterCount > dataDims[dataDimensionIx].count) {
                    llvm::errs() << "Out of bounds access to dimension " << dataDimensionIx
                                 << " of " << data.name() << data.shape()
                                 << " with iteration count " << loopIterCount
                                 << " while generating " << type << "\n";
                    assert(false && "Out of bounds access");
                }
                stride = computeStride(dataDims, dataDimensionIx);
                break;
            }
        }

        auto strideVal = stride.intVal.value() * elementSize;
        if (strideVal == 0) {
            if (repeatTagIx >= sizeof(repeatTags) / sizeof(repeatTags[0])) {
                llvm::errs() << "Error, NDL " << type << " has too many repeats\n";
                assert(false && "Too many repeats");
            }
            auto repeatCount = loopIterCount;
            auto tag = repeatTags[repeatTagIx++];
            if (tag == RegDimTag::M && repeatCount > 256 && !bigM) {
                // M only support small repeats, use next tag
                tag = repeatTags[repeatTagIx++];
            }
            else if (tag == RegDimTag::N && bigM && prevDimIsRepeat) {
                // Cumulate all repeats in a single M
                tag = RegDimTag::M;
                repeatCount *= ndlDims.back().count;
                ndlDims.pop_back();
                repeatTagIx--;
            }
            ndlDims.push_back({DimType::H, tag, repeatCount});
            prevDimIsRepeat = true;
        }
        else {
            if (fuseW && strideTagIx) {
                auto &wDim = ndlDims.back();
                llvm::errs() << "wDim.count: " << wDim.count << " wDim.stride : " << wDim.stride
                             << " strideVal: " << strideVal << "\n";
                assert(wDim.count * wDim.stride == strideVal && "Cannot fuse");
                wDim.count *= loopIterCount;
            }
            else {
                constexpr int strideTagsCount = sizeof(strideTags) / sizeof(strideTags[0]);
                assert(strideTagIx < strideTagsCount && "Too many strides");
                auto tag = strideTags[strideTagIx++];
                ndlDims.push_back({DimType::H, tag, loopIterCount, strideVal});
            }
            prevDimIsRepeat = false;
        }
    }
}

int SlicePrivate::innerIterCount(const std::vector<Loop> &forStack, int iv) {
    int count = 1;
    for (int i = forStack.size() - 1; i >= iv; i--) {
        count *= forStack[i].count;
    }
    return count;
}

int SlicePrivate::outerIterCount(const std::vector<Loop> &forStack, int iv) {
    int count = 1;
    for (int i = iv - 1; i >= 0; i--) {
        count *= forStack[i].count;
    }
    return count;
}

void SlicePrivate::aluSetMode(torq_hw::ALUOp0Mode op0, torq_hw::ALUOp1Mode op1) {
    for (auto &mode0 : _cfg.alu_op0_mode) {
        mode0 = op0;
    }
    for (auto &mode1 : _cfg.alu_op1_mode) {
        mode1 = op1;
    }
}

void SlicePrivate::aluSetNumberFormat(DType dtype) {
    if (isFloat(dtype)) {
        _cfg.alu_format = torq_hw::NumberFormat::BF;
    }
    else if (isInt(dtype)) {
        _cfg.alu_format = torq_hw::NumberFormat::I;
    }
    else {
        assert(false && "Unsupported data type for ALU number format");
    }
}

void SlicePrivate::actSetNumberFormat(DType dtype) {
    if (isFloat(dtype)) {
        _cfg.act_format = torq_hw::NumberFormat::BF;
    }
    else if (isInt(dtype)) {
        _cfg.act_format = torq_hw::NumberFormat::I;
    }
    else {
        assert(false && "Unsupported data type for ACT number format");
    }
}

// return value rounded up to the next valid value, 0 if too big
static int roundUp(int value, const int *validValues) {
    while (int validValue = *validValues++) {
        if (value <= validValue) {
            return validValue;
        }
    }
    return 0;
}

void SlicePrivate::cedr(const IData &idata, const uint32_t weightSize) {
    Shape shape = idata.subShape();
    const int elementSize = sizeofType(idata.elementType());
    SGBlockInfo bi = sgElementCount(shape, backDimCount(shape));
    int biCount = elementSize;

    // Generate CEDR to load the data from IRAM to ALU
    // For integer operations with 16b weights each data bytes in the block is duplicated so that
    // it will be processed with 2 different weight bytes
    RegNdlDimsData cedrDims;
    if (weightSize > 1 && isInt(idata.elementType())) {
        assert(weightSize == 2 && "Only int8 or int16 weights supported for now");
        cedrDims.push_back({DimType::L, RegDimTag::I, weightSize});
        biCount *= weightSize;
    }
    cedrDims.push_back({DimType::L, RegDimTag::B, elementSize, 1});
    static const int sValidDataCount[] = {8, 16, 32, 64, 0};
    int dCount = biCount * bi.size;
    int dSize = roundUp(dCount, sValidDataCount) / biCount;
    if (!dSize) {
        llvm::errs() << "Invalid CEDR block size: " << dSize << "\n";
        assert(false && "Invalid CEDR block size");
    }

    cedrDims.push_back({DimType::L, RegDimTag::D, dSize, elementSize});
    if (bi.sgGroups > 1 || bi.outerGroups > 1) {
        // Not sure if stride is actually used by HW here
        int stride = bi.stride.intVal.has_value() ? bi.stride.intVal.value() : 1;
        static const int sValidGroupCount[] = {1, 4, 8, 16, 32, 0};
        int groupCount = roundUp(bi.sgGroups * bi.outerGroups, sValidGroupCount);
        if (!groupCount) {
            llvm::errs() << "Invalid CEDR group size: " << groupCount << "\n";
            assert(false && "Invalid CEDR group size");
        }
        cedrDims.push_back({DimType::L, RegDimTag::G, groupCount, stride * elementSize});
    }

    // Now add hdims for each loop deeper the iram load
    addDims(NdlType::CEDR, cedrDims, idata, _iram.loadNesting);

    // Actual T count will be adjusted later
    cedrDims.push_back({DimType::H, RegDimTag::T, 1});

    _ndls.add(NdlType::CEDR, cedrDims);
}

void SlicePrivate::cedw(const IData &idata) {
    Shape shape = idata.subShape();
    // assert(shape.size() <= 1);
    const int elementSize = sizeofType(idata.elementType());
    SGBlockInfo bi = sgElementCount(shape, backDimCount(shape));

    // Generate CEDW to load the data from DEDR to IRAM
    // TODO: check that the data fits IRAM and that dataShape has natural strides
    RegNdlDimsData regDims;
    regDims.push_back({DimType::L, RegDimTag::B, elementSize, 1});
    // Note we could as well always transfer the entire IRAM segment without penalty here
    int cnt =
        bi.sgGroups > 1 || bi.outerGroups > 1 ? HwInfo::iram_seg_width / elementSize : bi.size;
    regDims.push_back({DimType::L, RegDimTag::D, cnt, elementSize});
    if (bi.outerGroups > 1) {
        // Fill the IRAM with the data from DEDR
        regDims.push_back({DimType::H, RegDimTag::S, bi.outerGroups, HwInfo::iram_seg_width});
    }

    // Repeat for as many times as DEDR
    auto dedr = _ndls.getMemNdl(NdlType::DEDR);
    assert(dedr && "DEDR NDL not defined");
    auto ramWrCount = iterationCount(dedr);
    regDims.push_back({DimType::H, RegDimTag::T, ramWrCount / iterationCount(regDims)});
    _ndls.add(NdlType::CEDW, regDims);

    ndlToStr(NdlType::CEDW, _ndls.getRegNdl(NdlType::CEDW));
}

void SlicePrivate::cewr(const WData &wdata, bool outer, bool repeatWeight) {
    Shape shape = wdata.subShape();
    const int weightSize = sizeofType(wdata.elementType());
    const int weightBlockSize = backDimCount(shape);

    RegNdlDimsData cewrDims;

    // Generate CEWR to load the weights from WRAM to ALU
    cewrDims.push_back({DimType::L, RegDimTag::B, weightSize, 1});

    // repeatWeight is true means repeat same weightSize byte weight for
    // 256/weightSize times in alu computation,
    // repeatWeight is false means use weightBlockSize different weight for
    // 256/(weightBlockSize*weightSize) times in alu computation
    // if not repeatWeight avoid using a count of 1 since that would put the ALU in repeat mode
    cewrDims.push_back({DimType::L, RegDimTag::D, repeatWeight ? 1 : max(weightBlockSize, 2), 0});

    // Mainly used for int16/int32 elementwise multiplication operations.
    // For example, for int16, this means repeating the elementSize (2 bytes) times the weight
    // in order to compute with 16-bit data, resulting in a 48-bit output from combining two 24-bit
    // intermediate results.
    if ((_iram.elementType == DType::int16 || _iram.elementType == DType::int32) &&
        _cfg.alu_op0_mode[0] == torq_hw::ALUOp0Mode::MUL && !repeatWeight) {
        cewrDims.push_back({DimType::L, RegDimTag::J, sizeofType(_iram.elementType), 0});
    }

    auto repeat = 1;
    if (outer) {
        // Distribute the weights over all the MACs
        // It should be possible to do this by default but for some reason the HWAPI doesn't support
        // this if we also have an higher lever S with a stride that doesn't match:
        // (ss==0 || ss==g*b)
        // Not clear if this is an HW limitation or API limitation.
        repeat = weightBlockSize;
        cewrDims.push_back({DimType::L, RegDimTag::G, repeat, weightSize});
    }

    // Now add hdims for each loop deeper the wram load
    addDims(NdlType::CEWR, cewrDims, wdata, _wram.loadNesting, true, true);

    // Actual T count will be adjusted later
    cewrDims.push_back({DimType::H, RegDimTag::T, 1});
    _ndls.add(NdlType::CEWR, cewrDims);
}

void SlicePrivate::ceww(const WData &wdata) {
    Shape shape = wdata.subShape();
    // assert(shape.size() <= 1 && "WData shape must be up to 1 for now");
    const int weightSize = sizeofType(wdata.elementType());
    const int weightBlockSize = denseElementCount(shape);
    assert(weightBlockSize > 0 && "Block empty or not dense");

    // Generate CEWW to load the data from DEWR to WRAM
    // TODO: check that the data fits WRAM and that dataShape has natural strides
    RegNdlDimsData regDims;
    regDims.push_back({DimType::L, RegDimTag::B, weightSize, 1});               // One element
    regDims.push_back({DimType::L, RegDimTag::D, weightBlockSize, weightSize}); // Block
    regDims.push_back({DimType::L, RegDimTag::G});                              // No group

    // Repeat for as many times as DEWR
    auto dewr = _ndls.getMemNdl(NdlType::DEWR);
    assert(dewr && "DEWR NDL not defined");
    auto ramWrCount = iterationCount(dewr);
    regDims.push_back({DimType::H, RegDimTag::T, ramWrCount / iterationCount(regDims)});

    _ndls.add(NdlType::CEWW, regDims);
    ndlToStr(NdlType::CEWW, _ndls.getRegNdl(NdlType::CEWW));
}

void SlicePrivate::acbr(const BData &bdata) {
    Shape shape = bdata.subShape();
    const int elementSize = sizeofType(bdata.elementType());
    const int blockSize = backDimCount(shape);

    // Generate ACBR to load the data from BRAM to ACT
    RegNdlDimsData acbrDims;
    acbrDims.push_back({DimType::L, RegDimTag::B, blockSize * elementSize, 1});

    // Now add hdims for each loop deeper the bram load
    addDims(NdlType::ACBR, acbrDims, bdata, _bram.loadNesting, true);

    // Number of read from bram must match the number of writes (overall H iterations in DEBR)
    // TODO: handle the case with multiple DEBR NDLs
    auto debr = _ndls.getMemNdl(NdlType::DEBR);
    assert(debr && "DEBR NDL not defined");
    auto ramWrCount = iterationCount(debr);
    acbrDims.push_back({DimType::H, RegDimTag::T, ramWrCount});

    _ndls.add(NdlType::ACBR, acbrDims);
    ndlToStr(NdlType::ACBR, _ndls.getRegNdl(NdlType::ACBR));
}

void SlicePrivate::acbw(const BData &bdata) {
    Shape shape = bdata.subShape();
    const int elementSize = sizeofType(bdata.elementType());
    const int blockSize = backDimCount(shape);

    // Generate ACBW to load the data from DEBR to BRAM
    // TODO: check that the data fits BRAM and that dataShape has natural strides
    RegNdlDimsData regDims;
    // /!\ todo: this sould start from where the previous H loop left!
    regDims.push_back({DimType::L, RegDimTag::B, blockSize * elementSize, 1});
    regDims.push_back({DimType::L, RegDimTag::D, 1, 0}); // No block
    regDims.push_back({DimType::L, RegDimTag::G});       // No group

    // Repeat for as many times as DEBR
    auto debr = _ndls.getMemNdl(NdlType::DEBR);
    assert(debr && "DEBR NDL not defined");
    auto ramWrCount = iterationCount(debr);
    regDims.push_back({DimType::H, RegDimTag::T, ramWrCount});

    _ndls.add(NdlType::ACBW, regDims);
    ndlToStr(NdlType::ACBW, _ndls.getRegNdl(NdlType::ACBW));
}

void SlicePrivate::acpr(const PData &pdata) {
    Shape shape = pdata.subShape();
    const int elementSize = sizeofType(pdata.elementType());
    const int blockSize = backDimCount(shape);

    assert(elementSize <= HwInfo::pdat_width && "Invalid data element size");
    assert(blockSize <= HwInfo::act_width && "Block size must fit ACT width");

    // Generate ACPR to load the partials from PRAM to ACT
    RegNdlDimsData acprDims;
    acprDims.push_back({DimType::L, RegDimTag::B, elementSize, 1});
    static const int sValidDataSizes[] = {2, 4, 8, 16, 0};
    int dSize = roundUp(blockSize, sValidDataSizes);
    if (!dSize) {
        llvm::errs() << "Invalid ACPR block size: " << dSize << "\n";
        assert(false && "Invalid ACPR block size");
    }
    acprDims.push_back({DimType::L, RegDimTag::D, dSize, HwInfo::pdat_width});
    acprDims.push_back({DimType::L, RegDimTag::G});

    // Now add hdims for each loop deeper the last endfor (end of the ALU accumulate)
    addDims(NdlType::ACPR, acprDims, pdata, _pram.loadNesting, false, true);
    // And repeat for the number of executions of ALU
    acprDims.push_back({DimType::H, RegDimTag::T, outerIterCount(_forStack, _pram.loadNesting)});

    _ndls.add(NdlType::ACPR, acprDims);
    ndlToStr(NdlType::ACPR, _ndls.getRegNdl(NdlType::ACPR));
}

void SlicePrivate::cepr(const PData &pdata) {
    Shape shape = pdata.subShape();
    const int elementSize = sizeofType(pdata.elementType());
    const int blockSize = backDimCount(shape);

    assert(elementSize <= HwInfo::pdat_width && "Invalid data element size");
    assert(blockSize <= HwInfo::mac_count && "Block size must fit ALU width");

    RegNdlDimsData ceprDims;
    ceprDims.push_back({DimType::L, RegDimTag::B, elementSize, 1});
    ceprDims.push_back({DimType::L, RegDimTag::D, elementCount(pdata.shape()), HwInfo::pdat_width});

    ceprDims.push_back({DimType::H, RegDimTag::N, innerIterCount(_stackCopy, _pram.loadNesting)});
    ceprDims.push_back({DimType::H, RegDimTag::T, outerIterCount(_stackCopy, _pram.loadNesting)});

    // Update iteration count for CEDR and CEWR to match CEPR
    // These counters are not really used, but we set them to consistent values to avoid confusion
    int ceprCount = iterationCount(ceprDims);
    if (auto cedr = _ndls.getRegNdl(NdlType::CEDR)) {
        cedr->dims.back().count = div_ceil(ceprCount, iterationCount(cedr->dims));
        ndlToStr(NdlType::CEDR, _ndls.getRegNdl(NdlType::CEDR));
    }
    if (auto cewr = _ndls.getRegNdl(NdlType::CEWR)) {
        cewr->dims.back().count = div_ceil(ceprCount, iterationCount(cewr->dims));
        ndlToStr(NdlType::CEWR, _ndls.getRegNdl(NdlType::CEWR));
    }

    _ndls.add(NdlType::CEPR, ceprDims);
    ndlToStr(NdlType::CEPR, _ndls.getRegNdl(NdlType::CEPR));
}

void SlicePrivate::memNdl(NdlType ndlType, const LData &data, int appendBlockSize) {
    MemNdlDimsData ndlDims;
    int offset = addMemNdlDims(ndlType, ndlDims, data, appendBlockSize);
    _ndls.add(ndlType, ndlDims, offset);
    ndlToStr(ndlType, _ndls.getMemNdl(ndlType));
}

void SlicePrivate::dedr(const LData &data) { memNdl(NdlType::DEDR, data); }

void SlicePrivate::dewr(const LData &data) { memNdl(NdlType::DEWR, data); }

void SlicePrivate::debr(const LData &data) { memNdl(NdlType::DEBR, data); }

void SlicePrivate::deqw(const LData &output, int appendBlockSize) {
    memNdl(NdlType::DEQW, output, appendBlockSize);
}

void SlicePrivate::ref(const LData &data) {
    MemNdlDimsData refNdlDims;
    if (_inputChannelHeight || _inputChannelWidth) {
        refNdlDims.push_back({DimType::H, MemDimTag::X, _inputChannelWidth, 0});
        refNdlDims.push_back({DimType::H, MemDimTag::Y, _inputChannelHeight, 0});
    }

    _ndls.add(NdlType::REF, refNdlDims);
    ndlToStr(NdlType::REF, _ndls.getMemNdl(NdlType::REF));
}

void SlicePrivate::acpw(DType dataType, const uint32_t weightSize) {
    const uint32_t dataSize = sizeofType(_iram.elementType);
    if (isInt(dataType) && ((weightSize == 2) || (dataSize > 1 && weightSize == 1))) {
        auto actIterationCount = outerIterCount(_forStack, _forStack.size());
        RegNdlDimsData acpwDims;
        acpwDims.push_back({DimType::L, RegDimTag::B, HwInfo::pdat_width, 1});
        acpwDims.push_back({DimType::L, RegDimTag::D, weightSize * dataSize, HwInfo::pdat_width});
        acpwDims.push_back(
            {DimType::L, RegDimTag::G, HwInfo::act_width / (weightSize * dataSize),
             HwInfo::pdat_width * weightSize * dataSize}
        );
        acpwDims.push_back({DimType::H, RegDimTag::T, actIterationCount});

        _ndls.add(NdlType::ACPW, acpwDims);
        ndlToStr(NdlType::ACPW, _ndls.getRegNdl(NdlType::ACPW));
    }
}

PData SlicePrivate::aluAccumulate(const IData &idata, torq_hw::ALUOp1Mode acc) {
    // Configure the alu in bypass mode with the specified accumulate operation
    aluSetMode(ALUOp0Mode::DBYP, acc);
    const DType dataType = idata.elementType();
    aluSetNumberFormat(dataType);

    // Weights are not used in bypass mode (no need to generate cewr)
    _wram.elementType = DType::none;
    _cfg.alu_d_unsigned = 0;
    if (isUnsigned(dataType) || isFloat(dataType)) {
        _cfg.alu_d_unsigned = 0b1111;
    }
    else if (dataType == DType::int16) {
        _cfg.alu_d_unsigned = 0b101;
    }
    else if (dataType == DType::int32) {
        _cfg.alu_d_unsigned = 0b0111;
    }
    _cfg.alu_w_unsigned = 0;
    _cfg.alu_format =
        isFloat(idata.elementType()) ? torq_hw::NumberFormat::BF : torq_hw::NumberFormat::I;

    // Check the input data
    auto iShape = idata.subShape();
    // TODO: check that iShape[rank - 1] is dense
    const int blockSize = elementCount(iShape);
    assert(blockSize <= HwInfo::max_input && "Block size too big");
    _iram.elementType = idata.elementType();

    cedr(idata, 0);

    // Save a copy of current stack, will be needed later to add N and T dimensions
    _stackCopy = _forStack;

    // Remember stack nesting level when pram is loaded
    // Will be overwritten if we are inside a for loop
    _pram.loadNesting = _forStack.size();

    // Automatically infer the shape of the pram data
    int actWidth = isFloat(dataType) ? HwInfo::act_width / 2 : HwInfo::act_width;
    int actBlockCount = div_ceil(blockSize, actWidth);
    int actBlockSize = actBlockCount > 1 ? actWidth : blockSize;
    DType pDataType = isInt(dataType) ? DType::int32 : dataType;
    Shape pramShape = {actBlockCount, {actBlockSize, (HwInfo::pdat_width / sizeofType(pDataType))}};
    PData pdata(pramShape, pDataType);
    debugData(pdata);
    return pdata;
}

PData SlicePrivate::aluProductAccumulate(
    const IData &idata, const WData &wdata, ALUOp1Mode acc, bool outer, bool repeatWeight
) {
    // Configure the required multiply and accumulate operations
    aluSetMode(ALUOp0Mode::MUL, acc);
    aluSetNumberFormat(idata.elementType());

    // Get weight shape
    auto wShape = wdata.subShape();
    LLVM_DEBUG(llvm::dbgs() << "wShape: " << wShape.size() << "\n";);

    const DType weightType = wdata.elementType();
    const int weightSize = sizeofType(weightType);
    const int weightBlockSize = elementCount(wShape);
    _wram.elementType = weightType;

    // Each data bytes is duplicated and processed with each weight byte
    // There will be a total of "weightSize" partials for each data
    auto iShape = idata.subShape();
    const DType dataType = idata.elementType();

    LLVM_DEBUG(llvm::dbgs() << "aluProductAccumulate iShape: " << iShape.size() << "\n";);

    _iram.elementType = dataType;

    _cfg.alu_d_unsigned = dataType == DType::uint8 ? 0b1111 : 0;
    _cfg.alu_w_unsigned = weightType == DType::int16 ? 0b101 : 0;
    if (dataType == DType::int16 && weightType == DType::int16) {
        _cfg.alu_d_unsigned = 3;
        _cfg.alu_w_unsigned = 5;
    }

    cedr(idata, weightSize);
    cewr(wdata, outer, repeatWeight);

    // Save a copy of current stack, will be needed later to add N and T dimensions
    _stackCopy = _forStack;

    // Remember stack nesting level when pram is loaded
    // Will be overwritten if we are inside a for loop
    _pram.loadNesting = _forStack.size();

    // In some cases multiple partials are required to store the ALU result
    int partialElementWidth = isFloat(dataType) ? 2 : weightSize * sizeofType(dataType);
    assert(iShape.size() > 0 && "Input shape cannot be empty");
    int blockSize = iShape[iShape.size() - 1].count;
    blockSize *= partialElementWidth;
    if (blockSize > HwInfo::max_input) {
        llvm::errs() << "Actual block size: " << blockSize << " > " << HwInfo::max_input << "\n";
        assert(false && "Block size too big");
    }

    // Automatically infer the shape of the pram data
    int actWidth = HwInfo::act_width;
    int actBlockCount = div_ceil(blockSize, actWidth);
    int actBlockSize = actBlockCount > 1 ? actWidth : blockSize;
    bool resUnsigned = isUnsigned(dataType) && isUnsigned(weightType);
    DType resType = isInt(dataType) ? (resUnsigned ? DType::uint32 : DType::int32) : DType::fp32;
    Shape pramShape = {actBlockCount, {actBlockSize, (HwInfo::pdat_width / sizeofType(resType))}};
    if (outer) {
        // In this case pram.shape becomes 4D
        pramShape.insert(pramShape.begin(), weightBlockSize);
    }

    PData pdata(pramShape, resType);
    debugData(pdata);
    return pdata;
}

QData SlicePrivate::actClamp(
    const PData &pdata, int actShift, int actZeroPoint, int actClipMin, int actClipMax,
    torq_hw::ACTMode actMode
) {
    const DType dataType = _iram.elementType;
    const int dataSize = sizeofType(dataType);
    const DType weightType = _wram.elementType;
    const int weightSize = sizeofType(weightType);
    const DType partialType = pdata.elementType();
    DType resultType = dataType;
    int resultCount = elementCount(pdata.subShape());

    // Check that data, weight and partial types are compatible
    assert(
        isInt(dataType) && isInt(partialType) && !isFloat(weightType) ||
        isFloat(dataType) && isFloat(partialType) && !isInt(weightType)
    );

    // Configure the ACT unit
    _cfg.act_rsh = actShift;
    _cfg.act_lsh = {0, 0, 0, 0};
    _cfg.act_clip_min = actClipMin;
    _cfg.act_clip_max = actClipMax;
    _cfg.act_zero_point = actZeroPoint;
    _cfg.act_sum_bits = 0;
    _cfg.act_mode = actMode;
    actSetNumberFormat(pdata.elementType());

    if (isInt(partialType)) {
        if (weightSize == 2 && dataSize == 1) {
            // We have 2 partials for each data
            // Each partial is processed with different left shifts
            _cfg.act_lsh = {0, 8, 0, 8};
            resultType = DType::int16;
            resultCount /= 2;
        }
        else if (weightSize == 1 && dataSize == 2) {
            // We have 2 partials for each data
            // Each partial is processed with different left shifts
            _cfg.alu_d_unsigned = 5;
            _cfg.alu_w_unsigned = 0;
            _cfg.act_lsh = {0, 8, 0, 8};
            resultType = DType::int16;
            resultCount /= 2;
        }
        else if (dataSize == 2 && weightSize == 2) {
            // We have 4 partials for each data
            // Each partial is processed with different left shifts
            _cfg.act_lsh = {0, 8, 8, 16};
            _cfg.act_sum_bits = 48;
            resultType = DType::int32;
            resultCount /= 4;
        }
        else if (dataSize == 4 && weightSize == 1) {
            _cfg.alu_d_unsigned = 7;
            _cfg.alu_w_unsigned = 0;
            _cfg.act_lsh = {0, 8, 16, 24};
            _cfg.act_sum_bits = 48;
            resultType = DType::int32;
            resultCount /= 4;
        }
        else if (dataSize == 1 && weightSize == 1) {
            _cfg.act_lsh = {0, 0, 0, 0};
            resultType = DType::int8;
        }
        else if (weightSize == 0) {
            // No weight applied, keep original data type
        }
        else {
            llvm::errs() << "Unsupported data and weight size combination: dataSize=" << dataSize
                         << ", weightSize=" << weightSize << "\n";
            assert(false && "Invalid data and weight size");
        }
    }
    else if (isFloat(partialType)) {
        _cfg.act_sum_bits = sizeofType(partialType) * 8;
        if (isFloat(weightType)) {
            // ALU floating point results use 2 partials each
            assert(resultCount % 2 == 0);
            resultCount /= 2;
        }
    }
    else {
        assert(false && "Unsupported partial type");
    }

    acpr(pdata);
    cepr(pdata);
    acpw(dataType, weightSize);

    QData qdata({resultCount}, resultType);
    debugData(qdata);
    return qdata;
}

QData SlicePrivate::actRescaleClamp(
    const PData &pdata, const BData &bdata, int actShift, int actZeroPoint, int actClipMin,
    int actClipMax, torq_hw::ACTMode actMode
) {
    acbr(bdata);
    QData qdata = actClamp(pdata, actShift, actZeroPoint, actClipMin, actClipMax, actMode);
    return qdata;
}

//
// Slice class
//

Slice::Slice(const std::string &name)
    : d{new SlicePrivate(name)}, // Internal state common to the slice and its subcomponents
      iram{d.get()}, wram{d.get()}, bram{d.get()}, alu{d.get()}, act{d.get()} {}

Slice::~Slice() {}

const std::string &Slice::name() const { return d->_name; }

IterVar Slice::forall(int count) {
    int nesting = d->_forStack.size();
    d->_forStack.push_back({count, torq_hw::MemDimTag::O});
    return IterVar{nesting, *d};
}

void Slice::endfor() {
    assert(!d->_forStack.empty());
    d->_forStack.pop_back();
    d->_pram.loadNesting = d->_forStack.size();
}

Iterator Slice::iterate(int count) {
    // Just create an iterator object that will take care of calling forall/endfor
    return Iterator(*this, count);
}

Iterator Slice::iterate(const std::vector<int> &counts) { return Iterator(*this, counts); }

static void checkTypeCompatibility(const Data &destData, const Data &sourceData) {
    const DType sType = sourceData.elementType();
    const DType dType = destData.elementType();
    bool elemTypeCompatible = (isFloat(sType) && isFloat(dType)) || (isInt(sType) && isInt(dType));
    if (!elemTypeCompatible) {
        llvm::errs() << "Mismatching assignment. ";
        llvm::errs() << "Destination: " << destData << ", ";
        llvm::errs() << "Source: " << sourceData << "\n";
        assert(false && "Data type mismatch");
    }
}

static void checkCompatibility(const Data &destData, const Data &sourceData) {
    checkTypeCompatibility(destData, sourceData);

    const Shape destShape = destData.subShape();
    const Shape sourceShape = sourceData.subShape();
    const int destRank = destShape.size();
    const int sourceRank = sourceShape.size();
    if (destRank > 2) {
        llvm::errs() << "Invalid assignment. ";
        llvm::errs() << "Destination: " << destData << " has rank " << destRank << "\n";
        assert(false && "Destination rank error");
    }
    bool shapeMatch = destRank == sourceRank;
    if (shapeMatch) {
        for (size_t i = 0; i < destRank; ++i) {
            if (destShape[i].count != sourceShape[i].count) {
                shapeMatch = false;
                break;
            }
        }
    }
    else {
        // Accept assignment of a vector with a single element to a scalar
        if (destRank == 0 && sourceRank == 1 && sourceShape[0].count == 1) {
            shapeMatch = true;
        }
        // Also accept a vector assigned to a 2-dim matrix with same total number of elements
        // This is needed to support output scattering
        else if (destRank == 2 && sourceRank == 1 &&
                 destShape[0].count * destShape[1].count == sourceShape[0].count) {
            shapeMatch = true;
        }
    }
    if (!shapeMatch) {
        llvm::errs() << "Mismatching assignment. ";
        llvm::errs() << "Destination: " << destData << ", ";
        llvm::errs() << "Source: " << sourceData << "\n";
        assert(false && "Data shape mismatch");
    }
    if (destRank > 1) {
        // Check that the last dimension is dense (except if even-odd split)
        auto innerStride = destShape.back().stride;
        SGBlockInfo dummyBlockInfo;

        if (!isEvenOddSplit(destShape, dummyBlockInfo) &&
            (innerStride.exprVal.has_value() ||
             (innerStride.intVal.has_value() && innerStride.intVal.value() != 1))) {
            llvm::errs() << "Destination inner dimension is not dense: " << destData << "\n";
            assert(false && "Inner dimension must be dense");
        }
    }
}

void Slice::append(const LData &output, const QData &data) {
    assert(data.indexes().size() == 0 && "QData can't be indexed during append");
    assert(data.shape().size() == 1 && "Unexpected QData rank during append");
    checkTypeCompatibility(output, data);
    d->deqw(output, backDimCount(data.subShape()));
}

void Slice::store(const LData &output, const QData &data) {
    assert(data.indexes().size() == 0 && "QData can't be indexed during store");
    assert(data.shape().size() == 1 && "Unexpected QData rank during store");
    checkCompatibility(output, data);
    d->deqw(output);
}

void Slice::store(const LData &output, int value) {
    // Check that no values loaded in RAM by mistake
    if (d->_iram.loadNesting >= 0 || d->_wram.loadNesting >= 0 || d->_bram.loadNesting >= 0) {
        llvm::errs() << "Error: values loaded in memory not used";
        assert(false);
    }
    const Shape subShape = output.subShape();
    assert(subShape.size() == 0 && "output must be a scalar");

    // Create here mandatory REF NDL
    d->ref(output);

    // Be sure ALU and ACT are completely disabled
    d->_cfg.alu_disable = 0xFFFF;
    d->_cfg.act_disable = 0xF;
    d->aluSetMode(torq_hw::ALUOp0Mode::DBYP, torq_hw::ALUOp1Mode::BXOR);

    // Configure the value to be stored as pad value
    setPadding({}, value);
    d->deqw(output);
}
int Slice::scatter() const { return getBusScatterGather(NdlType::DEQW); }

void Slice::setKernel(const LRTBDim &lrtb) { d->_cfg.kernel = lrtb; }

void Slice::setPadding(const LRTBDim &lrtb, int padValue) {
    d->_cfg.pad.left = lrtb.left > 0;
    d->_cfg.pad.right = lrtb.right > 0;
    d->_cfg.pad.top = lrtb.top;
    d->_cfg.pad.bottom = lrtb.bottom;
    d->_cfg.pad_value = padValue;
}

void Slice::setInputChannelShape(int height, int width) {
    auto refNdl = d->_ndls.getMemNdl(NdlType::REF);
    assert(!refNdl && "Must be called before loading the input");
    d->_inputChannelHeight = height;
    d->_inputChannelWidth = width;
}

void Slice::segment(const std::vector<int> &nchw, int channelStride) {
    if (nchw.empty()) {
        return;
    }

    assert(nchw.size() == 4 && "Invalid segmentation nchw dimensions");
    int h = nchw[2];
    int w = nchw[3];
    assert(h % 2 == 0 && w % 2 == 0 && "Segmentation must be even");
    assert(channelStride > 0 && "Channel stride must be positive");
    assert(channelStride % 4 == 0 && "Channel stride must be multiple of 4");

    // Re-edit DEQW dimensions to segment the output in 4 quadrants
    // Look for the inside-channel dimensions (A TAG) and clear their stride
    // (Inside-channel offset will advance with special X,Y dimensions)
    auto &ndlDims = d->_ndls.getMemNdl(NdlType::DEQW)->dims;
    bool hasInsideChannelDims = false;
    for (auto i = 0; i < ndlDims.size(); i++) {
        if (ndlDims[i].tag == MemDimTag::A) {
            ndlDims[i].setIntStride(0);
            hasInsideChannelDims = true;
        }
    }
    if (!hasInsideChannelDims) {
        // No inside-channel dimensions: assume there are no channel dimensions and clear all
        // existing H strides
        for (auto &dim : ndlDims) {
            if (dim.type == DimType::H) {
                dim.setIntStride(0);
            }
        }
        // Add a new explicit channel dimension with the specified stride
        ndlDims.push_back({DimType::H, MemDimTag::G, nchw[1], channelStride});
    }

    // The previous dimension is the channel dimension, divide its stride by 4
    auto segmentStride = channelStride / 4;
    ndlDims.push_back({DimType::S, MemDimTag::X, 2, segmentStride});
    ndlDims.push_back({DimType::S, MemDimTag::X, w / 2, 1});
    ndlDims.push_back({DimType::S, MemDimTag::Y, 2, segmentStride * 2});
    ndlDims.push_back({DimType::S, MemDimTag::Y, h / 2, w / 2});
}

torq_hw::SliceCFGAttr Slice::getCfgAttr(MLIRContext *ctx) const {
    return d->_cfg.toSliceCFGAttr(ctx);
}

const torq_hw::Ndls &Slice::getNdls() const {
#if 0
    // This doesn't work yet. Check conv-stride1.mlir vs conv-stride1-notile-5ch.mlir with:
    // biasScale.getShape()[1].count = 4;

    // Check consistency
    auto acbr = d->_ndls.getRegNdl(NdlType::ACBR);
    auto acpr = d->_ndls.getRegNdl(NdlType::ACPR);
    int acbrRepeat = (acbr && acbr->dims.size() > 2) ? acbr->dims[acbr->dims.size() - 2].count : 1;
    if (acbr && acpr && iterationCount(acbr->dims) != iterationCount(acpr->dims) * acbrRepeat) {
        llvm::errs() << "The biasScale read don't match the number of executions of ACT\n";
        llvm::errs() << "Mismatching iteration counts:\n";
        llvm::errs() << "  BiasScale: " << iterationCount(acbr->dims) << "/" << acbrRepeat << "\n";
        llvm::errs() << "  ACT: " << iterationCount(acpr->dims) << "\n";
        llvm::errs() << "  DEQW: " << iterationCount(d->_ndls.getMemNdl(NdlType::DEQW)) << "\n";
        assert(false && "Iteration count mismatch");
    }
#endif
    // Just return the NDLs
    return d->_ndls;
}

//
// Memory Units
//

void SliceRam::checkLoadSize(const Data &data) {
    debugData(data);

    int dataSize = elementCount(data.shape()) * sizeofType(data.elementType());
    if (dataSize > size()) {
        llvm::errs() << name() << " load: " << data << " (" << dataSize
                     << " bytes) exceeds capacity of " << size() << " bytes\n";
        assert(false && "Ram width exceeded");
    }
}

const char *IRam::name() const { return "IRam"; }

int IRam::size() const { return HwInfo::iram_seg_width * HwInfo::iram_seg; }

IData IRam::load(const LData &data) {
    // Create here mandatory REF NDL
    d->ref(data);
    d->dedr(data);

    d->_iram.loadNesting = d->_forStack.size();
    d->_iram.elementType = data.elementType();

    Shape iramShape = data.subShape();
    auto idata = IData(iramShape, data.elementType());
    d->cedw(idata);

    // Adjust IRAM shape strides to natural strides, in IRAM data is contiguous
    for (auto &iramShapeItem : iramShape) {
        iramShapeItem.stride = {};
    }

    if (iramShape.size() > 0) {
        SGBlockInfo bi =
            sgElementCount(iramShape, backDimCount(iramShape), getBusScatterGather(NdlType::DEDR));
        if (bi.stride.intVal.has_value()) {
            // Set the actual stride in the IRAM
            // The blocks read with gather are not contiguous in IRAM but have a fixed stride.
            iramShape[iramShape.size() - 1].stride = HwInfo::iram_seg_width / bi.sgGroups;
        }
    }

    idata.setShape(iramShape);

    checkLoadSize(idata);
    return idata;
}

int IRam::gather() const { return getBusScatterGather(NdlType::DEDR); }

const char *WRam::name() const { return "WRam"; }

int WRam::size() const {
    // WRam can store up to 36 elements
    return HwInfo::wram_seg_width + 4;
}

WData WRam::load(const LData &data) {
    d->dewr(data);

    d->_wram.loadNesting = d->_forStack.size();
    d->_wram.elementType = data.elementType();

    auto wdata = WData(data.subShape(), data.elementType());
    d->ceww(wdata);

    checkLoadSize(wdata);
    return wdata;
}

const char *BRam::name() const { return "BRam"; }

int BRam::size() const { return width() * HwInfo::breg_width; }

int BRam::width() const {
    // BRam can store up to 4 {bias,scale} pairs
    return 4;
}

BData BRam::load(const LData &data) {
    d->debr(data);

    d->_bram.loadNesting = d->_forStack.size();
    d->_bram.elementType = data.elementType();

    // TODO add extra iterations to load the data size between specified indexes and dataShape[1]
    // /!\ not clear how this would affect iterationCount(ndlDims)

    auto bdata = BData(data.subShape(), data.elementType());
    d->acbw(bdata);

    checkLoadSize(bdata);
    return bdata;
}

//
// Arithmetic Logic Unit
//

PData Alu::accumulate(const IData &idata, torq_hw::ALUOp1Mode acc) {
    return d->aluAccumulate(idata, acc);
}

PData Alu::load(const IData &idata) { return d->aluAccumulate(idata, ALUOp1Mode::BOR); }

PData Alu::scalarProductAccumulate(const IData &idata, const WData &wdata, ALUOp1Mode acc) {
    // Check wdata is a scalar or tensor with shape {1}
    auto iShape = idata.subShape();
    auto wShape = wdata.subShape();
    if (iShape.size() > 1) {
        llvm::errs() << "Invalid input shape for scalarProductAccumulate: " << iShape << "\n";
        assert(false && "Invalid input shape");
    }
    if (wShape.size() > 1 || (wShape.size() == 1 && wShape[0].count != 1)) {
        llvm::errs() << "Invalid weight shape for scalarProductAccumulate: " << wShape << "\n";
        assert(false && "Invalid weight shape");
    }

    return d->aluProductAccumulate(idata, wdata, acc, false, true);
}

PData Alu::multiScalarProductAccumulate(const IData &idata, const WData &wdata, ALUOp1Mode acc) {
    // Check wdata is a scalar or tensor with shape {1}
    auto iShape = idata.subShape();
    auto wShape = wdata.subShape();
    if (iShape.size() != 2) {
        llvm::errs() << "Invalid input shape for scalarProductAccumulate: " << iShape << "\n";
        assert(false && "Invalid input shape");
    }
    if (wShape.size() != 1) {
        llvm::errs() << "Invalid weight shape for scalarProductAccumulate: " << wShape << "\n";
        assert(false && "Invalid weight shape");
    }

    // Enable "outer" mode since we have multiple weights to apply them to different input vectors
    return d->aluProductAccumulate(idata, wdata, acc, true, true);
}

PData Alu::outerProductAccumulate(const IData &idata, const WData &wdata, ALUOp1Mode acc) {
    return d->aluProductAccumulate(idata, wdata, acc, true, true);
}

PData Alu::elementwiseProductAccumulate(const IData &idata, const WData &wdata, ALUOp1Mode acc) {
    return d->aluProductAccumulate(idata, wdata, acc, false, false);
}

int Alu::iWidth(DType iType, DType wType, int weightWidth) const {
    if (isFloat(iType)) {
        assert(!isInt(wType) && "Weight type can't be int for float input");
        if (weightWidth <= 0) {
            return HwInfo::max_input / sizeofType(iType);
        }
        else if (weightWidth <= 4) {
            assert(iType == DType::bf16 && wType == DType::bf16);
            return 32;
        }
        else if (weightWidth <= 8) {
            assert(iType == DType::bf16 && wType == DType::bf16);
            return 16;
        }
    }
    else if (isInt(iType)) {
        assert(!isFloat(wType) && "Weight type can't be float for int input");
        int inputElementSize = sizeofType(iType);
        int weightElementSize = wType == DType::none ? 1 : sizeofType(wType);
        assert(inputElementSize * weightElementSize <= sizeof(int32_t));
        int n = HwInfo::max_input / inputElementSize / weightElementSize;
        if (weightWidth <= 4) {
            return n;
        }
        else if (weightWidth <= 8) {
            return n / 2;
        }
        else if (weightWidth <= 16) {
            return n / 4;
        }
    }
    return 0;
}

int Alu::wWidth(DType wType) const {
    int weightSize = wType == DType::none ? 1 : sizeofType(wType);
    return HwInfo::wram_seg_width / weightSize;
}

int Alu::kerWidth() const { return 3; }

//
// Activation Unit
//

QData Act::load(const PData &pdata) {
    bool isFlt = isFloat(pdata.elementType());
    constexpr int32_t minusInf = 0xff800000;
    constexpr int32_t plusInf = 0x7f800000;
    const int32_t clipMin = isFlt ? minusInf : std::numeric_limits<int32_t>::min();
    const int32_t clipMax = isFlt ? plusInf : std::numeric_limits<int32_t>::max();
    return d->actClamp(pdata, 0, 0, clipMin, clipMax, torq_hw::ACTMode::ACT);
}

QData Act::clamp(const PData &pdata, int clipMin, int clipMax, torq_hw::ACTMode actMode) {
    return d->actClamp(pdata, 0, 0, clipMin, clipMax, actMode);
}

QData Act::rescaleClamp(
    const PData &pdata, const BData &bdata, int shift, int zeroPoint, int clipMin, int clipMax,
    torq_hw::ACTMode actMode
) {
    return d->actRescaleClamp(pdata, bdata, shift, zeroPoint, clipMin, clipMax, actMode);
}

int Act::width(DType iType, DType wType) const {
    if (isFloat(iType)) {
        return HwInfo::act_width / 2;
    }
    else if (isInt(iType)) {
        assert(!isFloat(wType) && "Weight type can't be float for int input");
        return HwInfo::act_width /
               (wType == DType::none ? 1 : sizeofType(wType) * sizeofType(iType));
    }
    return 0;
}

//
// Iterator class
//

Iterator::Iterator(Slice &kernel, int count) : _kernel{kernel}, _iterVars{kernel.forall(count)} {}

Iterator::Iterator(Slice &kernel, const std::vector<int> &counts) : _kernel{kernel} {
    for (auto count : counts) {
        _iterVars.push_back(kernel.forall(count));
    }
}

Iterator::Iterator(const Iterator &other)
    : _kernel{other._kernel}, _iterVars{other._iterVars}, _isCopy{true} {}

Iterator::Iterator(Iterator &&other)
    : _kernel{other._kernel}, _iterVars{std::move(other._iterVars)}, _isCopy{other._isCopy} {}

Iterator::~Iterator() {
    if (!_isCopy) {
        for (int i = 0; i < _iterVars.size(); ++i) {
            _kernel.endfor();
        }
    }
}

Iterator Iterator::reverse() & {
    // We have been called from a named object, we don't want to change the original so
    // we make a clone and return that one
    Iterator clone(*this);
    for (auto &iv : clone._iterVars) {
        iv.reverse();
    }
    return clone;
}

Iterator Iterator::reverse() && {
    // We have been called from a temporary, this could be the one when the loop is created
    // so if we clone it the for loop will be destroyed immediately when the temporary goes away.
    // In this case we move it so that the temporary destructor will do nothing.
    for (auto &iv : _iterVars) {
        iv.reverse();
    }
    return std::move(*this);
}

Iterator Iterator::operator/(int divisor) {
    assert(_iterVars.size() == 1 && "div can be applied only to single iterators");
    Iterator clone(*this);
    clone._iterVars.back().setDivisor(divisor);
    return clone;
}

Iterator Iterator::operator%(int modulo) {
    assert(_iterVars.size() == 1 && "mod can be applied only to single iterators");
    Iterator clone(*this);
    clone._iterVars.back().setModulo(modulo);
    return clone;
}

//
// Dimension manipulation utilities
//

static void partitionByIndexParity1D(LData &data) {
    Shape shape = data.shape();
    int rank = shape.size();
    assert(rank >= 1 && "Data must have at least 1 dimensions");
    assert(sizeofType(data.elementType()) <= 2 && "Element size must be 1 or 2 bytes");
    Shape subShape(&shape[rank - 1], &shape[rank]);

    // Check the last dim is dense
    assert(denseElementCount(subShape) > 0 && "Not dense inner dimension");
    int w = subShape[0].count;
    // To be verified if the below assert is really needed
    assert(w % 2 == 0 && "Last dimension must be even for partitioning");

    shape.pop_back();
    shape.push_back({(w / 2), 1});
    shape.push_back({2, (w / 2)});

    // Update data shape
    data.setShape(shape);
}

static void partitionByIndexParity2D(LData &data) {
    Shape shape = data.shape();
    int rank = shape.size();
    assert(rank >= 2 && "Data must have at least 2 dimensions");
    assert(sizeofType(data.elementType()) <= 2 && "Element size must be 1 or 2 bytes");
    Shape subShape(&shape[rank - 2], &shape[rank]);

    // Check the last two dims are dense
    // To be verified if subShape[0] really needs to be dense or we can somehow handle a stride
    assert(denseElementCount(subShape) > 0 && "Not dense inner dimension");
    // To be verified if the below assert is really needed
    int h = subShape[0].count;
    int w = subShape[1].count;
    // Note: this works also if h or w are odd
    int subFrameSize = (h * w) / 4;

    shape.pop_back();
    shape.pop_back();
    // Note: use normal div here instead of div_ceil, as in the original ndl-based implementation
    shape.push_back({(h / 2), (w / 2)});
    shape.push_back({2, (subFrameSize * 2)});
    shape.push_back({(w / 2), 1});
    shape.push_back({2, subFrameSize});

    // Update data shape
    data.setShape(shape);
}

// Return the number of dense dimensions at the innermost side of the data
static int denseDims(const LData &data) {
    int denseCnt = 0;
    // TODO: we can do better
    while (denseCnt < data.shape().size() && denseElementCount(data.shape(), denseCnt + 1) > 0) {
        denseCnt++;
    }
    return denseCnt;
}

static void fuse(LData &data, int count) {
    int fused = 1;
    for (; (count < 0 || fused < count) && data.shape().size() > 1 &&
           denseElementCount(data.shape(), 2) > 0;
         fused++) {
        // Merge last two dimensions
        int rank = data.shape().size();
        data.getShape()[rank - 2].count *= data.getShape().back().count;
        data.getShape()[rank - 2].stride = data.getShape().back().stride;
        data.getShape().pop_back();
    }
    assert(fused >= count && "Could not fuse the requested number of dimensions");
}

static void vectorize(LData &data, int vectorSize, int vectorStride) {
    if (data.denseDims() == 0) {
        // This is a scalar tensor or a tensor where even the innermost dimension is not dense
        // Add a placeholder dense dimension of size 1 to be able to vectorize
        data.getShape().push_back(ShapeItem{1});
    }
    if (vectorStride <= 0) {
        vectorStride = vectorSize;
    }

    // Remove dimension to vectorize
    auto shape = data.shape();
    int rank = shape.size();
    int itemCount = shape.back().count;
    shape.pop_back();

    // Append new vectorized dimensions.
    // Use the 'Main' tag for the count dimension to indicate it's the main vectorized dimension.
    // If itemCount <= vectorSize create one single vector of that size (this optimization is not
    // done if the vectorSize != vectorStride, this is needed for convolution with frame size
    // exactly equal to vectorSize because in that case we have to respect the vectorStride to
    // take into account padding).
    int vectCount = div_ceil(itemCount, vectorStride);
    shape.push_back(ShapeItem{vectCount, vectorStride, ShapeItem::Tag::Main});
    shape.push_back(ShapeItem{vectCount > 1 || vectorSize != vectorStride ? vectorSize : itemCount}
    );

    // Set the stride of the previous dim (if any) if it doesn't have a stride yet
    if (rank > 1 && !shape[rank - 1].stride.hasVal()) {
        shape[rank - 1].stride = Stride{itemCount};
    }

    data.setShape(shape);
}

static void subviewDim(LData &data, int dimIndex, int offset, int count) {
    auto shape = data.getShape();
    int rank = shape.size();
    assert(dimIndex >= 0 && dimIndex < rank && "Dimension index out of range");
    if (!(count > 0 && offset >= 0 && offset + count <= shape[dimIndex].count)) {
        llvm::errs() << "Subview out of range: dimIndex=" << dimIndex << ", offset=" << offset
                     << ", count=" << count << ", dimSize=" << shape[dimIndex].count << "\n";
        assert(false && "Subview out of range");
    }
    auto stride = computeStride(shape, dimIndex);
    assert(!stride.exprVal.has_value() && "Cannot subview dimension with expr stride");
    int intStride = stride.intVal.value();

    // Adjust dimension & offset
    int itemCountInDim = shape[dimIndex].count;
    data.setOffset(data.offset() + offset * intStride);
    shape[dimIndex].count = count;

    // Set the stride of the previous dim (if any) if it doesn't have a stride yet
    if (dimIndex > 1 && !shape[dimIndex - 1].stride.hasVal()) {
        shape[dimIndex - 1].stride = Stride{itemCountInDim};
    }

    data.setShape(shape);
}

static void
reshapeDim(LData &data, int dimIndex, const std::vector<int> &sizes, bool allowNonMultiple) {
    int rank = data.shape().size();
    assert(dimIndex < rank && "Invalid dimension to reshape");

    // Check the total number of elements in sizes[] corresponds to that in the dimension to reshape
    int itemCount = 1;
    for (auto size : sizes) {
        assert((size > 0 || (size == -1 && itemCount > 0)) && "Sizes must be positive or -1");
        itemCount *= size;
    }
    int implicitDimSize = 0;
    if (itemCount < 0) {
        // Deduce implicit dimension
        itemCount = -itemCount;
        if (data.dim(dimIndex) % itemCount != 0 && !allowNonMultiple) {
            llvm::errs() << "Reshape size mismatch: dimension " << dimIndex << " size "
                         << data.dim(dimIndex) << " not multiple of " << itemCount << "\n";
            assert(false && "Reshape size mismatch");
        }
        implicitDimSize = div_ceil(data.dim(dimIndex), itemCount);
    }
    else if ((allowNonMultiple && itemCount > data.dim(dimIndex)) ||
             (!allowNonMultiple && itemCount != data.dim(dimIndex))) {
        llvm::errs() << "Reshape size mismatch: dimension " << dimIndex << " has size "
                     << data.dim(dimIndex) << " but reshape sizes have total size " << itemCount
                     << "\n";
        assert(false && "Reshape size mismatch");
    }

    // Remove dimension to reshape
    auto shape = data.shape();
    ShapeItem reshapedItem = shape[dimIndex];
    shape.erase(shape.begin() + dimIndex, shape.begin() + dimIndex + 1);

    // Append new reshaped dimensions
    for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
        reshapedItem.count = *it > 0 ? *it : implicitDimSize;
        shape.insert(shape.begin() + dimIndex, reshapedItem);
        // Use natural stride for next dimensions
        reshapedItem.stride = reshapedItem.stride.intVal.has_value()
                                  ? Stride(reshapedItem.stride.intVal.value() * reshapedItem.count)
                                  : Stride{};
    }

    data.setShape(shape);
}

} // namespace mlir::syna::torq
