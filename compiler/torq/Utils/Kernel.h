// Copyright 2024 SYNAPTICS inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Transforms/DialectConversion.h"

#include "torq/Dialect/TorqHW/TorqHWAttrs.h"

namespace mlir::syna {

namespace torq {

class Slice;
class SlicePrivate;
class Iterator;

// A stride can be specified as either a constant value or an affine expression
struct Stride {
    Stride() {}
    Stride(const Stride &) = default;
    Stride &operator=(const Stride &) = default;
    Stride(int64_t stride) : intVal(int(stride)) {}
    Stride(AffineExpr stride) : exprVal(stride) {}
    bool hasVal() const { return intVal.has_value() || exprVal.has_value(); }

    std::optional<int> intVal{};
    std::optional<AffineExpr> exprVal{};
};

// ShapeItem represents a dimension of a tensor in terms of its size and optional stride
struct ShapeItem {
    ShapeItem(int64_t count) : count(int(count)) {}
    ShapeItem(int64_t count, int64_t intStride) : count(int(count)), stride(int(intStride)) {}
    ShapeItem(int64_t count, AffineExpr exprStride) : count(int(count)), stride(exprStride) {}
    ShapeItem(const ShapeItem &) = default;
    ShapeItem &operator=(const ShapeItem &) = default;

    int count;
    Stride stride;
};

// Dimensions and strides of a tensor
using Shape = std::vector<ShapeItem>;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Shape &shape);

// Total number of elements in the shape
int elementCount(const Shape &shape);

// Loop iteration variable
class IterVar {
  public:
    IterVar(int iterId) : _iterId(iterId) {}

    void reverse() { _reverse = !_reverse; }

    // Check if iteration variable is reversed
    bool isReverse() const { return _reverse; }

    operator int() const { return _iterId; }

  private:
    int _iterId;

    // If reverse go backward from last element to the first
    bool _reverse{};
};

// Indexes of a tensor
using Indexes = std::vector<IterVar>;

// Data types for tensor elements
enum class DType { none, uint8, uint16, uint32, int8, int16, int32, bf16, fp32 };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DType dtype);

// return size in bytes of the given type
int sizeofType(DType type);

// return true if the given type is a floating point type
bool isFloat(DType type);

// return true if the given type is an integer type
bool isInt(DType type);

// return true if the given type is an unsigned type
bool isUnsigned(DType type);

// return the DType corresponding to the given MLIR type
// none if no corresponding type exists
DType getDType(mlir::Type mlirType);

// Represent a data tensor in memory
// Data is defined by a shape, and it is possible to access a subset of it using indexes.
// For example the following represents a 4D tensor with 16 channels:
//  auto tensor = Data({1,16,3,3});
// We can access a channel of the tensor using: tensor[n][c] where "n" and "c" are IterVar
// Only IterVar can be used as indexes, constants and other integers are not allowed.
class Data {
  public:
    // Create a data tensor representation with the given shape and element type
    Data(Data &&) = default;
    Data &operator=(const Data &) = default;

    // Get data shape
    const Shape &shape() const;

    // Get i-th dimension of the shape, equivalent to shape()[i].count
    // if i is negative it is intended relative to the end of the shape as numpy
    int dim(int i) const;

    // Get all the dimensions of the shape
    std::vector<int> dims() const;

    // Get dimensions of the shape in the range [begin, end)
    // If end is negative it is intended relative to the end of the shape as numpy
    // eg if shape is {1,3,16,8} and begin=1, end=-1 will return {3,16}
    std::vector<int> dims(int begin, int end) const;

    // Set data shape
    void setShape(const Shape &shape);

    // Get data element type
    DType elementType() const;

    // Change data element type
    // Provide a different view on the data, no actual data conversion is performed.
    // Note: this is like C casting, use responsibly
    void setElementType(DType elType);

    // Get current indexes
    const Indexes &indexes() const;

    // Return the shape of the indexed sub-block of the data tensor
    // so if t has shape {1,3,16,8} and ix = {i, j} will return a shape of {16,8}
    Shape subShape() const;

    // Return data name
    const std::string &name() const;

    // Return data offset
    int offset() const;
    void setOffset(int offset);

  protected:
    Data(const std::string &, const Shape &, DType, int offset);
    Data(const std::string &, const Shape &, const Indexes &ix, DType, int offs);
    Data(const std::string &, const Shape &, const Indexes &ix, DType, int offs, IterVar);
    Data(const std::string &, const Shape &, const Indexes &ix, DType, int offs, const Indexes &);

  private:
    std::string _name;
    Shape _shape{};
    Indexes _ix{};
    DType _elementType;
    int _offset;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Data &shape);

// Wrapper to create different types of Data
template <class T> class DataT : public Data {
  public:
    using Data::Data;
    DataT(const Shape &shape, DType elementType) : Data(T::name(), shape, elementType, 0) {}
    DataT(const Shape &shape, DType elementType, int offset)
        : Data(T::name(), shape, elementType, offset) {}
    T operator[](IterVar index) const {
        return T(T::name(), shape(), indexes(), elementType(), offset(), index);
    }
    T operator[](const Indexes &ixs) const {
        return T(T::name(), shape(), indexes(), elementType(), offset(), ixs);
    }
};

// Data in LRAM
class LData : public DataT<LData> {
    using DataT::DataT;

  public:
    LData(const Shape &shape, DType elementType) : DataT(shape, elementType) {}
    LData(const Shape &shape, DType elementType, int offs) : DataT(shape, elementType, offs) {}
    LData(const MemRefType &type);
    static std::string name() { return "LData"; }
};

// Data in IRAM
class IData : public DataT<IData> {
    using DataT::DataT;
    IData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    static std::string name() { return "IData"; }
};

// Data in WRAM
class WData : public DataT<WData> {
    using DataT::DataT;
    WData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    static std::string name() { return "WData"; }
};

// Data in BRAM
class BData : public DataT<BData> {
    using DataT::DataT;
    BData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    static std::string name() { return "BData"; }
};

// Data in PRAM
// Nodiscard to ensure results from computations are not discarded since PData must normally
// be declared outside the loop where it can be computed
class [[nodiscard]] PData : public DataT<PData> {
    using DataT::DataT;
    PData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    PData() : DataT(Shape{}, DType::none) {}
    static std::string name() { return "PData"; }
};

// Output data
class QData : public DataT<QData> {
    using DataT::DataT;
    QData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    static std::string name() { return "QData"; }
};

// Represents a logical component of the Slice
class SliceComponent {
  public:
    SliceComponent(SlicePrivate *priv) : d{priv} {}

  protected:
    SlicePrivate *d;
};

// Slice RAM
class SliceRam : public SliceComponent {
    using SliceComponent::SliceComponent;

  public:
    virtual const char *name() const = 0;
    virtual int size() const = 0;
    virtual ~SliceRam() = default;

  protected:
    void checkLoadSize(const Data &data);
};

// Input RAM
class IRam : public SliceRam {
    using SliceRam::SliceRam;

  public:
    // Load the IRAM with (input) data
    IData load(const LData &data);

    // Get max gathering supported in the load() operation
    // Gathering indicates the max number of different addresses that can be generated in parallel.
    // Note: only blocks of specific sizes can be gathered (16, 32).
    int gather() const;

    const char *name() const override;
    int size() const override;
};

// Weight RAM
class WRam : public SliceRam {
    using SliceRam::SliceRam;

  public:
    // Load the WRAM with (weights) data
    WData load(const LData &data);

    const char *name() const override;
    int size() const override;
};

// Bias & Scale RAM
class BRam : public SliceRam {
    using SliceRam::SliceRam;

  public:
    // Load the BRAM with bias and scale
    BData load(const LData &data);

    const char *name() const override;
    int size() const override;

    // Max number of bias/scale pairs that can fit
    int width() const;
};

// Arithmetic Logic Unit
class Alu : SliceComponent {
    using SliceComponent::SliceComponent;

  public:
    // Load input data of shape {N} to PRam
    // N can be any value up to iWidth(iType)
    PData load(const IData &idata);

    // Accumulate an input of shape {N}
    // N can be any value up to iWidth(iType)
    // idata: input tensor data in iram
    // acc: accumulate operation (accumulate with ALUOp1Mode::BOR is equivalent to load()
    // return: pram data of shape {ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    // if N < act::width the result will be {1, N}:pType
    PData accumulate(const IData &idata, torq_hw::ALUOp1Mode acc);

    // Multiply an input of shape {N} with a scalar weight (or vector of shape {1}) and accumulate
    // N can be any value up to iWidth(iType, wType)
    // idata: input tensor data in iram
    // wdata: weight tensor data in wram
    // acc: accumulate operation
    // return: pram data of shape {ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    // if N < act::width the result will be {1, N}:pType
    PData scalarProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Outer product of an input of shape {N} with a weight vector of shape {M} and accumulate
    // N can be up to iWidth(iType, wType) (NOT any value, only some power of 2 for now TODO FIXME)
    // idata: input tensor data in iram
    // wdata: weight tensor data in wram
    // acc: accumulate operation
    // return: pram data of shape {M, ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    PData outerProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Multiply an input of shape {N} with a weight vector of shape {N} and accumulate
    // N can be any value up to wWidth(wType)
    // idata: input tensor data in iram
    // wdata: weight tensor data in wram
    // acc: accumulate operation
    // return: pram data of shape {ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    // if N < act::width the result will be {1, N}:pType
    PData elementwiseProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Max number of input items that can be processed in parallel for the given in and weight type
    int iWidth(DType iType, DType wType = DType::none) const;

    // Max number of weight items that can be processed in parallel for the given input width
    int wWidth(DType wType, int inputWidth = 0) const;
};

// Activation Unit
class Act : SliceComponent {
    using SliceComponent::SliceComponent;

  public:
    // Load partials of shape {N} to QData
    // N can be up to width(iType)
    QData load(const PData &pdata);

    // Perform clamp on partials of shape {N}
    // N can be up to width(iType)
    // clamped return: result data of shape {M}
    // M can be less than M if multiple partials are combined to compute each result value
    QData clamp(
        const PData &pdata, int clipMin, int clipMax,
        torq_hw::ACTMode actMode = torq_hw::ACTMode::ACT
    );

    // Perform rescaling and clamp on partials of shape {N}
    // N can be up to width(iType)
    // return: result data of shape {M}
    // M can be less than M if multiple partials are combined to compute each result value
    QData rescaleClamp(
        const PData &pdata, const BData &bdata, int shift, int zeroPoint, int clipMin, int clipMax,
        torq_hw::ACTMode actMode = torq_hw::ACTMode::ACT
    );

    // Max number of output items that can be computed in parallel for the given in and weight type
    int width(DType iType, DType wType = DType::none) const;
};

// Create a kernel for a slice
class Slice {
    std::unique_ptr<SlicePrivate> d;

  public:
    Slice();
    ~Slice();

    // Start a for loop iterating over all the values in the range [0, count - 1]
    // returns the loop iteration variable
    IterVar forall(int count, torq_hw::MemDimTag tag = torq_hw::MemDimTag::O);

    // Terminate a for loop
    void endfor();

    // Create an iterator (equivalent to forall/endfor)
    Iterator iterate(int count, torq_hw::MemDimTag tag = torq_hw::MemDimTag::O);
    Iterator iterate(const std::vector<int> &counts);

    // Store the computed result to LRAM
    // output is a vector of shape {N}, data must be a dense vector of the same shape {N}
    // If N in the output shape is 1, data can also be a scalar.
    // In alternative both output and data can be an array of shape {s, N} where s is a power of
    // two up to Slice::scatter() with dense inner dim.
    // If the first dimension is not dense then must be s*N == alu.width() or N == 1
    // The case N == 1 is only supported for item size <= 2.
    // Data types must be compatible, not necessarily the same, fp16 is compatible with float32,
    // integer types are compatible with each other.
    void store(const LData &output, const QData &data);

    // Store a value to LRAM
    // output must represent a scalar
    // Can only be used if iram, wram and bram haven't been loaded
    void store(const LData &output, int value);

    // Get max scattering supported by the slice in the store() operation
    // Scattering indicates the max number of different addresses that can be generated in parallel
    int scatter() const;

    // Configure the height and width of each input channel
    // Mandatory for convolutions
    void setInputChannelShape(int height, int width);

    // Configure the kernel size on each side, for example a 5x5 kernel will have {2, 2, 2, 2}
    // Mandatory for convolutions
    void setKernel(const std::vector<int> &lrtb);

    // Configure the input padding margin and value
    // For example a 1-pixel padding on left, right, top, bottom will have {1, 1, 1, 1}
    // Mandatory for convolutions
    void setPadding(const std::vector<int> &lrtb, int padValue);

    // Segment the output in 4 quadrants of shape {N, C, 4, H/2, W/2}
    // outShape: NCHW shape of the output tensor. if empty the output is not segmented.
    // channelStride: stride of the channel dimension
    // Note: loops on channel data if any must have been marked with MemDimTag::A
    void segment(const std::vector<int> &nchw, int channelStride);

    // Get the slice configuration
    torq_hw::SliceCFGAttr getCfgAttr(MLIRContext *ctx) const;

    // Get the NDLs
    const torq_hw::Ndls &getNdls() const;

    // Subunits, each has its own instruction set
    IRam iram;
    WRam wram;
    BRam bram;
    Alu alu;
    Act act;
};

// Helper class that calls forall in the constructor and endfor when going out of scope
class Iterator {
  public:
    Iterator(const Iterator &) = delete;
    Iterator(Iterator &&);
    Iterator &operator=(const Iterator &) = delete;
    Iterator(Slice &kernel, int count, torq_hw::MemDimTag tag);
    Iterator(Slice &kernel, const std::vector<int> &counts);
    ~Iterator();

    // Revert the iterator direction, from last to first
    Iterator reverse();

    operator bool() const { return true; }
    operator Indexes() const { return _iterVars; }

  private:
    Slice &_kernel;
    Indexes _iterVars;
};

// This macro is just a syntactic helper to write Torq loops with iterators in a natural way eg:
// For (auto block = kernel.iterate(blockCount)) {
//
// }
// Above "block" represents an iterator that calls endfor at the end of the block
#define For if

} // namespace torq

} // namespace mlir::syna
