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

// A stride can be specified as either a constant value or an affine expression.
// Constant strides and offsets always represent items (as in memrefs).
// Affine expression represent bytes.
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
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Stride &shape);

// ShapeItem represents a dimension of a tensor in terms of its size and optional stride
struct ShapeItem {
    enum class Tag {
        None,
        Main,       // Main (vectorized) data dimension
        KernelRows, // Dimension associated to kernel row index
        KernelCols  // Dimension associated to kernel column index
    };
    ShapeItem(int64_t count) : count(int(count)) {}
    ShapeItem(int64_t count, Stride stride, Tag tag = Tag::None)
        : count(int(count)), stride(stride), tag(tag) {}
    ShapeItem(const ShapeItem &) = default;
    ShapeItem &operator=(const ShapeItem &) = default;

    int count;
    Stride stride;
    // Tag is only used in input LData to support automatic data padding in convolutions
    Tag tag{Tag::None};
};

// Dimensions and strides of a tensor
class Shape : public std::vector<ShapeItem> {
  public:
    using std::vector<ShapeItem>::vector;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Shape &shape);

// Loop iteration variable
class IterVar {
  public:
    // Create from a constant index. We can make this explicit if needed.
    IterVar(int constIndex) : _iterId(-1), _constIndex{constIndex} {}

    // Create from a loop id. Internal usage only.
    IterVar(int iterId, const SlicePrivate &) : _iterId(iterId) {}
    int iterId() const { return _iterId; }

    void reverse() { _reverse = !_reverse; }
    void setDivisor(int divisor) { _divisor = divisor; }
    void setModulo(int modulo) { _modulo = modulo; }

    bool isReverse() const { return _reverse; }
    int divisor() const { return _divisor; }
    int modulo() const { return _modulo; }
    int constIndex() const { return _constIndex; }

  private:
    int _iterId;

    // If reverse go backward from last element to the first
    bool _reverse{};

    // Division mode. The actual count for the NDL dim is computed by dividing the
    // corresponding loop count by _divisor.
    int _divisor{0};

    // Modulo mode. An additional dim with count of _modulo is added to the corresponding NDL dim.
    int _modulo{0};

    // Support indexing a dimension with a constant index
    int _constIndex{0};
};

// Indexes of a tensor
using Indexes = std::vector<IterVar>;
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const IterVar &iv);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Indexes &indexes);

// Data types for tensor elements
enum class DType { none, uint8, uint16, uint32, int8, int16, int32, bf16, fp32 };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const DType dtype);

// Dimension indicators for vectorized data
struct Vectorized {
    enum {
        Vectors = -2, // The data vectors
        Elements = -1 // The data elements
    };
};

// Dimension indicators for NCHW data, quite common
struct NCHW {
    enum { N, C, H, W };
};

// Dimension structure for Height and Width
struct HWDim {
    HWDim() = default;
    HWDim(int h, int w) : h(h), w(w) {}
    int h{};
    int w{};
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const HWDim &shape);

// Dimension structure for Left, Right, Top, Bottom
// Can be used for 2D kernel size and padding specification
struct LRTBDim {
    enum { Left = 0, Right = 1, Top = 2, Bottom = 3 };
    LRTBDim() = default;
    LRTBDim(::llvm::ArrayRef<int64_t> lrtb);

    // Create symmetric border from kernel h,w size
    static LRTBDim symmetric(const HWDim &dim);

    int left{};
    int right{};
    int top{};
    int bottom{};
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const LRTBDim &shape);

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
    Data(const Data &) = default;
    Data &operator=(const Data &) = default;

    // Get data shape
    const Shape &shape() const;

    // Get i-th dimension of the shape, equivalent to shape()[i].count
    // if i is negative it is intended relative to the end of the shape as numpy
    int dim(int i) const;

    // Get all the dimensions of the shape
    std::vector<int> dims() const;

    // Get dimensions of the shape in the range [begin, end)
    // If begin or end is negative it is intended relative to the end of the shape as numpy
    // eg if shape is {1,3,16,8} and begin=1, end=-1 will return {3,16}
    std::vector<int> dims(int begin, int end) const;

    // Get R/W data shape
    Shape &getShape();

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

    // Return data offset (in items)
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
    LData(const Value value);
    static std::string name() { return "LData"; }

    // Dimension manipulation methods

    // Return the number of contiguous dense dimensions at the end of the data shape
    int denseDims() const;

    // Fuse count dimensions at the end of the data shape into a single dimension.
    // Asserts if rank < count or some of the dimensions are not dense.
    // If count is -1 all dense dimensions at the end are fused.
    LData &fuse(int count);

    // Same as fuse(n) but with explicitly specified dimensions
    // The dimensions must be the last ones in the shape and dense.
    LData &fuse(const std::vector<int> &dims);

    // Vectorize the last dimension into vectors of the specified size and stride.
    // The last dimension must be dense.
    // If the number of elements is not multiple of vectorStride the last vector
    // will actually go beyond the end of the data shape, in any case the stride
    // of the dimension above is kept to the actual element count.
    // If vectorStride is not specified it is assumed to be the same as vectorSize
    // Example:
    // data shape: {1, 16, 5, 5}, vectorSize: 4, dims: {2, 3}
    // resulting shape: {1, {16,stride:25}, 7, 4}
    LData &vectorize(int vectorSize, int vectorStride = 0);

    // Create a sub-view of the specified dimension.
    LData &subviewDim(int dimIndex, int offset, int count);

    // Reshape the specified dimension.
    // dimIndex: index of the dimension to reshape
    // newDims: new dimensions that will replace the specified one, if one of them is -1
    // its size is inferred from the size of the original dimension and the other new dimensions
    // asserts if the product of the new dimensions is not equal to the size of the original one
    // unless allowNonMultiple is true
    LData &reshapeDim(int dimIndex, const std::vector<int> &newDims, bool allowNonMultiple = false);

    // Insert a new dimension at the specified index
    LData &insertDim(int dimIndex, const ShapeItem &item);

    // Erase the dimension at the specified index
    LData &eraseDim(int dimIndex);

    // Move a dimension from fromDimIndex to toDimIndex
    // the 'toDimIndex' is the index after removing the dimension at 'fromDimIndex'
    LData &moveDim(int fromDimIndex, int toDimIndex);

    // Broadcast dimensions to match the shape of 'other' data
    // The current shape must have the same rank as 'other'
    // All dimensions with size 1 are expanded to match the corresponding dimension in 'other'
    // but with stride 0, so that no data duplication occours.
    // Asserts if the shapes are not compatible for broadcasting
    LData &broadcastAs(const LData &other);

    // Reorganize the last dimension into two sub-blocks containing elements
    // with even and odd indexes respectively
    LData &partitionByIndexParity1D();

    // Reorganize the last two dimensions into 4 quadrants containing elements
    // with even-even, even-odd, odd-even and odd-odd indexes respectively
    LData &partitionByIndexParity2D();
};

// Data in IRAM
class IData : public DataT<IData> {
    using DataT::DataT;
    IData(const Shape &shape, DType elementType, int offs) = delete;

  public:
    static std::string name() { return "IData"; }

    // Insert a new dimension at the specified index
    IData &insertDim(int dimIndex, const ShapeItem &item);
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

    // It's not possible to index BData (not supported by HW)
    BData operator[](IterVar index) const = delete;
    BData operator[](const Indexes &ixs) const = delete;
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

    // PData can be a 2D or 3D tensor. It is already vectorized correctly for the ACT processing.
    // If 3D the Outer dimension represents multiple groups (eg from outer products).
    enum { Outer = -3, Vectors = -2, Elements = -1 };
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
    // N can be any value up to act.width(iType)
    // idata: input tensor data in iram
    // return: pram data of shape {N}:pType
    // where pType is int32 for integer input, iType for float input
    PData load(const IData &idata);

    // Accumulate an input of shape {N}
    // N can be any value up to act.width(iType)
    // idata: input tensor data in iram
    // acc: accumulate operation (accumulate with ALUOp1Mode::BOR is equivalent to load()
    // return: pram data of shape {ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, iType for float input
    // if N < act::width the result will be {1, N}:pType
    PData accumulate(const IData &idata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC);

    // Multiply an input of shape {N} with a scalar weight (or vector of shape {1}) and accumulate
    // N can be any value up to iWidth(iType, wType)
    // idata: input tensor data in iram (A scalar is also accepted)
    // wdata: weight tensor data in wram
    // acc: accumulate operation
    // return: pram data of shape {ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    // if N < act::width the result will be {1, N}:pType
    PData scalarProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Like scalarProductAccumulate but the input must be [M,N] and the weight a vector of shape [M]
    // the scalarProductAccumulate is performed for each of the M input rows producing an output
    // of shape [M, ceil(N/act::width), act::width]
    PData multiScalarProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Outer product of an input of shape {N} with a weight vector of shape {M} and accumulate
    // N can be up to iWidth(iType, wType) (A scalar is also accepted for both input and weight)
    // idata: input tensor data in iram
    // wdata: weight tensor data in wram
    // acc: accumulate operation
    // return: pram data of shape {M, ceil(N/act::width), act::width}:pType
    // where pType is int32 for integer input, fp32 for float input
    PData outerProductAccumulate(
        const IData &idata, const WData &wdata, torq_hw::ALUOp1Mode acc = torq_hw::ALUOp1Mode::ACC
    );

    // Multiply an input of shape {N} with a weight vector of shape {N} and accumulate
    // N can be any value up to wWidth(wType) (A scalar is also accepted for both input and weight)
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
    // weightWidth if specified indicates the number of weights that will be used in outerProduct
    int iWidth(DType iType, DType wType = DType::none, int weightWidth = 0) const;

    // Max number of weight items that can be processed in parallel
    int wWidth(DType wType) const;

    // Max kernel with that can be handled directly by the ALU.
    // Above this threshold it is necessary to load each row multple times.
    int kerWidth() const;
};

// Activation Unit
class Act : SliceComponent {
    using SliceComponent::SliceComponent;

  public:
    // Load partials of shape {N} to QData
    // N can be up to width(iType)
    QData load(const PData &pdata);

    // Perform clamp on partials of shape {N} or {1, N}
    // N can be up to width(iType)
    // clamped return: result data of shape {M}
    // M can be less than M if multiple partials are combined to compute each result value
    // In case of floating point partials clipMax and clipMin are interpreted as float values.
    QData clamp(
        const PData &pdata, int clipMin, int clipMax,
        torq_hw::ACTMode actMode = torq_hw::ACTMode::ACT
    );

    // Perform rescaling and clamp on partials of shape {N} or {1, N}
    // N can be up to width(iType)
    // return: result data of shape {M}
    // M can be less than N if multiple partials are combined to compute each result value
    //
    // In case of integer partials:
    // bdata must have shape {2}:int32 or {BS,2}:int32 where BS can be 1, 2 or 4
    // Each value i in the generated QData is computed according to the following pseudocode:
    // int32_t t0 = pdata + bdata[i % BS][0];
    // int64_t t1 = t0 * bdata[i % BS][1];
    // int64_t t2 = t1 + (1 << (shift * 4 - 1));
    // int64_t t3 = t2 >> (shift * 4);
    // int64_t t4 = t3 + zeroPoint;
    // int16_t t5 = (t4 < clipMin) ? clipMin : (t4 > clipMax) ? clipMax : t4;
    // qdata = t5;
    //
    // In case of floating point partials the rescaling is not available so
    // bdata must have shape {1}:float32 or {BS,1}:float32 where BS can be 1, 2 or 4
    // Furthermore shift must be 0 and clipMax and clipMin are interpreted as float values.
    // Each value i in the generated QData is computed according to the following pseudocode:
    // float t0 = pdata + bdata[i % BS][0];
    // float t1 = (t0<clipMin) ? clipMin : (t0>clipMax) ? clipMax : t0;
    // qdata = t1;
    QData rescaleClamp(
        const PData &pdata, const BData &bdata, int shift, int zeroPoint, int clipMin, int clipMax
    );

    // Max number of output items that can be computed in parallel for the given in and weight type
    // The flag biasScalePerItem indicates if a different bias/scale is required for each item
    // (in which case the width may be reduced)
    int width(DType iType, DType wType = DType::none, bool biasScalePerItem = false) const;
};

// Create a kernel for a slice
class Slice {
    std::unique_ptr<SlicePrivate> d;

  public:
    // Create a slice kernel. Name is for debugging purposes only
    Slice(const std::string &name = {});
    ~Slice();

    // Return the name associated to the slice
    const std::string &name() const;

    // Start a for loop iterating over all the values in the range [0, count - 1]
    // returns the loop iteration variable
    IterVar forall(int count);

    // Terminate a for loop
    void endfor();

    // Create an iterator (equivalent to forall/endfor)
    Iterator iterate(int count);
    Iterator iterate(const std::vector<int> &counts);

    // Store the computed result to LRAM
    // Data types must be compatible, not necessarily the same, fp16 is compatible with float32,
    // integer types are compatible with each other.
    // QData is a vector of shape [n], output can be:
    //  - a dense vector of the same shape [n]
    //  - a scalar if QData has shape [1]
    //  - an array of shape [{g, s}, n] where g is up to Slice::scatter() (2) with dense inner dim.
    //    and s is the stride between the two inner blocks of size n (no need to be contiguous)
    //    In this case it must be s*n == alu.width() or n == 1
    //    The case n == 1 is only supported for item size <= 2
    //  - an array of shape [{n,1},{2,s}] (even-odd split mode) only supported for item size <= 2
    void store(const LData &output, const QData &data);

    // Store a value to LRAM
    // output must represent a scalar
    // Can only be used if iram, wram and bram haven't been loaded
    void store(const LData &output, int value);

    // Append the computed result to the specified position in the output tensor.
    // The result is appended after the other results already appended at that position,
    // according to the shape and stride of the indexed subtensor.
    // In case the result data would go beyond the end of the indexed subtensor
    // the data in excess is discarded.
    // QData is a vector of shape [n], output can be:
    //  - a dense vector of any rank and size
    //  - a scalar if QData has shape [1] (TBV)
    //  - a 2-dim array of shape [{n,1},{2,s}] (even-odd split mode), s is the stride between blocks
    //  - a 4-dim array of shape [{n,1},{2,s},{m,1},{2,t}] (4-quadrants even-odd segmentation mode)
    // Example, filling a tensor of 2 channels of shape 16x3, with a result of shape 5:
    // auto out = LData({2, 16, 3}, DType::int8); // output tensor
    // auto res = QData({5}, DType::int8); // computed result to append
    // For (auto n = kernel.iterate(2))
    //   For (auto i = kernel.iterate(div_ceil(16 * 3, 5)))
    //     kernel.append(out[n], res);
    // The extra 2 elements in each channels are discarded.
    void append(const LData &output, const QData &data);

    // Get max scattering supported by the slice in the store() operation
    // Scattering indicates the max number of different addresses that can be generated in parallel
    int scatter() const;

    // Configure the height and width of each input channel
    // Mandatory for convolutions
    void setInputChannelShape(int height, int width);

    // Configure the kernel size on each side, for example a 5x5 kernel will have {2, 2, 2, 2}
    // Mandatory for convolutions
    void setKernel(const LRTBDim &lrtb);

    // Configure the input padding margin and value
    // For example a 1-pixel padding on left, right, top, bottom will have {1, 1, 1, 1}
    // Mandatory for convolutions
    void setPadding(const LRTBDim &lrtb, int padValue);

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
    Iterator(const Iterator &);
    Iterator(Iterator &&);
    Iterator &operator=(const Iterator &) = delete;
    Iterator(Slice &kernel, int count);
    Iterator(Slice &kernel, const std::vector<int> &counts);
    ~Iterator();

    // Revert the iterator direction, from last to first
    Iterator reverse() &;
    Iterator reverse() &&;

    // Divide the corresponding loop count by divisor
    // Only supported for LRAM-based iterations
    Iterator operator/(int divisor);

    // Add an additional dimension with count of divisor
    // Only supported for internal-memory-based iterations
    // To be used in combination with division mode on the same iter variable with the same value.
    // The two modes together allow to split a loop in two parts, an outer part used to load data
    // from LRam to internal memory, and an inner part to load data from internal memory to ALU.
    // The original loop count doesn't have to be multiple of the modulo.
    Iterator operator%(int divisor);

    operator bool() const { return true; }
    operator Indexes() const { return _iterVars; }

  private:
    Slice &_kernel;
    Indexes _iterVars;
    bool _isCopy{};
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Iterator &iterator);

// This macro is just a syntactic helper to write Torq loops with iterators in a natural way eg:
// For (auto block = kernel.iterate(blockCount)) {
//
// }
// Above "block" represents an iterator that calls endfor at the end of the block
#define For if

//
// Utility functions
//

// return the DType corresponding to the given MLIR type
// none if no corresponding type exists
DType getDType(mlir::Type mlirType);

// return size in bytes of the given type
int sizeofType(DType type);

// return true if the given type is a floating point type
bool isFloat(DType type);

// return true if the given type is an integer type
bool isInt(DType type);

// return true if the given type is an unsigned type
bool isUnsigned(DType type);

// Return the max value for the given DType as an integer constant
int32_t maxVal(DType type);

// Return the min value for the given DType as an integer constant
int32_t minVal(DType type);

// Return true if the given type supports scale in the activation unit
bool hasScale(DType type);

// Return the number of values in each entry of the scale/bias vector for the given type
// (some types don't have scale but just bias)
int scaleBiasWidth(DType type);

// Total number of elements in the shape TODO: make this a method
int elementCount(const Shape &shape);

} // namespace torq

} // namespace mlir::syna
