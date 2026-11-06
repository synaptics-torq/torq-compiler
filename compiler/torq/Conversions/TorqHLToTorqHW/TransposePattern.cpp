// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-lower-torqhl-transpose"

/*

Transposes have different cases that can be optimized differently.

1) Any transpose that keeps the outer (batch) dimensions in place and swaps the two groups of
adjacent dimensions that follow, the second group ending with the innermost dimension, is a 2D
transpose of a (rows, cols) matrix for each batch, with rows and cols the product of the dimensions
in each group. For example NCHW -> NHWC is a (C, H*W) transpose and NHWC -> NCHW a (H*W, C) one.
This only requires the dimensions in each group to be contiguous (so that they can be fused).
This can take full advantage of HW support for 2D transposes, efficiency will be reduced if the
number of rows or columns is much less then 32.

/!\: this can't be applied if the data type is int8 and the number of input rows is < 32.
This is because we generate output in parallel for two overlapping 32-bytes rows and it is not clear
how HW handles collision in this case (is it deterministic or not?)
The issue doesn't occour for int16 (we generate one row at a time in the right order) or if
the number of rows is == 32 (aligned) or > 32 (we generate each output rows in multiple steps in the
right order.
Unfortunately there is no way to generate output one row at a time for int8, except if the
HW has supportACPRs32.

2) if multiple axis are swapped, we can use the generalized transpose which is very slow but
doesn't have any limitation nor aligment requirement.

3) if J is not the innermost axis, or size of the data type > 2 we can can again use the
generalized transpose

4) optimized generalized transpose for innermost dimension size

5) if multiple axis are swapped, with one axis being the innermost dimension, we could probably
combine the optimized transpose for innermost dimension with generalized transpose for the others


Notes:

N1) regarding alignment requirements, since we generate output in the right order there is no need
to pad the output rows to multiple of 32 bytes, we just have to ensure the overall output tensor
size is a multiple of 1024 *bytes* (32x32).

N2) for small transposes check if possible to load/store only a part of the 1024 bytes block

N3) introduced packed types so that we can keep the original input/output type with the correct
strides which would not be possible if we convert the output type to int32

*/

namespace mlir::syna::torq {

static std::vector<int64_t> getReversePerm(std::vector<int64_t> perm) {
    std::vector<int64_t> reversePerm(perm.size());
    for (int i = 0; i < perm.size(); i++) {
        // find position of i in permutation
        auto it = std::find(perm.begin(), perm.end(), i);
        assert(it != perm.end() && "Permutation is not a permutation");
        reversePerm[i] = std::distance(perm.begin(), it);
    }
    return reversePerm;
}

// Fuse dimensions [begin, end) of shape into a single dimension.
// Returns nullopt if the dimensions are not contiguous
static std::optional<ShapeItem> fuseDims(const Shape &shape, int begin, int end) {
    ShapeItem fused = shape[end - 1];
    for (int i = end - 2; i >= begin; i--) {
        if (shape[i].count == 1) {
            continue;
        }
        if (fused.count == 1) {
            fused = shape[i];
            continue;
        }
        if (shape[i].stride.intVal != fused.count * fused.stride.intVal.value()) {
            return std::nullopt;
        }
        fused.count *= shape[i].count;
    }
    return fused;
}

// Reshape input and output of a transpose that swaps two groups of dimensions (see
// isGroupSwapTranspose()) to 2D transposes: input (batch..., rows, cols) and output
// (batch..., cols, rows). Returns false if the dimensions in a group are not contiguous.
static bool reshapeTo2D(LData &input, LData &output, int batchDims, int rowDims) {
    const Shape &in = input.shape();
    const Shape &out = output.shape();
    int rank = in.size();
    int colDims = rank - batchDims - rowDims;
    auto inRows = fuseDims(in, batchDims, batchDims + rowDims);
    auto inCols = fuseDims(in, batchDims + rowDims, rank);
    auto outCols = fuseDims(out, batchDims, batchDims + colDims);
    auto outRows = fuseDims(out, batchDims + colDims, rank);
    if (!inRows || !inCols || !outCols || !outRows) {
        return false;
    }
    Shape in2D(in.begin(), in.begin() + batchDims);
    in2D.push_back(*inRows);
    in2D.push_back(*inCols);
    Shape out2D(out.begin(), out.begin() + batchDims);
    out2D.push_back(*outCols);
    out2D.push_back(*outRows);
    input.setShape(in2D);
    output.setShape(out2D);
    return true;
}

static Slice convertBlock(LData input, LData output) {
    // In/Out tensors dimensions before reshaping
    struct Dim {
        enum { NonDenseDims, H = -2, W = -1 };
    };

    struct In : Vectorized {
        enum { HGroups = -6, HBlocks, HLinePairs, HLinesInPair };
    };
    struct Out {
        enum { HGroups = -5, HWriteGroups, RowsPerWrite, WChunks, ElementsInChunk };
    };

    int elSize = sizeofType(input.elementType());
    assert(elSize <= 2);

    Slice slice("transpose-block");

    // Lines that are transposed in one go
    const int blocksInGroup = 4;
    const int linesInBlock = 8;
    const int transposeBlockSize = blocksInGroup * linesInBlock;
    int vectorSize = transposeBlockSize / elSize;
    input.forceReshapeDim(Dim::H, {-1, blocksInGroup, linesInBlock / 2, 2}).vectorize(vectorSize);

    // i8: write out chunks of 32 values, 2 chunks in nearby lines per cycle (64 bytes total)
    // i16/bf16: write out chunks of 32 values, 1 chunk per cycle (64 bytes total)
    // /!\ the last chunk of a row is written whole: we assume that the output tensor has been
    // either padded to whole blocks or cut to whole chunks by the caller
    int rowsPerWrite = elSize == 1 ? 2 : 1;
    output.forceReshapeDim(Dim::H, {-1, transposeBlockSize / elSize / rowsPerWrite, rowsPerWrite});
    output.forceReshapeDim(-1, {-1, transposeBlockSize});

    For(auto ndd = slice.iterate(input.dims(Dim::NonDenseDims, In::HGroups))) {
        // Process input rows in reverse order so that we generate the output from right to left,
        // this avoids corrupting the output already generated when size of output rows is not
        // a multiple of 32 bytes.
        For(auto hg = slice.iterate(input.dim(In::HGroups)).reverse()) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto hb = slice.iterate(input.dim(In::HBlocks))) { // 4 blocks
                    // Load 8 lines of 32:i8 or 16:i16
                    IData idata = slice.iram.load(input[ndd][hg][hb][":"][":"][iv]);
                    pdata = slice.alu.transpose(idata);
                }
                For(auto outLinePair = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.load(pdata[outLinePair]);
                    slice.store(output[ndd][iv][outLinePair][":"][hg], res);
                }
            }
        }
    }

    return slice;
}

#define TORQ_TRANSPOSE_WRITE_SEQUENTIAL false

// Generalized transpose of input to output, the output being given in the order of the input dims
// (its dims have the strides of the transposed dims), reading the input sequentially
static Slice convertGeneralizedShapes(LData input, LData output) {
    DType elementType = input.elementType();
    int elementSize = sizeofType(elementType);
    Slice slice("transpose-generalized");
    Shape inputShape = input.shape();
    Shape outputShape = output.shape();

    int endDim = inputShape.size();
    ShapeItem &topDim = inputShape.back();
    ShapeItem &outTopDim = outputShape.back();
#if TORQ_TRANSPOSE_WRITE_SEQUENTIAL
    bool topDimIsDense = topDim.stride.intVal.value() == 1;
#else
    bool topDimIsDense = outTopDim.stride.intVal.value() == 1;
#endif

    //  TODO: handle the case when dim is a power of 2
    //  bool topDimIsPowerOfTwo = dim.count & (dim.count - 1) == 0;
    if (topDimIsDense && topDim.count > 1 && topDim.count < slice.act.width(elementType)) {
        // Top dim is dense and smaller than ACT width, can be done in one go
        endDim -= 1;
    }
    else if (topDim.count % 2 == 0 && elementSize <= 2 && !TORQ_TRANSPOSE_WRITE_SEQUENTIAL) {
        // We can use even/odd scatter feature in DEQW. The shape adjustment here is also fine for
        // DEDR gather, but that doesn't support gethering 1 element at time
        // so can't be used for this purpose.
        // What we do here is to split the last dim (col) in two, and use the scatter feature to
        // process two elements at a time. The two elements are contiguous in input but separated
        // by one line in output.
        int scatter = 2;
        topDim.count /= scatter;
        int inStride = topDim.stride.intVal.value();
        topDim.stride.intVal = inStride * scatter;
        inputShape.push_back({scatter, inStride});

        outTopDim.count /= scatter;
        int outStride = outTopDim.stride.intVal.value();
        outTopDim.stride.intVal = outStride * scatter;
        outputShape.push_back({scatter, outStride});

        input.setShape(inputShape);
        output.setShape(outputShape);
    }

    For(auto ii = slice.iterate(input.dims(0, endDim))) {
        IData idata = slice.iram.load(input[ii]);
        PData pdata = slice.alu.load(idata);
        QData res = slice.act.load(pdata);
        slice.store(output[ii], res);
    }

    return slice;
}

static Slice convertGeneralized(torq_hl::TransposeOp op, PatternRewriter &rewriter) {
    // Generalized transpose
    ArrayRef<int64_t> perm = op.getPerm();
    LData input(op.getInput());
    LData output(op.getInit());

#if TORQ_TRANSPOSE_WRITE_SEQUENTIAL
    // Write sequential
    // The input can have any number of dimensions with any stride
    Shape inputShape;
    for (int i = 0; i < input.shape().size(); ++i) {
        inputShape.push_back(input.shape()[perm[i]]);
    }

    // The output can have any number of dimensions with any stride
    Shape outputShape;
    for (int i = 0; i < output.shape().size(); ++i) {
        outputShape.push_back(output.shape()[i]);
    }
#else
    // Read sequential
    Shape inputShape;
    for (int i = 0; i < input.shape().size(); ++i) {
        inputShape.push_back(input.shape()[i]);
    }

    // The output can have any number of dimensions with any stride
    auto revPerm = getReversePerm(perm);
    Shape outputShape;
    for (int i = 0; i < output.shape().size(); ++i) {
        outputShape.push_back(output.shape()[revPerm[i]]);
    }
#endif

    input.setShape(inputShape);
    output.setShape(outputShape);
    return convertGeneralizedShapes(input, output);
}

template <>
LogicalResult
TransposePattern::transform(torq_hl::TransposeOp op, PatternRewriter &rewriter) const {
    LData input(op.getInput());
    LData output(op.getInit());

    // Check that the input/output tensors are compatible
    ArrayRef<int64_t> perm = op.getPerm();
    for (int i = 0; i < input.shape().size(); i++) {
        if (perm[i] < 0 || perm[i] >= input.shape().size()) {
            return op.emitError("Invalid permutation dimension: ") << perm[i];
        }
        if (input.dim(perm[i]) != output.dim(i)) {
            return op.emitError("Input and output shapes are incompatible for permutation");
        }
    }

    // Use the block transpose if this is a 2D transpose of each batch (useBlockTranspose()) and the
    // groups of dims can be fused
    int batchDims = 0;
    int rowDims = 0;
    SmallVector<int64_t> inputShape;
    for (int i = 0; i < input.shape().size(); i++) {
        inputShape.push_back(input.dim(i));
    }
    bool useBlockTranspose =
        torq::useBlockTranspose(
            perm, inputShape, sizeofType(input.elementType()), batchDims, rowDims
        ) &&
        reshapeTo2D(input, output, batchDims, rowDims) && input.denseDims() >= 1 &&
        output.denseDims() >= 1;

    SmallVector<Slice> slices;
    if (useBlockTranspose && usePaddedBlockTranspose(batchDims, perm.size())) {
        // Swap of the last two dims: the output is padded to whole blocks (see
        // TransposeOp::getKernelEncoding()), so the block kernel transposes it all
        slices.push_back(convertBlock(input, output));
    }
    else if (useBlockTranspose) {
        // The block kernel transposes groups of 32 input rows into 32 elements of the output rows,
        // and vectors of 32 bytes of the input cols into as many output rows. To keep its writes
        // inside the output without padding it, it only transposes input rows [0, rows32) and
        // cols [0, cols32), rounded down to these sizes, and the generalized kernel transposes the
        // tails into the same output: the input cols [cols32, cols) of all rows, then the input
        // rows [rows32, rows) of cols [0, cols32).
        // Input: (batch..., rows, cols), output: (batch..., cols, rows)
        const int r = batchDims, c = batchDims + 1;
        const int64_t rows = input.dim(r), cols = input.dim(c);
        constexpr int64_t blockSize = 32;
        const int64_t vectorSize = blockSize / sizeofType(input.elementType());
        const int64_t rows32 = rows - rows % blockSize;
        const int64_t cols32 = cols - cols % vectorSize;
        auto piece = [&](int64_t r0, int64_t r1, int64_t c0, int64_t c1, bool block) {
            LData in = input;
            LData out = output;
            in.subviewDim(r, r0, r1 - r0).subviewDim(c, c0, c1 - c0);
            out.subviewDim(r, c0, c1 - c0).subviewDim(c, r0, r1 - r0);
            if (block) {
                slices.push_back(convertBlock(in, out));
                return;
            }
            // The generalized kernel takes the output in the order of the input dims
            Shape outShape = out.shape();
            std::swap(outShape[r], outShape[c]);
            out.setShape(outShape);
            slices.push_back(convertGeneralizedShapes(in, out));
        };
        piece(0, rows32, 0, cols32, true);
        if (cols32 < cols) {
            piece(0, rows, cols32, cols, false);
        }
        if (rows32 < rows) {
            piece(rows32, rows, 0, cols32, false);
        }
    }
    else {
        slices.push_back(convertGeneralized(op, rewriter));
    }

    for (Slice &slice : slices) {
        slice.createOp(rewriter, op.getLoc());
    }
    rewriter.eraseOp(op);
    return success();
}

} // namespace mlir::syna::torq
