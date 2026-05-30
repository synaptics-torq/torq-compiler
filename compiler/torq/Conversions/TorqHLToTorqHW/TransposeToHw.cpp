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

1) Any transpose of 2-axis I, J in a tensor of rank N where J is the innermost dimension (J == N-1)
can be seen as a direct 2D transpose (even if I and J are not contiguous).
We just have to flatten everything to the left of I into a single row dimension:
(outer_batch × d(I) × middle,  d(J))
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

using namespace mlir::syna::torq_hw;

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

static Slice convertBlock(torq_hl::TransposeOp op, PatternRewriter &rewriter) {
    // In/Out tensors dimensions before reshaping
    struct Dim {
        enum { NonDenseDims, H = -2, W = -1 };
    };

    struct In {
        enum { HGroups = -6, HBlocks, Vectors, HLinePairs, HLinesInPair, Elements };
    };
    struct Out {
        enum { HGroups = -5, HLinePairs, Vectors, HLinesInPair, Elements };
    };

    LData input(op.getInput());
    LData output(op.getInit());
    DType elementType = input.elementType();
    int elementSize = sizeofType(elementType);
    assert(elementSize <= 2);

    Slice slice("transpose-block");

    // Lines that are transposed in one go
    const int blocksInGroup = 4;
    const int linesInBlock = 8;
    const int transposeBlockSize = blocksInGroup * linesInBlock;
    int vectorSize = transposeBlockSize / sizeofType(elementType);
    input.reshapeDim(Dim::H, {-1, blocksInGroup, linesInBlock / 2, 2}, true).vectorize(vectorSize);
    // Move the vectors dimension just after the HBlocks, so that we can load multiple lines at once
    input.moveDim(Vectorized::Vectors, In::Vectors);

    if (elementSize == 2) {
        // Write out chunks of 32 int16, 1 chunk per cycle (64 bytes total)
        output.reshapeDim(Dim::H, {-1, transposeBlockSize / elementSize}, true);
        output.vectorize(transposeBlockSize);
    }
    else {
        // Write out chunks of 32 int8, 2 chunks in nearby lines per cycle (64 bytes total)
        output.reshapeDim(Dim::H, {-1, transposeBlockSize / 2, 2}, true);
        output.vectorize(transposeBlockSize);
        output.moveDim(-2, Out::Vectors);
    }

    For(auto ndd = slice.iterate(input.dims(Dim::NonDenseDims, In::HGroups))) {
        // Process input rows in reverse order so that we generate the output from right to left,
        // this avoids corrupting the output already generated when size of output rows is not
        // a multiple of 32 bytes.
        For(auto hg = slice.iterate(input.dim(In::HGroups)).reverse()) {
            For(auto iv = slice.iterate(input.dim(In::Vectors))) {
                PData pdata;
                For(auto hb = slice.iterate(input.dim(In::HBlocks))) {     // 4 blocks
                    IData idata = slice.iram.load(input[ndd][hg][hb][iv]); // 8lines 32:i8 or 16:i16
                    pdata = slice.alu.transpose(idata);
                }
                For(auto outLinePair = slice.iterate(pdata.dim(PData::Vectors))) {
                    QData res = slice.act.load(pdata[outLinePair]);
                    slice.store(output[ndd][iv][outLinePair][hg], res);
                }
            }
        }
    }

    return slice;
}

static Slice convertGeneralized(torq_hl::TransposeOp op, PatternRewriter &rewriter) {
    // Generalized transpose
    ArrayRef<int64_t> perm = op.getPerm();

    // Input
    auto input_type = llvm::cast<MemRefType>(op.getInput().getType());
    auto input_shape = input_type.getShape();
    auto input_strides = getEncodedStridesElements(input_type);
    DType elementType = getDType(input_type.getElementType());
    auto elementSize = sizeofType(elementType);

    // Output
    auto output_type = llvm::cast<MemRefType>(op.getInit().getType());
    auto output_shape = output_type.getShape();
    auto output_strides = getEncodedStridesElements(output_type);

    Slice slice("transpose-generalized");

#define TORQ_TRANSPOSE_WRITE_SEQUENTIAL false
#if TORQ_TRANSPOSE_WRITE_SEQUENTIAL
    // Write sequential
    // The input can have any number of dimensions with any stride
    Shape inputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        inputShape.push_back(
            {input_shape[perm[i]], input_strides[perm[i]] * sizeofType(elementType)}
        );
    }

    // The output can have any number of dimensions with any stride
    Shape outputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        outputShape.push_back({output_shape[i], output_strides[i]});
    }
#else
    // Read sequential
    Shape inputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        inputShape.push_back({input_shape[i], input_strides[i]});
    }

    // The output can have any number of dimensions with any stride
    auto revPerm = getReversePerm(perm);
    Shape outputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        outputShape.push_back({output_shape[revPerm[i]], output_strides[revPerm[i]]});
    }
#endif

    LData input(inputShape, elementType);
    LData output(outputShape, elementType);

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

LogicalResult convertToHw(torq_hl::TransposeOp op, PatternRewriter &rewriter) {
    if (supportedByOptimizedTranspose(op.getPerm())) {
        return failure();
    }

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

    // Check permutation is identity except for the last two dimensions which are swapped
    int rank = input.shape().size();
    bool transpose2D = true;
    for (int i = 0; i < rank - 2; i++) {
        if (perm[i] != i) {
            transpose2D = false;
        }
    }
    transpose2D &= rank >= 2 && perm[rank - 1] == rank - 2 && perm[rank - 2] == rank - 1;

    int inElSize = sizeofType(input.elementType());
    bool useBlockTranspose = transpose2D && inElSize <= 2 && input.denseDims() >= 1 &&
                             output.denseDims() >= 1 && input.dim(rank - 2) >= 32 &&
                             input.dim(rank - 1) >= 32;
    Slice slice = useBlockTranspose ? convertBlock(op, rewriter) : convertGeneralized(op, rewriter);

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                        // Operation to replace
        slice.name(),              // Task name
        ValueRange{op.getInput()}, // Input tensor
        ValueRange{},              // Weights
        ValueRange{},              // BiasScale tensor,
        ValueRange{op.getInit()},  // Output tensor initializer
        ValueRange{},              // Symbols
        slice.getCfgAttr(rewriter.getContext()),
        slice.getNdls() // NDLs
    );
    return success();
}

} // namespace mlir::syna::torq
