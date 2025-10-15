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

LogicalResult convertToHw(torq_hl::TransposeOp op, PatternRewriter &rewriter) {
    if (supportedByOptimizedTranspose(op.getPerm())) {
        return failure();
    }

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

    // Check that the input/output tensors are compatible
    for (int i = 0; i < input_shape.size(); i++) {
        if (perm[i] < 0 || perm[i] >= input_shape.size()) {
            return op.emitError("Invalid permutation dimension: ") << perm[i];
        }
        if (input_shape[perm[i]] != output_shape[i]) {
            return op.emitError("Input and output shapes are incompatible for permutation");
        }
    }

    Slice slice;

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
        outputShape.push_back({output_shape[i], output_strides[i] * sizeofType(elementType)});
    }
#else
    // Read sequential
    Shape inputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        inputShape.push_back({input_shape[i], input_strides[i] * sizeofType(elementType)});
    }

    // The output can have any number of dimensions with any stride
    auto revPerm = getReversePerm(perm);
    Shape outputShape;
    for (int i = 0; i < input_shape.size(); ++i) {
        outputShape.push_back(
            {output_shape[revPerm[i]], output_strides[revPerm[i]] * sizeofType(elementType)}
        );
    }
#endif

    LData input(inputShape, elementType);
    LData output(outputShape, elementType);

    int endDim = inputShape.size();
    ShapeItem &topDim = inputShape.back();
    ShapeItem &outTopDim = outputShape.back();
#if TORQ_TRANSPOSE_WRITE_SEQUENTIAL
    bool topDimIsDense = topDim.stride.intVal.value() == sizeofType(elementType);
#else
    bool topDimIsDense = outTopDim.stride.intVal.value() == sizeofType(elementType);
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

    rewriter.replaceOpWithNewOp<torq_hw::SliceTaskOp>(
        op,                        // Operation to replace
        "transpose",               // Task name
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
