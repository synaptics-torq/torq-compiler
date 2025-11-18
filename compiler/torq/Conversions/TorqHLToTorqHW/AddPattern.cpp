// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Patterns.h"

#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/EncodingUtils.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "torq-lower-torqhl-add"

using namespace mlir::syna::torq_hw;

namespace mlir::syna::torq {

// Non-dense Input 1 is loaded from the data path, and Input 2, which is a scalar constant,
// is loaded once as bias/scale. The ALU operates in bypass mode.
FailureOr<SliceTaskOp> buildScalarNonDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

// Dense Input 1 is loaded from the data path, and Input 2, which is a scalar constant,
// is loaded once as bias/scale. The ALU operates in bypass mode.
FailureOr<SliceTaskOp> buildScalarDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

// ALU is working with MUL and ACC operations on 32 input pixels at a time.
// The 2 inputs are loaded from 2 different addresses.
// Each input value is copied in two nearby ALU entries and multiplied by the same 16-bits weight.
// We have 4 weights bytes in total that are precomputed and stored in lram.
FailureOr<SliceTaskOp> buildNonScalarTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

LogicalResult convertToHw(torq_hl::AddOp op, PatternRewriter &rewriter);

template <>
LogicalResult AddPattern::transform(torq_hl::AddOp addOp, PatternRewriter &rewriter) const {

    BinaryOpParams<torq_hl::AddOp> params(addOp, rewriter);
    if (failed(prepareParams(params))) {
        return failure();
    }

    // check if input and output shapes match
    if (params.input1Shape.size() != params.outputShape.size()) {
        return addOp.emitError() << "Input and output shapes must match, got input shape: "
                                 << params.input1Shape.size()
                                 << " and output shape: " << params.outputShape.size();
    }
    for (size_t i = 0; i < params.input1Shape.size(); ++i) {
        if (params.input1Shape[i] != params.outputShape[i]) {
            return addOp.emitError()
                   << "Input and output shapes must match, got input shape: "
                   << params.input1Shape[i] << " and output shape: " << params.outputShape[i];
        }
    }
    FailureOr<SliceTaskOp> sliceTaskOp = failure();

    if (addOp.getRhsIsScalar() && params.input2Shape.size() == 1) {
        bool is_dense = isDenseInMemory(params.type1);
        if (!is_dense) {
            // Case 1: Input2 is scalar and Input1 is non-contiguous.
            // In this path, we use the least efficient implementation.
            // Throughput is limited by the number of elements in the last dimension,
            // and we access each dimention using the associated stride.
            // Input2 is treated as bias/scale.
            sliceTaskOp = buildScalarNonDenseTaskOp(params);
        }
        else {
            // Case 2: Input2 is scalar and Input1 is contiguous.
            // This allows an efficient implementation: Input1 is streamed from the data bus,
            // Input2 is treated as bias/scale.
            sliceTaskOp = buildScalarDenseTaskOp(params);
        }
    }
    else {
        // Case 3: Both inputs are non-scalar tensors.
        // The two tensors must have the same shape and strides.
        // Skip stride mismatch check for batch dimension (dim 0) if batch size is 1,
        // since differing strides won't affect computation in that case.
        for (size_t i = 0; i < params.input1Strides.size(); ++i) {
            if (params.input1Shape[i] != params.input2Shape[i])
                return addOp.emitError("Add input shapes must match");
            if (i == 0 && params.input1Shape[0] == 1)
                continue;
            if (params.input1Strides[i] != params.input2Strides[i]) {
                return addOp.emitError("Add input strides must matchh");
            }
        }

        sliceTaskOp = buildNonScalarTaskOp(params);
    }

    if (failed(sliceTaskOp)) {
        return failure();
    }
    rewriter.replaceOp(addOp, sliceTaskOp->getOperation()->getResults());

    return success();
}

FailureOr<SliceTaskOp> buildScalarDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params) {

    Slice slice;
    // By default consider the entire input as a single channel
    const uint32_t channelCount = 1;
    uint32_t channelSize = getEncodedTotalSizeBytes(params.type1);
    DType elementType = getDType(params.inputElementType);

    auto shift = params.op.getShiftFactor();
    auto min = params.op.getOutputMin();
    auto max = params.op.getOutputMax();
    auto zp = params.op.getOutputZp();

    // Input data that are processed at once by ALU
    const uint32_t blockSize = slice.alu.iWidth(elementType);

    // Input buffer size must be a multiple of the block size
    if (channelSize % blockSize != 0) {
        return params.op.emitError("Channel size must be multiple of " + std::to_string(blockSize));
    }

    channelSize /= params.inputElementSize; // Convert to elements count

    const uint32_t blocksPerChannel = div_ceil(channelSize, blockSize);
    const int32_t blockCount = blocksPerChannel * channelCount;

    // Data processed by ACT at once
    int actBlockSize = slice.act.width(elementType);

    LData input({blockCount, blockSize}, getDType(params.inputElementType));
    int biasNum = 2;
    if (getDType(params.inputElementType) == DType::bf16) {
        biasNum = 1;
    }
    LData biasScale({biasNum}, getDType(params.op.getScaleBias().getType().getElementType()));

    LData output(
        {blockCount, blockSize / actBlockSize, actBlockSize}, getDType(params.inputElementType)
    );

    BData bdata = slice.bram.load(biasScale);
    For(auto b = slice.iterate(blockCount)) {
        IData idata = slice.iram.load(input[b]);
        PData pdata = slice.alu.load(idata);
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.rescaleClamp(pdata[a], bdata, shift, zp, min, max);
            slice.store(output[b][a], res);
        }
    }

    auto sliceTaskOp = params.rewriter.create<SliceTaskOp>(
        params.loc,                                     // Location
        params.op.getName(),                            // Operation to replace
        ValueRange{params.op.getInput1()},              // Input tensor
        ValueRange{},                                   // Weights
        ValueRange{params.op.getScaleBias()},           // BiasScale tensor,
        ValueRange{params.op.getInit()},                // Output tensor initializer
        ValueRange{},                                   // Symbols
        slice.getCfgAttr(params.rewriter.getContext()), // Slice configuration
        slice.getNdls()                                 // NDLs
    );
    return sliceTaskOp;
}

FailureOr<SliceTaskOp> buildScalarNonDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params) {

    Slice slice;
    DType elementType = getDType(params.inputElementType);
    Shape inputShape;

    auto shift = params.op.getShiftFactor();
    auto min = params.op.getOutputMin();
    auto max = params.op.getOutputMax();
    auto zp = params.op.getOutputZp();

    for (int i = 0; i < params.input1Shape.size() - 1; ++i) {
        inputShape.push_back({params.input1Shape[i], params.input1Strides[i]});
    }
    int lastDimSize = params.input1Shape.back();

    if (lastDimSize <= slice.act.width(elementType)) {
        inputShape.push_back({lastDimSize, 1});
    }
    else {
        // If the last dimension is larger than act width, we need to split it into blocks
        int blocks = div_ceil(lastDimSize, slice.act.width(elementType));
        inputShape.push_back({blocks, slice.act.width(elementType)});
        inputShape.push_back({slice.act.width(elementType), 1});
    }

    // The output is a dense version of the input
    Shape outputShape;
    for (int i = 0; i < inputShape.size(); ++i) {
        outputShape.push_back(inputShape[i]);
    }
    int biasNum = 2;
    if (getDType(params.inputElementType) == DType::bf16) {
        biasNum = 1;
    }
    LData biasScale({biasNum}, getDType(params.op.getScaleBias().getType().getElementType()));
    LData input(inputShape, elementType, params.inputOffsetBytes);
    LData output(outputShape, elementType);
    BData bdata = slice.bram.load(biasScale);

    For(auto ii = slice.iterate(input.dims(0, -1))) {
        IData idata = slice.iram.load(input[ii]);
        PData pdata = slice.alu.load(idata);
        QData res = slice.act.rescaleClamp(pdata, bdata, shift, zp, min, max);
        slice.store(output[ii], res);
    }

    auto sliceTaskOp = params.rewriter.create<SliceTaskOp>(
        params.loc,
        params.op.getName(),                            // Operation to replace
        ValueRange{params.op.getInput1()},              // Input tensor
        ValueRange{},                                   // Weights
        ValueRange{params.op.getScaleBias()},           // BiasScale tensor,
        ValueRange{params.op.getInit()},                // Output tensor initializer
        ValueRange{},                                   // Symbols
        slice.getCfgAttr(params.rewriter.getContext()), // Slice configuration
        slice.getNdls()                                 // NDLs
    );
    return sliceTaskOp;
}

FailureOr<SliceTaskOp> buildNonScalarTaskOp(BinaryOpParams<torq_hl::AddOp> &params) {
    auto op = params.op;
    const auto shift = op.getShiftFactor();
    const auto min = op.getOutputMin();
    const auto max = op.getOutputMax();
    const auto zp = op.getOutputZp();

    // Define data tensors
    LData input(params.input1);
    LData output(params.init);
    LData weights(op.getWeights());
    LData biasScale(op.getScaleBias());
    assert(weights.dim(0) == 2);
    assert(biasScale.dim(0) == scaleBiasWidth(input.elementType()));

    // Dimensions of the input data for processing
    struct In : Vectorized {
        enum {
            InputTensors, // Selects between the two input tensors
            DataDim,      // First (non-dense) data dimension if any
        };
    };

    // Add an additional dimension of size 2 at the beginning to represent the 2 input tensors
    // The difference between the two input addresses is used as offset to fetch the data
    auto inputDiff = getAffineDimExpr(1, params.ctx) - getAffineDimExpr(0, params.ctx);
    input.insertDim(In::InputTensors, {2, inputDiff});

    // Take care of output segmentation if requested
    int denseDims = std::min(input.denseDims(), output.denseDims());
    if (op.getSegmentOutput()) {
        // In this case vectorize on H & W dimensions only, so the output can be segmented
        denseDims = 2;
        output.partitionByIndexParity2D();
    }

    // Vectorize the input tensor
    Slice slice;
    int vectorSize = slice.alu.iWidth(input.elementType(), weights.elementType());
    input.fuse(denseDims).vectorize(vectorSize);

    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);
    For(auto ii = slice.iterate(input.dims(In::DataDim, In::Vectors))) {
        For(auto dv = slice.iterate(input.dim(In::Vectors))) {
            PData pdata;
            For(auto i = slice.iterate(input.dim(In::InputTensors))) {
                IData idata = slice.iram.load(input[i][ii][dv]);
                pdata = slice.alu.scalarProductAccumulate(idata, wdata[i]);
            }
            For(auto a = slice.iterate(pdata.dim(0))) {
                QData res = slice.act.rescaleClamp(pdata[a], bdata, shift, zp, min, max);
                slice.append(output[ii], res);
            }
        }
    }

    auto input1Addr = params.rewriter.create<GetAddressOp>(params.loc, params.input1).getAddress();
    auto input2Addr = params.rewriter.create<GetAddressOp>(params.loc, params.input2).getAddress();
    auto sliceTaskOp = params.rewriter.create<SliceTaskOp>(
        params.loc,                               // Operation to replace
        op.getName(),                             // Task name
        ValueRange{params.input1, params.input2}, // Input tensor
        ValueRange{op.getWeights()},              // Weights
        ValueRange{op.getScaleBias()},            // BiasScale tensor
        ValueRange{op.getInit()},                 // Output tensor initializer
        ValueRange{input1Addr, input2Addr},       // Symbols used to compute the NDLs
        slice.getCfgAttr(params.ctx),             // Slice configuration
        slice.getNdls()
    );
    return sliceTaskOp;
}

} // namespace mlir::syna::torq
