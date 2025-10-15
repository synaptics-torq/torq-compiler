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
//
// 1. Read the input data in blocks of 32 pixels, duplicating each of them in 2 ALU entries.
// 2. Read the weights in blocks of 4 bytes (2 weights).
// 3. Feed the first input data to ALU with first 16bits weight and process
// 4. Feed the second input data to ALU with next 16bits weight and process
// 5. The ALU will output 64 bytes of data.
// 6. Read the bias & scale
// 7. The output is rescaled to int8 in blocks of 8
// 8. The output is written back to LRAM
// 9. Take the next block of 32 pixels and repeat the process.
// 10. Repeat the process for the next channel
FailureOr<SliceTaskOp> buildDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

LogicalResult convertToHw(torq_hl::AddOp op, PatternRewriter &rewriter);

template <>
LogicalResult AddPattern::transform(torq_hl::AddOp addOp, PatternRewriter &rewriter) const {

    BinaryOpParams<torq_hl::AddOp> params(addOp, rewriter);
    if (failed(prepareParams(params))) {
        return failure();
    }
    // Skip stride mismatch check for batch dimension (dim 0) if batch size is 1,
    // since differing strides won't affect computation in that case.
    for (size_t i = 0; i < params.input1Strides.size(); ++i) {
        if (i == 0 && params.input1Shape[0] == 1)
            continue;

        // FIXME: we should change input strides somewhere if it is from subview
        // otherwise we cannot get correct strides to get things work as we have 2 inputs whose
        // strides might be different but it is needed for hardware computation

        // if (params.input1Strides[i] != params.input2Strides[i]) {
        //     return addOp.emitError("Input strides must matchh");
        // }
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

    if (addOp.getRhsIsConst() && params.input2Shape.size() == 1) {
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
        // Case 3: Both inputs are non-scalar, dense layout tensors.
        sliceTaskOp = buildDenseTaskOp(params);
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
    const int elementSize = sizeofType(elementType);

    auto shift = params.op.getShiftFactor();
    auto min = params.op.getOutputMin();
    auto max = params.op.getOutputMax();
    auto zp = params.op.getOutputZp();

    for (int i = 0; i < params.input1Shape.size() - 1; ++i) {
        inputShape.push_back({params.input1Shape[i], params.input1Strides[i] * elementSize});
    }
    int lastDimSize = params.input1Shape.back();

    if (lastDimSize <= slice.act.width(elementType)) {
        inputShape.push_back({lastDimSize, 1 * elementSize});
    }
    else {
        // If the last dimension is larger than act width, we need to split it into blocks
        int blocks = div_ceil(lastDimSize, slice.act.width(elementType));
        inputShape.push_back({blocks, slice.act.width(elementType) * elementSize});
        inputShape.push_back({slice.act.width(elementType), 1 * elementSize});
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

FailureOr<SliceTaskOp> buildDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params) {

    Slice slice;
    auto shift = params.op.getShiftFactor();
    auto min = params.op.getOutputMin();
    auto max = params.op.getOutputMax();
    auto zp = params.op.getOutputZp();

    // By default consider the entire input as a single channel
    const uint32_t channelCount = 1;
    uint32_t channelSize = getEncodedTotalSizeBytes(params.type1);

    // Number of inputs to be added
    const uint32_t inputCount = 2;

    // We have a single 16-bits weight for each input
    DType weightType = params.inputElementType.isInteger(32) ? DType::int8 : DType::int16;
    DType inputType = getDType(params.inputElementType);

    // Input data that are processed at once by ALU
    uint32_t blockSize = slice.alu.iWidth(inputType, weightType);

    auto segment_output = params.op.getSegmentOutput();
    if (segment_output) {
        // If output is segmented, we have to process channel by channel
        if (params.outputShape.size() != 4) {
            return params.op.emitError("Output segmentation only available for rank 4 tensors");
        }
        channelSize = params.input1Strides[1];
        assert(false && "AddOp segment output is not supported yet");
    }

    // Input buffer size must be a multiple of the block size
    if (channelSize % blockSize != 0) {
        return params.op.emitError("Channel size must be multiple of " + std::to_string(blockSize));
    }

    channelSize /= params.inputElementSize; // Convert to elements count

    const uint32_t blocksPerChannel = div_ceil(channelSize, blockSize);
    const int32_t blockCount = blocksPerChannel * channelCount;

    // The difference between the two input addresses is used as offset in DEDR to fetch the
    // data
    auto input1Address =
        params.rewriter.create<GetAddressOp>(params.loc, params.input1).getAddress();
    auto input2Address =
        params.rewriter.create<GetAddressOp>(params.loc, params.input2).getAddress();
    auto inputDiff = getAffineDimExpr(1, params.ctx) - getAffineDimExpr(0, params.ctx);

    // Data processed by ACT at once
    uint32_t actBlockSize = slice.act.width(inputType, weightType);

    LData input(
        {{inputCount, inputDiff}, blockCount, blockSize}, getDType(params.inputElementType)
    );

    LData weights({inputCount}, getDType(params.op.getWeights().getType().getElementType()));
    int biasNum = 2;
    if (getDType(params.inputElementType) == DType::bf16) {
        biasNum = 1;
    }
    LData biasScale({biasNum}, getDType(params.op.getScaleBias().getType().getElementType()));

    LData output(
        {blockCount, blockSize / actBlockSize, actBlockSize},
        getDType(params.op.getInit().getType().getElementType())
    );

    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);
    For(auto b = slice.iterate(blockCount)) {
        PData pdata;
        For(auto i = slice.iterate(inputCount)) {
            IData idata = slice.iram.load(input[i][b]);
            pdata = slice.alu.scalarProductAccumulate(idata, wdata[i]);
        }
        For(auto a = slice.iterate(blockSize / actBlockSize)) {
            QData res = slice.act.rescaleClamp(pdata[a], bdata, shift, zp, min, max);
            slice.store(output[b][a], res);
        }
    }

    auto sliceTaskOp = params.rewriter.create<SliceTaskOp>(
        params.loc,                               // Operation to replace
        params.op.getName(),                      // Task name
        ValueRange{params.input1, params.input2}, // Input tensor
        ValueRange{params.op.getWeights()},       // Weights
        ValueRange{params.op.getScaleBias()},     // BiasScale tensor
        ValueRange{params.op.getInit()},          // Output tensor initializer
        ValueRange{input1Address, input2Address}, // Symbols used to compute the NDLs
        slice.getCfgAttr(params.ctx),             // Slice configuration
        slice.getNdls()
    );
    return sliceTaskOp;
}

} // namespace mlir::syna::torq
