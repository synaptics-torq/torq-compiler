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
// is loaded once as bias/scale. The ALU multiplies Input 1 by the scalar weight (-1 for
// `c - x`).
FailureOr<SliceTaskOp> buildScalarNonDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

// Dense Input 1 is loaded from the data path, and Input 2, which is a scalar constant,
// is loaded once as bias/scale. The ALU multiplies Input 1 by the scalar weight (-1 for
// `c - x`).
FailureOr<SliceTaskOp> buildScalarDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params);

// ALU is working with MUL and ACC operations on 32 input pixels at a time.
// The 2 inputs are loaded from 2 different addresses. For non-fp32 data, each
// input value is copied in two nearby ALU entries and multiplied by the same
// 16-bits weight (4 weights bytes in total precomputed in lram). fp32 data
// cannot use the MUL mode, which only takes 16-bit weights, so it is added
// with the accumulate mode's implicit unit weights instead.
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
            // Special case: when output segmentation is enabled, the height dimension
            // can be larger by 1 (e.g., input = 1x64x33x64xi8 and output = 1x64x34x64xi8).
            if (i == 2 && addOp.getSegmentOutput() &&
                params.input1Shape[i] + 1 == params.outputShape[i])
                continue;
            return addOp.emitError()
                   << "Input and output shapes must match, got input shape: "
                   << params.input1Shape[i] << " and output shape: " << params.outputShape[i];
        }
    }
    FailureOr<SliceTaskOp> sliceTaskOp = failure();

    if (addOp.getRhsIsScalar() && params.input2Shape.size() == 1) {
        // The dense path streams input1 as one linear block and writes the output as a single
        // sweep; the DEQW it generates uses the XY view (S dims), which discards any unaligned
        // tail, so it is safe for any physically contiguous data. isDenseInMemory only
        // recognizes identity layouts, but packed strided subviews (e.g. the halves of a
        // channel-sliced activation) are contiguous too, so accept both layouts when the data
        // is packed. The output must be packed for the sweep route; the identity-input route
        // keeps not checking the output: the non-dense fallback assumes the output is laid out
        // like input1 (it must not see a gapped one), while the dense builder derives its
        // write NDL from the output's own layout, which the DEQW bounds check verifies.
        bool is_dense = isDenseInMemory(params.type1) || (isContiguousInMemory(params.type1) &&
                                                          isContiguousInMemory(params.outputType));
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
        // input2 is loaded as a data tensor here, so element types must match.
        if (params.type1.getElementType() != params.type2.getElementType())
            return addOp.emitError("Add input element types must match");
        if (params.input1Shape != params.input2Shape) {
            return addOp.emitError("Add input shapes must match");
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
    Slice slice("add");
    // By default consider the entire input as a single channel
    DType elementType = getDType(params.inputElementType);

    auto shift = params.op.getShiftFactor();
    auto min = params.op.getOutputMin();
    auto max = params.op.getOutputMax();
    auto zp = params.op.getOutputZp();

    int biasNum = 2;
    if (getDType(params.inputElementType) == DType::bf16) {
        biasNum = 1;
    }
    LData biasScale({biasNum}, getDType(params.op.getScaleBias().getType().getElementType()));

    LData input(params.op.getInput1());
    LData output(params.op.getInit());
    LData weights({1}, params.inputElementType.isInteger(32) ? DType::int8 : elementType);
    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    // Dimensions of the input data for processing
    struct In : Vectorized {
        enum {
            NonDenseDims, // First (non-dense) data dimension if any
        };
    };
    int denseDims = std::min(input.denseDims(), output.denseDims());
    int vectorSize = slice.alu.iWidth(input.elementType(), weights.elementType());
    input.fuse(denseDims).vectorize(vectorSize);

    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            IData idata = slice.iram.load(input[ndd][iv]);
            PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                QData res = slice.act.rescaleClamp(pdata[av], bdata, shift, zp, min, max);
                slice.append(output[ndd], res);
            }
        }
    }

    auto sliceTaskOp = SliceTaskOp::create(
        params.rewriter,
        params.loc,                                     // Location
        params.op.getName(),                            // Operation to replace
        ValueRange{params.op.getInput1()},              // Input tensor
        ValueRange{params.op.getWeights()},             // Weights
        ValueRange{params.op.getScaleBias()},           // BiasScale tensor,
        ValueRange{params.op.getInit()},                // Output tensor initializer
        ValueRange{},                                   // Symbols
        slice.getCfgAttr(params.rewriter.getContext()), // Slice configuration
        slice.getNdls()                                 // NDLs
    );
    return sliceTaskOp;
}

// The largest block size not above `maxBlock` that divides `size`.
static int largestDivisorAtMost(int size, int maxBlock) {
    for (int block = std::min(size, maxBlock); block > 1; --block) {
        if (size % block == 0) {
            return block;
        }
    }
    return 1;
}

FailureOr<SliceTaskOp> buildScalarNonDenseTaskOp(BinaryOpParams<torq_hl::AddOp> &params) {
    Slice slice("add");
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
    // The input is multiplied by the scalar weight, as in buildScalarDenseTaskOp: it is
    // -1 for `c - x`, which a plain ALU load would turn into `x + c`.
    DType weightType = params.inputElementType.isInteger(32) ? DType::int8 : elementType;

    // A last dimension larger than the act width is split into equal blocks: a
    // partial last block would make the NDLs read and write past the end of the row.
    // The width depends on the weight type: one activation pass holds 4 int16 x int16
    // or int32 x int8 products, but 8 bf16 or 16 int8 ones.
    int blockSize = largestDivisorAtMost(lastDimSize, slice.act.width(elementType, weightType));
    if (blockSize == lastDimSize) {
        inputShape.push_back({lastDimSize, 1});
    }
    else {
        inputShape.push_back({lastDimSize / blockSize, blockSize});
        inputShape.push_back({blockSize, 1});
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
    LData weights({1}, weightType);
    WData wdata = slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);

    For(auto ii = slice.iterate(input.dims(0, -1))) {
        IData idata = slice.iram.load(input[ii]);
        PData pdata = slice.alu.scalarProductAccumulate(idata, wdata);
        For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
            QData res = slice.act.rescaleClamp(pdata[av], bdata, shift, zp, min, max);
            slice.append(output[ii], res);
        }
    }

    auto sliceTaskOp = SliceTaskOp::create(
        params.rewriter, params.loc,
        params.op.getName(),                            // Operation to replace
        ValueRange{params.op.getInput1()},              // Input tensor
        ValueRange{params.op.getWeights()},             // Weights
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
    LData input1(params.input1);
    LData input2(params.input2);
    LData output(params.init);
    LData weights(op.getWeights());
    LData biasScale(op.getScaleBias());
    assert(weights.dim(0) == 2);
    assert(biasScale.dim(0) == biasScaleWidth(input1.elementType()));

    // The ALU MUL mode only supports 16-bit weights, so fp32 tensors are added
    // with the accumulate mode, whose implicit weights are 1.0 (exact for f32).
    // It cannot apply a -1.0: the sign-invert accumulate (SACC) alternates
    // [1, -1, 1, -1, ...] within a single data load, and the two tensors are
    // loaded separately here. AddOpPattern (createBf16Add) checks the weight
    // constant and rejects any non-unit-weight f32 tensor-tensor op before the
    // torq_hl.add is created, so those fall back to the host. At this stage the
    // weights are a slice-program argument DMA'd to LRAM rather than an
    // inspectable constant, so assert the producer's "sub" label instead of
    // failing the compilation.
    const bool isFp32 = input1.elementType() == DType::fp32;
    if (isFp32) {
        assert(
            op.getName() != "sub" && "f32 tensor-vs-tensor sub must be rejected in AddOpPattern"
        );
    }

    // A per-channel scale_bias arrives as the 2-D [channels, pairWidth] table. Serving it needs
    // the channel dimension blocked and the BRAM reloaded per block, the way FMAPattern /
    // Conv2DToHw / DWToHw do; the single load below would read only the first block. This was an
    // assert, which a release build compiles out and then reads past the table, so check it.
    if (biasScale.dims().size() != 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "add: per-channel scale_bias needs blocked BRAM loads, not supported\n"
        );
        return failure();
    }
    if (biasScale.dim(0) != biasScaleWidth(input1.elementType())) {
        LLVM_DEBUG(llvm::dbgs() << "add: unexpected scale_bias width\n");
        return failure();
    }

    // Dimensions of the input data for processing
    struct In : Vectorized {
        enum {
            InputTensors, // Selects between the two input tensors
            NonDenseDims, // First (non-dense) data dimension if any
        };
    };

    // Take care of output segmentation if requested
    int denseInDims = std::min(input1.denseDims(), input2.denseDims());
    int denseOutDims = output.denseDims();
    int denseDims = std::min(denseInDims, denseOutDims);
    if (op.getSegmentOutput()) {
        // In this case vectorize on H & W dimensions only, so the output can be segmented
        assert(denseOutDims >= 2 && "Output must have at least 2 dense dims for segmentation");
        denseDims = 2;
        output.partitionByIndexParity2D();
    }

    // Vectorize the input tensors and stack them
    Slice slice("add");
    int vectorSize = isFp32
                         ? slice.alu.iWidthForAccumulateClamp(input1.elementType(), ALUOp1Mode::ACC)
                         : slice.alu.iWidth(input1.elementType(), weights.elementType());
    input1.fuse(denseDims).vectorize(vectorSize);
    input2.fuse(denseDims).vectorize(vectorSize);
    auto input = stack(params.rewriter, input1, input2);

    WData wdata = isFp32 ? WData({}, DType::none) : slice.wram.load(weights);
    BData bdata = slice.bram.load(biasScale);
    For(auto ndd = slice.iterate(input.dims(In::NonDenseDims, In::Vectors))) {
        For(auto iv = slice.iterate(input.dim(In::Vectors))) {
            PData pdata;
            For(auto it = slice.iterate(input.dim(In::InputTensors))) {
                IData idata = slice.iram.load(input[it][ndd][iv]);
                if (isFp32) {
                    pdata = slice.alu.accumulate(idata);
                }
                else {
                    pdata = slice.alu.scalarProductAccumulate(idata, wdata[it]);
                }
            }
            For(auto av = slice.iterate(pdata.dim(PData::Vectors))) {
                QData res = slice.act.rescaleClamp(pdata[av], bdata, shift, zp, min, max);
                slice.append(output[ndd], res);
            }
        }
    }

    return slice.createSliceTaskOp(params.rewriter, params.loc);
}

} // namespace mlir::syna::torq
