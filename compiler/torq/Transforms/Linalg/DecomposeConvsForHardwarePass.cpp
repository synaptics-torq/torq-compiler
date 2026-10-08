// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// DecomposeConvsForHardware Pass
//===----------------------------------------------------------------------===//
//
// Rewrites named linalg convolutions that fall outside the Torq hardware
// limits into equivalent combinations of supported convolutions.  Limits:
//   - strides in {1, 2}
//   - dilations == 1
//   - kernel dims < 8 for regular convs and for depthwise convs with no
//     degenerate axis (unit extent and unit kernel); a 1D depthwise conv
//     may reach -torq-max-depthwise-kernel-dim (a hyperparameter, default
//     1024) on its live axis, but only with stride 1 (the dedicated 1-D
//     depthwise lowering that admits large kernels is stride-1 only)
//
// The linalg shape verifier requires every conv's output to be exactly the
// valid-conv output of its inputs (out = (in - (k - 1) * d - 1) / s + 1,
// with in >= (k - 1) * d + 1), so the slice sizes chosen below keep the
// generated sub-convs exact fits as well.
//
// Three rewrites (algorithms developed in scripts/mlir.py), applied to a
// fixed point by the greedy driver, so compositions such as dilated +
// strided + oversized kernels decompose completely:
//
// 1. rewriteDilatedConv (dilate_conv): fires when any dilation > 1.  With
//    rstride = stride/gcd(stride, dilation) and
//    rdilation = dilation/gcd(stride, dilation), the output positions split
//    into residue classes; for each offset o in {0, stride, ...,
//    dilation*rstride - stride} a stride-rstride, dilation-1 conv over the
//    o-aligned, dilation-thinned input slice is inserted into the output at
//    position o/stride (step rdilation).
//
// 2. rewriteNoPadStride (no_pad_stride): fires when a conv has a stride the
//    hardware cannot take (>= 3, or == 2 with a kernel above the stride-2
//    cap).  Per axis the filter is sampled at a thinning factor g in
//    {stride, stride/2}: for each offset r in {0, ..., g-1} a conv of the
//    r-aligned, g-thinned input with the corresponding thinned filter taps,
//    running at stride stride/g, contributes one residue class of the
//    output; the sum over offsets reproduces the strided conv.  Since
//    stride 2 is native hardware, g = stride/2 (two residue classes per
//    stride-2 conv, one elementwise add) is preferred whenever the thinned
//    kernel still fits; the axis is only split all the way to stride 1
//    (g = stride) when it must be.  If a fully split axis would still carry
//    an oversized kernel, every axis is split to stride 1 so that
//    rewriteSplitKernel (stride-1 only) can fire afterwards.  The thinned
//    input is sized to subStride * (out - 1) + filterPrime elements: the
//    full thinned slice of a valid conv is exactly that long.
//
// 3. rewriteSplitKernel (split_kernel): fires when stride == 1, dilation ==
//    1 and any kernel dim > max.  The kernel is split into max-sized chunks
//    at offsets 0, max, 2*max, ...; each chunk is a stride-1 conv of
//    input[off:] with the chunked filter, whose input is sized
//    input - kernel + chunk so every chunk (not just the final one) outputs
//    exactly the original extent; the sum reproduces the original.
//
// Supported ops (channels in NCHW/NCW positions, which the rest of the
// pipeline guarantees): linalg.conv_1d_ncw_fcw, linalg.conv_2d_nchw_fchw,
// linalg.conv_3d_ncdhw_fcdhw, linalg.depthwise_conv_1d_ncw_cw,
// linalg.depthwise_conv_2d_nchw_chw, linalg.depthwise_conv_3d_ncdhw_cdhw.
//===----------------------------------------------------------------------===//

#include "Passes.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/LayoutTransformUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <numeric>
#include <optional>

#define DEBUG_TYPE "torq-decompose-convs-for-hardware"

namespace mlir::syna::torq {

namespace {

/// The Torq hardware supports regular convs with kernel dims < 8.
constexpr int64_t kMaxRegularKernelDim = 7;

/// The Torq hardware supports depthwise convs with a much larger kernel dim;
/// this is a hyperparameter that can change.
static llvm::cl::opt<int64_t> clMaxDepthwiseKernelDim(
    "torq-max-depthwise-kernel-dim",
    llvm::cl::desc("Maximum spatial kernel dimension supported by the Torq "
                   "hardware for 1D depthwise convolutions; larger kernels "
                   "are decomposed"),
    llvm::cl::init(1024), llvm::cl::Hidden
);

/// Extract the named conv index attributes (strides/dilations, stored as
/// dense i64 tensors) to int64 values.
static SmallVector<int64_t> toI64Array(Attribute attr) {
    SmallVector<int64_t> out;
    if (auto elements = dyn_cast<DenseIntElementsAttr>(attr)) {
        for (APInt v : elements)
            out.push_back(v.getSExtValue());
    }
    else if (auto array = dyn_cast<DenseI64ArrayAttr>(attr)) {
        for (int64_t v : array.asArrayRef())
            out.push_back(v);
    }
    return out;
}

/// Convert static index values to the OpFoldResult lists expected by the
/// tensor.extract_slice / tensor.insert_slice builders.
static SmallVector<OpFoldResult> toIndexList(PatternRewriter &rewriter, ArrayRef<int64_t> values) {
    SmallVector<OpFoldResult> out;
    out.reserve(values.size());
    for (int64_t v : values)
        out.push_back(rewriter.getIndexAttr(v));
    return out;
}

/// Common view over the six supported named convolution ops.  Channels are
/// assumed to be in NCHW/NCW positions.
struct ConvInfo {
    Location loc;
    int64_t sdim;
    bool depthwise;
    Value input, filter, init;
    int64_t batch, inC, outC;
    SmallVector<int64_t> activ, kernel, out;
    SmallVector<int64_t> strides, dilations;
    Type elemType;
};

template <typename ConvOpT> struct ConvOpTrait;

template <> struct ConvOpTrait<linalg::Conv1DNcwFcwOp> {
    static constexpr int64_t sdim = 1;
    static constexpr bool depthwise = false;
};
template <> struct ConvOpTrait<linalg::Conv2DNchwFchwOp> {
    static constexpr int64_t sdim = 2;
    static constexpr bool depthwise = false;
};
template <> struct ConvOpTrait<linalg::Conv3DNcdhwFcdhwOp> {
    static constexpr int64_t sdim = 3;
    static constexpr bool depthwise = false;
};
template <> struct ConvOpTrait<linalg::DepthwiseConv1DNcwCwOp> {
    static constexpr int64_t sdim = 1;
    static constexpr bool depthwise = true;
};
template <> struct ConvOpTrait<linalg::DepthwiseConv2DNchwChwOp> {
    static constexpr int64_t sdim = 2;
    static constexpr bool depthwise = true;
};
template <> struct ConvOpTrait<linalg::DepthwiseConv3DNcdhwCdhwOp> {
    static constexpr int64_t sdim = 3;
    static constexpr bool depthwise = true;
};

template <typename ConvOpT> static std::optional<ConvInfo> getConvInfo(ConvOpT op) {
    constexpr int64_t S = ConvOpTrait<ConvOpT>::sdim;
    constexpr bool kDepthwise = ConvOpTrait<ConvOpT>::depthwise;
    Value input = op.getInputs()[0];
    Value filter = op.getInputs()[1];
    Value init = op.getOutputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto filterType = dyn_cast<RankedTensorType>(filter.getType());
    auto initType = dyn_cast<RankedTensorType>(init.getType());
    if (!inputType || !filterType || !initType || !inputType.hasStaticShape() ||
        !filterType.hasStaticShape() || !initType.hasStaticShape())
        return std::nullopt;
    if (inputType.getRank() != 2 + S || initType.getRank() != 2 + S ||
        filterType.getRank() != (kDepthwise ? 1 + S : 2 + S))
        return std::nullopt;

    SmallVector<int64_t> activ(inputType.getShape().begin() + 2, inputType.getShape().end());
    SmallVector<int64_t> out(initType.getShape().begin() + 2, initType.getShape().end());
    ArrayRef<int64_t> kernelShape =
        kDepthwise ? filterType.getShape().drop_front(1) : filterType.getShape().drop_front(2);
    SmallVector<int64_t> kernel(kernelShape.begin(), kernelShape.end());
    SmallVector<int64_t> strides = toI64Array(op.getStrides());
    SmallVector<int64_t> dilations = toI64Array(op.getDilations());
    if (strides.size() != static_cast<size_t>(S) || dilations.size() != static_cast<size_t>(S))
        return std::nullopt;
    // Location has no default constructor, so ConvInfo must be aggregate-
    // initialized rather than default-constructed.
    return ConvInfo{
        op.getLoc(),
        S,
        kDepthwise,
        input,
        filter,
        init,
        inputType.getShape()[0],
        inputType.getShape()[1],
        initType.getShape()[1],
        activ,
        kernel,
        out,
        strides,
        dilations,
        initType.getElementType()
    };
}

/// Build a named conv of type ConvOpT with the given strides/dilations.
template <typename ConvOpT>
static Value buildNamedConv(
    PatternRewriter &rewriter, const ConvInfo &info, Value input, Value filter, Value init,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations
) {
    return ConvOpT::create(
               rewriter, info.loc, init.getType(), ValueRange{input, filter}, ValueRange{init},
               rewriter.getI64TensorAttr(strides), rewriter.getI64TensorAttr(dilations)
    )
        ->getResult(0);
}

/// Elementwise tensor add: arith.addf for float element types, arith.addi for
/// integers (mirrors the Add op in scripts/mlir.py).
static Value addTensors(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs) {
    if (isa<FloatType>(cast<RankedTensorType>(lhs.getType()).getElementType()))
        return arith::AddFOp::create(rewriter, loc, lhs, rhs);
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
}

/// Iterate the cartesian product of the ranges [0, count[d]) calling fn(idx)
/// for each combination.  Returns true if fn signals a stop.
static bool forEachCombo(
    int64_t sdim, ArrayRef<int64_t> count, llvm::function_ref<bool(ArrayRef<int64_t>)> fn
) {
    SmallVector<int64_t> idx(sdim, 0);
    while (true) {
        if (fn(idx))
            return true;
        int64_t d = sdim - 1;
        while (d >= 0 && idx[d] + 1 == count[d]) {
            idx[d] = 0;
            --d;
        }
        if (d < 0)
            return false;
        ++idx[d];
    }
}

/// Per-axis kernel caps for convs that may still be further rewritten:
///  - capS1[d]: the largest kernel dim a stride-1 sub-conv may carry on axis
///    d before rewriteSplitKernel must fire.  For a 1D depthwise conv (a
///    genuine 1-D depthwise op, or a higher-rank one with a degenerate
///    spatial axis) this is the depthwise hyperparameter; elsewhere the
///    regular cap.
///  - capS2[d]: the same for a stride-2 sub-conv.  The dedicated 1-D
///    depthwise lowering that admits kernels above the regular cap is
///    stride-1 only, so a stride-2 depthwise sub-conv is bound by the
///    regular cap even on the live axis.
static void axisKernelCaps(
    const ConvInfo &info, int64_t maxKernelDim, SmallVectorImpl<int64_t> &capS1,
    SmallVectorImpl<int64_t> &capS2
) {
    int64_t S = info.sdim;
    capS1.assign(S, maxKernelDim);
    capS2.assign(S, kMaxRegularKernelDim);
    if (info.depthwise && S > 1) {
        bool is1D = false;
        for (int64_t d = 0; d < S; ++d)
            if (info.activ[d] == 1 && info.kernel[d] == 1)
                is1D = true;
        if (!is1D)
            capS1.assign(S, kMaxRegularKernelDim);
    }
}

/// True when the conv image carries backward padding (border zeros in
/// memory): a tensor.pad, or a fill written by tensor.insert_slice, fed
/// through any number of extract_slice ops.  Such convs are the EK path's
/// jurisdiction (it forges the tall-kernel border from memory); the sliced
/// sub-convs these rewrites emit would end inside the padded region, which
/// foldBackwardPadding cannot fold.  Mirrors convImageIsPrePadded in
/// Conversions/LinalgToTorqHL/Conv2DPattern.cpp.
static bool imageHasBackwardPadding(Value image) {
    Value v = image;
    while (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>())
        v = extract.getSource();
    if (v.getDefiningOp<tensor::PadOp>())
        return true;
    if (auto insert = v.getDefiningOp<tensor::InsertSliceOp>())
        return insert.getDest().getDefiningOp<linalg::FillOp>() != nullptr;
    return false;
}

//===----------------------------------------------------------------------===//
// 1. Dilated conv decomposition (dilate_conv in scripts/mlir.py)
//===----------------------------------------------------------------------===////

template <typename ConvOpT>
static LogicalResult
rewriteDilatedConv(ConvOpT op, PatternRewriter &rewriter, const ConvInfo &info) {

    constexpr int64_t S = ConvOpTrait<ConvOpT>::sdim;
    if (llvm::all_of(info.dilations, [](int64_t d) { return d == 1; }))
        return failure();
    LLVM_DEBUG(llvm::dbgs() << "[DecomposeConvs] decomposing dilated conv: " << op << "\n");

    SmallVector<int64_t> rstride(S), rdilation(S), count(S);
    for (int64_t d = 0; d < S; ++d) {
        int64_t g = std::gcd(info.strides[d], info.dilations[d]);
        rstride[d] = info.strides[d] / g;
        rdilation[d] = info.dilations[d] / g;
        // offsets = 0, stride, 2*stride, ... < dilation * rstride
        count[d] = (info.dilations[d] * rstride[d] - 1) / info.strides[d] + 1;
    }

    Value current = info.init;
    forEachCombo(S, count, [&](ArrayRef<int64_t> idx) {
        SmallVector<int64_t> inOff(S), inSize(S), inStride(S);
        SmallVector<int64_t> outOff(S), outSize(S), outStride(S);
        bool skip = false;
        for (int64_t d = 0; d < S; ++d) {
            int64_t off = idx[d] * info.strides[d];
            // Length of the dilation-thinned input starting at off (0 when
            // off is at or past the input end).
            int64_t lattice =
                off < info.activ[d] ? (info.activ[d] - off - 1) / info.dilations[d] + 1 : 0;
            if (lattice < info.kernel[d]) {
                // No original output of this residue class fits the kernel.
                skip = true;
                break;
            }
            // Kernel placements in the thinned slice; each maps to one
            // original output at off/stride + q * rdilation.
            int64_t outDim = (lattice - info.kernel[d]) / rstride[d] + 1;
            inSize[d] = (outDim - 1) * rstride[d] + info.kernel[d];
            inOff[d] = off;
            inStride[d] = info.dilations[d];
            outOff[d] = off / info.strides[d];
            outSize[d] = outDim;
            outStride[d] = rdilation[d];
        }
        if (skip)
            return false;

        SmallVector<int64_t> aOff(2 + S, 0), aSize(2 + S, 1), aStr(2 + S, 1);
        aSize[0] = info.batch;
        aSize[1] = info.inC;
        for (int64_t d = 0; d < S; ++d) {
            aOff[2 + d] = inOff[d];
            aSize[2 + d] = inSize[d];
            aStr[2 + d] = inStride[d];
        }
        Value extracted = tensor::ExtractSliceOp::create(
            rewriter, info.loc, info.input, toIndexList(rewriter, aOff),
            toIndexList(rewriter, aSize), toIndexList(rewriter, aStr)
        );

        SmallVector<int64_t> subInitShape = {info.batch, info.outC};
        for (int64_t d = 0; d < S; ++d)
            subInitShape.push_back(outSize[d]);
        Value subInit = createZeroFilledTensor(rewriter, info.loc, subInitShape, info.elemType);

        SmallVector<int64_t> ones(S, 1);
        Value subConv =
            buildNamedConv<ConvOpT>(rewriter, info, extracted, info.filter, subInit, rstride, ones);

        SmallVector<int64_t> oOff(2 + S, 0), oSize(2 + S, 1), oStr(2 + S, 1);
        oSize[0] = info.batch;
        oSize[1] = info.outC;
        for (int64_t d = 0; d < S; ++d) {
            oOff[2 + d] = outOff[d];
            oSize[2 + d] = outSize[d];
            oStr[2 + d] = outStride[d];
        }
        current = tensor::InsertSliceOp::create(
            rewriter, info.loc, subConv, current, toIndexList(rewriter, oOff),
            toIndexList(rewriter, oSize), toIndexList(rewriter, oStr)
        );
        return false;
    });

    rewriter.replaceOp(op, current);
    return success();
}

//===----------------------------------------------------------------------===//
// 2. Strided conv decomposition (no_pad_stride in scripts/mlir.py)
//===----------------------------------------------------------------------===//

template <typename ConvOpT>
static LogicalResult rewriteNoPadStride(
    ConvOpT op, PatternRewriter &rewriter, const ConvInfo &info, int64_t maxKernelDim
) {
    constexpr int64_t S = ConvOpTrait<ConvOpT>::sdim;
    if (llvm::any_of(info.dilations, [](int64_t d) { return d != 1; }))
        return failure();

    // Per-axis thinning factor (see the pass header): g = stride keeps the
    // old behaviour (stride-1 sub-convs, one per residue class); g = stride/2
    // stops at the native stride 2 and halves the sub-conv count and the
    // elementwise adds.  Prefer the stop when the thinned kernel fits the
    // stride-2 cap.
    SmallVector<int64_t> capS1, capS2;
    axisKernelCaps(info, maxKernelDim, capS1, capS2);
    SmallVector<int64_t> thin = info.strides;
    if (S != 3) { // 3D is not supported by the rest of the pipeline yet: full split
        for (int64_t d = 0; d < S; ++d)
            if (info.strides[d] % 2 == 0 &&
                (info.kernel[d] - 1) / (info.strides[d] / 2) + 1 <= capS2[d])
                thin[d] = info.strides[d] / 2;
        // 2-D depthwise: the HW depthwise path only takes the stride pairs
        // (1, 1), (2, 1) and (2, 2) (see depthwiseStridesCapable in
        // Conversions/LinalgToTorqHL/Conv2DPattern.cpp), so the W axis may
        // stop at stride 2 only when the H axis does as well.
        if (S == 2 && info.depthwise && thin[1] < info.strides[1] && thin[0] >= info.strides[0])
            thin[1] = info.strides[1];
    }
    // A sub-conv whose kernel still exceeds the stride-1 cap needs
    // rewriteSplitKernel afterwards, which only handles stride-1 convs:
    // split every axis down to stride 1 so that it can fire.
    for (int64_t d = 0; d < S; ++d)
        if ((info.kernel[d] - 1) / thin[d] + 1 > capS1[d]) {
            thin = info.strides;
            break;
        }
    if (llvm::all_of(thin, [](int64_t t) { return t == 1; }))
        return failure(); // stride in {1, 2} and the kernels fit

    LLVM_DEBUG(llvm::dbgs() << "[DecomposeConvs] decomposing strided conv: " << op << "\n");

    SmallVector<int64_t> subStride(S);
    for (int64_t d = 0; d < S; ++d)
        subStride[d] = info.strides[d] / thin[d];

    SmallVector<int64_t> count(S);
    for (int64_t d = 0; d < S; ++d)
        count[d] = thin[d]; // offsets 0 .. thin-1

    Value current = info.init;
    forEachCombo(S, count, [&](ArrayRef<int64_t> idx) {
        SmallVector<int64_t> aOff(2 + S, 0), aSize(2 + S, 1), aStr(2 + S, 1);
        aSize[0] = info.batch;
        aSize[1] = info.inC;
        SmallVector<int64_t> wOff, wSize, wStr;
        if (info.depthwise) {
            wOff = {0};
            wSize = {info.inC};
            wStr = {1};
        }
        else {
            wOff = {0, 0};
            wSize = {info.outC, info.inC};
            wStr = {1, 1};
        }
        bool skip = false;
        for (int64_t d = 0; d < S; ++d) {
            int64_t off = idx[d];
            if (off >= info.kernel[d]) {
                skip = true;
                break;
            }
            // Thinned filter taps and the exact-fit thinned input size for
            // them (see the pass header).
            int64_t filterPrime = (info.kernel[d] - off - 1) / thin[d] + 1;
            int64_t activPrime = subStride[d] * (info.out[d] - 1) + filterPrime;
            aOff[2 + d] = off;
            aSize[2 + d] = activPrime;
            aStr[2 + d] = thin[d];
            wOff.push_back(off);
            wSize.push_back(filterPrime);
            wStr.push_back(thin[d]);
        }
        if (skip)
            return false;

        Value extractedInput = tensor::ExtractSliceOp::create(
            rewriter, info.loc, info.input, toIndexList(rewriter, aOff),
            toIndexList(rewriter, aSize), toIndexList(rewriter, aStr)
        );
        Value extractedFilter = tensor::ExtractSliceOp::create(
            rewriter, info.loc, info.filter, toIndexList(rewriter, wOff),
            toIndexList(rewriter, wSize), toIndexList(rewriter, wStr)
        );

        SmallVector<int64_t> subInitShape = {info.batch, info.outC};
        for (int64_t d = 0; d < S; ++d)
            subInitShape.push_back(info.out[d]);
        Value subInit = createZeroFilledTensor(rewriter, info.loc, subInitShape, info.elemType);

        SmallVector<int64_t> ones(S, 1);
        Value subConv = buildNamedConv<ConvOpT>(
            rewriter, info, extractedInput, extractedFilter, subInit, subStride, ones
        );
        current = addTensors(rewriter, info.loc, current, subConv);
        return false;
    });

    rewriter.replaceOp(op, current);
    return success();
}

//===----------------------------------------------------------------------===//
// 3. Large-kernel decomposition (split_kernel in scripts/mlir.py)
//===----------------------------------------------------------------------===//

template <typename ConvOpT>
static LogicalResult rewriteSplitKernel(
    ConvOpT op, PatternRewriter &rewriter, const ConvInfo &info, int64_t maxKernelDim
) {
    constexpr int64_t S = ConvOpTrait<ConvOpT>::sdim;
    if (llvm::any_of(info.dilations, [](int64_t d) { return d != 1; }))
        return failure();
    if (llvm::any_of(info.strides, [](int64_t s) { return s != 1; }))
        return failure();
    // Per-axis kernel caps (see the pass header): a depthwise conv with a
    // degenerate spatial axis (unit extent and unit kernel) is a 1D conv and
    // its live axis may use the depthwise cap; every other case is bound by
    // the regular cap.  The chunks are stride-1 convs, so the stride-1 caps
    // apply.
    SmallVector<int64_t> axisCap, capS2;
    axisKernelCaps(info, maxKernelDim, axisCap, capS2);
    bool withinLimits = true;
    for (int64_t d = 0; d < S; ++d)
        withinLimits = withinLimits && info.kernel[d] <= axisCap[d];
    if (withinLimits)
        return failure();
    LLVM_DEBUG(llvm::dbgs() << "[DecomposeConvs] decomposing large-kernel conv: " << op << "\n");

    SmallVector<int64_t> count(S);
    for (int64_t d = 0; d < S; ++d)
        count[d] = (info.kernel[d] - 1) / axisCap[d] + 1; // offsets 0, cap, 2*cap, ...

    SmallVector<int64_t> ones(S, 1);
    Value current = info.init;
    forEachCombo(S, count, [&](ArrayRef<int64_t> idx) {
        SmallVector<int64_t> aOff(2 + S, 0), aSize(2 + S, 1), aStr(2 + S, 1);
        aSize[0] = info.batch;
        aSize[1] = info.inC;
        SmallVector<int64_t> wOff, wSize, wStr;
        if (info.depthwise) {
            wOff = {0};
            wSize = {info.inC};
            wStr = {1};
        }
        else {
            wOff = {0, 0};
            wSize = {info.outC, info.inC};
            wStr = {1, 1};
        }
        for (int64_t d = 0; d < S; ++d) {
            int64_t off = idx[d] * axisCap[d];
            int64_t chunk = std::min(info.kernel[d] - off, axisCap[d]);
            // Exact-fit input size for this chunk (see the pass header).
            aOff[2 + d] = off;
            aSize[2 + d] = info.activ[d] - info.kernel[d] + chunk;
            wOff.push_back(off);
            wSize.push_back(chunk);
            wStr.push_back(1);
        }

        Value extractedInput = tensor::ExtractSliceOp::create(
            rewriter, info.loc, info.input, toIndexList(rewriter, aOff),
            toIndexList(rewriter, aSize), toIndexList(rewriter, aStr)
        );
        Value extractedFilter = tensor::ExtractSliceOp::create(
            rewriter, info.loc, info.filter, toIndexList(rewriter, wOff),
            toIndexList(rewriter, wSize), toIndexList(rewriter, wStr)
        );

        SmallVector<int64_t> subInitShape = {info.batch, info.outC};
        for (int64_t d = 0; d < S; ++d)
            subInitShape.push_back(info.out[d]);
        Value subInit = createZeroFilledTensor(rewriter, info.loc, subInitShape, info.elemType);

        Value subConv = buildNamedConv<ConvOpT>(
            rewriter, info, extractedInput, extractedFilter, subInit, ones, ones
        );
        current = addTensors(rewriter, info.loc, current, subConv);
        return false;
    });

    rewriter.replaceOp(op, current);
    return success();
}

//===----------------------------------------------------------------------===//
// Pattern and pass
//===----------------------------------------------------------------------===//

// True when the rewrites may decompose a conv whose image carries backward
// padding ("same"-style borders in memory).  The sub-convs they emit take
// strided input slices that end inside the padded region, which
// foldBackwardPadding cannot fold into a kernel halo - so only halo-free
// (kernel-1 on every axis) sub-convs are safe there.  Mirrors
// rewriteNoPadStride's thinning decision; everything else (dilated convs,
// stride-1 axes with real kernels) needs a halo and stays with the EK path.
template <typename ConvOpT>
static bool paddedImageDecompositionSafe(const ConvInfo &info, int64_t maxKernelDim) {
    constexpr int64_t S = ConvOpTrait<ConvOpT>::sdim;
    if (llvm::any_of(info.dilations, [](int64_t d) { return d != 1; }))
        return false;
    SmallVector<int64_t> capS1, capS2;
    axisKernelCaps(info, maxKernelDim, capS1, capS2);
    SmallVector<int64_t> thin = info.strides;
    if (S != 3) {
        for (int64_t d = 0; d < S; ++d)
            if (info.strides[d] % 2 == 0 &&
                (info.kernel[d] - 1) / (info.strides[d] / 2) + 1 <= capS2[d])
                thin[d] = info.strides[d] / 2;
        if (S == 2 && info.depthwise && thin[1] < info.strides[1] && thin[0] >= info.strides[0])
            thin[1] = info.strides[1];
    }
    for (int64_t d = 0; d < S; ++d)
        if ((info.kernel[d] - 1) / thin[d] + 1 > capS1[d]) {
            thin = info.strides;
            break;
        }
    for (int64_t d = 0; d < S; ++d) {
        if (info.strides[d] == 1) {
            // No stride rewrite on this axis: the sub-convs keep the full
            // kernel, which needs a halo.
            if (info.kernel[d] != 1)
                return false;
        }
        else if (info.kernel[d] > thin[d]) {
            // Thinned sub-kernel is still > 1: halo needed.
            return false;
        }
    }
    return true;
}

template <typename ConvOpT>
class DecomposeConvForHardwarePattern : public OpRewritePattern<ConvOpT> {
  public:
    DecomposeConvForHardwarePattern(MLIRContext *context, int64_t maxKernelDim)
        : OpRewritePattern<ConvOpT>(context, /*benefit=*/1), maxKernelDim(maxKernelDim) {}

    LogicalResult matchAndRewrite(ConvOpT op, PatternRewriter &rewriter) const override {
        auto infoOr = getConvInfo<ConvOpT>(op);
        if (!infoOr)
            return rewriter.notifyMatchFailure(op, "conv has unsupported (dynamic) shape");
        const ConvInfo &info = *infoOr;
        // Convs whose image carries backward padding ("same"-style borders in
        // memory) are handled by the EK path; the sliced sub-convs these
        // rewrites emit end inside the padded region, which
        // foldBackwardPadding cannot fold.  The exception is a decomposition
        // whose sub-convs are kernel-1 on every axis (halo-free): their strided
        // slices may end in the pad, the pad zeros are simply read as data.
        // That is what makes e.g. a stride-3 kernel-3 padded conv1d
        // decomposable into three stride-1 kernel-1 sub-convs.
        if (imageHasBackwardPadding(info.input) &&
            !paddedImageDecompositionSafe<ConvOpT>(info, maxKernelDim))
            return rewriter.notifyMatchFailure(op, "image has backward padding (EK path)");
        if (succeeded(rewriteDilatedConv<ConvOpT>(op, rewriter, info)))
            return success();
        if (succeeded(rewriteNoPadStride<ConvOpT>(op, rewriter, info, maxKernelDim)))
            return success();
        if (succeeded(rewriteSplitKernel<ConvOpT>(op, rewriter, info, maxKernelDim)))
            return success();
        return rewriter.notifyMatchFailure(op, "conv is within hardware limits");
    }

  private:
    int64_t maxKernelDim;
};

// A conv1d that outputs a single point is a GEMM, not a convolution:
//   out[n, o] = sum_c sum_l W[o, c, l] * x[n, c, l]
// (dilation 1, the window covers the whole input; the stride is irrelevant
// with one output position, which selects tap 0).  Lowering it to a
// linalg.matmul over [N, C*kL] x [C*kL, O] lets the fully-connected kernel
// take it, far cheaper than the S2D-inflated (C*b channels) or residue-split
// conv forms on these degenerate geometries.  Restriction: no materialized
// pad (input spatial size == window size); a SAME conv with a single output
// point has an unknown forward-pad offset at this level and stays on the
// decomposition path.
struct SinglePointConv1DtoMatmulPattern : public OpRewritePattern<linalg::Conv1DNcwFcwOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult
    matchAndRewrite(linalg::Conv1DNcwFcwOp op, PatternRewriter &rewriter) const override {
        auto infoOr = getConvInfo<linalg::Conv1DNcwFcwOp>(op);
        if (!infoOr)
            return rewriter.notifyMatchFailure(op, "not a static conv1d");
        const ConvInfo &info = *infoOr;
        if (op.getNumDpsInputs() != 2)
            return rewriter.notifyMatchFailure(op, "biased conv1d stays on the conv path");
        if (info.out[0] != 1)
            return rewriter.notifyMatchFailure(op, "output has more than one point");
        if (info.dilations[0] != 1 || info.activ[0] != info.kernel[0])
            return rewriter.notifyMatchFailure(
                op, "window does not cover the whole input (pad or dilation)"
            );
        if (!isAllZerosTensor(info.init))
            return rewriter.notifyMatchFailure(op, "init is not all zeros");

        int64_t N = info.batch, C = info.inC, O = info.outC, kL = info.kernel[0];
        auto filterTy = cast<RankedTensorType>(info.filter.getType());
        auto outTy = cast<RankedTensorType>(info.init.getType());

        // [N, C, kL] -> [N, C*kL]
        Value flatIn =
            tensor::CollapseShapeOp::create(
                rewriter, info.loc, info.input, SmallVector<ReassociationIndices>{{0}, {1, 2}}
            ).getResult();

        // [O, C, kL] -> [C, kL, O] -> [C*kL, O]; canonicalization folds the
        // transpose into the weight constant when the filter is a constant.
        auto transTy = RankedTensorType::get({C, kL, O}, filterTy.getElementType());
        Value transInit =
            rewriter.create<tensor::EmptyOp>(info.loc, transTy, ValueRange{}).getResult();
        Value trans = linalg::TransposeOp::create(
                          rewriter, info.loc, info.filter, transInit, ArrayRef<int64_t>{1, 2, 0}
        )
                          .getResult()[0];
        Value flatW = tensor::CollapseShapeOp::create(
                          rewriter, info.loc, trans, SmallVector<ReassociationIndices>{{0, 1}, {2}}
        ).getResult();

        // matmul [N, C*kL] x [C*kL, O] -> [N, O], accumulating in the conv's
        // output element type (the pre-decomposition conv1d->matmul unfold
        // had the same form; a zero fill init keeps the matmul's accumulating
        // semantics explicit for the downstream patterns).
        auto mmTy = RankedTensorType::get({N, O}, outTy.getElementType());
        Value mmInitEmpty =
            rewriter.create<tensor::EmptyOp>(info.loc, mmTy, ValueRange{}).getResult();
        Value mmInit =
            linalg::FillOp::create(
                rewriter, info.loc, createZeroConstant(rewriter, info.loc, mmTy.getElementType()),
                mmInitEmpty
            )
                .getResult(0);
        Value mm =
            linalg::MatmulOp::create(
                rewriter, info.loc, TypeRange{mmTy}, ValueRange{flatIn, flatW}, ValueRange{mmInit}
            )
                .getResult(0);

        Value result =
            tensor::ExpandShapeOp::create(
                rewriter, info.loc, outTy, mm, SmallVector<ReassociationIndices>{{0}, {1, 2}}
            ).getResult();

        rewriter.replaceOp(op, result);
        return success();
    }
};

class DecomposeConvsForHardwarePass
    : public PassWrapper<DecomposeConvsForHardwarePass, OperationPass<func::FuncOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DecomposeConvsForHardwarePass)

    DecomposeConvsForHardwarePass() = default;
    DecomposeConvsForHardwarePass(const DecomposeConvsForHardwarePass &other) = default;

    StringRef getArgument() const override { return "torq-decompose-convs-for-hardware"; }

    StringRef getDescription() const override {
        return "Decompose named linalg convolutions outside Torq hardware limits "
               "(stride in {1, 2}, dilation 1, kernel dim < 8 for regular convs, "
               "kernel dim <= the depthwise limit) into sums of supported "
               "convolutions";
    }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    }

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext *context = funcOp.getContext();
        RewritePatternSet patterns(context);
        constexpr int64_t kMax = kMaxRegularKernelDim;
        // The single-point matmul form wins over the decomposition for the
        // convs it matches; the decomposition then sees nothing to do.
        patterns.add<SinglePointConv1DtoMatmulPattern>(context, /*benefit=*/2);
        patterns.add<DecomposeConvForHardwarePattern<linalg::Conv1DNcwFcwOp>>(context, kMax);
        patterns.add<DecomposeConvForHardwarePattern<linalg::Conv2DNchwFchwOp>>(context, kMax);
        patterns.add<DecomposeConvForHardwarePattern<linalg::Conv3DNcdhwFcdhwOp>>(context, kMax);
        patterns.add<DecomposeConvForHardwarePattern<linalg::DepthwiseConv1DNcwCwOp>>(
            context, clMaxDepthwiseKernelDim
        );
        patterns.add<DecomposeConvForHardwarePattern<linalg::DepthwiseConv2DNchwChwOp>>(
            context, clMaxDepthwiseKernelDim
        );
        patterns.add<DecomposeConvForHardwarePattern<linalg::DepthwiseConv3DNcdhwCdhwOp>>(
            context, clMaxDepthwiseKernelDim
        );
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

std::unique_ptr<Pass> createDecomposeConvsForHardwarePass() {
    return std::make_unique<DecomposeConvsForHardwarePass>();
}

} // namespace mlir::syna::torq
