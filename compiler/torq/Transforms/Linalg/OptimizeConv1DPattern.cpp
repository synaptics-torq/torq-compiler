// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"
#include "torq/Utils/HwInfo.h"
#include "torq/Utils/TorqUtils.h"

#include "algorithm"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-conv1d-pattern"

namespace mlir::syna::torq {

// Elementwise f32 -> bf16 truncf as a parallel linalg.generic; rank is taken from `value`.
static Value emitTruncfToBF16(PatternRewriter &rewriter, Location loc, Value value) {
    auto srcType = cast<RankedTensorType>(value.getType());
    int64_t rank = srcType.getRank();
    auto bf16Type = rewriter.getBF16Type();
    auto resultType = RankedTensorType::get(srcType.getShape(), bf16Type);
    Value init = tensor::EmptyOp::create(rewriter, loc, srcType.getShape(), bf16Type).getResult();
    auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(
        rewriter, loc, TypeRange{resultType}, ValueRange{value}, ValueRange{init},
        ArrayRef<AffineMap>{identityMap, identityMap}, iterators,
        [&](OpBuilder &b, Location l, ValueRange args) {
            auto truncf = arith::TruncFOp::create(b, l, bf16Type, args[0]);
            linalg::YieldOp::create(b, l, truncf.getResult());
        }
    );
    return generic.getResult(0);
}

// True when `op` is an elementwise f32 -> bf16 truncf of its only input and does
// nothing else (a fused clamp + truncf, for example, is not).
static bool isPlainTruncfF32ToBF16(linalg::GenericOp op) {
    if (op.getNumDpsInputs() != 1 || op.getNumResults() != 1 ||
        op.getNumParallelLoops() != op.getNumLoops() ||
        !llvm::all_of(op.getIndexingMapsArray(), [](AffineMap m) { return m.isIdentity(); })) {
        return false;
    }
    Block &body = op.getRegion().front();
    if (body.getOperations().size() != 2) {
        return false;
    }
    auto truncf = dyn_cast<arith::TruncFOp>(&body.front());
    return truncf && truncf.getIn() == body.getArgument(0) && truncf.getType().isBF16() &&
           truncf.getIn().getType().isF32() &&
           body.getTerminator()->getOperand(0) == truncf.getResult();
}

/// Reassociation for a tensor::expand_shape that appends a unit extent
/// after the last dimension of a rank-`rank` tensor: {d0}, {d1}, ...,
/// {d_{rank-2}}, {d_{rank-1}, d_rank}.
static SmallVector<ReassociationIndices> appendUnitDimReassociation(int64_t rank) {
    SmallVector<ReassociationIndices> reassoc;
    reassoc.reserve(rank);
    for (int64_t i = 0; i + 1 < rank; ++i)
        reassoc.push_back({i});
    reassoc.push_back({rank - 1, rank});
    return reassoc;
}

/// The same tensor type with a unit extent appended after the last dimension.
static RankedTensorType appendUnitDimType(RankedTensorType type) {
    SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
    shape.push_back(1);
    return RankedTensorType::get(shape, type.getElementType());
}

/// Maps a 1D convolution op to the 2D op it promotes to (the added spatial
/// extent is set to 1) and to the 1D filter rank ([F,C,W] for the regular
/// conv, [C,W] for the depthwise one).

template <typename Conv1DOp> struct Conv1DPromotionTrait;

template <> struct Conv1DPromotionTrait<linalg::Conv1DNcwFcwOp> {
    using Conv2DOp = linalg::Conv2DNchwFchwOp;
    static constexpr int64_t filterRank = 3;
    static constexpr bool isDepthwise = false;
};
template <> struct Conv1DPromotionTrait<linalg::DepthwiseConv1DNcwCwOp> {
    using Conv2DOp = linalg::DepthwiseConv2DNchwChwOp;
    static constexpr int64_t filterRank = 2;
    static constexpr bool isDepthwise = true;
};

static llvm::cl::opt<int64_t> clConv1dMatmulMaxPoints(
    "torq-conv1d-matmul-max-points",
    llvm::cl::desc("Largest number of output points of a stride > 2 conv1d that is lowered "
                   "to a matmul over its gathered input windows"),
    llvm::cl::init(4), llvm::cl::Hidden
);

/// Shared claim predicate: a regular conv1d with dilation 1, an all-zero init and no bias
/// input that is a GEMM over its input windows, which DecomposeConvsForHardwarePass lowers
/// to linalg.matmul (Conv1DtoMatmulPattern):
/// - a single output point whose window covers the whole input;
/// - or, for a stride above 2 (which the conv2d form cannot take) that does not divide the
///   kernel, batch 1 and at most --torq-conv1d-matmul-max-points output points.
/// When the stride divides the kernel, the S2D fold needs no zero-padded kernel taps and is
/// cheaper; otherwise the matmul is cheaper up to about 4 points (compile-time estimate of
/// bf16 sweeps with C 1..128, K 5..127, stride 3..64, both sim chips).
bool isConv1dMatmulCandidate(linalg::Conv1DNcwFcwOp op) {
    if (op.getNumDpsInputs() != 2)
        return false; // a bias input keeps the conv on the conv path
    auto inTy = dyn_cast<RankedTensorType>(op.getInputs()[0].getType());
    auto fltTy = dyn_cast<RankedTensorType>(op.getInputs()[1].getType());
    auto outTy = dyn_cast<RankedTensorType>(op.getOutputs()[0].getType());
    if (!inTy || !fltTy || !outTy || !inTy.hasStaticShape() || !fltTy.hasStaticShape() ||
        !outTy.hasStaticShape())
        return false; // not a static conv1d
    if (op.getDilations().getValues<int64_t>()[0] != 1)
        return false;
    int64_t points = outTy.getShape()[2];
    if (points == 1) {
        if (inTy.getShape()[2] != fltTy.getShape()[2])
            return false; // window does not cover the whole input (pad or dilation)
    }
    else {
        int64_t stride = op.getStrides().getValues<int64_t>()[0];
        if (stride <= 2 || fltTy.getShape()[2] % stride == 0 || outTy.getShape()[0] != 1 ||
            points > clConv1dMatmulMaxPoints)
            return false;
    }
    return isAllZerosTensor(op.getOutputs()[0]);
}

/// Promotes a 1D convolution to its 2D counterpart with the added spatial
/// extent set to 1.  One shared body for the regular and depthwise 1D
/// convs, parameterized like ConvOpTrait in DecomposeConvsForHardwarePass.cpp.
///
/// The 2D op then lowers through the established conv2d path, where a
/// depthwise conv with a unit extent takes the 1D-depthwise lowering and may
/// carry large kernels on the live axis.
template <typename Conv1DOp>
struct Conv1DtoConv2DPromotionPattern : public OpRewritePattern<Conv1DOp> {
    using Conv2DOp = typename Conv1DPromotionTrait<Conv1DOp>::Conv2DOp;

    // Strides outside [minStride, maxStride] are not promoted.  The defaults
    // keep the HW-legal promotion range [1, 2]; the S2D pass registers
    // (minStride = 3, maxStride = large) so that only out-of-range conv1ds are
    // promoted early, for its stride-into-channels fold — everything below
    // keeps its existing promotion path in OptimizeLinalgForTorqPass.
    const int64_t minStride;
    const int64_t maxStride;

    Conv1DtoConv2DPromotionPattern(
        MLIRContext *context, int64_t minStride = 1, int64_t maxStride = 2
    )
        : OpRewritePattern<Conv1DOp>(context), minStride(minStride), maxStride(maxStride) {}

    LogicalResult matchAndRewrite(Conv1DOp convOp, PatternRewriter &rewriter) const override {
        auto loc = convOp.getLoc();

        Value input = convOp.getInputs()[0];
        Value filter = convOp.getInputs()[1];
        Value output = convOp.getOutputs()[0];

        auto inputType = cast<RankedTensorType>(input.getType());
        auto filterType = cast<RankedTensorType>(filter.getType());
        auto outputType = cast<RankedTensorType>(output.getType());
        auto strides = convOp.getStrides().template getValues<int64_t>();
        auto dilations = convOp.getDilations().template getValues<int64_t>();

        if (inputType.getRank() != 3 || outputType.getRank() != 3 ||
            filterType.getRank() != Conv1DPromotionTrait<Conv1DOp>::filterRank) {
            return rewriter.notifyMatchFailure(convOp, "unexpected rank");
        }
        if (strides[0] < minStride || strides[0] > maxStride || dilations[0] > 1) {
            return rewriter.notifyMatchFailure(
                convOp, "stride outside promotion range or dilation > 1"
            );
        }

        // A conv1d with one or a few output points is a GEMM over its input
        // windows; DecomposeConvsForHardwarePass lowers it to a matmul for the
        // fully-connected kernel, which beats both the S2D stride-into-channels
        // fold (C-inflated conv) and the residue split.  Bail exactly where
        // that pattern claims (isConv1dMatmulCandidate): a conv it does not
        // claim (a bias folded into the init, say) must still be promoted, or
        // nothing lowers it.
        if (!Conv1DPromotionTrait<Conv1DOp>::isDepthwise &&
            isConv1dMatmulCandidate(cast<linalg::Conv1DNcwFcwOp>(convOp))) {
            return rewriter.notifyMatchFailure(convOp, "conv1d is matmul territory");
        }

        // Append the unit extent: [N,C,W] -> [N,C,W,1], filter likewise.
        auto expandedInput = tensor::ExpandShapeOp::create(
                                 rewriter, loc, appendUnitDimType(inputType), input,
                                 appendUnitDimReassociation(inputType.getRank())
        )
                                 .getResult();
        auto expandedFilter = tensor::ExpandShapeOp::create(
                                  rewriter, loc, appendUnitDimType(filterType), filter,
                                  appendUnitDimReassociation(filterType.getRank())
        )
                                  .getResult();
        auto expandedOutput = tensor::ExpandShapeOp::create(
                                  rewriter, loc, appendUnitDimType(outputType), output,
                                  appendUnitDimReassociation(outputType.getRank())
        )
                                  .getResult();

        // 1D strides/dilations to 2D (unit on the added dimension).
        SmallVector<int64_t> strides2d = {strides[0], 1};
        SmallVector<int64_t> dilations2d = {dilations[0], 1};
        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto stridesAttr2d = DenseIntElementsAttr::get(attrType, strides2d);
        auto dilationsAttr2d = DenseIntElementsAttr::get(attrType, dilations2d);

        auto conv2d = Conv2DOp::create(
            rewriter, loc, expandedOutput.getType(), ValueRange{expandedInput, expandedFilter},
            ValueRange{expandedOutput}, stridesAttr2d, dilationsAttr2d
        );

        // Re-emit an f32->bf16 truncf that immediately consumed the 1D conv
        // result; the original generic is dropped below once the collapse is
        // wired up.
        auto conv2dResult = conv2d.getResult(0);
        auto currentResult = conv2dResult;

        linalg::GenericOp truncfGenericOp = nullptr;
        auto conv1dResult = convOp.getResult(0);
        if (conv1dResult.hasOneUse()) {
            auto userOp = conv1dResult.use_begin()->getOwner();
            auto genericOp = llvm::dyn_cast<linalg::GenericOp>(userOp);
            // Only a plain truncf is re-emitted; any other user (e.g. a clamp fused
            // with the truncf) keeps reading the collapsed conv2d result.
            if (genericOp && isPlainTruncfF32ToBF16(genericOp)) {
                currentResult = emitTruncfToBF16(rewriter, loc, conv2dResult);
                truncfGenericOp = genericOp;
            }
        }

        // Collapse the unit extent back off: [N,F,Ow,1] -> [N,F,Ow].
        auto collapsedResultType = RankedTensorType::get(
            SmallVector<int64_t>(outputType.getShape().begin(), outputType.getShape().end()),
            cast<RankedTensorType>(currentResult.getType()).getElementType()
        );
        auto collapsedResult =
            tensor::CollapseShapeOp::create(
                rewriter, loc, collapsedResultType, currentResult, appendUnitDimReassociation(3)
            )
                .getResult();

        rewriter.replaceOp(convOp, collapsedResult);

        if (truncfGenericOp) {
            truncfGenericOp->replaceAllUsesWith(ValueRange{collapsedResult});
            rewriter.eraseOp(truncfGenericOp);
        }

        return success();
    }
};

/// Re-encodes a W-live 1D depthwise conv (unit image H extent and unit kH) into
/// the H-live orientation ([N,C,1,W] -> [N,C,W,1], weight [C,1,KW] -> [C,KW,1])
/// when the dedicated 1D depthwise lowering cannot handle its channels. That
/// lowering packs the weight channels into fixed blocks of 16 (bf16) or 32
/// (other) and requires the channel count to be a multiple of the block size,
/// while the tall-kernel EK path taken by the H-live conv has no such
/// requirement. The swap is exact: a 1D depthwise conv is unchanged under a
/// spatial axis swap (strides and dilations swap with it).
struct DepthwiseConv2DWLiveToHLivePattern
    : public OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp> {
    using OpRewritePattern<linalg::DepthwiseConv2DNchwChwOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(
        linalg::DepthwiseConv2DNchwChwOp convOp, PatternRewriter &rewriter
    ) const override {
        auto loc = convOp.getLoc();

        auto imageType = cast<RankedTensorType>(convOp.getInputs()[0].getType());
        auto weightType = cast<RankedTensorType>(convOp.getInputs()[1].getType());
        auto outputType = cast<RankedTensorType>(convOp.getOutputs()[0].getType());
        auto strides = convOp.getStrides().template getValues<int64_t>();
        auto dilations = convOp.getDilations().template getValues<int64_t>();

        if (imageType.getRank() != 4 || weightType.getRank() != 3 || outputType.getRank() != 4)
            return rewriter.notifyMatchFailure(convOp, "unexpected rank");
        // Only re-orient the degenerate W-live encoding [N,C,1,W] (weight
        // [C,1,KW]), and only when the kernel exceeds the 2D cap: smaller
        // kernels lower fine in place.
        if (imageType.getShape()[2] != 1 || weightType.getShape()[1] != 1 ||
            weightType.getShape()[2] <= HwInfo::nss_max_kernel_width)
            return rewriter.notifyMatchFailure(convOp, "not a wide W-live conv");
        // The dedicated 1D lowering (and the H-live path) expect stride-1,
        // dilation-1 convs; the decomposition pass guarantees that for the
        // chunks this pattern sees.
        if (strides[0] != 1 || strides[1] != 1 || dilations[0] != 1 || dilations[1] != 1)
            return rewriter.notifyMatchFailure(convOp, "stride/dilation not 1");
        // The dedicated 1D lowering is selected for bf16 images and fails
        // when the channels do not fit its fixed packing blocks; that is the
        // exact case this rewrite rescues.
        if (!imageType.getElementType().isBF16())
            return rewriter.notifyMatchFailure(convOp, "image not bf16");
        const int64_t ch = imageType.getShape()[1];
        const int64_t inner = weightType.getElementType().isBF16() ? 16 : 32;
        if (ch % inner == 0)
            return rewriter.notifyMatchFailure(convOp, "channels fit 1D packing");

        SmallVector<int64_t> dataPerm = {0, 1, 3, 2};
        SmallVector<int64_t> weightPerm = {0, 2, 1};

        auto permuteType = [](RankedTensorType type, ArrayRef<int64_t> perm) {
            SmallVector<int64_t> shape(perm.size());
            for (size_t i = 0; i < perm.size(); ++i)
                shape[i] = type.getShape()[perm[i]];
            return RankedTensorType::get(shape, type.getElementType());
        };
        auto transpose = [&](Value value, ArrayRef<int64_t> perm) -> Value {
            auto type = cast<RankedTensorType>(value.getType());
            auto transposedType = permuteType(type, perm);
            auto init =
                tensor::EmptyOp::create(
                    rewriter, loc, transposedType.getShape(), transposedType.getElementType()
                )
                    .getResult();
            return linalg::TransposeOp::create(rewriter, loc, value, init, perm).getResult()[0];
        };

        SmallVector<int64_t> strides2d = {strides[1], strides[0]};
        SmallVector<int64_t> dilations2d = {dilations[1], dilations[0]};
        auto attrType = RankedTensorType::get({2}, rewriter.getIntegerType(64));
        auto conv2d = linalg::DepthwiseConv2DNchwChwOp::create(
            rewriter, loc, permuteType(outputType, dataPerm),
            ValueRange{
                transpose(convOp.getInputs()[0], dataPerm),
                transpose(convOp.getInputs()[1], weightPerm)
            },
            ValueRange{transpose(convOp.getOutputs()[0], dataPerm)},
            DenseIntElementsAttr::get(attrType, strides2d),
            DenseIntElementsAttr::get(attrType, dilations2d)
        );

        rewriter.replaceOp(convOp, transpose(conv2d.getResult(0), dataPerm));
        return success();
    }
};

/// Sink a static tensor.extract_slice past an elementwise linalg.generic whose every
/// input is an identically-parameterised extract_slice, i.e. rewrite
///   elementwise(slice(a), slice(b), ...) -> slice(elementwise(a, b, ...)).
///
/// A strided ConvTranspose lowers to a conv whose output is a few columns wider than the real
/// result; the epilogue is `conv -> extract_slice(Ow+k -> Ow) -> addf(per-channel bias) -> truncf`.
/// The crop drops the extra ConvTranspose columns and may also absorb a downstream slice the
/// compiler folded in here, so it can shrink the width by more than one element. That slice
/// between the conv result and the bias add stops the bias/truncf from fusing into the conv
/// epilogue (matmul or native conv1d), leaving an fp32 add with no NPU lowering that is illegal
/// under --torq-disable-host. Slicing commutes with positional elementwise ops, so moving it
/// below the add/truncf makes the conv->add->truncf chain contiguous again for the downstream
/// epilogue fusion.
struct SinkExtractSliceThroughElementwisePattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    struct SliceInputsInfo {
        tensor::ExtractSliceOp firstSlice;
        RankedTensorType srcType;
        RankedTensorType resultType;
        SmallVector<int64_t> staticSizes;
        SmallVector<Value> sources;
    };

    LogicalResult
    checkElementwiseSignature(linalg::GenericOp genericOp, PatternRewriter &rewriter) const {
        // Target the ConvTranspose bias epilogue elementwise(slice(a), slice(b), ...): a binary
        // (or wider) op whose operands are all the same crop. A single-input crop is just a
        // rescale/cast/copy on a sliced tensor (very common in conv backbones, e.g. per-channel
        // requantize after a channel split); sinking those does nothing for bias fusion and only
        // risks reshaping unrelated epilogues, so require at least two inputs.
        if (genericOp.getNumResults() != 1 || genericOp.getNumDpsInputs() < 2) {
            return rewriter.notifyMatchFailure(genericOp, "need single result and >=2 inputs");
        }

        // Elementwise: every iterator parallel and every indexing map an identity.
        if (!llvm::all_of(genericOp.getIteratorTypesArray(), [](utils::IteratorType t) {
                return t == utils::IteratorType::parallel;
            })) {
            return rewriter.notifyMatchFailure(genericOp, "not all-parallel");
        }
        for (AffineMap m : genericOp.getIndexingMapsArray()) {
            if (!m.isIdentity()) {
                return rewriter.notifyMatchFailure(genericOp, "non-identity indexing map");
            }
        }
        return success();
    }

    LogicalResult collectSliceInputs(
        linalg::GenericOp genericOp, PatternRewriter &rewriter, SliceInputsInfo &sliceInfo
    ) const {
        // Every input must be an extract_slice with identical, fully-static slice parameters
        // taken from an equally-shaped source: slicing the same region from each operand commutes
        // with the positional elementwise body. Collect the slice sources as we validate.
        auto firstSlice = genericOp.getDpsInputs().front().getDefiningOp<tensor::ExtractSliceOp>();
        if (!firstSlice) {
            return rewriter.notifyMatchFailure(genericOp, "inputs are not extract_slice");
        }
        auto srcType = cast<RankedTensorType>(firstSlice.getSource().getType());
        if (!srcType.hasStaticShape() ||
            llvm::any_of(firstSlice.getStaticOffsets(), ShapedType::isDynamic) ||
            llvm::any_of(firstSlice.getStaticSizes(), ShapedType::isDynamic) ||
            llvm::any_of(firstSlice.getStaticStrides(), ShapedType::isDynamic)) {
            return rewriter.notifyMatchFailure(genericOp, "dynamic slice");
        }

        SmallVector<Value> sources;
        sources.reserve(genericOp.getNumDpsInputs());
        for (Value in : genericOp.getDpsInputs()) {
            auto s = in.getDefiningOp<tensor::ExtractSliceOp>();
            if (!s) {
                return rewriter.notifyMatchFailure(genericOp, "input is not extract_slice");
            }
            if (s.getSource().getType() != srcType ||
                s.getStaticOffsets() != firstSlice.getStaticOffsets() ||
                s.getStaticSizes() != firstSlice.getStaticSizes() ||
                s.getStaticStrides() != firstSlice.getStaticStrides()) {
                return rewriter.notifyMatchFailure(genericOp, "mismatched slice params");
            }
            sources.push_back(s.getSource());
        }

        sliceInfo.firstSlice = firstSlice;
        sliceInfo.srcType = srcType;
        sliceInfo.resultType = cast<RankedTensorType>(genericOp.getResult(0).getType());
        sliceInfo.staticSizes.assign(
            firstSlice.getStaticSizes().begin(), firstSlice.getStaticSizes().end()
        );
        sliceInfo.sources = std::move(sources);
        return success();
    }

    LogicalResult checkConvEpilogueSliceShapeAndSources(
        linalg::GenericOp genericOp, PatternRewriter &rewriter, SliceInputsInfo &sliceInfo
    ) const {
        // Only sink genuine convolution epilogues: the crop must keep every source dimension
        // except exactly one (the spatial width), and at least one sliced source must be a
        // convolution result. The dropped-column count is left unbounded because a strided
        // ConvTranspose's extra columns plus any folded-in downstream slice can remove more than
        // one element. Requiring a convolution source keeps this scoped to conv+bias epilogues
        // (where the bias fuses into the conv) and, together with the identical slice-parameter
        // check above, excludes unrelated backbone crops such as channel splits.
        // getStaticSizes() is indexed by source dimension (rank reduction only drops unit result
        // dims), so it lines up with the source shape here.
        int64_t shrunkDims = 0;
        for (auto [sliceSize, srcDim] :
             llvm::zip_equal(sliceInfo.staticSizes, sliceInfo.srcType.getShape())) {
            if (sliceSize != srcDim) {
                ++shrunkDims;
            }
        }
        if (shrunkDims != 1) {
            return rewriter.notifyMatchFailure(
                genericOp, "slice does not crop exactly one dimension"
            );
        }
        if (llvm::none_of(sliceInfo.sources, [](Value src) {
                return src.getDefiningOp<linalg::ConvolutionOpInterface>() != nullptr;
            })) {
            return rewriter.notifyMatchFailure(
                genericOp, "no convolution source to fuse the bias into"
            );
        }
        return success();
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
        if (failed(checkElementwiseSignature(genericOp, rewriter))) {
            return failure();
        }

        SliceInputsInfo sliceInfo;
        if (failed(collectSliceInputs(genericOp, rewriter, sliceInfo))) {
            return failure();
        }
        if (failed(checkConvEpilogueSliceShapeAndSources(genericOp, rewriter, sliceInfo))) {
            return failure();
        }

        auto loc = genericOp.getLoc();
        Type outElemType = sliceInfo.resultType.getElementType();
        auto newResultType = RankedTensorType::get(sliceInfo.srcType.getShape(), outElemType);

        Value newInit =
            tensor::EmptyOp::create(rewriter, loc, sliceInfo.srcType.getShape(), outElemType)
                .getResult();

        int64_t rank = sliceInfo.srcType.getRank();
        auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
        SmallVector<AffineMap> maps(sliceInfo.sources.size() + 1, identityMap);
        SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);

        auto newGeneric = linalg::GenericOp::create(
            rewriter, loc, TypeRange{newResultType}, ValueRange(sliceInfo.sources),
            ValueRange{newInit}, maps, iterators
        );
        rewriter.cloneRegionBefore(
            genericOp.getRegion(), newGeneric.getRegion(), newGeneric.getRegion().end()
        );

        // Re-apply the original (possibly rank-reducing) slice on the elementwise result.
        Value newSlice = tensor::ExtractSliceOp::create(
            rewriter, loc, sliceInfo.resultType, newGeneric.getResult(0),
            sliceInfo.firstSlice.getMixedOffsets(), sliceInfo.firstSlice.getMixedSizes(),
            sliceInfo.firstSlice.getMixedStrides()
        );
        rewriter.replaceOp(genericOp, newSlice);
        return success();
    }
};

// Registers the conv1d->conv2d promotion patterns for strides in
// [minStride, maxStride]; see the pattern for what the range means.
void populateConv1dPromotionPatterns(
    MLIRContext *context, RewritePatternSet &patterns, int64_t minStride, int64_t maxStride
) {
    patterns.insert<Conv1DtoConv2DPromotionPattern<linalg::Conv1DNcwFcwOp>>(
        context, minStride, maxStride
    );
    patterns.insert<Conv1DtoConv2DPromotionPattern<linalg::DepthwiseConv1DNcwCwOp>>(
        context, minStride, maxStride
    );
}

void populateOptimizeConv1DPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    // Move the ConvTranspose trailing-column slice below its elementwise bias/truncf epilogue
    // so the bias fuses into the conv (needed for the native conv1d path).
    patterns.insert<SinkExtractSliceThroughElementwisePattern>(context);

    // Default conv1d lowering: promote 1D convs to their 2D counterparts with
    // the added spatial extent set to 1, so they flow through the established
    // conv2d path. By the time this pass runs,
    // DecomposeConvsForHardwarePass has already rewritten out-of-limits convs
    // to in-limits ones. Wide W-live 1D depthwise convs whose channels do not
    // fit the dedicated 1D lowering's fixed packing blocks are re-encoded into
    // the H-live orientation by DepthwiseConv2DWLiveToHLivePattern.
    populateConv1dPromotionPatterns(context, patterns, /*minStride=*/1, /*maxStride=*/2);
    patterns.insert<DepthwiseConv2DWLiveToHLivePattern>(context);
}

} // namespace mlir::syna::torq
