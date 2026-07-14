// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Conversions/LinalgToTorqHL/QuantPatternUtils.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "torq/Conversions/LinalgToTorqHL/MatchingFunctions.h"
#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#define DEBUG_TYPE "linalg-torq-q-conv2d-pattern"

namespace mlir::syna::torq {

namespace {

static std::optional<int32_t> getScalarI32Const(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantOp>();
    if (!cst)
        return std::nullopt;
    auto ia = dyn_cast<IntegerAttr>(cst.getValue());
    if (!ia)
        return std::nullopt;
    return static_cast<int32_t>(ia.getInt());
}

// Compute -inputZp * sum(adjustedWeights) per output channel. adjustedWeights
// are already sign-adjusted for weight_zp, so this term cancels the input_zp
// offset introduced by keeping the input tensor in its as-quantized (unsigned)
// form.
static Value
computeInputZpCorrection(Value adjustedWeights, int32_t inputZp, PatternRewriter &rewriter) {
    auto wTy = cast<RankedTensorType>(adjustedWeights.getType());
    auto loc = adjustedWeights.getLoc();
    auto i32Ty = rewriter.getI32Type();
    SmallVector<int64_t> biasShape{wTy.getShape()[0]};
    auto reduceInitTy = RankedTensorType::get(biasShape, i32Ty);
    auto reduceInit =
        arith::ConstantOp::create(rewriter, loc, reduceInitTy, rewriter.getZeroAttr(reduceInitTy))
            .getResult();

    SmallVector<int64_t> reduceDims;
    for (int64_t i = 1; i < wTy.getRank(); ++i)
        reduceDims.push_back(i);

    auto sumW = linalg::ReduceOp::create(
                    rewriter, loc, adjustedWeights, reduceInit, reduceDims,
                    [](OpBuilder &b, Location loc, ValueRange args) {
                        auto toI32 = [&](Value v) -> Value {
                            if (v.getType() == b.getI32Type())
                                return v;
                            return arith::ExtSIOp::create(b, loc, b.getI32Type(), v);
                        };
                        auto extL = toI32(args[0]);
                        auto extR = toI32(args[1]);
                        auto sum = arith::AddIOp::create(b, loc, extL, extR);
                        linalg::YieldOp::create(b, loc, ArrayRef<Value>{sum});
                    }
    ).getResult(0);

    auto zpConst = arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(-inputZp));
    auto mulInit = tensor::EmptyOp::create(rewriter, loc, biasShape, i32Ty).getResult();
    return linalg::GenericOp::create(
               rewriter, loc, mulInit.getType(), ValueRange{sumW, zpConst}, ValueRange{mulInit},
               SmallVector<AffineMap>{
                   rewriter.getMultiDimIdentityMap(1),
                   AffineMap::get(1, 0, {}, rewriter.getContext()),
                   rewriter.getMultiDimIdentityMap(1)
               },
               SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
               [](OpBuilder &b, Location loc, ValueRange args) {
                   auto mul = arith::MulIOp::create(b, loc, args[0], args[1]);
                   linalg::YieldOp::create(b, loc, ArrayRef<Value>{mul});
               }
    ).getResult(0);
}

// Add two per-channel i32 bias tensors.
static Value addPerChannelBias(Value lhs, Value rhs, PatternRewriter &rewriter) {
    auto ty = cast<RankedTensorType>(lhs.getType());
    auto loc = lhs.getLoc();
    auto init =
        tensor::EmptyOp::create(rewriter, loc, ty.getShape(), ty.getElementType()).getResult();
    return linalg::GenericOp::create(
               rewriter, loc, ty, ValueRange{lhs, rhs}, ValueRange{init},
               SmallVector<AffineMap>{
                   rewriter.getMultiDimIdentityMap(1), rewriter.getMultiDimIdentityMap(1),
                   rewriter.getMultiDimIdentityMap(1)
               },
               SmallVector<utils::IteratorType>{utils::IteratorType::parallel},
               [](OpBuilder &b, Location loc, ValueRange args) {
                   auto sum = arith::AddIOp::create(b, loc, args[0], args[1]);
                   linalg::YieldOp::create(b, loc, ArrayRef<Value>{sum});
               }
    ).getResult(0);
}
// Compute a (multiplier, shift) pair that approximates `scale` as
//   scale ~= multiplier / 2^shift
// The Torq convolution hardware requires the shift amount to be a multiple of
// 4, so we choose the largest multiple-of-4 shift whose rounded multiplier still
// fits in a signed 32-bit integer.  The scale is taken from the Q graph
// (dequant_scale / quant_scale), so the shift is derived from the actual graph
// constants rather than a hardcoded value.
static bool computeMultiplierAndShift(double scale, int32_t &multiplier, int32_t &shift) {
    if (scale == 0.0) {
        multiplier = 0;
        // Shift is irrelevant when the multiplier is zero, but keep it a
        // hardware-legal multiple of 4.
        shift = 28;
        return true;
    }
    if (scale < 0.0)
        return false;

    // Largest multiple-of-4 shift supported by the current hardware encoding.
    constexpr int32_t kMaxShift = 60;
    for (int32_t s = kMaxShift; s >= 0; s -= 4) {
        int64_t mult = std::llround(scale * static_cast<double>(1LL << s));
        if (mult >= std::numeric_limits<int32_t>::min() &&
            mult <= std::numeric_limits<int32_t>::max()) {
            multiplier = static_cast<int32_t>(mult);
            shift = s;
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] computeMultiplierAndShift scale=" << scale
                                    << " multiplier=" << multiplier << " shift=" << shift << "\n";);
            return true;
        }
    }
    LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] computeMultiplierAndShift failed for scale="
                            << scale << "\n";);
    return false;
}

static bool isElementwiseAddI32(linalg::GenericOp op) {
    if (!op || op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
        return false;
    if (!op.getRegion().hasOneBlock())
        return false;
    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1)
        return false;
    auto addi = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();
    if (!addi)
        return false;
    auto lhs = dyn_cast<BlockArgument>(addi.getLhs());
    auto rhs = dyn_cast<BlockArgument>(addi.getRhs());
    if (!lhs || !rhs)
        return false;
    return lhs.getOwner() == op.getBody() && rhs.getOwner() == op.getBody();
}

static Value extractBiasOperand(linalg::GenericOp addOp, Value convOutput) {
    Value biasOperand =
        addOp->getOperand(0) == convOutput ? addOp->getOperand(1) : addOp->getOperand(0);
    if (auto broadcastOp = biasOperand.getDefiningOp<linalg::BroadcastOp>())
        biasOperand = broadcastOp.getInput();
    return biasOperand;
}

// If `v` is a constant fp tensor or a stride-1 slice of one, return the
// element attribute and the actual tensor type. Returns nullptr on failure.
static DenseFPElementsAttr getConstantFPTensorAttr(Value v, RankedTensorType &outTy) {
    if (auto cstOp = v.getDefiningOp<arith::ConstantOp>()) {
        outTy = dyn_cast<RankedTensorType>(v.getType());
        auto attr = dyn_cast<DenseFPElementsAttr>(cstOp.getValue());
        LLVM_DEBUG(if (!attr) llvm::dbgs(
                   ) << "[getConstantFPTensorAttr] direct constant is not DenseFPElementsAttr\n";);
        return attr;
    }

    auto extractSliceOp = v.getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractSliceOp) {
        LLVM_DEBUG(llvm::dbgs(
                   ) << "[getConstantFPTensorAttr] input is not constant or extract_slice: "
                     << *v.getDefiningOp() << "\n";);
        return nullptr;
    }
    auto sourceCst = extractSliceOp.getSource().getDefiningOp<arith::ConstantOp>();
    if (!sourceCst) {
        LLVM_DEBUG(llvm::dbgs(
                   ) << "[getConstantFPTensorAttr] extract_slice source is not constant\n";);
        return nullptr;
    }
    auto sourceDense = dyn_cast<DenseFPElementsAttr>(sourceCst.getValue());
    if (!sourceDense) {
        LLVM_DEBUG(llvm::dbgs() << "[getConstantFPTensorAttr] extract_slice source constant is not "
                                   "DenseFPElementsAttr\n";);
        return nullptr;
    }
    outTy = dyn_cast<RankedTensorType>(v.getType());
    if (!outTy)
        return nullptr;

    auto offsets = extractSliceOp.getStaticOffsets();
    auto sizes = extractSliceOp.getStaticSizes();
    auto strides = extractSliceOp.getStaticStrides();
    if (offsets.size() != 1 || sizes.size() != 1 || strides.size() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[getConstantFPTensorAttr] extract_slice rank != 1\n";);
        return nullptr;
    }
    int64_t offset = offsets[0];
    int64_t size = sizes[0];
    int64_t stride = strides[0];
    if (ShapedType::isDynamic(offset) || ShapedType::isDynamic(size) ||
        ShapedType::isDynamic(stride)) {
        LLVM_DEBUG(llvm::dbgs(
                   ) << "[getConstantFPTensorAttr] extract_slice has dynamic offset/size/stride\n";
        );
        return nullptr;
    }

    SmallVector<APFloat> slicedVals;
    slicedVals.reserve(size);
    auto sourceValues = sourceDense.getValues<APFloat>();
    auto it = sourceValues.begin();
    std::advance(it, offset);
    for (int64_t i = 0; i < size; ++i) {
        slicedVals.push_back(*it);
        if (i + 1 < size)
            std::advance(it, stride);
    }
    LLVM_DEBUG(llvm::dbgs() << "[getConstantFPTensorAttr] sliced constant of size " << size
                            << " from offset " << offset << "\n";);
    return DenseFPElementsAttr::get(outTy, slicedVals);
}

// Build a constant i32 tensor (optionally sliced) from a known i32 source
// constant by applying the PT2E quant formula:
//   out = clamp(round(in * scale) + zp, min, max)
static Value buildQuantConstantFromI32Source(
    linalg::GenericOp op, Value sourceCst, double scale, double zp, double min, double max,
    ArrayRef<OpFoldResult> sliceOffsets, ArrayRef<OpFoldResult> sliceSizes,
    ArrayRef<OpFoldResult> sliceStrides, PatternRewriter &rewriter
) {
    auto cstOp = sourceCst.getDefiningOp<arith::ConstantOp>();
    if (!cstOp)
        return nullptr;
    auto dense = dyn_cast<DenseIntElementsAttr>(cstOp.getValue());
    auto sourceTy = dyn_cast<RankedTensorType>(sourceCst.getType());
    if (!dense || !sourceTy || !sourceTy.getElementType().isInteger(32))
        return nullptr;

    int64_t iMin = std::max<int64_t>(
        std::llround(min), static_cast<int64_t>(std::numeric_limits<int32_t>::min())
    );
    int64_t iMax = std::min<int64_t>(
        std::llround(max), static_cast<int64_t>(std::numeric_limits<int32_t>::max())
    );
    int64_t iZp = std::llround(zp);
    SmallVector<int32_t> outVals;
    outVals.reserve(dense.size());
    for (APInt apv : dense.getValues<APInt>()) {
        double dv = static_cast<double>(static_cast<int32_t>(apv.getSExtValue())) * scale;
        int64_t iv = std::llround(dv) + iZp;
        if (iv < iMin)
            iv = iMin;
        if (iv > iMax)
            iv = iMax;
        outVals.push_back(static_cast<int32_t>(iv));
    }

    auto outTy = RankedTensorType::get(sourceTy.getShape(), rewriter.getI32Type());
    Value fullCst;
    {
        OpBuilder::InsertionGuard g(rewriter);
        if (auto funcOp = op->getParentOfType<func::FuncOp>())
            rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        fullCst = arith::ConstantOp::create(
                      rewriter, op.getLoc(), outTy, rewriter.getI32TensorAttr(outVals)
        )
                      .getResult();
    }

    if (sliceOffsets.empty())
        return fullCst;

    int64_t staticSize = -1;
    if (auto sizeAttr = dyn_cast<Attribute>(sliceSizes[0]))
        if (auto intAttr = dyn_cast<IntegerAttr>(sizeAttr))
            staticSize = intAttr.getInt();
    SmallVector<int64_t> resultShape;
    if (staticSize >= 0)
        resultShape.push_back(staticSize);
    else
        resultShape.push_back(ShapedType::kDynamic);
    auto sliceTy = RankedTensorType::get(resultShape, rewriter.getI32Type());
    return tensor::ExtractSliceOp::create(
               rewriter, op.getLoc(), sliceTy, fullCst, SmallVector<OpFoldResult>{sliceOffsets[0]},
               SmallVector<OpFoldResult>{sliceSizes[0]},
               SmallVector<OpFoldResult>{
                   sliceStrides.empty() ? rewriter.getIndexAttr(1) : sliceStrides[0]
               }
    )
        .getResult();
}

// Fold a per-channel PT2E-style quant generic whose body is
//   divf/mulf -> roundeven -> addf zp -> maxf -> minf -> fptosi
// into a constant i32 tensor. This avoids the heavy computeArithConst path
// (which needs additional dialects loaded) for the simple bias quant case.
static Value foldQuantGenericToConstant(linalg::GenericOp op, PatternRewriter &rewriter) {
    double scale, zp, min, max;
    if (!matchQuantGeneric(op, scale, zp, min, max))
        return nullptr;
    if (scale == 0.0)
        return nullptr;
    if (op.getNumDpsInputs() != 1)
        return nullptr;
    Value input = op.getDpsInputOperand(0)->get();

    RankedTensorType inTy;
    auto dense = getConstantFPTensorAttr(input, inTy);
    if (dense) {
        int64_t iMin = std::max<int64_t>(
            std::llround(min), static_cast<int64_t>(std::numeric_limits<int32_t>::min())
        );
        int64_t iMax = std::min<int64_t>(
            std::llround(max), static_cast<int64_t>(std::numeric_limits<int32_t>::max())
        );
        int64_t iZp = std::llround(zp);
        SmallVector<int32_t> outVals;
        outVals.reserve(dense.size());
        for (const APFloat &f : dense.getValues<APFloat>()) {
            double dv = f.convertToDouble();
            int64_t iv = std::llround(dv / scale) + iZp;
            if (iv < iMin)
                iv = iMin;
            if (iv > iMax)
                iv = iMax;
            outVals.push_back(static_cast<int32_t>(iv));
        }
        auto outTy = RankedTensorType::get(inTy.getShape(), rewriter.getI32Type());
        return arith::ConstantOp::create(
                   rewriter, op.getLoc(), outTy, rewriter.getI32TensorAttr(outVals)
        )
            .getResult();
    }

    // The input may itself be a dequant generic operating on a constant i32
    // tensor (or a slice of one). Fold the dequant+quant chain into a single
    // constant slice so the Q conv rewrite can avoid creating a runtime
    // linalg.generic for the bias.
    auto dequantOp = dyn_cast<linalg::GenericOp>(input.getDefiningOp());
    double dequantScale;
    int32_t ignoredDequantZp = 0;
    if (!dequantOp || !matchDequantGeneric(dequantOp, dequantScale, ignoredDequantZp))
        return nullptr;

    Value dequantInput = dequantOp.getDpsInputOperand(0)->get();
    SmallVector<OpFoldResult> sliceOffsets;
    SmallVector<OpFoldResult> sliceSizes;
    SmallVector<OpFoldResult> sliceStrides;
    Value source = dequantInput;
    while (auto extract = source.getDefiningOp<tensor::ExtractSliceOp>()) {
        sliceOffsets = extract.getMixedOffsets();
        sliceSizes = extract.getMixedSizes();
        sliceStrides = extract.getMixedStrides();
        source = extract.getSource();
    }

    return buildQuantConstantFromI32Source(
        op, source, dequantScale / scale, zp, min, max, sliceOffsets, sliceSizes, sliceStrides,
        rewriter
    );
}

// Compute -inputZp * sum(adjustedWeights) per output channel directly from the
// constant weight tensor. This avoids creating a linalg.reduce that cannot be
// folded by computeArithConst in pipelines where the constant-folding JIT is
// not available (e.g., the tile-fit check pipeline).
static Value
buildInputZpCorrectionConstant(Value weights, int32_t inputZp, PatternRewriter &rewriter) {
    // Look through tensor.extract_slice to reach the underlying constant and
    // remember the outermost slice parameters so we can apply the same channel
    // slice to the correction tensor.
    SmallVector<OpFoldResult> sliceOffsets;
    SmallVector<OpFoldResult> sliceSizes;
    SmallVector<OpFoldResult> sliceStrides;
    Value source = weights;
    while (auto extract = source.getDefiningOp<tensor::ExtractSliceOp>()) {
        sliceOffsets = extract.getMixedOffsets();
        sliceSizes = extract.getMixedSizes();
        sliceStrides = extract.getMixedStrides();
        source = extract.getSource();
    }

    auto cstOp = source.getDefiningOp<arith::ConstantOp>();
    if (!cstOp)
        return nullptr;
    auto dense = dyn_cast<ElementsAttr>(cstOp.getValue());
    if (!dense)
        return nullptr;
    auto wTy = dyn_cast<RankedTensorType>(source.getType());
    if (!wTy || wTy.getRank() != 4 || !wTy.getElementType().isInteger(8))
        return nullptr;

    ArrayRef<int64_t> shape = wTy.getShape();
    SmallVector<int64_t> strides(wTy.getRank(), 1);
    for (int64_t i = wTy.getRank() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];

    int64_t C = shape[0];
    SmallVector<int32_t> sums(C, 0);
    int64_t flatIdx = 0;
    for (APInt apv : dense.getValues<APInt>()) {
        int64_t c = flatIdx / strides[0];
        int8_t v = static_cast<int8_t>(apv.getSExtValue());
        sums[c] += static_cast<int32_t>(v);
        ++flatIdx;
    }

    int32_t factor = -inputZp;
    SmallVector<int32_t> correction;
    correction.reserve(C);
    for (int32_t s : sums)
        correction.push_back(s * factor);

    auto outTy = RankedTensorType::get({C}, rewriter.getI32Type());

    // Hoist the full correction constant to the function entry so it is not
    // duplicated inside a tiled loop.
    Value fullCorrection;
    {
        OpBuilder::InsertionGuard g(rewriter);
        if (auto funcOp = cstOp->getParentOfType<func::FuncOp>())
            rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        fullCorrection = arith::ConstantOp::create(
                             rewriter, cstOp.getLoc(), outTy, rewriter.getI32TensorAttr(correction)
        )
                             .getResult();
    }

    if (source == weights)
        return fullCorrection;

    // The original weights were sliced; apply the same channel-dim slice to the
    // correction tensor so the per-channel correction matches the conv output.
    // The offset may be dynamic (e.g. a loop induction variable), so keep it as
    // an OpFoldResult rather than forcing it to be a constant.
    if (sliceOffsets.empty() || sliceSizes.empty())
        return nullptr;
    OpFoldResult offset = sliceOffsets[0];
    OpFoldResult size = sliceSizes[0];
    OpFoldResult stride = sliceStrides.empty() ? rewriter.getIndexAttr(1) : sliceStrides[0];

    int64_t staticSize = -1;
    if (auto sizeAttr = dyn_cast<Attribute>(size))
        if (auto intAttr = dyn_cast<IntegerAttr>(sizeAttr))
            staticSize = intAttr.getInt();

    SmallVector<int64_t> resultShape;
    if (staticSize >= 0)
        resultShape.push_back(staticSize);
    else
        resultShape.push_back(ShapedType::kDynamic);
    auto sliceTy = RankedTensorType::get(resultShape, rewriter.getI32Type());
    return tensor::ExtractSliceOp::create(
               rewriter, cstOp.getLoc(), sliceTy, fullCorrection, SmallVector<OpFoldResult>{offset},
               SmallVector<OpFoldResult>{size}, SmallVector<OpFoldResult>{stride}
    )
        .getResult();
}

// Elementwise add of two 1-D i32 tensors that are constants or slices of
// constants. Returns a (possibly sliced) constant tensor, or nullptr if either
// operand is not a constant/slice.
static Value addI32SlicesOfConstants(Value lhs, Value rhs, PatternRewriter &rewriter) {
    auto resolve = [](Value v, Value &cst, SmallVector<OpFoldResult> &offsets,
                      SmallVector<OpFoldResult> &sizes, SmallVector<OpFoldResult> &strides) {
        offsets.clear();
        sizes.clear();
        strides.clear();
        Value src = v;
        while (auto extract = src.getDefiningOp<tensor::ExtractSliceOp>()) {
            offsets = extract.getMixedOffsets();
            sizes = extract.getMixedSizes();
            strides = extract.getMixedStrides();
            src = extract.getSource();
        }
        cst = src;
        return isa<arith::ConstantOp>(cst.getDefiningOp());
    };

    Value lhsCst, rhsCst;
    SmallVector<OpFoldResult> lhsOffsets, lhsSizes, lhsStrides;
    SmallVector<OpFoldResult> rhsOffsets, rhsSizes, rhsStrides;
    if (!resolve(lhs, lhsCst, lhsOffsets, lhsSizes, lhsStrides) ||
        !resolve(rhs, rhsCst, rhsOffsets, rhsSizes, rhsStrides))
        return nullptr;

    auto lhsOp = dyn_cast<arith::ConstantOp>(lhsCst.getDefiningOp());
    auto rhsOp = dyn_cast<arith::ConstantOp>(rhsCst.getDefiningOp());
    if (!lhsOp || !rhsOp)
        return nullptr;
    auto lhsDense = dyn_cast<DenseIntElementsAttr>(lhsOp.getValue());
    auto rhsDense = dyn_cast<DenseIntElementsAttr>(rhsOp.getValue());
    if (!lhsDense || !rhsDense)
        return nullptr;
    auto ty = dyn_cast<RankedTensorType>(lhsCst.getType());
    if (!ty || ty != rhsCst.getType() || !ty.getElementType().isInteger(32))
        return nullptr;

    SmallVector<int32_t> out;
    out.reserve(ty.getNumElements());
    auto lhsVals = lhsDense.getValues<APInt>();
    auto rhsVals = rhsDense.getValues<APInt>();
    for (size_t i = 0; i < ty.getNumElements(); ++i) {
        int32_t a = static_cast<int32_t>(lhsVals[i].getSExtValue());
        int32_t b = static_cast<int32_t>(rhsVals[i].getSExtValue());
        out.push_back(a + b);
    }

    Value fullSum;
    {
        OpBuilder::InsertionGuard g(rewriter);
        if (auto funcOp = lhsOp->getParentOfType<func::FuncOp>())
            rewriter.setInsertionPointToStart(&funcOp.getBody().front());
        fullSum =
            arith::ConstantOp::create(rewriter, lhsOp.getLoc(), ty, rewriter.getI32TensorAttr(out))
                .getResult();
    }

    if (lhsOffsets.empty())
        return fullSum;

    int64_t staticSize = -1;
    if (auto sizeAttr = dyn_cast<Attribute>(lhsSizes[0]))
        if (auto intAttr = dyn_cast<IntegerAttr>(sizeAttr))
            staticSize = intAttr.getInt();
    SmallVector<int64_t> resultShape;
    if (staticSize >= 0)
        resultShape.push_back(staticSize);
    else
        resultShape.push_back(ShapedType::kDynamic);
    auto sliceTy = RankedTensorType::get(resultShape, rewriter.getI32Type());
    return tensor::ExtractSliceOp::create(
               rewriter, lhsOp.getLoc(), sliceTy, fullSum, SmallVector<OpFoldResult>{lhsOffsets[0]},
               SmallVector<OpFoldResult>{lhsSizes[0]},
               SmallVector<OpFoldResult>{
                   lhsStrides.empty() ? rewriter.getIndexAttr(1) : lhsStrides[0]
               }
    )
        .getResult();
}

static Value
buildInterleavedBiasScale(Value biasConst, int32_t multiplier, PatternRewriter &rewriter) {
    auto cstOp = dyn_cast<arith::ConstantOp>(biasConst.getDefiningOp());
    if (!cstOp)
        return nullptr;
    auto dense = dyn_cast<DenseIntElementsAttr>(cstOp.getValue());
    if (!dense)
        return nullptr;
    auto biasTy = dyn_cast<RankedTensorType>(biasConst.getType());
    if (!biasTy || biasTy.getRank() != 1)
        return nullptr;
    int64_t C = biasTy.getShape()[0];
    SmallVector<int32_t> vals;
    vals.reserve(C * 2);
    for (auto apint : dense.getValues<APInt>()) {
        vals.push_back(static_cast<int32_t>(apint.getSExtValue()));
        vals.push_back(multiplier);
    }
    auto sbTy = RankedTensorType::get({C * 2}, rewriter.getI32Type());
    return arith::ConstantOp::create(
               rewriter, biasConst.getLoc(), sbTy, rewriter.getI32TensorAttr(vals)
    )
        .getResult();
}

// Build a {C*2} i32 scale_bias tensor by interleaving a dynamic per-channel
// i32 bias (shape {C}) with a scalar multiplier. The result layout is
// [bias_0, mult, bias_1, mult, ...].
static Value buildDynamicInterleavedBiasScale(
    Value bias, int32_t multiplier, Location loc, PatternRewriter &rewriter
) {
    auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasTy || biasTy.getRank() != 1 || biasTy.getElementType() != rewriter.getI32Type())
        return nullptr;
    int64_t C = biasTy.getShape()[0];
    if (ShapedType::isDynamic(C))
        return nullptr;

    SmallVector<int32_t> multVals(C, multiplier);
    auto multTy = RankedTensorType::get({C}, rewriter.getI32Type());
    auto multCst =
        arith::ConstantOp::create(rewriter, loc, multTy, rewriter.getI32TensorAttr(multVals))
            .getResult();

    auto sbTy = RankedTensorType::get({C * 2}, rewriter.getI32Type());
    auto init =
        tensor::EmptyOp::create(rewriter, loc, sbTy.getShape(), sbTy.getElementType()).getResult();
    // Place biases at even offsets (stride 2) and scales at odd offsets (stride 2)
    // so the final layout is [bias_0, mult, bias_1, mult, ...].
    auto withBias = tensor::InsertSliceOp::create(
        rewriter, loc, bias, init, SmallVector<OpFoldResult>{rewriter.getIndexAttr(0)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(C)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(2)}
    );
    auto scaleBias = tensor::InsertSliceOp::create(
        rewriter, loc, multCst, withBias, SmallVector<OpFoldResult>{rewriter.getIndexAttr(1)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(C)},
        SmallVector<OpFoldResult>{rewriter.getIndexAttr(2)}
    );
    return scaleBias;
}

struct QConv2dConvert : public OpRewritePattern<linalg::Conv2DNchwFchwQOp> {
  private:
    const bool _markFuseGroups;

    // Match the dequant -> quant chain that follows the conv (or the optional
    // bias add).  On success `dequantOp` and `quantOp` are populated together
    // with their scales/zero-points.
    LogicalResult matchQConv2DQuantChain(
        linalg::Conv2DNchwFchwQOp convOp, Value chain, PatternRewriter &rewriter,
        linalg::GenericOp &dequantOp, double &dequantScale, linalg::GenericOp &quantOp,
        double &quantScale, double &quantZp, double &quantMin, double &quantMax
    ) const {
        if (!chain.hasOneUse()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[QConv2dConvert] chain does not have single use before dequant, uses: "
                << std::distance(chain.getUsers().begin(), chain.getUsers().end()) << "\n"
            );
            return rewriter.notifyMatchFailure(
                convOp, "chain does not have single use before dequant"
            );
        }
        dequantOp = dyn_cast<linalg::GenericOp>(*chain.getUsers().begin());
        int32_t ignoredDequantZp = 0;
        if (!dequantOp || !matchDequantGeneric(dequantOp, dequantScale, ignoredDequantZp)) {
            if (dequantOp)
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] failed to match Q dequant, user: "
                                 << *chain.getUsers().begin() << "\n"
                );
            return rewriter.notifyMatchFailure(convOp, "failed to match Q dequant");
        }
        Value dequantResult = dequantOp->getResult(0);
        if (!dequantResult.hasOneUse()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] dequant result has "
                             << std::distance(
                                    dequantResult.getUsers().begin(), dequantResult.getUsers().end()
                                )
                             << " uses\n"
            );
            return rewriter.notifyMatchFailure(convOp, "dequant result has multiple uses");
        }

        quantOp = dyn_cast<linalg::GenericOp>(*dequantResult.getUsers().begin());
        if (!quantOp || !matchQuantGeneric(quantOp, quantScale, quantZp, quantMin, quantMax)) {
            if (quantOp)
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] failed to match Q quant, user: "
                                 << *dequantResult.getUsers().begin() << "\n"
                );
            return rewriter.notifyMatchFailure(convOp, "failed to match Q quant");
        }
        return success();
    }

    // Build the {C*2} interleaved scale_bias tensor for the conv.  `bias` may be
    // the original bias value, a bias quant generic, or null (meaning a zero
    // bias is created).  This function folds the bias quant generic and applies
    // the input-zero-point correction when possible.  On success it returns the
    // scale_bias value and, via out parameters, any correction ops that must be
    // cleaned up after the conv chain has been replaced.
    Value buildQConv2DScaleBias(
        linalg::Conv2DNchwFchwQOp convOp, RankedTensorType convOutTy, Value bias,
        linalg::GenericOp addOp, Value torqWeights, int32_t inputZp, int32_t multiplier,
        PatternRewriter &rewriter, Operation *&zpCorrectionMulOp, Operation *&zpCorrectionAddOp,
        bool &foldedZpCorrection
    ) const {
        Location loc = convOp.getLoc();

        // Create a zero bias when no bias add is present.
        if (!bias) {
            int64_t C = convOutTy.getShape()[1];
            auto zeroTy = RankedTensorType::get({C}, rewriter.getI32Type());
            bias = arith::ConstantOp::create(rewriter, loc, zeroTy, rewriter.getI32IntegerAttr(0))
                       .getResult();
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] using zero bias\n");
        }

        // Fold the bias quant generic (if present) to a constant i32 tensor when
        // possible. For tiled convs inside loops the bias may be a dynamic slice
        // of a constant; in that case keep the generic output as a dynamic bias
        // and build the scale_bias tensor at runtime.
        if (auto biasGeneric = dyn_cast<linalg::GenericOp>(bias.getDefiningOp())) {
            if (auto foldedBias = foldQuantGenericToConstant(biasGeneric, rewriter)) {
                bias = foldedBias;
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] folded bias quant generic to constant\n"
                );
            }
            else {
                bias = biasGeneric->getResult(0);
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] using dynamic bias quant generic output\n"
                );
            }
        }
        auto biasTy = dyn_cast<RankedTensorType>(bias.getType());
        if (!biasTy || biasTy.getRank() != 1 || biasTy.getElementType() != rewriter.getI32Type()) {
            return nullptr;
        }

        zpCorrectionMulOp = nullptr;
        zpCorrectionAddOp = nullptr;
        foldedZpCorrection = true;
        if (inputZp != 0) {
            Value correction = buildInputZpCorrectionConstant(torqWeights, inputZp, rewriter);
            if (!correction) {
                // Fall back to the generic reduce path when the weights are not a
                // constant that we can fold directly.
                correction = computeInputZpCorrection(torqWeights, inputZp, rewriter);
                zpCorrectionMulOp = correction.getDefiningOp();
                bias = addPerChannelBias(bias, correction, rewriter);
                zpCorrectionAddOp = bias.getDefiningOp();
                FailureOr<Value> foldedBias2 = computeArithConst(bias, true, {});
                if (failed(foldedBias2)) {
                    foldedZpCorrection = false;
                }
                else {
                    bias = *foldedBias2;
                }
            }
            else {
                if (auto foldedBias = addI32SlicesOfConstants(bias, correction, rewriter)) {
                    bias = foldedBias;
                }
                else {
                    bias = addPerChannelBias(bias, correction, rewriter);
                    zpCorrectionAddOp = bias.getDefiningOp();
                    foldedZpCorrection = false;
                }
            }
        }

        if (dyn_cast<arith::ConstantOp>(bias.getDefiningOp())) {
            return buildInterleavedBiasScale(bias, multiplier, rewriter);
        }
        return buildDynamicInterleavedBiasScale(bias, multiplier, loc, rewriter);
    }

    // Replace the conv → (add) → dequant → quant chain with a single
    // torq_hl.conv2d using the pre-built weights and scale_bias tensor.
    LogicalResult rewriteQConv2DChain(
        linalg::Conv2DNchwFchwQOp convOp, linalg::GenericOp addOp, linalg::GenericOp dequantOp,
        linalg::GenericOp quantOp, Value input, Value torqWeights, Value scaleBias,
        const PaddingInfo &padInfo, int32_t inputZp, int32_t multiplier, int32_t shift,
        double quantZp, double quantMin, double quantMax, PatternRewriter &rewriter,
        Operation *zpCorrectionMulOp, Operation *zpCorrectionAddOp, bool foldedZpCorrection
    ) const {
        Location loc = convOp.getLoc();

        if (auto padOp = convOp->getOperand(0).getDefiningOp<tensor::PadOp>()) {
            if (llvm::all_of(padInfo.lrtbPad, [](int64_t p) { return p == 0; })) {
                return rewriter.notifyMatchFailure(convOp, "failed to fold padding");
            }
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] padding info lrtb=[" << padInfo.lrtbPad[0] << ","
                         << padInfo.lrtbPad[1] << "," << padInfo.lrtbPad[2] << ","
                         << padInfo.lrtbPad[3] << "]\n"
        );

        auto dilations = convOp.getDilations().getValues<int64_t>();
        std::vector<int64_t> finalDilationVec(dilations.begin(), dilations.end());
        Value dilatedWeights = torqWeights;
        auto mWts = getDilatedWts(torqWeights, finalDilationVec, /*isDW1DStride1=*/false, rewriter);
        if (!failed(mWts))
            dilatedWeights = *mWts;

        std::vector<int64_t> strideVec = attrValuesAsVec(convOp.getStrides());

        Value convInit = quantOp.getDpsInitOperand(0)->get();
        auto outTy = cast<RankedTensorType>(convInit.getType());

        int32_t outputZp = static_cast<int32_t>(std::llround(quantZp));
        int32_t outputMin = static_cast<int32_t>(std::llround(quantMin));
        int32_t outputMax = static_cast<int32_t>(std::llround(quantMax));

        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] creating torq_hl.conv2d, outTy=" << outTy << "\n"
        );
        {
            OpBuilder::InsertionGuard g(rewriter);
            rewriter.setInsertionPoint(quantOp);

            Value convResult = torq_hl::Conv2DOp::create(
                                   rewriter, loc, outTy, convInit, inputZp, /*weight_zp=*/0,
                                   outputZp, outputMin, outputMax, shift,
                                   /*groups=*/1, padInfo.lrtbPad, strideVec, finalDilationVec,
                                   torq_hl::VectorizationModeEnum::None, dilatedWeights, scaleBias,
                                   input, /*nhwc_input=*/false
            )
                                   .getResult(0);

            rewriter.replaceOp(convOp, convOp.getDpsInitOperand(0)->get());
            if (addOp)
                rewriter.replaceOp(addOp, addOp.getDpsInitOperand(0)->get());
            rewriter.replaceOp(dequantOp, dequantOp.getDpsInitOperand(0)->get());
            rewriter.replaceOp(quantOp, convResult);
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] replaced Q chain with torq_hl.conv2d\n");
        }

        // Erase the intermediate ops created for the input-zero-point correction
        // only when the corrected bias was folded to a constant. Otherwise the
        // generic bias add is still live as an input to the runtime scale_bias.
        if (foldedZpCorrection) {
            if (zpCorrectionAddOp)
                rewriter.eraseOp(zpCorrectionAddOp);
            if (zpCorrectionMulOp) {
                if (auto reduceOp = zpCorrectionMulOp->getOperand(0).getDefiningOp())
                    rewriter.eraseOp(reduceOp);
                rewriter.eraseOp(zpCorrectionMulOp);
            }
        }

        return success();
    }

  public:
    using OpRewritePattern::OpRewritePattern;
    QConv2dConvert(MLIRContext *context, bool markFuseGroups, PatternBenefit benefit = 1)
        : OpRewritePattern<linalg::Conv2DNchwFchwQOp>(context, benefit),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::Conv2DNchwFchwQOp convOp, PatternRewriter &rewriter) const override {
        LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] matching conv: " << convOp << "\n");
        TorqStructuredOpMatcher<linalg::Conv2DNchwFchwQOp> matcher;
        if (!matcher.addPredicate(notMarkedFuseGroupIf(_markFuseGroups)).match(convOp)) {
            return rewriter.notifyMatchFailure(convOp, "Q conv match failed");
        }

        Value convOutput = convOp->getResult(0);
        auto convOutTy = dyn_cast<RankedTensorType>(convOutput.getType());
        if (!convOutTy || !convOutTy.getElementType().isInteger(32)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] expected i32 conv output, got: "
                             << convOutput.getType() << "\n"
            );
            return rewriter.notifyMatchFailure(convOp, "expected i32 conv output");
        }

        auto maybeInputZp = getScalarI32Const(convOp->getOperand(2));
        auto maybeWeightZp = getScalarI32Const(convOp->getOperand(3));
        if (!maybeInputZp || !maybeWeightZp) {
            LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] input/weight zp not constant scalar\n");
            return rewriter.notifyMatchFailure(convOp, "input/weight zp not constant scalar");
        }
        int32_t inputZp = *maybeInputZp;
        int32_t weightZp = *maybeWeightZp;

        Value bias = nullptr;
        linalg::GenericOp addOp = nullptr;
        Value chain = convOutput;
        if (convOutput.hasOneUse()) {
            auto user = dyn_cast<linalg::GenericOp>(*convOutput.getUsers().begin());
            if (user && isElementwiseAddI32(user)) {
                addOp = user;
                bias = extractBiasOperand(user, convOutput);
                chain = user->getResult(0);
                LLVM_DEBUG(llvm::dbgs() << "[QConv2dConvert] found bias add\n");
            }
            else {
                LLVM_DEBUG(
                    llvm::dbgs() << "[QConv2dConvert] conv output user is not add: "
                                 << *convOutput.getUsers().begin() << "\n"
                );
            }
        }
        else {
            LLVM_DEBUG(
                llvm::dbgs() << "[QConv2dConvert] conv output has "
                             << std::distance(
                                    convOutput.getUsers().begin(), convOutput.getUsers().end()
                                )
                             << " uses\n"
            );
        }

        linalg::GenericOp dequantOp;
        double dequantScale;
        linalg::GenericOp quantOp;
        double quantScale, quantZp, quantMin, quantMax;
        if (failed(matchQConv2DQuantChain(
                convOp, chain, rewriter, dequantOp, dequantScale, quantOp, quantScale, quantZp,
                quantMin, quantMax
            ))) {
            return failure();
        }

        Value input = convOp->getOperand(0);
        Value weightsValue = convOp->getOperand(1);
        Value output = quantOp->getResult(0);

        // Fold backward padding so the pad op (if any) is included in the fusion
        // group and the conv can use hardware padding.
        PaddingInfo padInfo = foldBackwardPadding(input, rewriter, /*nchw=*/true);

        // Marking mode: annotate the conv → add → dequant → quant chain so it can be
        // tiled and fused as a single group; do not rewrite yet.
        if (_markFuseGroups) {
            markFuseGroupBackward(
                output, {input, weightsValue}, rewriter,
                convOp->template getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID)
            );
            return success();
        }

        if (quantScale == 0.0) {
            return rewriter.notifyMatchFailure(convOp, "quant scale is zero");
        }
        double scaleFactor = dequantScale / quantScale;
        int32_t multiplier, shift;
        if (!computeMultiplierAndShift(scaleFactor, multiplier, shift)) {
            return rewriter.notifyMatchFailure(convOp, "failed to compute multiplier/shift");
        }
        LLVM_DEBUG(
            llvm::dbgs() << "[QConv2dConvert] multiplier=" << multiplier << " shift=" << shift
                         << "\n"
        );

        std::optional<Value> optionalWeightZpV;
        if (weightZp != 0)
            optionalWeightZpV = convOp->getOperand(3);
        ScaleClampInfo dummyScInfo;
        Value torqWeights = preConversionWeights(
            weightsValue, Permutation::none(), optionalWeightZpV, dummyScInfo, rewriter,
            /*isDepthwise=*/false
        );

        Operation *zpCorrectionMulOp = nullptr;
        Operation *zpCorrectionAddOp = nullptr;
        bool foldedZpCorrection = true;
        Value scaleBias = buildQConv2DScaleBias(
            convOp, convOutTy, bias, addOp, torqWeights, inputZp, multiplier, rewriter,
            zpCorrectionMulOp, zpCorrectionAddOp, foldedZpCorrection
        );
        if (!scaleBias) {
            return rewriter.notifyMatchFailure(convOp, "failed to build scale_bias");
        }

        return rewriteQConv2DChain(
            convOp, addOp, dequantOp, quantOp, input, torqWeights, scaleBias, padInfo, inputZp,
            multiplier, shift, quantZp, quantMin, quantMax, rewriter, zpCorrectionMulOp,
            zpCorrectionAddOp, foldedZpCorrection
        );
    }
};

} // namespace

void populateLinalgToTorqHLQConv2DPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<QConv2dConvert>(context, markFuseGroups);
}

} // namespace mlir::syna::torq
