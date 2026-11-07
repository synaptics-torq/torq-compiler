// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Conversion/TorchOnnxToTorch/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

#define DEBUG_TYPE "torq-legacy-onnx-quantization"

// Rewrite the quantized_decomposed Q/DQ ops that the ONNX importer emits since IREE 3.13
// (torch-mlir 239cb08) back to the quantized tensor form it emitted before:
//
//   quantized_decomposed.dequantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
//   -> aten.dequantize.self(aten._make_per_tensor_quantized_tensor(x, scale, zp))
//
//   quantized_decomposed.quantize_per_tensor(x, scale, zp, qmin, qmax, dtype)
//   -> aten.int_repr(aten.quantize_per_tensor(x, scale, zp, qdtype))
//
// and the per-channel equivalents. torch-mlir's FuseQuantizedOps only fuses the old form into
// integer convolutions and matmuls; the new form reaches Torq as float ops between dequantize
// and quantize generics.
//
// TODO: torch-mlir leaves this fusion to the backends (llvm/torch-mlir#4600) and will remove
// FuseQuantizedOps. Fuse dq -> op -> q in Torq at the linalg level, then remove this pass.

using namespace mlir::torch;

namespace mlir::syna::torq_hl {

namespace {

// The quantized tensor type for an integer tensor type (si8 -> !torch.qint8, ...), or null.
static Torch::ValueTensorType getQuantizedType(Type type) {
    auto qType = onnx_c::getQTorchTypeFromTorchIntType(type);
    if (!qType || !isa<Torch::QInt8Type, Torch::QUInt8Type, Torch::QInt16Type, Torch::QInt32Type>(
                      qType.getDtype()
                  ))
        return {};
    return qType;
}

// The old form clamps to the full range of the integer type: only rewrite ops whose bounds are
// that range.
static bool hasFullRangeBounds(Value quantMin, Value quantMax, Type intType) {
    int64_t min, max;
    if (!matchPattern(quantMin, Torch::m_TorchConstantInt(&min)) ||
        !matchPattern(quantMax, Torch::m_TorchConstantInt(&max)))
        return false;
    auto type = dyn_cast<IntegerType>(intType);
    if (!type)
        return false;
    unsigned width = type.getWidth();
    if (type.isUnsigned())
        return min == 0 && max == static_cast<int64_t>((uint64_t(1) << width) - 1);
    return min == -(int64_t(1) << (width - 1)) && max == (int64_t(1) << (width - 1)) - 1;
}

// A torch.constant.int with the scalar type of a quantized tensor type.
static Value
getQuantizedDtypeConstant(PatternRewriter &rewriter, Location loc, Torch::ValueTensorType qType) {
    return Torch::ConstantIntOp::create(
        rewriter, loc,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(Torch::getScalarTypeForType(qType.getDtype()
        )))
    );
}

template <typename DequantizeOp>
static LogicalResult rewriteDequantize(DequantizeOp op, PatternRewriter &rewriter) {
    auto inputType = cast<Torch::ValueTensorType>(op.getInput().getType());
    auto qType = getQuantizedType(inputType);
    if (!qType || !hasFullRangeBounds(op.getQuantMin(), op.getQuantMax(), inputType.getDtype()))
        return rewriter.notifyMatchFailure(op, "not an integer dequantize over the full range");

    Value quantized;
    if constexpr (std::is_same_v<DequantizeOp, Torch::QuantizedDecomposedDequantizePerTensorOp>) {
        quantized = Torch::Aten_MakePerTensorQuantizedTensorOp::create(
            rewriter, op.getLoc(), qType, op.getInput(), op.getScale(), op.getZeroPoint()
        );
    }
    else {
        if (isa<Torch::NoneType>(op.getZeroPoints().getType()))
            return rewriter.notifyMatchFailure(op, "per-channel dequantize without zero points");
        quantized = Torch::Aten_MakePerChannelQuantizedTensorOp::create(
            rewriter, op.getLoc(), qType, op.getInput(), op.getScales(), op.getZeroPoints(),
            op.getAxis()
        );
    }
    rewriter.replaceOpWithNewOp<Torch::AtenDequantizeSelfOp>(op, op.getType(), quantized);
    return success();
}

template <typename QuantizeOp>
static LogicalResult rewriteQuantize(QuantizeOp op, PatternRewriter &rewriter) {
    auto resultType = cast<Torch::ValueTensorType>(op.getType());
    auto qType = getQuantizedType(resultType);
    if (!qType || !hasFullRangeBounds(op.getQuantMin(), op.getQuantMax(), resultType.getDtype()))
        return rewriter.notifyMatchFailure(op, "not an integer quantize over the full range");

    Value dtype = getQuantizedDtypeConstant(rewriter, op.getLoc(), qType);
    Value quantized;
    if constexpr (std::is_same_v<QuantizeOp, Torch::QuantizedDecomposedQuantizePerTensorOp>) {
        quantized = Torch::AtenQuantizePerTensorOp::create(
            rewriter, op.getLoc(), qType, op.getInput(), op.getScale(), op.getZeroPoint(), dtype
        );
    }
    else {
        quantized = Torch::AtenQuantizePerChannelOp::create(
            rewriter, op.getLoc(), qType, op.getInput(), op.getScales(), op.getZeroPoints(),
            op.getAxis(), dtype
        );
    }
    rewriter.replaceOpWithNewOp<Torch::AtenIntReprOp>(op, resultType, quantized);
    return success();
}

template <typename Op> struct LegacyQuantizationPattern : public OpRewritePattern<Op> {
    using OpRewritePattern<Op>::OpRewritePattern;

    LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
        if constexpr (std::is_same_v<Op, Torch::QuantizedDecomposedDequantizePerTensorOp> ||
                      std::is_same_v<Op, Torch::QuantizedDecomposedDequantizePerChannelOp>)
            return rewriteDequantize(op, rewriter);
        else
            return rewriteQuantize(op, rewriter);
    }
};

class LegacyOnnxQuantizationPass
    : public impl::LegacyOnnxQuantizationBase<LegacyOnnxQuantizationPass> {
  public:
    using LegacyOnnxQuantizationBase<LegacyOnnxQuantizationPass>::LegacyOnnxQuantizationBase;

    void runOnOperation() override {
        MLIRContext *ctx = &getContext();
        RewritePatternSet patterns(ctx);
        patterns.add<
            LegacyQuantizationPattern<Torch::QuantizedDecomposedDequantizePerTensorOp>,
            LegacyQuantizationPattern<Torch::QuantizedDecomposedDequantizePerChannelOp>,
            LegacyQuantizationPattern<Torch::QuantizedDecomposedQuantizePerTensorOp>,
            LegacyQuantizationPattern<Torch::QuantizedDecomposedQuantizePerChannelOp>>(ctx);
        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
            return signalPassFailure();
    }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>> createLegacyOnnxQuantizationPass() {
    return std::make_unique<LegacyOnnxQuantizationPass>();
}

} // namespace mlir::syna::torq_hl
