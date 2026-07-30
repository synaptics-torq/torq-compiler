// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Quantization pattern utilities
//
// ONNX quantized models lower to linalg.generic ops whose bodies encode the
// dequant / quant arithmetic (int→float conversion, clamp bounds, scale
// rounding, etc.).  Different producers (torch-mlir, ONNX Runtime, TOSA, ...)
// and integer representations (signed vs unsigned) emit slightly different IR
// bodies.  Each supported body shape is matched by a standalone "variant"
// matcher that returns a canonical info struct (DequantInfo or QuantInfo) on
// success, nullopt on mismatch.
//
// Full matching flow
// ------------------
//
//   linalg.generic candidate (looks like DQ or Q)
//            |
//            v
//   +---------------------------+
//   |  matchDequantGeneric      |
//   |  or matchQuantGeneric     |
//   +---------------------------+
//            |
//            v
//   +---------------------------+
//   |   variant dispatcher      |
//   |  (tries matchers in order)|
//   +---------------------------+
//            |
//     +------+------+
//     |             |
//     v             v
// +--------+   +----------+
// | signed |   | scale-one|
// | variant|   | variant  |
// +--------+   +----------+
//     |
//     v
// +---------------------------+
// | DequantInfo / QuantInfo   |
// +---------------------------+
//
// The public entry points (matchDequantGeneric, matchQuantGeneric) are the
// dispatchers.  Adding support for a new IR shape is one function plus one
// entry in the variant array.  Variant matchers use raw function pointers
// (DequantVariantFn, QuantVariantFn) rather than std::function to avoid heap
// allocation in the critical path.
//
// Quantized compute-op chains
// ---------------------------
// Patterns share QuantizedOpChain to represent the two common quantized
// compute-op shapes.  Intermediate ops (pad, transpose, bias add,
// div-by-count, ...) are handled by the caller before invoking the shared
// input/output matchers.
//
// 1. QDQ chain
//
//      int8 tensor
//          |
//          v
//      +----------------------------+
//      | DequantizeLinear (generic) |  <-- inputDequantOp
//      |  fp32 = (int8 - zp_in) * scale_in |
//      +----------------------------+
//          |
//          v
//      +----------------------+
//      | intermediate ops     |  (pad, transpose, reshape, ...)
//      +----------------------+
//          |
//          v
//      +----------------+
//      |  compute op    |  (e.g. arith.addf, arith.mulf, sigmoid)
//      |   f32 compute  |
//      +----------------+
//          |
//          v
//      +----------------------+
//      | intermediate ops     |  (scale mul, clamp, ...)
//      +----------------------+
//          |
//          v
//      +-----------------------------+
//      |  QuantizeLinear (generic)   |  <-- quantOp
//      |  int8 = round(fp32/scale_out) + zp_out, clamp(min,max) |
//      +-----------------------------+
//          |
//          v
//      int8 tensor
//
// 2. QLinear / quantized-compute rescale chain
//
//      int8 input
//          |
//          v
//      +-------------------------------+
//      | quantized compute op          |  (e.g. linalg.conv_2d_nchw_fchw_q)
//      | produces i32 accumulators     |
//      +-------------------------------+
//          |
//          v
//      +----------------------+
//      | intermediate ops     |  (bias add, div-by-count, ...)
//      +----------------------+
//          |
//          v
//      +-------------------------------+
//      | DequantizeLinear (generic)    |  <-- outputDequantOp
//      | fp32 = (i32 - zp_dq) * scale_dq |
//      +-------------------------------+
//          |
//          v
//       fp32 result
//          |
//          v
//      +-------------------------------+
//      | QuantizeLinear (generic)      |  <-- quantOp
//      | int8 = round(fp32/scale_q) + zp_q, clamp(min,max) |
//      +-------------------------------+
//          |
//          v
//      int8 output

#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include <optional>

namespace mlir::syna::torq {

// Extract a scalar float constant from a value defined by arith.constant.
std::optional<double> getQFloatScalar(Value v);

// Extract a scalar float constant from a value inside a linalg.generic body,
// looking through arith.truncf / arith.extf and block arguments that map to
// constant inputs.
std::optional<double> getQGenericFloatConstant(Value v, linalg::GenericOp op);

// Canonical dequant metadata.
//
// Different quantization producers (torch-mlir, ONNX Runtime, TOSA, etc.) may
// lower dequantize to slightly different linalg.generic bodies.  Each producer
// gets its own "variant" matcher that extracts the same canonical information.
struct DequantInfo {
    double scale = 1.0;
    int32_t zp = 0;
};

// Per-variant dequant matchers.
//
// Each variant is self-contained; adding a new producer never risks breaking an
// existing one.  The unified entry point (matchDequantGeneric) tries variants
// in order and returns on the first match.

// Signed integer dequant:
//   (input - zp) * scale
// represented as either:
//   [arith.extsi] -> arith.sitofp -> arith.mulf(scale)
// or with an intermediate arith.subi(zp) before sitofp.
std::optional<DequantInfo> matchDequantSigned(linalg::GenericOp op);

// Scale-one signed integer dequant:
//   (input - zp)
// represented as:
//   [arith.extsi] -> arith.sitofp
// or with an intermediate arith.subi(zp) before sitofp.
// This appears when the scale mulf/divf has been folded away by canonicalization.
std::optional<DequantInfo> matchDequantScaleOne(linalg::GenericOp op);

// Unsigned integer dequant (ONNX INT4 QDQ with block_size):
//   (input - fp_zp) * scale
// represented as:
//   [arith.extui] -> arith.uitofp -> (arith.subf(fp_zp)) -> arith.mulf(scale)
std::optional<DequantInfo> matchDequantUnsigned(linalg::GenericOp op);

// Match any supported dequant linalg.generic.  Tries all registered variants.
bool matchDequantGeneric(linalg::GenericOp op, double &scale, int32_t &zp);

// Canonical quant metadata.
//
// Just like dequant, different producers may lower quantize to different
// linalg.generic bodies.  Each variant extracts the same canonical information.
struct QuantInfo {
    double scale = 1.0;
    double zp = 0.0;
    double min = 0.0;
    double max = 0.0;
};

// Per-variant quant matchers.

// Signed integer quant:
//   divf/mulf -> math.roundeven -> arith.addf(zp) -> arith.maximumf(min)
//   -> arith.minimumf(max) -> arith.fptosi
std::optional<QuantInfo> matchQuantSigned(linalg::GenericOp op);

// Scale-one signed integer quant:
//   math.roundeven -> arith.addf(zp) -> arith.maximumf(min) -> arith.minimumf(max)
//   -> arith.fptosi
// This appears when the scale divf/mulf has been folded away by canonicalization.
std::optional<QuantInfo> matchQuantScaleOne(linalg::GenericOp op);

// Walk backward from `v` through a quant clamp pair (maxf/minf), extracting the
// constant bounds and the value that feeds the rest of the quant body.
// Both orderings are accepted.  Exposed so other patterns can reuse it.
bool extractQuantClampBounds(Value v, linalg::GenericOp op, double &min, double &max, Value &data);

// Match any supported quant linalg.generic.  Tries all registered variants.
bool matchQuantGeneric(linalg::GenericOp op, double &scale, double &zp, double &min, double &max);

// Compute a (multiplier, shift) pair that approximates `scale` as
//   scale ~= multiplier / 2^shift
// The Torq hardware requires the shift amount to be a multiple of 4, so we
// choose the largest multiple-of-4 shift whose rounded multiplier still fits
// in a signed 32-bit integer.  Returns false for negative scales (which cannot
// be represented) or if no shift fits.
bool computeMultiplierAndShift(double scale, int32_t &multiplier, int32_t &shift);

// Canonical quantization state for a compute op surrounded by QDQ or QLinear
// rescale ops.  Patterns populate this and then share the rescale math and
// chain erasure logic.
//
// Two common shapes are supported:
//   - QDQ:  DQ -> f32 compute op -> Q
//           inputDequantOp is set; outputDequantOp is null.
//   - QLinear rescale:  quantized compute op -> DQ -> Q
//           inputDequantOp is null; outputDequantOp is set.
struct QuantizedOpChain {
    DequantInfo inputInfo;
    DequantInfo outputDequantInfo;
    QuantInfo outputInfo;
    linalg::GenericOp inputDequantOp = nullptr;
    linalg::GenericOp outputDequantOp = nullptr;
    linalg::GenericOp quantOp = nullptr;
    double extraScale = 1.0;

    bool hasInputDQ() const { return inputDequantOp != nullptr; }
    bool hasOutputDQ() const { return outputDequantOp != nullptr; }
    bool hasQuant() const { return quantOp != nullptr; }

    // Effective input scale for the compute op.  For QDQ this is the input DQ
    // scale; for QLinear-style rescale it is the output DQ scale.
    double inputScale() const { return hasOutputDQ() ? outputDequantInfo.scale : inputInfo.scale; }

    // Effective input zero-point for the compute op.  For QDQ this is the input
    // DQ zero-point; for QLinear-style rescale it is the output DQ zero-point.
    int32_t inputZp() const { return hasOutputDQ() ? outputDequantInfo.zp : inputInfo.zp; }
};

// Match a dequant generic that produces `candidate`.  On success populates
// `chain.inputInfo` and `chain.inputDequantOp`.  The caller is responsible for
// walking through any allowed intermediate ops (pad, transpose, etc.) to obtain
// `candidate`.
LogicalResult
matchInputQuantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain);

// Match a dequant generic that consumes `candidate`.  On success populates
// `chain.outputDequantInfo` and `chain.outputDequantOp`.  This is the
// QLinear-style output rescale DQ; the caller is responsible for matching the
// final quant generic (and for handling any ops between the DQ and Q).
LogicalResult
matchOutputDequantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain);

// Match a quant generic that consumes `candidate`.  On success populates
// `chain.outputInfo` and `chain.quantOp`.  The caller handles intermediate ops
// (bias add, scale generic, div-by-count, etc.) between the compute op and
// the quant generic.
LogicalResult
matchOutputQuantization(Value candidate, PatternRewriter &rewriter, QuantizedOpChain &chain);

// Deferred constant-resolution helpers.
//
// These builders emit the zero-point correction / scale_bias chains as
// Host-marked linalg ops instead of folding them eagerly in the pattern.  The
// Host-marked ops survive the pre-conversion target and are resolved later by
// CompileTimeConstOutlinePass / CompileTimeConstComputePass (the "JIT"
// constant pipeline).  Shared by the quantized weighted-op patterns
// (QConv2D, QMatmul, ...).

// Extract a scalar i32 constant from a value defined by arith.constant.
std::optional<int32_t> getScalarI32Const(Value v);

// Compute -inputZp * sum(adjustedWeights) per output channel.  adjustedWeights
// are already sign-adjusted for weight_zp, so this term cancels the input_zp
// offset introduced by keeping the input tensor in its as-quantized (unsigned)
// form.  Works for any rank: all non-output-channel dims are reduced.
Value computeInputZpCorrection(Value adjustedWeights, int32_t inputZp, PatternRewriter &rewriter);

// Add two per-channel i32 bias tensors (Host-marked, folded at compile time).
Value addPerChannelBias(Value lhs, Value rhs, PatternRewriter &rewriter);

// Build a {C*2} i32 scale_bias tensor by interleaving a dynamic per-channel
// i32 bias (shape {C}) with a scalar multiplier.  The result layout is
// [bias_0, mult, bias_1, mult, ...].
Value buildDynamicInterleavedBiasScale(
    Value bias, int32_t multiplier, Location loc, PatternRewriter &rewriter
);

} // namespace mlir::syna::torq
