// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Quantization pattern utilities
//
// ONNX quantized models lower to linalg.generic ops whose body encodes the
// dequant / quant arithmetic (arithemtics for int→float conversion, clamp
// bounds, scale rounding, etc.).  Different ONNX opset versions and integer
// representations (signed vs unsigned) produce slightly different IR shapes.
//
// Architecture: multi-flavor dispatch.
//
// Each supported IR shape is a standalone "flavor" matcher that returns a
// canonical info struct (DequantInfo or QuantInfo) on success, nullopt on
// mismatch.  The public entry points (matchDequantGeneric, matchQuantGeneric)
// iterate over registered flavors and return on the first match.  Adding
// support for a new IR shape is one function + one entry in the flavor array.
//
// Flavor matchers use raw function pointers (DequantFlavorFn, QuantFlavorFn)
// rather than std::function to avoid heap allocation in the critical path.

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
// gets its own "flavor" matcher that extracts the same canonical information.
struct DequantInfo {
    double scale = 1.0;
    int32_t zp = 0;
};

// Per-flavor dequant matchers.
//
// Each flavor is self-contained; adding a new producer never risks breaking an
// existing one.  The unified entry point (matchDequantGeneric) tries flavors
// in order and returns on the first match.

// Signed integer dequant:
//   (input - zp) * scale
// represented as either:
//   [arith.extsi] -> arith.sitofp -> arith.mulf(scale)
// or with an intermediate arith.subi(zp) before sitofp.
std::optional<DequantInfo> matchDequantSigned(linalg::GenericOp op);

// Unsigned integer dequant (ONNX INT4 QDQ with block_size):
//   (input - fp_zp) * scale
// represented as:
//   [arith.extui] -> arith.uitofp -> (arith.subf(fp_zp)) -> arith.mulf(scale)
std::optional<DequantInfo> matchDequantUnsigned(linalg::GenericOp op);

// Match any supported dequant linalg.generic.  Tries all registered flavors.
bool matchDequantGeneric(linalg::GenericOp op, double &scale, int32_t &zp);

// Canonical quant metadata.
//
// Just like dequant, different producers may lower quantize to different
// linalg.generic bodies.  Each flavor extracts the same canonical information.
struct QuantInfo {
    double scale = 1.0;
    double zp = 0.0;
    double min = 0.0;
    double max = 0.0;
};

// Per-flavor quant matchers.

// Signed integer quant:
//   divf/mulf -> math.roundeven -> arith.addf(zp) -> arith.maximumf(min)
//   -> arith.minimumf(max) -> arith.fptosi
std::optional<QuantInfo> matchQuantSigned(linalg::GenericOp op);

// Match any supported quant linalg.generic.  Tries all registered flavors.
bool matchQuantGeneric(linalg::GenericOp op, double &scale, double &zp, double &min, double &max);

// Compute a (multiplier, shift) pair that approximates `scale` as
//   scale ~= multiplier / 2^shift
// The Torq hardware requires the shift amount to be a multiple of 4, so we
// choose the largest multiple-of-4 shift whose rounded multiplier still fits
// in a signed 32-bit integer.  Returns false for negative scales (which cannot
// be represented) or if no shift fits.
bool computeMultiplierAndShift(double scale, int32_t &multiplier, int32_t &shift);

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
