// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// DeriveQuantParamsPattern
// ------------------------
// Lower torq_hl.derive_quant_params to a short sequence of slice_tasks inside
// one program body. The op consumes the reduced min_x/max_x scalars and
// produces the per-tensor DynamicQuantize scale/zero-point.
//
// `$init` is one <N x bf16> scratch+output blob: each task reads/writes fixed
// slots of it, addressed by an LData built from the op.getInit() Value and then
// indexed by slot (see slotView). Later tasks read slots earlier tasks wrote —
// safe because the tasks execute sequentially within the single program.
// bf16, i16 and i8 intermediates share slots by bit-reinterpret
// (re-typing an LData just re-addresses the same bytes: bf16 slot k, i16 slot k,
// and the low byte i8 slot 2k all alias).
//
// Slot layout (bf16 element index unless noted). The results lead, scratch follows:
//   [0] inv_scale (result)     [5] neg_min_adj (kept for zp)
//   [1] zp (result)            [6] expo   = 0x7E80 - x            (i16)
//   [2] scale (result)         [7] lutVal16 = zext(lutVal8)       (i16, [6]+1)
//   [3] neg_min (= -min_x)     [8] lutVal8 = LUT[mant8]           (i8 @ byte 16)
//   [4] max_adj                [9] recip = bitcast(expo+lutVal16) (bf16)
//                             [10] range_safe (recip in)
//
// Tasks (each 1 ALU pass + 1 ACT pass + 1 store):
//   [4] max_adj     = clamp(max_x, 0, +inf)
//   [3] neg_min     = -min_x                              (ACT NEG, full range)
//   [5] neg_min_adj = clamp([3], 0, +inf)
//  [10] range_safe  = clamp([4] + [5], eps, +inf)         (2-elt accumulate)
//   reciprocal (full-range exponent+mantissa, bit-identical to the linalg
//   BfloatReciprocalPattern, DecomposeLinalgOpsPattern.cpp:446):
//   [6] expo        = 0x7E80 - x   with x = i16 bits of range_safe   (ACT affine)
//   [8] lutVal8     = mantissaLUT[x & 0xFF]                          (ACT LUT)
//   [7] lutVal16    = zext(lutVal8)
//   [9] recip       = bitcast<bf16>(expo + lutVal16)                 (2-elt accumulate)
//   [0] inv_scale   = recip * 255                                    (WRAM const mul_weight[0])
//   [2] scale       = range_safe * (1/255)                           (WRAM const mul_weight[1])
//   [1] zp          = neg_min_adj * inv_scale                        (two runtime bf16 scalars)

#include "Patterns.h"
#include "torq/Utils/Kernel.h"
#include "torq/Utils/TorqUtils.h"

#include "llvm/Support/Debug.h"

#include <cstdint>
#include <cstring>

#define DEBUG_TYPE "torq-lower-torqhl"

namespace mlir::syna::torq {

namespace {

// Blob slot layout. The op's results occupy the leading slots; the rest is kernel scratch.
enum Slot {
    INV_SCALE = 0, // result
    ZP = 1,        // result
    SCALE = 2,     // result
    NEG_MIN = 3,
    MAX_ADJ = 4,
    NEG_MIN_ADJ = 5,
    EXPO = 6,     // i16
    LUTVAL16 = 7, // i16 (contiguous with EXPO for the 2-elt recombine)
    LUTVAL8 = 8,  // i8 (low byte only)
    RECIP = 9,    // bf16
    RANGE_SAFE = 10,
    BLOB_SLOTS = 11
};

// Slots of the {2}:bf16 mul_weight constant operand.
enum MulWeight {
    W_255 = 0,    // 255.0, the uint8 range      -> inv_scale = recip * 255
    W_INV_255 = 1 // 1/255                       -> scale = range_safe / 255
};

// One-sided divide-by-zero guard on the range: scale = range/255 stays finite for an
// all-equal input.
constexpr float kRangeEps = 1e-12f;

// The reciprocal magic constant from BfloatReciprocalPattern: recip bits =
// (0x7E80 - x) + mantissaLUT[x & 0xFF], computed on the i16 bit-pattern of x.
constexpr int32_t kReciprocalExpConst = 0x7E80;

// Bit pattern of a float32, for the act clamp bounds.
static int32_t f32Bits(float f) {
    int32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

// 256-entry packed mantissa LUT: a 128-entry base, duplicated to 256, with the
// per-index correction lut[i] += i%128 done in int8_t (wrapping), then sign-extended
// and shifted left 8 into an i32 word.
static SmallVector<int32_t> buildReciprocalPackedLut() {
    static const int8_t base128[128] = {
        (int8_t)-0x80, 0x7E, 0x7C, 0x7A, 0x78, 0x76, 0x75, 0x73, 0x71, 0x6F, 0x6D, 0x6C, 0x6A,
        0x68,          0x67, 0x65, 0x64, 0x62, 0x60, 0x5F, 0x5D, 0x5C, 0x5A, 0x59, 0x58, 0x56,
        0x55,          0x53, 0x52, 0x51, 0x4F, 0x4E, 0x4D, 0x4C, 0x4A, 0x49, 0x48, 0x47, 0x45,
        0x44,          0x43, 0x42, 0x41, 0x40, 0x3F, 0x3D, 0x3C, 0x3B, 0x3A, 0x39, 0x38, 0x37,
        0x36,          0x35, 0x34, 0x33, 0x32, 0x31, 0x30, 0x2F, 0x2E, 0x2D, 0x2C, 0x2C, 0x2B,
        0x2A,          0x29, 0x28, 0x27, 0x26, 0x25, 0x25, 0x24, 0x23, 0x22, 0x21, 0x21, 0x20,
        0x1F,          0x1E, 0x1E, 0x1D, 0x1C, 0x1B, 0x1B, 0x1A, 0x19, 0x18, 0x18, 0x17, 0x16,
        0x16,          0x15, 0x14, 0x14, 0x13, 0x12, 0x12, 0x11, 0x10, 0x10, 0x0F, 0x0E, 0x0E,
        0x0D,          0x0D, 0x0C, 0x0B, 0x0B, 0x0A, 0x0A, 0x09, 0x09, 0x08, 0x07, 0x07, 0x06,
        0x06,          0x05, 0x05, 0x04, 0x04, 0x03, 0x03, 0x02, 0x02, 0x01, 0x01
    };
    SmallVector<int32_t> packed(256);
    for (int i = 0; i < 256; ++i) {
        int8_t v = static_cast<int8_t>(base128[i % 128] + static_cast<int8_t>(i % 128));
        packed[i] = static_cast<int32_t>(v) << 8;
    }
    return packed;
}

// Index the `count` elements of type `dt` starting at element `slot` of the 1-D tensor behind
// `value`. The LData is built from the MLIR Value, so shape and strides come from the actual
// tensor, and `slot` then indexes into it (so it is range-checked against the tensor extent).
static LData slotView(Value value, DType dt, int slot, int count = 1) {
    LData data(value);
    data.bitCast(dt);
    return data.subviewDim(0, slot, count);
}

// Emit one scalar clamp slice_task: out[outSlot] = clamp(mode(in[inSlot]), lo, hi).
static void emitScalarClamp(
    PatternRewriter &rewriter, Location loc, StringRef name, Value src, int inSlot, DType inDt,
    Value dst, int outSlot, DType outDt, int loClip, int hiClip, torq_hw::ACTMode mode
) {
    Slice slice(name.str());
    IData idata = slice.iram.load(slotView(src, inDt, inSlot));
    PData pdata = slice.alu.load(idata);
    QData res = slice.act.clamp(pdata, loClip, hiClip, mode);
    slice.store(slotView(dst, outDt, outSlot), res);
    slice.createSliceTaskOp(rewriter, loc);
}

// Emit out[outSlot] = clamp(sum(in[aSlot], in[aSlot+1]), lo, hi) as a 2-element
// accumulate over the contiguous pair (aSlot, aSlot+1).
// `dt` selects float (bf16 range_safe) vs integer (i16 reciprocal recombine).
static void emitScalarSumClamp(
    PatternRewriter &rewriter, Location loc, StringRef name, Value blob, int aSlot, DType dt,
    int outSlot, int loClip, int hiClip
) {
    Slice slice(name.str());
    LData inData = slotView(blob, dt, aSlot, /*count=*/2);
    PData pdata;
    For(auto r = slice.iterate(2)) {
        IData idata = slice.iram.load(inData[r]);
        pdata = slice.alu.accumulate(idata, torq_hw::ALUOp1Mode::ACC);
    }
    QData res = slice.act.clamp(pdata, loClip, hiClip, torq_hw::ACTMode::ACT);
    slice.store(slotView(blob, dt, outSlot), res);
    slice.createSliceTaskOp(rewriter, loc);
}

// Emit out[outSlot] = (in[inSlot] + bias) * scale, with bias/scale from a {2}:i32
// scale_bias operand (bdata[0]=bias, bdata[1]=scale).
// Optionally applies a piecewise-linear ACT LUT (the reciprocal mantissa
// table). Integer-only path (shift=0, no rounding, full i32 clip).
static void emitScalarRescale(
    PatternRewriter &rewriter, Location loc, StringRef name, Value src, int inSlot, DType inDt,
    Value scaleBias, Value dst, int outSlot, DType outDt, ArrayRef<int32_t> lut
) {
    Slice slice(name.str());
    if (!lut.empty()) {
        ArrayRef<int32_t> lutRef = lut;
        slice.act.setLUT(lutRef);
    }
    BData bdata = slice.bram.load(LData(scaleBias));
    IData idata = slice.iram.load(slotView(src, inDt, inSlot));
    PData pdata = slice.alu.load(idata);
    QData res =
        slice.act.rescaleClamp(pdata, bdata, 0, 0, minVal(DType::int32), maxVal(DType::int32));
    slice.store(slotView(dst, outDt, outSlot), res);
    slice.createSliceTaskOp(rewriter, loc);
}

// Emit out[outSlot] = clamp(d[dSlot] * w[wSlot], lo, hi) as a bf16 scalar product
// (D through IRAM, W through WRAM).
// `wOperand` is either a constant weight buffer (the 255.0 mul_weight) or another blob slot
// (inv_scale).
static void emitScalarMul(
    PatternRewriter &rewriter, Location loc, StringRef name, Value dOperand, int dSlot,
    Value wOperand, int wSlot, Value dst, int outSlot, int loClip, int hiClip
) {
    Slice slice(name.str());
    IData idata = slice.iram.load(slotView(dOperand, DType::bf16, dSlot));
    WData wdata = slice.wram.load(slotView(wOperand, DType::bf16, wSlot));
    PData pdata = slice.alu.elementwiseProductAccumulate(idata, wdata);
    QData res = slice.act.clamp(pdata, loClip, hiClip, torq_hw::ACTMode::ACT);
    slice.store(slotView(dst, DType::bf16, outSlot), res);
    slice.createSliceTaskOp(rewriter, loc);
}

} // namespace

template <>
LogicalResult DeriveQuantParamsPattern::transform(
    torq_hl::DeriveQuantParamsOp op, PatternRewriter &rewriter
) const {
    Location loc = op.getLoc();
    Value blob = op.getInit();
    const int32_t posInf = 0x7f800000;
    const int32_t negInf = static_cast<int32_t>(0xff800000);
    const int32_t zero = f32Bits(0.0f);
    const int32_t eps = f32Bits(kRangeEps);
    const int32_t iMin = minVal(DType::int32);
    const int32_t iMax = maxVal(DType::int32);
    using M = torq_hw::ACTMode;
    using D = DType;

    SmallVector<int32_t> recipLut = buildReciprocalPackedLut();

    // ---- scalar param math (bf16, off the x path) ----
    // [4] max_adj = clamp(max_x, 0, +inf)
    emitScalarClamp(
        rewriter, loc, "dqp_max_adj", op.getMaxX(), 0, D::bf16, blob, MAX_ADJ, D::bf16, zero,
        posInf, M::ACT
    );
    // [3] neg_min = -min_x  (full-range negate)
    emitScalarClamp(
        rewriter, loc, "dqp_neg_min", op.getMinX(), 0, D::bf16, blob, NEG_MIN, D::bf16, negInf,
        posInf, M::NEG
    );
    // [5] neg_min_adj = clamp([3], 0, +inf)
    emitScalarClamp(
        rewriter, loc, "dqp_neg_min_adj", blob, NEG_MIN, D::bf16, blob, NEG_MIN_ADJ, D::bf16, zero,
        posInf, M::ACT
    );
    // [10] range_safe = clamp([4] + [5], eps, +inf)  (MAX_ADJ, NEG_MIN_ADJ are contiguous)
    emitScalarSumClamp(
        rewriter, loc, "dqp_range_safe", blob, MAX_ADJ, D::bf16, RANGE_SAFE, eps, posInf
    );

    // ---- reciprocal: recip = 1/range_safe, full-range exponent+mantissa ----
    // [6] expo = 0x7E80 - x, with x = i16 bits of range_safe:
    //     (x + (-0x7E80)) * (-1) via the scalar-affine ACT path.
    emitScalarRescale(
        rewriter, loc, "dqp_recip_expo", blob, RANGE_SAFE, D::int16, op.getSubScaleBias(), blob,
        EXPO, D::int16, /*lut=*/{}
    );
    // [8] lutVal8 = mantissaLUT[x & 0xFF]: the low byte of range_safe (i8 slot 2*RANGE_SAFE)
    //     indexes the 256-entry packed LUT; scale_bias {-128, 128} decodes the base<<8 word.
    emitScalarRescale(
        rewriter, loc, "dqp_recip_lut", blob, 2 * RANGE_SAFE, D::int8, op.getTableScaleBias(), blob,
        2 * LUTVAL8, D::int8, /*lut=*/recipLut
    );
    // [7] lutVal16 = zext(lutVal8): read the LUT byte unsigned, widen to i16.
    emitScalarClamp(
        rewriter, loc, "dqp_recip_zext", blob, 2 * LUTVAL8, D::uint8, blob, LUTVAL16, D::int16,
        iMin, iMax, M::ACT
    );
    // [9] recip = bitcast<bf16>(expo + lutVal16)  (EXPO, LUTVAL16 are contiguous i16)
    emitScalarSumClamp(rewriter, loc, "dqp_recip", blob, EXPO, D::int16, RECIP, iMin, iMax);

    // ---- results ----
    // [0] inv_scale = recip * 255   (WRAM constant 255.0)
    emitScalarMul(
        rewriter, loc, "dqp_inv_scale", blob, RECIP, op.getMulWeight(), W_255, blob, INV_SCALE,
        negInf, posInf
    );
    // [2] scale = range_safe * (1/255)   (WRAM constant 1/255)
    // NOTE:
    // Computed straight from range_safe, so it is ONNX's exact y_scale and carries none of the
    // reciprocal's error. It is therefore NOT the exact inverse of [0]: inv_scale is a bf16
    // LUT approximation of 255/range_safe, so scale * inv_scale = 1 + eps with eps the LUT's
    // relative error (~1 bf16 ULP). A dequantize using scale carries that eps as a systematic
    // relative error. Deriving scale from inv_scale instead would need a second LUT reciprocal
    // and trade eps for eps'.
    emitScalarMul(
        rewriter, loc, "dqp_scale", blob, RANGE_SAFE, op.getMulWeight(), W_INV_255, blob, SCALE,
        negInf, posInf
    );
    // [1] zp = neg_min_adj * inv_scale   (two runtime bf16 scalars, both blob slots)
    emitScalarMul(
        rewriter, loc, "dqp_zp", blob, NEG_MIN_ADJ, blob, INV_SCALE, blob, ZP, negInf, posInf
    );

    rewriter.eraseOp(op);
    return success();
}

} // namespace mlir::syna::torq
