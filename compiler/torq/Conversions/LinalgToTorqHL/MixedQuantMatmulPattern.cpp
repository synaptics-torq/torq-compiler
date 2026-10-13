// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Dialect/TorqHW/TorqHWInfo.h"
#include "torq/Utils/TorqHw.h"

#include "mlir/IR/DialectResourceBlobManager.h"

#include "torq/Utils/ComputeConstants.h"
#include "torq/Utils/ConversionUtils.h"

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MathExtras.h"

#include <cmath>
#include <limits>
#include <optional>

#define DEBUG_TYPE "linalg-torq-matmul-pattern"

static llvm::cl::opt<bool> clDisableMixedQuantConversion(
    "torq-disable-mixed-quant-raise", llvm::cl::desc("Disable the mixed-quant matmul conversion"),
    llvm::cl::init(false)
);

namespace mlir::syna::torq {

// ============================================================================
// Mixed-quant matmul: convert a (tiled) batch_matmul over a block-dequantized
// weight into torq_hl.mixed_quant_matmul, so the int weights stream to the
// kernel's cast-on-load path instead of being dequantized into LRAM.
//
// Matched chain (as produced by the importer and preserved by tiling):
//   batch_matmul(x[1,1,K], expand(collapse(
//       dequant_generic(wq[nB,B,tn] int, scale[nB,1,tn], zp))))
// wq/scale/zp may be constants or extract_slices of constants (the tiled
// form); the pattern reads the PARENT constants' bytes, builds the full-width
// combined wscale table [2nB+1, N] = scale | -zp*scale | ones (or [nB, N] for
// zp = 0) and the lane-replicated activation once, outside the tile loop, and
// slices both with the tile's own offset. The parent constants are marked
// torq-split-constant so SplitConstantsPass materializes each tile's window
// after unrolling (the parents die with the dequant, so splitting is free).
// ============================================================================

namespace {

// MQMM alignment contract -- single source of truth for the paddings in the
// mark and materialize paths below.
// The kernel (MixedQuantMatMulPattern) drains kMqmmDrain bf16 output columns
// per pass and pads ragged tiles to whole groups; the raise must size
// wscale/scratch with that same width (the kernel asserts the equality).
// TileAndFuse splits the N window uniformly across HwInfo::slice_count
// tiles, so at mark time the window must be a multiple of
// kMqmmWindowAlign; at materialize time the tile offset/width are checked
// against kMqmmDrain and the wscale table bounds. If tiling ever stops being
// a uniform slice_count-way split, the checks fail the match and the matmul
// falls back to the generic dequant path.
constexpr int64_t kMqmmDrain = HwInfo::act_width / 2;                  // 8
constexpr int64_t kMqmmWindowAlign = kMqmmDrain * HwInfo::slice_count; // 16

// bf16 <-> float helpers. Blob bytes are little-endian per the ONNX raw_data
// spec (our producers), so read the 16-bit pattern explicitly LE rather than
// assuming the host is LE. The combined wscale table is written back as a
// native uint16_t array through getFromRawBuffer(), which is host-native on
// both sides and needs no conversion.
float bf16ToFloat(const char *p) {
    uint16_t bits = llvm::support::endian::read16le(reinterpret_cast<const uint8_t *>(p));
    return APFloat(APFloat::BFloat(), APInt(16, bits)).convertToFloat();
}

uint16_t floatToBf16(float f) {
    APFloat a(f);
    bool losesInfo;
    a.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &losesInfo);
    return (uint16_t)a.bitcastToAPInt().getZExtValue();
}

// One i4 element of a raw byte buffer, sign-extended. Used for resource
// blobs only. The layout is a producer contract (ONNX raw_data, as emitted
// by the importers): two elements per byte, first element in the LOW nibble,
// signed. The one-per-byte case matches the inline layout MLIR uses (8-bit
// storage per sub-byte element) and is kept as a defensive case for
// importers that unpack. Which case applies is decided ONCE by
// getMqmmConst from the pinned blob size (it sets i4Packed) -- the nibble
// order itself carries no metadata in the blob, so the convention is pinned
// on the producer side (the importers emit exactly this layout; the
// branch's gemma q4 testdata case validates it end-to-end).
int8_t i4At(ArrayRef<char> bytes, bool packed, int64_t i) {
    uint8_t v = packed ? (uint8_t)bytes[i / 2] >> (4 * (i & 1)) & 0xF : (uint8_t)bytes[i] & 0xF;
    return (int8_t)(v >= 8 ? v - 16 : v);
}

// An operand that is either a constant or an extract_slice of one: the
// parent constant value, an element reader, and the tile's offset along the
// last (N) dim.
//
// Element access goes through the official API per backing:
//  - inline DenseElementsAttr: getValues<APInt/APFloat>() — layout-agnostic
//    iterators (getRawData() is documented as internal storage, and the
//    inline byte order of multi-byte elements is host-native);
//  - resource blobs: MLIR defines no element layout for them (the base
//    builder does no size checking and there is no
//    DenseResourceElementsAttrBase<T> for i4/bf16), so the bytes are
//    interpreted directly. The layout is the producer's — in our flow it is
//    ONNX raw_data verbatim: fixed-width little-endian, i4 packed two per
//    byte with the first element in the low nibble. getMqmmConst pins the
//    expected blob sizes.
struct SlicedConstOperand {
    Value parent;
    std::optional<DenseIntOrFPElementsAttr> dense; // inline backing
    ArrayRef<char> bytes;                          // blob backing, if !dense
    int64_t count = 0;                             // parent element count
    bool splat = false;                            // inline only
    bool i4Packed = false;                         // blob only: 2 i4 per byte
    OpFoldResult offN;

    Type elemType() const { return cast<ShapedType>(parent.getType()).getElementType(); }

    // Element i (row-major over the parent) as a float. Supports i4/i8/bf16.
    float at(int64_t i) const {
        const int64_t idx = splat ? 0 : i;
        Type et = elemType();
        if (dense) {
            if (et.isBF16()) {
                auto v = dense->getValues<APFloat>();
                return (*(v.begin() + idx)).convertToFloat();
            }
            assert(et.isInteger(8) || et.isInteger(4));
            auto iv = dense->getValues<APInt>();
            return (float)(*(iv.begin() + idx)).getSExtValue();
        }
        // Blob: ONNX raw_data layout (see i4At / bf16ToFloat).
        if (et.isBF16())
            return bf16ToFloat(bytes.data() + 2 * idx);
        if (et.isInteger(8))
            return (float)(int8_t)bytes[idx];
        assert(et.isInteger(4));
        return (float)i4At(bytes, i4Packed, idx);
    }

    // True if any element is non-zero (the hasZp probe). Element-wise for
    // both backings, so the i8 -128 / i4 high-nibble-8 / odd-count cases the
    // old uint16 word scan missed are handled.
    bool anyNonZero() const {
        if (dense) {
            if (elemType().isBF16())
                return llvm::any_of(dense->getValues<APFloat>(), [](APFloat f) {
                    return !f.isZero();
                });
            return llvm::any_of(dense->getValues<APInt>(), [](APInt v) { return !v.isZero(); });
        }
        if (elemType().isBF16()) {
            // Little-endian words per the ONNX spec; the sign bit may carry
            // -0.0, which is numerically zero.
            const uint8_t *raw = reinterpret_cast<const uint8_t *>(bytes.data());
            for (size_t p = 0; p + 1 < bytes.size(); p += 2) {
                if (llvm::support::endian::read16le(raw + p) & 0x7fff)
                    return true;
            }
            return false;
        }
        return llvm::any_of(bytes, [](char c) { return c != 0; });
    }
};

// Element reader for the parent constant of a sliced operand (see
// SlicedConstOperand for the per-backing layout contracts).
bool getMqmmConst(Value v, SlicedConstOperand &out) {
    Attribute a;
    if (!matchPattern(v, m_Constant(&a)))
        return false;
    out.count = cast<ShapedType>(v.getType()).getNumElements();
    if (auto d = dyn_cast<DenseIntOrFPElementsAttr>(a)) {
        out.dense = d;
        out.splat = d.isSplat();
        return true;
    }
    if (auto r = dyn_cast<DenseResourceElementsAttr>(a)) {
        if (auto *blob = r.getRawHandle().getBlob()) {
            out.bytes = blob->getData();
            // Pin the producer contract (ONNX raw_data, as emitted by the
            // importers): i4 packed (count/2) or unpacked (count), i8 count,
            // bf16 2*count. A foreign producer fails the match instead of
            // corrupting weights. The size pinning also decides the i4
            // packing layout (stored in i4Packed); the remaining convention
            // details (low-nibble-first i4, little-endian bf16) carry no
            // metadata in the blob itself; they are pinned on the producer
            // side (the branch's gemma q4 testdata case validates the
            // convention end-to-end) and the bf16 side is additionally
            // validated by the normal-range sweep in the wscale
            // construction.
            int64_t bw = cast<ShapedType>(v.getType()).getElementTypeBitWidth();
            int64_t sz = (int64_t)out.bytes.size();
            bool ok = bw == 4    ? (sz == out.count / 2 || sz == out.count)
                      : bw == 8  ? sz == out.count
                      : bw == 16 ? sz == 2 * out.count
                                 : false;
            if (!ok)
                return false;
            out.i4Packed = (bw == 4 && sz == out.count / 2);
            return true;
        }
    }
    return false;
}

// Sum of two index offsets. Folds constants and emits a plain affine.apply
// for dynamic ones (no makeComposedFoldedAffineApply: composing producer maps
// can assert on the shapes the fit probes build).
OpFoldResult addIndexOfrs(OpBuilder &b, Location loc, OpFoldResult a, OpFoldResult c) {
    auto aC = getConstantIntValue(a);
    auto cC = getConstantIntValue(c);
    if (cC && *cC == 0)
        return a;
    if (aC && *aC == 0)
        return c;
    if (aC && cC)
        return b.getIndexAttr(*aC + *cC);
    AffineExpr d0, d1;
    bindDims(b.getContext(), d0, d1);
    return affine::AffineApplyOp::create(
               b, loc, AffineMap::get(2, 0, d0 + d1),
               ValueRange{
                   getValueOrCreateConstantIndexOp(b, loc, a),
                   getValueOrCreateConstantIndexOp(b, loc, c)
               }
    )
        .getResult();
}

// Walks a chain of extract_slices (multi-level tiling nests them) down to the
// constant, requiring each level to be a full-height column window, and
// composes the per-level offsets into one total column offset.
FailureOr<SlicedConstOperand> resolveSlicedConst(Value v, OpBuilder &b, Location loc) {
    SlicedConstOperand r;
    r.offN = b.getIndexAttr(0);
    while (auto slice = v.getDefiningOp<tensor::ExtractSliceOp>()) {
        auto srcTy = cast<RankedTensorType>(slice.getSource().getType());
        auto offs = slice.getMixedOffsets();
        for (size_t i = 0; i + 1 < offs.size(); ++i) {
            if (!isConstantIntValue(offs[i], 0))
                return failure();
        }
        auto sizes = slice.getStaticSizes();
        for (int64_t i = 0; i + 1 < (int64_t)sizes.size(); ++i) {
            if (sizes[i] != srcTy.getDimSize(i))
                return failure();
        }
        r.offN = addIndexOfrs(b, loc, r.offN, offs.back());
        v = slice.getSource();
    }
    r.parent = v;
    if (!getMqmmConst(r.parent, r))
        return failure();
    return r;
}

// One zero point of the parent constant, as a float. Zero points are stored
// per block and per column, and may be bf16, i8 or i4.
float zeroPointAt(const SlicedConstOperand &zp, int64_t zpW, int64_t g, int64_t col) {
    return zp.at(g * zpW + (col % zpW));
}

// Widen a row-major constant's innermost dim from N to N8 with zeros.
Value padConstToWidth(
    const SlicedConstOperand &c, int64_t N, int64_t N8, OpBuilder &b, Location loc
) {
    auto ty = cast<RankedTensorType>(c.parent.getType());
    SmallVector<int64_t> shape(ty.getShape());
    shape.back() = N8;
    auto newTy = RankedTensorType::get(shape, ty.getElementType());
    if (c.splat) {
        auto attr =
            cast<DenseElementsAttr>(cast<arith::ConstantOp>(c.parent.getDefiningOp()).getValue());
        return arith::ConstantOp::create(b, loc, attr.resizeSplat(newTy));
    }
    const int64_t rows = c.count / N;
    if (c.dense) {
        // Official value-based construction: read through the element
        // iterators, append zero pad columns, let MLIR own the storage.
        // Copy row by row: row r starts at r * N in the source and at
        // r * N8 in the padded result.
        Type et = ty.getElementType();
        if (et.isBF16()) {
            SmallVector<APFloat> vals(rows * N8, APFloat(APFloat::BFloat()));
            auto src = c.dense->getValues<APFloat>();
            for (int64_t r = 0; r < rows; ++r)
                for (int64_t col = 0; col < N; ++col)
                    vals[r * N8 + col] = *(src.begin() + (r * N + col));
            return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(newTy, vals));
        }
        int64_t bw = et.isInteger(4) ? 4 : 8;
        SmallVector<APInt> vals(rows * N8, APInt(bw, 0));
        auto src = c.dense->getValues<APInt>();
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t col = 0; col < N; ++col)
                vals[r * N8 + col] = *(src.begin() + (r * N + col));
        return arith::ConstantOp::create(b, loc, DenseElementsAttr::get(newTy, vals));
    }
    // Blob: the only access is bytes; the row stride derives from the actual
    // byte count (i4 packed: N/2 bytes/row, or byte-per-element).
    const int64_t oldRow = (int64_t)c.bytes.size() / rows;
    const int64_t newRow = oldRow * N8 / N;
    SmallVector<char> nb(rows * newRow, 0);
    for (int64_t r = 0; r < rows; ++r)
        memcpy(&nb[r * newRow], c.bytes.data() + r * oldRow, oldRow);
    return arith::ConstantOp::create(b, loc, DenseElementsAttr::getFromRawBuffer(newTy, nb));
}

// The kernel drains kMqmmDrain output columns per pass and tiling splits
// the window uniformly across HwInfo::slice_count tiles, so N must be a
// multiple of kMqmmWindowAlign (= drain * slices; 16 on SL2610). For a
// ragged chain (e.g. a vocab-sized lm_head), rewrite it at mark time into
// the same chain over constants zero-padded to the next multiple of
// kMqmmWindowAlign plus a result slice back to N; the driver then marks the
// padded chain like any aligned one. Padded columns have scale 0, so they
// compute to 0 and are discarded by the slice.
LogicalResult padRaggedChain(
    linalg::BatchMatmulOp srcOp, linalg::GenericOp dq, Value x, const SlicedConstOperand &wq,
    const SlicedConstOperand &scale, const SlicedConstOperand *zp, int64_t nB, int64_t B,
    PatternRewriter &rw
) {
    Location loc = srcOp.getLoc();
    auto wqTy = cast<RankedTensorType>(wq.parent.getType());
    const int64_t N = wqTy.getDimSize(2);
    const int64_t N8 = llvm::alignTo(N, kMqmmWindowAlign);
    if (wqTy.getElementType().isInteger(4) && N % 2 != 0)
        return rw.notifyMatchFailure(srcOp, "cannot pad odd-width i4 rows");
    // require the unfused full-width constants as the direct dequant operands
    if (dq.getDpsInputOperand(0)->get() != wq.parent ||
        dq.getDpsInputOperand(1)->get() != scale.parent ||
        (zp && dq.getDpsInputOperand(2)->get() != zp->parent))
        return rw.notifyMatchFailure(srcOp, "ragged chain is already sliced");
    auto scTy = cast<RankedTensorType>(scale.parent.getType());
    if (scTy.getDimSize(scTy.getRank() - 1) != N)
        return rw.notifyMatchFailure(srcOp, "scale is not per-column");
    if (zp && !zp->splat) {
        auto zpTy = cast<RankedTensorType>(zp->parent.getType());
        if (zpTy.getDimSize(zpTy.getRank() - 1) != N)
            return rw.notifyMatchFailure(srcOp, "zero-point is not per-column");
    }
    auto fill = srcOp.getOutputs()[0].getDefiningOp<linalg::FillOp>();
    if (!fill)
        return rw.notifyMatchFailure(srcOp, "matmul init is not a fill");
    for (const SlicedConstOperand *c : {&wq, &scale, zp}) {
        // Inline layout is guaranteed by MLIR's element iterators; only blob
        // strides need checking.
        if (!c || c->splat || c->dense)
            continue;
        auto ty = cast<RankedTensorType>(c->parent.getType());
        int64_t rows = 1;
        for (int64_t i = 0; i + 1 < ty.getRank(); ++i)
            rows *= ty.getDimSize(i);
        const int64_t sz = (int64_t)c->bytes.size();
        if (sz % rows != 0 || (sz / rows * N8) % N != 0)
            return rw.notifyMatchFailure(srcOp, "unpaddable constant layout");
    }

    rw.setInsertionPoint(srcOp);
    SmallVector<Value> ins = {
        padConstToWidth(wq, N, N8, rw, loc), padConstToWidth(scale, N, N8, rw, loc)
    };
    if (zp)
        ins.push_back(padConstToWidth(*zp, N, N8, rw, loc));

    Type dqEt = cast<RankedTensorType>(dq.getResult(0).getType()).getElementType();
    Value dqInit = tensor::EmptyOp::create(rw, loc, ArrayRef<int64_t>{nB, B, N8}, dqEt);
    auto dqP = linalg::GenericOp::create(
        rw, loc, TypeRange{dqInit.getType()}, ins, ValueRange{dqInit}, dq.getIndexingMapsArray(),
        dq.getIteratorTypesArray()
    );
    rw.cloneRegionBefore(dq.getRegion(), dqP.getRegion(), dqP.getRegion().end());

    Value coll = tensor::CollapseShapeOp::create(
        rw, loc, dqP.getResult(0), SmallVector<ReassociationIndices>{{0, 1}, {2}}
    );
    Value expd = tensor::ExpandShapeOp::create(
        rw, loc, RankedTensorType::get({1, nB * B, N8}, dqEt), coll,
        SmallVector<ReassociationIndices>{{0, 1}, {2}}
    );
    Type yEt = cast<RankedTensorType>(srcOp.getResult(0).getType()).getElementType();
    Value yEmpty = tensor::EmptyOp::create(rw, loc, ArrayRef<int64_t>{1, 1, N8}, yEt);
    Value yInit =
        linalg::FillOp::create(rw, loc, ValueRange{fill.getInputs()[0]}, ValueRange{yEmpty})
            .getResult(0);
    auto mmP = linalg::BatchMatmulOp::create(
        rw, loc, TypeRange{yInit.getType()}, ValueRange{x, expd}, ValueRange{yInit}
    );
    // Keep the pre-assigned fuse-group UIDs alive on the padded chain (the
    // ragged originals die with it).
    auto copyId = [&](Operation *from, Operation *to) {
        if (auto a = from->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID))
            to->setAttr(TORQ_FUSE_GROUP_ID, a);
    };
    copyId(dq, dqP);
    copyId(fill, yInit.getDefiningOp());
    copyId(srcOp, mmP);

    auto one = rw.getIndexAttr(1), zero = rw.getIndexAttr(0);
    Value res = tensor::ExtractSliceOp::create(
        rw, loc, mmP->getResult(0), ArrayRef<OpFoldResult>{zero, zero, zero},
        ArrayRef<OpFoldResult>{one, one, rw.getIndexAttr(N)}, ArrayRef<OpFoldResult>{one, one, one}
    );
    rw.replaceOp(srcOp, res);
    return success();
}

// The matched block-dequant chain: the dequant generic, its resolved parent
// constants (wq/scale/zp), and the tile window. Filled by matchBlockQuantChain.
struct MqmmChain {
    linalg::GenericOp dq;
    Value x, wqV, zpV;
    SlicedConstOperand wq, scale;
    std::optional<SlicedConstOperand> zp;
    int64_t K, nB, B, N, tn, outerTn = 0;
    bool hasZp = false;
};

// Match: batch_matmul(x[1,1,K], expand(collapse(dequant(wq, scale, zp))))
// with constant wq/scale/zp. On failure records the reason on srcOp.
LogicalResult matchBlockQuantChain(linalg::BatchMatmulOp srcOp, PatternRewriter &rw, MqmmChain &c) {
    Location loc = srcOp.getLoc();
    Value x = srcOp.getInputs()[0];
    auto xTy = dyn_cast<RankedTensorType>(x.getType());
    if (!xTy || xTy.getRank() != 3 || xTy.getDimSize(0) != 1 || xTy.getDimSize(1) != 1 ||
        !xTy.getElementType().isBF16())
        return rw.notifyMatchFailure(srcOp, "x is not [1,1,K] bf16");
    const int64_t K = xTy.getDimSize(2);
    auto yTy = dyn_cast<RankedTensorType>(srcOp.getOutputs()[0].getType());
    if (!yTy || !yTy.getElementType().isBF16())
        return rw.notifyMatchFailure(srcOp, "matmul output is not bf16");

    // Two tiled shapes reach the conversion. When the chain was marked as
    // a fuse group, tiling fuses the dequant into the loop and slices its
    // OPERANDS. When it was not, the dequant stays full-width outside the
    // loop and the loop slices its RESULT: an extract_slice sits between
    // the expand and the matmul, and carries the tile's offset instead.
    Value rhs = srcOp.getInputs()[1];
    OpFoldResult outerOff;
    int64_t outerTn = 0;
    if (auto outer = rhs.getDefiningOp<tensor::ExtractSliceOp>()) {
        auto offs = outer.getMixedOffsets();
        for (size_t i = 0; i + 1 < offs.size(); ++i) {
            if (!isConstantIntValue(offs[i], 0))
                return rw.notifyMatchFailure(srcOp, "outer slice not a column window");
        }
        outerOff = offs.back();
        outerTn = outer.getStaticSizes().back();
        rhs = outer.getSource();
    }
    if (auto exp = rhs.getDefiningOp<tensor::ExpandShapeOp>())
        rhs = exp.getSrc();
    if (auto col = rhs.getDefiningOp<tensor::CollapseShapeOp>())
        rhs = col.getSrc();
    auto dq = rhs.getDefiningOp<linalg::GenericOp>();
    if (!dq || dq.getNumDpsInputs() < 2 || dq.getNumDpsInputs() > 3)
        return rw.notifyMatchFailure(srcOp, "rhs is not a 2/3-input dequant generic");
    if (!matchMatmulRhsBlockQuantizedBody(dq))
        return rw.notifyMatchFailure(srcOp, "generic body is not (weight - zero-point) * scale");
    if (dequantHasUnsignedZeroPoint(dq))
        return rw.notifyMatchFailure(srcOp, "unsigned zero-point is not supported");

    Value wqV = dq.getDpsInputOperand(0)->get();
    Value scaleV = dq.getDpsInputOperand(1)->get();
    Value zpV = dq.getNumDpsInputs() == 3 ? dq.getDpsInputOperand(2)->get() : Value();
    auto wqTy = dyn_cast<RankedTensorType>(wqV.getType());
    if (!wqTy || wqTy.getRank() != 3 ||
        !(wqTy.getElementType().isInteger(8) || wqTy.getElementType().isInteger(4)))
        return rw.notifyMatchFailure(srcOp, "weights are not [nB,B,N] i8/i4");
    const int64_t nB = wqTy.getDimSize(0), B = wqTy.getDimSize(1);
    const int64_t tn = outerTn ? outerTn : wqTy.getDimSize(2);
    if (nB * B != K)
        return rw.notifyMatchFailure(srcOp, "block shape does not cover K");
    auto wq = resolveSlicedConst(wqV, rw, loc);
    auto scale = resolveSlicedConst(scaleV, rw, loc);
    if (failed(wq) || failed(scale) || wq->splat || scale->splat)
        return rw.notifyMatchFailure(srcOp, "weights/scale are not resolvable constants");
    const int64_t N = cast<RankedTensorType>(wq->parent.getType()).getDimSize(2);
    auto scaleTy = dyn_cast<RankedTensorType>(scale->parent.getType());
    if (!scaleTy || scaleTy.getRank() != 3 || scaleTy.getDimSize(0) != nB ||
        scaleTy.getDimSize(1) != 1 || scaleTy.getDimSize(2) != N ||
        !scaleTy.getElementType().isBF16())
        return rw.notifyMatchFailure(srcOp, "scale is not [nB, 1, N] bf16");
    FailureOr<SlicedConstOperand> zp;
    Type zpEt;
    bool hasZp = false;
    if (zpV) {
        zp = resolveSlicedConst(zpV, rw, loc);
        if (failed(zp))
            return rw.notifyMatchFailure(srcOp, "zero-point is not a resolvable constant");
        zpEt = cast<RankedTensorType>(zp->parent.getType()).getElementType();
        if (!(zpEt.isInteger(8) || zpEt.isInteger(4) || zpEt.isBF16()))
            return rw.notifyMatchFailure(srcOp, "unsupported zero-point type");
        auto zpTy = cast<RankedTensorType>(zp->parent.getType());
        if (!zp->splat && (zpTy.getRank() != 3 || zpTy.getDimSize(0) != nB ||
                           zpTy.getDimSize(1) != 1 || zpTy.getDimSize(2) != N))
            return rw.notifyMatchFailure(srcOp, "zero-point is not [nB, 1, N]");
        hasZp = zp->anyNonZero();
    }

    if (outerTn) {
        // unfused shape: the outer slice carries the tile window on top of
        // whatever window the dequant operands already have
        wq->offN = addIndexOfrs(rw, loc, wq->offN, outerOff);
        scale->offN = addIndexOfrs(rw, loc, scale->offN, outerOff);
    }

    c = MqmmChain{dq, x, wqV, zpV, *wq, *scale, std::nullopt, K, nB, B, N, tn, outerTn, hasZp};
    if (zpV)
        c.zp = std::move(*zp);
    return success();
}

// Contract check (see kMqmmDrain): the table carries exactly one drain
// group of slack, and the kernel assumes a drain-aligned tile window.
// Uniform slice_count-way tiling guarantees both; if tiling ever changes
// out from under the mark-time assumption, fail the match (generic
// dequant path) instead of mis-sizing the DMA/scratch. A dynamic offset
// (workgroup-indexed tiling) is not checked here: the baseline behavior
// forms MQMM with dynamic offsets, whose alignment follows from the
// tiler's uniform stride (a follow-up hardens this to check the stride
// expression itself).
LogicalResult checkTileAlignmentContract(
    linalg::BatchMatmulOp srcOp, const MqmmChain &c, int64_t paddedTn, int64_t paddedN,
    PatternRewriter &rw
) {
    const auto offN = getConstantIntValue(c.wq.offN);
    if (offN.has_value()) {
        if (*offN < 0 || *offN % kMqmmDrain != 0 || *offN + paddedTn > paddedN)
            return rw.notifyMatchFailure(
                srcOp, "MQMM alignment contract violated: tile offset/width "
                       "not drain-aligned (uniform slice_count-way tiling "
                       "assumed at mark time)"
            );
        return success();
    }
    if (paddedTn > paddedN)
        return rw.notifyMatchFailure(
            srcOp, "MQMM alignment contract violated: tile wider than "
                   "wscale table slack"
        );
    return success();
}

// The combined scale table [wsRows, paddedN] = scale | -zp*scale | ones, built
// once from the parent bytes at the parent constant's position (outside any
// tile loop); each tile slices its own columns out of it.
struct WscaleTable {
    Value wsCombined;
    int64_t paddedTn, wsRows;
};

LogicalResult buildWscaleTable(
    linalg::BatchMatmulOp srcOp, const MqmmChain &c, PatternRewriter &rw, Location loc, Type bf16,
    WscaleTable &out
) {
    const int64_t scaleW = cast<RankedTensorType>(c.scale.parent.getType()).getDimSize(2);
    const int64_t zpW = c.zp ? cast<RankedTensorType>(c.zp->parent.getType()).getDimSize(2) : 0;
    const int64_t xLanes = kMqmmDrain;
    const int64_t paddedTn = llvm::alignTo(c.tn, xLanes);
    const int64_t paddedN = llvm::alignTo(c.N, xLanes) + xLanes;

    if (failed(checkTileAlignmentContract(srcOp, c, paddedTn, paddedN, rw)))
        return failure();

    const int64_t wsRows = c.hasZp ? 2 * c.nB + 1 : c.nB;
    // Row bands of the table: nB scale rows, then (with zp) nB -zp*scale
    // rows, then a single row of ones the kernel multiplies xsum by.
    const int64_t scaleBand = 0;
    const int64_t nzsBand = c.nB * paddedN;
    const int64_t onesBand = 2 * c.nB * paddedN;
    SmallVector<uint16_t> wsFull(wsRows * paddedN);
    for (int64_t g = 0; g < c.nB; ++g) {
        for (int64_t col = 0; col < c.N; ++col) {
            const int64_t i = g * paddedN + col;
            const float sVal = c.scale.at(g * scaleW + (col % scaleW));
            // Byte-order guard: a valid dequant scale is 0 (dead block,
            // relied on by padRaggedChain) or a normal-range float. A
            // denormal/NaN/huge reading means the values are not valid
            // little-endian bf16 (byte-swapped or corrupt producer) ->
            // fail the match instead of corrupting the table. A
            // byte-swapped blob has the right size, so it is inlined
            // upstream (InlineDenseResourcesPass) and reaches this
            // check through the inline path; inline-constant models are
            // layout-agnostic and unaffected.
            if (sVal != 0.0f && (!std::isfinite(sVal) || sVal < 1e-30f || sVal > 1e30f))
                return rw.notifyMatchFailure(
                    srcOp, "MQMM scale is not a valid normal-range value "
                           "(byte-swapped or corrupt producer)"
                );
            float zVal = 0.0f;
            if (c.hasZp) {
                zVal = zeroPointAt(*c.zp, zpW, g, col);
                if (!std::isfinite(zVal) || zVal < -1e30f || zVal > 1e30f)
                    return rw.notifyMatchFailure(
                        srcOp, "MQMM zero-point is not a valid value "
                               "(byte-swapped or corrupt producer)"
                    );
            }
            wsFull[scaleBand + i] = floatToBf16(sVal);
            if (c.hasZp)
                wsFull[nzsBand + i] = floatToBf16(-sVal * zVal);
        }
    }
    if (c.hasZp) {
        for (int64_t i = 0; i < paddedN; ++i)
            wsFull[onesBand + i] = floatToBf16(1.0f);
    }
    OpBuilder::InsertionGuard guard(rw);
    if (Operation *parentDef = c.scale.parent.getDefiningOp()) {
        rw.setInsertionPointAfter(parentDef);
    }
    Value wsCombined = arith::ConstantOp::create(
        rw, loc,
        DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get({wsRows, paddedN}, bf16),
            ArrayRef<char>((char *)wsFull.data(), wsFull.size() * 2)
        )
    );
    out = WscaleTable{wsCombined, paddedTn, wsRows};
    return success();
}

// x replicated into the drain lanes ([K] -> [K, kMqmmDrain]), built once next
// to x. torq_hl (not linalg) broadcast: under applyPartialConversion a new
// linalg op would fail legalization and roll back this whole rewrite.
Value buildLaneReplicatedX(PatternRewriter &rw, Location loc, Value x, int64_t K, Type bf16) {
    rw.setInsertionPointAfterValue(x);
    Value x1 =
        tensor::CollapseShapeOp::create(rw, loc, x, SmallVector<ReassociationIndices>{{0, 1, 2}});
    Value xInit = tensor::EmptyOp::create(rw, loc, ArrayRef<int64_t>{K, kMqmmDrain}, bf16);
    auto op = torq_hl::BroadcastOp::create(
        rw, loc, xInit.getType(), xInit, rw.getDenseI64ArrayAttr({1}), x1
    );
    return op.getOutput();
}

// The per-tile MQMM op at the matmul's position: wscale slice, scratch, a
// zero-padded init for ragged tiles, and the slice back to the tile width.
Value buildTileResult(
    linalg::BatchMatmulOp srcOp, const MqmmChain &c, const WscaleTable &table, Value xRep,
    PatternRewriter &rw, Location loc, Type bf16
) {
    rw.setInsertionPoint(srcOp);
    auto one = rw.getIndexAttr(1);
    Value wsTile = tensor::ExtractSliceOp::create(
        rw, loc, table.wsCombined, ArrayRef<OpFoldResult>{rw.getIndexAttr(0), c.wq.offN},
        ArrayRef<OpFoldResult>{rw.getIndexAttr(table.wsRows), rw.getIndexAttr(table.paddedTn)},
        ArrayRef<OpFoldResult>{one, one}
    );
    const int64_t scrSize = c.hasZp ? 2 * c.nB * table.paddedTn + 8 * c.nB : c.nB * table.paddedTn;
    Value scratch = tensor::EmptyOp::create(rw, loc, ArrayRef<int64_t>{scrSize}, bf16);
    // A tile narrower than a whole group computes into a wider zero-filled
    // destination; the extra columns are dropped from the result.
    Value init = srcOp.getOutputs()[0];
    auto initType = cast<RankedTensorType>(init.getType());
    if (table.paddedTn != c.tn) {
        SmallVector<int64_t> paddedShape(initType.getShape());
        paddedShape.back() = table.paddedTn;
        Value paddedInit = tensor::EmptyOp::create(rw, loc, paddedShape, initType.getElementType());
        // torq_hl (not linalg) fill: a new linalg op would fail
        // legalization here and roll the whole rewrite back.
        init = torq_hl::FillOp::create(rw, loc, paddedInit.getType(), paddedInit, 0).getOutput();
    }

    auto mqmm = torq_hl::MixedQuantMatMulOp::create(
        rw, loc, init.getType(), init, c.B, xRep, c.wqV, wsTile, scratch
    );

    Value result = mqmm->getResult(0);
    if (table.paddedTn != c.tn) {
        SmallVector<OpFoldResult> offsets(initType.getRank(), rw.getIndexAttr(0));
        SmallVector<OpFoldResult> strides(initType.getRank(), rw.getIndexAttr(1));
        SmallVector<OpFoldResult> sizes;
        for (int64_t dim : initType.getShape())
            sizes.push_back(rw.getIndexAttr(dim));
        result = tensor::ExtractSliceOp::create(rw, loc, result, offsets, sizes, strides);
    }
    return result;
}

struct MixedQuantMatmulRewrite final : public OpRewritePattern<linalg::BatchMatmulOp> {
    const bool _markFuseGroups;

    MixedQuantMatmulRewrite(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern<linalg::BatchMatmulOp>(context, /*benefit=*/2),
          _markFuseGroups(markFuseGroups) {}

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp srcOp, PatternRewriter &rw) const override {
        if (clDisableMixedQuantConversion)
            return rw.notifyMatchFailure(srcOp, "disabled by flag");
        if (_markFuseGroups && isMarkedFuseGroup(srcOp))
            return rw.notifyMatchFailure(srcOp, "Already marked");
        Location loc = srcOp.getLoc();

        MqmmChain c;
        if (failed(matchBlockQuantChain(srcOp, rw, c)))
            return failure();

        // Pre-tiling (MarkPatternsForTileAndFuse): mark the chain as one fuse
        // group so TileAndFuse tiles the matmul and fuses the dequant into the
        // tile loop; conversion happens post-tiling on the sliced chain.
        if (_markFuseGroups) {
            if (c.tn % kMqmmWindowAlign != 0) {
                if (c.outerTn)
                    return rw.notifyMatchFailure(srcOp, "ragged sliced chain");
                return padRaggedChain(
                    srcOp, c.dq, c.x, c.wq, c.scale, c.zp ? &*c.zp : nullptr, c.nB, c.B, rw
                );
            }
            auto id = srcOp->getAttrOfType<IntegerAttr>(TORQ_FUSE_GROUP_ID);
            markFuseGroupBackward(srcOp.getResult(0), {c.x}, rw, id);
            return success();
        }
        auto bf16 = rw.getBF16Type();

        WscaleTable table;
        if (failed(buildWscaleTable(srcOp, c, rw, loc, bf16, table)))
            return failure();
        Value xRep = buildLaneReplicatedX(rw, loc, c.x, c.K, bf16);
        Value result = buildTileResult(srcOp, c, table, xRep, rw, loc, bf16);
        rw.replaceOp(srcOp, result);
        return success();
    }
};

} // namespace

void populateLinalgToTorqHLMixedQuantMatmulPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<MixedQuantMatmulRewrite>(context, markFuseGroups);
}
} // namespace mlir::syna::torq
