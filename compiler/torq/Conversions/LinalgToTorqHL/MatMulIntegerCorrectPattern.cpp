// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lower RaiseMatMulInteger's i32 zero-point corrections to `torq_hl.zero_point_correct`.
// The column and row terms get a pattern each: there is one bias record per ACT pass.
//
// Matched on the correction rather than on the producing `linalg.matmul`, which by now is
// often already a `torq_hl.matmul`, and may have been tiled.

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-torq-matmul-integer-correct"

namespace mlir::syna::torq {

namespace {

// Column sums in the even entries, 0 in the odd ones: what one MAC against `a_zp` turns into
// the `{bias, scale}` pairs the ACT reads.
//
// A tiling pass may have put a `tensor.extract_slice` in front of the raise's constant to walk
// the columns in chunks, and its offset can be a loop induction variable -- so interleave the
// whole underlying constant and re-slice at twice the offset, static or dynamic.
static Value buildInterleavedColsum(Value negColsum, PatternRewriter &rewriter, Location loc) {
    auto interleave = [&](Value base) -> Value {
        auto constOp = base.getDefiningOp<arith::ConstantOp>();
        auto attr = constOp ? dyn_cast<DenseIntElementsAttr>(constOp.getValue()) : nullptr;
        if (!attr || !attr.getType().getElementType().isInteger(32))
            return nullptr;
        SmallVector<APInt> values;
        values.reserve(2 * attr.getNumElements());
        for (const APInt &e : attr.getValues<APInt>()) {
            values.push_back(e);
            values.emplace_back(32, 0);
        }
        auto ty = RankedTensorType::get({int64_t(values.size())}, rewriter.getIntegerType(32));
        return arith::ConstantOp::create(rewriter, loc, ty, DenseIntElementsAttr::get(ty, values))
            .getResult();
    };

    auto sliceOp = negColsum.getDefiningOp<tensor::ExtractSliceOp>();
    if (!sliceOp)
        return interleave(negColsum);

    if (sliceOp.getMixedOffsets().size() != 1 || sliceOp.getMixedSizes().size() != 1 ||
        !llvm::all_of(sliceOp.getStaticStrides(), [](int64_t st) { return st == 1; }))
        return nullptr;
    Value base = interleave(sliceOp.getSource());
    if (!base)
        return nullptr;

    // Double the offset and the size; a dynamic offset is doubled in the IR.
    OpFoldResult offset = sliceOp.getMixedOffsets()[0];
    OpFoldResult size = sliceOp.getMixedSizes()[0];
    auto twice = [&](OpFoldResult v) -> OpFoldResult {
        if (auto attr = dyn_cast<Attribute>(v))
            return rewriter.getIndexAttr(2 * cast<IntegerAttr>(attr).getInt());
        Value two = arith::ConstantIndexOp::create(rewriter, loc, 2);
        return arith::MulIOp::create(rewriter, loc, cast<Value>(v), two).getResult();
    };
    auto sizeAttr = dyn_cast<Attribute>(size);
    if (!sizeAttr)
        return nullptr; // a dynamic column count has no static result type below
    const int64_t doubledSize = 2 * cast<IntegerAttr>(sizeAttr).getInt();
    return tensor::ExtractSliceOp::create(
               rewriter, loc, RankedTensorType::get({doubledSize}, rewriter.getIntegerType(32)),
               base, ArrayRef<OpFoldResult>{twice(offset)},
               ArrayRef<OpFoldResult>{rewriter.getIndexAttr(doubledSize)},
               ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)}
    )
        .getResult();
}

struct MatMulIntegerCorrectPattern : public OpRewritePattern<linalg::GenericOp> {
    MatMulIntegerCorrectPattern(MLIRContext *context, bool markFuseGroups)
        // Ahead of the generic elementwise lowerings, which would otherwise claim the
        // `sitofp` on its own and leave the correction behind.
        : OpRewritePattern(context, /*benefit=*/3), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp corrOp, PatternRewriter &rewriter) const override {
        // Discovery is driven from the matmul; claiming here would give the group a second
        // principal.
        if (_markFuseGroups)
            return failure();

        // The column sums index the N dimension, so the correction is at least rank 1.
        auto corrTy = dyn_cast<RankedTensorType>(corrOp.getResult(0).getType());
        if (!corrTy || !corrTy.hasStaticShape() || corrTy.getRank() < 1 ||
            !corrTy.getElementType().isInteger(32))
            return failure();
        if (corrOp.getNumDpsInputs() != 3 || corrOp.getNumDpsInits() != 1 ||
            !linalg::isElementwise(corrOp) || !corrOp.getRegion().hasOneBlock())
            return failure();

        // body: addi(raw, muli(zext(a_zp), neg_colsum))
        auto yieldOp = cast<linalg::YieldOp>(corrOp.getBody()->getTerminator());
        auto addOp = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();
        if (!addOp || addOp->getBlock() != corrOp.getBody())
            return failure();
        auto isArg = [&](Value v, unsigned index) {
            auto arg = dyn_cast<BlockArgument>(v);
            return arg && arg.getOwner() == corrOp.getBody() && arg.getArgNumber() == index;
        };
        if (!isArg(addOp.getLhs(), 0))
            return failure();
        auto mulOp = addOp.getRhs().getDefiningOp<arith::MulIOp>();
        if (!mulOp || mulOp->getBlock() != corrOp.getBody() || !isArg(mulOp.getRhs(), 2))
            return failure();
        Value widened = mulOp.getLhs();
        if (auto extOp = widened.getDefiningOp<arith::ExtUIOp>())
            widened = extOp.getIn();
        else if (auto truncOp = widened.getDefiningOp<arith::TruncIOp>())
            widened = truncOp.getIn();
        if (!isArg(widened, 1))
            return failure();

        // (d...) -> (0) for the zero point, (d...) -> (dLast) for the column sums.
        MLIRContext *ctx = corrOp.getContext();
        const unsigned rank = corrTy.getRank();
        SmallVector<AffineMap> maps = corrOp.getIndexingMapsArray();
        if (maps.size() != 4 || !maps[3].isIdentity() || maps[1].getNumResults() != 1 ||
            maps[1].getResult(0) != getAffineConstantExpr(0, ctx) || maps[2].getNumResults() != 1 ||
            maps[2].getResult(0) != getAffineDimExpr(rank - 1, ctx))
            return failure();

        const int64_t n = corrTy.getShape().back();
        auto negColsumTy = dyn_cast<RankedTensorType>(corrOp.getDpsInputs()[2].getType());
        if (!negColsumTy || negColsumTy.getRank() != 1 || negColsumTy.getShape()[0] != n ||
            !negColsumTy.getElementType().isInteger(32))
            return failure();

        Value aZp = corrOp.getDpsInputs()[1];
        auto aZpTy = dyn_cast<RankedTensorType>(aZp.getType());
        if (!aZpTy || aZpTy.getRank() != 1 || aZpTy.getShape()[0] != 1 ||
            aZpTy.getElementType().getIntOrFloatBitWidth() != 8)
            return failure();

        Location loc = corrOp.getLoc();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(corrOp);

        // `-a_zp * colsum_W[n]` in the bias slots, 1 in the scale slots.
        Value interleaved = buildInterleavedColsum(corrOp.getDpsInputs()[2], rewriter, loc);
        if (!interleaved)
            return failure();
        auto tableTy = RankedTensorType::get({2 * n}, rewriter.getIntegerType(32));
        Value table = torq_hl::ZeroPointBiasTableOp::create(
                          rewriter, loc, tableTy, createInitTensor(corrOp, rewriter, tableTy),
                          createI32Const(rewriter, corrOp, std::vector<int32_t>{0, 1, 1, 1}), aZp,
                          interleaved, /*weights_unsigned=*/true
        )
                          .getResult(0);

        // The corrected value is the exact ONNX result, so it cannot overflow i32.
        auto correctOp = torq_hl::ZeroPointCorrectOp::create(
            rewriter, loc, corrTy, createInitTensor(corrOp, rewriter, corrTy),
            std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), table,
            corrOp.getDpsInputs()[0], /*per_row=*/false
        );

        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Lowered zero-point correction to " << correctOp
                         << "\n"
        );
        rewriter.replaceOp(corrOp, correctOp.getResult(0));
        return success();
    }

  private:
    const bool _markFuseGroups;
};

// The residual weight zero point's row term, `col - drs[m, 0]`, where `drs` is the `[M, 2]`
// result of the `delta * rowsum_A` matmul. Its second column is zero by construction, so
// collapsing it to `[2*M]` already has the record's interleaving; a signed -1 factor negates
// the values into the bias.
// Peel the explicit `[M] -> [.., M, N]` broadcast a standardization pass materializes when a
// binary elementwise op reads an operand through a non-identity map. Null when `v` is not that.
static Value peelRowBroadcast(Value v, int64_t m, unsigned rank) {
    auto bcastOp = v.getDefiningOp<linalg::GenericOp>();
    if (!bcastOp || bcastOp.getNumDpsInputs() != 1 || bcastOp.getNumDpsInits() != 1 ||
        !bcastOp.getRegion().hasOneBlock())
        return nullptr;
    auto yieldOp = cast<linalg::YieldOp>(bcastOp.getBody()->getTerminator());
    auto arg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
    if (!arg || arg.getOwner() != bcastOp.getBody() || arg.getArgNumber() != 0)
        return nullptr;

    SmallVector<AffineMap> maps = bcastOp.getIndexingMapsArray();
    MLIRContext *ctx = bcastOp.getContext();
    if (maps.size() != 2 || !maps[1].isIdentity() || maps[0].getNumDims() != rank ||
        maps[0].getNumResults() != 1 || maps[0].getResult(0) != getAffineDimExpr(rank - 2, ctx))
        return nullptr;

    Value source = bcastOp.getDpsInputs()[0];
    auto sourceTy = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceTy || sourceTy.getRank() != 1 || sourceTy.getShape()[0] != m)
        return nullptr;
    return source;
}

struct MatMulIntegerRowCorrectPattern : public OpRewritePattern<linalg::GenericOp> {
    MatMulIntegerRowCorrectPattern(MLIRContext *context, bool markFuseGroups)
        : OpRewritePattern(context, /*benefit=*/3), _markFuseGroups(markFuseGroups) {}

    LogicalResult
    matchAndRewrite(linalg::GenericOp corrOp, PatternRewriter &rewriter) const override {
        if (_markFuseGroups)
            return failure();

        // The row term indexes the M dimension, so the correction is at least rank 2.
        auto corrTy = dyn_cast<RankedTensorType>(corrOp.getResult(0).getType());
        if (!corrTy || !corrTy.hasStaticShape() || corrTy.getRank() < 2 ||
            !corrTy.getElementType().isInteger(32))
            return failure();
        if (corrOp.getNumDpsInputs() != 2 || corrOp.getNumDpsInits() != 1 ||
            !linalg::isElementwise(corrOp) || !corrOp.getRegion().hasOneBlock())
            return failure();

        // body: subi(col, drs)
        auto yieldOp = cast<linalg::YieldOp>(corrOp.getBody()->getTerminator());
        auto subOp = yieldOp.getOperand(0).getDefiningOp<arith::SubIOp>();
        if (!subOp || subOp->getBlock() != corrOp.getBody())
            return failure();
        auto isArg = [&](Value v, unsigned index) {
            auto arg = dyn_cast<BlockArgument>(v);
            return arg && arg.getOwner() == corrOp.getBody() && arg.getArgNumber() == index;
        };
        if (!isArg(subOp.getLhs(), 0) || !isArg(subOp.getRhs(), 1))
            return failure();

        // The row term is read either through a (d...) -> (dM) map or through the broadcast
        // materialized in its place.
        MLIRContext *ctx = corrOp.getContext();
        const unsigned rank = corrTy.getRank();
        const int64_t m = corrTy.getShape()[rank - 2];
        SmallVector<AffineMap> maps = corrOp.getIndexingMapsArray();
        if (maps.size() != 3 || !maps[0].isIdentity() || !maps[2].isIdentity())
            return failure();
        Value rowTerm = corrOp.getDpsInputs()[1];
        if (maps[1].isIdentity()) {
            rowTerm = peelRowBroadcast(rowTerm, m, rank);
            if (!rowTerm)
                return failure();
        }
        else if (maps[1].getNumResults() != 1 ||
                 maps[1].getResult(0) != getAffineDimExpr(rank - 2, ctx)) {
            return failure();
        }

        // Recover the [M, 2] source: collapsed, it is already the record's interleaving.
        auto sliceOp = rowTerm.getDefiningOp<tensor::ExtractSliceOp>();
        if (!sliceOp)
            return failure();
        auto drsTy = dyn_cast<RankedTensorType>(sliceOp.getSource().getType());
        if (!drsTy || drsTy.getRank() != 2 || drsTy.getShape() != ArrayRef<int64_t>{m, 2} ||
            !drsTy.getElementType().isInteger(32) ||
            sliceOp.getStaticOffsets() != ArrayRef<int64_t>{0, 0} ||
            sliceOp.getStaticSizes() != ArrayRef<int64_t>{m, 1} ||
            sliceOp.getStaticStrides() != ArrayRef<int64_t>{1, 1})
            return failure();

        Location loc = corrOp.getLoc();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(corrOp);

        auto i32Ty = rewriter.getIntegerType(32);
        auto flatTy = RankedTensorType::get({2 * m}, i32Ty);
        Value flat = tensor::CollapseShapeOp::create(
                         rewriter, loc, flatTy, sliceOp.getSource(),
                         ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}}
        ).getResult();
        auto negOneTy = RankedTensorType::get({1}, rewriter.getIntegerType(8));
        Value negOne = arith::ConstantOp::create(
                           rewriter, loc, negOneTy,
                           DenseIntElementsAttr::get(negOneTy, {APInt(8, -1, /*isSigned=*/true)})
        )
                           .getResult();
        Value table = torq_hl::ZeroPointBiasTableOp::create(
                          rewriter, loc, flatTy, createInitTensor(corrOp, rewriter, flatTy),
                          createI32Const(rewriter, corrOp, std::vector<int32_t>{0, 1, 1, 1}),
                          negOne, flat, /*weights_unsigned=*/false
        )
                          .getResult(0);

        auto correctOp = torq_hl::ZeroPointCorrectOp::create(
            rewriter, loc, corrTy, createInitTensor(corrOp, rewriter, corrTy),
            std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), table,
            corrOp.getDpsInputs()[0], /*per_row=*/true
        );

        LLVM_DEBUG(
            llvm::dbgs() << "[" DEBUG_TYPE "] Lowered row zero-point correction to " << correctOp
                         << "\n"
        );
        rewriter.replaceOp(corrOp, correctOp.getResult(0));
        return success();
    }

  private:
    const bool _markFuseGroups;
};

} // namespace

void populateLinalgToTorqHLMatMulIntegerCorrectPatterns(
    MLIRContext *context, RewritePatternSet &patterns, bool markFuseGroups
) {
    patterns.insert<MatMulIntegerCorrectPattern, MatMulIntegerRowCorrectPattern>(
        context, markFuseGroups
    );
}

} // namespace mlir::syna::torq
