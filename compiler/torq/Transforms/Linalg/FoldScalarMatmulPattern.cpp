// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-optimize-linalg-for-torq"

namespace mlir::syna::torq {

enum class FoldScalarMatmulMode { Enable, Disable };

static llvm::cl::opt<FoldScalarMatmulMode> clFoldScalarMatmulMode(
    "torq-fold-scalar-matmul-mode",
    llvm::cl::desc("Choose how to handle a matmul whose reduction dim and output feature dim "
                   "are both one"),
    llvm::cl::values(
        clEnumValN(
            FoldScalarMatmulMode::Enable, "enable",
            "With a unit reduction dim the matmul is the elementwise product of its operands, "
            "so rewrite it as that product instead of running a contraction kernel. "
            "Value-preserving; on by default"
        ),
        clEnumValN(FoldScalarMatmulMode::Disable, "disable", "Leave unit-reduction matmuls as is")
    ),
    llvm::cl::init(FoldScalarMatmulMode::Enable)
);

namespace {

/// The shared shape contract of both subpatterns: static shapes, the unit
/// reduction dim (k) on the left-hand operand and both unit dims (k and n) on
/// the right-hand operand, equal leading batch extents, and a result shaped
/// like the left-hand operand.
bool isScalarMatmulShape(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs, ArrayRef<int64_t> res) {
    if (lhs.size() != 2 && lhs.size() != 3)
        return false;
    if (lhs.size() != rhs.size() || lhs.size() != res.size())
        return false;
    if (llvm::any_of(lhs, ShapedType::isDynamic) || llvm::any_of(rhs, ShapedType::isDynamic) ||
        llvm::any_of(res, ShapedType::isDynamic))
        return false;
    if (lhs.back() != 1)
        return false;
    if (rhs[rhs.size() - 2] != 1 || rhs.back() != 1)
        return false;
    if (lhs.size() == 3 && lhs.front() != rhs.front())
        return false;
    return lhs == res;
}

/// Replaces a k=1/n=1 matmul with the elementwise product of its operands.
///
///   rank-3:  %r = linalg.batch_matmul ins(%a : [B,M,1], %b : [B,1,1]) outs(%init : [B,M,1])
///   rank-2:  %r = linalg.matmul       ins(%a : [M,1],    %b : [1,1])   outs(%init : [M,1])
///
/// becomes
///
///   %r = linalg.generic
///        {indexing_maps = [identity, drop-the-row-dim, identity],
///         iterator_types = [parallel, ...]}
///        ins(%a, %b) outs(%init) {
///     %p = arith.mulf %x, %y : bf16
///     linalg.yield %p : bf16
/// }
///
/// With k == 1 the contraction is a single term, so the result is the
/// elementwise product of the operands (the right operand broadcast over the
/// rows). Operands and result are bf16, so the product is computed in bf16 as
/// the matmul's single accumulation would. The matmul's init is reused as the
/// generic's init; the body never reads it.
template <typename MatmulOp>
LogicalResult foldScalarMatmul(MatmulOp matmul, PatternRewriter &rewriter) {
    Value lhs = matmul.getInputs()[0];
    Value rhs = matmul.getInputs()[1];
    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto resTy = dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
    if (!lhsTy || !rhsTy || !resTy)
        return failure();
    if (!isScalarMatmulShape(lhsTy.getShape(), rhsTy.getShape(), resTy.getShape()))
        return failure();
    Type elem = lhsTy.getElementType();
    if (!elem.isBF16() || rhsTy.getElementType() != elem || resTy.getElementType() != elem)
        return failure();
    // The named matmul ops may carry user-defined broadcast/transpose maps
    // (verifier-legal); only the standard (b,m,k)/(b,k,n)/(b,m,n) layout is the
    // shape the rewrite above is valid for.
    if (matmul.hasUserDefinedMaps())
        return failure();

    int64_t rank = lhsTy.getRank();
    MLIRContext *ctx = rewriter.getContext();
    SmallVector<AffineMap> maps(3);
    maps[0] = AffineMap::getMultiDimIdentityMap(rank, ctx);
    // The right operand drops the row dim: rank-3 (d0, 0, d2), rank-2 (0, d1).
    SmallVector<AffineExpr> rhsExprs;
    rhsExprs.reserve(rank);
    for (int64_t i = 0; i < rank; ++i)
        rhsExprs.push_back(
            i == rank - 2 ? getAffineConstantExpr(0, ctx) : getAffineDimExpr(i, ctx)
        );
    maps[1] = AffineMap::get(rank, 0, rhsExprs, ctx);
    maps[2] = AffineMap::getMultiDimIdentityMap(rank, ctx);

    Value init = matmul.getDpsInits()[0];
    // The named matmul body accumulates into its init (addf(init, product));
    // the bare-product rewrite is value-identical only for a zero init.
    if (!isAllZerosTensor(init))
        return failure();
    auto generic = linalg::GenericOp::create(
        rewriter, matmul.getLoc(), TypeRange{resTy}, ValueRange{lhs, rhs}, ValueRange{init}, maps,
        SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
            Value p = arith::MulFOp::create(b, loc, args[0], args[1]).getResult();
            linalg::YieldOp::create(b, loc, p);
        }
    );
    rewriter.replaceOp(matmul, generic.getResult(0));
    LLVM_DEBUG(
        llvm::dbgs() << "[torq-fold-scalar-matmul] folded unit-reduction matmul to an "
                        "elementwise product "
                     << resTy << "\n"
    );
    return success();
}

/// Folds a k=1/n=1 rank-3 matmul (a batched row-scale product) into the
/// elementwise product of its operands.
class FoldScalarMatmulBatchPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
  public:
    explicit FoldScalarMatmulBatchPattern(MLIRContext *ctx)
        : OpRewritePattern<linalg::BatchMatmulOp>(ctx, /*benefit=*/2) {
        setDebugName("FoldScalarMatmulPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::BatchMatmulOp matmul, PatternRewriter &rewriter) const override {
        return foldScalarMatmul(matmul, rewriter);
    }
};

/// Folds a k=1/n=1 rank-2 matmul into the elementwise product of its operands.
class FoldScalarMatmulPattern : public OpRewritePattern<linalg::MatmulOp> {
  public:
    explicit FoldScalarMatmulPattern(MLIRContext *ctx)
        : OpRewritePattern<linalg::MatmulOp>(ctx, /*benefit=*/2) {
        setDebugName("FoldScalarMatmulPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::MatmulOp matmul, PatternRewriter &rewriter) const override {
        return foldScalarMatmul(matmul, rewriter);
    }
};

} // namespace

void populateFoldScalarMatmulPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
    if (clFoldScalarMatmulMode != FoldScalarMatmulMode::Enable)
        return;
    patterns.add<FoldScalarMatmulBatchPattern>(ctx);
    patterns.add<FoldScalarMatmulPattern>(ctx);
}

} // namespace mlir::syna::torq
