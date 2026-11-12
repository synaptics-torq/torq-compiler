// Copyright 2026 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Split the zero-point correction of an integer matmul into ops the slices run.
//
// IREE's QuantizedMatmulToMatmul turns a matmul with zero points (e.g. a `tosa.matmul` of two
// activations, as in an attention block) into a plain matmul, the row sums of A, the column
// sums of B and one element-wise i32 correction:
//
//   out[b, m, n] = mm[b, m, n] - rowsum[b, m] * zb - colsum[b, n] * za + K * za * zb
//
// With constant weights the sums fold away, but with two activations they are computed at
// runtime, and the correction (i32 multiplies, operands broadcast along different dimensions,
// in one body) has no slice kernel, so it runs on CSS.
//
// The zero points are constants, so each product becomes shifts and adds on its small sum
// tensor, the constant term folds into one of them, and each term is then added to `mm` with a
// broadcast add or sub. i32 arithmetic wraps the same way in both forms, so the result is
// exact.

#include "PassesDetail.h"
#include "Patterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "torq-split-matmul-zero-point-correction"

namespace mlir::syna::torq {

namespace {

// One input of the correction, with the factor it is multiplied by in the sum.
struct Term {
    Value input;
    AffineMap map;
    int64_t factor;
};

// The correction as a sum: `constant + sum(factor * input)`.
struct LinearSum {
    SmallVector<Term> terms;
    int64_t constant = 0;
    bool hasConstantMul = false;
};

// The value of a scalar constant input: an i32 constant, or a splat i32 tensor.
static std::optional<int64_t> getScalarConstant(Value v) {
    APInt value;
    if (matchPattern(v, m_ConstantInt(&value)))
        return value.getSExtValue();
    DenseIntElementsAttr attr;
    if (matchPattern(v, m_Constant(&attr)) && attr.isSplat())
        return attr.getSplatValue<APInt>().getSExtValue();
    return std::nullopt;
}

class CorrectionMatcher {
  public:
    explicit CorrectionMatcher(linalg::GenericOp op) : op(op) {}

    // Decompose the yielded value as a sum of the inputs times constants.
    bool match(LinearSum &sum) {
        auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
        return yieldOp.getNumOperands() == 1 && collect(yieldOp.getOperand(0), 1, sum);
    }

  private:
    // The value of a scalar constant: an input that is one, or one the body captures.
    std::optional<int64_t> getConstantArg(Value v) {
        auto arg = dyn_cast<BlockArgument>(v);
        if (!arg)
            return getScalarConstant(v);
        if (arg.getOwner() != op.getBody() || arg.getArgNumber() >= op.getNumDpsInputs())
            return std::nullopt;
        return getScalarConstant(op.getDpsInputs()[arg.getArgNumber()]);
    }

    bool collect(Value v, int64_t factor, LinearSum &sum) {
        if (auto constant = getConstantArg(v)) {
            sum.constant += factor * *constant;
            return true;
        }
        if (auto arg = dyn_cast<BlockArgument>(v)) {
            if (arg.getOwner() != op.getBody() || arg.getArgNumber() >= op.getNumDpsInputs())
                return false;
            OpOperand *operand = op.getDpsInputOperand(arg.getArgNumber());
            sum.terms.push_back({operand->get(), op.getMatchingIndexingMap(operand), factor});
            return true;
        }
        Operation *def = v.getDefiningOp();
        if (!def || def->getBlock() != op.getBody())
            return false;
        if (auto add = dyn_cast<arith::AddIOp>(def))
            return collect(add.getLhs(), factor, sum) && collect(add.getRhs(), factor, sum);
        if (auto sub = dyn_cast<arith::SubIOp>(def))
            return collect(sub.getLhs(), factor, sum) && collect(sub.getRhs(), -factor, sum);
        if (auto mul = dyn_cast<arith::MulIOp>(def)) {
            Value lhs = mul.getLhs(), rhs = mul.getRhs();
            auto constant = getConstantArg(rhs);
            if (!constant) {
                std::swap(lhs, rhs);
                constant = getConstantArg(rhs);
            }
            // Only a product by a constant keeps the sum linear; the factors are zero points,
            // so they stay far from overflowing int64.
            if (!constant || std::abs(*constant) > (1 << 16))
                return false;
            sum.hasConstantMul = true;
            return collect(lhs, factor * *constant, sum);
        }
        return false;
    }

    linalg::GenericOp op;
};

// Whether v is computed from constants only, in a few ops: then the sum is a constant-weight
// one, which QMatmulToFCConvert and the MatMulInteger raise fold into the matmul bias.
static bool isComputedFromConstants(Value v) {
    SmallVector<Value> worklist{v};
    llvm::SmallPtrSet<Operation *, 16> visited;
    while (!worklist.empty()) {
        Operation *def = worklist.pop_back_val().getDefiningOp();
        // A block argument is a runtime input.
        if (!def)
            return false;
        if (!visited.insert(def).second)
            continue;
        if (visited.size() > 32)
            return false;
        // Only the tensors carry the data: a source of them that isn't a constant, such as a
        // load from a dispatch binding, is a runtime input.
        bool hasTensorOperand = false;
        for (Value operand : def->getOperands()) {
            if (isa<RankedTensorType>(operand.getType())) {
                worklist.push_back(operand);
                hasTensorOperand = true;
            }
        }
        if (!hasTensorOperand && !matchPattern(def, m_Constant()) && !isa<tensor::EmptyOp>(def))
            return false;
    }
    return true;
}

// The non-adjacent form of m > 0: the digits (shift, +1 or -1) with m = sum(digit << shift),
// most significant first. It has the fewest non-zero digits, so the fewest shifts and adds.
static SmallVector<std::pair<int64_t, int64_t>> getNonAdjacentForm(uint64_t m) {
    SmallVector<std::pair<int64_t, int64_t>> digits;
    for (int64_t shift = 0; m != 0; ++shift, m >>= 1) {
        if (m & 1) {
            int64_t digit = (m & 3) == 3 ? -1 : 1;
            digits.push_back({shift, digit});
            m = digit < 0 ? m + 1 : m - 1;
        }
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

class CorrectionBuilder {
  public:
    CorrectionBuilder(PatternRewriter &rewriter, Location loc) : rewriter(rewriter), loc(loc) {}

    // `lhs <op> rhs`, with `rhs` read through `rhsMap` (a broadcast) in the iteration space of
    // `lhs`, which is also the shape of the result.
    template <typename OpTy> Value binary(Value lhs, Value rhs, AffineMap rhsMap) {
        auto type = cast<RankedTensorType>(lhs.getType());
        AffineMap identity = rewriter.getMultiDimIdentityMap(type.getRank());
        Value init = tensor::EmptyOp::create(rewriter, loc, type.getShape(), type.getElementType());
        SmallVector<utils::IteratorType> iterators(type.getRank(), utils::IteratorType::parallel);
        return linalg::GenericOp::create(
                   rewriter, loc, type, ValueRange{lhs, rhs}, ValueRange{init},
                   ArrayRef<AffineMap>{identity, rhsMap, identity}, iterators,
                   [](OpBuilder &b, Location l, ValueRange args) {
                       Value result = OpTy::create(b, l, args[0], args[1]);
                       linalg::YieldOp::create(b, l, result);
                   }
        ).getResult(0);
    }

    // `x <op> c` with a constant c, read from a splat i32 tensor of ones like the TOSA ops
    // lower it.
    template <typename OpTy> Value binaryConstant(Value x, int64_t c) {
        auto type = cast<RankedTensorType>(x.getType());
        SmallVector<int64_t> ones(type.getRank(), 1);
        auto constType = RankedTensorType::get(ones, rewriter.getI32Type());
        Value constant = arith::ConstantOp::create(
            rewriter, loc, constType,
            // Wraps to i32 like the arithmetic it replaces.
            DenseElementsAttr::get(constType, APInt(64, static_cast<uint64_t>(c), true).trunc(32))
        );
        SmallVector<AffineExpr> zeros(type.getRank(), rewriter.getAffineConstantExpr(0));
        AffineMap map = AffineMap::get(type.getRank(), 0, zeros, rewriter.getContext());
        return binary<OpTy>(x, constant, map);
    }

    Value identityBinary(Value lhs, Value rhs, bool add) {
        AffineMap identity =
            rewriter.getMultiDimIdentityMap(cast<RankedTensorType>(lhs.getType()).getRank());
        return add ? binary<arith::AddIOp>(lhs, rhs, identity)
                   : binary<arith::SubIOp>(lhs, rhs, identity);
    }

    // `x * m` for m > 0, with shifts and adds.
    Value multiply(Value x, uint64_t m) {
        Value result;
        for (auto [shift, digit] : getNonAdjacentForm(m)) {
            Value shifted = shift == 0 ? x : binaryConstant<arith::ShLIOp>(x, shift);
            // The most significant digit of the non-adjacent form is +1.
            result = result ? identityBinary(result, shifted, digit > 0) : shifted;
        }
        return result;
    }

  private:
    PatternRewriter &rewriter;
    Location loc;
};

// The row or column sum of an i8 matmul operand, `sum_k extsi(x)` in a one-reduction generic,
// as a cast to i32 and a `linalg.reduce`, which the reduce kernel takes. Returns `sum` itself
// when it is not one.
static Value rewriteSumAsReduce(PatternRewriter &rewriter, Value sum) {
    auto op = sum.getDefiningOp<linalg::GenericOp>();
    if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
        op.getNumReductionLoops() != 1 || !op.hasPureTensorSemantics())
        return sum;
    Value input = op.getDpsInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.getElementType().isInteger(8) ||
        !getElementTypeOrSelf(sum.getType()).isInteger(32))
        return sum;

    // yield(addi(extsi(in), out)), in either operand order.
    Block *body = op.getBody();
    auto addOp =
        cast<linalg::YieldOp>(body->getTerminator()).getOperand(0).getDefiningOp<arith::AddIOp>();
    if (!addOp || addOp->getBlock() != body)
        return sum;
    Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
    if (lhs == body->getArgument(1))
        std::swap(lhs, rhs);
    auto extOp = lhs.getDefiningOp<arith::ExtSIOp>();
    if (rhs != body->getArgument(1) || !extOp || extOp.getIn() != body->getArgument(0))
        return sum;

    // From 0, and the input dims other than the reduced one, in order, give the output.
    auto fillOp = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    APInt fillValue;
    if (!fillOp || !matchPattern(fillOp.value(), m_ConstantInt(&fillValue)) || !fillValue.isZero())
        return sum;
    AffineMap inMap = op.getMatchingIndexingMap(op.getDpsInputOperand(0));
    AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
    if (!inMap.isPermutation() || !outMap.isProjectedPermutation())
        return sum;
    SmallVector<unsigned> reductionDims;
    op.getReductionDims(reductionDims);
    std::optional<unsigned> reducedPosition =
        inMap.getResultPosition(rewriter.getAffineDimExpr(reductionDims.front()));
    SmallVector<AffineExpr> kept;
    for (auto [position, expr] : llvm::enumerate(inMap.getResults()))
        if (position != *reducedPosition)
            kept.push_back(expr);
    if (ArrayRef<AffineExpr>(kept) != outMap.getResults())
        return sum;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto wideType = RankedTensorType::get(inputType.getShape(), rewriter.getI32Type());
    AffineMap identity = rewriter.getMultiDimIdentityMap(inputType.getRank());
    Value wideInit =
        tensor::EmptyOp::create(rewriter, loc, inputType.getShape(), rewriter.getI32Type());
    Value wide =
        linalg::GenericOp::create(
            rewriter, loc, wideType, ValueRange{input}, ValueRange{wideInit},
            ArrayRef<AffineMap>{identity, identity},
            SmallVector<utils::IteratorType>(inputType.getRank(), utils::IteratorType::parallel),
            [&](OpBuilder &b, Location l, ValueRange args) {
                Value ext = arith::ExtSIOp::create(b, l, b.getI32Type(), args[0]);
                linalg::YieldOp::create(b, l, ext);
            }
        ).getResult(0);
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, ValueRange{wide}, ValueRange{op.getDpsInits()[0]},
        ArrayRef<int64_t>{static_cast<int64_t>(*reducedPosition)},
        [](OpBuilder &b, Location l, ValueRange args) {
            Value add = arith::AddIOp::create(b, l, args[0], args[1]);
            linalg::YieldOp::create(b, l, add);
        }
    );
    rewriter.replaceOp(op, reduceOp.getResults());
    return reduceOp.getResult(0);
}

struct SplitMatmulZeroPointCorrectionPattern : public OpRewritePattern<linalg::GenericOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
        if (op.getNumResults() != 1 || op.getNumDpsInits() != 1 || !type ||
            !type.hasStaticShape() || !type.getElementType().isInteger(32) ||
            op.getNumParallelLoops() != op.getNumLoops() ||
            !op.getMatchingIndexingMap(op.getDpsInitOperand(0)).isIdentity())
            return failure();
        for (Value input : op.getDpsInputs())
            if (getElementTypeOrSelf(input.getType()) != rewriter.getI32Type())
                return failure();

        LinearSum sum;
        if (!CorrectionMatcher(op).match(sum) || !sum.hasConstantMul)
            return rewriter.notifyMatchFailure(op, "not a sum of inputs times constants");

        // The accumulator is read in full and added once; every other input is a broadcast
        // (lower rank, no constant indices), multiplied by a non-zero constant.
        Value acc;
        SmallVector<Term> broadcasts;
        for (const Term &term : sum.terms) {
            if (term.map.isIdentity()) {
                if (acc || term.factor != 1)
                    return rewriter.notifyMatchFailure(op, "expected one full-size input");
                acc = term.input;
                continue;
            }
            if (!term.map.isProjectedPermutation() || term.map.getNumResults() >= type.getRank())
                return rewriter.notifyMatchFailure(op, "expected broadcast corrections");
            if (isComputedFromConstants(term.input))
                return rewriter.notifyMatchFailure(op, "constant sums fold into the matmul");
            // An i8 zero point has at most 4 digits.
            if (getNonAdjacentForm(std::abs(term.factor)).size() > 4)
                return rewriter.notifyMatchFailure(op, "factor needs too many shifts and adds");
            if (term.factor != 0)
                broadcasts.push_back(term);
        }
        if (!acc || cast<RankedTensorType>(acc.getType()).getShape() != type.getShape())
            return rewriter.notifyMatchFailure(op, "expected the accumulator as an input");
        // Only matmul corrections: a convolution with runtime weights has the same form, but
        // its kernels take the zero point themselves.
        auto accOp = acc.getDefiningOp<linalg::LinalgOp>();
        if (!accOp || !linalg::isaContractionOpInterface(accOp))
            return rewriter.notifyMatchFailure(op, "the accumulator is not a matmul");
        if (broadcasts.empty())
            return rewriter.notifyMatchFailure(op, "no correction to split");
        // Wraps the same way as the i32 arithmetic it replaces.
        int64_t constant = static_cast<int32_t>(static_cast<uint32_t>(sum.constant));

        CorrectionBuilder builder(rewriter, op.getLoc());
        Value result = acc;
        for (auto [index, term] : llvm::enumerate(broadcasts)) {
            // `factor * x` as `sign * (|factor| * x)`, so that the sign goes in the add or sub
            // of the full-size tensor.
            bool add = term.factor > 0;
            Value product =
                builder.multiply(rewriteSumAsReduce(rewriter, term.input), std::abs(term.factor));
            if (index == 0 && constant != 0)
                product =
                    builder.binaryConstant<arith::AddIOp>(product, add ? constant : -constant);
            result = add ? builder.binary<arith::AddIOp>(result, product, term.map)
                         : builder.binary<arith::SubIOp>(result, product, term.map);
        }

        LLVM_DEBUG(llvm::dbgs() << "Split matmul zero-point correction " << op << "\n");
        rewriter.replaceOp(op, result);
        return success();
    }
};

} // namespace

void populateSplitMatmulZeroPointCorrectionPatterns(
    MLIRContext *context, RewritePatternSet &patterns
) {
    patterns.add<SplitMatmulZeroPointCorrectionPattern>(context);
}

} // namespace mlir::syna::torq
