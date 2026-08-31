// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "torq-precalc-silu-i8-pattern"

namespace mlir::syna::torq {

namespace {

static llvm::cl::opt<bool> clDisablePreCalcSiluI8(
    "torq-disable-precalc-silu-i8",
    llvm::cl::desc("Keep a quantized SiLU as a multiply of two rescaled paths instead of "
                   "precomputing it into the sigmoid table. The precomputation is on by "
                   "default; this is the escape hatch for a graph whose LRAM allocation it "
                   "pushes over the limit"),
    llvm::cl::init(false)
);

constexpr int kTableSize = 256;
constexpr int kTableBias = 128;

/// TOSA apply_scale_32, as TosaToArith lowers it. The caller must have checked the
/// shift is in the range the specification allows; `1 << (shift - 1)` is undefined
/// outside it.
static int32_t
constantFoldApplyScale32(int64_t value, int32_t multiplier, int32_t shift, bool doubleRound) {
    assert(shift >= 2 && shift <= 62 && "apply_scale shift out of the TOSA range");
    int64_t round = int64_t(1) << (shift - 1);
    if (doubleRound && shift > 31) {
        round += value >= 0 ? (int64_t(1) << 30) : -(int64_t(1) << 30);
    }
    int64_t result = (value * int64_t(multiplier) + round) >> shift;
    return static_cast<int32_t>(std::clamp(
        result, int64_t(std::numeric_limits<int32_t>::min()),
        int64_t(std::numeric_limits<int32_t>::max())
    ));
}

/// Predicate for the matcher's element-type checks. Not to be confused with
/// PatternUtils' isI8Type(Value), which peels a shaped type first; the matcher hands
/// over an element type already.
static bool isI8Elem(Type type) { return type.isInteger(8); }

/// The value an elementwise generic yields.
static Value yieldedValue(linalg::GenericOp op) {
    return cast<linalg::YieldOp>(op.getBody()->getTerminator()).getOperand(0);
}

/// Everything precomputing a SiLU needs: where it starts and ends, and every
/// constant the arithmetic between them uses.
struct SiluI8 {
    linalg::GenericOp mulOp;        ///< muli of the two rescaled arms
    linalg::GenericOp activationOp; ///< produces the i8 activation both arms read
    linalg::GenericOp outRescaleOp; ///< takes the product back down to i8
    SmallVector<int8_t> sigmoid;    ///< the table the sigmoid arm looks up
    int32_t directZp, tableZp;      ///< zero-points the two identity rescales move
    int32_t multiplier, shift;      ///< scale of the output rescale
    int32_t outZp;
    bool doubleRound;

    /// The result the whole chain gives for an i8 activation value.
    int8_t evaluate(int v) const {
        int64_t a = int64_t(v) - directZp;
        int64_t b = int64_t(sigmoid[v + kTableBias]) - tableZp;
        int32_t scaled = constantFoldApplyScale32(a * b, multiplier, shift, doubleRound);
        return static_cast<int8_t>(std::clamp<int32_t>(
            scaled + outZp, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()
        ));
    }
};

/// yield extsi(muli(%arg0, %arg1)).

static bool isMul(linalg::GenericOp op) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numInputs(2).numOutputs(1).allParallel().match(op))
        return false;

    Value lhs, rhs;
    return matchPattern(
               yieldedValue(op), m_Op<arith::ExtSIOp>(m_Op<arith::MulIOp>(
                                     matchers::m_Any(&lhs), matchers::m_Any(&rhs)
                                 ))
           ) &&
           isa<BlockArgument>(lhs) && isa<BlockArgument>(rhs);
}

/// One arm of the diamond: a rescale whose scale is 1, so it only moves the
/// zero-point.
struct IdentityRescale {
    int32_t zp;
    linalg::GenericOp input;
};

/// yield [trunci](apply_scale(subi(extsi(%arg), zp), mult, shift)) with
/// mult/2^shift == 1.
///
/// The trunci and the subi are both optional -- the first when the arm stays i32, the
/// second when the zero-point is 0 -- so the body is walked rather than matched with
/// one yieldChain, which cannot express an optional link.
static std::optional<IdentityRescale> matchIdentityRescale(linalg::GenericOp op) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numInputs(1).numOutputs(1).allParallel().inputElementType(0, isI8Elem).match(op))
        return std::nullopt;

    Value scaled = yieldedValue(op);
    if (auto truncOp = scaled.getDefiningOp<arith::TruncIOp>())
        scaled = truncOp.getIn();

    Value value;
    APInt multiplier, shift;
    if (!matchPattern(
            scaled, m_Op<tosa::ApplyScaleOp>(
                        matchers::m_Any(&value), m_ConstantInt(&multiplier), m_ConstantInt(&shift)
                    )
        ))
        return std::nullopt;
    // Shift 0 is fine here, unlike in the output rescale below: this arm only has to
    // satisfy mult == 1 << shift, and 1 << 0 is a valid identity.
    int64_t shiftValue = shift.getSExtValue();
    if (shiftValue < 0 || shiftValue > 62 ||
        multiplier.getSExtValue() != (int64_t(1) << shiftValue))
        return std::nullopt;

    IdentityRescale rescale{0, op.getInputs()[0].getDefiningOp<linalg::GenericOp>()};
    if (!rescale.input)
        return std::nullopt;
    if (value.getDefiningOp<arith::SubIOp>()) {
        APInt zp;
        if (!matchPattern(value, m_Op<arith::SubIOp>(matchers::m_Any(), m_ConstantInt(&zp))))
            return std::nullopt;
        rescale.zp = static_cast<int32_t>(zp.getSExtValue());
    }
    return rescale;
}

/// The sigmoid arm: a lookup into a constant table.
struct TableLookup {
    SmallVector<int8_t> table;
    linalg::GenericOp input;
};

/// yield extract(%lut, index_cast(%arg) + 128) over a 256-entry i8 constant.
static std::optional<TableLookup> matchTableLookup(linalg::GenericOp op) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numInputs(1).numOutputs(1).allParallel().inputElementType(0, isI8Elem).match(op))
        return std::nullopt;

    DenseElementsAttr lut;
    Value index;
    if (!matchPattern(
            yieldedValue(op), m_Op<tensor::ExtractOp>(m_Constant(&lut), matchers::m_Any(&index))
        ))
        return std::nullopt;
    if (!lut.getElementType().isInteger(8) || lut.getNumElements() != kTableSize)
        return std::nullopt;
    // The bias is what turns a signed i8 into a table index. Without checking it the
    // entries computed below would line up against the wrong inputs.
    APInt bias;
    if (!matchPattern(
            index,
            m_Op<arith::AddIOp>(m_Op<arith::IndexCastOp>(matchers::m_Any()), m_ConstantInt(&bias))
        ) ||
        bias.getSExtValue() != kTableBias)
        return std::nullopt;

    TableLookup lookup{{}, op.getInputs()[0].getDefiningOp<linalg::GenericOp>()};
    if (!lookup.input)
        return std::nullopt;
    lookup.table.assign(lut.getValues<int8_t>().begin(), lut.getValues<int8_t>().end());
    return lookup;
}

/// yield trunci(minsi(maxsi(addi(apply_scale(%arg, mult, shift), zp)))) down to i8,
/// with a scale this pattern can replay. Fills the scale fields of `silu`.
static bool matchOutputRescale(linalg::GenericOp op, SiluI8 &silu) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numOutputs(1).allParallel().outputElementType(0, isI8Elem).match(op))
        return false;

    Value scaled;
    APInt zp;
    if (!matchPattern(
            yieldedValue(op),
            m_Op<arith::TruncIOp>(m_Op<arith::MinSIOp>(
                m_Op<arith::MaxSIOp>(
                    m_Op<arith::AddIOp>(matchers::m_Any(&scaled), m_ConstantInt(&zp)),
                    matchers::m_Any()
                ),
                matchers::m_Any()
            ))
        ))
        return false;

    APInt multiplier, shift;
    if (!matchPattern(
            scaled, m_Op<tosa::ApplyScaleOp>(
                        matchers::m_Any(), m_ConstantInt(&multiplier), m_ConstantInt(&shift)
                    )
        ))
        return false;
    auto applyScaleOp = cast<tosa::ApplyScaleOp>(scaled.getDefiningOp());
    // Replaying apply_scale needs a shift the specification allows: `1 << (shift - 1)`
    // is undefined outside it.
    int64_t shiftValue = shift.getSExtValue();
    if (shiftValue < 2 || shiftValue > 62)
        return false;

    silu.outZp = static_cast<int32_t>(zp.getSExtValue());
    silu.multiplier = static_cast<int32_t>(multiplier.getSExtValue());
    silu.shift = static_cast<int32_t>(shiftValue);
    silu.doubleRound = applyScaleOp.getRoundingMode() == tosa::RoundingMode::DOUBLE_ROUND;
    return true;
}

/// Walk out from a mul generic to the whole SiLU. Fails on anything this pattern
/// cannot replay exactly.
static std::optional<SiluI8> matchSiluI8(linalg::GenericOp mulOp) {
    if (!isMul(mulOp) || !mulOp.getResult(0).hasOneUse())
        return std::nullopt;

    auto lhs = mulOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
    auto rhs = mulOp.getInputs()[1].getDefiningOp<linalg::GenericOp>();
    if (!lhs || !rhs)
        return std::nullopt;
    auto lhsArm = matchIdentityRescale(lhs);
    auto rhsArm = matchIdentityRescale(rhs);
    if (!lhsArm || !rhsArm)
        return std::nullopt;

    // One arm comes through the sigmoid table, the other straight from the activation.
    IdentityRescale tableArm = *lhsArm, directArm = *rhsArm;
    auto lookup = matchTableLookup(tableArm.input);
    if (!lookup) {
        std::swap(tableArm, directArm);
        lookup = matchTableLookup(tableArm.input);
        if (!lookup)
            return std::nullopt;
    }
    // Both arms have to read the same activation, or this is two SiLUs sharing a mul.
    if (lookup->input != directArm.input)
        return std::nullopt;

    SiluI8 silu;
    silu.mulOp = mulOp;
    silu.activationOp = directArm.input;
    silu.sigmoid = std::move(lookup->table);
    silu.directZp = directArm.zp;
    silu.tableZp = tableArm.zp;
    silu.outRescaleOp = dyn_cast<linalg::GenericOp>(*mulOp.getResult(0).getUsers().begin());
    if (!silu.outRescaleOp || !matchOutputRescale(silu.outRescaleOp, silu))
        return std::nullopt;
    return silu;
}

/// Precompute a quantized SiLU into the sigmoid table it already carries.
///
/// Both ends being i8 is what makes this possible: the activation takes 256 values,
/// so the mul and the rescales that follow the sigmoid lookup have only 256 possible
/// results and one table reproduces the chain entry for entry. What goes away with
/// the mul are the i16/i32 tensors its two arms materialize -- those, not the op
/// count, are what dominate the traffic.
struct PreCalcSiluI8Pattern : public OpRewritePattern<linalg::GenericOp> {
    PreCalcSiluI8Pattern(MLIRContext *context) : OpRewritePattern(context) {
        setDebugName("PreCalcSiluI8Pattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp mulOp, PatternRewriter &rewriter) const override {
        auto silu = matchSiluI8(mulOp);
        if (!silu)
            return rewriter.notifyMatchFailure(mulOp, "not an i8-in/i8-out quantized SiLU");

        // Entry i stands for the activation value i - 128, matching how the table
        // body indexes a signed i8.
        SmallVector<int8_t> table(kTableSize);
        for (int i = 0; i < kTableSize; ++i)
            table[i] = silu->evaluate(i - kTableBias);

        Location loc = mulOp.getLoc();
        auto outType = cast<RankedTensorType>(silu->outRescaleOp.getResult(0).getType());
        auto i8Type = rewriter.getIntegerType(8);
        rewriter.setInsertionPoint(silu->outRescaleOp);

        auto tableType = RankedTensorType::get({kTableSize}, i8Type);
        Value tableConst = arith::ConstantOp::create(
            rewriter, loc, DenseElementsAttr::get(tableType, llvm::ArrayRef(table))
        );

        // An elementwise generic mapping each i8 activation to table[value + 128].
        auto identityMap = rewriter.getMultiDimIdentityMap(outType.getRank());
        SmallVector<AffineMap> maps{identityMap, identityMap};
        SmallVector<utils::IteratorType> iterTypes(
            outType.getRank(), utils::IteratorType::parallel
        );
        Value init = tensor::EmptyOp::create(rewriter, loc, outType.getShape(), i8Type).getResult();

        auto lookupOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{outType}, ValueRange{silu->activationOp.getResult(0)},
            ValueRange{init}, maps, iterTypes
        );
        Block *body = rewriter.createBlock(
            &lookupOp.getRegion(), lookupOp.getRegion().end(), TypeRange{i8Type, i8Type},
            SmallVector<Location>(2, loc)
        );
        rewriter.setInsertionPointToStart(body);
        Value index = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), body->getArgument(0)
        );
        Value biased = arith::AddIOp::create(
            rewriter, loc, index, arith::ConstantIndexOp::create(rewriter, loc, kTableBias)
        );
        linalg::YieldOp::create(
            rewriter, loc,
            tensor::ExtractOp::create(rewriter, loc, tableConst, ValueRange{biased}).getResult()
        );

        rewriter.replaceOp(silu->outRescaleOp, lookupOp.getResult(0));
        // Erase explicitly so the driver cannot hand this pattern the same op again.
        // The identity rescales and the old table lose their last use with it.
        rewriter.eraseOp(mulOp);
        return success();
    }
};

} // namespace

void populatePreCalcSiluI8Patterns(MLIRContext *context, RewritePatternSet &patterns) {
    if (!clDisablePreCalcSiluI8)
        patterns.add<PreCalcSiluI8Pattern>(context);
}

} // namespace mlir::syna::torq
