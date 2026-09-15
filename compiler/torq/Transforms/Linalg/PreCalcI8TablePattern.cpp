// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"
#include "Patterns.h"

#include "torq/Utils/TorqMatcherBase.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <limits>

#define DEBUG_TYPE "torq-precalc-i8-table-pattern"

namespace mlir::syna::torq {

namespace {

static llvm::cl::opt<bool> clDisablePreCalcI8Table(
    "torq-disable-precalc-i8-table",
    llvm::cl::desc("Keep an i8-in/i8-out elementwise region as the ops the frontend emitted "
                   "instead of precomputing it into a single 256-entry lookup table. The "
                   "precomputation is on by default; this is the escape hatch for a graph whose "
                   "LRAM allocation it pushes over the limit"),
    llvm::cl::init(false)
);

/// An i8 activation takes 256 values, so one entry per value reproduces any function of it.
constexpr int kTableSize = 256;
/// What turns a signed i8 into a table index, matching how TosaToLinalg indexes an i8
/// tosa.table (TosaToLinalg.cpp, TableConverter i8 path).
constexpr int kTableBias = 128;

/// TOSA apply_scale_32, as TosaToArith lowers it (TosaToArith.cpp,
/// ApplyScaleGenericOpConverter). `round` follows the lowering's `(1 << shift) >> 1`, which
/// is 0 rather than undefined at shift 0.
static int32_t
constantFoldApplyScale32(int64_t value, int32_t multiplier, int32_t shift, bool doubleRound) {
    assert(shift >= 0 && shift <= 62 && "apply_scale shift out of the TOSA range");
    int64_t round = shift > 0 ? (int64_t(1) << (shift - 1)) : 0;
    if (doubleRound && shift > 31) {
        round += value >= 0 ? (int64_t(1) << 30) : -(int64_t(1) << 30);
    }
    int64_t result = (value * int64_t(multiplier) + round) >> shift;
    return static_cast<int32_t>(std::clamp(
        result, int64_t(std::numeric_limits<int32_t>::min()),
        int64_t(std::numeric_limits<int32_t>::max())
    ));
}

/// Predicate for the matcher's element-type checks. Not to be confused with PatternUtils'
/// isI8Type(Value), which peels a shaped type first; the matcher hands over an element type
/// already.
static bool isI8Elem(Type type) { return type.isInteger(8); }

/// The value an elementwise generic yields.
static Value yieldedValue(linalg::GenericOp op) {
    return cast<linalg::YieldOp>(op.getBody()->getTerminator()).getOperand(0);
}

/// Bit width of an integer or index type, as the width checks below use it.
static unsigned scalarWidth(Type type) {
    if (isa<IndexType>(type))
        return 64;
    return type.getIntOrFloatBitWidth();
}

/// Whether `value` is representable as a signed integer of `width` bits.
static bool fitsSigned(int64_t value, unsigned width) {
    if (width >= 64)
        return true;
    int64_t limit = int64_t(1) << (width - 1);
    return value >= -limit && value < limit;
}

/// The single scalar a shaped constant stands for, if it has one: either a splat or a tensor
/// whose dimensions are all unit, as a rank-reduced broadcast constant is.
static std::optional<int64_t> constantScalar(Value value) {
    DenseElementsAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
        return std::nullopt;
    if (!attr.getElementType().isIntOrIndex())
        return std::nullopt;
    if (attr.isSplat())
        return attr.getSplatValue<APInt>().getSExtValue();
    if (attr.getNumElements() == 1)
        return (*attr.value_begin<APInt>()).getSExtValue();
    return std::nullopt;
}

/// Replays an i8-in/i8-out elementwise region at compile time, one activation value at a
/// time.
///
/// Intermediates are carried at 64 bits and only an explicit arith.trunci narrows them,
/// because the widths in this IR are carriers for the backend matchers rather than
/// truncating operations. CastI32MulPattern narrows an i32 multiply to `arith.muli : i16`
/// after checking only that each arm fits i16 (CastI32MulPattern.cpp, isRescaleSafeForI16),
/// not that their product does -- and the product of two arms that do fit can exceed i16.
/// Measured on the simulator, the hardware still computes such a product wide: with the fold
/// disabled the torq result matches the llvm-cpu reference bit for bit. So a value an
/// explicit trunci cannot hold is one this replay cannot predict, and the fold declines
/// rather than guessing at the wrap.
class RegionEvaluator {
  public:
    explicit RegionEvaluator(Value source) : source(source) {}

    /// The value `root` yields for one i8 activation value, or nullopt if the region uses
    /// something this evaluator cannot replay.
    std::optional<int64_t> evaluate(Value root, int8_t activation) {
        env.clear();
        env.try_emplace(source, int64_t(activation));
        return evalValue(root);
    }

  private:
    /// A tensor-level value, collapsed to the one scalar it carries.
    std::optional<int64_t> evalValue(Value value) {
        auto cached = env.find(value);
        if (cached != env.end())
            return cached->second;

        std::optional<int64_t> result;
        if (auto scalar = constantScalar(value)) {
            result = scalar;
        }
        else if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
            result = evalValue(expand.getSrc());
        }
        else if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
            result = evalValue(collapse.getSrc());
        }
        else if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
            result = evalGeneric(generic);
        }
        if (!result)
            return std::nullopt;
        env.try_emplace(value, *result);
        return result;
    }

    /// One elementwise generic: bind its inputs, then walk its body.
    ///
    /// The structure is not re-checked here. Every generic this reaches was matched at match
    /// time -- a region member by isElementwiseOnVarying, a constant arm by
    /// isCompileTimeScalar -- and the answer does not depend on the activation, so repeating
    /// it once per activation value would only rebuild the same matcher 256 times.
    std::optional<int64_t> evalGeneric(linalg::GenericOp op) {
        Block *body = op.getBody();
        for (auto [operand, arg] : llvm::zip(op.getInputs(), body->getArguments())) {
            auto scalar = evalValue(operand);
            if (!scalar)
                return std::nullopt;
            env.try_emplace(arg, *scalar);
        }

        for (Operation &bodyOp : body->without_terminator()) {
            auto result = evalBodyOp(&bodyOp);
            if (!result)
                return std::nullopt;
            env.try_emplace(bodyOp.getResult(0), *result);
        }
        return evalScalar(yieldedValue(op));
    }

    /// A scalar bound by a block argument or an earlier body op, or one the body captured.
    /// The frontend hoists the scale and zero-point constants out of the bodies that use
    /// them, so a captured constant is the common case rather than the exception.
    std::optional<int64_t> evalScalar(Value value) {
        auto it = env.find(value);
        if (it != env.end())
            return it->second;
        if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                return intAttr.getValue().getSExtValue();
        return std::nullopt;
    }

    /// The closed set of body ops a quantized activation region is built from. Anything else
    /// fails the match rather than being approximated.
    std::optional<int64_t> evalBodyOp(Operation *op) {
        if (op->getNumResults() != 1)
            return std::nullopt;
        unsigned width = scalarWidth(op->getResult(0).getType());

        if (isa<arith::ConstantOp>(op))
            return evalScalar(op->getResult(0));
        if (auto extract = dyn_cast<tensor::ExtractOp>(op))
            return evalExtract(extract);
        if (auto applyScale = dyn_cast<tosa::ApplyScaleOp>(op))
            return evalApplyScale(applyScale);

        // Everything below takes its operands from the environment.
        SmallVector<int64_t> operands;
        for (Value operand : op->getOperands()) {
            auto scalar = evalScalar(operand);
            if (!scalar)
                return std::nullopt;
            operands.push_back(*scalar);
        }

        // A widening cast keeps the value the carrier already holds.
        if (isa<arith::ExtSIOp, arith::IndexCastOp>(op))
            return operands[0];
        if (isa<arith::ExtUIOp>(op)) {
            unsigned from = scalarWidth(op->getOperand(0).getType());
            if (from >= 64)
                return operands[0];
            return int64_t(uint64_t(operands[0]) & ((uint64_t(1) << from) - 1));
        }
        if (isa<arith::TruncIOp>(op)) {
            if (!fitsSigned(operands[0], width))
                return std::nullopt;
            return operands[0];
        }
        int64_t result = 0;
        if (isa<arith::AddIOp>(op)) {
            if (__builtin_add_overflow(operands[0], operands[1], &result))
                return std::nullopt;
            return result;
        }
        if (isa<arith::SubIOp>(op)) {
            if (__builtin_sub_overflow(operands[0], operands[1], &result))
                return std::nullopt;
            return result;
        }
        if (isa<arith::MulIOp>(op)) {
            if (__builtin_mul_overflow(operands[0], operands[1], &result))
                return std::nullopt;
            return result;
        }
        if (isa<arith::MaxSIOp>(op))
            return std::max(operands[0], operands[1]);
        if (isa<arith::MinSIOp>(op))
            return std::min(operands[0], operands[1]);
        return std::nullopt;
    }

    /// A lookup into a constant table. The table is captured from outside the body, so it is
    /// read from the op rather than from the environment.
    std::optional<int64_t> evalExtract(tensor::ExtractOp op) {
        DenseElementsAttr table;
        if (!matchPattern(op.getTensor(), m_Constant(&table)))
            return std::nullopt;
        if (!table.getElementType().isIntOrIndex() || table.getType().getRank() != 1)
            return std::nullopt;
        if (op.getIndices().size() != 1)
            return std::nullopt;

        auto offset = evalScalar(op.getIndices()[0]);
        if (!offset)
            return std::nullopt;
        // A replay must not read past the table the graph carries: an out-of-range index is
        // undefined at runtime, so there is no value to bake in.
        if (*offset < 0 || *offset >= table.getNumElements())
            return std::nullopt;
        return (*(table.value_begin<APInt>() + *offset)).getSExtValue();
    }

    std::optional<int64_t> evalApplyScale(tosa::ApplyScaleOp op) {
        auto value = evalScalar(op.getValue());
        auto multiplier = evalScalar(op.getMultiplier());
        auto shift = evalScalar(op.getShift());
        if (!value || !multiplier || !shift)
            return std::nullopt;
        // Replaying apply_scale needs a shift the specification allows, and a multiplier the
        // i32 operand can hold.
        if (*shift < 0 || *shift > 62 || !fitsSigned(*multiplier, 32))
            return std::nullopt;
        return int64_t(constantFoldApplyScale32(
            *value, static_cast<int32_t>(*multiplier), static_cast<int32_t>(*shift),
            op.getRoundingMode() == tosa::RoundingMode::DOUBLE_ROUND
        ));
    }

    Value source;
    /// Both tensor-level values and body scalars, cleared for each activation value.
    DenseMap<Value, int64_t> env;
};

/// The region a single lookup table can replace: every generic between one varying i8 value
/// and one i8 result.
struct I8TableRegion {
    linalg::GenericOp root;                 ///< yields the i8 result
    Value source;                           ///< the one varying value, of i8 element type
    SmallVector<linalg::GenericOp> members; ///< the generics the rewrite replaces
};

/// Whether `op` is a map RegionEvaluator can replay: one result, every loop parallel, and a
/// body that writes its init rather than reading it. The init block argument holds whatever
/// the output tensor already contains, which a replay has no value for.
static bool isReplayableMap(linalg::GenericOp op) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numOutputs(1).allParallel().match(op))
        return false;
    Block *body = op.getBody();
    return body->getArgument(body->getNumArguments() - 1).use_empty();
}

/// Whether `value` collapses to one scalar without knowing the activation, which is what
/// makes an operand a constant arm rather than part of the region. The recursion mirrors
/// RegionEvaluator::evalValue, so a value this accepts is one the evaluator can fold.
static bool isCompileTimeScalar(Value value, DenseMap<Value, bool> &memo) {
    auto cached = memo.find(value);
    if (cached != memo.end())
        return cached->second;
    // A cycle cannot be folded, and claiming so while the answer is still being computed
    // keeps the recursion finite.
    memo.try_emplace(value, false);

    bool result = false;
    if (constantScalar(value)) {
        result = true;
    }
    else if (auto expand = value.getDefiningOp<tensor::ExpandShapeOp>()) {
        result = isCompileTimeScalar(expand.getSrc(), memo);
    }
    else if (auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>()) {
        result = isCompileTimeScalar(collapse.getSrc(), memo);
    }
    else if (auto generic = value.getDefiningOp<linalg::GenericOp>()) {
        result = isReplayableMap(generic) && llvm::all_of(generic.getInputs(), [&](Value input) {
                     return isCompileTimeScalar(input, memo);
                 });
    }
    memo[value] = result;
    return result;
}

/// Whether `op` maps element i of its varying inputs to element i of its result, which is
/// what lets one table stand for the whole region.
///
/// `inputVaries` says which inputs carry the activation. A constant input is uniform, so any
/// map over it reads the same value and its map is left unconstrained -- which is what a
/// rank-reduced broadcast constant needs.
static bool isElementwiseOnVarying(
    linalg::GenericOp op, ArrayRef<bool> inputVaries, RankedTensorType rootType
) {
    if (!isReplayableMap(op))
        return false;

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || resultType.getShape() != rootType.getShape())
        return false;

    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    if (maps.size() != op.getNumOperands() || maps.size() != inputVaries.size() + 1)
        return false;
    AffineMap identity = AffineMap::getMultiDimIdentityMap(
        cast<linalg::LinalgOp>(op.getOperation()).getNumLoops(), op.getContext()
    );
    // The init map has to be the identity as well, or the result element the body computes
    // is not the one the table would be indexed for.
    if (maps.back() != identity)
        return false;
    for (auto [index, varies] : llvm::enumerate(inputVaries))
        if (varies && maps[index] != identity)
            return false;
    return true;
}

/// Whether the region may take `op` over rather than read its result as the activation. A
/// generic something outside the region also reads has to stay: absorbing it would leave
/// both forms in the graph instead of replacing one.
static bool isAbsorbable(Value operand) {
    auto producer = operand.getDefiningOp<linalg::GenericOp>();
    return producer && producer.getResult(0).hasOneUse();
}

/// Collect the region between `source` and `root`, absorbing every generic on the way.
/// Fails on anything the rewrite cannot reproduce exactly.
static std::optional<I8TableRegion>
collectRegion(linalg::GenericOp root, Value source, RankedTensorType rootType) {
    DenseMap<Value, bool> foldable;
    SetVector<linalg::GenericOp> members;
    SmallVector<linalg::GenericOp> worklist{root};
    members.insert(root);

    while (!worklist.empty()) {
        linalg::GenericOp op = worklist.pop_back_val();

        SmallVector<bool> inputVaries;
        for (Value operand : op.getInputs())
            inputVaries.push_back(!isCompileTimeScalar(operand, foldable));
        if (!isElementwiseOnVarying(op, inputVaries, rootType))
            return std::nullopt;

        for (auto [operand, operandVaries] : llvm::zip(op.getInputs(), inputVaries)) {
            // A constant arm is already known to collapse to one scalar; how it is shaped
            // and how many ops build it do not matter, and it stays outside the region.
            if (!operandVaries)
                continue;
            if (operand == source)
                continue;
            if (!isAbsorbable(operand))
                return std::nullopt;
            auto producer = operand.getDefiningOp<linalg::GenericOp>();
            if (members.insert(producer))
                worklist.push_back(producer);
        }
    }

    // A lone generic is already a single elementwise op; folding it would only rewrite it as
    // a table, and the pattern would then be handed the result again.
    if (members.size() < 2)
        return std::nullopt;
    return I8TableRegion{root, source, members.takeVector()};
}

/// Every value the region could read as its activation: an i8 value shaped like the result,
/// on a path from `root` backward through generics the region may absorb. Deepest last, so a
/// caller trying them in reverse tries the largest region first.
static SmallVector<Value> activationCandidates(linalg::GenericOp root, RankedTensorType rootType) {
    /// A region big enough to exhaust this is not one a quantized activation produces.
    constexpr int kMaxCandidates = 32;

    SetVector<Value> candidates;
    SetVector<linalg::GenericOp> visited;
    SmallVector<linalg::GenericOp> worklist{root};
    visited.insert(root);

    while (!worklist.empty() && candidates.size() < kMaxCandidates) {
        linalg::GenericOp op = worklist.pop_back_val();
        for (Value operand : op.getInputs()) {
            auto operandType = dyn_cast<RankedTensorType>(operand.getType());
            if (operandType && operandType.getShape() == rootType.getShape() &&
                operandType.getElementType().isInteger(8))
                candidates.insert(operand);
            if (!isAbsorbable(operand))
                continue;
            auto producer = operand.getDefiningOp<linalg::GenericOp>();
            if (visited.insert(producer))
                worklist.push_back(producer);
        }
    }
    return candidates.takeVector();
}

/// Walk out from an i8-producing generic to the whole region one table can replace.
static std::optional<I8TableRegion> matchI8TableRegion(linalg::GenericOp root) {
    TorqStructuredOpMatcher<> matcher;
    if (!matcher.numOutputs(1).allParallel().outputElementType(0, isI8Elem).match(root))
        return std::nullopt;
    auto rootType = dyn_cast<RankedTensorType>(root.getResult(0).getType());
    if (!rootType)
        return std::nullopt;

    // Which value is the activation is a choice, not something the walk can read off: an
    // intermediate i8 tensor looks exactly like the activation from the root. Taking the
    // deepest one that works keeps the region as large as it can be, and a region that stops
    // short still leaves the pattern a smaller correct fold on the next round.
    SmallVector<Value> candidates = activationCandidates(root, rootType);
    for (Value source : llvm::reverse(candidates)) {
        if (auto region = collectRegion(root, source, rootType))
            return region;
    }
    return std::nullopt;
}

/// Precompute an i8-in/i8-out elementwise region into a single 256-entry lookup table: the
/// activation takes 256 values, so however many ops the region holds it has only 256
/// possible results. What goes away with the region are the i16/i32 tensors its intermediate
/// ops materialize, not the op count.
struct PreCalcI8TablePattern : public OpRewritePattern<linalg::GenericOp> {
    PreCalcI8TablePattern(MLIRContext *context) : OpRewritePattern(context) {
        setDebugName("PreCalcI8TablePattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp root, PatternRewriter &rewriter) const override {
        auto region = matchI8TableRegion(root);
        if (!region)
            return rewriter.notifyMatchFailure(
                root, "not an i8-in/i8-out elementwise region of one activation"
            );
        LLVM_DEBUG(
            llvm::dbgs() << "[PreCalcI8Table] " << region->members.size()
                         << " generics over activation " << region->source << "\n"
        );

        // Entry i stands for the activation value i - 128, matching how the table body
        // indexes a signed i8.
        RegionEvaluator evaluator(region->source);
        SmallVector<int8_t> table(kTableSize);
        for (int i = 0; i < kTableSize; ++i) {
            auto value = evaluator.evaluate(root.getResult(0), int8_t(i - kTableBias));
            if (!value) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[PreCalcI8Table] not replayable at activation "
                                 << (i - kTableBias) << "\n"
                );
                return rewriter.notifyMatchFailure(root, "region is not replayable");
            }
            table[i] = static_cast<int8_t>(*value);
        }

        // A region that gives the same answer for every activation value is a constant, not
        // a table. Rewriting it leaves a generic that only yields that constant, and the
        // backend has no table op to lower it to, so the ops the frontend emitted have to
        // stay -- they do compile.
        if (llvm::all_equal(table))
            return rewriter.notifyMatchFailure(root, "region does not depend on the activation");

        Location loc = root.getLoc();
        auto outType = cast<RankedTensorType>(root.getResult(0).getType());
        auto i8Type = rewriter.getIntegerType(8);
        rewriter.setInsertionPoint(root);

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
            rewriter, loc, TypeRange{outType}, ValueRange{region->source}, ValueRange{init}, maps,
            iterTypes
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

        rewriter.replaceOp(root, lookupOp.getResult(0));
        // Erase explicitly so the driver cannot hand this pattern the same ops again. A
        // member is discovered while its consumer is being walked, so discovery order puts
        // every consumer before its producer and each erase drops the last use of the next;
        // the old tables and constants lose their last use with them.
        for (linalg::GenericOp member : region->members) {
            if (member != root && member.getResult(0).use_empty())
                rewriter.eraseOp(member);
        }
        return success();
    }
};

} // namespace

void populatePreCalcI8TablePatterns(MLIRContext *context, RewritePatternSet &patterns) {
    if (!clDisablePreCalcI8Table)
        patterns.add<PreCalcI8TablePattern>(context);
}

} // namespace mlir::syna::torq
