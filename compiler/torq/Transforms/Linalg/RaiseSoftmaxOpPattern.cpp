// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Utils/ExecutorAssignment.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "torq/Utils/TorqUtils.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#define DEBUG_TYPE "torq-raise-sofmax"

namespace mlir::syna::torq {

namespace {

// Walk forward through addi-tree consumers from `start` until `expected` is
// reached. The addi-tree top forks (reciprocal-multiply also reads it), so
// single-use is required only to advance past a node, not when checking
// arrival.
bool walkAddiChainTo(Value start, linalg::GenericOp expected, int maxDepth = 10) {
    Value current = start;
    for (int hop = 0; hop < maxDepth; ++hop) {
        if (llvm::is_contained(current.getUsers(), expected.getOperation()))
            return true;
        if (!current.hasOneUse())
            return false;
        auto genericOp = dyn_cast<linalg::GenericOp>(*current.getUsers().begin());
        if (!genericOp)
            return false;
        if (!cast<linalg::YieldOp>(genericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::AddIOp>())
            return false;
        current = genericOp.getResult(0);
    }
    return false;
}

// Collect the table-form generics consuming `i16Input` as their lookup input.
SmallVector<std::pair<linalg::GenericOp, LinalgTableMatchInfo>, 4> gatherTableUsers(Value i16Input
) {
    SmallVector<std::pair<linalg::GenericOp, LinalgTableMatchInfo>, 4> tables;
    for (Operation *user : i16Input.getUsers()) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(user)) {
            LinalgTableMatchInfo info;
            if (succeeded(matchI16InterpolatedTable(genericOp, info)) && info.input == i16Input)
                tables.emplace_back(genericOp, info);
        }
    }
    return tables;
}

// One of the four i16 sub-tables in TFLite's int8 softmax exp() approximation.
struct TflInt8SoftmaxLane {
    int shift;
    bool isRight;
    DenseIntElementsAttr lut;        // tensor<513xi16>
    linalg::GenericOp shiftConsumer; // shli/shrsi generic feeding the addi tree
};

// Build one lane from a table op and its associated match info.
FailureOr<TflInt8SoftmaxLane>
extractLane(linalg::GenericOp tableOp, const LinalgTableMatchInfo &info) {
    assert(info.lutData.getNumElements() == 513 && "guaranteed by matchI16InterpolatedTable");
    for (Operation *user : tableOp.getResult(0).getUsers()) {
        auto genericOp = dyn_cast<linalg::GenericOp>(user);
        if (!genericOp || genericOp.getNumDpsInputs() != 2)
            continue;
        for (Operation &innerOp : genericOp.getBody()->getOperations()) {
            bool isRight = isa<arith::ShRSIOp>(innerOp);
            if (!isRight && !isa<arith::ShLIOp>(innerOp))
                continue;
            APInt shiftAmt;
            if (!matchPattern(genericOp.getInputs()[1], m_ConstantInt(&shiftAmt))) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[SoftmaxTflite] table shift consumer has non-constant shift\n"
                );
                return failure();
            }
            return TflInt8SoftmaxLane{
                static_cast<int>(shiftAmt.getSExtValue()),
                isRight,
                info.lutData,
                genericOp,
            };
        }
    }
    LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] table result has no shift consumer\n");
    return failure();
}

// Verify the four-lane shift signature is exactly {17L, 9L, 1L, 7R}.
LogicalResult verifyLaneShiftSignature(ArrayRef<TflInt8SoftmaxLane> lanes) {
    assert(lanes.size() == 4);
    auto count = [&](bool isRight, int shift) {
        return llvm::count_if(lanes, [=](const TflInt8SoftmaxLane &lane) {
            return lane.isRight == isRight && lane.shift == shift;
        });
    };
    if (count(false, 17) != 1 || count(false, 9) != 1 || count(false, 1) != 1 ||
        count(true, 7) != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] lane shifts don't match {17L, 9L, 1L, 7R}\n");
        return failure();
    }
    return success();
}

// Verify every lane's addi-tree path converges to `expected`.
LogicalResult
verifyLanesConvergeAt(ArrayRef<TflInt8SoftmaxLane> lanes, linalg::GenericOp expected) {
    for (const auto &lane : lanes) {
        linalg::GenericOp consumer = lane.shiftConsumer;
        if (!walkAddiChainTo(consumer.getResult(0), expected)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] lane does not converge to the replace target\n"
            );
            return failure();
        }
    }
    return success();
}

// Verify the four table-form generics sharing `i16Input` combine as
// (T_a << 17) + (T_b << 9) + (T_c << 1) + (T_d >> 7), and that the four-lane
// sum is consumed by a rounded right-shift by 12. The shift signature
// reflects how TFLite's reference int8 softmax packs four bytes of a 32-bit
// fixed-point exp value across the LUTs. See TF's
// `getTosaConst32bitSoftmaxExpTable` in legalize_utils.cc for the byte layout.
// Returns one lane per table in gather order.
FailureOr<SmallVector<TflInt8SoftmaxLane, 4>>
verifyTflInt8SoftmaxFourTables(Value i16Input, linalg::GenericOp expectedRoundShift) {
    auto tables = gatherTableUsers(i16Input);
    if (tables.size() != 4) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SoftmaxTflite] expected 4 table users sharing i16 input, got "
                         << tables.size() << "\n"
        );
        return failure();
    }

    SmallVector<TflInt8SoftmaxLane, 4> lanes;
    for (auto &[tableOp, info] : tables) {
        auto lane = extractLane(tableOp, info);
        if (failed(lane))
            return failure();
        lanes.push_back(*lane);
    }

    if (failed(verifyLaneShiftSignature(lanes)))
        return failure();
    if (failed(verifyLanesConvergeAt(lanes, expectedRoundShift)))
        return failure();
    return lanes;
}

// Recover `input_scale * beta` (bf16) from the four sub-tables. The combine
// output at LUT index `idx` reproduces what the integer pipeline computes for
// i16 input value (idx - kLutMidpoint) * kI16Step, and approximates
//   peakValue * exp(i16Input * scalePerI16Unit),
// so the ratio of two indices recovers scalePerI16Unit. Multiplying by
// kI16Step converts that back to the i8 input scale (the integer pipeline
// rescales i8 by 128 before reaching the LUT). This mirrors TF's int8 softmax
// legalization exactly. Only meaningful for IR that came through that pipeline.
// Caller must have already verified the lane signature.
FailureOr<double> recoverInputScale(ArrayRef<TflInt8SoftmaxLane> lanes) {
    constexpr int kLutMidpoint = 256;
    constexpr int kI16Step = 128;
    constexpr int kSampleIdx = 192;
    constexpr int64_t kSampleI16Input = (kSampleIdx - kLutMidpoint) * kI16Step; // -8192

    auto combineLanesAt = [](ArrayRef<TflInt8SoftmaxLane> lanes, int idx) -> int64_t {
        int64_t sum = 0;
        for (const auto &lane : lanes) {
            int64_t base = lane.lut.getValues<APInt>()[idx].getSExtValue();
            int64_t v = base << 7; // matches the <<7 inside the table body
            sum += lane.isRight ? (v >> lane.shift) : (v << lane.shift);
        }
        return (sum + (1LL << 11)) >> 12; // round-right-shift by 12 (verified by caller)
    };

    int64_t peakValue = combineLanesAt(lanes, kLutMidpoint); // i16 input = 0, exp(0) = 1
    if (peakValue <= 0) {
        LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] combine at LUT midpoint is not positive\n");
        return failure();
    }
    int64_t refValue = combineLanesAt(lanes, kSampleIdx); // well within typical LUT range
    if (refValue <= 0 || refValue >= peakValue) {
        LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] combine at sample idx out of expected range\n");
        return failure();
    }

    double ratio = static_cast<double>(refValue) / static_cast<double>(peakValue);
    double scalePerI16Unit = std::log(ratio) / kSampleI16Input;
    double bf16Scale = scalePerI16Unit * kI16Step;

    // Typical TFLite int8 softmax scales fall in [0.001, 0.5].
    if (bf16Scale <= 0.001 || bf16Scale >= 0.5) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SoftmaxTflite] recovered scale " << bf16Scale
                         << " is outside [0.001, 0.5]\n"
        );
        return failure();
    }
    return bf16Scale;
}

} // namespace

bool checkIdentityLikeIndexingMaps(linalg::GenericOp op) {
    for (auto [operandIdx, indexingMap] : llvm::enumerate(op.getIndexingMapsArray())) {
        if (indexingMap.isIdentity()) {
            continue;
        }

        for (auto [idx, result] : llvm::enumerate(indexingMap.getResults())) {
            if (auto dim = dyn_cast<AffineDimExpr>(result)) {
                if (dim.getPosition() != idx) {
                    return false;
                }
            }
            else if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
                if (constExpr.getValue() != 0) {
                    return false;
                }
                auto operandType = op.getOperand(operandIdx).getType();
                if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
                    if (shapedType.getDimSize(idx) != 1) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

template <class T> SmallVector<Value> findElementwiseOpInputs(Value value) {
    SmallVector<Value> inputs;
    auto op = value.getDefiningOp<linalg::GenericOp>();
    if (!op) {
        return inputs;
    }
    // Make sure there is only 1 output
    if (op.getNumDpsInits() != 1) {
        return inputs;
    }
    // Make sure there is no permutations or other indexing maps than identity
    if (!checkIdentityLikeIndexingMaps(op)) {
        return inputs;
    }
    // Make sure the terminator of the generic op returns the result of a truncf op
    auto innerOp = op.getBody()->getTerminator()->getOperand(0).getDefiningOp<T>();
    if (!innerOp) {
        return inputs;
    }
    if (innerOp->getNumOperands() != op.getNumDpsInputs()) {
        return inputs;
    }
    for (unsigned i = 0; i < op.getNumDpsInputs(); ++i) {
        auto innerInput = dyn_cast<BlockArgument>(innerOp->getOperands()[i]);
        // Make sure the inner op input(s) are the input(s) of the generic op
        if (!innerInput || innerInput.getArgNumber() != i ||
            innerInput.getOwner() != op.getBody()) {
            return inputs;
        }
        inputs.push_back(op.getInputs()[i]);
    }
    return inputs;
}

struct ReductionOpMatch {
    SmallVector<Value> inputs;
    unsigned reductionDim;
};

template <class T> std::optional<ReductionOpMatch> matchReductionOp(Value value) {
    SmallVector<Value> inputs;
    auto op = value.getDefiningOp<linalg::GenericOp>();
    if (!op) {
        return std::nullopt;
    }

    auto result = dyn_cast<OpResult>(value);
    if (!result || result.getOwner() != op.getOperation()) {
        return std::nullopt;
    }
    unsigned resultNumber = result.getResultNumber();

    SmallVector<unsigned> reductionDims;
    for (auto [i, iteratorType] : llvm::enumerate(op.getIteratorTypesArray())) {
        if (iteratorType == utils::IteratorType::reduction) {
            reductionDims.push_back(i);
        }
    }
    if (reductionDims.size() != 1) {
        return std::nullopt;
    }

    if (resultNumber >= op.getNumDpsInits()) {
        return std::nullopt;
    }

    // Make sure there is no permutations or other indexing maps than identity
    if (!checkIdentityLikeIndexingMaps(op)) {
        return std::nullopt;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || resultNumber >= yieldOp.getNumOperands()) {
        return std::nullopt;
    }

    auto innerOp = yieldOp.getOperand(resultNumber).getDefiningOp<T>();
    if (!innerOp) {
        return std::nullopt;
    }

    unsigned expectedOutputArg = op.getNumDpsInputs() + resultNumber;
    for (Value operand : innerOp->getOperands()) {
        auto innerOperand = dyn_cast<BlockArgument>(operand);
        // Make sure the inner op operands are the operands of the generic op
        if (!innerOperand || innerOperand.getOwner() != op.getBody()) {
            return std::nullopt;
        }

        unsigned argNumber = innerOperand.getArgNumber();
        if (argNumber >= op.getNumOperands()) {
            return std::nullopt;
        }

        OpOperand &opOperand = op->getOpOperand(argNumber);
        if (op.isDpsInput(&opOperand)) {
            inputs.push_back(opOperand.get());
            continue;
        }

        if (argNumber != expectedOutputArg) {
            return std::nullopt;
        }
    }

    if (inputs.empty()) {
        return std::nullopt;
    }

    return ReductionOpMatch{inputs, reductionDims.front()};
}

template <class T>
SmallVector<Value> findReductionOpInputs(Value value, unsigned int reductionDim) {
    auto match = matchReductionOp<T>(value);
    if (!match || match->reductionDim != reductionDim) {
        return {};
    }
    return match->inputs;
}

/// Matches a linalg.generic operation that broadcasts along the last dimension.
/// Pattern:
///   %broadcast = linalg.generic
///     {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
///                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
///      iterator_types = ["parallel", "parallel", "parallel"]}
///     ins(%input : tensor<1x1916xf32>) outs(%out : tensor<1x1916x2xf32>) {
///       ^bb0(%in: f32, %out: f32):
///         linalg.yield %in : f32
///     } -> tensor<1x1916x2xf32>
///
/// @param value The value to check if it's a broadcast operation
/// @return The input to the broadcast operation if matched, nullptr otherwise
Value matchBroadcastOp(Value value) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp) {
        return nullptr;
    }

    // Should have exactly one input
    if (genericOp.getNumDpsInputs() != 1) {
        return nullptr;
    }

    // Check that body just yields the input
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return nullptr;
    }

    auto blockArg = dyn_cast<BlockArgument>(yieldOp.getOperand(0));
    if (!blockArg || blockArg.getArgNumber() != 0 || blockArg.getOwner() != genericOp.getBody()) {
        return nullptr;
    }

    // Verify it's a broadcast along the last dimension
    auto inputType = dyn_cast<RankedTensorType>(genericOp.getInputs()[0].getType());
    auto outputType = dyn_cast<RankedTensorType>(genericOp.getResult(0).getType());

    if (!inputType || !outputType) {
        return nullptr;
    }

    // Output should have one more dimension than input
    if (outputType.getRank() != inputType.getRank() + 1) {
        return nullptr;
    }

    return genericOp.getInputs()[0];
}

/// Matches a tensor.expand_shape operation that adds a trailing dimension of size 1.
/// Supports two patterns:
/// 1. Splitting last dimension [1] into [1, 1]
/// 2. Splitting last dimension [N] into [N, 1]
///
/// Pattern 1:
///   %expanded = tensor.expand_shape %input [[0], [1], ..., [n-1], [n, n+1]]
///   : tensor<d0xd1x...xd_{n-1}x1xf32> into tensor<d0xd1x...xd_{n-1}x1x1xf32>
///
/// Pattern 2:
///   %expanded = tensor.expand_shape %input [[0], [1, 2]]
///   : tensor<1x1916xf32> into tensor<1x1916x1xf32>
///
/// @param value The value to check if it's an expand_shape operation
/// @return The input to the expand_shape operation if matched, nullptr otherwise
Value matchExpandShapeOp(Value value) {
    auto expandOp = value.getDefiningOp<tensor::ExpandShapeOp>();
    if (!expandOp) {
        return nullptr;
    }

    auto inputType = dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
    if (!inputType) {
        return nullptr;
    }

    auto outputType = dyn_cast<RankedTensorType>(expandOp.getResult().getType());
    if (!outputType) {
        return nullptr;
    }

    // Verify output rank is exactly one more than input rank
    if (outputType.getRank() != inputType.getRank() + 1) {
        return nullptr;
    }

    auto reassociations = expandOp.getReassociationIndices();
    int inputRank = inputType.getRank();

    // Verify the reassociation groups: [[0], [1], ..., [n-1], [n, n+1]]
    // Should have (inputRank) groups total
    if (reassociations.size() != inputRank) {
        return nullptr;
    }

    // Check all groups except the last one: each should contain exactly one dimension
    for (int i = 0; i < inputRank - 1; ++i) {
        if (reassociations[i].size() != 1 || reassociations[i][0] != i) {
            return nullptr;
        }
    }

    // Check the last group: should contain exactly two consecutive dimensions
    int lastGroupIdx = inputRank - 1;
    if (reassociations[lastGroupIdx].size() != 2 ||
        reassociations[lastGroupIdx][0] != lastGroupIdx ||
        reassociations[lastGroupIdx][1] != lastGroupIdx + 1) {
        return nullptr;
    }

    // Verify the last dimension of output is 1 (the newly added trailing dimension)
    if (outputType.getDimSize(inputRank) != 1) {
        return nullptr;
    }

    // Verify the second-to-last output dimension matches the last input dimension
    if (outputType.getDimSize(inputRank - 1) != inputType.getDimSize(inputRank - 1)) {
        return nullptr;
    }

    return expandOp.getSrc();
}

/// Matches either a tensor.expand_shape or linalg.generic broadcast operation
/// @param value The value to check
/// @return The input to the broadcast/expand operation if matched, nullptr otherwise
Value matchBroadcastOrExpandShape(Value value) {
    Value result = matchExpandShapeOp(value);
    if (result) {
        return result;
    }
    return matchBroadcastOp(value);
}

/// Matches a linalg.generic that computes reciprocal: 1.0 / input
/// The constant 1.0 can be either:
/// 1. A captured value from outside the block (e.g., %cst_3)
/// 2. A constant defined inside the block
///
/// Pattern:
///   %recip = linalg.generic ins(%sum) outs(%out) {
///     ^bb0(%in: f32, %out: f32):
///       %result = arith.divf %cst_one, %in : f32
///       linalg.yield %result : f32
///   }
///
/// @param value The value to check if it's a reciprocal operation
/// @return The input to the reciprocal (the denominator) if matched, nullptr otherwise
Value matchReciprocalOp(Value value) {
    auto genericOp = value.getDefiningOp<linalg::GenericOp>();
    if (!genericOp) {
        return nullptr;
    }

    // Should have exactly one input
    if (genericOp.getNumDpsInputs() != 1) {
        return nullptr;
    }

    // Should have exactly one output
    if (genericOp.getNumDpsInits() != 1) {
        return nullptr;
    }

    // Check the body contains divf yielded directly
    auto yieldOp = dyn_cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        return nullptr;
    }

    auto divOp = yieldOp.getOperand(0).getDefiningOp<arith::DivFOp>();
    if (!divOp) {
        return nullptr;
    }

    // Check that the divisor (rhs) is the block argument corresponding to the input
    auto rhsArg = dyn_cast<BlockArgument>(divOp.getRhs());
    if (!rhsArg || rhsArg.getArgNumber() != 0 || rhsArg.getOwner() != genericOp.getBody()) {
        return nullptr;
    }

    // Check that the numerator (lhs) is a constant 1.0
    // It could be a captured value from outside or a constant inside the block
    Value lhs = divOp.getLhs();

    // Try to get the constant value - could be from inside or outside the block
    FloatAttr constValue;
    if (auto constOp = lhs.getDefiningOp<arith::ConstantOp>()) {
        constValue = dyn_cast<FloatAttr>(constOp.getValue());
    }

    if (!constValue || !constValue.getValue().isExactlyValue(1.0)) {
        return nullptr;
    }

    return genericOp.getInputs()[0];
}

class RaiseSoftmaxOnnx : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {

        auto truncInput = findElementwiseOpInputs<arith::TruncFOp>(op.getResult(0));
        if (truncInput.size() != 1) {
            // with --torq-convert-dtypes, there is no truncf.  Fine to ignore
            truncInput = {op.getResult(0)};
        }
        auto divInputs = findElementwiseOpInputs<arith::DivFOp>(truncInput[0]);
        if (divInputs.size() != 2) {
            return failure();
        }
        auto expInputs = findElementwiseOpInputs<math::ExpOp>(divInputs[0]);
        if (expInputs.size() != 1) {
            return failure();
        }
        auto addMatch = matchReductionOp<arith::AddFOp>(divInputs[1]);
        if (!addMatch || addMatch->inputs.size() != 1) {
            // with --torq-convert-dtypes, the pattern here looks a bit
            // different.  Check for that case
            auto emptyGeneric = divInputs[1].getDefiningOp<linalg::GenericOp>();
            if (!emptyGeneric)
                return failure();
            auto reshape = emptyGeneric.getInputs()[0].getDefiningOp<tensor::ReshapeOp>();
            if (!reshape)
                return failure();
            addMatch = matchReductionOp<arith::AddFOp>(reshape.getOperands()[0]);
            if (!addMatch || addMatch->inputs.size() != 1)
                return failure();
        }
        if (addMatch->inputs[0] != divInputs[0]) {
            return failure();
        }
        auto subInputs = findElementwiseOpInputs<arith::SubFOp>(expInputs[0]);
        if (subInputs.size() != 2) {
            return failure();
        }
        auto expandInput = matchExpandShapeOp(subInputs[1]);
        if (!expandInput) {
            return failure();
        }
        auto maxInputs =
            findReductionOpInputs<arith::MaximumFOp>(expandInput, addMatch->reductionDim);
        if (maxInputs.size() != 1) {
            return failure();
        }
        if (maxInputs[0] != subInputs[0]) {
            return failure();
        }
        auto extInputs = findElementwiseOpInputs<arith::ExtFOp>(subInputs[0]);
        if (extInputs.size() != 1) {
            // with --torq-convert-dtypes, there is no extf.  Fine to ignore.
            extInputs = maxInputs;
        }

        rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
            op, op.getResultTypes(), extInputs[0], op.getOutputs()[0], addMatch->reductionDim
        );

        return success();
    }
};

/// Matches the decomposed softmax pattern and rewrites it to linalg.softmax
///
/// Softmax formula: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
///
/// Pattern matched (working backwards from result):
///   %result = mul(%exp, %recip)
///   %recip = 1.0 / %sum_expanded
///   %sum_expanded = expand_shape(%sum)
///   %sum = reduce_add(%exp)
///   %exp = exp(%sub)
///   %sub = %input - %max_expanded
///   %max_expanded = expand_shape(%max)
///   %max = reduce_max(%input)
///
/// Constraints verified:
///   - sum and max reduce the same dimension
///   - max and sub operate on the same input
///   - sum reduces the exp output
///
/// If the original tosa.mul has shift != 0, it lowers to arith.mulf + shift operations, which
/// won't match the simple arith::MulFOp pattern here.
class RaiseDecomposedSoftmax : public OpRewritePattern<linalg::GenericOp> {
  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        // Match the final multiplication (exp * reciprocal)
        auto mulInputs = findElementwiseOpInputs<arith::MulFOp>(op.getResult(0));
        if (mulInputs.size() != 2) {
            return failure();
        }

        // Multiplication is commutative, so try both orderings
        // First try: mulInputs[0] = exp, mulInputs[1] = recip
        // Second try: mulInputs[0] = recip, mulInputs[1] = exp
        for (int ordering = 0; ordering < 2; ++ordering) {
            Value expValue = mulInputs[ordering];
            Value recipBroadcastValue = mulInputs[1 - ordering];

            // Match reciprocal broadcast (could be a linalg.generic that broadcasts a collapsed
            // reciprocal)
            Value recipValue = matchBroadcastOrExpandShape(recipBroadcastValue);
            if (!recipValue) {
                recipValue = recipBroadcastValue;
            }
            else {
                auto collapseOp = recipValue.getDefiningOp<tensor::CollapseShapeOp>();
                if (collapseOp) {
                    recipValue = collapseOp.getSrc();
                }
            }

            // Match reciprocal computation (1.0 / sum)
            Value sumExpandedValue = matchReciprocalOp(recipValue);
            if (!sumExpandedValue) {
                continue; // Try other ordering
            }

            // Match sum expansion (tensor.expand_shape)
            Value sumReducedValue = matchExpandShapeOp(sumExpandedValue);
            if (!sumReducedValue) {
                continue;
            }

            // Match sum reduction (linalg.reduce with addf)
            auto sumReduceOp = sumReducedValue.getDefiningOp<linalg::ReduceOp>();
            if (!sumReduceOp) {
                continue;
            }

            auto sumYield = dyn_cast<linalg::YieldOp>(sumReduceOp.getBody()->getTerminator());
            if (!sumYield || sumYield.getNumOperands() != 1) {
                continue;
            }
            auto sumAddOp = sumYield.getOperand(0).getDefiningOp<arith::AddFOp>();
            if (!sumAddOp) {
                continue;
            }

            auto sumDimensions = sumReduceOp.getDimensions();
            if (sumDimensions.size() != 1) {
                continue;
            }
            int64_t reductionDim = sumDimensions[0];

            // Verify input to sum is exp
            if (sumReduceOp.getInputs()[0] != expValue) {
                continue;
            }

            // Match exponential (math.exp)
            auto expInputs = findElementwiseOpInputs<math::ExpOp>(expValue);
            if (expInputs.size() != 1) {
                continue;
            }
            Value subValue = expInputs[0];

            // Match subtraction (input - max)
            auto subInputs = findElementwiseOpInputs<arith::SubFOp>(subValue);
            if (subInputs.size() != 2) {
                continue;
            }
            Value inputValue = subInputs[0];
            Value maxBroadcastValue = subInputs[1];

            // Match max broadcast (tensor.expand_shape or linalg.generic broadcast)
            Value maxReducedValue = matchBroadcastOrExpandShape(maxBroadcastValue);
            if (!maxReducedValue) {
                continue;
            }

            // Match max reduction (linalg.reduce with maximumf)
            auto maxReduceOp = maxReducedValue.getDefiningOp<linalg::ReduceOp>();
            if (!maxReduceOp) {
                continue;
            }

            auto maxYield = dyn_cast<linalg::YieldOp>(maxReduceOp.getBody()->getTerminator());
            if (!maxYield || maxYield.getNumOperands() != 1) {
                continue;
            }
            auto maxOp = maxYield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
            if (!maxOp) {
                continue;
            }

            auto maxDimensions = maxReduceOp.getDimensions();
            if (maxDimensions.size() != 1 || maxDimensions[0] != reductionDim) {
                continue;
            }

            // Verify input to max is the same as input to subtraction
            if (maxReduceOp.getInputs()[0] != inputValue) {
                continue;
            }

            // All constraints matched - rewrite to linalg.softmax
            auto softmaxOp = rewriter.replaceOpWithNewOp<linalg::SoftmaxOp>(
                op, op.getResultTypes(), inputValue, op.getOutputs()[0], reductionDim
            );
            setTargetExecutorIfForced(softmaxOp, rewriter, "softmax");
            return success();
        }

        return failure();
    }
};

// Matches the linalg.generic shape TFLite legalizes int8 softmax to:
//
//   i8 input -> scale (i8->i32) -> max-reduce + sub-max -> narrow (i32->i16)
//     -> four tosa.table i16 LUTs combined as (T_a<<17)+(T_b<<9)+(T_c<<1)+(T_d>>7)
//        and round-shifted right by 12
//     -> sum-reduce + ctlz + reciprocal multiply -> rounded right-shift
//     -> apply_scale(mult=2^30, shift=30) + addi(-128) -> i8 result
//
// Reference (TF, convertSoftmaxOp int8 path + getTosaConst32bitSoftmaxExpTable):
//   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc
//   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc
// Canonical example IR: tests/testdata/tosa_ops/mbv2-softmax-i8.mlir
//
class RaiseSoftmaxTflite : public OpRewritePattern<linalg::GenericOp> {
  private:
    struct MatchInfo {
        Value i8Input;                   // i8 tensor entering the pipeline
        int64_t reductionDim;            // softmax axis
        linalg::GenericOp replaceTarget; // op whose result rewriteSoftmax replaces
        double inputScale;               // bf16 cast scale (input_scale * beta)
    };

  public:
    using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter &rewriter) const override {
        auto info = matchSoftmax(op);
        if (failed(info))
            return failure();
        rewriteSoftmax(op, *info, rewriter);
        return success();
    }

  private:
    FailureOr<MatchInfo> matchSoftmax(linalg::GenericOp op) const {
        if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] not single-input single-output generic\n");
            return failure();
        }

        if (auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
            !resultType || !resultType.getElementType().isInteger(8)) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] result is not i8 ranked tensor\n");
            return failure();
        }

        if (!dyn_cast<RankedTensorType>(op.getInputs()[0].getType()) ||
            !dyn_cast<RankedTensorType>(op.getInputs()[0].getType())
                 .getElementType()
                 .isInteger(32)) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] input is not i32 ranked tensor\n");
            return failure();
        }

        if (!cast<linalg::YieldOp>(op.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::TruncIOp>()) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] body does not yield arith.trunci\n");
            return failure();
        }

        // Body must be TFLite-canonical: apply_scale(mult=2^30, shift=30) +
        // addi(ozp=-128). rewriteSoftmax feeds `softmax_prob * 256` (i32) into
        // apply_scale's input expecting it to pass through unchanged, which
        // requires effective scale = mult / 2^shift = 1.
        {
            tosa::ApplyScaleOp applyScale;
            arith::AddIOp addOzp;
            for (Operation &innerOp : op.getBody()->getOperations()) {
                if (auto as = dyn_cast<tosa::ApplyScaleOp>(&innerOp)) {
                    if (applyScale) {
                        LLVM_DEBUG(
                            llvm::dbgs() << "[SoftmaxTflite] multiple tosa.apply_scale in body\n"
                        );
                        return failure();
                    }
                    applyScale = as;
                }
                else if (auto add = dyn_cast<arith::AddIOp>(&innerOp)) {
                    if (addOzp) {
                        LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] multiple arith.addi in body\n");
                        return failure();
                    }
                    addOzp = add;
                }
            }
            if (!applyScale || !addOzp) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[SoftmaxTflite] body missing tosa.apply_scale or arith.addi\n"
                );
                return failure();
            }

            if (APInt c; !matchPattern(applyScale.getMultiplier(), m_ConstantInt(&c)) ||
                         c.getSExtValue() != (1 << 30)) {
                LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] apply_scale multiplier != 2^30\n");
                return failure();
            }
            if (APInt c;
                !matchPattern(applyScale.getShift(), m_ConstantInt(&c)) || c.getSExtValue() != 30) {
                LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] apply_scale shift != 30\n");
                return failure();
            }

            // ozp may be either operand.
            APInt ozp;
            if (!matchPattern(addOzp.getRhs(), m_ConstantInt(&ozp)) &&
                !matchPattern(addOzp.getLhs(), m_ConstantInt(&ozp))) {
                LLVM_DEBUG(
                    llvm::dbgs() << "[SoftmaxTflite] addi has no constant operand for ozp\n"
                );
                return failure();
            }
            if (ozp.getSExtValue() != -128) {
                LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] output zero-point != -128\n");
                return failure();
            }
        }

        // Walking back: rounded-right-shift generic ins(scaled_exp, shift_amount).
        auto shrsiGenericOp = op.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!shrsiGenericOp || shrsiGenericOp.getNumDpsInputs() != 2) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] rounded-right-shift generic feeding op missing\n"
            );
            return failure();
        }
        if (arith::ShRSIOp shrsi; !isRoundingRightShiftOp(shrsiGenericOp, shrsi)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] shrsi generic body is not a rounded right shift\n"
            );
            return failure();
        }

        // shift_amount = subi(cst, ctlz(expand_shape(sum_reduce))).
        auto subiShiftGenericOp = shrsiGenericOp.getInputs()[1].getDefiningOp<linalg::GenericOp>();
        if (!subiShiftGenericOp || subiShiftGenericOp.getNumDpsInputs() != 2 ||
            subiShiftGenericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] shift-amount subi generic missing\n");
            return failure();
        }
        if (!cast<linalg::YieldOp>(subiShiftGenericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::SubIOp>()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] shift-amount body does not yield arith.subi\n"
            );
            return failure();
        }

        auto ctlzGenericOp = subiShiftGenericOp.getInputs()[1].getDefiningOp<linalg::GenericOp>();
        if (!ctlzGenericOp || ctlzGenericOp.getNumDpsInputs() != 1 ||
            ctlzGenericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] ctlz generic missing or wrong shape\n");
            return failure();
        }

        if (!cast<linalg::YieldOp>(ctlzGenericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<math::CountLeadingZerosOp>()) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] ctlz body does not yield math.ctlz\n");
            return failure();
        }

        Value sumReducedValue = matchExpandShapeOp(ctlzGenericOp.getInputs()[0]);
        if (!sumReducedValue) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] could not trace through expand_shape to sum\n"
            );
            return failure();
        }

        auto sumReduceOp = sumReducedValue.getDefiningOp<linalg::ReduceOp>();
        if (!sumReduceOp || sumReduceOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] sum linalg.reduce missing or wrong shape\n"
            );
            return failure();
        }

        if (!cast<linalg::YieldOp>(sumReduceOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::AddIOp>()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] sum-reduce body does not yield arith.addi\n"
            );
            return failure();
        }

        auto sumDimensions = sumReduceOp.getDimensions();
        if (sumDimensions.size() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] sum reduction is not over a single dim\n");
            return failure();
        }
        int64_t reductionDim = sumDimensions[0];

        // post-table-sum rounded right-shift by 12.
        // Its addi-tree input drives both the table-walk start and the lane
        // convergence target.
        auto postSumShiftOp = sumReduceOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!postSumShiftOp || postSumShiftOp.getNumDpsInputs() != 2) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] post-table-sum shrsi generic missing\n");
            return failure();
        }
        if (arith::ShRSIOp shrsi; !isRoundingRightShiftOp(postSumShiftOp, shrsi)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] post-table-sum body is not a rounded right shift\n"
            );
            return failure();
        }
        if (APInt rsAmt; !matchPattern(postSumShiftOp.getInputs()[1], m_ConstantInt(&rsAmt)) ||
                         rsAmt.getSExtValue() != 12) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] post-table-sum round-right-shift amount != 12\n"
            );
            return failure();
        }

        // Walk inputs[0] chain to find the table-form generic holding the exp LUT.
        Value current = postSumShiftOp.getInputs()[0];
        linalg::GenericOp tableGenericOp;
        LinalgTableMatchInfo tableInfo;
        while (auto genericOp = current.getDefiningOp<linalg::GenericOp>()) {
            if (succeeded(matchI16InterpolatedTable(genericOp, tableInfo))) {
                tableGenericOp = genericOp;
                break;
            }
            if (genericOp.getNumDpsInputs() == 0)
                break;
            current = genericOp.getInputs()[0];
        }

        if (!tableGenericOp) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] no table-form linalg.generic found in chain\n"
            );
            return failure();
        }

        // tableInfo.input <- i32->i16 narrow <- subi(scaled_input, max) <- i8->i32 scale <- i8
        // input.
        auto i16GenericOp = tableInfo.input.getDefiningOp<linalg::GenericOp>();
        if (!i16GenericOp || i16GenericOp.getNumDpsInputs() != 1 ||
            i16GenericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] i32->i16 narrow generic missing\n");
            return failure();
        }
        if (!cast<linalg::YieldOp>(i16GenericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::TruncIOp>()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] i32->i16 narrow body does not yield arith.trunci\n"
            );
            return failure();
        }

        auto subMaxGenericOp = i16GenericOp.getInputs()[0].getDefiningOp<linalg::GenericOp>();
        if (!subMaxGenericOp || subMaxGenericOp.getNumDpsInputs() != 2 ||
            subMaxGenericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] sub-max generic missing or wrong shape\n");
            return failure();
        }

        if (!cast<linalg::YieldOp>(subMaxGenericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::SubIOp>()) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] sub-max body does not yield arith.subi\n");
            return failure();
        }

        // subi's second input is the expanded max reduction.
        Value maxReducedValue = matchBroadcastOrExpandShape(subMaxGenericOp.getInputs()[1]);
        if (!maxReducedValue) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[SoftmaxTflite] could not trace through broadcast/expand_shape to max\n"
            );
            return failure();
        }

        auto maxReduceOp = maxReducedValue.getDefiningOp<linalg::ReduceOp>();
        if (!maxReduceOp || maxReduceOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] max linalg.reduce missing or wrong shape\n"
            );
            return failure();
        }

        if (!cast<linalg::YieldOp>(maxReduceOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<arith::MaxSIOp>()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] max-reduce body does not yield arith.maxsi\n"
            );
            return failure();
        }

        // Max and sum must reduce the same dim.
        auto maxDimensions = maxReduceOp.getDimensions();
        if (maxDimensions.size() != 1 || maxDimensions[0] != reductionDim) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] max and sum do not reduce the same dim\n");
            return failure();
        }

        Value scaledInputValue = subMaxGenericOp.getInputs()[0];
        auto scaledInputGenericOp = scaledInputValue.getDefiningOp<linalg::GenericOp>();
        if (!scaledInputGenericOp || scaledInputGenericOp.getNumDpsInputs() != 1 ||
            scaledInputGenericOp.getNumDpsInits() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] scaled-input generic missing\n");
            return failure();
        }
        if (!cast<linalg::YieldOp>(scaledInputGenericOp.getBody()->getTerminator())
                 .getOperand(0)
                 .getDefiningOp<tosa::ApplyScaleOp>()) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "[SoftmaxTflite] scaled-input body does not yield tosa.apply_scale\n"
            );
            return failure();
        }

        // Max-reduce and sub-max must share their input.
        if (maxReduceOp.getInputs()[0] != scaledInputValue) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SoftmaxTflite] max-reduce and sub-max do not share scaled input\n"
            );
            return failure();
        }

        Value inputValue = scaledInputGenericOp.getInputs()[0];
        if (auto inputTensorType = dyn_cast<RankedTensorType>(inputValue.getType());
            !inputTensorType || !inputTensorType.getElementType().isInteger(8)) {
            LLVM_DEBUG(llvm::dbgs() << "[SoftmaxTflite] original input is not i8 ranked tensor\n");
            return failure();
        }

        // The four-table signature is the load-bearing TFLite-specific check.
        auto lanes = verifyTflInt8SoftmaxFourTables(tableInfo.input, postSumShiftOp);
        if (failed(lanes))
            return failure();
        FailureOr<double> extracted = recoverInputScale(*lanes);
        if (failed(extracted))
            return failure();

        return MatchInfo{
            inputValue,
            reductionDim,
            shrsiGenericOp,
            *extracted,
        };
    }

    // Replaces info.replaceTarget with a bf16 detour (i8 -> bf16 cast -> linalg.softmax -> *256 ->
    // fptosi). the matched op's quant arithmetic are not erased here.
    void
    rewriteSoftmax(linalg::GenericOp op, const MatchInfo &info, PatternRewriter &rewriter) const {
        auto loc = op.getLoc();
        auto bf16Type = rewriter.getBF16Type();
        auto i32Type = rewriter.getI32Type();
        auto inputTensorType = cast<RankedTensorType>(info.i8Input.getType());
        auto inputShape = inputTensorType.getShape();
        auto bf16TensorType = RankedTensorType::get(inputShape, bf16Type);
        auto i32TensorType = RankedTensorType::get(inputShape, i32Type);
        auto rank = inputTensorType.getRank();
        auto identityMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
        SmallVector<utils::IteratorType> parallelIterators(rank, utils::IteratorType::parallel);

        // Keep cast and scale in separate generics. `isTorqCastOp` rejects fused form
        // and forces CSS/host instead of slice.

        // i8 -> bf16.
        auto emptyBf16Cast = tensor::EmptyOp::create(rewriter, loc, inputShape, bf16Type);
        auto castToBf16 = linalg::GenericOp::create(
            rewriter, loc, TypeRange{bf16TensorType}, ValueRange{info.i8Input},
            ValueRange{emptyBf16Cast}, ArrayRef<AffineMap>{identityMap, identityMap},
            parallelIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto fp = arith::SIToFPOp::create(b, l, bf16Type, args[0]);
                linalg::YieldOp::create(b, l, fp.getResult());
            }
        );

        // Scale by input_scale * beta.
        auto emptyBf16Scale = tensor::EmptyOp::create(rewriter, loc, inputShape, bf16Type);
        auto scaleToBf16 = linalg::GenericOp::create(
            rewriter, loc, TypeRange{bf16TensorType}, ValueRange{castToBf16.getResult(0)},
            ValueRange{emptyBf16Scale}, ArrayRef<AffineMap>{identityMap, identityMap},
            parallelIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto scale =
                    arith::ConstantOp::create(b, l, b.getFloatAttr(bf16Type, info.inputScale));
                auto scaled = arith::MulFOp::create(b, l, args[0], scale);
                linalg::YieldOp::create(b, l, scaled.getResult());
            }
        );

        auto emptyBf16Out = tensor::EmptyOp::create(rewriter, loc, inputShape, bf16Type);
        auto softmaxOp = linalg::SoftmaxOp::create(
            rewriter, loc, TypeRange{bf16TensorType}, scaleToBf16.getResult(0), emptyBf16Out,
            info.reductionDim
        );

        // softmax_prob * 256. Assumes the matched rescale's effective scale = 1;
        // non-trivial quantizers would need 256 * 2^shift / mult.
        auto emptyBf16Mul = tensor::EmptyOp::create(rewriter, loc, inputShape, bf16Type);
        auto scaledBf16 = linalg::GenericOp::create(
            rewriter, loc, TypeRange{bf16TensorType}, ValueRange{softmaxOp.getResult()},
            ValueRange{emptyBf16Mul}, ArrayRef<AffineMap>{identityMap, identityMap},
            parallelIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto scale = arith::ConstantOp::create(b, l, b.getFloatAttr(bf16Type, 256.0));
                auto scaled = arith::MulFOp::create(b, l, args[0], scale);
                linalg::YieldOp::create(b, l, scaled.getResult());
            }
        );

        // bf16 -> i32. round_even to match `isTorqCastOp`'s "f2i".
        auto emptyI32 = tensor::EmptyOp::create(rewriter, loc, inputShape, i32Type);
        auto scaledI32 = linalg::GenericOp::create(
            rewriter, loc, TypeRange{i32TensorType}, ValueRange{scaledBf16.getResult(0)},
            ValueRange{emptyI32}, ArrayRef<AffineMap>{identityMap, identityMap}, parallelIterators,
            [&](OpBuilder &b, Location l, ValueRange args) {
                auto rounded = math::RoundEvenOp::create(b, l, args[0]);
                auto asI32 = arith::FPToSIOp::create(b, l, i32Type, rounded);
                linalg::YieldOp::create(b, l, asI32.getResult());
            }
        );

        rewriter.replaceOp(info.replaceTarget, scaledI32.getResult(0));

        setTargetExecutorIfForced(softmaxOp, rewriter, "softmax");
    }
};

void populateRaiseSoftmaxOpPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<RaiseSoftmaxOnnx>(context);
    patterns.add<RaiseDecomposedSoftmax>(context);
    patterns.add<RaiseSoftmaxTflite>(context);
}

} // namespace mlir::syna::torq
