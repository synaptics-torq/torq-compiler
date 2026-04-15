// Copyright 2024 SYNAPTICS Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PassesDetail.h"

#include "torq/Conversions/LinalgToTorqHL/PatternUtils.h"
#include "torq/Conversions/LinalgToTorqHL/Patterns.h"
#include "torq/Dialect/TorqHL/TorqHLOps.h"
#include "torq/Utils/ConversionUtils.h"
#include "torq/Utils/TorqUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "numeric"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "torq-optimize-generic-to-specific-op-pattern"

namespace mlir::syna::torq {

namespace {

/// Check if a linalg.generic is a "mul" operation:
///   ^bb0(%in0: i16, %in1: i16, %out: i32):
///     %0 = arith.muli %in0, %in1 : i16
///     %1 = arith.extsi %0 : i16 to i32
///     linalg.yield %1 : i32
/// Returns the muli op if matched, nullptr otherwise.
static arith::MulIOp matchMulGeneric(linalg::GenericOp op) {
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Checking if op is a mul generic...\n");

    if (op.getNumDpsInputs() != 2) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a mul: expected 2 inputs, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return nullptr;
    }

    if (op.getNumDpsInits() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a mul: expected 1 init\n");
        return nullptr;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a mul: not all loops are parallel\n");
        return nullptr;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a mul: bad yield\n");
        return nullptr;
    }

    // The yield value should be extsi(muli(...))
    Value yieldVal = yieldOp.getOperand(0);
    auto extsiOp = yieldVal.getDefiningOp<arith::ExtSIOp>();
    if (!extsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a mul: yield is not extsi\n");
        return nullptr;
    }

    auto muliOp = extsiOp.getIn().getDefiningOp<arith::MulIOp>();
    if (!muliOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a mul: extsi input is not muli\n");
        return nullptr;
    }

    // Both muli operands should come from block arguments (the two inputs)
    auto lhs = muliOp.getLhs();
    auto rhs = muliOp.getRhs();
    if (!isa<BlockArgument>(lhs) || !isa<BlockArgument>(rhs)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a mul: muli operands are not block args\n"
        );
        return nullptr;
    }

    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Matched mul generic!\n");
    return muliOp;
}

/// Check if a linalg.generic is a "rescale with identity scale" (multiplier/2^shift ≈ 1):
///   ^bb0(%in: i8, %out: i16):
///     %0 = arith.extsi %in : i8 to i32
///     %1 = arith.subi %0, %c_zp : i32          (input zero-point subtraction)
///     %2 = tosa.apply_scale %1, %c_mult, %c_shift : (i32, i32, i8) -> i16
///     linalg.yield %2 : i16
///
/// We verify that multiplier/2^shift ≈ 1.0 (meaning this rescale is essentially
/// just subtracting the zero-point and not actually changing the scale).
///
/// On success, returns the input zero-point value and the defining linalg.generic input.
/// If no subi is found, zp is set to 0.
static bool
matchIdentityRescale(linalg::GenericOp op, int32_t &outZp, linalg::GenericOp &outInputOp) {
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Checking if op is an identity rescale...\n");
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Op: " << op << "\n");

    if (op.getNumDpsInputs() != 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: expected 1 input, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return false;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: not all loops parallel\n"
        );
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: bad yield\n");
        return false;
    }

    Value val = yieldOp.getOperand(0);
    auto truncOp = dyn_cast_or_null<arith::TruncIOp>(val.getDefiningOp());
    if (truncOp) {
        val = truncOp.getIn();
    }
    auto applyScaleOp = dyn_cast_or_null<tosa::ApplyScaleOp>(val.getDefiningOp());
    if (!applyScaleOp) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: yield is not apply_scale\n"
        );
        return false;
    }

    // Get multiplier and shift as scalar constants
    auto maybeMult = getConstIntValue(applyScaleOp.getMultiplier());
    auto maybeShift = getConstIntValue(applyScaleOp.getShift());
    if (!maybeMult || !maybeShift) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "[SwishMatch]   -> Not identity rescale: multiplier or shift not constant\n"
        );
        return false;
    }

    int64_t mult = *maybeMult;
    int64_t shift = *maybeShift;

    // Check multiplier / 2^shift ≈ 1.0
    // For example, mult=1073741824 (2^30) and shift=30 => 1073741824/2^30 = 1.0
    double scaleValue = static_cast<double>(mult) / std::pow(2.0, static_cast<double>(shift));
    LLVM_DEBUG(
        llvm::dbgs() << "[SwishMatch]   Multiplier=" << mult << ", Shift=" << shift
                     << ", scale=" << scaleValue << "\n"
    );

    if (std::abs(scaleValue - 1.0) > 1e-6) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: scale " << scaleValue
                         << " is not ~1.0\n"
        );
        return false;
    }

    // Now walk into the apply_scale input to find subi (zero-point) and extsi
    Value scaleInput = applyScaleOp.getValue();

    // Check for subi (zero-point subtraction)
    int32_t zp = 0;
    if (auto subiOp = scaleInput.getDefiningOp<arith::SubIOp>()) {
        auto maybeZp = getConstIntValue(subiOp.getRhs());
        if (!maybeZp) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: subi rhs not constant\n"
            );
            return false;
        }
        zp = static_cast<int32_t>(*maybeZp);
        scaleInput = subiOp.getLhs();
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Found input zero-point: " << zp << "\n");
    }
    else {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   No subi found, zero-point = 0\n");
    }

    // Check for extsi
    if (auto extsiOp = scaleInput.getDefiningOp<arith::ExtSIOp>()) {
        scaleInput = extsiOp.getIn();
    }

    // scaleInput should now be a block argument
    if (!isa<BlockArgument>(scaleInput)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: final input not a block arg\n"
        );
        return false;
    }

    // Check that the generic's input comes from another linalg.generic
    Value inputTensor = op.getInputs()[0];
    auto inputGeneric = inputTensor.getDefiningOp<linalg::GenericOp>();
    if (!inputGeneric) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not identity rescale: input not a linalg.generic\n"
        );
        return false;
    }

    outZp = zp;
    outInputOp = inputGeneric;

    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Matched identity rescale! zp=" << zp << "\n");
    return true;
}

/// Check if a linalg.generic is a "table" (lookup) operation:
///   ^bb0(%in: i8, %out: i8):
///     %0 = arith.index_cast %in : i8 to index
///     %1 = arith.addi %0, %c128 : index
///     %2 = tensor.extract %table[%1] : tensor<256xi8>
///     linalg.yield %2 : i8
///
/// Returns true if matched, and sets outInputOp to the input linalg.generic.
static bool matchTableOp(linalg::GenericOp op, linalg::GenericOp &outInputOp) {
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Checking if op is a table (LUT) operation...\n");
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Op: " << op << "\n");

    if (op.getNumDpsInputs() != 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a table: expected 1 input, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return false;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a table: not all loops parallel\n");
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a table: bad yield\n");
        return false;
    }

    // The yield value should be a tensor.extract
    auto extractOp = yieldOp.getOperand(0).getDefiningOp<tensor::ExtractOp>();
    if (!extractOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not a table: yield is not tensor.extract\n");
        return false;
    }

    // The extract index should come from arith.addi(arith.index_cast(%in), cst)
    auto indices = extractOp.getIndices();
    if (indices.size() != 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a table: extract has " << indices.size()
                         << " indices (expected 1)\n"
        );
        return false;
    }

    Value indexVal = indices[0];
    // Check for addi (offset by 128 for signed-to-unsigned conversion)
    if (auto addiOp = indexVal.getDefiningOp<arith::AddIOp>()) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Found addi in table index computation\n");
        indexVal = addiOp.getLhs();
    }

    // Check for index_cast
    auto indexCastOp = indexVal.getDefiningOp<arith::IndexCastOp>();
    if (!indexCastOp) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a table: index does not come from index_cast\n"
        );
        return false;
    }

    // The index_cast input should be a block argument
    Value castInput = indexCastOp.getIn();
    if (!isa<BlockArgument>(castInput)) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a table: index_cast input not a block arg\n"
        );
        return false;
    }

    // Check the table source tensor shape (should be 256 for i8 LUT)
    auto tableType = dyn_cast<RankedTensorType>(extractOp.getTensor().getType());
    if (tableType) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Table tensor shape: " << tableType << "\n");
    }

    // Now get the input generic op
    Value inputTensor = op.getInputs()[0];
    auto inputGeneric = inputTensor.getDefiningOp<linalg::GenericOp>();
    if (!inputGeneric) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not a table: input is not a linalg.generic\n"
        );
        return false;
    }

    outInputOp = inputGeneric;
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Matched table operation!\n");
    return true;
}

/// Check if a linalg.generic is a "rescale from i32 to i8" with clamp:
///   ^bb0(%in: i32, %out: i8):
///     %0 = tosa.apply_scale %in, %mult, %shift {double_round = true} : (i32, i32, i8) -> i32
///     %1 = arith.addi %0, %c_zp : i32          (output zero-point addition)
///     %2 = arith.maxsi %1, %c-128 : i32        (clamp min)
///     %3 = arith.minsi %2, %c127 : i32         (clamp max)
///     %4 = arith.trunci %3 : i32 to i8
///     linalg.yield %4 : i8
static bool matchOutputRescale(linalg::GenericOp op, int32_t &outZp) {
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Checking if op is an output rescale (i32->i8)...\n");
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Op: " << op << "\n");

    // Need at least 1 input (the data tensor). Additional inputs are allowed
    // for per-channel multiplier/shift tensors fed to tosa.apply_scale.
    if (op.getNumDpsInputs() < 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not output rescale: expected at least 1 input, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return false;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   -> Not output rescale: not all loops parallel\n"
        );
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not output rescale: bad yield\n");
        return false;
    }

    // Walk from yield backwards: trunci -> minsi -> maxsi -> addi -> apply_scale
    Value val = yieldOp.getOperand(0);

    auto trunciOp = val.getDefiningOp<arith::TruncIOp>();
    if (!trunciOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not output rescale: no trunci\n");
        return false;
    }
    val = trunciOp.getIn();

    auto minsiOp = val.getDefiningOp<arith::MinSIOp>();
    if (!minsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not output rescale: no minsi\n");
        return false;
    }
    val = minsiOp.getLhs();

    auto maxsiOp = val.getDefiningOp<arith::MaxSIOp>();
    if (!maxsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not output rescale: no maxsi\n");
        return false;
    }
    val = maxsiOp.getLhs();

    // Check for addi (output zero-point)
    int32_t zp = 0;
    if (auto addiOp = val.getDefiningOp<arith::AddIOp>()) {
        auto maybeZp = getConstIntValue(addiOp.getRhs());
        if (maybeZp) {
            zp = static_cast<int32_t>(*maybeZp);
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch]   Found output zero-point in rescale: " << zp << "\n"
            );
        }
        val = addiOp.getLhs();
    }

    // Should be tosa.apply_scale
    auto applyScaleOp = val.getDefiningOp<tosa::ApplyScaleOp>();
    if (!applyScaleOp) {
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Not output rescale: no apply_scale\n");
        return false;
    }

    // Log the scale values
    auto maybeMult = getConstIntValue(applyScaleOp.getMultiplier());
    auto maybeShift = getConstIntValue(applyScaleOp.getShift());
    if (maybeMult && maybeShift) {
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   Output rescale: multiplier=" << *maybeMult
                         << ", shift=" << *maybeShift << "\n"
        );
    }

    outZp = zp;
    LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   -> Matched output rescale! zp=" << zp << "\n");
    return true;
}

/// Check if a linalg.generic is a "subi with constant" operation:
///   ^bb0(%in: iN, %out: iN):
///     %0 = arith.subi %in, %cst : iN
///     linalg.yield %0 : iN
/// Returns true if matched, and sets outZp to the constant subtracted.
static bool matchSubiRescale(linalg::GenericOp op, int64_t &outZp) {
    LLVM_DEBUG(llvm::dbgs() << "[FuseMul] Checking if op is a subi rescale...\n");

    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: expected 1 input, 1 init\n");
        return false;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: not all loops parallel\n");
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: bad yield\n");
        return false;
    }

    auto subiOp = yieldOp.getOperand(0).getDefiningOp<arith::SubIOp>();
    if (!subiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: yield is not subi\n");
        return false;
    }

    // LHS should be block argument, RHS should be constant
    if (!isa<BlockArgument>(subiOp.getLhs())) {
        LLVM_DEBUG(
            llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: subi lhs is not block arg\n"
        );
        return false;
    }

    auto maybeZp = getConstIntValue(subiOp.getRhs());
    if (!maybeZp) {
        LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Not a subi rescale: subi rhs is not constant\n");
        return false;
    }

    outZp = *maybeZp;
    LLVM_DEBUG(llvm::dbgs() << "[FuseMul]   -> Matched subi rescale! zp=" << outZp << "\n");
    return true;
}

/// ============================================================================
/// SwishActivationOpPattern
/// ============================================================================
///
/// Matches the quantized swish activation diamond pattern:
///
///        %input (i32)
///            |
///        %rescale_common (rescale i32 → i8, with apply_scale + zp + clamp + trunci)
///          /           \
///   %table_op        %rescale_a (identity rescale i8 → i16, mult/2^shift ≈ 1)
///   (LUT: index_cast     |
///    + addi(128)         |
///    + tensor.extract)   |
///         |              |
///   %rescale_b           |
///   (identity rescale    |
///    i8 → i16,           |
///    mult/2^shift ≈ 1)   |
///         \             /
///          %mul_op (muli i16 + extsi → i32)
///
/// The key insight is that both arms of the diamond originate from the same
/// common rescale (%rescale_common), forming the pattern:
///   swish(x) = x * sigmoid(x)
/// where the table implements the sigmoid function on the quantized value.
///
/// The rewrite only touches the table path:
///   - rescale_a (direct path) is left unchanged
///   - table(i8→i8) + rescale_b(i8→i16) are fused into a single
///     modified table(i8→i16) whose i16 LUT bakes in rescale_b's zero-point
///
class SwishActivationOpPattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    SwishActivationOpPattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/0) {
        setDebugName("SwishActivationOpPattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp mulOp, PatternRewriter &rewriter) const override {

        LLVM_DEBUG(
            llvm::dbgs() << "\n===== [SwishMatch] ===== Starting SwishActivationOpPattern "
                            "match on op:\n"
                         << mulOp << "\n"
        );

        // ---------------------------------------------------------------
        // Step 1: Match the mul op (muli of two i16 inputs → i32 output)
        // ---------------------------------------------------------------
        auto muliOp = matchMulGeneric(mulOp);
        if (!muliOp) {
            return rewriter.notifyMatchFailure(mulOp, "[SwishMatch] Op is not a mul generic");
        }

        Value lhsTensor = mulOp.getInputs()[0];
        Value rhsTensor = mulOp.getInputs()[1];

        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Mul op matched. LHS input: " << lhsTensor << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Mul op matched. RHS input: " << rhsTensor << "\n");

        // Both inputs must come from linalg.generic ops (the identity rescales)
        auto lhsGeneric = lhsTensor.getDefiningOp<linalg::GenericOp>();
        auto rhsGeneric = rhsTensor.getDefiningOp<linalg::GenericOp>();

        if (!lhsGeneric || !rhsGeneric) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch] FAIL: One or both mul inputs are not linalg.generic\n"
            );
            return rewriter.notifyMatchFailure(
                mulOp, "[SwishMatch] Mul inputs must be linalg.generic ops"
            );
        }

        // ---------------------------------------------------------------
        // Step 2: Both inputs must be identity rescales (mult/2^shift ≈ 1)
        // ---------------------------------------------------------------
        int32_t lhsZp = 0, rhsZp = 0;
        linalg::GenericOp lhsInputOp, rhsInputOp;

        bool lhsIsRescale = matchIdentityRescale(lhsGeneric, lhsZp, lhsInputOp);
        bool rhsIsRescale = matchIdentityRescale(rhsGeneric, rhsZp, rhsInputOp);

        if (!lhsIsRescale || !rhsIsRescale) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch] FAIL: Not both mul inputs are identity rescales. "
                             << "lhsIsRescale=" << lhsIsRescale << ", rhsIsRescale=" << rhsIsRescale
                             << "\n"
            );
            return rewriter.notifyMatchFailure(
                mulOp, "[SwishMatch] Both mul inputs must be identity rescales (mult/2^shift ≈ 1)"
            );
        }

        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Both inputs are identity rescales.\n");
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   LHS rescale input zp=" << lhsZp
                         << ", input op: " << lhsInputOp << "\n"
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch]   RHS rescale input zp=" << rhsZp
                         << ", input op: " << rhsInputOp << "\n"
        );

        // ---------------------------------------------------------------
        // Step 3: Go back further — one path should have a table op,
        //         the other should go directly to the common rescale.
        //         Try both orderings: (lhs=table, rhs=direct) or vice versa.
        // ---------------------------------------------------------------
        linalg::GenericOp tableOp;
        linalg::GenericOp tableInputOp;
        linalg::GenericOp directRescaleInputOp;
        int32_t tablePathZp = 0;
        int32_t directPathZp = 0;

        // Try LHS path as table
        bool lhsIsTable = matchTableOp(lhsInputOp, tableInputOp);
        if (lhsIsTable) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch] LHS path contains a table op. RHS is direct path.\n"
            );
            tableOp = lhsInputOp;
            directRescaleInputOp = rhsInputOp;
            tablePathZp = lhsZp;
            directPathZp = rhsZp;
        }
        else {
            // Try RHS path as table
            bool rhsIsTable = matchTableOp(rhsInputOp, tableInputOp);
            if (rhsIsTable) {
                LLVM_DEBUG(
                    llvm::dbgs()
                    << "[SwishMatch] RHS path contains a table op. LHS is direct path.\n"
                );
                tableOp = rhsInputOp;
                directRescaleInputOp = lhsInputOp;
                tablePathZp = rhsZp;
                directPathZp = lhsZp;
            }
            else {
                LLVM_DEBUG(
                    llvm::dbgs() << "[SwishMatch] FAIL: Neither path contains a table op.\n"
                );
                LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   LHS input op: " << lhsInputOp << "\n");
                LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   RHS input op: " << rhsInputOp << "\n");
                return rewriter.notifyMatchFailure(
                    mulOp, "[SwishMatch] One of the two paths must contain a table (LUT) operation"
                );
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "[SwishMatch] Table op: " << tableOp << "\n");
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch] Table input op (source of table): " << tableInputOp
                         << "\n"
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch] Direct rescale input op: " << directRescaleInputOp << "\n"
        );
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch] Table path zp=" << tablePathZp
                         << ", Direct path zp=" << directPathZp << "\n"
        );

        // ---------------------------------------------------------------
        // Step 4: Check the diamond — both paths must originate from the
        //         same common source (the output rescale i32 → i8).
        //         tableInputOp == directRescaleInputOp
        // ---------------------------------------------------------------
        if (tableInputOp.getOperation() != directRescaleInputOp.getOperation()) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch] FAIL: Diamond not formed! "
                             << "Table input and direct path input are different ops.\n"
            );
            LLVM_DEBUG(llvm::dbgs() << "[SwishMatch]   Table input op:  " << tableInputOp << "\n");
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch]   Direct input op: " << directRescaleInputOp << "\n"
            );
            return rewriter.notifyMatchFailure(
                mulOp, "[SwishMatch] Both paths must meet at the same common source (diamond)"
            );
        }

        linalg::GenericOp commonRescaleOp = tableInputOp;
        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch] Diamond confirmed! Common rescale op: " << commonRescaleOp
                         << "\n"
        );

        // ---------------------------------------------------------------
        // Step 5: Verify the common source is an output rescale (i32 → i8)
        //         with apply_scale + zp + clamp + trunci
        // ---------------------------------------------------------------
        int32_t commonRescaleZp = 0;
        if (!matchOutputRescale(commonRescaleOp, commonRescaleZp)) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishMatch] FAIL: Common source is not an output rescale "
                             << "(expected apply_scale + addi(zp) + clamp + trunci pattern)\n"
            );
            return rewriter.notifyMatchFailure(
                mulOp, "[SwishMatch] Common source must be an output rescale (i32 → i8 with clamp)"
            );
        }

        LLVM_DEBUG(
            llvm::dbgs() << "[SwishMatch] Common output rescale matched! zp=" << commonRescaleZp
                         << "\n"
        );

        // ---------------------------------------------------------------
        // Step 6: SUCCESS — this is a swish activation!
        //         Log all the matched components.
        // ---------------------------------------------------------------
        LLVM_DEBUG({
            llvm::dbgs() << "\n";
            llvm::dbgs() << "[SwishMatch] ====================================\n";
            llvm::dbgs() << "[SwishMatch]  SWISH ACTIVATION PATTERN MATCHED!  \n";
            llvm::dbgs() << "[SwishMatch] ====================================\n";
            llvm::dbgs() << "[SwishMatch] Pattern summary:\n";
            llvm::dbgs() << "[SwishMatch]   Mul op:             " << mulOp << "\n";
            llvm::dbgs() << "[SwishMatch]   LHS identity rescale (to mul): " << lhsGeneric << "\n";
            llvm::dbgs() << "[SwishMatch]   RHS identity rescale (to mul): " << rhsGeneric << "\n";
            llvm::dbgs() << "[SwishMatch]   Table (LUT) op:     " << tableOp << "\n";
            llvm::dbgs() << "[SwishMatch]   Common rescale op:  " << commonRescaleOp << "\n";
            llvm::dbgs() << "[SwishMatch]   Common rescale zp:  " << commonRescaleZp << "\n";
            llvm::dbgs() << "[SwishMatch]   Table-path zp:      " << tablePathZp << "\n";
            llvm::dbgs() << "[SwishMatch]   Direct-path zp:     " << directPathZp << "\n";
            llvm::dbgs() << "[SwishMatch]   Common rescale input: "
                         << commonRescaleOp.getInputs()[0] << "\n";
            llvm::dbgs() << "[SwishMatch] ====================================\n\n";
        });

        // ---------------------------------------------------------------
        // Step 7: Rewrite the table path only.
        //
        // BEFORE:
        //   commonRescale(i32→i8)
        //     ├── rescale_a(i8→i16): extsi + subi(zp) + apply_scale(1)  [direct path]
        //     └── table(i8→i8) → rescale_b(i8→i16): extsi + subi(zp) + apply_scale(1)  [table path]
        //   mul(i16 × i16 → i32)
        //
        // AFTER:
        //   commonRescale(i32→i8) [unchanged]
        //     ├── rescale_a(i8→i16) [unchanged]
        //     └── tableModified(i8→i16): index_cast + addi(128) + extract from i16 LUT
        //           (rescale_b is fully fused into table: LUT values = original_i8 - tablePathZp)
        //   mul(i16 × i16 → i32) [unchanged]
        // ---------------------------------------------------------------

        Location loc = mulOp.getLoc();
        Value commonRescaleResult = commonRescaleOp.getResult(0);
        auto commonResultType = cast<RankedTensorType>(commonRescaleResult.getType());
        auto shape = commonResultType.getShape();
        auto i16Type = rewriter.getIntegerType(16);
        auto i8Type = rewriter.getIntegerType(8);
        auto i16TensorType = RankedTensorType::get(shape, i16Type);

        // Identify which rescale is on the table path
        linalg::GenericOp tableRescaleGeneric; // the identity rescale on the table path

        if (lhsIsTable) {
            tableRescaleGeneric = lhsGeneric; // LHS = table path rescale
        }
        else {
            tableRescaleGeneric = rhsGeneric; // RHS = table path rescale
        }

        LLVM_DEBUG(llvm::dbgs() << "[SwishRewrite] Starting rewrite...\n");

        // ---- 7a: Build i16 LUT with rescale_b fused in ----
        // Fuse rescale_b into the table: each LUT entry becomes
        //   newLUT[i] = (int16_t)oldLUT[i] - tablePathZp
        // The table now directly outputs rescaled i16 values.

        Block &oldTableBlock = tableOp.getRegion().front();
        auto oldTableYield = cast<linalg::YieldOp>(oldTableBlock.getTerminator());
        auto oldExtractOp = oldTableYield.getOperand(0).getDefiningOp<tensor::ExtractOp>();
        assert(oldExtractOp && "Expected table yield to come from tensor.extract");
        Value oldTableTensor = oldExtractOp.getTensor();

        rewriter.setInsertionPointAfter(commonRescaleOp);
        auto oldTableType = cast<RankedTensorType>(oldTableTensor.getType());
        auto i16TableType = RankedTensorType::get(oldTableType.getShape(), i16Type);

        LLVM_DEBUG(
            llvm::dbgs() << "[SwishRewrite] Fusing rescale_b (zp=" << tablePathZp
                         << ") into table LUT values\n"
        );

        Value i16TableTensor;
        if (auto oldConstOp = oldTableTensor.getDefiningOp<arith::ConstantOp>()) {
            if (auto denseAttr = dyn_cast<DenseElementsAttr>(oldConstOp.getValue())) {
                SmallVector<int16_t> i16Values;
                for (int8_t val : denseAttr.getValues<int8_t>()) {
                    i16Values.push_back(
                        static_cast<int16_t>(val) - static_cast<int16_t>(tablePathZp)
                    );
                }
                auto i16DenseAttr = DenseElementsAttr::get(i16TableType, llvm::ArrayRef(i16Values));
                i16TableTensor = arith::ConstantOp::create(rewriter, loc, i16DenseAttr);
                LLVM_DEBUG(
                    llvm::dbgs() << "[SwishRewrite] Created i16 table constant "
                                 << "(each entry = original - " << tablePathZp << ")\n"
                );
            }
        }

        if (!i16TableTensor) {
            LLVM_DEBUG(
                llvm::dbgs() << "[SwishRewrite] FAIL: Could not create i16 table constant\n"
            );
            return rewriter.notifyMatchFailure(
                mulOp, "[SwishRewrite] Table tensor is not a foldable constant"
            );
        }

        // ---- 7b: Build modified table: i8 → i16 ----
        // Input: i8 directly from commonRescale (no cast needed)
        // Body: clone old table index computation (index_cast + addi(128))
        //       but extract from the i16 LUT → yields i16 directly
        // Output: i16 (rescale_b zero-point already baked into LUT values)

        SmallVector<AffineMap> tableMaps;
        int64_t rank = commonResultType.getRank();
        auto identityMap = rewriter.getMultiDimIdentityMap(rank);
        tableMaps.push_back(identityMap); // input
        tableMaps.push_back(identityMap); // output

        SmallVector<utils::IteratorType> tableIterTypes(rank, utils::IteratorType::parallel);

        Value tableInit = tensor::EmptyOp::create(rewriter, loc, shape, i16Type).getResult();

        auto tableNewOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{i16TensorType}, ValueRange{commonRescaleResult},
            ValueRange{tableInit}, tableMaps, tableIterTypes
        );

        {
            // Body: i8 input → index_cast + addi(128) + extract from i16 table → i16 output
            Block *tableBlock = rewriter.createBlock(
                &tableNewOp.getRegion(), tableNewOp.getRegion().end(), TypeRange{i8Type, i16Type},
                SmallVector<Location>(2, loc)
            );
            rewriter.setInsertionPointToStart(tableBlock);

            // Clone the old table body, mapping old i8 block arg directly
            // and replacing tensor.extract to use the i16 table
            IRMapping mapping;
            mapping.map(oldTableBlock.getArgument(0), tableBlock->getArgument(0));
            mapping.map(oldTableBlock.getArgument(1), tableBlock->getArgument(1));

            for (auto &op : oldTableBlock.without_terminator()) {
                if (auto extractOp = dyn_cast<tensor::ExtractOp>(&op)) {
                    // Replace: extract from i8 table → extract from i16 table
                    SmallVector<Value> remappedIndices;
                    for (Value idx : extractOp.getIndices()) {
                        remappedIndices.push_back(mapping.lookupOrDefault(idx));
                    }
                    auto newExtract =
                        tensor::ExtractOp::create(rewriter, loc, i16TableTensor, remappedIndices);
                    mapping.map(extractOp.getResult(), newExtract.getResult());
                }
                else {
                    Operation *cloned = rewriter.clone(op, mapping);
                    for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults())) {
                        mapping.map(oldRes, newRes);
                    }
                }
            }

            // tensor.extract now returns i16 directly — yield it
            Value tableOut = mapping.lookupOrDefault(oldTableYield.getOperand(0));
            linalg::YieldOp::create(rewriter, loc, tableOut);
        }

        LLVM_DEBUG(
            llvm::dbgs() << "[SwishRewrite] Created modified table (i8->i16): " << tableNewOp
                         << "\n"
        );

        // ---- 7c: Replace only the table-path rescale ----
        // rescale_a (direct path) stays unchanged.
        // tableRescaleGeneric (rescale_b) is replaced by the new table output.
        rewriter.replaceOp(tableRescaleGeneric, tableNewOp.getResult(0));

        // Erase the old table op (it's no longer needed since we built a new one)
        if (tableOp->use_empty()) {
            rewriter.eraseOp(tableOp);
        }

        LLVM_DEBUG({
            llvm::dbgs() << "[SwishRewrite] ====================================\n";
            llvm::dbgs() << "[SwishRewrite]  SWISH REWRITE COMPLETE!            \n";
            llvm::dbgs() << "[SwishRewrite] ====================================\n";
        });

        return success();
    }
};

} // anonymous namespace

void populateSwishActivationPatterns(MLIRContext *context, RewritePatternSet &patterns) {
    patterns.add<SwishActivationOpPattern>(context);
}

} // namespace mlir::syna::torq