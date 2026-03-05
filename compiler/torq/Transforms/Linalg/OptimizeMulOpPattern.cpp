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

#define DEBUG_TYPE "torq-optimize-mul-op-pattern"

namespace mlir::syna::torq {

extern llvm::cl::opt<bool> clEnableTorqHLTiling;

namespace {

/// Check if a linalg.generic is a "mul" operation:
///   ^bb0(%in0: i16, %in1: i16, %out: i32):
///     %0 = arith.muli %in0, %in1 : i16
///     %1 = arith.extsi %0 : i16 to i32
///     linalg.yield %1 : i32
/// Returns the muli op if matched, nullptr otherwise.
static arith::MulIOp matchMulGeneric(linalg::GenericOp op) {
    LLVM_DEBUG(llvm::dbgs() << "[MulMatch] Checking if op is a mul generic...\n");

    if (op.getNumDpsInputs() != 2) {
        LLVM_DEBUG(
            llvm::dbgs() << "[MulMatch]   -> Not a mul: expected 2 inputs, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return nullptr;
    }

    if (op.getNumDpsInits() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: expected 1 init\n");
        return nullptr;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: not all loops are parallel\n");
        return nullptr;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: bad yield\n");
        return nullptr;
    }

    // The yield value should be extsi(muli(...))
    Value yieldVal = yieldOp.getOperand(0);
    auto extsiOp = yieldVal.getDefiningOp<arith::ExtSIOp>();
    if (!extsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: yield is not extsi\n");
        return nullptr;
    }

    auto muliOp = extsiOp.getIn().getDefiningOp<arith::MulIOp>();
    if (!muliOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: extsi input is not muli\n");
        return nullptr;
    }

    // Both muli operands should come from block arguments (the two inputs)
    auto lhs = muliOp.getLhs();
    auto rhs = muliOp.getRhs();
    if (!isa<BlockArgument>(lhs) || !isa<BlockArgument>(rhs)) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not a mul: muli operands are not block args\n");
        return nullptr;
    }

    LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Matched mul generic!\n");
    return muliOp;
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
    LLVM_DEBUG(llvm::dbgs() << "[MulMatch] Checking if op is an output rescale (i32->i8)...\n");
    LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   Op: " << op << "\n");

    // Need at least 1 input (the data tensor). Additional inputs are allowed
    // for per-channel multiplier/shift tensors fed to tosa.apply_scale.
    if (op.getNumDpsInputs() < 1) {
        LLVM_DEBUG(
            llvm::dbgs() << "[MulMatch]   -> Not output rescale: expected at least 1 input, got "
                         << op.getNumDpsInputs() << "\n"
        );
        return false;
    }

    if (op.getNumParallelLoops() != op.getNumLoops()) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: not all loops parallel\n");
        return false;
    }

    auto yieldOp = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != 1) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: bad yield\n");
        return false;
    }

    // Walk from yield backwards: trunci -> minsi -> maxsi -> addi -> apply_scale
    Value val = yieldOp.getOperand(0);

    auto trunciOp = val.getDefiningOp<arith::TruncIOp>();
    if (!trunciOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: no trunci\n");
        return false;
    }
    val = trunciOp.getIn();

    auto minsiOp = val.getDefiningOp<arith::MinSIOp>();
    if (!minsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: no minsi\n");
        return false;
    }
    val = minsiOp.getLhs();

    auto maxsiOp = val.getDefiningOp<arith::MaxSIOp>();
    if (!maxsiOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: no maxsi\n");
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
                llvm::dbgs() << "[MulMatch]   Found output zero-point in rescale: " << zp << "\n"
            );
        }
        val = addiOp.getLhs();
    }

    // Should be tosa.apply_scale
    auto applyScaleOp = val.getDefiningOp<tosa::ApplyScaleOp>();
    if (!applyScaleOp) {
        LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Not output rescale: no apply_scale\n");
        return false;
    }

    // Log the scale values
    auto maybeMult = getConstIntValue(applyScaleOp.getMultiplier());
    auto maybeShift = getConstIntValue(applyScaleOp.getShift());
    if (maybeMult && maybeShift) {
        LLVM_DEBUG(
            llvm::dbgs() << "[MulMatch]   Output rescale: multiplier=" << *maybeMult
                         << ", shift=" << *maybeShift << "\n"
        );
    }

    outZp = zp;
    LLVM_DEBUG(llvm::dbgs() << "[MulMatch]   -> Matched output rescale! zp=" << zp << "\n");
    return true;
}

/// ============================================================================
/// FuseMulRescalePattern
/// ============================================================================
///
/// Fuses a mul generic followed by an output rescale into a single linalg.generic.
///
/// BEFORE:
///   %mul = linalg.generic ins(%a, %b : i16, i16) outs(%init_i32 : i32) {
///     %0 = arith.muli %in0, %in1 : i16
///     %1 = arith.extsi %0 : i16 to i32
///     linalg.yield %1 : i32
///   }
///   %rescale = linalg.generic ins(%mul : i32) outs(%init_i8 : i8) {
///     %0 = tosa.apply_scale %in, %mult, %shift : (i32, i32, i8) -> i32
///     %1 = arith.addi %0, %zp : i32
///     %2 = arith.maxsi %1, %c-128 : i32
///     %3 = arith.minsi %2, %c127 : i32
///     %4 = arith.trunci %3 : i32 to i8
///     linalg.yield %4 : i8
///   }
///
/// AFTER:
///   %fused = linalg.generic ins(%a, %b : i16, i16) outs(%init_i8 : i8) {
///     %0 = arith.muli %in0, %in1 : i16
///     %1 = arith.extsi %0 : i16 to i32
///     %2 = tosa.apply_scale %1, %mult, %shift : (i32, i32, i8) -> i32
///     %3 = arith.addi %2, %zp : i32
///     %4 = arith.maxsi %3, %c-128 : i32
///     %5 = arith.minsi %4, %c127 : i32
///     %6 = arith.trunci %5 : i32 to i8
///     linalg.yield %6 : i8
///   }
///
class FuseMulRescalePattern : public OpRewritePattern<linalg::GenericOp> {
  public:
    FuseMulRescalePattern(MLIRContext *context)
        : OpRewritePattern<linalg::GenericOp>(context, /*benefit=*/1) {
        setDebugName("FuseMulRescalePattern");
    }

    LogicalResult
    matchAndRewrite(linalg::GenericOp rescaleOp, PatternRewriter &rewriter) const override {

        LLVM_DEBUG(
            llvm::dbgs() << "\n===== [FuseMulRescale] ===== Starting match on op:\n"
                         << rescaleOp << "\n"
        );

        // ---------------------------------------------------------------
        // Step 1: Match the output rescale pattern (apply_scale + zp + clamp + trunci)
        // ---------------------------------------------------------------
        int32_t outputZp = 0;
        if (!matchOutputRescale(rescaleOp, outputZp)) {
            return rewriter.notifyMatchFailure(
                rescaleOp, "[FuseMulRescale] Op is not an output rescale"
            );
        }

        // ---------------------------------------------------------------
        // Step 2: Check that the rescale's input comes from a mul generic
        // ---------------------------------------------------------------
        Value rescaleInput = rescaleOp.getInputs()[0];
        auto mulOp = rescaleInput.getDefiningOp<linalg::GenericOp>();
        if (!mulOp) {
            return rewriter.notifyMatchFailure(
                rescaleOp, "[FuseMulRescale] Rescale input is not a linalg.generic"
            );
        }

        auto muliOp = matchMulGeneric(mulOp);
        if (!muliOp) {
            return rewriter.notifyMatchFailure(
                rescaleOp, "[FuseMulRescale] Rescale input is not a mul generic"
            );
        }

        // ---------------------------------------------------------------
        // Step 3: Check that the mul result is only used by this rescale
        // ---------------------------------------------------------------
        if (!mulOp.getResult(0).hasOneUse()) {
            return rewriter.notifyMatchFailure(
                rescaleOp, "[FuseMulRescale] Mul result has multiple uses, cannot fuse"
            );
        }

        // ---------------------------------------------------------------
        // Step 4: Verify loop counts match
        // ---------------------------------------------------------------
        if (mulOp.getNumLoops() != rescaleOp.getNumLoops()) {
            return rewriter.notifyMatchFailure(
                rescaleOp, "[FuseMulRescale] Mul and rescale have different loop counts"
            );
        }

        LLVM_DEBUG(llvm::dbgs() << "[FuseMulRescale] Matched! Mul op: " << mulOp << "\n");
        LLVM_DEBUG(llvm::dbgs() << "[FuseMulRescale] Output zp=" << outputZp << "\n");

        // ---------------------------------------------------------------
        // Step 5: Build the fused linalg.generic
        // ---------------------------------------------------------------
        Location loc = rescaleOp.getLoc();

        Value mulLhs = mulOp.getInputs()[0];
        Value mulRhs = mulOp.getInputs()[1];
        Value rescaleInit = rescaleOp.getDpsInits()[0];

        // Indexing maps: mul's two input maps + rescale's output map
        SmallVector<AffineMap> fusedMaps;
        auto mulMaps = mulOp.getIndexingMapsArray();
        auto rescaleMaps = rescaleOp.getIndexingMapsArray();
        fusedMaps.push_back(mulMaps[0]);     // mul input 0
        fusedMaps.push_back(mulMaps[1]);     // mul input 1
        fusedMaps.push_back(rescaleMaps[1]); // rescale output

        SmallVector<utils::IteratorType> iterTypes(
            mulOp.getNumLoops(), utils::IteratorType::parallel
        );

        auto resultType = cast<RankedTensorType>(rescaleOp.getResult(0).getType());

        auto fusedOp = rewriter.create<linalg::GenericOp>(
            loc, TypeRange{resultType}, ValueRange{mulLhs, mulRhs}, ValueRange{rescaleInit},
            fusedMaps, iterTypes
        );

        // Build the fused body
        {
            auto in0Type = cast<RankedTensorType>(mulLhs.getType()).getElementType();
            auto in1Type = cast<RankedTensorType>(mulRhs.getType()).getElementType();
            auto outType = resultType.getElementType();

            Block *fusedBlock = rewriter.createBlock(
                &fusedOp.getRegion(), fusedOp.getRegion().end(),
                TypeRange{in0Type, in1Type, outType}, SmallVector<Location>(3, loc)
            );
            rewriter.setInsertionPointToStart(fusedBlock);

            // Clone mul body (without yield), mapping block args
            Block &mulBlock = mulOp.getRegion().front();
            IRMapping mulMapping;
            mulMapping.map(mulBlock.getArgument(0), fusedBlock->getArgument(0));
            mulMapping.map(mulBlock.getArgument(1), fusedBlock->getArgument(1));

            for (auto &op : mulBlock.without_terminator()) {
                Operation *cloned = rewriter.clone(op, mulMapping);
                for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults())) {
                    mulMapping.map(oldRes, newRes);
                }
            }

            // Get the mul's yield value (the extsi i32 result)
            auto mulYield = cast<linalg::YieldOp>(mulBlock.getTerminator());
            Value mulResult = mulMapping.lookupOrDefault(mulYield.getOperand(0));

            // Clone rescale body (without yield), mapping its input to the mul result
            Block &rescaleBlock = rescaleOp.getRegion().front();
            IRMapping rescaleMapping;
            rescaleMapping.map(rescaleBlock.getArgument(0), mulResult);
            rescaleMapping.map(rescaleBlock.getArgument(1), fusedBlock->getArgument(2));

            for (auto &op : rescaleBlock.without_terminator()) {
                Operation *cloned = rewriter.clone(op, rescaleMapping);
                for (auto [oldRes, newRes] : llvm::zip(op.getResults(), cloned->getResults())) {
                    rescaleMapping.map(oldRes, newRes);
                }
            }

            // Yield the rescale's final result (the trunci i8)
            auto rescaleYield = cast<linalg::YieldOp>(rescaleBlock.getTerminator());
            Value finalResult = rescaleMapping.lookupOrDefault(rescaleYield.getOperand(0));
            rewriter.create<linalg::YieldOp>(loc, finalResult);
        }

        LLVM_DEBUG(llvm::dbgs() << "[FuseMulRescale] Created fused op: " << fusedOp << "\n");

        // Replace the rescale op with the fused op
        rewriter.replaceOp(rescaleOp, fusedOp.getResult(0));

        // Erase the old mul op (now dead)
        if (mulOp->use_empty()) {
            rewriter.eraseOp(mulOp);
        }

        LLVM_DEBUG({
            llvm::dbgs() << "[FuseMulRescale] ====================================\n";
            llvm::dbgs() << "[FuseMulRescale]  MUL+RESCALE FUSION COMPLETE!       \n";
            llvm::dbgs() << "[FuseMulRescale] ====================================\n";
        });

        return success();
    }
};

} // namespace

void populateMulPatterns(MLIRContext *context, RewritePatternSet &patterns) {

    // Fuse linalg.generic(mul) + linalg.generic(rescale) at the linalg level
    // to avoid torq-tile tiling them individually. The fused
    // linalg.generic(mul+rescale) is then pattern-matched during torq-hl
    // lowering.
    if (clEnableTorqHLTiling) {
        patterns.add<FuseMulRescalePattern>(context);
    }
}

} // namespace mlir::syna::torq